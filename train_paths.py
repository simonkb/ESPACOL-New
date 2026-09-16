#!/usr/bin/env python3
"""Train PATHS under the locked ORIGIN split/evaluation protocol."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping

import torch

sys.path.insert(0, os.path.dirname(__file__))

from configs.paths_config import PathsConfig
from Datasets.origin_data import (
    ORIGIN_PREPROCESSING_VERSION,
    class_histogram,
    load_origin_items,
    make_origin_loaders,
    split_origin_items,
    split_paths_are_disjoint,
    validate_ordinal_labels,
)
from models.paths import build_paths_model
from train_origin import (
    DATASET_DEFAULTS,
    guard_and_write_split_manifest,
    parse_folds,
    parse_scales,
    set_seed,
    split_signature,
    write_json_atomic,
)
from training.paths_trainer import PathsTrainer


def _parse_probes(value: str) -> tuple[float, ...]:
    try:
        probes = tuple(float(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError("PGF probes must be comma-separated floats") from exc
    if not probes:
        raise argparse.ArgumentTypeError("at least one PGF probe is required")
    return probes


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="PATHS PGF continuation-spectrum severity training",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dataset", choices=tuple(DATASET_DEFAULTS), default="aptos")
    parser.add_argument("--data_root", default=None)
    parser.add_argument("--labels_csv", default=None)
    parser.add_argument("--image_column", default=None)
    parser.add_argument("--label_column", default=None)
    parser.add_argument("--image_dir", default=None)
    parser.add_argument("--n_classes", type=int, default=None)
    parser.add_argument("--n_folds", type=int, default=None)
    parser.add_argument("--val_fraction", type=float, default=0.1)
    parser.add_argument("--folds", default="0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run_dir", default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--include_test", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--fundus_preprocessing", action="store_true")

    base = parser.add_argument_group("audited V3 base")
    base.add_argument("--v3_checkpoint", required=True)
    base.add_argument("--v3_sha256", required=True)
    base.add_argument("--image_size", type=int, default=640)
    base.add_argument("--encoder", choices=("convnext_tiny",), default="convnext_tiny")
    base.add_argument("--scales", type=parse_scales, default=parse_scales("s4,s8,s16,s32"))
    base.add_argument("--projection_dim", type=int, default=128)
    base.add_argument("--reference_count", type=float, default=4096.0)
    base.add_argument("--atom_rate_init", type=float, default=1e-6)
    base.add_argument("--prior_rate_init", type=float, default=1e-4)
    base.add_argument("--boundary_scale_init", type=float, default=1.0)
    base.add_argument("--total_rate_cap", type=float, default=64.0)
    base.add_argument("--prior_rate_cap", type=float, default=1.0)
    base.add_argument("--boundary_scale_cap", type=float, default=2.0)
    base.add_argument("--rate_roundoff_margin", type=float, default=1.0)
    base.add_argument("--mask_valid_fraction", type=float, default=0.5)
    base.add_argument("--grad_checkpoint", action="store_true")
    base.add_argument(
        "--decision_rule",
        choices=("class_map", "posterior_median", "rounded_expected"),
        default="class_map",
    )

    paths = parser.add_argument_group("PATHS architecture")
    paths.add_argument("--pgf_probes", type=_parse_probes, default=_parse_probes("0.05,0.20,0.50,0.80"))
    paths.add_argument("--correction_cap", type=float, default=3.0)
    paths.add_argument("--correction_gain_init", type=float, default=0.05)
    paths.add_argument("--correction_strength", type=float, default=1.0)
    paths.add_argument("--risk_set_alpha", type=float, default=0.5)
    paths.add_argument("--rps_weight", type=float, default=0.25)

    training = parser.add_argument_group("training")
    training.add_argument("--epochs", type=int, default=None)
    training.add_argument("--batch_size", type=int, default=8)
    training.add_argument("--num_workers", type=int, default=8)
    training.add_argument("--paths_encoder_lr", type=float, default=1e-5)
    training.add_argument("--paths_base_lr", type=float, default=5e-5)
    training.add_argument("--paths_refiner_lr", type=float, default=5e-4)
    training.add_argument("--weight_decay", type=float, default=1e-5)
    training.add_argument("--grad_clip_norm", type=float, default=5.0)
    training.add_argument("--correction_only_epochs", type=int, default=3)
    training.add_argument("--freeze_encoder_epochs", type=int, default=3)
    training.add_argument("--lr_factor", type=float, default=0.2)
    training.add_argument("--lr_patience", type=int, default=5)
    training.add_argument("--lr_min", type=float, default=1e-6)
    training.add_argument("--early_stopping_patience", type=int, default=12)
    training.add_argument(
        "--checkpoint_selection",
        choices=("acc_then_qwk", "acc_qwk_score"),
        default="acc_then_qwk",
    )
    training.add_argument("--selection_qwk_weight", type=float, default=0.10)
    amp = training.add_mutually_exclusive_group()
    amp.add_argument("--amp", dest="amp", action="store_true")
    amp.add_argument("--no_amp", dest="amp", action="store_false")
    amp.set_defaults(amp=True)
    training.add_argument("--amp_init_scale", type=float, default=4096.0)
    training.add_argument("--amp_unfreeze_scale", type=float, default=256.0)
    training.add_argument("--amp_growth_interval", type=int, default=2000)
    training.add_argument("--amp_max_consecutive_skips", type=int, default=8)
    training.add_argument("--certificate_samples", type=int, default=8)
    return parser


def _setup_logging(run_dir: str) -> logging.Logger:
    Path(run_dir).mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(Path(run_dir) / "train.log"),
        ],
        force=True,
    )
    return logging.getLogger("train_paths")


def _scalar_summary(result: Mapping[str, Any]) -> dict[str, float | int]:
    return {
        key: value
        for key, value in result.items()
        if isinstance(value, (float, int)) and not isinstance(value, bool)
    }


def main() -> None:
    args = build_parser().parse_args()
    if args.include_test and args.skip_test:
        raise ValueError("--include_test and --skip_test are mutually exclusive")
    defaults = DATASET_DEFAULTS[args.dataset]
    root = args.data_root or str(defaults["root"])
    n_folds = args.n_folds or int(defaults["n_folds"])
    epochs = args.epochs or int(defaults["epochs"])
    run_dir = args.run_dir or f"runs/paths_{args.dataset}_v2"
    fundus = bool(defaults["fundus"]) or (
        args.dataset == "generic" and args.fundus_preprocessing
    )

    items = load_origin_items(
        args.dataset,
        root,
        labels_csv=args.labels_csv,
        image_column=args.image_column,
        label_column=args.label_column,
        image_dir=args.image_dir,
    )
    inferred = validate_ordinal_labels(items)
    num_classes = int(args.n_classes or defaults["n_classes"] or inferred)
    validate_ordinal_labels(items, num_classes)
    cfg = PathsConfig(
        dataset=args.dataset,
        n_classes=num_classes,
        n_folds=n_folds,
        val_fraction=args.val_fraction,
        run_dir=run_dir,
        preprocessing_version=(
            ORIGIN_PREPROCESSING_VERSION
            if fundus
            else "origin-generic-square-full-mask-v1"
        ),
        seed=args.seed,
        img_size=args.image_size,
        encoder=args.encoder,
        pretrained=True,
        evidence_scales=args.scales,
        projection_dim=args.projection_dim,
        grad_checkpoint=args.grad_checkpoint,
        mask_valid_fraction=args.mask_valid_fraction,
        reference_count=args.reference_count,
        atom_rate_init=args.atom_rate_init,
        prior_rate_init=args.prior_rate_init,
        boundary_scale_init=args.boundary_scale_init,
        total_rate_cap=args.total_rate_cap,
        prior_rate_cap=args.prior_rate_cap,
        boundary_scale_cap=args.boundary_scale_cap,
        rate_roundoff_margin=args.rate_roundoff_margin,
        atom_mode="cumulative",
        decision_rule=args.decision_rule,
        rps_weight=args.rps_weight,
        evidence_budget_weight=0.0,
        class_weighting="none",
        stratified_batches=False,
        pgf_probes=args.pgf_probes,
        correction_cap=args.correction_cap,
        correction_gain_init=args.correction_gain_init,
        correction_strength=args.correction_strength,
        correction_only_epochs=args.correction_only_epochs,
        paths_encoder_lr=args.paths_encoder_lr,
        paths_base_lr=args.paths_base_lr,
        paths_refiner_lr=args.paths_refiner_lr,
        risk_set_alpha=args.risk_set_alpha,
        warm_start_checkpoint=args.v3_checkpoint,
        warm_start_sha256=args.v3_sha256,
        certificate_samples=args.certificate_samples,
        epochs=epochs,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        # Retained inherited fields mirror the outer optimizer range; PATHS
        # itself uses the three explicit groups above.
        lr=args.paths_encoder_lr,
        head_lr=args.paths_refiner_lr,
        weight_decay=args.weight_decay,
        grad_clip_norm=args.grad_clip_norm,
        encoder_freeze_epochs=args.freeze_encoder_epochs,
        scheduler="plateau",
        lr_factor=args.lr_factor,
        lr_patience=args.lr_patience,
        lr_min=args.lr_min,
        early_stopping_patience=args.early_stopping_patience,
        checkpoint_selection=args.checkpoint_selection,
        selection_qwk_weight=args.selection_qwk_weight,
        amp=args.amp,
        amp_init_scale=args.amp_init_scale,
        amp_unfreeze_scale=args.amp_unfreeze_scale,
        amp_growth_interval=args.amp_growth_interval,
        amp_max_consecutive_skips=args.amp_max_consecutive_skips,
        resume=args.resume,
        labels_csv=args.labels_csv,
        image_column=args.image_column,
        label_column=args.label_column,
        image_dir=args.image_dir,
    )
    log = _setup_logging(cfg.run_dir)
    set_seed(cfg.seed)
    device = torch.device(
        "cuda" if torch.cuda.is_available() else
        "mps" if torch.backends.mps.is_available() else "cpu"
    )
    folds = parse_folds(args.folds, cfg.n_folds)
    log.info("PATHS configuration: %s", asdict(cfg))
    log.info(
        "dataset=%s images=%d classes=%s device=%s scope=%s",
        cfg.dataset,
        len(items),
        class_histogram(items, cfg.n_classes),
        device,
        "outer_test_after_selection" if args.include_test else "inner_validation_only",
    )
    if args.include_test:
        log.warning("OUTER TEST UNLOCKED; do not use it for further model selection")

    fold_results: list[dict[str, Any]] = []
    for fold in folds:
        set_seed(cfg.seed + fold)
        train_items, val_items, test_items = split_origin_items(
            cfg.dataset,
            items,
            fold,
            n_folds=cfg.n_folds,
            val_fraction=cfg.val_fraction,
            seed=cfg.seed,
        )
        if not split_paths_are_disjoint(train_items, val_items, test_items):
            raise RuntimeError(f"fold {fold} contains overlapping paths")
        signature = split_signature(
            ("train", train_items),
            ("validation", val_items),
            ("locked_test", test_items),
        )
        fold_dir = Path(cfg.run_dir) / f"fold{fold}"
        manifest = {
            "schema": "paths-split-v2",
            "dataset": cfg.dataset,
            "fold": fold,
            "signature": signature,
            "evaluation_scope": (
                "outer_test_after_selection" if args.include_test
                else "inner_validation_only"
            ),
            "counts": {
                "train": len(train_items),
                "validation": len(val_items),
                "locked_test": len(test_items),
            },
            "histograms": {
                "train": class_histogram(train_items, cfg.n_classes),
                "validation": class_histogram(val_items, cfg.n_classes),
                "locked_test": class_histogram(test_items, cfg.n_classes),
            },
        }
        guard_and_write_split_manifest(fold_dir, manifest, resume=cfg.resume)
        log.info("fold=%d split=%s", fold, manifest)
        loaders = make_origin_loaders(
            train_items,
            val_items,
            test_items,
            image_size=cfg.img_size,
            batch_size=cfg.batch_size,
            num_workers=0 if device.type == "mps" else cfg.num_workers,
            pin_memory=device.type == "cuda",
            seed=cfg.seed + fold,
            fundus=fundus,
            stratified=False,
        )
        model = build_paths_model(
            num_classes=cfg.n_classes,
            encoder_name=cfg.encoder,
            # Every V3 tensor is restored below; do not download/consult an
            # external pretrained cache before the hash-bound load.
            pretrained=False,
            evidence_scales=cfg.evidence_scales,
            projection_dim=cfg.projection_dim,
            reference_count=cfg.reference_count,
            atom_mode=cfg.atom_mode,
            hybrid_cumulative_init=cfg.hybrid_cumulative_init,
            atom_rate_init=cfg.atom_rate_init,
            prior_rate_init=cfg.prior_rate_init,
            boundary_scale_init=cfg.boundary_scale_init,
            total_rate_cap=cfg.total_rate_cap,
            prior_rate_cap=cfg.prior_rate_cap,
            boundary_scale_cap=cfg.boundary_scale_cap,
            rate_roundoff_margin=cfg.rate_roundoff_margin,
            evidence_dropout=cfg.evidence_dropout,
            mask_valid_fraction=cfg.mask_valid_fraction,
            grad_checkpoint=cfg.grad_checkpoint,
            paths_probe_z=cfg.pgf_probes,
            paths_correction_cap=cfg.correction_cap,
            paths_gain_init=cfg.correction_gain_init,
            paths_strength=cfg.correction_strength,
        )
        trainer = PathsTrainer(
            model,
            *loaders,
            cfg,
            fold_dir,
            fold=fold,
            split_signature=signature,
            device=device,
        )
        result = trainer.fit(evaluate_test=args.include_test)
        record = {"fold": fold, **result}
        fold_results.append(record)
        write_json_atomic(fold_dir / "result.json", record)

    summary_name = (
        "final_results.json"
        if sorted(folds) == list(range(cfg.n_folds))
        else "final_results_folds_" + "_".join(map(str, folds)) + ".json"
    )
    write_json_atomic(Path(cfg.run_dir) / summary_name, fold_results)
    rows = [{"fold": item["fold"], **_scalar_summary(item)} for item in fold_results]
    keys = sorted({key for row in rows for key in row if key != "fold"})
    if keys:
        with (Path(cfg.run_dir) / summary_name.replace(".json", ".csv")).open(
            "w", newline=""
        ) as stream:
            writer = csv.DictWriter(stream, fieldnames=["fold", *keys])
            writer.writeheader()
            writer.writerows(rows)
    log.info("PATHS results written to %s", Path(cfg.run_dir) / summary_name)


if __name__ == "__main__":
    main()
