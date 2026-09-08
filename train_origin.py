#!/usr/bin/env python3
"""Train ORIGIN with a validation-first, test-locked protocol.

The outer cross-validation fold is constructed and content-addressed at
startup, but it is never evaluated unless ``--include_test`` is supplied
explicitly.  This makes the safe architecture-development behaviour the
command-line default rather than a convention that can be forgotten.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import os
import random
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))

from configs.origin_config import OriginConfig
from Datasets.origin_data import (
    ORIGIN_PREPROCESSING_VERSION,
    class_histogram,
    load_origin_items,
    make_origin_loaders,
    split_origin_items,
    split_paths_are_disjoint,
    validate_ordinal_labels,
)
from models.origin import build_origin_model
from training.origin_trainer import OriginTrainer


DATASET_DEFAULTS = {
    "aptos": {
        "root": "Datasets/aptos2019-blindness-detection",
        "n_classes": 5,
        "n_folds": 5,
        "fundus": True,
        "epochs": 35,
    },
    "dr": {
        "root": "Datasets/DR",
        "n_classes": 5,
        "n_folds": 10,
        "fundus": True,
        "epochs": 75,
    },
    "busi": {
        "root": "Datasets/BUSI",
        "n_classes": 3,
        "n_folds": 5,
        "fundus": False,
        "epochs": 50,
    },
    "generic": {
        "root": "Datasets/generic",
        "n_classes": None,
        "n_folds": 5,
        "fundus": False,
        "epochs": 50,
    },
}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def setup_logging(run_dir: str) -> logging.Logger:
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
    return logging.getLogger("train_origin")


def parse_folds(value: str, n_folds: int) -> list[int]:
    if value.lower() == "all":
        return list(range(n_folds))
    folds = [int(part.strip()) for part in value.split(",") if part.strip()]
    if not folds:
        raise ValueError("at least one fold must be requested")
    if len(folds) != len(set(folds)):
        raise ValueError(f"duplicate folds are not allowed: {folds}")
    invalid = [fold for fold in folds if fold < 0 or fold >= n_folds]
    if invalid:
        raise ValueError(f"invalid folds {invalid}; expected 0..{n_folds - 1}")
    return folds


def parse_scales(value: str) -> tuple[str, ...]:
    scales: list[str] = []
    for token in value.split(","):
        token = token.strip().lower()
        if not token:
            continue
        scales.append(token if token.startswith("s") else f"s{token}")
    if not scales:
        raise argparse.ArgumentTypeError("at least one evidence scale is required")
    allowed = {"s4", "s8", "s16", "s32"}
    invalid = sorted(set(scales) - allowed)
    if invalid:
        raise argparse.ArgumentTypeError(
            f"invalid evidence scales {invalid}; choose from s4,s8,s16,s32"
        )
    if len(scales) != len(set(scales)):
        raise argparse.ArgumentTypeError("evidence scales must be unique")
    return tuple(scales)


def split_signature(*named_splits) -> str:
    """Content-address a split using root-independent relative image paths.

    A basename alone is not an image identity: generic datasets commonly use
    repeated names such as ``image.png`` in different class or patient
    directories.  We therefore retain the complete path below the common
    dataset root while deliberately excluding the machine-specific prefix.
    """

    materialized = [
        (str(split_name), list(items)) for split_name, items in named_splits
    ]
    resolved_paths = [
        Path(path).expanduser().resolve(strict=False)
        for _, items in materialized
        for path, _ in items
    ]
    if not resolved_paths:
        raise ValueError("cannot sign an empty collection of splits")
    common_root = Path(os.path.commonpath([str(path) for path in resolved_paths]))
    if common_root in resolved_paths:
        common_root = common_root.parent

    digest = hashlib.sha256()
    for split_name, items in materialized:
        digest.update(f"[{split_name}]\n".encode())
        identities: list[tuple[str, int, Path]] = []
        for path, label in items:
            resolved = Path(path).expanduser().resolve(strict=False)
            try:
                relative = resolved.relative_to(common_root).as_posix()
            except ValueError:  # Different drives on Windows, for portability.
                relative = resolved.as_posix()
            identities.append((relative, int(label), resolved))
        for relative, label, resolved in sorted(
            identities, key=lambda item: (item[0], item[1])
        ):
            digest.update(f"{relative}\t{label}\n".encode())
            if resolved.is_file():
                digest.update(str(resolved.stat().st_size).encode("ascii"))
                digest.update(b"\0")
                with resolved.open("rb") as stream:
                    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                        digest.update(chunk)
            else:
                # Unit tests use virtual paths. Real training launchers verify
                # every image before this function is called.
                digest.update(b"<missing>")
            digest.update(b"\0")
    return digest.hexdigest()


_PROTECTED_FOLD_ARTIFACTS = (
    "best.pth",
    "last.pth",
    "history.csv",
    "result.json",
    "validation_certificates.json",
)


def guard_and_write_split_manifest(
    fold_dir: str | Path,
    manifest: dict[str, Any],
    *,
    resume: bool,
) -> None:
    """Persist split provenance without overwriting an authoritative run."""

    directory = Path(fold_dir)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "split_manifest.json"
    if resume:
        if not path.is_file():
            raise FileNotFoundError(
                f"resume requires the original split manifest: {path}"
            )
        with path.open() as stream:
            existing = json.load(stream)
        identity_fields = ("dataset", "fold", "signature", "counts", "histograms")
        mismatches = [
            field for field in identity_fields
            if existing.get(field) != manifest.get(field)
        ]
        if mismatches:
            raise ValueError(
                "resume split manifest mismatch in " + ", ".join(mismatches)
            )
        return

    existing_artifacts = [
        name for name in _PROTECTED_FOLD_ARTIFACTS if (directory / name).exists()
    ]
    if existing_artifacts:
        raise FileExistsError(
            "fresh run would overwrite fold artifacts: "
            + ", ".join(existing_artifacts)
        )
    write_json_atomic(path, manifest)


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    return value


def write_json_atomic(path: str | Path, value: Any) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w") as stream:
            json.dump(_json_safe(value), stream, indent=2)
            stream.write("\n")
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="ORIGIN conserved ordinal generator training",
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
    test_group = parser.add_mutually_exclusive_group()
    test_group.add_argument(
        "--include_test",
        action="store_true",
        help="evaluate the locked outer fold after validation model selection",
    )
    test_group.add_argument(
        "--skip_test",
        action="store_true",
        help="explicit spelling of the default validation-only behaviour",
    )
    parser.add_argument(
        "--fundus_preprocessing",
        action="store_true",
        help="apply canonical retinal support to a generic CSV dataset",
    )

    model = parser.add_argument_group("model")
    model.add_argument("--image_size", type=int, default=640)
    model.add_argument("--encoder", choices=("convnext_tiny",), default="convnext_tiny")
    model.add_argument("--no_pretrained", action="store_true")
    model.add_argument("--scales", type=parse_scales, default=parse_scales("s4,s8,s16,s32"))
    model.add_argument("--projection_dim", type=int, default=128)
    model.add_argument("--reference_count", type=float, default=4096.0)
    model.add_argument("--atom_rate_init", type=float, default=1e-6)
    model.add_argument("--prior_rate_init", type=float, default=1e-4)
    model.add_argument("--boundary_scale_init", type=float, default=1.0)
    model.add_argument(
        "--atom_mode",
        choices=("cumulative", "independent", "hybrid"),
        default="cumulative",
    )
    model.add_argument("--hybrid_cumulative_init", type=float, default=0.9)
    model.add_argument("--evidence_dropout", type=float, default=0.0)
    model.add_argument("--mask_valid_fraction", type=float, default=0.5)
    model.add_argument("--grad_checkpoint", action="store_true")
    model.add_argument(
        "--decision_rule",
        choices=("posterior_median", "class_map", "rounded_expected"),
        default="class_map",
    )

    loss = parser.add_argument_group("loss")
    loss.add_argument("--rps_weight", type=float, default=0.25)
    loss.add_argument("--evidence_budget_weight", type=float, default=0.0)
    loss.add_argument("--evidence_budget_delay_epochs", type=int, default=5)
    loss.add_argument(
        "--class_weighting",
        choices=("none", "effective_num", "inverse_frequency"),
        default="none",
    )
    loss.add_argument("--effective_num_beta", type=float, default=0.999)
    loss.add_argument("--class_weight_cap", type=float, default=10.0)
    loss.add_argument(
        "--allow_weighted_likelihood",
        action="store_true",
        help="acknowledge that weighted NLL changes the population likelihood target",
    )

    training = parser.add_argument_group("training")
    training.add_argument("--epochs", type=int, default=None)
    training.add_argument("--batch_size", type=int, default=8)
    training.add_argument("--num_workers", type=int, default=4)
    training.add_argument("--lr", type=float, default=1e-4)
    training.add_argument("--head_lr", type=float, default=5e-4)
    training.add_argument("--weight_decay", type=float, default=1e-5)
    training.add_argument("--grad_clip_norm", type=float, default=5.0)
    training.add_argument("--freeze_encoder_epochs", type=int, default=2)
    training.add_argument(
        "--scheduler",
        choices=("plateau",),
        default="plateau",
        help="prospectively fixed ReduceLROnPlateau scheduler on validation loss",
    )
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
    training.add_argument("--stratified_batches", action="store_true")
    amp_group = training.add_mutually_exclusive_group()
    amp_group.add_argument("--amp", dest="amp", action="store_true")
    amp_group.add_argument("--no_amp", dest="amp", action="store_false")
    amp_group.set_defaults(amp=True)
    training.add_argument("--amp_init_scale", type=float, default=4096.0)
    training.add_argument(
        "--amp_unfreeze_scale",
        type=float,
        default=256.0,
        help="one-time AMP loss scale used when the frozen encoder enters backprop",
    )
    training.add_argument("--amp_growth_interval", type=int, default=2000)
    training.add_argument("--amp_max_consecutive_skips", type=int, default=8)
    return parser


def _scalar_summary(result: dict[str, Any]) -> dict[str, float | int]:
    return {
        key: value
        for key, value in result.items()
        if isinstance(value, (float, int)) and not isinstance(value, bool)
    }


def main() -> None:
    args = build_parser().parse_args()
    defaults = DATASET_DEFAULTS[args.dataset]
    data_root = args.data_root or str(defaults["root"])
    n_folds = args.n_folds or int(defaults["n_folds"])
    epochs = args.epochs or int(defaults["epochs"])
    run_dir = args.run_dir or f"runs/origin_{args.dataset}"
    fundus = bool(defaults["fundus"]) or (
        args.dataset == "generic" and args.fundus_preprocessing
    )

    items = load_origin_items(
        args.dataset,
        data_root,
        labels_csv=args.labels_csv,
        image_column=args.image_column,
        label_column=args.label_column,
        image_dir=args.image_dir,
    )
    inferred_classes = validate_ordinal_labels(items)
    requested_classes = args.n_classes or defaults["n_classes"] or inferred_classes
    validate_ordinal_labels(items, int(requested_classes))
    preprocessing_version = (
        ORIGIN_PREPROCESSING_VERSION
        if fundus
        else "origin-generic-square-full-mask-v1"
    )
    cfg = OriginConfig(
        dataset=args.dataset,
        n_classes=int(requested_classes),
        n_folds=n_folds,
        val_fraction=args.val_fraction,
        run_dir=run_dir,
        preprocessing_version=preprocessing_version,
        seed=args.seed,
        img_size=args.image_size,
        encoder=args.encoder,
        pretrained=not args.no_pretrained,
        evidence_scales=args.scales,
        projection_dim=args.projection_dim,
        grad_checkpoint=args.grad_checkpoint,
        mask_valid_fraction=args.mask_valid_fraction,
        reference_count=args.reference_count,
        atom_rate_init=args.atom_rate_init,
        prior_rate_init=args.prior_rate_init,
        boundary_scale_init=args.boundary_scale_init,
        atom_mode=args.atom_mode,
        hybrid_cumulative_init=args.hybrid_cumulative_init,
        evidence_dropout=args.evidence_dropout,
        force_decoder_fp64=True,
        decision_rule=args.decision_rule,
        rps_weight=args.rps_weight,
        evidence_budget_weight=args.evidence_budget_weight,
        evidence_budget_delay_epochs=args.evidence_budget_delay_epochs,
        class_weighting=args.class_weighting,
        effective_num_beta=args.effective_num_beta,
        class_weight_cap=args.class_weight_cap,
        allow_weighted_likelihood=args.allow_weighted_likelihood,
        epochs=epochs,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        lr=args.lr,
        head_lr=args.head_lr,
        weight_decay=args.weight_decay,
        grad_clip_norm=args.grad_clip_norm,
        encoder_freeze_epochs=args.freeze_encoder_epochs,
        scheduler=args.scheduler,
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
        stratified_batches=args.stratified_batches,
        labels_csv=args.labels_csv,
        image_column=args.image_column,
        label_column=args.label_column,
        image_dir=args.image_dir,
    )
    if cfg.class_weighting != "none" and not cfg.allow_weighted_likelihood:
        raise ValueError(
            "class weighting changes the likelihood target; add "
            "--allow_weighted_likelihood to acknowledge this explicitly"
        )

    log = setup_logging(cfg.run_dir)
    set_seed(cfg.seed)
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    fold_indices = parse_folds(args.folds, cfg.n_folds)
    log.info("ORIGIN configuration: %s", asdict(cfg))
    log.info(
        "dataset=%s images=%d classes=%s device=%s evaluation_scope=%s",
        cfg.dataset,
        len(items),
        class_histogram(items, cfg.n_classes),
        device,
        "outer_test_after_selection" if args.include_test else "inner_validation_only",
    )
    if args.include_test:
        log.warning(
            "OUTER TEST UNLOCKED explicitly by --include_test; do not use these "
            "results for further architecture or hyperparameter selection"
        )

    fold_results: list[dict[str, Any]] = []
    for fold in fold_indices:
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
            raise RuntimeError(f"fold {fold} contains overlapping image paths")
        signature = split_signature(
            ("train", train_items),
            ("validation", val_items),
            ("locked_test", test_items),
        )
        fold_dir = Path(cfg.run_dir) / f"fold{fold}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        split_manifest = {
            "schema": "origin-split-v1",
            "dataset": cfg.dataset,
            "fold": fold,
            "signature": signature,
            "evaluation_scope": (
                "outer_test_after_selection" if args.include_test else "inner_validation_only"
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
        guard_and_write_split_manifest(
            fold_dir,
            split_manifest,
            resume=cfg.resume,
        )
        log.info("fold=%d split=%s", fold, split_manifest)

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
            stratified=cfg.stratified_batches,
        )
        model = build_origin_model(
            num_classes=cfg.n_classes,
            encoder_name=cfg.encoder,
            # A resume immediately restores the complete encoder state, so it
            # must not depend on an external torchvision weight download/cache.
            pretrained=cfg.pretrained and not cfg.resume,
            evidence_scales=cfg.evidence_scales,
            projection_dim=cfg.projection_dim,
            reference_count=cfg.reference_count,
            atom_rate_init=cfg.atom_rate_init,
            prior_rate_init=cfg.prior_rate_init,
            boundary_scale_init=cfg.boundary_scale_init,
            atom_mode=cfg.atom_mode,
            hybrid_cumulative_init=cfg.hybrid_cumulative_init,
            evidence_dropout=cfg.evidence_dropout,
            mask_valid_fraction=cfg.mask_valid_fraction,
            grad_checkpoint=cfg.grad_checkpoint,
        )
        trainer = OriginTrainer(
            model,
            *loaders,
            cfg,
            str(fold_dir),
            fold=fold,
            split_signature=signature,
            device=device,
        )
        result = trainer.fit(evaluate_test=args.include_test)
        fold_result = {"fold": fold, **result}
        fold_results.append(fold_result)
        write_json_atomic(fold_dir / "result.json", fold_result)

    summary_name = (
        "final_results.json"
        if sorted(fold_indices) == list(range(cfg.n_folds))
        else "final_results_folds_" + "_".join(map(str, fold_indices)) + ".json"
    )
    write_json_atomic(Path(cfg.run_dir) / summary_name, fold_results)
    scalar_rows = [{"fold": item["fold"], **_scalar_summary(item)} for item in fold_results]
    scalar_keys = sorted({key for row in scalar_rows for key in row if key != "fold"})
    if scalar_keys:
        csv_path = Path(cfg.run_dir) / summary_name.replace(".json", ".csv")
        with csv_path.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=["fold", *scalar_keys])
            writer.writeheader()
            writer.writerows(scalar_rows)
    log.info("ORIGIN results written to %s", Path(cfg.run_dir) / summary_name)


if __name__ == "__main__":
    main()
