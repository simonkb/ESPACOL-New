#!/usr/bin/env python3
"""Train MOSAIC on APTOS first, then EyePACS after the architecture passes."""

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

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))

from configs.config import MOSAICConfig
from Datasets.mosaic_data import (
    MOSAIC_PREPROCESSING_VERSION,
    aptos_fold,
    class_histogram,
    eyepacs_fold,
    load_aptos_items,
    load_eyepacs_items,
    make_mosaic_loaders,
)
from models.mosaic_model import build_mosaic_model
from training.mosaic_trainer import MosaicTrainer


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
    return logging.getLogger("train_mosaic")


def parse_folds(value: str, n_folds: int) -> list[int]:
    if value.lower() == "all":
        return list(range(n_folds))
    folds = [int(part.strip()) for part in value.split(",") if part.strip()]
    if not folds:
        raise ValueError("at least one fold must be requested")
    invalid = [fold for fold in folds if fold < 0 or fold >= n_folds]
    if invalid:
        raise ValueError(f"invalid folds {invalid}; expected 0..{n_folds - 1}")
    if len(set(folds)) != len(folds):
        raise ValueError(f"duplicate folds are not allowed: {folds}")
    return folds


def default_run_dir(dataset: str) -> str:
    """Keep APTOS and EyePACS artifacts separate unless explicitly overridden."""

    if dataset not in {"aptos", "dr"}:
        raise ValueError(f"unknown MOSAIC dataset {dataset!r}")
    return f"runs/mosaic_{dataset}"


def summary_filename(folds: list[int], n_folds: int) -> str:
    """Give independent fold jobs collision-free root summaries."""

    if sorted(folds) == list(range(n_folds)):
        return "final_results.csv"
    suffix = "_".join(str(fold) for fold in folds)
    return f"final_results_folds_{suffix}.csv"


def json_safe(value):
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    return value


def write_json_atomic(path: str | Path, value) -> None:
    """Write a JSON artifact without exposing a partially written file."""

    destination = Path(path)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w") as stream:
            json.dump(json_safe(value), stream, indent=2)
            stream.write("\n")
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def split_signature(*named_splits) -> str:
    """Content-address one fold without depending on its absolute data root."""

    digest = hashlib.sha256()
    for split_name, items in named_splits:
        digest.update(f"[{split_name}]\n".encode("utf-8"))
        canonical = sorted((Path(path).name, int(label)) for path, label in items)
        for image_name, label in canonical:
            digest.update(f"{image_name}\t{label}\n".encode("utf-8"))
    return digest.hexdigest()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="MOSAIC minimum ordinal proof training",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dataset", choices=("aptos", "dr"), default="aptos")
    parser.add_argument("--data_root", default=None)
    parser.add_argument("--labels_csv", default=None)
    parser.add_argument(
        "--run_dir",
        default=None,
        help="Output root; defaults to runs/mosaic_<dataset>.",
    )
    parser.add_argument("--folds", default="0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--skip_test",
        action="store_true",
        help="Do not evaluate the outer test split (required during architecture selection)",
    )

    parser.add_argument("--image_size", type=int, default=896)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=35)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--local_stage", choices=("rf_small", "rf_medium", "rf_large"), default="rf_medium")
    parser.add_argument("--evidence_dim", type=int, default=128)
    parser.add_argument("--grad_checkpoint", action="store_true")
    parser.add_argument("--no_pretrained", action="store_true")

    parser.add_argument("--max_count", type=int, default=32)
    parser.add_argument("--count_block_size", type=int, default=64)
    parser.add_argument("--proof_epsilon", type=float, default=0.02)
    parser.add_argument("--necessity_fraction", type=float, default=0.5)
    parser.add_argument("--normal_expected_count", type=float, default=0.5)
    parser.add_argument("--dense_warmup_epochs", type=int, default=4)
    parser.add_argument("--proof_ramp_epochs", type=int, default=4)
    parser.add_argument("--dense_loss_weight", type=float, default=0.1)

    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--head_lr", type=float, default=5e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--early_stop_patience", type=int, default=10)
    parser.add_argument(
        "--no_amp",
        action="store_true",
        help="Disable CUDA mixed precision for numerical A/B diagnosis.",
    )
    parser.add_argument("--amp_init_scale", type=float, default=8192.0)
    parser.add_argument("--amp_growth_interval", type=int, default=2000)
    parser.add_argument("--amp_max_consecutive_skips", type=int, default=8)
    parser.add_argument("--transition_weighting", choices=("effective_num", "inverse_frequency", "none"), default="effective_num")
    parser.add_argument("--effective_num_beta", type=float, default=0.999)
    parser.add_argument("--transition_weight_cap", type=float, default=10.0)
    parser.add_argument(
        "--decision_rule",
        choices=(
            "rounded_expected",
            "class_map",
            "posterior_median",
            "deweighted_mean_round",
            "deweighted_class_map",
            "deweighted_posterior_median",
        ),
        default="posterior_median",
        help=(
            "Parameter-free point decision derived from the selected proof; "
            "posterior_median is prospectively locked after the APTOS and "
            "EyePACS development-fold audits."
        ),
    )
    parser.add_argument(
        "--stratified_batches",
        action="store_true",
        help="Usually leave off: the at-risk likelihood already corrects imbalance.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    n_folds = 5 if args.dataset == "aptos" else 10
    run_dir = args.run_dir or default_run_dir(args.dataset)
    root = args.data_root or (
        "Datasets/aptos2019-blindness-detection" if args.dataset == "aptos" else "Datasets/DR"
    )
    cfg = MOSAICConfig(
        dataset=args.dataset.upper(),
        n_folds=n_folds,
        run_dir=run_dir,
        seed=args.seed,
        resume=args.resume,
        img_size=args.image_size,
        batch_size=args.batch_size,
        epochs=args.epochs,
        num_workers=args.num_workers,
        local_stage=args.local_stage,
        evidence_dim=args.evidence_dim,
        grad_checkpoint=args.grad_checkpoint,
        pretrained=not args.no_pretrained,
        max_count=args.max_count,
        count_block_size=args.count_block_size,
        proof_epsilon=args.proof_epsilon,
        necessity_fraction=args.necessity_fraction,
        normal_expected_count=args.normal_expected_count,
        dense_warmup_epochs=args.dense_warmup_epochs,
        proof_ramp_epochs=args.proof_ramp_epochs,
        dense_loss_weight=args.dense_loss_weight,
        lr=args.lr,
        head_lr=args.head_lr,
        weight_decay=args.weight_decay,
        early_stop_patience=args.early_stop_patience,
        amp=not args.no_amp,
        amp_init_scale=args.amp_init_scale,
        amp_growth_interval=args.amp_growth_interval,
        amp_max_consecutive_skips=args.amp_max_consecutive_skips,
        transition_weighting=args.transition_weighting,
        effective_num_beta=args.effective_num_beta,
        transition_weight_cap=args.transition_weight_cap,
        decision_rule=args.decision_rule,
        stratified=args.stratified_batches,
    )
    if cfg.preprocessing_version != MOSAIC_PREPROCESSING_VERSION:
        raise RuntimeError(
            "MOSAICConfig.preprocessing_version does not match the active data "
            f"pipeline ({cfg.preprocessing_version!r} != "
            f"{MOSAIC_PREPROCESSING_VERSION!r})"
        )
    log = setup_logging(cfg.run_dir)
    set_seed(cfg.seed)
    device = torch.device(
        "cuda" if torch.cuda.is_available() else
        "mps" if torch.backends.mps.is_available() else "cpu"
    )
    if args.dataset == "aptos":
        items = load_aptos_items(root, args.labels_csv)
    else:
        items = load_eyepacs_items(root, args.labels_csv)
    log.info("MOSAIC configuration: %s", asdict(cfg))
    log.info("dataset=%s images=%d classes=%s device=%s", args.dataset, len(items), class_histogram(items), device)

    fold_results = []
    fold_indices = parse_folds(args.folds, n_folds)
    for fold in fold_indices:
        set_seed(cfg.seed + fold)
        if args.dataset == "aptos":
            train_items, val_items, test_items = aptos_fold(
                items,
                fold,
                n_folds=n_folds,
                val_fraction=cfg.val_fraction,
                seed=cfg.seed,
            )
        else:
            train_items, val_items, test_items = eyepacs_fold(
                items,
                fold,
                n_folds=n_folds,
                val_fraction=cfg.val_fraction,
                seed=cfg.seed,
            )
        log.info(
            "fold=%d train=%d val=%d test=%d train_classes=%s",
            fold,
            len(train_items),
            len(val_items),
            len(test_items),
            class_histogram(train_items),
        )
        loaders = make_mosaic_loaders(
            train_items,
            val_items,
            test_items,
            image_size=cfg.img_size,
            batch_size=cfg.batch_size,
            num_workers=0 if device.type == "mps" else cfg.num_workers,
            pin_memory=device.type == "cuda",
            seed=cfg.seed + fold,
            stratified=cfg.stratified,
        )
        model = build_mosaic_model(
            num_classes=cfg.n_classes,
            image_size=cfg.img_size,
            local_stage=cfg.local_stage,
            local_dim=cfg.evidence_dim,
            pretrained=cfg.pretrained,
            grad_checkpoint=cfg.grad_checkpoint,
            initial_abnormal_count=cfg.normal_expected_count,
            max_count=cfg.max_count,
            sufficiency_tolerance=cfg.proof_epsilon,
            complement_suppression=cfg.necessity_fraction,
            count_implementation=cfg.count_implementation,
            count_block_size=cfg.count_block_size,
        )
        fold_dir = os.path.join(cfg.run_dir, f"fold{fold}")
        trainer = MosaicTrainer(
            model,
            *loaders,
            cfg,
            fold_dir,
            [label for _, label in train_items],
            fold=fold,
            split_signature=split_signature(
                ("train", train_items),
                ("validation", val_items),
                ("test", test_items),
            ),
            device=device,
        )
        result = trainer.fit(evaluate_test=not args.skip_test)
        fold_results.append({"fold": fold, **result})
        write_json_atomic(
            os.path.join(fold_dir, "best_validation_metrics.json"),
            result["best_validation_metrics"],
        )
        if result["test_evaluated"]:
            write_json_atomic(os.path.join(fold_dir, "test_metrics.json"), result)

    summary_path = os.path.join(
        cfg.run_dir,
        summary_filename(fold_indices, n_folds),
    )
    scalar_keys = sorted(
        key for key, value in fold_results[0].items()
        if key != "fold" and isinstance(value, (float, int))
    )
    with open(summary_path, "w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["fold", *scalar_keys])
        writer.writeheader()
        for result in fold_results:
            writer.writerow({key: result[key] for key in ["fold", *scalar_keys]})
        writer.writerow(
            {
                "fold": "mean",
                **{key: float(np.mean([item[key] for item in fold_results])) for key in scalar_keys},
            }
        )
        writer.writerow(
            {
                "fold": "std",
                **{key: float(np.std([item[key] for item in fold_results])) for key in scalar_keys},
            }
        )
    log.info("results written to %s", summary_path)


if __name__ == "__main__":
    main()
