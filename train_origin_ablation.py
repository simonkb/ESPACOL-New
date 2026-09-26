#!/usr/bin/env python3
"""Train one pre-registered ORIGIN fold ablation without opening the outer test.

The script deliberately reuses ORIGIN's audited data, augmentation, split,
optimizer, scheduler, and checkpoint-selection code.  A checked-in variant
registry owns the small set of allowed architectural/objective changes.  The
locked outer fold is evaluated later by one suite-level release job, only after
every variant has completed inner-validation model selection.
"""

from __future__ import annotations

import csv
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

from configs.origin_ablation_config import OriginAblationConfig
from Datasets.origin_data import (
    ORIGIN_PREPROCESSING_VERSION,
    class_histogram,
    load_origin_items,
    make_origin_loaders,
    split_origin_items,
    split_paths_are_disjoint,
    validate_ordinal_labels,
)
from models.origin_ablation import build_origin_ablation_model
from scripts.origin_fold9_ablation_common import (
    TRAINING_STREAM_SEED,
    get_variant_spec,
)
from train_origin import (
    DATASET_DEFAULTS,
    build_parser as build_origin_parser,
    guard_and_write_split_manifest,
    parse_folds,
    set_seed,
    setup_logging,
    split_signature,
    write_json_atomic,
)
from training.origin_ablation_trainer import OriginAblationTrainer


def build_parser():
    parser = build_origin_parser()
    parser.description = (
        "Train one locked ORIGIN ablation using inner validation only"
    )
    parser.add_argument(
        "--variant",
        required=True,
        help="pre-registered variant name from origin_fold9_ablation_common.py",
    )
    return parser


def _scalar_summary(result: dict[str, Any]) -> dict[str, float | int]:
    return {
        key: value
        for key, value in result.items()
        if isinstance(value, (float, int)) and not isinstance(value, bool)
    }


def _resolved_model_settings(args: Any, variant: str) -> dict[str, Any]:
    spec = get_variant_spec(variant)
    overrides = dict(spec.get("config_overrides", {}))
    settings: dict[str, Any] = {
        "evidence_scales": tuple(args.scales),
        "atom_mode": str(args.atom_mode),
        "rps_weight": float(args.rps_weight),
        "evidence_budget_weight": float(args.evidence_budget_weight),
    }
    settings.update(overrides)
    if "evidence_scales" in settings:
        settings["evidence_scales"] = tuple(settings["evidence_scales"])
    return settings


def main() -> None:
    args = build_parser().parse_args()
    if args.include_test:
        raise ValueError(
            "train_origin_ablation.py never evaluates the outer test; use "
            "the suite-level outer-release job after every variant is frozen"
        )
    if not args.skip_test:
        raise ValueError(
            "ablation training requires the explicit --skip_test authorization"
        )

    variant_spec = get_variant_spec(args.variant)
    variant = str(variant_spec["name"])
    settings = _resolved_model_settings(args, variant)
    defaults = DATASET_DEFAULTS[args.dataset]
    data_root = args.data_root or str(defaults["root"])
    n_folds = args.n_folds or int(defaults["n_folds"])
    epochs = args.epochs or int(defaults["epochs"])
    run_dir = args.run_dir or f"runs/origin_ablation_{variant}_{args.dataset}"
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
    cfg = OriginAblationConfig(
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
        evidence_scales=settings["evidence_scales"],
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
        atom_mode=settings["atom_mode"],
        hybrid_cumulative_init=args.hybrid_cumulative_init,
        evidence_dropout=args.evidence_dropout,
        force_decoder_fp64=True,
        decision_rule=args.decision_rule,
        rps_weight=settings["rps_weight"],
        evidence_budget_weight=settings["evidence_budget_weight"],
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
        ablation_variant=variant,
        ablation_description=str(variant_spec["description"]),
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
    log.info("ORIGIN ablation configuration: %s", asdict(cfg))
    log.info(
        "variant=%s dataset=%s images=%d classes=%s device=%s "
        "evaluation_scope=inner_validation_only",
        variant,
        cfg.dataset,
        len(items),
        class_histogram(items, cfg.n_classes),
        device,
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
            "schema": "origin-fold9-ablation-split-v1",
            "dataset": cfg.dataset,
            "fold": fold,
            "ablation_variant": variant,
            "signature": signature,
            "evaluation_scope": "inner_validation_only_outer_locked",
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
            (),  # The locked outer partition is not materialized by workers.
            image_size=cfg.img_size,
            batch_size=cfg.batch_size,
            num_workers=0 if device.type == "mps" else cfg.num_workers,
            pin_memory=device.type == "cuda",
            seed=cfg.seed + fold,
            fundus=fundus,
            stratified=cfg.stratified_batches,
        )
        model = build_origin_ablation_model(cfg)
        parameter_count = sum(parameter.numel() for parameter in model.parameters())
        trainable_parameter_count = sum(
            parameter.numel() for parameter in model.parameters() if parameter.requires_grad
        )
        trainer = OriginAblationTrainer(
            model,
            *loaders,
            cfg,
            str(fold_dir),
            fold=fold,
            split_signature=signature,
            device=device,
        )
        # Keep the stochastic training stream identical across variants. Model
        # constructors consume different amounts of RNG state (for example, a
        # pooled head has a different parameter topology from ORIGIN). Reset
        # only after construction so shuffle order, worker seeds, augmentation,
        # stochastic depth, and dropout do not become hidden ablation changes.
        training_stream_seed = cfg.seed + fold + 100_000
        if cfg.dataset == "dr" and fold == 9 and training_stream_seed != TRAINING_STREAM_SEED:
            raise AssertionError("fold-9 training stream differs from the locked protocol")
        set_seed(training_stream_seed)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        started = time.monotonic()
        result = trainer.fit(evaluate_test=False)
        wall_time = time.monotonic() - started
        result.update(
            {
                "ablation_variant": variant,
                "ablation_description": cfg.ablation_description,
                "parameter_count": int(parameter_count),
                "trainable_parameter_count": int(trainable_parameter_count),
                "training_wall_time_seconds": float(wall_time),
                "training_efficiency_scope": (
                    "resumed_segment_only_not_cross_variant_comparable"
                    if cfg.resume
                    else "complete_training_process"
                ),
                "training_stream_seed": int(training_stream_seed),
                "training_peak_cuda_memory_bytes": (
                    int(torch.cuda.max_memory_allocated(device))
                    if device.type == "cuda"
                    else None
                ),
            }
        )
        fold_result = {"fold": fold, **result}
        fold_results.append(fold_result)
        write_json_atomic(fold_dir / "result.json", fold_result)

    summary_name = "ablation_validation_results_folds_" + "_".join(
        map(str, fold_indices)
    ) + ".json"
    write_json_atomic(Path(cfg.run_dir) / summary_name, fold_results)
    scalar_rows = [
        {"fold": item["fold"], **_scalar_summary(item)} for item in fold_results
    ]
    scalar_keys = sorted(
        {key for row in scalar_rows for key in row if key != "fold"}
    )
    if scalar_keys:
        csv_path = Path(cfg.run_dir) / summary_name.replace(".json", ".csv")
        with csv_path.open("w", newline="") as stream:
            writer = csv.DictWriter(
                stream, fieldnames=["fold", *scalar_keys]
            )
            writer.writeheader()
            writer.writerows(scalar_rows)
    log.info("ORIGIN ablation results written to %s", Path(cfg.run_dir) / summary_name)


if __name__ == "__main__":
    main()
