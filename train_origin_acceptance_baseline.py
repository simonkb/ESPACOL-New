#!/usr/bin/env python3
"""Train one pre-registered acceptance baseline without opening outer test."""

from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
import time
from typing import Any

import torch

from configs.origin_acceptance_baseline_config import (
    OriginAcceptanceBaselineConfig,
)
from Datasets.origin_data import (
    ORIGIN_PREPROCESSING_VERSION,
    class_histogram,
    load_origin_items,
    make_origin_loaders,
    split_origin_items,
    split_paths_are_disjoint,
    validate_ordinal_labels,
)
from models.origin_acceptance_baselines import build_origin_acceptance_baseline
from scripts.origin_acceptance_baseline_common import (
    AcceptanceTrainingTask,
    BASELINE_SPECS,
    PROTOCOL_ID,
    PROTOCOL_SHA256,
    SPLIT_SEED,
    SPARSE_L1_DELAY_EPOCHS,
    artifact_manifest,
    canonical_sha256,
    task_at,
    worker_run_dir,
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
from training.origin_acceptance_baseline_trainer import (
    OriginAcceptanceBaselineTrainer,
)


_WORKER_ARTIFACTS = (
    "split_manifest.json",
    "warm_start_floor.json",
    "best_learned.pth",
    "last_learned.pth",
    "learned_history.csv",
    "validation_certificates.json",
    "result.json",
)


def build_parser():
    parser = build_origin_parser()
    parser.description = (
        "Train one locked ORIGIN acceptance baseline using inner validation only"
    )
    parser.add_argument("--variant", choices=tuple(BASELINE_SPECS), default=None)
    parser.add_argument("--training_seed", type=int, default=None)
    parser.add_argument(
        "--protocol_scope", choices=("canary", "full"), required=True
    )
    parser.add_argument(
        "--task_index",
        type=int,
        default=None,
        help="stable zero-based index in the checked-in scope task map",
    )
    return parser


def _resolve_task(args: Any) -> AcceptanceTrainingTask:
    if args.task_index is not None:
        return task_at(args.protocol_scope, args.task_index)
    if args.variant is None:
        raise ValueError("--variant is required when --task_index is omitted")
    folds = parse_folds(args.folds, int(args.n_folds or DATASET_DEFAULTS[args.dataset]["n_folds"]))
    if len(folds) != 1:
        raise ValueError("one worker invocation must own exactly one outer fold")
    return AcceptanceTrainingTask(
        dataset=args.dataset,
        fold=folds[0],
        baseline_variant=args.variant,
        training_seed=(42 if args.training_seed is None else args.training_seed),
    )


def _guard_worker_directory(fold_dir: Path, *, resume: bool) -> None:
    protected = (*_WORKER_ARTIFACTS, "artifact_manifest.json", "TRAINING_COMPLETE.json")
    existing = [name for name in protected if (fold_dir / name).exists()]
    if not resume and existing:
        raise FileExistsError(
            "fresh acceptance-baseline run would overwrite: " + ", ".join(existing)
        )


def _require_canary_gate(worker_dir: Path) -> None:
    """Verify, rather than merely notice, the prerequisite canary gate."""

    # Registered layout: <experiment-root>/full/training/<task-key>.
    try:
        experiment_root = worker_dir.resolve().parents[2]
    except IndexError as exc:
        raise ValueError("full worker run directory is too shallow") from exc
    gate_path = experiment_root / "canary" / "CANARY_PASSED.json"
    if not gate_path.is_file():
        raise FileNotFoundError(f"full training requires canary gate: {gate_path}")
    with gate_path.open(encoding="utf-8") as stream:
        gate = json.load(stream)
    checksum = gate.get("content_checksum_sha256")
    unsigned = dict(gate)
    unsigned.pop("content_checksum_sha256", None)
    if checksum != canonical_sha256(unsigned):
        raise ValueError("canary gate checksum mismatch")
    if (
        gate.get("status") != "passed"
        or gate.get("scope") != "canary"
        or gate.get("protocol_sha256") != PROTOCOL_SHA256
    ):
        raise ValueError("canary gate does not authorize this full protocol")


def main() -> None:
    args = build_parser().parse_args()
    if args.include_test:
        raise ValueError(
            "training never evaluates the locked outer fold; use the suite-level release"
        )
    if not args.skip_test:
        raise ValueError("training requires explicit --skip_test authorization")
    if args.seed != SPLIT_SEED:
        raise ValueError(
            f"the registered outer/inner split seed is fixed at {SPLIT_SEED}"
        )
    if args.evidence_budget_weight != 0.0:
        raise ValueError("acceptance baselines require --evidence_budget_weight 0")

    task = _resolve_task(args)
    defaults = DATASET_DEFAULTS[task.dataset]
    data_root = args.data_root or str(defaults["root"])
    n_folds = args.n_folds or int(defaults["n_folds"])
    if n_folds != int(defaults["n_folds"]):
        raise ValueError("protocol workers cannot change the registered fold count")
    epochs = args.epochs or int(defaults["epochs"])
    run_root = args.run_dir or "runs/origin_acceptance_baselines"
    worker_dir = (
        Path(run_root)
        if args.run_dir is not None
        else worker_run_dir(run_root, args.protocol_scope, task)
    )
    if args.protocol_scope == "full":
        _require_canary_gate(worker_dir)
    fundus = bool(defaults["fundus"])
    spec = BASELINE_SPECS[task.baseline_variant]

    items = load_origin_items(
        task.dataset,
        data_root,
        labels_csv=args.labels_csv,
        image_column=args.image_column,
        label_column=args.label_column,
        image_dir=args.image_dir,
    )
    inferred_classes = validate_ordinal_labels(items)
    requested_classes = args.n_classes or defaults["n_classes"] or inferred_classes
    validate_ordinal_labels(items, int(requested_classes))
    cfg = OriginAcceptanceBaselineConfig(
        dataset=task.dataset,
        n_classes=int(requested_classes),
        n_folds=n_folds,
        val_fraction=args.val_fraction,
        run_dir=str(worker_dir),
        preprocessing_version=ORIGIN_PREPROCESSING_VERSION,
        seed=SPLIT_SEED,
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
        total_rate_cap=args.total_rate_cap,
        prior_rate_cap=args.prior_rate_cap,
        boundary_scale_cap=args.boundary_scale_cap,
        rate_roundoff_margin=args.rate_roundoff_margin,
        atom_mode=args.atom_mode,
        hybrid_cumulative_init=args.hybrid_cumulative_init,
        evidence_dropout=args.evidence_dropout,
        force_decoder_fp64=True,
        decision_rule=args.decision_rule,
        rps_weight=args.rps_weight,
        evidence_budget_weight=0.0,
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
        baseline_variant=task.baseline_variant,
        ablation_variant=task.baseline_variant,
        baseline_description=str(spec["description"]),
        training_seed=task.training_seed,
        sparse_l1_weight=float(spec["sparse_l1_weight"]),
        sparse_l1_delay_epochs=SPARSE_L1_DELAY_EPOCHS,
        protocol_id=PROTOCOL_ID,
    )

    log = setup_logging(str(worker_dir))
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    log.info("acceptance baseline configuration: %s", asdict(cfg))
    log.info(
        "protocol=%s protocol_sha256=%s scope=%s task=%s images=%d classes=%s",
        PROTOCOL_ID,
        PROTOCOL_SHA256,
        args.protocol_scope,
        task.key,
        len(items),
        class_histogram(items, cfg.n_classes),
    )

    train_items, val_items, locked_test_items = split_origin_items(
        cfg.dataset,
        items,
        task.fold,
        n_folds=cfg.n_folds,
        val_fraction=cfg.val_fraction,
        seed=cfg.seed,
    )
    if not split_paths_are_disjoint(train_items, val_items, locked_test_items):
        raise RuntimeError(f"fold {task.fold} contains overlapping paths")
    signature = split_signature(
        ("train", train_items),
        ("validation", val_items),
        ("locked_test", locked_test_items),
    )
    fold_dir = worker_dir / f"fold{task.fold}"
    fold_dir.mkdir(parents=True, exist_ok=True)
    _guard_worker_directory(fold_dir, resume=cfg.resume)
    split_manifest = {
        "schema": "origin-acceptance-split-v1",
        "protocol_id": PROTOCOL_ID,
        "protocol_sha256": PROTOCOL_SHA256,
        "scope": args.protocol_scope,
        "task": asdict(task),
        "dataset": cfg.dataset,
        "fold": task.fold,
        "baseline_variant": task.baseline_variant,
        "split_seed": cfg.seed,
        "training_seed": cfg.training_seed,
        "signature": signature,
        "evaluation_scope": "inner_validation_only_outer_locked",
        "counts": {
            "train": len(train_items),
            "validation": len(val_items),
            "locked_test": len(locked_test_items),
        },
        "histograms": {
            "train": class_histogram(train_items, cfg.n_classes),
            "validation": class_histogram(val_items, cfg.n_classes),
            "locked_test": class_histogram(locked_test_items, cfg.n_classes),
        },
    }
    guard_and_write_split_manifest(fold_dir, split_manifest, resume=cfg.resume)

    loader_seed = cfg.training_seed + task.fold
    loaders = make_origin_loaders(
        train_items,
        val_items,
        (),  # Locked outer items are never materialized by training workers.
        image_size=cfg.img_size,
        batch_size=cfg.batch_size,
        num_workers=0 if device.type == "mps" else cfg.num_workers,
        pin_memory=device.type == "cuda",
        seed=loader_seed,
        fundus=fundus,
        stratified=cfg.stratified_batches,
    )
    set_seed(loader_seed)
    model = build_origin_acceptance_baseline(cfg)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    trainable_parameter_count = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    trainer = OriginAcceptanceBaselineTrainer(
        model,
        *loaders,
        cfg,
        fold_dir,
        fold=task.fold,
        split_signature=signature,
        device=device,
    )
    training_stream_seed = cfg.training_seed + task.fold + 100_000
    set_seed(training_stream_seed)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    started = time.monotonic()
    result = trainer.fit(evaluate_test=False)
    result.update(
        {
            "schema": "origin-acceptance-training-result-v1",
            "protocol_id": PROTOCOL_ID,
            "protocol_sha256": PROTOCOL_SHA256,
            "scope": args.protocol_scope,
            "task": asdict(task),
            "split_signature": signature,
            "parameter_count": int(parameter_count),
            "trainable_parameter_count": int(trainable_parameter_count),
            "training_stream_seed": int(training_stream_seed),
            "training_wall_time_seconds": float(time.monotonic() - started),
            "training_peak_cuda_memory_bytes": (
                int(torch.cuda.max_memory_allocated(device))
                if device.type == "cuda"
                else None
            ),
            "outer_test_items_materialized_by_worker": False,
            "test_evaluated": False,
        }
    )
    write_json_atomic(fold_dir / "result.json", result)
    manifest = artifact_manifest(
        fold_dir,
        required_names=_WORKER_ARTIFACTS,
        task=task,
        scope=args.protocol_scope,
    )
    write_json_atomic(fold_dir / "artifact_manifest.json", manifest)
    completion = {
        "schema": "origin-acceptance-training-complete-v1",
        "protocol_id": PROTOCOL_ID,
        "protocol_sha256": PROTOCOL_SHA256,
        "task": asdict(task),
        "artifact_manifest_checksum": manifest["content_checksum_sha256"],
        "test_evaluated": False,
    }
    write_json_atomic(fold_dir / "TRAINING_COMPLETE.json", completion)
    log.info("completed acceptance worker %s at %s", task.key, fold_dir)


if __name__ == "__main__":
    main()
