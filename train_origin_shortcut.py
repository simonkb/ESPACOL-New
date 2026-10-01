#!/usr/bin/env python3
"""Train preregistered APTOS controlled-shortcut arms with ORIGIN.

This entry point intentionally reuses the unchanged ORIGIN model, loss, and
trainer.  Only the dataset is wrapped, so comparisons with clean checkpoints
cannot acquire a separate classifier bypass or optimization policy.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

from benchmarks.ordinal_shortcut import (
    ORDINAL_SHORTCUT_VERSION,
    SHORTCUT_FAMILIES,
    OrdinalShortcutDataset,
    OrdinalShortcutProtocol,
    make_shortcut_loaders,
)
from configs.origin_config import OriginConfig
from Datasets.origin_data import (
    OriginFundusTransform,
    OriginImageDataset,
    class_histogram,
    load_aptos_items,
    split_origin_items,
    split_paths_are_disjoint,
    validate_ordinal_labels,
)
from models.origin import build_origin_model
from train_origin import (
    build_parser as build_origin_parser,
    guard_and_write_split_manifest,
    parse_folds,
    set_seed,
    setup_logging,
    split_signature,
    write_json_atomic,
)
from training.origin_trainer import OriginTrainer


SHORTCUT_ARMS = ("shortcut", "cue_only", "clean")


def build_parser():
    parser = build_origin_parser()
    parser.description = "ORIGIN controlled ordinal-shortcut benchmark training"
    parser.set_defaults(dataset="aptos")
    benchmark = parser.add_argument_group("controlled shortcut benchmark")
    benchmark.add_argument("--shortcut_family", choices=SHORTCUT_FAMILIES, default="localized")
    benchmark.add_argument("--shortcut_arm", choices=SHORTCUT_ARMS, default="shortcut")
    benchmark.add_argument("--shortcut_seed", type=int, default=271828)
    benchmark.add_argument(
        "--split_seed",
        type=int,
        default=42,
        help="fixed split seed, deliberately separate from the training seed",
    )
    benchmark.add_argument("--shortcut_strength", type=float, default=0.30)
    benchmark.add_argument("--marker_radius_fraction", type=float, default=0.040)
    return parser


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _write_frozen_protocol(path: Path, payload: dict[str, Any], *, resume: bool) -> None:
    signed = dict(payload)
    signed["content_checksum_sha256"] = _canonical_sha256(payload)
    if path.exists():
        with path.open(encoding="utf-8") as stream:
            existing = json.load(stream)
        if existing != signed:
            raise ValueError(f"existing shortcut protocol differs from this invocation: {path}")
        if not resume:
            raise FileExistsError(f"fresh run refuses existing shortcut protocol: {path}")
        return
    if resume:
        raise FileNotFoundError(f"resume requires shortcut protocol: {path}")
    write_json_atomic(path, signed)


def _base_dataset(items, image_size: int, *, augment: bool) -> OriginImageDataset:
    return OriginImageDataset(
        items,
        OriginFundusTransform(size=image_size, augment=augment),
    )


def _wrapped_dataset(
    items,
    *,
    image_size: int,
    augment: bool,
    family: str,
    arm: str,
    protocol: OrdinalShortcutProtocol,
    held_out_domain: bool,
) -> OrdinalShortcutDataset:
    condition = {
        "shortcut": "aligned",
        "cue_only": "cue_only",
        "clean": "clean",
    }[arm]
    return OrdinalShortcutDataset(
        _base_dataset(items, image_size, augment=augment),
        family=family,
        condition=condition,
        position_domain="unseen" if held_out_domain else "seen",
        appearance_domain="unseen" if held_out_domain else "seen",
        protocol=protocol,
    )


def _model_from_cfg(cfg: OriginConfig):
    return build_origin_model(
        num_classes=cfg.n_classes,
        encoder_name=cfg.encoder,
        pretrained=cfg.pretrained and not cfg.resume,
        evidence_scales=cfg.evidence_scales,
        projection_dim=cfg.projection_dim,
        reference_count=cfg.reference_count,
        atom_rate_init=cfg.atom_rate_init,
        prior_rate_init=cfg.prior_rate_init,
        boundary_scale_init=cfg.boundary_scale_init,
        total_rate_cap=cfg.total_rate_cap,
        prior_rate_cap=cfg.prior_rate_cap,
        boundary_scale_cap=cfg.boundary_scale_cap,
        rate_roundoff_margin=cfg.rate_roundoff_margin,
        atom_mode=cfg.atom_mode,
        hybrid_cumulative_init=cfg.hybrid_cumulative_init,
        evidence_dropout=cfg.evidence_dropout,
        mask_valid_fraction=cfg.mask_valid_fraction,
        grad_checkpoint=cfg.grad_checkpoint,
    )


def main() -> None:
    args = build_parser().parse_args()
    if args.dataset != "aptos":
        raise ValueError("the preregistered three-seed pilot is APTOS-only")
    if args.fundus_preprocessing:
        raise ValueError("APTOS already uses the canonical fundus preprocessing")
    data_root = args.data_root or "Datasets/aptos2019-blindness-detection"
    n_folds = args.n_folds or 5
    epochs = args.epochs or 35
    run_dir = args.run_dir or (
        f"runs/origin_shortcut_aptos/{args.shortcut_arm}/"
        f"{args.shortcut_family}/seed{args.seed}"
    )
    items = load_aptos_items(data_root, args.labels_csv)
    validate_ordinal_labels(items, 5)

    cfg = OriginConfig(
        dataset="aptos",
        n_classes=5,
        n_folds=n_folds,
        val_fraction=args.val_fraction,
        run_dir=run_dir,
        preprocessing_version=f"{ORDINAL_SHORTCUT_VERSION}+canonical-square-fixed-ellipse-v1",
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
        stratified_batches=False,
        labels_csv=args.labels_csv,
    )
    if args.stratified_batches:
        raise ValueError("the shortcut pilot fixes ordinary shuffled training batches")
    if cfg.class_weighting != "none" and not cfg.allow_weighted_likelihood:
        raise ValueError("weighted likelihood requires explicit acknowledgement")

    shortcut_protocol = OrdinalShortcutProtocol(
        seed=args.shortcut_seed,
        strength=args.shortcut_strength,
        marker_radius_fraction=args.marker_radius_fraction,
    )
    training_cfg = asdict(cfg)
    training_cfg.update(
        {
            "shortcut_arm": args.shortcut_arm,
            "shortcut_family": args.shortcut_family,
            "shortcut_protocol": shortcut_protocol.as_dict(),
            "shortcut_protocol_signature": shortcut_protocol.signature,
            "split_seed": int(args.split_seed),
        }
    )

    log = setup_logging(run_dir)
    set_seed(cfg.seed)
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    folds = parse_folds(args.folds, cfg.n_folds)
    log.info(
        "shortcut arm=%s family=%s training_seed=%d split_seed=%d protocol=%s",
        args.shortcut_arm,
        args.shortcut_family,
        cfg.seed,
        args.split_seed,
        shortcut_protocol.signature,
    )

    fold_results: list[dict[str, Any]] = []
    for fold in folds:
        set_seed(cfg.seed + fold)
        train_items, validation_items, test_items = split_origin_items(
            "aptos",
            items,
            fold,
            n_folds=cfg.n_folds,
            val_fraction=cfg.val_fraction,
            seed=args.split_seed,
        )
        if not split_paths_are_disjoint(train_items, validation_items, test_items):
            raise RuntimeError(f"fold {fold} contains overlapping image paths")
        signature = split_signature(
            ("train", train_items),
            ("validation", validation_items),
            ("locked_test", test_items),
        )
        fold_dir = Path(run_dir) / f"fold{fold}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        split_manifest = {
            "schema": "origin-shortcut-split-v1",
            "dataset": "aptos",
            "fold": fold,
            "split_seed": int(args.split_seed),
            "training_seed": int(cfg.seed),
            "signature": signature,
            "evaluation_scope": (
                "outer_test_after_selection" if args.include_test else "inner_validation_only"
            ),
            "counts": {
                "train": len(train_items),
                "validation": len(validation_items),
                "locked_test": len(test_items),
            },
            "histograms": {
                "train": class_histogram(train_items, 5),
                "validation": class_histogram(validation_items, 5),
                "locked_test": class_histogram(test_items, 5),
            },
            "shortcut_arm": args.shortcut_arm,
            "shortcut_family": args.shortcut_family,
            "shortcut_protocol_signature": shortcut_protocol.signature,
            "train_domain": {"position": "seen", "appearance": "seen"},
            "outer_domain": {"position": "unseen", "appearance": "unseen"},
        }
        # The shared guard expects the ordinary schema only through identity
        # fields; the shortcut-specific fields remain additional provenance.
        guard_and_write_split_manifest(fold_dir, split_manifest, resume=cfg.resume)
        protocol_payload = {
            "schema": "origin-ordinal-shortcut-protocol-v1",
            "dataset": "aptos",
            "fold": fold,
            "shortcut_arm": args.shortcut_arm,
            "shortcut_family": args.shortcut_family,
            "training_seed": int(cfg.seed),
            "split_seed": int(args.split_seed),
            "split_signature": signature,
            "shortcut": shortcut_protocol.as_dict(),
            "train_condition": {
                "shortcut": "aligned",
                "cue_only": "cue_only",
                "clean": "clean",
            }[args.shortcut_arm],
            "train_domain": {"position": "seen", "appearance": "seen"},
            "test_domain": {"position": "unseen", "appearance": "unseen"},
            "factorial_states": 16,
        }
        _write_frozen_protocol(
            fold_dir / "shortcut_protocol.json", protocol_payload, resume=cfg.resume
        )

        train_dataset = _wrapped_dataset(
            train_items,
            image_size=cfg.img_size,
            augment=True,
            family=args.shortcut_family,
            arm=args.shortcut_arm,
            protocol=shortcut_protocol,
            held_out_domain=False,
        )
        validation_dataset = _wrapped_dataset(
            validation_items,
            image_size=cfg.img_size,
            augment=False,
            family=args.shortcut_family,
            arm=args.shortcut_arm,
            protocol=shortcut_protocol,
            held_out_domain=False,
        )
        test_dataset = _wrapped_dataset(
            test_items,
            image_size=cfg.img_size,
            augment=False,
            family=args.shortcut_family,
            arm=args.shortcut_arm,
            protocol=shortcut_protocol,
            held_out_domain=True,
        )
        loaders = make_shortcut_loaders(
            train_dataset,
            validation_dataset,
            test_dataset,
            batch_size=cfg.batch_size,
            num_workers=0 if device.type == "mps" else cfg.num_workers,
            pin_memory=device.type == "cuda",
            seed=cfg.seed + fold,
        )
        model = _model_from_cfg(cfg)
        trainer = OriginTrainer(
            model,
            *loaders,
            training_cfg,
            fold_dir,
            fold=fold,
            split_signature=signature,
            device=device,
        )
        result = trainer.fit(evaluate_test=args.include_test)
        fold_result = {
            "fold": fold,
            "shortcut_arm": args.shortcut_arm,
            "shortcut_family": args.shortcut_family,
            "training_seed": cfg.seed,
            "split_seed": args.split_seed,
            "shortcut_protocol_signature": shortcut_protocol.signature,
            **result,
        }
        fold_results.append(fold_result)
        write_json_atomic(fold_dir / "result.json", fold_result)

    summary = Path(run_dir) / (
        "final_results.json"
        if sorted(folds) == list(range(cfg.n_folds))
        else "final_results_folds_" + "_".join(map(str, folds)) + ".json"
    )
    write_json_atomic(summary, fold_results)
    log.info("controlled-shortcut results written to %s", summary)


if __name__ == "__main__":
    main()
