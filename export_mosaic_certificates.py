#!/usr/bin/env python3
"""Export and independently replay MOSAIC prediction certificates."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from configs.config import MOSAICConfig
from Datasets.mosaic_data import (
    MOSAIC_PREPROCESSING_VERSION,
    MosaicFundusTransform,
    MosaicImageDataset,
    aptos_fold,
    eyepacs_fold,
    load_aptos_items,
    load_eyepacs_items,
)
from inference.mosaic_certificate import (
    build_mosaic_certificate,
    save_mosaic_certificate,
    verify_mosaic_certificate,
)
from models.mosaic_model import build_mosaic_model
from training.mosaic_trainer import mosaic_implementation_signature


def _file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _split_signature(*named_splits) -> str:
    digest = hashlib.sha256()
    for split_name, items in named_splits:
        digest.update(f"[{split_name}]\n".encode())
        for path, label in sorted((Path(path).name, int(label)) for path, label in items):
            digest.update(f"{path}\t{label}\n".encode())
    return digest.hexdigest()


def _require_matching_implementation_signature(
    checkpoint: dict,
    *,
    active_signature: str | None = None,
) -> str:
    """Reject checkpoints produced by a different executable MOSAIC stack."""

    stored_signature = checkpoint.get("implementation_signature")
    if not isinstance(stored_signature, str) or not stored_signature:
        raise ValueError(
            "checkpoint has no MOSAIC implementation signature; certificate "
            "provenance would be incomplete"
        )
    if active_signature is None:
        active_signature = mosaic_implementation_signature()
    if stored_signature != active_signature:
        raise ValueError(
            "checkpoint MOSAIC implementation signature does not match the "
            "active source tree; refusing to export a certificate computed "
            "under different code"
        )
    return active_signature


def _verification_fields(report: dict | None) -> tuple[bool | None, str]:
    """Represent skipped replay distinctly from successful verification."""

    if report is None:
        return None, "not_run"
    replay_ok = bool(report["ok"])
    return replay_ok, "passed" if replay_ok else "failed"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export replayable MOSAIC certificates",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset", choices=("aptos", "dr"), required=True)
    parser.add_argument("--data_root", default=None)
    parser.add_argument("--labels_csv", default=None)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--split", choices=("validation", "test"), default="test")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="0 exports every image; intended only for an arbitrary replay smoke test",
    )
    parser.add_argument(
        "--per_grade_limit",
        type=int,
        default=0,
        help="Export up to this many deterministic samples per grade (0 disables)",
    )
    parser.add_argument("--no_verify", action="store_true")
    args = parser.parse_args()
    if args.limit < 0 or args.per_grade_limit < 0:
        raise ValueError("certificate limits must be non-negative")
    if args.limit > 0 and args.per_grade_limit > 0:
        raise ValueError("use either --limit or --per_grade_limit, not both")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    stored = checkpoint.get("config", {})
    if "preprocessing_version" not in stored:
        raise ValueError(
            "checkpoint predates MOSAIC's canonical preprocessing identity; "
            "it cannot produce a faithful certificate"
        )
    active_implementation_signature = _require_matching_implementation_signature(
        checkpoint
    )
    cfg = MOSAICConfig(
        **{
            key: value
            for key, value in stored.items()
            if key in MOSAICConfig.__dataclass_fields__
        }
    )
    # Checkpoints created before the explicit proof-decoder audit used rounded
    # expected grade.  Preserve that provenance instead of inheriting the new
    # configuration default when the stored field is absent.
    cfg.decision_rule = stored.get("decision_rule", "rounded_expected")
    criterion_state = checkpoint.get("criterion_state")
    if (
        not isinstance(criterion_state, dict)
        or "transition_weights" not in criterion_state
    ):
        raise ValueError(
            "checkpoint has no training-fold transition weights; a v3 "
            "decision certificate cannot replay its decoder"
        )
    transition_weights = torch.as_tensor(
        criterion_state["transition_weights"], dtype=torch.float32
    ).cpu()
    if tuple(transition_weights.shape) != (cfg.n_classes - 1, 2):
        raise ValueError(
            "checkpoint transition weights do not match the configured "
            "ordinal boundaries"
        )
    if not bool(torch.isfinite(transition_weights).all()) or bool(
        (transition_weights <= 0.0).any()
    ):
        raise ValueError(
            "checkpoint transition weights must be finite and strictly positive"
        )
    if cfg.preprocessing_version != MOSAIC_PREPROCESSING_VERSION:
        raise ValueError(
            "checkpoint preprocessing version does not match the active "
            f"pipeline ({cfg.preprocessing_version!r} != "
            f"{MOSAIC_PREPROCESSING_VERSION!r})"
        )
    if cfg.dataset.lower() != args.dataset:
        raise ValueError(
            f"checkpoint dataset is {cfg.dataset!r}, requested {args.dataset!r}"
        )
    checkpoint_fold = checkpoint.get("fold")
    if checkpoint_fold is None or int(checkpoint_fold) != args.fold:
        raise ValueError(
            f"checkpoint fold is {checkpoint_fold!r}, requested fold {args.fold}"
        )
    root = args.data_root or (
        "Datasets/aptos2019-blindness-detection"
        if args.dataset == "aptos"
        else "Datasets/DR"
    )
    if args.dataset == "aptos":
        all_items = load_aptos_items(root, args.labels_csv)
        train, validation, test = aptos_fold(
            all_items,
            args.fold,
            n_folds=5,
            val_fraction=cfg.val_fraction,
            seed=cfg.seed,
        )
    else:
        all_items = load_eyepacs_items(root, args.labels_csv)
        train, validation, test = eyepacs_fold(
            all_items,
            args.fold,
            n_folds=10,
            val_fraction=cfg.val_fraction,
            seed=cfg.seed,
        )
    current_signature = _split_signature(
        ("train", train), ("validation", validation), ("test", test)
    )
    if checkpoint.get("split_signature") != current_signature:
        raise ValueError(
            "checkpoint split signature does not match the current dataset partition"
        )
    items = validation if args.split == "validation" else test
    if args.per_grade_limit > 0:
        counts: dict[int, int] = {}
        selected = []
        for item in items:
            grade = int(item[1])
            if counts.get(grade, 0) < args.per_grade_limit:
                selected.append(item)
                counts[grade] = counts.get(grade, 0) + 1
        items = selected
    elif args.limit > 0:
        items = items[: args.limit]

    dataset = MosaicImageDataset(
        items,
        MosaicFundusTransform(cfg.img_size, augment=False),
    )
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=cfg.num_workers,
    )
    model = build_mosaic_model(
        num_classes=cfg.n_classes,
        image_size=cfg.img_size,
        local_stage=cfg.local_stage,
        local_dim=cfg.evidence_dim,
        pretrained=False,
        grad_checkpoint=False,
        initial_abnormal_count=cfg.normal_expected_count,
        max_count=cfg.max_count,
        sufficiency_tolerance=cfg.proof_epsilon,
        complement_suppression=cfg.necessity_fraction,
        count_implementation=cfg.count_implementation,
        count_block_size=cfg.count_block_size,
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.configure_proof_decoder(cfg.decision_rule, transition_weights)
    model.eval()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_digest = _file_sha256(args.checkpoint)
    manifest_rows = []
    with torch.no_grad():
        for images, masks, _labels, indices in loader:
            local_index = int(indices[0])
            image_path, true_grade = items[local_index]
            result = model(
                images.to(device),
                masks.to(device),
                project=True,
                # The builder computes exact fixed-proof effects on CPU.  This
                # avoids thousands of tiny GPU launches for a large proof.
                return_pivotality=False,
            )
            sample_id = Path(image_path).stem
            certificate = build_mosaic_certificate(
                result.evidence,
                lattice_metadata=result.lattice,
                valid_mask=result.valid_mask,
                sample_index=0,
                sample_id=sample_id,
                sufficiency_tolerance=cfg.proof_epsilon,
                complement_suppression=cfg.necessity_fraction,
                decision_rule=cfg.decision_rule,
                transition_weights=transition_weights,
                provenance={
                    "checkpoint_sha256": checkpoint_digest,
                    "source_image_sha256": _file_sha256(image_path),
                    "source_image_name": Path(image_path).name,
                    "true_grade": int(true_grade),
                    "dataset": args.dataset,
                    "fold": int(args.fold),
                    "split": args.split,
                    "split_signature": current_signature,
                    "implementation_signature": active_implementation_signature,
                    "preprocessing": {
                        "version": MOSAIC_PREPROCESSING_VERSION,
                        "image_size": int(cfg.img_size),
                        "crop": "dominant_nonblack_field",
                        "resize": "direct_fundus_field_to_square",
                        "normalization": "imagenet",
                        "augmentation": False,
                        "mask": "fixed_centered_ellipse",
                    },
                },
            )
            direct_grade = int(result.predicted_grade[0].detach().cpu())
            certified_grade = int(certificate["prediction"]["predicted_grade"])
            if direct_grade != certified_grade:
                raise RuntimeError(
                    "configured model prediction and independently rebuilt "
                    f"certificate decision disagree ({direct_grade} != "
                    f"{certified_grade})"
                )
            report = None
            if not args.no_verify:
                report = verify_mosaic_certificate(certificate, raise_on_error=True)
            replay_ok, replay_status = _verification_fields(report)
            certificate_path = output_dir / f"{sample_id}.json"
            save_mosaic_certificate(certificate, certificate_path)
            manifest_rows.append(
                {
                    "sample_id": sample_id,
                    "image_path": image_path,
                    "true_grade": true_grade,
                    "predicted_grade": certificate["prediction"]["predicted_grade"],
                    "expected_grade": certificate["prediction"]["expected_grade"],
                    "certificate": str(certificate_path),
                    "replay_ok": replay_ok,
                    "replay_status": replay_status,
                }
            )

    with (output_dir / "manifest.csv").open("w", newline="") as stream:
        fieldnames = list(manifest_rows[0]) if manifest_rows else []
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        if manifest_rows:
            writer.writeheader()
            writer.writerows(manifest_rows)
    summary = {
        "checkpoint": args.checkpoint,
        "dataset": args.dataset,
        "fold": args.fold,
        "split": args.split,
        "decision_rule": cfg.decision_rule,
        "implementation_signature": active_implementation_signature,
        "limit": args.limit,
        "per_grade_limit": args.per_grade_limit,
        "certificates": len(manifest_rows),
        "all_replay_ok": (
            None
            if args.no_verify
            else all(bool(row["replay_ok"]) for row in manifest_rows)
        ),
        "replay_status": "not_run" if args.no_verify else "passed",
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
