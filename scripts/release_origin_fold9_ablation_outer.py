#!/usr/bin/env python3
"""One-shot locked-outer release for the paired EyePACS fold-9 ablation.

No model is evaluated until *every* selected variant has a valid, checksummed
inner-validation training marker.  The release then reconstructs every model
from its selected checkpoint, evaluates the identical locked fold-9 partition,
and atomically publishes per-image probabilities plus checksummed manifests.
"""

from __future__ import annotations

import argparse
import csv
import gc
import inspect
import os
import sys
from dataclasses import fields
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
from torch.utils.data import DataLoader

from configs.origin_ablation_config import OriginAblationConfig
from Datasets.origin_data import (
    OriginFundusTransform,
    OriginImageDataset,
    class_histogram,
    load_origin_items,
    split_origin_items,
    split_paths_are_disjoint,
)
from models.origin_ablation import build_origin_ablation_model
from scripts.aggregate_origin_fold9_ablations import CSV_FIELDS, metrics_from_rows
from scripts.origin_fold9_ablation_common import (
    DATASET,
    EXPECTED_SPLIT_COUNTS,
    EXPECTED_SPLIT_HISTOGRAMS,
    EXPECTED_SPLIT_SIGNATURE,
    FOLD,
    N_IMAGES,
    TRAIN_MARKER_SCHEMA,
    canonical_sha256,
    file_sha256,
    read_json,
    validate_split_manifest,
    validate_variant_config,
    verify_checksummed_payload,
    verify_protocol,
    write_json_atomic,
)
from train_origin import split_signature
from training.origin_ablation_trainer import (
    _canonical_sha256,
    _truthful_architecture_record,
    origin_ablation_implementation_signature,
)


def _field(output: object, name: str) -> torch.Tensor:
    value = output.get(name) if isinstance(output, Mapping) else getattr(output, name)
    if not torch.is_tensor(value):
        raise TypeError(f"ablation output {name!r} is not a tensor")
    return value


def _relative_image_path(path: str, data_root: Path) -> str:
    resolved = Path(path).expanduser().resolve(strict=False)
    try:
        return resolved.relative_to(data_root.resolve()).as_posix()
    except ValueError:
        return resolved.as_posix()


def _expected_training_paths(root: Path, variant: str) -> dict[str, Path]:
    fold_dir = root / "workers" / variant / f"fold{FOLD}"
    return {
        "checkpoint": fold_dir / "best.pth",
        "result": fold_dir / "result.json",
        "split_manifest": fold_dir / "split_manifest.json",
        "history": fold_dir / "history.csv",
        "validation_certificate": fold_dir / "validation_certificates.json",
    }


def verify_all_training_markers(
    *, root: Path, protocol: Mapping[str, Any], variants: tuple[str, ...]
) -> dict[str, dict[str, Any]]:
    """Complete fail-closed marker pass performed before any outer inference."""

    verified: dict[str, dict[str, Any]] = {}
    split_signature_seen: str | None = None
    for variant in variants:
        paths = _expected_training_paths(root, variant)
        marker_path = paths["checkpoint"].parent / "ABLATION_TRAIN_COMPLETE.json"
        if not marker_path.is_file():
            raise FileNotFoundError(
                f"outer release remains locked; missing training marker for {variant}: "
                f"{marker_path}"
            )
        marker = read_json(marker_path)
        verify_checksummed_payload(marker, schema=TRAIN_MARKER_SCHEMA)
        expected_identity = {
            "dataset": DATASET,
            "fold": FOLD,
            "variant": variant,
            "evaluation_scope": "inner_validation_only_outer_locked",
            "outer_test_evaluated": False,
            "protocol_checksum_sha256": protocol["content_checksum_sha256"],
            "launch_commit": protocol["launch_commit"],
            "split_signature": EXPECTED_SPLIT_SIGNATURE,
        }
        for key, expected in expected_identity.items():
            if marker.get(key) != expected:
                raise AssertionError(
                    f"{variant} training marker {key} mismatch: "
                    f"{marker.get(key)!r} != {expected!r}"
                )
        if marker.get("artifact_paths") != {
            name: str(path.resolve()) for name, path in paths.items()
        }:
            raise AssertionError(f"{variant} training marker artifact paths changed")
        if set(marker.get("artifact_sha256", {})) != set(paths):
            raise AssertionError(f"{variant} training marker artifact set is incomplete")
        for name, path in paths.items():
            if not path.is_file():
                raise FileNotFoundError(f"{variant} training artifact is missing: {path}")
            if file_sha256(path) != marker["artifact_sha256"][name]:
                raise AssertionError(f"{variant} training artifact changed: {path}")

        split = read_json(paths["split_manifest"])
        validate_split_manifest(split)
        if split.get("ablation_variant") != variant:
            raise AssertionError(f"{variant} split manifest variant mismatch")
        result = read_json(paths["result"])
        if result.get("fold") != FOLD or result.get("test_evaluated") is not False:
            raise AssertionError(f"{variant} result does not preserve the outer-test seal")
        if result.get("ablation_variant") != variant:
            raise AssertionError(f"{variant} result variant identity changed")
        if result.get("test") not in (None, {}):
            raise AssertionError(f"{variant} result contains forbidden outer-test metrics")
        if result.get("best_epoch") != marker.get("best_epoch"):
            raise AssertionError(f"{variant} marker/result selected epoch mismatch")
        if result.get("training_efficiency_scope") != marker.get(
            "training_efficiency_scope"
        ):
            raise AssertionError(
                f"{variant} marker/result training-efficiency scope mismatch"
            )
        for key in (
            "parameter_count",
            "trainable_parameter_count",
            "training_wall_time_seconds",
        ):
            value = result.get(key)
            if not isinstance(value, (int, float)) or isinstance(value, bool) or value <= 0:
                raise AssertionError(f"{variant} result has invalid {key}: {value!r}")
        if split_signature_seen is None:
            split_signature_seen = str(split["signature"])
        elif split["signature"] != split_signature_seen:
            raise AssertionError("ablation variants do not share one locked split")
        verified[variant] = {
            "marker_path": marker_path,
            "marker": marker,
            "paths": paths,
            "split": split,
            "result": result,
        }
    if tuple(verified) != variants:
        raise AssertionError("not every protocol variant passed the training-marker gate")
    return verified


def _reconstruct_outer_items(
    *, cfg: OriginAblationConfig, data_root: Path, split: Mapping[str, Any]
) -> tuple[list[tuple[str, int]], list[tuple[str, int]], list[tuple[str, int]]]:
    items = load_origin_items(
        cfg.dataset,
        str(data_root),
        labels_csv=cfg.labels_csv,
        image_column=cfg.image_column,
        label_column=cfg.label_column,
        image_dir=cfg.image_dir,
    )
    if len(items) != N_IMAGES:
        raise AssertionError(f"EyePACS size changed: {len(items)} != {N_IMAGES}")
    train_items, validation_items, outer_items = split_origin_items(
        cfg.dataset,
        items,
        FOLD,
        n_folds=cfg.n_folds,
        val_fraction=cfg.val_fraction,
        seed=cfg.seed,
    )
    if not split_paths_are_disjoint(train_items, validation_items, outer_items):
        raise AssertionError("reconstructed fold-9 split has overlapping image paths")
    counts = {
        "train": len(train_items),
        "validation": len(validation_items),
        "locked_test": len(outer_items),
    }
    histograms = {
        "train": class_histogram(train_items, cfg.n_classes),
        "validation": class_histogram(validation_items, cfg.n_classes),
        "locked_test": class_histogram(outer_items, cfg.n_classes),
    }
    signature = split_signature(
        ("train", train_items),
        ("validation", validation_items),
        ("locked_test", outer_items),
    )
    if counts != EXPECTED_SPLIT_COUNTS or counts != split.get("counts"):
        raise AssertionError("reconstructed split counts differ from the locked protocol")
    if histograms != EXPECTED_SPLIT_HISTOGRAMS or histograms != split.get("histograms"):
        raise AssertionError("reconstructed split histograms differ from the locked protocol")
    if signature != EXPECTED_SPLIT_SIGNATURE or signature != split.get("signature"):
        raise AssertionError("reconstructed split signature differs from the locked protocol")
    patient_sets = [
        {Path(path).stem.rsplit("_", 1)[0] for path, _ in subset}
        for subset in (train_items, validation_items, outer_items)
    ]
    if any(patient_sets[left] & patient_sets[right] for left in range(3) for right in range(left + 1, 3)):
        raise AssertionError("EyePACS patient leaks across reconstructed fold-9 splits")
    return train_items, validation_items, outer_items


def _reconstruct_validated_model(
    *, variant: str, verified: Mapping[str, Any]
) -> tuple[OriginAblationConfig, torch.nn.Module]:
    """Reconstruct one selected checkpoint without touching the outer split.

    This validation is deliberately independent of ``data_root`` and of any
    dataset/loader construction.  The release performs it for *all* variants
    before the first outer image is materialized, so an incompatible checkpoint
    late in protocol order cannot produce a partial outer evaluation.
    """

    state = torch.load(
        verified["paths"]["checkpoint"], map_location="cpu", weights_only=False
    )
    if state.get("schema") != "origin-checkpoint-v3":
        raise ValueError(f"{variant} selected checkpoint has an unexpected schema")
    if state.get("fold") != FOLD or state.get("split_signature") != EXPECTED_SPLIT_SIGNATURE:
        raise AssertionError(f"{variant} checkpoint fold/split identity changed")
    config_values = state.get("config")
    if not isinstance(config_values, Mapping):
        raise TypeError(f"{variant} checkpoint configuration is missing")
    allowed = {field.name for field in fields(OriginAblationConfig)}
    cfg = OriginAblationConfig(
        **{key: value for key, value in config_values.items() if key in allowed}
    )
    if cfg.ablation_variant != variant:
        raise AssertionError(f"{variant} checkpoint reconstructs another variant")
    validate_variant_config(config_values, variant)
    marker = verified["marker"]
    for key in (
        "implementation_signature",
        "architecture_signature",
        "config_signature",
    ):
        marker_key = "checkpoint_config_signature" if key == "config_signature" else key
        if state.get(key) != marker.get(marker_key):
            raise AssertionError(f"{variant} checkpoint/marker {key} mismatch")
    active_implementation = origin_ablation_implementation_signature()
    if state.get("implementation_signature") != active_implementation:
        raise AssertionError(
            f"{variant} checkpoint was produced by another ablation implementation"
        )
    model = build_origin_ablation_model(cfg, pretrained=False)
    architecture = _truthful_architecture_record(model)
    if architecture != state.get("architecture"):
        raise AssertionError(f"{variant} reconstructed architecture differs from checkpoint")
    if _canonical_sha256(architecture) != state.get("architecture_signature"):
        raise AssertionError(f"{variant} reconstructed architecture signature differs")
    model.load_state_dict(state["model_state"], strict=True)
    del state
    return cfg, model


def validate_all_checkpoint_reconstructions(
    *, verified: Mapping[str, Mapping[str, Any]], variants: tuple[str, ...]
) -> None:
    """Strict-load every selected checkpoint before any outer inference."""

    for variant in variants:
        _, model = _reconstruct_validated_model(
            variant=variant, verified=verified[variant]
        )
        del model
        gc.collect()


def _evaluate_variant(
    *,
    variant: str,
    verified: Mapping[str, Any],
    protocol: Mapping[str, Any],
    data_root: Path,
    output_dir: Path,
    final_output_dir: Path,
    num_workers: int,
) -> dict[str, Any]:
    cfg, model = _reconstruct_validated_model(
        variant=variant, verified=verified
    )
    marker = verified["marker"]

    train_items, validation_items, outer_items = _reconstruct_outer_items(
        cfg=cfg, data_root=data_root, split=verified["split"]
    )
    del train_items, validation_items
    gc.collect()

    loader = DataLoader(
        OriginImageDataset(outer_items, OriginFundusTransform(cfg.img_size, augment=False)),
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=False,
    )
    device = torch.device("cuda")
    model.to(device).eval()
    accepts_pixel_mask = "pixel_valid_mask" in inspect.signature(model.forward).parameters
    accepts_fp64 = "force_decoder_fp64" in inspect.signature(model.forward).parameters
    probabilities: list[torch.Tensor] = []
    expected_grades: list[torch.Tensor] = []
    predictions: list[torch.Tensor] = []
    labels_all: list[torch.Tensor] = []
    indices_all: list[torch.Tensor] = []
    with torch.inference_mode():
        for images, pixel_masks, labels, indices in loader:
            kwargs: dict[str, Any] = {}
            if accepts_pixel_mask:
                kwargs["pixel_valid_mask"] = pixel_masks.to(device, non_blocking=True)
            if accepts_fp64:
                kwargs["force_decoder_fp64"] = True
            with torch.autocast(device_type="cuda", enabled=bool(cfg.amp)):
                output = model(images.to(device, non_blocking=True), **kwargs)
            probs = _field(output, "class_probs").detach().cpu().double()
            expected = _field(output, "expected_grade").detach().cpu().double()
            class_map = _field(output, "class_map").detach().cpu().long()
            if probs.shape[1] != cfg.n_classes:
                raise AssertionError(f"{variant} produced an invalid posterior shape")
            probabilities.append(probs)
            expected_grades.append(expected)
            predictions.append(class_map)
            labels_all.append(labels.detach().cpu().long())
            indices_all.append(torch.as_tensor(indices, dtype=torch.long).cpu())

    probs = torch.cat(probabilities)
    expected = torch.cat(expected_grades)
    predicted = torch.cat(predictions).clamp(0, cfg.n_classes - 1)
    labels = torch.cat(labels_all)
    indices = torch.cat(indices_all)
    if indices.tolist() != list(range(len(outer_items))):
        raise AssertionError(f"{variant} outer loader order/coverage changed")
    if labels.tolist() != [int(label) for _, label in outer_items]:
        raise AssertionError(f"{variant} outer loader labels differ from locked split")
    if not torch.equal(predicted, probs.argmax(dim=1)):
        raise AssertionError(f"{variant} class-MAP output differs from argmax posterior")
    replayed_expected = probs @ torch.arange(cfg.n_classes, dtype=torch.float64)
    if not torch.allclose(expected, replayed_expected, rtol=1e-7, atol=2e-7):
        raise AssertionError(f"{variant} expected grade does not replay from posterior")

    rows: list[dict[str, Any]] = []
    for index, (path, label) in enumerate(outer_items):
        row: dict[str, Any] = {
            "variant": variant,
            "fold": FOLD,
            "outer_index": index,
            "image_path_relative": _relative_image_path(path, data_root),
            "patient_id": Path(path).stem.rsplit("_", 1)[0],
            "true_label": int(label),
            "primary_prediction": int(predicted[index]),
            "expected_grade": float(expected[index]),
        }
        row.update(
            {f"p_grade_{grade}": float(probs[index, grade]) for grade in range(cfg.n_classes)}
        )
        rows.append(row)
    metrics = metrics_from_rows(rows)

    output_dir.mkdir(parents=True, exist_ok=False)
    csv_path = output_dir / "outer_predictions.csv"
    with csv_path.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    checkpoint_path = verified["paths"]["checkpoint"]
    manifest: dict[str, Any] = {
        "schema": "origin-fold9-ablation-outer-predictions-v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": DATASET,
        "fold": FOLD,
        "variant": variant,
        "scope": "locked_outer_fold9_after_all_inner_validation_markers",
        "decision_rule": "class_map",
        "n": len(rows),
        "class_histogram": class_histogram(outer_items, cfg.n_classes),
        "split_signature": EXPECTED_SPLIT_SIGNATURE,
        "protocol_checksum_sha256": protocol["content_checksum_sha256"],
        "training_marker_checksum_sha256": marker["content_checksum_sha256"],
        "launch_commit": protocol["launch_commit"],
        "implementation_signature": marker["implementation_signature"],
        "architecture_signature": marker["architecture_signature"],
        "protocol_variant_config_signature": marker[
            "protocol_variant_config_signature"
        ],
        "checkpoint_config_signature": marker["checkpoint_config_signature"],
        "checkpoint_epoch": marker["best_epoch"],
        "checkpoint_sha256": file_sha256(checkpoint_path),
        "result_sha256": file_sha256(verified["paths"]["result"]),
        "architecture_declared": marker["architecture_declared"],
        "training_summary": {
            "best_epoch": marker["best_epoch"],
            "validation": marker["validation"],
            "parameter_count": verified["result"].get("parameter_count"),
            "trainable_parameter_count": verified["result"].get(
                "trainable_parameter_count"
            ),
            "training_wall_time_seconds": verified["result"].get(
                "training_wall_time_seconds"
            ),
            "training_peak_cuda_memory_bytes": verified["result"].get(
                "training_peak_cuda_memory_bytes"
            ),
            "training_efficiency_scope": verified["result"].get(
                "training_efficiency_scope"
            ),
        },
        "release_implementation_sha256": file_sha256(Path(__file__).resolve()),
        "csv_path": str((final_output_dir / "outer_predictions.csv").resolve()),
        "csv_sha256": file_sha256(csv_path),
        "metrics": metrics,
    }
    manifest["content_checksum_sha256"] = canonical_sha256(manifest)
    write_json_atomic(output_dir / "outer_predictions_manifest.json", manifest)
    del model
    torch.cuda.empty_cache()
    return manifest


def release(
    *, root: Path, protocol_path: Path, data_root: Path, num_workers: int
) -> dict[str, Any]:
    protocol = read_json(protocol_path)
    variants = verify_protocol(
        protocol, expected_root=root, expected_data_root=data_root
    )
    # This artifact gate is intentionally complete before outer-data/model
    # evaluation; the following CPU-only pass then strict-loads every model.
    verified = verify_all_training_markers(
        root=root, protocol=protocol, variants=variants
    )
    # Validate every checkpoint, including the last one in protocol order,
    # before constructing an outer dataset or invoking an evaluator.  Marker
    # verification alone cannot detect an incompatible state dict.
    validate_all_checkpoint_reconstructions(
        verified=verified, variants=variants
    )
    if not torch.cuda.is_available():
        raise RuntimeError("locked outer release requires CUDA")

    release_root = root / "release"
    if release_root.exists():
        raise FileExistsError(
            f"refusing to repeat or overwrite a locked outer release: {release_root}"
        )
    stale_staging = list(root.glob(".release.staging.*"))
    if stale_staging:
        raise FileExistsError(
            "a prior release attempt left staging artifacts; preserve and audit them "
            f"before any retry: {stale_staging}"
        )
    staging_root = root / f".release.staging.{os.getpid()}"
    staging_root.mkdir(parents=False, exist_ok=False)
    manifests: dict[str, dict[str, Any]] = {}
    try:
        for variant in variants:
            manifests[variant] = _evaluate_variant(
                variant=variant,
                verified=verified[variant],
                protocol=protocol,
                data_root=data_root,
                output_dir=staging_root / variant,
                final_output_dir=release_root / variant,
                num_workers=num_workers,
            )
        artifacts = {
            variant: {
                "csv_path": str(
                    (release_root / variant / "outer_predictions.csv").resolve()
                ),
                "csv_sha256": file_sha256(
                    staging_root / variant / "outer_predictions.csv"
                ),
                "manifest_path": str(
                    (release_root / variant / "outer_predictions_manifest.json").resolve()
                ),
                "manifest_sha256": file_sha256(
                    staging_root / variant / "outer_predictions_manifest.json"
                ),
            }
            for variant in variants
        }
        completion: dict[str, Any] = {
            "schema": "origin-fold9-ablation-outer-release-complete-v1",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "dataset": DATASET,
            "fold": FOLD,
            "scope": "single_nonadaptive_outer_release_after_all_training_markers",
            "protocol_checksum_sha256": protocol["content_checksum_sha256"],
            "ordered_variants": list(variants),
            "training_marker_checksums": {
                variant: verified[variant]["marker"]["content_checksum_sha256"]
                for variant in variants
            },
            "prediction_manifest_checksums": {
                variant: manifests[variant]["content_checksum_sha256"]
                for variant in variants
            },
            "artifacts": artifacts,
        }
        completion["content_checksum_sha256"] = canonical_sha256(completion)
        write_json_atomic(staging_root / "OUTER_RELEASE_COMPLETE.json", completion)
        os.replace(staging_root, release_root)
    except Exception:
        # Preserve partial staging evidence rather than silently deleting proof
        # that the outer partition may already have been opened.
        raise
    return completion


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--num-workers", type=int, default=8)
    args = parser.parse_args()
    completion = release(
        root=args.root.expanduser().resolve(),
        protocol_path=args.protocol.expanduser().resolve(),
        data_root=args.data_root.expanduser().resolve(),
        num_workers=args.num_workers,
    )
    print(
        "ORIGIN fold-9 outer release completed: "
        f"variants={len(completion['ordered_variants'])} root={args.root / 'release'}"
    )


if __name__ == "__main__":
    main()
