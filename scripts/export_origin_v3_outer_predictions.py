#!/usr/bin/env python3
"""Read-only export of one frozen ORIGIN-v3 locked outer fold."""

from __future__ import annotations

import argparse
import csv
import gc
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

from configs.origin_config import OriginConfig
from Datasets.origin_data import (
    OriginFundusTransform,
    OriginImageDataset,
    class_histogram,
    load_origin_items,
    split_origin_items,
    split_paths_are_disjoint,
)
from models.origin import build_origin_model
from scripts.origin_v3_cv_common import (
    ARCHITECTURE_SIGNATURE,
    BASE_COMMIT,
    DATASET_SPECS,
    IMPLEMENTATION_SIGNATURE,
    canonical_sha256,
    file_sha256,
    read_json,
    validate_pinned_config,
    values_close,
    verify_checksummed_payload,
    write_json_atomic,
)
from train_origin import split_signature
from training.origin_trainer import (
    _architecture_record,
    _canonical_sha256,
    _critical_config,
    evaluate_origin_predictions,
    origin_implementation_signature,
)


PRIMARY_METRICS = ("acc", "mae", "qwk", "ece", "balanced_acc", "macro_f1")


def field(output: object, name: str) -> torch.Tensor:
    value = output.get(name) if isinstance(output, Mapping) else getattr(output, name)
    if not torch.is_tensor(value):
        raise TypeError(f"ORIGIN output {name!r} is not a tensor")
    return value


def relative_image_path(path: str, data_root: Path) -> str:
    resolved = Path(path).expanduser().resolve(strict=False)
    try:
        return resolved.relative_to(data_root.resolve()).as_posix()
    except ValueError:
        return resolved.as_posix()


def export_predictions(
    *,
    checkpoint_path: Path,
    result_path: Path,
    split_manifest_path: Path,
    data_root: Path,
    protocol_path: Path,
    output_csv: Path,
    output_manifest: Path,
    num_workers: int,
) -> dict[str, Any]:
    if output_manifest.exists():
        raise FileExistsError("refusing to overwrite a completed outer-prediction export")
    if not torch.cuda.is_available():
        raise RuntimeError("production outer-prediction export requires CUDA")

    protocol = read_json(protocol_path)
    verify_checksummed_payload(protocol, schema="origin-v3-full-cv-protocol-v1")
    result = read_json(result_path)
    split_manifest = read_json(split_manifest_path)
    checkpoint_hash_before = file_sha256(checkpoint_path)
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if state.get("schema") != "origin-checkpoint-v3":
        raise ValueError("outer prediction export requires an ORIGIN-v3 checkpoint")
    config_values = dict(state.get("config", {}))
    allowed = {item.name for item in fields(OriginConfig)}
    cfg = OriginConfig(**{key: value for key, value in config_values.items() if key in allowed})
    dataset = cfg.dataset
    spec = DATASET_SPECS[dataset]
    fold = int(state["fold"])
    if fold not in range(int(spec["n_folds"])):
        raise ValueError(f"invalid checkpoint fold {fold}")

    expected_protocol = {
        "dataset": dataset,
        "n_folds": spec["n_folds"],
        "n_images": spec["n_images"],
        "base_commit": BASE_COMMIT,
        "implementation_signature": IMPLEMENTATION_SIGNATURE,
        "architecture_signature": ARCHITECTURE_SIGNATURE,
        "config_signature": spec["config_signature"],
        "evaluation_scope": "outer_test_after_selection",
    }
    for key, expected in expected_protocol.items():
        if protocol.get(key) != expected:
            raise AssertionError(f"protocol {key} mismatch")
    if Path(protocol["data_root"]).resolve() != data_root.resolve():
        raise AssertionError("prediction-export data root differs from protocol")
    expected_fold_dir = Path(protocol["cv_root"]) / "workers" / f"fold{fold}" / f"fold{fold}"
    expected_paths = {
        "checkpoint": expected_fold_dir / "best.pth",
        "result": expected_fold_dir / "result.json",
        "split manifest": expected_fold_dir / "split_manifest.json",
        "output CSV": expected_fold_dir / "outer_predictions.csv",
        "output manifest": expected_fold_dir / "outer_predictions_manifest.json",
    }
    observed_paths = {
        "checkpoint": checkpoint_path,
        "result": result_path,
        "split manifest": split_manifest_path,
        "output CSV": output_csv,
        "output manifest": output_manifest,
    }
    for name, expected_path in expected_paths.items():
        if observed_paths[name].resolve() != expected_path.resolve():
            raise AssertionError(f"{name} path is outside the locked fold directory")
    validate_pinned_config(protocol.get("frozen_config", {}), dataset)
    validate_pinned_config(config_values, dataset)

    if origin_implementation_signature() != IMPLEMENTATION_SIGNATURE:
        raise AssertionError("active ORIGIN implementation differs from frozen V3")
    for key, expected in {
        "fold": fold,
        "split_signature": split_manifest.get("signature"),
        "implementation_signature": IMPLEMENTATION_SIGNATURE,
        "architecture_signature": ARCHITECTURE_SIGNATURE,
        "config_signature": spec["config_signature"],
    }.items():
        if state.get(key) != expected:
            raise AssertionError(f"checkpoint {key} mismatch")
    if result.get("fold") != fold or result.get("best_epoch") != state.get("epoch"):
        raise AssertionError("result does not identify the selected checkpoint")
    if result.get("test_evaluated") is not True:
        raise AssertionError("outer predictions require an explicitly evaluated test fold")
    if split_manifest.get("schema") != "origin-split-v2":
        raise AssertionError("unexpected split manifest schema")
    if split_manifest.get("dataset") != dataset or split_manifest.get("fold") != fold:
        raise AssertionError("split manifest dataset/fold mismatch")
    if split_manifest.get("evaluation_scope") != "outer_test_after_selection":
        raise AssertionError("split manifest did not authorize outer evaluation")

    items = load_origin_items(
        dataset,
        str(data_root),
        labels_csv=cfg.labels_csv,
        image_column=cfg.image_column,
        label_column=cfg.label_column,
        image_dir=cfg.image_dir,
    )
    if len(items) != int(spec["n_images"]):
        raise AssertionError("dataset size changed before prediction export")
    train_items, validation_items, outer_items = split_origin_items(
        dataset,
        items,
        fold,
        n_folds=cfg.n_folds,
        val_fraction=cfg.val_fraction,
        seed=cfg.seed,
    )
    if not split_paths_are_disjoint(train_items, validation_items, outer_items):
        raise AssertionError("reconstructed split contains overlapping paths")
    expected_counts = {
        "train": len(train_items),
        "validation": len(validation_items),
        "locked_test": len(outer_items),
    }
    expected_histograms = {
        "train": class_histogram(train_items, cfg.n_classes),
        "validation": class_histogram(validation_items, cfg.n_classes),
        "locked_test": class_histogram(outer_items, cfg.n_classes),
    }
    if split_manifest.get("counts") != expected_counts:
        raise AssertionError("reconstructed split counts changed")
    if split_manifest.get("histograms") != expected_histograms:
        raise AssertionError("reconstructed split histograms changed")
    signature = split_signature(
        ("train", train_items),
        ("validation", validation_items),
        ("locked_test", outer_items),
    )
    if signature != split_manifest.get("signature") or signature != state.get("split_signature"):
        raise AssertionError("content-addressed split signature changed")

    model = build_origin_model(
        num_classes=cfg.n_classes,
        encoder_name=cfg.encoder,
        pretrained=False,
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
        grad_checkpoint=False,
    )
    architecture = _architecture_record(model)
    if state.get("architecture") != architecture:
        raise AssertionError("reconstructed architecture differs from checkpoint")
    if _canonical_sha256(architecture) != ARCHITECTURE_SIGNATURE:
        raise AssertionError("reconstructed architecture signature differs")
    critical_config = _critical_config(cfg)
    if state.get("critical_config") != critical_config:
        raise AssertionError("reconstructed critical config differs from checkpoint")
    if _canonical_sha256(critical_config) != spec["config_signature"]:
        raise AssertionError("reconstructed config signature differs")
    model.load_state_dict(state["model_state"], strict=True)
    del state, train_items, validation_items
    gc.collect()

    transform = OriginFundusTransform(cfg.img_size, augment=False)
    loader = DataLoader(
        OriginImageDataset(outer_items, transform),
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=False,
    )
    device = torch.device("cuda")
    model.to(device).eval()
    probabilities: list[torch.Tensor] = []
    cumulative: list[torch.Tensor] = []
    total_rates: list[torch.Tensor] = []
    expected_grades: list[torch.Tensor] = []
    class_maps: list[torch.Tensor] = []
    medians: list[torch.Tensor] = []
    rounded: list[torch.Tensor] = []
    labels_all: list[torch.Tensor] = []
    indices_all: list[torch.Tensor] = []
    with torch.inference_mode():
        for images, pixel_masks, labels, indices in loader:
            images = images.to(device, non_blocking=True)
            pixel_masks = pixel_masks.to(device, non_blocking=True)
            with torch.autocast(device_type="cuda", enabled=bool(cfg.amp)):
                output = model(
                    images,
                    pixel_valid_mask=pixel_masks,
                    force_decoder_fp64=True,
                )
            probabilities.append(field(output, "class_probs").detach().cpu().double())
            cumulative.append(field(output, "cumulative_probs").detach().cpu().double())
            total_rates.append(field(output, "total_rates").detach().cpu().double())
            expected = field(output, "expected_grade").detach().cpu().double()
            expected_grades.append(expected)
            class_maps.append(field(output, "class_map").detach().cpu().long())
            medians.append(field(output, "posterior_median").detach().cpu().long())
            rounded.append(expected.round().long().clamp(0, cfg.n_classes - 1))
            labels_all.append(labels.long().cpu())
            indices_all.append(torch.as_tensor(indices, dtype=torch.long).cpu())

    probs = torch.cat(probabilities)
    cumulative_probs = torch.cat(cumulative)
    rates = torch.cat(total_rates)
    expected = torch.cat(expected_grades)
    class_map = torch.cat(class_maps).clamp(0, cfg.n_classes - 1)
    posterior_median = torch.cat(medians).clamp(0, cfg.n_classes - 1)
    rounded_expected = torch.cat(rounded)
    labels = torch.cat(labels_all)
    indices = torch.cat(indices_all)
    expected_indices = list(range(len(outer_items)))
    if indices.tolist() != expected_indices:
        raise AssertionError("outer loader order/coverage changed")
    for index, label in enumerate(labels.tolist()):
        if int(outer_items[index][1]) != int(label):
            raise AssertionError("outer loader label differs from split item")

    decisions = {
        "class_map": class_map,
        "posterior_median": posterior_median,
        "rounded_expected": rounded_expected,
    }
    primary = decisions[cfg.decision_rule]
    metrics = evaluate_origin_predictions(probs, primary, labels)
    reported = result.get("test", {})
    for key in PRIMARY_METRICS:
        if not values_close(metrics.get(key), reported.get(key)):
            raise AssertionError(
                f"outer prediction export does not reproduce {key}: "
                f"{metrics.get(key)!r} != {reported.get(key)!r}"
            )
    for key in ("confusion", "per_grade_recall", "per_grade_support"):
        if not values_close(metrics.get(key), reported.get(key)):
            raise AssertionError(f"outer prediction export does not reproduce {key}")

    fieldnames = [
        "fold",
        "outer_index",
        "image_path_relative",
        "patient_id",
        "true_label",
        "primary_prediction",
        "class_map",
        "posterior_median",
        "rounded_expected",
        "expected_grade",
        *[f"p_grade_{grade}" for grade in range(cfg.n_classes)],
        *[f"p_gt_{boundary}" for boundary in range(cfg.n_classes - 1)],
        *[f"total_rate_{boundary}" for boundary in range(cfg.n_classes - 1)],
    ]
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_csv.with_name(f".{output_csv.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fieldnames)
            writer.writeheader()
            for index, (path, label) in enumerate(outer_items):
                image_path = relative_image_path(path, data_root)
                patient_id = Path(path).stem.rsplit("_", 1)[0] if dataset == "dr" else ""
                row: dict[str, Any] = {
                    "fold": fold,
                    "outer_index": index,
                    "image_path_relative": image_path,
                    "patient_id": patient_id,
                    "true_label": int(label),
                    "primary_prediction": int(primary[index]),
                    "class_map": int(class_map[index]),
                    "posterior_median": int(posterior_median[index]),
                    "rounded_expected": int(rounded_expected[index]),
                    "expected_grade": float(expected[index]),
                }
                row.update(
                    {f"p_grade_{grade}": float(probs[index, grade]) for grade in range(cfg.n_classes)}
                )
                row.update(
                    {
                        f"p_gt_{boundary}": float(cumulative_probs[index, boundary])
                        for boundary in range(cfg.n_classes - 1)
                    }
                )
                row.update(
                    {
                        f"total_rate_{boundary}": float(rates[index, boundary])
                        for boundary in range(cfg.n_classes - 1)
                    }
                )
                writer.writerow(row)
        os.replace(temporary, output_csv)
    finally:
        if temporary.exists():
            temporary.unlink()

    payload: dict[str, Any] = {
        "schema": "origin-v3-outer-predictions-v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": dataset,
        "fold": fold,
        "scope": "locked_outer_test_after_inner_validation_selection",
        "decision_rule": cfg.decision_rule,
        "n": len(outer_items),
        "class_histogram": class_histogram(outer_items, cfg.n_classes),
        "split_signature": signature,
        "protocol_checksum_sha256": protocol["content_checksum_sha256"],
        "launch_commit": protocol["launch_commit"],
        "implementation_signature": IMPLEMENTATION_SIGNATURE,
        "architecture_signature": ARCHITECTURE_SIGNATURE,
        "config_signature": spec["config_signature"],
        "checkpoint_epoch": int(result["best_epoch"]),
        "checkpoint_sha256": checkpoint_hash_before,
        "result_sha256": file_sha256(result_path),
        "export_implementation_sha256": file_sha256(Path(__file__).resolve()),
        "csv_path": str(output_csv.resolve()),
        "csv_sha256": file_sha256(output_csv),
        "metrics": metrics,
    }
    if file_sha256(checkpoint_path) != checkpoint_hash_before:
        raise AssertionError("checkpoint changed during outer-prediction export")
    payload["content_checksum_sha256"] = canonical_sha256(payload)
    write_json_atomic(output_manifest, payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("--num-workers", type=int, default=8)
    args = parser.parse_args()
    payload = export_predictions(
        checkpoint_path=args.checkpoint,
        result_path=args.result,
        split_manifest_path=args.split_manifest,
        data_root=args.data_root,
        protocol_path=args.protocol,
        output_csv=args.output_csv,
        output_manifest=args.output_manifest,
        num_workers=args.num_workers,
    )
    print(
        "ORIGIN-v3 outer predictions exported: "
        f"dataset={payload['dataset']} fold={payload['fold']} n={payload['n']} "
        f"csv={payload['csv_path']}"
    )


if __name__ == "__main__":
    main()
