#!/usr/bin/env python3
"""Aggregate and audit all locked outer folds of frozen ORIGIN-v3."""

from __future__ import annotations

import argparse
import csv
import math
import os
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Direct script execution puts only ``scripts/`` on sys.path.  The aggregate
# audit also imports the frozen dataset/split implementation from the repo.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from origin_v3_cv_common import (
    ARCHITECTURE_SIGNATURE,
    BASE_COMMIT,
    DATASET_SPECS,
    IMPLEMENTATION_SIGNATURE,
    canonical_sha256,
    file_sha256,
    metrics_from_confusion,
    read_json,
    sum_confusions,
    validate_pinned_config,
    values_close,
    verify_checksummed_payload,
    write_json_atomic,
)


PRIMARY_METRICS = ("acc", "mae", "qwk", "balanced_acc", "macro_f1")


def item_identity(item: tuple[str, int]) -> tuple[str, int]:
    return (str(Path(item[0]).expanduser().resolve(strict=False)), int(item[1]))


def aggregate(*, dataset: str, cv_root: Path, data_root: Path, protocol_path: Path) -> dict[str, Any]:
    import torch
    from Datasets.origin_data import class_histogram, load_origin_items, split_origin_items
    from training.origin_trainer import evaluate_origin_predictions

    spec = DATASET_SPECS[dataset]
    folds = list(range(int(spec["n_folds"])))
    protocol = read_json(protocol_path)
    verify_checksummed_payload(protocol, schema="origin-v3-full-cv-protocol-v1")
    for key, expected in {
        "dataset": dataset,
        "n_folds": spec["n_folds"],
        "n_images": spec["n_images"],
        "evaluation_scope": "outer_test_after_selection",
        "base_commit": BASE_COMMIT,
        "implementation_signature": IMPLEMENTATION_SIGNATURE,
        "architecture_signature": ARCHITECTURE_SIGNATURE,
        "config_signature": spec["config_signature"],
    }.items():
        if protocol.get(key) != expected:
            raise AssertionError(f"protocol {key}: expected {expected!r}, got {protocol.get(key)!r}")
    validate_pinned_config(protocol.get("frozen_config", {}), dataset)
    if Path(protocol["data_root"]).resolve() != data_root.resolve():
        raise AssertionError("aggregator data root differs from locked protocol")
    if Path(protocol["cv_root"]).resolve() != cv_root.resolve():
        raise AssertionError("aggregator output root differs from locked protocol")

    items = load_origin_items(dataset, str(data_root))
    if len(items) != int(spec["n_images"]):
        raise AssertionError(f"dataset size changed: expected {spec['n_images']}, got {len(items)}")
    all_identities = {item_identity(item) for item in items}
    if len(all_identities) != len(items):
        raise AssertionError("dataset contains duplicate path-label identities")
    all_paths = [identity[0] for identity in all_identities]
    if len(set(all_paths)) != len(items):
        raise AssertionError("dataset contains duplicate image paths")

    markers: list[dict[str, Any]] = []
    result_rows: list[dict[str, Any]] = []
    outer_sets: list[set[tuple[str, int]]] = []
    seen_signatures: set[str] = set()
    oof_rows: list[dict[str, str]] = []
    oof_fieldnames: list[str] | None = None
    oof_probabilities: list[list[float]] = []
    oof_predictions: list[int] = []
    oof_labels: list[int] = []
    expected_prediction_header = [
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
        *[f"p_grade_{grade}" for grade in range(5)],
        *[f"p_gt_{boundary}" for boundary in range(4)],
        *[f"total_rate_{boundary}" for boundary in range(4)],
    ]
    workers_root = cv_root / "workers"
    observed_worker_folds = {
        path.name
        for path in workers_root.iterdir()
        if path.is_dir() and path.name.startswith("fold")
    }
    expected_worker_folds = {f"fold{fold}" for fold in folds}
    if observed_worker_folds != expected_worker_folds:
        raise AssertionError(
            "worker fold set mismatch: "
            f"expected={sorted(expected_worker_folds)}, observed={sorted(observed_worker_folds)}"
        )
    for fold in folds:
        fold_dir = cv_root / "workers" / f"fold{fold}" / f"fold{fold}"
        marker_path = fold_dir / "CV_FOLD_COMPLETE.json"
        marker = read_json(marker_path)
        verify_checksummed_payload(marker, schema="origin-v3-cv-fold-complete-v1")
        for key, expected in {
            "dataset": dataset,
            "fold": fold,
            "evaluation_scope": "outer_test_after_selection",
            "protocol_checksum_sha256": protocol["content_checksum_sha256"],
            "launch_commit": protocol["launch_commit"],
            "implementation_signature": IMPLEMENTATION_SIGNATURE,
            "architecture_signature": ARCHITECTURE_SIGNATURE,
            "config_signature": spec["config_signature"],
        }.items():
            if marker.get(key) != expected:
                raise AssertionError(f"fold {fold} marker {key} mismatch")
        if marker["split_signature"] in seen_signatures:
            raise AssertionError("duplicate split signature across folds")
        seen_signatures.add(marker["split_signature"])
        expected_artifacts = {
            "result": fold_dir / "result.json",
            "split_manifest": fold_dir / "split_manifest.json",
            "checkpoint": fold_dir / "best.pth",
            "validation_certificate": fold_dir / "validation_certificates.json",
            "outer_predictions_csv": fold_dir / "outer_predictions.csv",
            "outer_predictions_manifest": fold_dir / "outer_predictions_manifest.json",
        }
        if set(marker["artifact_sha256"]) != set(expected_artifacts):
            raise AssertionError(f"fold {fold} marker artifact set is incomplete")
        if set(marker["artifact_paths"]) != set(expected_artifacts):
            raise AssertionError(f"fold {fold} marker artifact-path set is incomplete")
        for name, expected_hash in marker["artifact_sha256"].items():
            artifact = Path(marker["artifact_paths"][name])
            if artifact.resolve() != expected_artifacts[name].resolve():
                raise AssertionError(f"fold {fold} marker points outside its artifact directory")
            if file_sha256(artifact) != expected_hash:
                raise AssertionError(f"fold {fold} artifact changed after audit: {artifact}")

        train, validation, locked_test = split_origin_items(
            dataset,
            items,
            fold,
            n_folds=int(spec["n_folds"]),
            val_fraction=0.1,
            seed=42,
        )
        manifest = read_json(fold_dir / "split_manifest.json")
        expected_counts = {
            "train": len(train),
            "validation": len(validation),
            "locked_test": len(locked_test),
        }
        expected_histograms = {
            "train": class_histogram(train, 5),
            "validation": class_histogram(validation, 5),
            "locked_test": class_histogram(locked_test, 5),
        }
        if manifest.get("counts") != expected_counts or manifest.get("histograms") != expected_histograms:
            raise AssertionError(f"fold {fold} deterministic split no longer matches its manifest")
        split_sets = [
            {item_identity(item) for item in subset}
            for subset in (train, validation, locked_test)
        ]
        if any(split_sets[i] & split_sets[j] for i in range(3) for j in range(i + 1, 3)):
            raise AssertionError(f"fold {fold} has overlapping image splits")
        if set().union(*split_sets) != all_identities:
            raise AssertionError(f"fold {fold} does not partition the complete dataset")
        if dataset == "dr":
            patient_sets = [
                {Path(path).stem.rsplit("_", 1)[0] for path, _ in subset}
                for subset in (train, validation, locked_test)
            ]
            if any(patient_sets[i] & patient_sets[j] for i in range(3) for j in range(i + 1, 3)):
                raise AssertionError(f"EyePACS fold {fold} leaks a patient across splits")

        prediction_manifest = read_json(expected_artifacts["outer_predictions_manifest"])
        verify_checksummed_payload(
            prediction_manifest, schema="origin-v3-outer-predictions-v1"
        )
        if prediction_manifest.get("split_signature") != marker.get("split_signature"):
            raise AssertionError(f"fold {fold} prediction export split mismatch")
        if prediction_manifest.get("csv_sha256") != file_sha256(
            expected_artifacts["outer_predictions_csv"]
        ):
            raise AssertionError(f"fold {fold} prediction CSV checksum mismatch")
        with expected_artifacts["outer_predictions_csv"].open(
            newline="", encoding="utf-8"
        ) as stream:
            reader = csv.DictReader(stream)
            if reader.fieldnames is None:
                raise AssertionError(f"fold {fold} prediction CSV has no header")
            if list(reader.fieldnames) != expected_prediction_header:
                raise AssertionError(f"fold {fold} prediction CSV schema changed")
            if oof_fieldnames is None:
                oof_fieldnames = list(reader.fieldnames)
            elif list(reader.fieldnames) != oof_fieldnames:
                raise AssertionError("outer-prediction CSV schemas differ across folds")
            fold_rows = list(reader)
        if len(fold_rows) != len(locked_test):
            raise AssertionError(f"fold {fold} prediction row count changed")
        fold_identities: list[tuple[str, int]] = []
        fold_confusion = [[0 for _ in range(5)] for _ in range(5)]
        fold_probabilities: list[list[float]] = []
        fold_predictions: list[int] = []
        fold_labels: list[int] = []
        for expected_index, row in enumerate(fold_rows):
            if int(row["fold"]) != fold or int(row["outer_index"]) != expected_index:
                raise AssertionError(f"fold {fold} prediction row order changed")
            recorded_path = Path(row["image_path_relative"])
            resolved_path = (
                recorded_path.resolve(strict=False)
                if recorded_path.is_absolute()
                else (data_root / recorded_path).resolve(strict=False)
            )
            label = int(row["true_label"])
            prediction = int(row["primary_prediction"])
            if label not in range(5) or prediction not in range(5):
                raise AssertionError(f"fold {fold} prediction row has an invalid grade")
            if prediction != int(row["class_map"]):
                raise AssertionError(f"fold {fold} primary prediction is not class-MAP")
            identity = (str(resolved_path), label)
            if identity != item_identity(locked_test[expected_index]):
                raise AssertionError(f"fold {fold} prediction row differs from locked split order")
            fold_identities.append(identity)
            probabilities = [float(row[f"p_grade_{grade}"]) for grade in range(5)]
            if any(
                not math.isfinite(value) or value < 0.0 or value > 1.0
                for value in probabilities
            ) or not math.isclose(sum(probabilities), 1.0, rel_tol=1e-6, abs_tol=1e-6):
                raise AssertionError(f"fold {fold} contains invalid class probabilities")
            cumulative = [float(row[f"p_gt_{boundary}"]) for boundary in range(4)]
            rates = [float(row[f"total_rate_{boundary}"]) for boundary in range(4)]
            if any(not math.isfinite(value) or value < 0.0 or value > 1.0 for value in cumulative):
                raise AssertionError(f"fold {fold} contains invalid cumulative probabilities")
            if any(not math.isfinite(value) or value < 0.0 or value > 64.0 for value in rates):
                raise AssertionError(f"fold {fold} contains invalid total rates")
            if not math.isfinite(float(row["expected_grade"])):
                raise AssertionError(f"fold {fold} contains invalid expected grades")
            if dataset == "dr" and row["patient_id"] != Path(resolved_path).stem.rsplit("_", 1)[0]:
                raise AssertionError(f"fold {fold} contains an invalid patient identity")
            fold_probabilities.append(probabilities)
            fold_predictions.append(prediction)
            fold_labels.append(label)
            oof_probabilities.append(probabilities)
            oof_predictions.append(prediction)
            oof_labels.append(label)
            fold_confusion[label][prediction] += 1
            oof_rows.append(row)
        if len(set(fold_identities)) != len(fold_identities):
            raise AssertionError(f"fold {fold} prediction CSV contains duplicate images")
        if set(fold_identities) != split_sets[2]:
            raise AssertionError(f"fold {fold} prediction CSV does not cover its outer split")
        if fold_confusion != marker["outer_test"]["confusion"]:
            raise AssertionError(f"fold {fold} prediction CSV does not reproduce confusion")
        fold_metrics = evaluate_origin_predictions(
            torch.tensor(fold_probabilities, dtype=torch.float64),
            torch.tensor(fold_predictions, dtype=torch.long),
            torch.tensor(fold_labels, dtype=torch.long),
        )
        exported_metrics = prediction_manifest.get("metrics", {})
        # The fold-completion marker intentionally stores the compact metrics
        # needed for CV aggregation.  Per-grade recall remains bound to the
        # checksummed prediction export and is replayed from the CSV here, but
        # is not duplicated in the compact marker.
        for key in (*PRIMARY_METRICS, "ece", "confusion", "per_grade_support"):
            if not values_close(fold_metrics.get(key), marker["outer_test"].get(key)):
                raise AssertionError(f"fold {fold} CSV does not reproduce marker {key}")
        for key in (
            *PRIMARY_METRICS,
            "ece",
            "confusion",
            "per_grade_recall",
            "per_grade_support",
        ):
            if not values_close(fold_metrics.get(key), exported_metrics.get(key)):
                raise AssertionError(f"fold {fold} CSV does not reproduce export {key}")
        outer_sets.append(split_sets[2])
        outer = marker["outer_test"]
        result_rows.append(
            {"fold": fold, **{key: outer[key] for key in ("n", *PRIMARY_METRICS, "ece")}}
        )
        markers.append(marker)

    for left in range(len(outer_sets)):
        for right in range(left + 1, len(outer_sets)):
            if outer_sets[left] & outer_sets[right]:
                raise AssertionError(f"outer folds {left} and {right} overlap")
    outer_union = set().union(*outer_sets)
    if outer_union != all_identities:
        raise AssertionError(
            f"outer-fold coverage incomplete: union={len(outer_union)}, dataset={len(all_identities)}"
        )

    pooled = metrics_from_confusion(
        sum_confusions([marker["outer_test"]["confusion"] for marker in markers])
    )
    pooled_prediction_metrics = evaluate_origin_predictions(
        torch.tensor(oof_probabilities, dtype=torch.float64),
        torch.tensor(oof_predictions, dtype=torch.long),
        torch.tensor(oof_labels, dtype=torch.long),
    )
    for key in (*PRIMARY_METRICS, "confusion", "per_grade_recall", "per_grade_support"):
        if not values_close(pooled.get(key), pooled_prediction_metrics.get(key)):
            raise AssertionError(f"pooled prediction exports do not reproduce {key}")
    pooled["ece"] = float(pooled_prediction_metrics["ece"])
    fold_summary: dict[str, dict[str, float]] = {}
    for key in (*PRIMARY_METRICS, "ece"):
        values = [float(row[key]) for row in result_rows]
        fold_summary[key] = {
            "mean": statistics.mean(values),
            "sample_sd": statistics.stdev(values),
        }
    if len(oof_rows) != len(all_identities):
        raise AssertionError("OOF prediction exports do not contain exactly one row per image")

    if oof_fieldnames is None:
        raise AssertionError("no outer-prediction CSV schema was observed")
    oof_csv_path = cv_root / "CV_OOF_PREDICTIONS.csv"
    oof_temporary = oof_csv_path.with_name(f".{oof_csv_path.name}.{os.getpid()}.tmp")
    try:
        with oof_temporary.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=oof_fieldnames)
            writer.writeheader()
            writer.writerows(oof_rows)
        os.replace(oof_temporary, oof_csv_path)
    finally:
        if oof_temporary.exists():
            oof_temporary.unlink()

    summary: dict[str, Any] = {
        "schema": "origin-v3-full-cv-summary-v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": dataset,
        "evaluation_scope": "locked_outer_cross_validation_after_inner_validation_selection",
        "development_status": "final_protocol_evaluation_do_not_tune_on_outer_results",
        "launch_commit": protocol["launch_commit"],
        "base_commit": protocol["base_commit"],
        "implementation_signature": IMPLEMENTATION_SIGNATURE,
        "architecture_signature": ARCHITECTURE_SIGNATURE,
        "config_signature": spec["config_signature"],
        "protocol_checksum_sha256": protocol["content_checksum_sha256"],
        "fold_count": len(folds),
        "coverage": {
            "expected_images": int(spec["n_images"]),
            "observed_unique_outer_images": len(outer_union),
            "outer_folds_pairwise_disjoint": True,
            "outer_union_equals_dataset": True,
        },
        "fold_results": result_rows,
        "fold_mean_and_sample_sd": fold_summary,
        "pooled_oof_from_prediction_exports": pooled,
        "oof_predictions_csv": str(oof_csv_path.resolve()),
        "oof_predictions_csv_sha256": file_sha256(oof_csv_path),
        "fold_marker_sha256": {
            str(marker["fold"]): marker["content_checksum_sha256"] for marker in markers
        },
    }
    csv_path = cv_root / "CV_FOLD_RESULTS.csv"
    temporary = csv_path.with_name(f".{csv_path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(
                stream,
                fieldnames=["fold", "n", *PRIMARY_METRICS, "ece"],
            )
            writer.writeheader()
            writer.writerows(result_rows)
        os.replace(temporary, csv_path)
    finally:
        if temporary.exists():
            temporary.unlink()
    summary["fold_results_csv"] = str(csv_path.resolve())
    summary["fold_results_csv_sha256"] = file_sha256(csv_path)
    summary["content_checksum_sha256"] = canonical_sha256(summary)
    # The checksummed JSON is the completion marker, so publish it only after
    # every companion artifact has been written successfully.
    write_json_atomic(cv_root / "CV_SUMMARY.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=tuple(DATASET_SPECS), required=True)
    parser.add_argument("--cv-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    args = parser.parse_args()
    summary = aggregate(
        dataset=args.dataset,
        cv_root=args.cv_root,
        data_root=args.data_root,
        protocol_path=args.protocol,
    )
    pooled = summary["pooled_oof_from_prediction_exports"]
    means = summary["fold_mean_and_sample_sd"]
    print(f"ORIGIN-v3 {args.dataset} full CV audit passed")
    print(
        "pooled OOF: "
        f"acc={pooled['acc']:.4f} mae={pooled['mae']:.6f} "
        f"qwk={pooled['qwk']:.6f} bal_acc={pooled['balanced_acc']:.4f} "
        f"macro_f1={pooled['macro_f1']:.6f} ece={pooled['ece']:.6f}"
    )
    print(
        "fold mean±sample-SD accuracy: "
        f"{means['acc']['mean']:.4f} ± {means['acc']['sample_sd']:.4f}"
    )


if __name__ == "__main__":
    main()
