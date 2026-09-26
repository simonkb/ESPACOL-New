#!/usr/bin/env python3
"""Audit and aggregate the paired EyePACS fold-9 ORIGIN ablations.

The script consumes only the sealed outer-release artifacts.  It independently
replays every reported metric from the per-image probabilities, verifies that
all variants contain the same patients/images in the same order, and computes
paired patient-cluster bootstrap intervals against ``origin_full``.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.origin_fold9_ablation_common import (
    EXPECTED_SPLIT_COUNTS,
    EXPECTED_SPLIT_HISTOGRAMS,
    EXPECTED_SPLIT_SIGNATURE,
    canonical_sha256,
    file_sha256,
    read_json,
    values_close,
    verify_checksummed_payload,
    verify_protocol,
    write_json_atomic,
)
from scripts.origin_v3_cv_common import metrics_from_confusion


N_CLASSES = 5
REFERENCE_VARIANT = "origin_full"
METRIC_ORDER = (
    "acc",
    "mae",
    "qwk",
    "balanced_acc",
    "macro_f1",
    "ece",
    "expected_grade_mae",
)
HIGHER_IS_BETTER = {
    "acc": True,
    "mae": False,
    "qwk": True,
    "balanced_acc": True,
    "macro_f1": True,
    "ece": False,
    "expected_grade_mae": False,
}
CSV_FIELDS = [
    "variant",
    "fold",
    "outer_index",
    "image_path_relative",
    "patient_id",
    "true_label",
    "primary_prediction",
    "expected_grade",
    *[f"p_grade_{grade}" for grade in range(N_CLASSES)],
]


def _ece(probabilities: np.ndarray, predicted: np.ndarray, labels: np.ndarray) -> float:
    confidence = probabilities[np.arange(len(predicted)), predicted]
    edges = np.linspace(0.0, 1.0, 16)
    total = float(len(predicted))
    result = 0.0
    for index in range(15):
        if index == 0:
            members = (confidence >= edges[index]) & (confidence <= edges[index + 1])
        else:
            members = (confidence > edges[index]) & (confidence <= edges[index + 1])
        if not np.any(members):
            continue
        result += float(np.sum(members)) / total * abs(
            float(np.mean(predicted[members] == labels[members]))
            - float(np.mean(confidence[members]))
        )
    return result


def metrics_from_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Recompute all reported grading metrics from released CSV rows."""

    if not rows:
        raise ValueError("prediction rows are empty")
    labels = np.asarray([int(row["true_label"]) for row in rows], dtype=np.int64)
    predicted = np.asarray([int(row["primary_prediction"]) for row in rows], dtype=np.int64)
    expected = np.asarray([float(row["expected_grade"]) for row in rows], dtype=np.float64)
    probabilities = np.asarray(
        [[float(row[f"p_grade_{grade}"]) for grade in range(N_CLASSES)] for row in rows],
        dtype=np.float64,
    )
    if np.any(labels < 0) or np.any(labels >= N_CLASSES):
        raise ValueError("released labels lie outside the five-grade scale")
    if np.any(predicted < 0) or np.any(predicted >= N_CLASSES):
        raise ValueError("released predictions lie outside the five-grade scale")
    if not np.all(np.isfinite(probabilities)) or np.any(probabilities < 0.0):
        raise ValueError("released probabilities are not finite nonnegative values")
    if not np.allclose(probabilities.sum(axis=1), 1.0, rtol=1e-7, atol=1e-7):
        raise ValueError("released class probabilities do not sum to one")
    if not np.all(np.isfinite(expected)):
        raise ValueError("released expected grades are non-finite")
    replayed_expected = probabilities @ np.arange(N_CLASSES, dtype=np.float64)
    if not np.allclose(expected, replayed_expected, rtol=1e-7, atol=2e-7):
        raise ValueError("expected grade does not replay from class probabilities")
    if not np.array_equal(predicted, np.argmax(probabilities, axis=1)):
        raise ValueError("primary prediction is not the declared class-MAP decision")

    confusion = np.zeros((N_CLASSES, N_CLASSES), dtype=np.int64)
    np.add.at(confusion, (labels, predicted), 1)
    result = metrics_from_confusion(confusion.tolist())
    result["ece"] = _ece(probabilities, predicted, labels)
    result["expected_grade_mae"] = float(np.mean(np.abs(expected - labels)))
    return result


def _identity(row: Mapping[str, Any]) -> tuple[str, str, int, int]:
    return (
        str(row["image_path_relative"]),
        str(row["patient_id"]),
        int(row["true_label"]),
        int(row["outer_index"]),
    )


def _read_prediction_csv(path: Path, *, variant: str) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != CSV_FIELDS:
            raise AssertionError(
                f"{variant} prediction schema changed: {reader.fieldnames!r}"
            )
        rows = list(reader)
    for expected_index, row in enumerate(rows):
        if row["variant"] != variant or int(row["fold"]) != 9:
            raise AssertionError(f"{variant} CSV has an incorrect variant/fold identity")
        if int(row["outer_index"]) != expected_index:
            raise AssertionError(f"{variant} CSV row order is not the locked split order")
        if not row["patient_id"]:
            raise AssertionError(f"{variant} CSV omits an EyePACS patient identity")
    if len({_identity(row) for row in rows}) != len(rows):
        raise AssertionError(f"{variant} CSV contains duplicate image identities")
    return rows


def _cluster_sufficient_statistics(
    rows: Sequence[Mapping[str, Any]], patient_order: Sequence[str]
) -> dict[str, np.ndarray]:
    patient_to_index = {patient: index for index, patient in enumerate(patient_order)}
    confusion = np.zeros((len(patient_order), N_CLASSES, N_CLASSES), dtype=np.float64)
    bin_total = np.zeros((len(patient_order), 15), dtype=np.float64)
    bin_correct = np.zeros_like(bin_total)
    bin_confidence = np.zeros_like(bin_total)
    expected_error = np.zeros(len(patient_order), dtype=np.float64)
    edges = np.linspace(0.0, 1.0, 16)
    for row in rows:
        patient_index = patient_to_index[str(row["patient_id"])]
        label = int(row["true_label"])
        prediction = int(row["primary_prediction"])
        probabilities = np.asarray(
            [float(row[f"p_grade_{grade}"]) for grade in range(N_CLASSES)],
            dtype=np.float64,
        )
        confidence = probabilities[prediction]
        # Match torch.bucketize semantics used by the 15-bin ECE implementation:
        # intervals are [0, 1/15], then (b/15, (b+1)/15].
        bin_index = int(np.searchsorted(edges, confidence, side="left") - 1)
        bin_index = min(14, max(0, bin_index))
        confusion[patient_index, label, prediction] += 1.0
        bin_total[patient_index, bin_index] += 1.0
        bin_correct[patient_index, bin_index] += float(label == prediction)
        bin_confidence[patient_index, bin_index] += confidence
        expected_error[patient_index] += abs(float(row["expected_grade"]) - label)
    return {
        "confusion": confusion,
        "bin_total": bin_total,
        "bin_correct": bin_correct,
        "bin_confidence": bin_confidence,
        "expected_error": expected_error,
    }


def _metrics_from_weighted_statistics(
    weights: np.ndarray, statistics: Mapping[str, np.ndarray]
) -> dict[str, np.ndarray]:
    confusion = np.einsum("bp,pij->bij", weights, statistics["confusion"], optimize=True)
    support = confusion.sum(axis=2)
    predicted_support = confusion.sum(axis=1)
    total = support.sum(axis=1)
    diagonal = np.diagonal(confusion, axis1=1, axis2=2)
    coordinates = np.arange(N_CLASSES, dtype=np.float64)
    absolute_weights = np.abs(coordinates[:, None] - coordinates[None, :])
    quadratic_weights = (coordinates[:, None] - coordinates[None, :]) ** 2
    quadratic_weights /= float((N_CLASSES - 1) ** 2)

    acc = 100.0 * diagonal.sum(axis=1) / total
    mae = np.einsum("bij,ij->b", confusion, absolute_weights, optimize=True) / total
    observed = np.einsum("bij,ij->b", confusion, quadratic_weights, optimize=True)
    expected = np.einsum(
        "bi,bj,ij->b", support, predicted_support, quadratic_weights, optimize=True
    ) / total
    qwk = np.zeros_like(expected)
    np.divide(observed, expected, out=qwk, where=expected > 0.0)
    qwk = np.where(expected > 0.0, 1.0 - qwk, 0.0)
    recall = np.divide(diagonal, support, out=np.zeros_like(diagonal), where=support > 0.0)
    precision = np.divide(
        diagonal,
        predicted_support,
        out=np.zeros_like(diagonal),
        where=predicted_support > 0.0,
    )
    present = support > 0.0
    balanced_acc = 100.0 * np.sum(recall * present, axis=1) / np.maximum(
        1.0, present.sum(axis=1)
    )
    f1 = np.divide(
        2.0 * recall * precision,
        recall + precision,
        out=np.zeros_like(recall),
        where=(recall + precision) > 0.0,
    )
    macro_f1 = np.sum(f1 * present, axis=1) / np.maximum(1.0, present.sum(axis=1))
    bin_total = weights @ statistics["bin_total"]
    bin_correct = weights @ statistics["bin_correct"]
    bin_confidence = weights @ statistics["bin_confidence"]
    ece = np.sum(np.abs(bin_correct - bin_confidence), axis=1) / total
    expected_grade_mae = (weights @ statistics["expected_error"]) / total
    return {
        "acc": acc,
        "mae": mae,
        "qwk": qwk,
        "balanced_acc": balanced_acc,
        "macro_f1": macro_f1,
        "ece": ece,
        "expected_grade_mae": expected_grade_mae,
    }


def paired_patient_cluster_bootstrap(
    rows_by_variant: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    reference_variant: str = REFERENCE_VARIANT,
    samples: int = 10_000,
    seed: int = 26_092_026,
    chunk_size: int = 256,
) -> dict[str, Any]:
    """Paired bootstrap over EyePACS patients, keeping both eyes together."""

    if samples < 1:
        raise ValueError("bootstrap samples must be positive")
    if reference_variant not in rows_by_variant:
        raise ValueError("bootstrap reference is absent")
    reference_rows = rows_by_variant[reference_variant]
    reference_identities = [_identity(row) for row in reference_rows]
    for variant, rows in rows_by_variant.items():
        if [_identity(row) for row in rows] != reference_identities:
            raise AssertionError(f"{variant} rows are not paired to the reference")
    patient_order = list(dict.fromkeys(str(row["patient_id"]) for row in reference_rows))
    if any(not patient for patient in patient_order):
        raise AssertionError("patient-cluster bootstrap requires non-empty patient IDs")
    statistics = {
        variant: _cluster_sufficient_statistics(rows, patient_order)
        for variant, rows in rows_by_variant.items()
    }
    observed = {variant: metrics_from_rows(rows) for variant, rows in rows_by_variant.items()}
    candidates = [variant for variant in rows_by_variant if variant != reference_variant]
    draws: dict[str, dict[str, list[np.ndarray]]] = {
        variant: {metric: [] for metric in METRIC_ORDER} for variant in candidates
    }
    rng = np.random.default_rng(seed)
    probabilities = np.full(len(patient_order), 1.0 / len(patient_order), dtype=np.float64)
    completed = 0
    while completed < samples:
        batch = min(chunk_size, samples - completed)
        weights = rng.multinomial(len(patient_order), probabilities, size=batch).astype(
            np.float64, copy=False
        )
        reference_metrics = _metrics_from_weighted_statistics(
            weights, statistics[reference_variant]
        )
        for variant in candidates:
            candidate_metrics = _metrics_from_weighted_statistics(weights, statistics[variant])
            for metric in METRIC_ORDER:
                draws[variant][metric].append(
                    candidate_metrics[metric] - reference_metrics[metric]
                )
        completed += batch

    comparisons: dict[str, Any] = {}
    for variant in candidates:
        metric_payload: dict[str, Any] = {}
        for metric in METRIC_ORDER:
            values = np.concatenate(draws[variant][metric])
            delta = float(observed[variant][metric]) - float(observed[reference_variant][metric])
            lower, upper = np.percentile(values, [2.5, 97.5])
            probability_positive = float((np.sum(values > 0.0) + 0.5 * np.sum(values == 0.0)) / samples)
            improvement_probability = (
                probability_positive if HIGHER_IS_BETTER[metric] else 1.0 - probability_positive
            )
            descriptive_two_sided_tail_fraction = min(
                1.0,
                2.0
                * min(
                    (float(np.sum(values <= 0.0)) + 1.0) / (samples + 1.0),
                    (float(np.sum(values >= 0.0)) + 1.0) / (samples + 1.0),
                ),
            )
            metric_payload[metric] = {
                "candidate_minus_reference": delta,
                "bootstrap_mean_delta": float(np.mean(values)),
                "ci95_percentile": [float(lower), float(upper)],
                "bootstrap_fraction_candidate_better": improvement_probability,
                "descriptive_two_sided_bootstrap_tail_fraction": (
                    descriptive_two_sided_tail_fraction
                ),
                "higher_is_better": HIGHER_IS_BETTER[metric],
            }
        comparisons[variant] = metric_payload
    return {
        "method": "paired_patient_cluster_bootstrap_percentile",
        "reference_variant": reference_variant,
        "patient_clusters": len(patient_order),
        "images": len(reference_rows),
        "samples": samples,
        "seed": seed,
        "delta_orientation": "candidate_minus_reference",
        "inference_scope": (
            "descriptive paired resampling conditional on the previously observed "
            "fold9 cohort and one training seed"
        ),
        "multiplicity_adjustment": (
            "none; bootstrap tail fractions are descriptive and are not p-values"
        ),
        "comparisons": comparisons,
    }


def _format_delta(payload: Mapping[str, Any], *, digits: int = 4) -> str:
    delta = float(payload["candidate_minus_reference"])
    lower, upper = payload["ci95_percentile"]
    return f"{delta:+.{digits}f} [{float(lower):+.{digits}f}, {float(upper):+.{digits}f}]"


def _write_csv_atomic(path: Path, fieldnames: Sequence[str], rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _write_text_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(text, encoding="utf-8")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def aggregate(
    *,
    root: Path,
    protocol_path: Path,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    protocol = read_json(protocol_path)
    variants = list(verify_protocol(protocol, expected_root=root))
    policy = protocol["evaluation_policy"]
    if bootstrap_samples != policy["paired_patient_cluster_bootstrap_samples"]:
        raise ValueError("bootstrap sample count differs from the locked protocol")
    if bootstrap_seed != policy["paired_patient_cluster_bootstrap_seed"]:
        raise ValueError("bootstrap seed differs from the locked protocol")
    release_root = root / "release"
    completion_path = release_root / "OUTER_RELEASE_COMPLETE.json"
    completion = read_json(completion_path)
    verify_checksummed_payload(
        completion, schema="origin-fold9-ablation-outer-release-complete-v1"
    )
    if completion.get("protocol_checksum_sha256") != protocol["content_checksum_sha256"]:
        raise AssertionError("outer release is bound to a different protocol")
    if completion.get("ordered_variants") != variants:
        raise AssertionError("outer release variant order differs from protocol")

    rows_by_variant: dict[str, list[dict[str, str]]] = {}
    metrics_by_variant: dict[str, dict[str, Any]] = {}
    training_by_variant: dict[str, Mapping[str, Any]] = {}
    manifests: dict[str, Mapping[str, Any]] = {}
    reference_identities: list[tuple[str, str, int, int]] | None = None
    for variant in variants:
        variant_dir = release_root / variant
        csv_path = variant_dir / "outer_predictions.csv"
        manifest_path = variant_dir / "outer_predictions_manifest.json"
        recorded = completion.get("artifacts", {}).get(variant)
        if not isinstance(recorded, Mapping):
            raise AssertionError(f"completion marker omits {variant}")
        for name, path in {"csv": csv_path, "manifest": manifest_path}.items():
            expected = recorded.get(f"{name}_sha256")
            if not path.is_file() or file_sha256(path) != expected:
                raise AssertionError(f"{variant} released {name} is missing or changed")
        manifest = read_json(manifest_path)
        verify_checksummed_payload(
            manifest, schema="origin-fold9-ablation-outer-predictions-v1"
        )
        if manifest.get("variant") != variant or manifest.get("fold") != 9:
            raise AssertionError(f"{variant} release manifest identity changed")
        if manifest.get("split_signature") != EXPECTED_SPLIT_SIGNATURE:
            raise AssertionError(f"{variant} release split signature changed")
        if manifest.get("n") != EXPECTED_SPLIT_COUNTS["locked_test"]:
            raise AssertionError(f"{variant} release has the wrong fold-9 size")
        if manifest.get("class_histogram") != EXPECTED_SPLIT_HISTOGRAMS["locked_test"]:
            raise AssertionError(f"{variant} release class histogram changed")
        if manifest.get("protocol_checksum_sha256") != protocol["content_checksum_sha256"]:
            raise AssertionError(f"{variant} release manifest protocol mismatch")
        if manifest.get("csv_sha256") != file_sha256(csv_path):
            raise AssertionError(f"{variant} release CSV checksum mismatch")
        if manifest.get("csv_path") != str(csv_path.resolve()):
            raise AssertionError(f"{variant} release manifest CSV path changed")
        expected_manifest_checksum = completion.get(
            "prediction_manifest_checksums", {}
        ).get(variant)
        if manifest.get("content_checksum_sha256") != expected_manifest_checksum:
            raise AssertionError(f"{variant} manifest differs from release completion")
        rows = _read_prediction_csv(csv_path, variant=variant)
        if manifest.get("n") != len(rows):
            raise AssertionError(f"{variant} release row count differs from manifest")
        identities = [_identity(row) for row in rows]
        if reference_identities is None:
            reference_identities = identities
        elif identities != reference_identities:
            raise AssertionError(f"{variant} released identities/order differ from reference")
        metrics = metrics_from_rows(rows)
        for key in (*METRIC_ORDER, "confusion", "per_grade_support", "per_grade_recall"):
            if not values_close(metrics.get(key), manifest.get("metrics", {}).get(key)):
                raise AssertionError(f"{variant} manifest does not replay metric {key}")
        rows_by_variant[variant] = rows
        metrics_by_variant[variant] = metrics
        training = manifest.get("training_summary")
        if not isinstance(training, Mapping):
            raise AssertionError(f"{variant} release manifest omits training summary")
        training_by_variant[variant] = training
        manifests[variant] = manifest

    bootstrap = paired_patient_cluster_bootstrap(
        rows_by_variant,
        reference_variant=REFERENCE_VARIANT,
        samples=bootstrap_samples,
        seed=bootstrap_seed,
    )
    result_rows: list[dict[str, Any]] = []
    for variant in variants:
        metrics = metrics_by_variant[variant]
        row: dict[str, Any] = {
            "variant": variant,
            "n": metrics["n"],
            "best_epoch": training_by_variant[variant].get("best_epoch"),
            "parameter_count": training_by_variant[variant].get("parameter_count"),
            "trainable_parameter_count": training_by_variant[variant].get(
                "trainable_parameter_count"
            ),
            "training_wall_time_seconds": training_by_variant[variant].get(
                "training_wall_time_seconds"
            ),
            "training_peak_cuda_memory_bytes": training_by_variant[variant].get(
                "training_peak_cuda_memory_bytes"
            ),
            "training_efficiency_scope": training_by_variant[variant].get(
                "training_efficiency_scope"
            ),
            **{metric: metrics[metric] for metric in METRIC_ORDER},
        }
        if variant == REFERENCE_VARIANT:
            for metric in METRIC_ORDER:
                row[f"delta_{metric}"] = 0.0
                row[f"delta_{metric}_ci95_low"] = 0.0
                row[f"delta_{metric}_ci95_high"] = 0.0
        else:
            for metric in METRIC_ORDER:
                comparison = bootstrap["comparisons"][variant][metric]
                row[f"delta_{metric}"] = comparison["candidate_minus_reference"]
                row[f"delta_{metric}_ci95_low"] = comparison["ci95_percentile"][0]
                row[f"delta_{metric}_ci95_high"] = comparison["ci95_percentile"][1]
        result_rows.append(row)

    output_json = root / "FOLD9_ABLATION_RESULTS.json"
    output_csv = root / "FOLD9_ABLATION_RESULTS.csv"
    output_markdown = root / "FOLD9_ABLATION_RESULTS.md"
    csv_fields = [
        "variant",
        "n",
        "best_epoch",
        "parameter_count",
        "trainable_parameter_count",
        "training_wall_time_seconds",
        "training_peak_cuda_memory_bytes",
        "training_efficiency_scope",
        *METRIC_ORDER,
        *[
            field
            for metric in METRIC_ORDER
            for field in (
                f"delta_{metric}",
                f"delta_{metric}_ci95_low",
                f"delta_{metric}_ci95_high",
            )
        ],
    ]
    _write_csv_atomic(output_csv, csv_fields, result_rows)

    markdown = [
        "# ORIGIN fold-9 paired ablation results",
        "",
        f"Paired patient-cluster bootstrap: {bootstrap_samples:,} resamples, "
        f"seed {bootstrap_seed}; all deltas are candidate minus `origin_full`.",
        "",
        "| Variant | Params (M) | Best epoch | Accuracy (%) | Δ accuracy [95% CI] | MAP MAE | Δ MAE [95% CI] | QWK | Δ QWK [95% CI] | Balanced acc. (%) | Macro-F1 | Expected-grade MAE |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for variant in variants:
        metrics = metrics_by_variant[variant]
        if variant == REFERENCE_VARIANT:
            delta_acc = delta_mae = delta_qwk = "reference"
        else:
            comparison = bootstrap["comparisons"][variant]
            delta_acc = _format_delta(comparison["acc"])
            delta_mae = _format_delta(comparison["mae"])
            delta_qwk = _format_delta(comparison["qwk"])
        markdown.append(
            f"| `{variant}` | "
            f"{float(training_by_variant[variant]['parameter_count']) / 1e6:.2f} | "
            f"{int(training_by_variant[variant]['best_epoch'])} | "
            f"{metrics['acc']:.2f} | {delta_acc} | "
            f"{metrics['mae']:.4f} | {delta_mae} | {metrics['qwk']:.4f} | "
            f"{delta_qwk} | {metrics['balanced_acc']:.2f} | "
            f"{metrics['macro_f1']:.4f} | {metrics['expected_grade_mae']:.4f} |"
        )
    markdown.extend(
        [
            "",
            "Intervals are paired at patient level, so both eyes from a patient are always resampled together.",
            "Lower values are better for both MAE columns and ECE; higher values are better for the remaining grading metrics.",
            "This is a retrospective mechanism study on a previously observed fold with one training seed; the ten-fold run remains the main performance estimate.",
            "No multiplicity adjustment is applied, and descriptive bootstrap tail fractions are not p-values.",
            "",
        ]
    )
    _write_text_atomic(output_markdown, "\n".join(markdown))

    payload: dict[str, Any] = {
        "schema": "origin-fold9-ablation-results-v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": "dr",
        "fold": 9,
        "scope": "locked_outer_fold9_paired_ablation_after_inner_validation_selection",
        "development_status": "ablation_only_do_not_tune_on_outer_results",
        "protocol_checksum_sha256": protocol["content_checksum_sha256"],
        "release_completion_checksum_sha256": completion["content_checksum_sha256"],
        "ordered_variants": variants,
        "metrics": metrics_by_variant,
        "training_summaries": training_by_variant,
        "paired_patient_cluster_bootstrap": bootstrap,
        "prediction_manifest_checksums": {
            variant: manifests[variant]["content_checksum_sha256"] for variant in variants
        },
        "output_csv": str(output_csv.resolve()),
        "output_csv_sha256": file_sha256(output_csv),
        "output_markdown": str(output_markdown.resolve()),
        "output_markdown_sha256": file_sha256(output_markdown),
    }
    payload["content_checksum_sha256"] = canonical_sha256(payload)
    write_json_atomic(output_json, payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=26_092_026)
    args = parser.parse_args()
    payload = aggregate(
        root=args.root.expanduser().resolve(),
        protocol_path=args.protocol.expanduser().resolve(),
        bootstrap_samples=args.bootstrap_samples,
        bootstrap_seed=args.bootstrap_seed,
    )
    print(
        "ORIGIN fold-9 ablations aggregated: "
        f"variants={len(payload['ordered_variants'])} "
        f"output={args.root / 'FOLD9_ABLATION_RESULTS.json'}"
    )


if __name__ == "__main__":
    main()
