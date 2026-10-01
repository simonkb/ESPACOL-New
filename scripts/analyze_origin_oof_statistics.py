#!/usr/bin/env python3
"""Recompute and cluster-bootstrap ORIGIN OOF statistics.

The analysis consumes only checksum-sealed aggregate intervention artifacts.  It
never loads a model checkpoint.  Grade posteriors are deduplicated from the
four boundary-census rows for each image, while intervention AUCs remain paired
at image--boundary level.  EyePACS resampling keeps both eyes from a patient in
one cluster; APTOS uses the image as its cluster because patient identifiers are
not available.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import math
from pathlib import Path
import sys
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.audit_origin_oof_interventions import iter_jsonl_gzip
from scripts.origin_oof_intervention_common import AGGREGATE_SCHEMA, METHODS, ROW_SCHEMA
from scripts.origin_v3_cv_common import (
    canonical_sha256,
    file_sha256,
    read_json,
    verify_checksummed_payload,
    write_json_atomic,
)


STATISTICS_PROTOCOL_SCHEMA = "origin-oof-statistics-protocol-v1"
PROPER_SCORE_SCHEMA = "origin-oof-proper-score-calibration-v1"
GRADING_BOOTSTRAP_SCHEMA = "origin-oof-grading-cluster-bootstrap-v1"
INTERVENTION_BOOTSTRAP_SCHEMA = "origin-oof-intervention-paired-bootstrap-v1"
STATISTICS_MANIFEST_SCHEMA = "origin-oof-statistics-manifest-v1"

GRADING_METRICS = (
    "accuracy_percent",
    "mae",
    "qwk",
    "balanced_accuracy_percent",
    "macro_f1",
    "expected_grade_mae",
    "nll",
    "rps",
    "multiclass_brier",
)


@dataclass(frozen=True)
class StatisticsConfig:
    bootstrap_samples: int
    bootstrap_seed: int
    bootstrap_chunk_size: int
    confidence_level: float
    reliability_bins: int
    nll_probability_floor: float


@dataclass(frozen=True)
class GradeRow:
    dataset: str
    image_key: str
    cluster_id: str
    true_grade: int
    predicted_grade: int
    probabilities: tuple[float, ...]


def _stable_seed(seed: int, *parts: Any) -> int:
    message = ":".join((str(seed), *(str(part) for part in parts)))
    return int.from_bytes(hashlib.sha256(message.encode()).digest()[:8], "little")


def load_statistics_protocol(path: str | Path) -> tuple[dict[str, Any], StatisticsConfig]:
    payload = read_json(path)
    verify_checksummed_payload(payload, schema=STATISTICS_PROTOCOL_SCHEMA)
    if payload.get("bootstrap_unit") != "patient_cluster_for_dr_image_for_aptos":
        raise ValueError("statistics bootstrap unit changed")
    if payload.get("grading_metrics") != list(GRADING_METRICS):
        raise ValueError("statistics grading metric family changed")
    config = payload.get("config")
    if not isinstance(config, Mapping):
        raise TypeError("statistics protocol config is missing")
    result = StatisticsConfig(
        bootstrap_samples=int(config["bootstrap_samples"]),
        bootstrap_seed=int(config["bootstrap_seed"]),
        bootstrap_chunk_size=int(config["bootstrap_chunk_size"]),
        confidence_level=float(config["confidence_level"]),
        reliability_bins=int(config["reliability_bins"]),
        nll_probability_floor=float(config["nll_probability_floor"]),
    )
    if result.bootstrap_samples < 1000:
        raise ValueError("statistics bootstrap requires at least 1000 replicates")
    if result.bootstrap_chunk_size < 1:
        raise ValueError("bootstrap chunk size must be positive")
    if not 0.0 < result.confidence_level < 1.0:
        raise ValueError("confidence level must lie in (0,1)")
    if result.reliability_bins < 2:
        raise ValueError("at least two reliability bins are required")
    if not 0.0 < result.nll_probability_floor < 1e-3:
        raise ValueError("NLL floor must be positive and smaller than 1e-3")
    return payload, result


def _artifact_path(manifest_path: Path, artifact: Mapping[str, Any]) -> Path:
    path = Path(str(artifact.get("path", "")))
    if not path.is_file():
        raise FileNotFoundError(path)
    try:
        path.resolve().relative_to(manifest_path.parent.resolve())
    except ValueError:
        raise AssertionError("aggregate manifest points outside its artifact directory") from None
    if file_sha256(path) != artifact.get("sha256"):
        raise AssertionError(f"aggregate artifact checksum mismatch: {path}")
    if artifact.get("row_schema") not in (None, ROW_SCHEMA):
        raise AssertionError("aggregate artifact row schema changed")
    return path


def _validate_probabilities(probabilities: np.ndarray) -> None:
    if probabilities.ndim != 2 or probabilities.shape[1] < 2:
        raise ValueError("probabilities must have shape (N,K), K >= 2")
    if not np.all(np.isfinite(probabilities)) or np.any(probabilities < -1e-12):
        raise ValueError("probabilities must be finite and nonnegative")
    if not np.allclose(probabilities.sum(1), 1.0, rtol=1e-9, atol=2e-10):
        raise ValueError("probability rows do not sum to one")


def _confusion_metrics(confusion: np.ndarray) -> dict[str, float]:
    confusion = np.asarray(confusion, dtype=np.float64)
    if confusion.ndim == 2:
        confusion = confusion[None]
    total = confusion.sum(axis=(1, 2))
    if np.any(total <= 0):
        raise ValueError("confusion matrices must contain observations")
    support = confusion.sum(2)
    predicted_support = confusion.sum(1)
    diagonal = np.diagonal(confusion, axis1=1, axis2=2)
    classes = confusion.shape[1]
    coordinate = np.arange(classes, dtype=np.float64)
    absolute = np.abs(coordinate[:, None] - coordinate[None])
    quadratic = (coordinate[:, None] - coordinate[None]) ** 2
    quadratic /= float((classes - 1) ** 2)
    accuracy = 100.0 * diagonal.sum(1) / total
    mae = np.einsum("bij,ij->b", confusion, absolute) / total
    observed = np.einsum("bij,ij->b", confusion, quadratic)
    expected = np.einsum(
        "bi,bj,ij->b", support, predicted_support, quadratic, optimize=True
    ) / total
    observed_over_expected = np.divide(
        observed, expected, out=np.zeros_like(observed), where=expected > 0.0
    )
    qwk = np.where(expected > 0.0, 1.0 - observed_over_expected, 0.0)
    recall = np.divide(diagonal, support, out=np.zeros_like(diagonal), where=support > 0)
    precision = np.divide(
        diagonal, predicted_support, out=np.zeros_like(diagonal), where=predicted_support > 0
    )
    present = support > 0
    balanced = 100.0 * np.sum(recall * present, axis=1) / np.maximum(1, present.sum(1))
    f1 = np.divide(
        2 * recall * precision,
        recall + precision,
        out=np.zeros_like(recall),
        where=(recall + precision) > 0,
    )
    macro_f1 = np.sum(f1 * present, axis=1) / np.maximum(1, present.sum(1))
    return {
        "accuracy_percent": accuracy,
        "mae": mae,
        "qwk": qwk,
        "balanced_accuracy_percent": balanced,
        "macro_f1": macro_f1,
    }


def grading_metrics(
    probabilities: np.ndarray,
    labels: np.ndarray,
    predicted: np.ndarray,
    *,
    nll_probability_floor: float = 1e-15,
) -> dict[str, Any]:
    probabilities = np.asarray(probabilities, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    predicted = np.asarray(predicted, dtype=np.int64)
    _validate_probabilities(probabilities)
    n, classes = probabilities.shape
    if labels.shape != (n,) or predicted.shape != (n,):
        raise ValueError("labels and predictions must align with probabilities")
    if np.any(labels < 0) or np.any(labels >= classes):
        raise ValueError("labels are outside the class range")
    if not np.array_equal(predicted, np.argmax(probabilities, axis=1)):
        raise ValueError("stored decisions are not class-MAP decisions")
    confusion = np.zeros((classes, classes), dtype=np.float64)
    np.add.at(confusion, (labels, predicted), 1.0)
    result = {key: float(value[0]) for key, value in _confusion_metrics(confusion).items()}
    one_hot = np.eye(classes, dtype=np.float64)[labels]
    expected = probabilities @ np.arange(classes, dtype=np.float64)
    tails = np.flip(np.cumsum(np.flip(probabilities, axis=1), axis=1), axis=1)[:, 1:]
    targets = labels[:, None] > np.arange(classes - 1)[None]
    result.update(
        {
            "n": n,
            "expected_grade_mae": float(np.mean(np.abs(expected - labels))),
            "nll": float(
                -np.mean(np.log(np.maximum(probabilities[np.arange(n), labels], nll_probability_floor)))
            ),
            "rps": float(np.mean(np.mean((tails - targets) ** 2, axis=1))),
            "multiclass_brier": float(np.mean(np.sum((probabilities - one_hot) ** 2, axis=1))),
            "confusion": confusion.astype(np.int64).tolist(),
        }
    )
    return result


def threshold_reliability(
    probabilities: np.ndarray,
    labels: np.ndarray,
    *,
    bins: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    probabilities = np.asarray(probabilities, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    _validate_probabilities(probabilities)
    n, classes = probabilities.shape
    if labels.shape != (n,):
        raise ValueError("labels do not align with probabilities")
    tails = np.flip(np.cumsum(np.flip(probabilities, axis=1), axis=1), axis=1)[:, 1:]
    edges = np.linspace(0.0, 1.0, bins + 1)
    reliability: list[dict[str, Any]] = []
    metrics: list[dict[str, Any]] = []
    for boundary in range(classes - 1):
        probability = tails[:, boundary]
        target = (labels > boundary).astype(np.float64)
        assignments = np.searchsorted(edges, probability, side="left") - 1
        assignments = np.clip(assignments, 0, bins - 1)
        ece = 0.0
        for index in range(bins):
            members = assignments == index
            count = int(np.sum(members))
            mean_probability = None if count == 0 else float(np.mean(probability[members]))
            observed_frequency = None if count == 0 else float(np.mean(target[members]))
            gap = (
                None if count == 0
                else abs(float(mean_probability) - float(observed_frequency))
            )
            if gap is not None:
                ece += count / n * gap
            reliability.append(
                {
                    "boundary": boundary,
                    "bin_index": index,
                    "lower": float(edges[index]),
                    "upper": float(edges[index + 1]),
                    "interval": "closed_closed" if index == 0 else "open_closed",
                    "n": count,
                    "mean_probability": mean_probability,
                    "observed_frequency": observed_frequency,
                    "absolute_gap": gap,
                }
            )
        metrics.append(
            {
                "boundary": boundary,
                "n": n,
                "positive_rate": float(np.mean(target)),
                "mean_probability": float(np.mean(probability)),
                "binary_brier": float(np.mean((probability - target) ** 2)),
                "ece": float(ece),
            }
        )
    return metrics, reliability


def _cluster_grade_statistics(
    rows: Sequence[GradeRow], *, bins: int, nll_floor: float
) -> tuple[list[str], dict[str, np.ndarray]]:
    clusters = list(dict.fromkeys(f"{row.dataset}:{row.cluster_id}" for row in rows))
    lookup = {cluster: index for index, cluster in enumerate(clusters)}
    classes = len(rows[0].probabilities)
    shape = (len(clusters),)
    result = {
        "confusion": np.zeros(shape + (classes, classes), dtype=np.float64),
        "count": np.zeros(shape, dtype=np.float64),
        "nll": np.zeros(shape, dtype=np.float64),
        "rps": np.zeros(shape, dtype=np.float64),
        "brier": np.zeros(shape, dtype=np.float64),
        "expected_error": np.zeros(shape, dtype=np.float64),
        "bin_total": np.zeros(shape + (classes - 1, bins), dtype=np.float64),
        "bin_positive": np.zeros(shape + (classes - 1, bins), dtype=np.float64),
        "bin_probability": np.zeros(shape + (classes - 1, bins), dtype=np.float64),
    }
    edges = np.linspace(0.0, 1.0, bins + 1)
    for row in rows:
        index = lookup[f"{row.dataset}:{row.cluster_id}"]
        p = np.asarray(row.probabilities, dtype=np.float64)
        label, prediction = row.true_grade, row.predicted_grade
        tails = np.flip(np.cumsum(np.flip(p)))[1:]
        targets = label > np.arange(classes - 1)
        one_hot = np.eye(classes)[label]
        result["confusion"][index, label, prediction] += 1
        result["count"][index] += 1
        result["nll"][index] += -math.log(max(float(p[label]), nll_floor))
        result["rps"][index] += float(np.mean((tails - targets) ** 2))
        result["brier"][index] += float(np.sum((p - one_hot) ** 2))
        result["expected_error"][index] += abs(float(p @ np.arange(classes)) - label)
        assignments = np.clip(np.searchsorted(edges, tails, side="left") - 1, 0, bins - 1)
        for boundary, bin_index in enumerate(assignments):
            result["bin_total"][index, boundary, bin_index] += 1
            result["bin_positive"][index, boundary, bin_index] += float(targets[boundary])
            result["bin_probability"][index, boundary, bin_index] += float(tails[boundary])
    return clusters, result


def _grade_metrics_from_weights(weights: np.ndarray, stats: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    confusion = np.einsum("bc,cij->bij", weights, stats["confusion"], optimize=True)
    result = _confusion_metrics(confusion)
    count = weights @ stats["count"]
    for output, source in (
        ("expected_grade_mae", "expected_error"),
        ("nll", "nll"),
        ("rps", "rps"),
        ("multiclass_brier", "brier"),
    ):
        result[output] = (weights @ stats[source]) / count
    bin_total = np.einsum("bc,ckl->bkl", weights, stats["bin_total"], optimize=True)
    bin_positive = np.einsum("bc,ckl->bkl", weights, stats["bin_positive"], optimize=True)
    bin_probability = np.einsum("bc,ckl->bkl", weights, stats["bin_probability"], optimize=True)
    gaps = np.divide(
        np.abs(bin_positive - bin_probability),
        bin_total,
        out=np.zeros_like(bin_total),
        where=bin_total > 0,
    )
    boundary_ece = np.sum(bin_total / count[:, None, None] * gaps, axis=2)
    for boundary in range(boundary_ece.shape[1]):
        result[f"threshold_ece_y_gt_{boundary}"] = boundary_ece[:, boundary]
    return result


def _percentile_interval(values: np.ndarray, confidence: float) -> list[float]:
    tail = (1.0 - confidence) / 2.0
    return [float(value) for value in np.quantile(values, [tail, 1.0 - tail])]


def cluster_bootstrap_grading(
    rows: Sequence[GradeRow],
    *,
    config: StatisticsConfig,
    seed_label: str,
) -> dict[str, Any]:
    if not rows:
        raise ValueError("grading bootstrap has no rows")
    probabilities = np.asarray([row.probabilities for row in rows], dtype=np.float64)
    labels = np.asarray([row.true_grade for row in rows], dtype=np.int64)
    predicted = np.asarray([row.predicted_grade for row in rows], dtype=np.int64)
    point = grading_metrics(
        probabilities, labels, predicted,
        nll_probability_floor=config.nll_probability_floor,
    )
    threshold_metrics, _ = threshold_reliability(
        probabilities, labels, bins=config.reliability_bins
    )
    for record in threshold_metrics:
        point[f"threshold_ece_y_gt_{record['boundary']}"] = record["ece"]
    clusters, statistics = _cluster_grade_statistics(
        rows, bins=config.reliability_bins, nll_floor=config.nll_probability_floor
    )
    metric_names = [
        *GRADING_METRICS,
        *(f"threshold_ece_y_gt_{boundary}" for boundary in range(probabilities.shape[1] - 1)),
    ]
    draws: dict[str, list[np.ndarray]] = {name: [] for name in metric_names}
    rng = np.random.default_rng(_stable_seed(config.bootstrap_seed, seed_label))
    probability = np.full(len(clusters), 1.0 / len(clusters))
    completed = 0
    while completed < config.bootstrap_samples:
        batch = min(config.bootstrap_chunk_size, config.bootstrap_samples - completed)
        weights = rng.multinomial(len(clusters), probability, size=batch).astype(np.float64)
        metrics = _grade_metrics_from_weights(weights, statistics)
        for name in metric_names:
            draws[name].append(np.asarray(metrics[name], dtype=np.float64))
        completed += batch
    records = {}
    for name in metric_names:
        values = np.concatenate(draws[name])
        records[name] = {
            "estimate": float(point[name]),
            "bootstrap_mean": float(np.mean(values)),
            "ci_percentile": _percentile_interval(values, config.confidence_level),
        }
    return {
        "n_images": len(rows),
        "n_clusters": len(clusters),
        "bootstrap_samples": config.bootstrap_samples,
        "bootstrap_seed_derived": _stable_seed(config.bootstrap_seed, seed_label),
        "metrics": records,
    }


def _intervention_auc_fields(rows: Sequence[Mapping[str, Any]]) -> tuple[str, ...]:
    fields = sorted(
        {
            key
            for row in rows
            for key in row
            if key.startswith("auc_") and not key.startswith("auc_ranked_")
        }
    )
    if not fields:
        raise ValueError("intervention summaries contain no AUC fields")
    return tuple(fields)


def paired_intervention_bootstrap(
    rows: Sequence[Mapping[str, Any]],
    *,
    config: StatisticsConfig,
    seed_label: str,
) -> list[dict[str, Any]]:
    if not rows:
        return []
    by_key: dict[tuple[str, str, int], dict[str, Mapping[str, Any]]] = defaultdict(dict)
    for row in rows:
        key = (str(row["dataset"]), str(row["image_key"]), int(row["boundary"]))
        method = str(row["method"])
        if method in by_key[key]:
            raise AssertionError("duplicate intervention summary method")
        by_key[key][method] = row
    if any(set(methods) != set(METHODS) for methods in by_key.values()):
        raise AssertionError("intervention summaries are not fully paired by method")
    auc_fields = _intervention_auc_fields(rows)
    cluster_order = list(
        dict.fromkeys(
            f"{dataset}:{methods['ranked_native']['cluster_id']}"
            for (dataset, _, _), methods in by_key.items()
        )
    )
    cluster_lookup = {value: index for index, value in enumerate(cluster_order)}
    comparators = [method for method in METHODS if method != "ranked_native"]
    results: list[dict[str, Any]] = []
    probability = np.full(len(cluster_order), 1.0 / len(cluster_order))
    shape = (len(comparators), len(cluster_order), len(auc_fields))
    sums = np.zeros(shape, dtype=np.float64)
    counts = np.zeros_like(sums)
    ranked_sums = np.zeros_like(sums)
    comparator_sums = np.zeros_like(sums)
    for comparator_index, comparator in enumerate(comparators):
        for (dataset, _, _), methods in by_key.items():
            ranked, control = methods["ranked_native"], methods[comparator]
            cluster = f"{dataset}:{ranked['cluster_id']}"
            if cluster != f"{dataset}:{control['cluster_id']}":
                raise AssertionError("paired intervention rows changed cluster identity")
            cluster_index = cluster_lookup[cluster]
            for field_index, field in enumerate(auc_fields):
                left, right = ranked.get(field), control.get(field)
                if left is None or right is None:
                    continue
                left, right = float(left), float(right)
                if not math.isfinite(left) or not math.isfinite(right):
                    continue
                sums[comparator_index, cluster_index, field_index] += left - right
                ranked_sums[comparator_index, cluster_index, field_index] += left
                comparator_sums[comparator_index, cluster_index, field_index] += right
                counts[comparator_index, cluster_index, field_index] += 1
    # All comparators receive exactly the same cluster resamples.  This both
    # preserves cross-control pairing and avoids regenerating an expensive
    # patient bootstrap five times.
    bootstrap_seed = _stable_seed(config.bootstrap_seed, seed_label)
    rng = np.random.default_rng(bootstrap_seed)
    draws: list[list[list[np.ndarray]]] = [
        [[] for _ in auc_fields] for _ in comparators
    ]
    completed = 0
    while completed < config.bootstrap_samples:
        batch = min(config.bootstrap_chunk_size, config.bootstrap_samples - completed)
        weights = rng.multinomial(
            len(cluster_order), probability, size=batch
        ).astype(np.float64)
        numerator = np.einsum("bc,pcm->bpm", weights, sums, optimize=True)
        denominator = np.einsum("bc,pcm->bpm", weights, counts, optimize=True)
        delta = np.divide(
            numerator,
            denominator,
            out=np.full_like(numerator, np.nan),
            where=denominator > 0,
        )
        for comparator_index in range(len(comparators)):
            for field_index in range(len(auc_fields)):
                draws[comparator_index][field_index].append(
                    delta[:, comparator_index, field_index]
                )
        completed += batch
    for comparator_index, comparator in enumerate(comparators):
        for field_index, field in enumerate(auc_fields):
            denominator = float(counts[comparator_index, :, field_index].sum())
            if denominator <= 0:
                continue
            values = np.concatenate(draws[comparator_index][field_index])
            values = values[np.isfinite(values)]
            if len(values) != config.bootstrap_samples:
                raise AssertionError("a bootstrap draw omitted every paired observation")
            estimate = float(sums[comparator_index, :, field_index].sum() / denominator)
            results.append(
                {
                    "comparator": comparator,
                    "metric": field,
                    "n_paired_image_boundaries": int(denominator),
                    "n_clusters": len(cluster_order),
                    "ranked_native_mean": float(
                        ranked_sums[comparator_index, :, field_index].sum() / denominator
                    ),
                    "comparator_mean": float(
                        comparator_sums[comparator_index, :, field_index].sum() / denominator
                    ),
                    "ranked_minus_comparator": estimate,
                    "bootstrap_mean_difference": float(np.mean(values)),
                    "ci_percentile": _percentile_interval(values, config.confidence_level),
                    "bootstrap_fraction_ranked_better": float(
                        (np.sum(values > 0) + 0.5 * np.sum(values == 0)) / len(values)
                    ),
                    "bootstrap_seed_derived": bootstrap_seed,
                    "direction": "positive_favors_ranked_native",
                }
            )
    return results


def _read_aggregate(
    dataset: str, manifest_path: Path
) -> tuple[dict[str, Any], list[GradeRow], list[dict[str, Any]]]:
    manifest = read_json(manifest_path)
    verify_checksummed_payload(manifest, schema=AGGREGATE_SCHEMA)
    if manifest.get("dataset") != dataset:
        raise AssertionError("declared dataset differs from aggregate manifest")
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise TypeError("aggregate manifest has no artifact map")
    image_path = _artifact_path(manifest_path, artifacts["image_rows"])
    summary_path = _artifact_path(manifest_path, artifacts["summary_rows"])
    image_state: dict[str, GradeRow] = {}
    boundaries: dict[str, set[int]] = defaultdict(set)
    image_rows = 0
    for row in iter_jsonl_gzip(image_path):
        image_rows += 1
        if row.get("schema") != ROW_SCHEMA or row.get("row_type") != "image_boundary_census":
            raise AssertionError("unexpected image census row")
        if row.get("dataset") != dataset:
            raise AssertionError("image census dataset changed")
        key = str(row["image_key"])
        grade = GradeRow(
            dataset=dataset,
            image_key=key,
            cluster_id=str(row.get("cluster_id") or key),
            true_grade=int(row["true_grade"]),
            predicted_grade=int(row["predicted_grade"]),
            probabilities=tuple(float(value) for value in row["full_class_probabilities"]),
        )
        if key in image_state and image_state[key] != grade:
            raise AssertionError("boundary census rows disagree on image posterior")
        image_state[key] = grade
        boundary = int(row["boundary"])
        if boundary in boundaries[key]:
            raise AssertionError("duplicate image-boundary census row")
        boundaries[key].add(boundary)
    if image_rows != int(artifacts["image_rows"]["rows"]):
        raise AssertionError("image artifact row count changed")
    if len(image_state) != int(manifest["n_images"]):
        raise AssertionError("aggregate image coverage changed")
    for row in image_state.values():
        expected = set(range(len(row.probabilities) - 1))
        if boundaries[row.image_key] != expected:
            raise AssertionError("image is missing a boundary census row")
    summaries: list[dict[str, Any]] = []
    for row in iter_jsonl_gzip(summary_path):
        if row.get("schema") != ROW_SCHEMA or row.get("row_type") != "image_boundary_curve_summary":
            raise AssertionError("unexpected intervention summary row")
        if row.get("dataset") != dataset:
            raise AssertionError("intervention summary dataset changed")
        image_key = str(row["image_key"])
        if image_key not in image_state:
            raise AssertionError("intervention summary references an unknown image")
        if str(row.get("cluster_id") or image_key) != image_state[image_key].cluster_id:
            raise AssertionError("intervention summary changed image cluster identity")
        summaries.append(row)
    if len(summaries) != int(artifacts["summary_rows"]["rows"]):
        raise AssertionError("summary artifact row count changed")
    return manifest, list(image_state.values()), summaries


def _scope_records(
    grade_rows_by_dataset: Mapping[str, Sequence[GradeRow]],
    summaries_by_dataset: Mapping[str, Sequence[Mapping[str, Any]]],
) -> Iterable[tuple[str, str, list[GradeRow], list[Mapping[str, Any]]]]:
    all_grade = [row for dataset in sorted(grade_rows_by_dataset) for row in grade_rows_by_dataset[dataset]]
    all_summary = [row for dataset in sorted(summaries_by_dataset) for row in summaries_by_dataset[dataset]]
    yield "overall", "all", all_grade, all_summary
    for dataset in sorted(grade_rows_by_dataset):
        grade = list(grade_rows_by_dataset[dataset])
        summary = list(summaries_by_dataset[dataset])
        yield "dataset", dataset, grade, summary
        boundaries = sorted({int(row["boundary"]) for row in summary})
        for boundary in boundaries:
            yield (
                "dataset_boundary", f"{dataset}:Y>{boundary}", grade,
                [row for row in summary if int(row["boundary"]) == boundary],
            )


def analyze_oof_statistics(
    *,
    aggregate_manifests: Mapping[str, Path],
    protocol_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    if (output_dir / "statistics_manifest.json").exists():
        raise FileExistsError("refusing to overwrite completed OOF statistics")
    protocol, config = load_statistics_protocol(protocol_path)
    input_records = []
    grade_by_dataset: dict[str, list[GradeRow]] = {}
    summaries_by_dataset: dict[str, list[dict[str, Any]]] = {}
    for dataset, path in sorted(aggregate_manifests.items()):
        manifest, grade_rows, summaries = _read_aggregate(dataset, path)
        grade_by_dataset[dataset] = grade_rows
        summaries_by_dataset[dataset] = summaries
        input_records.append(
            {
                "dataset": dataset,
                "path": str(path.resolve()),
                "sha256": file_sha256(path),
                "content_checksum_sha256": manifest["content_checksum_sha256"],
                "n_images": manifest["n_images"],
            }
        )
    if not grade_by_dataset:
        raise ValueError("at least one aggregate manifest is required")
    output_dir.mkdir(parents=True, exist_ok=True)

    proper_records: list[dict[str, Any]] = []
    reliability_records: list[dict[str, Any]] = []
    grading_bootstrap_records: list[dict[str, Any]] = []
    intervention_records: list[dict[str, Any]] = []
    for scope_type, scope, grade_rows, summary_rows in _scope_records(
        grade_by_dataset, summaries_by_dataset
    ):
        if scope_type == "dataset_boundary":
            intervention_records.extend(
                {
                    "scope_type": scope_type, "scope": scope, **record
                }
                for record in paired_intervention_bootstrap(
                    summary_rows, config=config, seed_label=f"intervention:{scope}"
                )
            )
            continue
        probabilities = np.asarray([row.probabilities for row in grade_rows], dtype=np.float64)
        labels = np.asarray([row.true_grade for row in grade_rows], dtype=np.int64)
        predicted = np.asarray([row.predicted_grade for row in grade_rows], dtype=np.int64)
        metrics = grading_metrics(
            probabilities, labels, predicted,
            nll_probability_floor=config.nll_probability_floor,
        )
        threshold_metrics, reliability = threshold_reliability(
            probabilities, labels, bins=config.reliability_bins
        )
        proper_records.append(
            {
                "scope_type": scope_type, "scope": scope,
                "n_images": len(grade_rows), "metrics": metrics,
                "threshold_metrics": threshold_metrics,
            }
        )
        reliability_records.extend(
            {"scope_type": scope_type, "scope": scope, **record}
            for record in reliability
        )
        grading_bootstrap_records.append(
            {
                "scope_type": scope_type, "scope": scope,
                **cluster_bootstrap_grading(
                    grade_rows, config=config, seed_label=f"grading:{scope_type}:{scope}"
                ),
            }
        )
        intervention_records.extend(
            {"scope_type": scope_type, "scope": scope, **record}
            for record in paired_intervention_bootstrap(
                summary_rows, config=config, seed_label=f"intervention:{scope_type}:{scope}"
            )
        )

    common = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "statistics_protocol_checksum_sha256": protocol["content_checksum_sha256"],
        "inputs": input_records,
        "confidence_level": config.confidence_level,
    }
    payloads = {
        "proper_score_calibration": {
            "schema": PROPER_SCORE_SCHEMA,
            **common,
            "definitions": {
                "nll": "mean negative natural log target probability; probabilities floored only at protocol epsilon",
                "rps": "mean squared P(Y>k) error averaged over K-1 ordinal boundaries",
                "multiclass_brier": "mean unnormalized sum of K squared class-probability errors",
                "threshold_ece": "15-bin equal-width absolute calibration error for each P(Y>k)",
            },
            "records": proper_records,
            "reliability_records": reliability_records,
        },
        "grading_bootstrap": {
            "schema": GRADING_BOOTSTRAP_SCHEMA,
            **common,
            "method": "nonparametric_cluster_bootstrap_percentile",
            "bootstrap_unit": "EyePACS patient (both eyes together); APTOS image",
            "records": grading_bootstrap_records,
        },
        "intervention_bootstrap": {
            "schema": INTERVENTION_BOOTSTRAP_SCHEMA,
            **common,
            "method": "paired_nonparametric_cluster_bootstrap_percentile",
            "pairing_unit": "same image and ordinal boundary",
            "interpretation": "exact stored-ledger interventions; not causal pixel interventions",
            "multiplicity": "no adjustment; intervals and bootstrap fractions are descriptive",
            "records": intervention_records,
        },
    }
    filenames = {
        "proper_score_calibration": "OOF_PROPER_SCORE_CALIBRATION.json",
        "grading_bootstrap": "OOF_GRADING_CLUSTER_BOOTSTRAP.json",
        "intervention_bootstrap": "OOF_INTERVENTION_PAIRED_BOOTSTRAP.json",
    }
    artifacts: dict[str, Any] = {}
    for name, payload in payloads.items():
        payload["content_checksum_sha256"] = canonical_sha256(payload)
        path = output_dir / filenames[name]
        write_json_atomic(path, payload)
        artifacts[name] = {
            "path": str(path.resolve()),
            "sha256": file_sha256(path),
            "content_checksum_sha256": payload["content_checksum_sha256"],
            "records": len(payload["records"]),
        }
        if name == "proper_score_calibration":
            artifacts[name]["reliability_records"] = len(payload["reliability_records"])
    manifest: dict[str, Any] = {
        "schema": STATISTICS_MANIFEST_SCHEMA,
        **common,
        "analysis_implementation_sha256": file_sha256(Path(__file__).resolve()),
        "artifacts": artifacts,
        "scope": "checksum_sealed_complete_oof_posteriors_and_paired_stored_ledger_interventions",
    }
    manifest["content_checksum_sha256"] = canonical_sha256(manifest)
    write_json_atomic(output_dir / "statistics_manifest.json", manifest)
    return manifest


def _parse_manifest(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("aggregate manifest must be DATASET=PATH")
    dataset, path = value.split("=", 1)
    if not dataset or not path:
        raise argparse.ArgumentTypeError("aggregate manifest must be DATASET=PATH")
    return dataset, Path(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--aggregate-manifest", action="append", type=_parse_manifest, required=True,
        metavar="DATASET=PATH",
    )
    parser.add_argument(
        "--protocol", type=Path,
        default=REPO_ROOT / "scripts" / "protocols" / "origin_oof_statistics_protocol.json",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    manifests = dict(args.aggregate_manifest)
    if len(manifests) != len(args.aggregate_manifest):
        raise ValueError("duplicate dataset aggregate manifest")
    result = analyze_oof_statistics(
        aggregate_manifests=manifests,
        protocol_path=args.protocol,
        output_dir=args.output_dir,
    )
    print(
        "ORIGIN OOF statistics complete: "
        f"datasets={','.join(sorted(manifests))} "
        f"manifest={args.output_dir / 'statistics_manifest.json'} "
        f"checksum={result['content_checksum_sha256']}"
    )


if __name__ == "__main__":
    main()
