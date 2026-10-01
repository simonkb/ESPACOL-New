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
import subprocess
import sys
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.audit_origin_oof_interventions import iter_jsonl_gzip
from scripts.origin_oof_intervention_common import AGGREGATE_SCHEMA, METHODS, ROW_SCHEMA
from scripts.origin_v3_cv_common import (
    BASE_COMMIT,
    canonical_sha256,
    file_sha256,
    read_json,
    verify_checksummed_payload,
    write_json_atomic,
)


STATISTICS_PROTOCOL_SCHEMA = "origin-oof-statistics-protocol-v2"
PROPER_SCORE_SCHEMA = "origin-oof-proper-score-calibration-v1"
GRADING_BOOTSTRAP_SCHEMA = "origin-oof-grading-cluster-bootstrap-v1"
INTERVENTION_BOOTSTRAP_SCHEMA = "origin-oof-intervention-paired-bootstrap-v2"
GATE_A_SCHEMA = "origin-oof-gate-a-adjudication-v1"
RF_DIAGNOSTIC_SCHEMA = "origin-oof-rf-footprint-diagnostics-v1"
STATISTICS_MANIFEST_SCHEMA = "origin-oof-statistics-manifest-v2"

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
class GateAConfig:
    required_datasets: tuple[str, ...] = ("dr", "aptos")
    matched_controls: tuple[str, ...] = (
        "random_scale_count_stride_area",
        "random_scale_footprint",
    )
    deletion_auc_metric: str = "auc_deletion_tail_normalized_by_nominal_stride_area"
    retention_auc_metric: str = "auc_retention_tail_normalized_by_nominal_stride_area"
    minimum_auc_advantage: float = 0.10
    minimum_stratum_clusters: int = 30
    dr_positive_grades: tuple[int, ...] = (1, 2, 3, 4)
    dr_minimum_passing_grades: int = 3
    aptos_positive_grades: tuple[int, ...] = (1, 2, 4)
    aptos_minimum_passing_grades: int = 2
    target_effect: float = 0.50
    target_area_fraction: float = 0.10
    maximum_ranked_area_to_target: float = 0.10
    maximum_ranked_to_random_area_ratio: float = 0.50
    minimum_retention_preservation_advantage: float = 0.05
    minimum_metric_coverage: float = 0.90


DEFAULT_GATE_A_CONFIG = GateAConfig()


@dataclass(frozen=True)
class StatisticsConfig:
    bootstrap_samples: int
    bootstrap_seed: int
    bootstrap_chunk_size: int
    confidence_level: float
    reliability_bins: int
    nll_probability_floor: float
    gate_a: GateAConfig = DEFAULT_GATE_A_CONFIG


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


def _runtime_source_record() -> dict[str, Any]:
    """Record the exact statistics source without weakening checksum checks.

    A missing Git executable is represented explicitly.  Production Gate A is
    fail-closed unless a clean immutable commit is available; unit tests and
    exploratory callers can still obtain descriptive artifacts.
    """

    def command(*arguments: str) -> str | None:
        try:
            completed = subprocess.run(
                ("git", "-C", str(REPO_ROOT), *arguments),
                check=True, capture_output=True, text=True,
            )
        except (FileNotFoundError, subprocess.CalledProcessError):
            return None
        return completed.stdout.strip()

    commit = command("rev-parse", "HEAD")
    tracked_status = command("status", "--porcelain", "--untracked-files=no")
    return {
        "git_commit": commit,
        "git_commit_available": commit is not None and len(commit) == 40,
        "tracked_worktree_clean": tracked_status == "" if tracked_status is not None else False,
        "analysis_implementation_path": str(Path(__file__).resolve()),
        "analysis_implementation_sha256": file_sha256(Path(__file__).resolve()),
    }


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
    gate_values = payload.get("gate_a")
    if not isinstance(gate_values, Mapping):
        raise TypeError("statistics protocol Gate A contract is missing")
    grade_values = gate_values.get("positive_grade_strata")
    if not isinstance(grade_values, Mapping):
        raise TypeError("statistics protocol positive-grade contract is missing")
    area_values = gate_values.get("concentration")
    preservation_values = gate_values.get("retention_preservation")
    if not isinstance(area_values, Mapping) or not isinstance(preservation_values, Mapping):
        raise TypeError("statistics protocol concentration contract is missing")
    gate = GateAConfig(
        required_datasets=tuple(str(value) for value in gate_values["required_datasets"]),
        matched_controls=tuple(str(value) for value in gate_values["matched_controls"]),
        deletion_auc_metric=str(gate_values["deletion_auc_metric"]),
        retention_auc_metric=str(gate_values["retention_auc_metric"]),
        minimum_auc_advantage=float(gate_values["minimum_auc_advantage"]),
        minimum_stratum_clusters=int(gate_values["minimum_stratum_clusters"]),
        dr_positive_grades=tuple(int(value) for value in grade_values["dr"]["grades"]),
        dr_minimum_passing_grades=int(grade_values["dr"]["minimum_passing"]),
        aptos_positive_grades=tuple(
            int(value) for value in grade_values["aptos"]["grades"]
        ),
        aptos_minimum_passing_grades=int(
            grade_values["aptos"]["minimum_passing"]
        ),
        target_effect=float(area_values["target_local_attributable_effect"]),
        target_area_fraction=float(area_values["top_area_fraction"]),
        maximum_ranked_area_to_target=float(area_values["maximum_ranked_area_to_target"]),
        maximum_ranked_to_random_area_ratio=float(
            area_values["maximum_ranked_to_random_area_ratio"]
        ),
        minimum_retention_preservation_advantage=float(
            preservation_values["minimum_absolute_advantage"]
        ),
        minimum_metric_coverage=float(gate_values["minimum_metric_coverage"]),
    )
    result = StatisticsConfig(
        bootstrap_samples=int(config["bootstrap_samples"]),
        bootstrap_seed=int(config["bootstrap_seed"]),
        bootstrap_chunk_size=int(config["bootstrap_chunk_size"]),
        confidence_level=float(config["confidence_level"]),
        reliability_bins=int(config["reliability_bins"]),
        nll_probability_floor=float(config["nll_probability_floor"]),
        gate_a=gate,
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
    if set(gate.required_datasets) != {"dr", "aptos"}:
        raise ValueError("Gate A requires the complete EyePACS and APTOS OOF audits")
    if not gate.matched_controls or any(value not in METHODS for value in gate.matched_controls):
        raise ValueError("Gate A matched controls are invalid")
    if not 0.0 < gate.minimum_auc_advantage < 1.0:
        raise ValueError("Gate A AUC threshold must lie in (0,1)")
    if gate.minimum_stratum_clusters < 2:
        raise ValueError("Gate A strata require at least two clusters")
    if not 0.0 < gate.target_effect < 1.0 or not 0.0 < gate.target_area_fraction < 1.0:
        raise ValueError("Gate A concentration thresholds must lie in (0,1)")
    if not 0.0 < gate.maximum_ranked_to_random_area_ratio < 1.0:
        raise ValueError("Gate A area ratio must lie in (0,1)")
    if not 0.0 < gate.minimum_metric_coverage <= 1.0:
        raise ValueError("Gate A metric coverage must lie in (0,1]")
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
    auc_fields: Sequence[str] | None = None,
    comparators: Sequence[str] | None = None,
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
    auc_fields = tuple(auc_fields or _intervention_auc_fields(rows))
    if not auc_fields:
        raise ValueError("paired intervention bootstrap has no requested AUC fields")
    cluster_order = list(
        dict.fromkeys(
            f"{dataset}:{methods['ranked_native']['cluster_id']}"
            for (dataset, _, _), methods in by_key.items()
        )
    )
    cluster_lookup = {value: index for index, value in enumerate(cluster_order)}
    comparators = list(
        comparators or (method for method in METHODS if method != "ranked_native")
    )
    if not comparators or any(
        method == "ranked_native" or method not in METHODS for method in comparators
    ):
        raise ValueError("paired intervention comparator set is invalid")
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


def _boundary_outcome(true_grade: int, predicted_grade: int, boundary: int) -> str:
    true_positive = true_grade > boundary
    predicted_positive = predicted_grade > boundary
    if true_positive and predicted_positive:
        return "true_positive"
    if true_positive:
        return "false_negative"
    if predicted_positive:
        return "false_positive"
    return "true_negative"


_CURVE_POINT_FIELDS = (
    "requested_cell_fraction",
    "selected_cell_fraction",
    "selected_nominal_stride_area_fraction",
    "selected_target_boundary_mass_fraction",
    "deletion_tail_normalized",
    "retention_tail_normalized",
    "retention_map_preserved",
    "maximum_single_footprint_area_input_fraction",
    "summed_clipped_footprint_area_input_fraction",
    "clipped_union_footprint_area_input_fraction",
    "selected_count",
)


def _compact_curve_profiles(
    *,
    dataset: str,
    path: Path,
    expected_rows: int,
    image_state: Mapping[str, GradeRow],
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int, str], dict[str, Any]] = {}
    observed_rows = 0
    for row in iter_jsonl_gzip(path):
        observed_rows += 1
        if row.get("schema") != ROW_SCHEMA or row.get("row_type") != "image_boundary_curve_point":
            raise AssertionError("unexpected intervention curve row")
        if row.get("dataset") != dataset or row.get("method") not in METHODS:
            raise AssertionError("intervention curve identity changed")
        image_key = str(row["image_key"])
        if image_key not in image_state:
            raise AssertionError("intervention curve references an unknown image")
        grade = image_state[image_key]
        if str(row.get("cluster_id") or image_key) != grade.cluster_id:
            raise AssertionError("intervention curve changed image cluster identity")
        boundary = int(row["boundary"])
        method = str(row["method"])
        key = (image_key, boundary, method)
        profile = grouped.setdefault(
            key,
            {
                "dataset": dataset,
                "image_key": image_key,
                "cluster_id": grade.cluster_id,
                "true_grade": grade.true_grade,
                "predicted_grade": grade.predicted_grade,
                "correct": grade.true_grade == grade.predicted_grade,
                "boundary": boundary,
                "boundary_outcome": _boundary_outcome(
                    grade.true_grade, grade.predicted_grade, boundary
                ),
                "method": method,
                "points": [],
            },
        )
        point = {field: row.get(field) for field in _CURVE_POINT_FIELDS}
        point["budget_index"] = int(row["budget_index"])
        for field, value in row.items():
            if (
                field.startswith("selected_count_")
                and not field.endswith(("_sd", "_q025", "_q975"))
            ):
                point[field] = value
        profile["points"].append(point)
    if observed_rows != expected_rows:
        raise AssertionError("curve artifact row count changed")
    by_image_boundary: dict[tuple[str, int], set[str]] = defaultdict(set)
    result: list[dict[str, Any]] = []
    common_budget_signature: tuple[float, ...] | None = None
    for profile in grouped.values():
        profile["points"].sort(key=lambda value: int(value["budget_index"]))
        indices = [int(value["budget_index"]) for value in profile["points"]]
        if indices != list(range(len(indices))):
            raise AssertionError("curve profile has missing or duplicate budget indices")
        signature = tuple(float(value["requested_cell_fraction"]) for value in profile["points"])
        if common_budget_signature is None:
            common_budget_signature = signature
        elif signature != common_budget_signature:
            raise AssertionError("curve profiles changed the frozen budget grid")
        by_image_boundary[(profile["image_key"], int(profile["boundary"]))].add(
            str(profile["method"])
        )
        result.append(profile)
    if any(methods != set(METHODS) for methods in by_image_boundary.values()):
        raise AssertionError("curve profiles are not paired across every intervention method")
    return result


def _finite_xy(
    points: Sequence[Mapping[str, Any]], x_field: str, y_field: str
) -> tuple[np.ndarray, np.ndarray]:
    pairs = []
    for point in points:
        x, y = point.get(x_field), point.get(y_field)
        if x is None or y is None:
            continue
        x, y = float(x), float(y)
        if math.isfinite(x) and math.isfinite(y):
            pairs.append((x, y))
    if not pairs:
        return np.empty(0), np.empty(0)
    by_x: dict[float, list[float]] = defaultdict(list)
    for x, y in pairs:
        by_x[x].append(y)
    xs = np.asarray(sorted(by_x), dtype=np.float64)
    ys = np.asarray([np.mean(by_x[x]) for x in xs], dtype=np.float64)
    return xs, ys


def _interpolate_curve(
    points: Sequence[Mapping[str, Any]], x_field: str, y_field: str, target: float
) -> float | None:
    xs, ys = _finite_xy(points, x_field, y_field)
    if len(xs) == 0 or target < xs[0] - 1e-12 or target > xs[-1] + 1e-12:
        return None
    return float(np.interp(float(target), xs, ys))


def _area_to_effect(
    points: Sequence[Mapping[str, Any]], y_field: str, target: float
) -> tuple[float | None, float | None]:
    xs, ys = _finite_xy(points, "selected_nominal_stride_area_fraction", y_field)
    if len(xs) < 2:
        return None, None
    minimum_step = float(np.min(np.diff(ys))) if len(ys) > 1 else 0.0
    monotone = np.maximum.accumulate(ys)
    hits = np.flatnonzero(monotone >= target - 1e-12)
    if not len(hits):
        return None, minimum_step
    index = int(hits[0])
    if index == 0:
        return float(xs[0]), minimum_step
    x0, x1 = float(xs[index - 1]), float(xs[index])
    y0, y1 = float(monotone[index - 1]), float(monotone[index])
    if y1 <= y0 + 1e-15:
        return x1, minimum_step
    fraction = min(1.0, max(0.0, (target - y0) / (y1 - y0)))
    return float(x0 + fraction * (x1 - x0)), minimum_step


def derive_curve_endpoints(
    profiles: Sequence[Mapping[str, Any]],
    census: Mapping[tuple[str, int], Mapping[str, Any]],
    *,
    gate: GateAConfig,
) -> list[dict[str, Any]]:
    """Reduce full curves to the predeclared concentration/sufficiency endpoints."""

    result: list[dict[str, Any]] = []
    for profile in profiles:
        points = profile["points"]
        delete_area, delete_min_step = _area_to_effect(
            points, "deletion_tail_normalized", gate.target_effect
        )
        retain_area, retain_min_step = _area_to_effect(
            points, "retention_tail_normalized", gate.target_effect
        )
        record = {
            key: profile[key]
            for key in (
                "dataset", "image_key", "cluster_id", "true_grade",
                "predicted_grade", "correct", "boundary", "boundary_outcome", "method",
            )
        }
        record.update(
            {
                "area_to_50pct_deletion_effect": delete_area,
                "area_to_50pct_retention_effect": retain_area,
                "deletion_effect_at_10pct_area": _interpolate_curve(
                    points, "selected_nominal_stride_area_fraction",
                    "deletion_tail_normalized", gate.target_area_fraction,
                ),
                "retention_effect_at_10pct_area": _interpolate_curve(
                    points, "selected_nominal_stride_area_fraction",
                    "retention_tail_normalized", gate.target_area_fraction,
                ),
                "retention_map_preserved_at_10pct_area": _interpolate_curve(
                    points, "selected_nominal_stride_area_fraction",
                    "retention_map_preserved", gate.target_area_fraction,
                ),
                "deletion_curve_minimum_step": delete_min_step,
                "retention_curve_minimum_step": retain_min_step,
                "maximum_single_rf_footprint_at_10pct_area": _interpolate_curve(
                    points, "selected_nominal_stride_area_fraction",
                    "maximum_single_footprint_area_input_fraction", gate.target_area_fraction,
                ),
                "summed_rf_footprint_at_10pct_area": _interpolate_curve(
                    points, "selected_nominal_stride_area_fraction",
                    "summed_clipped_footprint_area_input_fraction", gate.target_area_fraction,
                ),
                "clipped_union_rf_footprint_at_10pct_area": _interpolate_curve(
                    points, "selected_nominal_stride_area_fraction",
                    "clipped_union_footprint_area_input_fraction", gate.target_area_fraction,
                ),
            }
        )
        census_row = census.get((str(profile["image_key"]), int(profile["boundary"])), {})
        geometry = census_row.get("scale_geometry", {})
        global_scales = {
            str(name) for name, value in geometry.items()
            if isinstance(value, Mapping) and value.get("theoretical_support_class") == "global_context"
        }
        selected = _interpolate_curve(
            points, "selected_nominal_stride_area_fraction", "selected_count",
            gate.target_area_fraction,
        )
        global_count = 0.0
        global_count_complete = selected is not None
        for scale in global_scales:
            value = _interpolate_curve(
                points, "selected_nominal_stride_area_fraction", f"selected_count_{scale}",
                gate.target_area_fraction,
            )
            if value is None:
                global_count_complete = False
                break
            global_count += value
        record["global_context_cell_fraction_at_10pct_area"] = (
            None
            if not global_count_complete or selected is None or selected <= 0.0
            else float(global_count / selected)
        )
        record["global_context_scales"] = sorted(global_scales)
        result.append(record)
    return result


_DERIVED_METRIC_DIRECTIONS = {
    "area_to_50pct_deletion_effect": "lower",
    "area_to_50pct_retention_effect": "lower",
    "deletion_effect_at_10pct_area": "higher",
    "retention_effect_at_10pct_area": "higher",
    "retention_map_preserved_at_10pct_area": "higher",
}


def paired_scalar_bootstrap(
    rows: Sequence[Mapping[str, Any]],
    *,
    config: StatisticsConfig,
    seed_label: str,
    comparators: Sequence[str] | None = None,
) -> list[dict[str, Any]]:
    """Cluster-bootstrap precomputed curve endpoints with method pairing."""

    if not rows:
        return []
    by_key: dict[tuple[str, str, int], dict[str, Mapping[str, Any]]] = defaultdict(dict)
    for row in rows:
        key = (str(row["dataset"]), str(row["image_key"]), int(row["boundary"]))
        method = str(row["method"])
        if method in by_key[key]:
            raise AssertionError("duplicate derived curve endpoint method")
        by_key[key][method] = row
    if any(set(methods) != set(METHODS) for methods in by_key.values()):
        raise AssertionError("derived curve endpoints are not fully method-paired")
    comparators = tuple(
        comparators or (method for method in METHODS if method != "ranked_native")
    )
    clusters = list(
        dict.fromkeys(
            f"{dataset}:{methods['ranked_native']['cluster_id']}"
            for (dataset, _, _), methods in by_key.items()
        )
    )
    cluster_lookup = {value: index for index, value in enumerate(clusters)}
    metrics = tuple(_DERIVED_METRIC_DIRECTIONS)
    shape = (len(comparators), len(clusters), len(metrics))
    sums = np.zeros(shape, dtype=np.float64)
    counts = np.zeros(shape, dtype=np.float64)
    ranked_sums = np.zeros(shape, dtype=np.float64)
    control_sums = np.zeros(shape, dtype=np.float64)
    for comparator_index, comparator in enumerate(comparators):
        for (dataset, _, _), methods in by_key.items():
            cluster = f"{dataset}:{methods['ranked_native']['cluster_id']}"
            cluster_index = cluster_lookup[cluster]
            for metric_index, metric in enumerate(metrics):
                left = methods["ranked_native"].get(metric)
                right = methods[comparator].get(metric)
                if left is None or right is None:
                    continue
                left, right = float(left), float(right)
                if not math.isfinite(left) or not math.isfinite(right):
                    continue
                direction = _DERIVED_METRIC_DIRECTIONS[metric]
                contrast = left - right if direction == "higher" else right - left
                sums[comparator_index, cluster_index, metric_index] += contrast
                ranked_sums[comparator_index, cluster_index, metric_index] += left
                control_sums[comparator_index, cluster_index, metric_index] += right
                counts[comparator_index, cluster_index, metric_index] += 1.0
    bootstrap_seed = _stable_seed(config.bootstrap_seed, seed_label, "derived_endpoints")
    rng = np.random.default_rng(bootstrap_seed)
    probability = np.full(len(clusters), 1.0 / len(clusters))
    draws: list[list[list[np.ndarray]]] = [
        [[] for _ in metrics] for _ in comparators
    ]
    completed = 0
    while completed < config.bootstrap_samples:
        batch = min(config.bootstrap_chunk_size, config.bootstrap_samples - completed)
        weights = rng.multinomial(len(clusters), probability, size=batch).astype(np.float64)
        numerator = np.einsum("bc,pcm->bpm", weights, sums, optimize=True)
        denominator = np.einsum("bc,pcm->bpm", weights, counts, optimize=True)
        contrast = np.divide(
            numerator, denominator, out=np.full_like(numerator, np.nan),
            where=denominator > 0,
        )
        for comparator_index in range(len(comparators)):
            for metric_index in range(len(metrics)):
                draws[comparator_index][metric_index].append(
                    contrast[:, comparator_index, metric_index]
                )
        completed += batch
    results: list[dict[str, Any]] = []
    for comparator_index, comparator in enumerate(comparators):
        for metric_index, metric in enumerate(metrics):
            total = float(counts[comparator_index, :, metric_index].sum())
            if total <= 0.0:
                continue
            values = np.concatenate(draws[comparator_index][metric_index])
            values = values[np.isfinite(values)]
            if len(values) != config.bootstrap_samples:
                raise AssertionError("derived endpoint bootstrap lost all observations")
            direction = _DERIVED_METRIC_DIRECTIONS[metric]
            results.append(
                {
                    "comparator": comparator,
                    "metric": metric,
                    "n_paired_image_boundaries": int(total),
                    "n_clusters": len(clusters),
                    "ranked_native_mean": float(
                        ranked_sums[comparator_index, :, metric_index].sum() / total
                    ),
                    "comparator_mean": float(
                        control_sums[comparator_index, :, metric_index].sum() / total
                    ),
                    "contrast_positive_favors_ranked": float(
                        sums[comparator_index, :, metric_index].sum() / total
                    ),
                    "bootstrap_mean_contrast": float(np.mean(values)),
                    "ci_percentile": _percentile_interval(values, config.confidence_level),
                    "bootstrap_fraction_ranked_better": float(
                        (np.sum(values > 0) + 0.5 * np.sum(values == 0)) / len(values)
                    ),
                    "bootstrap_seed_derived": bootstrap_seed,
                    "direction": (
                        "ranked_minus_comparator" if direction == "higher"
                        else "comparator_minus_ranked"
                    ),
                }
            )
    return results


def _read_aggregate(
    dataset: str, manifest_path: Path
) -> tuple[
    dict[str, Any], list[GradeRow], dict[tuple[str, int], dict[str, Any]],
    list[dict[str, Any]], list[dict[str, Any]],
]:
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
    census: dict[tuple[str, int], dict[str, Any]] = {}
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
        census[(key, boundary)] = row
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
        grade = image_state[image_key]
        enriched = dict(row)
        enriched["true_grade"] = grade.true_grade
        enriched["predicted_grade"] = grade.predicted_grade
        enriched["correct"] = grade.true_grade == grade.predicted_grade
        enriched["boundary_outcome"] = _boundary_outcome(
            grade.true_grade, grade.predicted_grade, int(row["boundary"])
        )
        summaries.append(enriched)
    if len(summaries) != int(artifacts["summary_rows"]["rows"]):
        raise AssertionError("summary artifact row count changed")
    profiles: list[dict[str, Any]] = []
    if "curve_rows" in artifacts:
        curve_path = _artifact_path(manifest_path, artifacts["curve_rows"])
        profiles = _compact_curve_profiles(
            dataset=dataset,
            path=curve_path,
            expected_rows=int(artifacts["curve_rows"]["rows"]),
            image_state=image_state,
        )
        summary_keys = {
            (str(row["image_key"]), int(row["boundary"]), str(row["method"]))
            for row in summaries
        }
        profile_keys = {
            (str(row["image_key"]), int(row["boundary"]), str(row["method"]))
            for row in profiles
        }
        if profile_keys != summary_keys:
            raise AssertionError("curve profiles and AUC summaries cover different method pairs")
    return manifest, list(image_state.values()), census, summaries, profiles


def _grading_scopes(
    grade_rows_by_dataset: Mapping[str, Sequence[GradeRow]],
) -> Iterable[tuple[str, str, list[GradeRow]]]:
    all_grade = [row for dataset in sorted(grade_rows_by_dataset) for row in grade_rows_by_dataset[dataset]]
    yield "overall", "all", all_grade
    for dataset in sorted(grade_rows_by_dataset):
        yield "dataset", dataset, list(grade_rows_by_dataset[dataset])


def _intervention_scopes(
    rows_by_dataset: Mapping[str, Sequence[Mapping[str, Any]]],
) -> Iterable[tuple[str, str, list[Mapping[str, Any]]]]:
    all_rows = [row for dataset in sorted(rows_by_dataset) for row in rows_by_dataset[dataset]]
    yield "overall", "all", all_rows
    for dataset in sorted(rows_by_dataset):
        rows = list(rows_by_dataset[dataset])
        yield "dataset", dataset, rows
        boundaries = sorted({int(row["boundary"]) for row in rows})
        for boundary in boundaries:
            yield (
                "dataset_boundary", f"{dataset}:Y>{boundary}",
                [row for row in rows if int(row["boundary"]) == boundary],
            )
        for correct in (True, False):
            selected = [row for row in rows if bool(row["correct"]) is correct]
            if selected:
                yield (
                    "dataset_correctness",
                    f"{dataset}:{'correct' if correct else 'error'}",
                    selected,
                )
        correct_positive = [
            row for row in rows
            if bool(row["correct"]) and int(row["true_grade"]) > 0
        ]
        if correct_positive:
            yield "dataset_correct_positive", f"{dataset}:correct_positive", correct_positive
        for grade in sorted({int(row["true_grade"]) for row in rows}):
            selected = [
                row for row in rows
                if bool(row["correct"]) and int(row["true_grade"]) == grade
            ]
            if selected:
                yield "dataset_correct_true_grade", f"{dataset}:grade={grade}", selected
        for grade in sorted({int(row["predicted_grade"]) for row in rows}):
            selected = [row for row in rows if int(row["predicted_grade"]) == grade]
            if selected:
                yield "dataset_predicted_grade", f"{dataset}:grade={grade}", selected
        for boundary in boundaries:
            boundary_rows = [row for row in rows if int(row["boundary"]) == boundary]
            for outcome in ("true_positive", "false_positive", "false_negative"):
                selected = [row for row in boundary_rows if row["boundary_outcome"] == outcome]
                if selected:
                    yield (
                        "dataset_boundary_outcome",
                        f"{dataset}:Y>{boundary}:{outcome}",
                        selected,
                    )


def _finite_values(rows: Sequence[Mapping[str, Any]], field: str) -> np.ndarray:
    values = []
    for row in rows:
        value = row.get(field)
        if value is None:
            continue
        value = float(value)
        if math.isfinite(value):
            values.append(value)
    return np.asarray(values, dtype=np.float64)


def _descriptive(values: np.ndarray, *, total: int) -> dict[str, Any]:
    values = np.asarray(values, dtype=np.float64)
    if len(values) == 0:
        return {
            "n": 0, "total": total, "coverage": 0.0, "mean": None,
            "median": None, "q25": None, "q75": None,
        }
    return {
        "n": int(len(values)),
        "total": int(total),
        "coverage": float(len(values) / max(total, 1)),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "q25": float(np.quantile(values, 0.25)),
        "q75": float(np.quantile(values, 0.75)),
    }


def rf_footprint_diagnostics(
    derived_rows: Sequence[Mapping[str, Any]],
    *,
    target_area_fraction: float,
) -> dict[str, Any]:
    ranked = [row for row in derived_rows if row["method"] == "ranked_native"]
    fields = (
        "maximum_single_rf_footprint_at_10pct_area",
        "summed_rf_footprint_at_10pct_area",
        "clipped_union_rf_footprint_at_10pct_area",
        "global_context_cell_fraction_at_10pct_area",
        "deletion_effect_at_10pct_area",
        "retention_effect_at_10pct_area",
    )
    groups: list[tuple[str, str, list[Mapping[str, Any]]]] = [("overall", "all", ranked)]
    for dataset in sorted({str(row["dataset"]) for row in ranked}):
        dataset_rows = [row for row in ranked if row["dataset"] == dataset]
        groups.append(("dataset", dataset, dataset_rows))
        for grade in sorted({int(row["true_grade"]) for row in dataset_rows}):
            groups.append(
                (
                    "dataset_true_grade", f"{dataset}:grade={grade}",
                    [row for row in dataset_rows if int(row["true_grade"]) == grade],
                )
            )
        for correct in (True, False):
            selected = [row for row in dataset_rows if bool(row["correct"]) is correct]
            if selected:
                groups.append(
                    (
                        "dataset_correctness",
                        f"{dataset}:{'correct' if correct else 'error'}",
                        selected,
                    )
                )
        for boundary in sorted({int(row["boundary"]) for row in dataset_rows}):
            selected = [row for row in dataset_rows if int(row["boundary"]) == boundary]
            groups.append(("dataset_boundary", f"{dataset}:Y>{boundary}", selected))
            for outcome in ("true_positive", "false_positive", "false_negative"):
                outcome_rows = [
                    row for row in selected if row["boundary_outcome"] == outcome
                ]
                if outcome_rows:
                    groups.append(
                        (
                            "dataset_boundary_outcome",
                            f"{dataset}:Y>{boundary}:{outcome}", outcome_rows,
                        )
                    )
    records = []
    for scope_type, scope, rows in groups:
        records.append(
            {
                "scope_type": scope_type,
                "scope": scope,
                "n_image_boundaries": len(rows),
                "metrics": {
                    field: _descriptive(_finite_values(rows, field), total=len(rows))
                    for field in fields
                },
            }
        )
    bins = ((0.0, 0.01), (0.01, 0.10), (0.10, 0.50), (0.50, math.inf))
    footprint_effect_records = []
    for dataset in sorted({str(row["dataset"]) for row in ranked}):
        dataset_rows = [row for row in ranked if row["dataset"] == dataset]
        for lower, upper in bins:
            selected = [
                row for row in dataset_rows
                if row.get("maximum_single_rf_footprint_at_10pct_area") is not None
                and lower < float(row["maximum_single_rf_footprint_at_10pct_area"]) <= upper
            ]
            footprint_effect_records.append(
                {
                    "dataset": dataset,
                    "rf_footprint_bin": f"({lower:g},{upper if math.isfinite(upper) else 'inf'}]",
                    "n_image_boundaries": len(selected),
                    "deletion_effect": _descriptive(
                        _finite_values(selected, "deletion_effect_at_10pct_area"),
                        total=len(selected),
                    ),
                    "retention_effect": _descriptive(
                        _finite_values(selected, "retention_effect_at_10pct_area"),
                        total=len(selected),
                    ),
                }
            )
    return {
        "schema": RF_DIAGNOSTIC_SCHEMA,
        "target_nominal_addressable_area_fraction": target_area_fraction,
        "interpretation": (
            "theoretical receptive-field disclosure for exact ledger cells; large or "
            "global supports preclude a fine-locality claim even when intervention effects are strong"
        ),
        "locality_claim_adjudicated": False,
        "records": records,
        "effect_by_maximum_single_rf_footprint_bin": footprint_effect_records,
    }


def _record_index(
    records: Sequence[Mapping[str, Any]], *, contrast_field: str
) -> dict[tuple[str, str, str, str], Mapping[str, Any]]:
    result = {}
    for row in records:
        key = (
            str(row["scope_type"]), str(row["scope"]),
            str(row["comparator"]), str(row["metric"]),
        )
        if key in result:
            raise AssertionError("duplicate scoped intervention inference record")
        if contrast_field not in row:
            raise AssertionError("intervention inference record has the wrong contrast contract")
        result[key] = row
    return result


def _clause_status(checks: Sequence[Mapping[str, Any]]) -> str:
    statuses = [str(check["status"]) for check in checks]
    if any(value == "insufficient_data" for value in statuses):
        return "insufficient_data"
    return "pass" if statuses and all(value == "pass" for value in statuses) else "fail"


def adjudicate_gate_a(
    *,
    config: StatisticsConfig,
    intervention_records: Sequence[Mapping[str, Any]],
    derived_inference_records: Sequence[Mapping[str, Any]],
    derived_by_dataset: Mapping[str, Sequence[Mapping[str, Any]]],
    input_issues: Sequence[str],
    runtime_source: Mapping[str, Any],
) -> dict[str, Any]:
    """Apply every preregistered Gate A clause without human result selection."""

    gate = config.gate_a
    auc = _record_index(intervention_records, contrast_field="ranked_minus_comparator")
    derived = _record_index(
        derived_inference_records, contrast_field="contrast_positive_favors_ranked"
    )
    auc_checks = []
    for dataset in gate.required_datasets:
        for metric in (gate.deletion_auc_metric, gate.retention_auc_metric):
            for comparator in gate.matched_controls:
                row = auc.get(
                    (
                        "dataset_correct_positive", f"{dataset}:correct_positive",
                        comparator, metric,
                    )
                )
                if row is None or int(row.get("n_clusters", 0)) < gate.minimum_stratum_clusters:
                    status = "insufficient_data"
                else:
                    status = (
                        "pass"
                        if float(row["ranked_minus_comparator"]) >= gate.minimum_auc_advantage
                        and float(row["ci_percentile"][0]) > 0.0
                        else "fail"
                    )
                auc_checks.append(
                    {
                        "dataset": dataset, "metric": metric, "comparator": comparator,
                        "minimum_advantage": gate.minimum_auc_advantage,
                        "estimate": None if row is None else row["ranked_minus_comparator"],
                        "ci_percentile": None if row is None else row["ci_percentile"],
                        "n_clusters": None if row is None else row["n_clusters"],
                        "status": status,
                    }
                )

    grade_checks = []
    grade_contracts = {
        "dr": (gate.dr_positive_grades, gate.dr_minimum_passing_grades),
        "aptos": (gate.aptos_positive_grades, gate.aptos_minimum_passing_grades),
    }
    dataset_grade_results = {}
    for dataset, (grades, required_passes) in grade_contracts.items():
        individual = []
        for grade in grades:
            subchecks = []
            for metric in (gate.deletion_auc_metric, gate.retention_auc_metric):
                for comparator in gate.matched_controls:
                    row = auc.get(
                        ("dataset_correct_true_grade", f"{dataset}:grade={grade}", comparator, metric)
                    )
                    if row is None or int(row.get("n_clusters", 0)) < gate.minimum_stratum_clusters:
                        status = "insufficient_data"
                    else:
                        status = "pass" if float(row["ranked_minus_comparator"]) > 0.0 else "fail"
                    subchecks.append(
                        {
                            "metric": metric, "comparator": comparator,
                            "estimate": None if row is None else row["ranked_minus_comparator"],
                            "n_clusters": None if row is None else row["n_clusters"],
                            "status": status,
                        }
                    )
            status = _clause_status(subchecks)
            individual.append({"grade": grade, "status": status, "checks": subchecks})
        adequate = [row for row in individual if row["status"] != "insufficient_data"]
        passes = sum(row["status"] == "pass" for row in adequate)
        dataset_status = (
            "insufficient_data" if len(adequate) < required_passes
            else "pass" if passes >= required_passes else "fail"
        )
        dataset_grade_results[dataset] = {
            "minimum_passing_grades": required_passes,
            "adequately_powered_grades": len(adequate),
            "passing_grades": passes,
            "status": dataset_status,
            "grades": individual,
        }
        grade_checks.append(dataset_grade_results[dataset])

    concentration_checks = []
    for dataset in gate.required_datasets:
        rows = [
            row for row in derived_by_dataset.get(dataset, ())
            if bool(row["correct"]) and int(row["true_grade"]) > 0
        ]
        by_method = {
            method: [row for row in rows if row["method"] == method] for method in METHODS
        }
        for mode in ("deletion", "retention"):
            area_field = f"area_to_50pct_{mode}_effect"
            effect_field = f"{mode}_effect_at_10pct_area"
            ranked_area = _descriptive(
                _finite_values(by_method["ranked_native"], area_field),
                total=len(by_method["ranked_native"]),
            )
            ranked_effect = _descriptive(
                _finite_values(by_method["ranked_native"], effect_field),
                total=len(by_method["ranked_native"]),
            )
            for comparator in gate.matched_controls:
                control_area = _descriptive(
                    _finite_values(by_method[comparator], area_field),
                    total=len(by_method[comparator]),
                )
                coverage_ok = min(ranked_area["coverage"], ranked_effect["coverage"], control_area["coverage"])
                if coverage_ok < gate.minimum_metric_coverage:
                    status = "insufficient_data"
                    area_ratio = None
                else:
                    area_ratio = (
                        None if control_area["median"] is None or control_area["median"] <= 0.0
                        else float(ranked_area["median"] / control_area["median"])
                    )
                    area_route = (
                        ranked_area["median"] is not None
                        and ranked_area["median"] <= gate.maximum_ranked_area_to_target
                        and area_ratio is not None
                        and area_ratio <= gate.maximum_ranked_to_random_area_ratio
                    )
                    top_area_route = (
                        ranked_effect["median"] is not None
                        and ranked_effect["median"] >= gate.target_effect
                    )
                    status = "pass" if area_route or top_area_route else "fail"
                concentration_checks.append(
                    {
                        "dataset": dataset, "mode": mode, "comparator": comparator,
                        "ranked_area_to_target": ranked_area,
                        "comparator_area_to_target": control_area,
                        "ranked_effect_at_target_area": ranked_effect,
                        "ranked_to_comparator_median_area_ratio": area_ratio,
                        "status": status,
                    }
                )

    preservation_checks = []
    metric = "retention_map_preserved_at_10pct_area"
    for dataset in gate.required_datasets:
        for comparator in gate.matched_controls:
            row = derived.get(
                (
                    "dataset_correct_positive", f"{dataset}:correct_positive",
                    comparator, metric,
                )
            )
            if row is None or int(row.get("n_clusters", 0)) < gate.minimum_stratum_clusters:
                status = "insufficient_data"
            else:
                status = (
                    "pass"
                    if float(row["contrast_positive_favors_ranked"])
                    >= gate.minimum_retention_preservation_advantage
                    and float(row["ci_percentile"][0]) > 0.0
                    else "fail"
                )
            preservation_checks.append(
                {
                    "dataset": dataset, "comparator": comparator, "metric": metric,
                    "minimum_absolute_advantage": gate.minimum_retention_preservation_advantage,
                    "estimate": None if row is None else row["contrast_positive_favors_ranked"],
                    "ci_percentile": None if row is None else row["ci_percentile"],
                    "n_clusters": None if row is None else row["n_clusters"],
                    "status": status,
                }
            )

    completeness_issues = list(input_issues)
    if not bool(runtime_source.get("git_commit_available")):
        completeness_issues.append("statistics runtime Git commit is unavailable")
    if not bool(runtime_source.get("tracked_worktree_clean")):
        completeness_issues.append("statistics runtime tracked worktree is not clean")
    clauses = {
        "auc_advantage": {"status": _clause_status(auc_checks), "checks": auc_checks},
        "positive_grade_strata": {
            "status": _clause_status(grade_checks), "datasets": dataset_grade_results,
        },
        "concentration": {
            "status": _clause_status(concentration_checks), "checks": concentration_checks,
        },
        "retain_only_preservation": {
            "status": _clause_status(preservation_checks), "checks": preservation_checks,
        },
    }
    if completeness_issues:
        overall_status = "insufficient_data"
        claim = "withheld_incomplete"
    elif all(value["status"] == "pass" for value in clauses.values()):
        overall_status = "pass"
        claim = "supported_by_preregistered_gate"
    elif any(value["status"] == "insufficient_data" for value in clauses.values()):
        overall_status = "insufficient_data"
        claim = "withheld_incomplete"
    else:
        overall_status = "fail"
        claim = "not_supported"
    return {
        "schema": GATE_A_SCHEMA,
        "gate": "A",
        "status": overall_status,
        "auditability_claim_status": claim,
        "claim_authorized": overall_status == "pass",
        "fail_closed": True,
        "input_completeness_issues": completeness_issues,
        "clauses": clauses,
        "thresholds": {
            "matched_controls": list(gate.matched_controls),
            "minimum_auc_advantage": gate.minimum_auc_advantage,
            "minimum_stratum_clusters": gate.minimum_stratum_clusters,
            "target_effect": gate.target_effect,
            "target_area_fraction": gate.target_area_fraction,
            "maximum_ranked_area_to_target": gate.maximum_ranked_area_to_target,
            "maximum_ranked_to_random_area_ratio": gate.maximum_ranked_to_random_area_ratio,
            "minimum_retention_preservation_advantage": (
                gate.minimum_retention_preservation_advantage
            ),
            "minimum_metric_coverage": gate.minimum_metric_coverage,
        },
    }


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
    input_issues: list[str] = []
    grade_by_dataset: dict[str, list[GradeRow]] = {}
    census_by_dataset: dict[str, dict[tuple[str, int], dict[str, Any]]] = {}
    summaries_by_dataset: dict[str, list[dict[str, Any]]] = {}
    profiles_by_dataset: dict[str, list[dict[str, Any]]] = {}
    for dataset, path in sorted(aggregate_manifests.items()):
        manifest, grade_rows, census, summaries, profiles = _read_aggregate(dataset, path)
        grade_by_dataset[dataset] = grade_rows
        census_by_dataset[dataset] = census
        summaries_by_dataset[dataset] = summaries
        profiles_by_dataset[dataset] = profiles
        if not profiles:
            input_issues.append(f"{dataset} aggregate has no complete curve artifact")
        input_records.append(
            {
                "dataset": dataset,
                "path": str(path.resolve()),
                "sha256": file_sha256(path),
                "content_checksum_sha256": manifest["content_checksum_sha256"],
                "n_images": manifest["n_images"],
                "model_frozen_base_commit": BASE_COMMIT,
                "audit_protocol_checksum_sha256": manifest.get(
                    "audit_protocol_checksum_sha256"
                ),
                "cv_protocol_checksum_sha256": manifest.get("cv_protocol_checksum_sha256"),
                "audit_config_sha256": manifest.get("audit_config_sha256"),
                "model_implementation_signature": manifest.get("implementation_signature"),
                "model_architecture_signature": manifest.get("architecture_signature"),
                "fold_audit_implementation_sha256": manifest.get(
                    "fold_audit_implementation_sha256"
                ),
                "fold_audit_common_implementation_sha256": manifest.get(
                    "fold_audit_common_implementation_sha256"
                ),
                "aggregate_implementation_sha256": manifest.get(
                    "aggregate_implementation_sha256"
                ),
            }
        )
    if not grade_by_dataset:
        raise ValueError("at least one aggregate manifest is required")
    missing_datasets = set(config.gate_a.required_datasets) - set(grade_by_dataset)
    if missing_datasets:
        input_issues.append(
            f"missing required OOF datasets: {','.join(sorted(missing_datasets))}"
        )
    audit_protocols = {
        str(record.get("audit_protocol_checksum_sha256")) for record in input_records
    }
    implementations = {
        str(record.get("model_implementation_signature")) for record in input_records
    }
    if len(audit_protocols) != 1 or "None" in audit_protocols:
        input_issues.append("input aggregates do not share one declared intervention protocol")
    if len(implementations) != 1 or "None" in implementations:
        input_issues.append("input aggregates do not share one model implementation signature")
    output_dir.mkdir(parents=True, exist_ok=True)

    proper_records: list[dict[str, Any]] = []
    reliability_records: list[dict[str, Any]] = []
    grading_bootstrap_records: list[dict[str, Any]] = []
    intervention_records: list[dict[str, Any]] = []
    for scope_type, scope, grade_rows in _grading_scopes(grade_by_dataset):
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
    for scope_type, scope, summary_rows in _intervention_scopes(summaries_by_dataset):
        primary_scope = scope_type in {"overall", "dataset", "dataset_boundary"}
        intervention_records.extend(
            {"scope_type": scope_type, "scope": scope, **record}
            for record in paired_intervention_bootstrap(
                summary_rows,
                config=config,
                seed_label=f"intervention:{scope_type}:{scope}",
                auc_fields=(
                    None
                    if primary_scope
                    else (config.gate_a.deletion_auc_metric, config.gate_a.retention_auc_metric)
                ),
                comparators=None if primary_scope else config.gate_a.matched_controls,
            )
        )

    derived_by_dataset: dict[str, list[dict[str, Any]]] = {}
    for dataset, profiles in profiles_by_dataset.items():
        derived_by_dataset[dataset] = derive_curve_endpoints(
            profiles, census_by_dataset[dataset], gate=config.gate_a
        )
    derived_inference_records: list[dict[str, Any]] = []
    for scope_type, scope, rows in _intervention_scopes(derived_by_dataset):
        primary_scope = scope_type in {"overall", "dataset", "dataset_boundary"}
        derived_inference_records.extend(
            {"scope_type": scope_type, "scope": scope, **record}
            for record in paired_scalar_bootstrap(
                rows,
                config=config,
                seed_label=f"derived:{scope_type}:{scope}",
                comparators=None if primary_scope else config.gate_a.matched_controls,
            )
        )

    runtime_source = _runtime_source_record()
    all_derived = [
        row for dataset in sorted(derived_by_dataset) for row in derived_by_dataset[dataset]
    ]
    rf_payload = rf_footprint_diagnostics(
        all_derived, target_area_fraction=config.gate_a.target_area_fraction
    )
    gate_payload = adjudicate_gate_a(
        config=config,
        intervention_records=intervention_records,
        derived_inference_records=derived_inference_records,
        derived_by_dataset=derived_by_dataset,
        input_issues=input_issues,
        runtime_source=runtime_source,
    )

    common = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "statistics_protocol_path": str(protocol_path.resolve()),
        "statistics_protocol_file_sha256": file_sha256(protocol_path),
        "statistics_protocol_checksum_sha256": protocol["content_checksum_sha256"],
        "statistics_runtime_source": runtime_source,
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
            "derived_curve_endpoint_records": derived_inference_records,
        },
        "rf_footprint_diagnostics": {
            **rf_payload,
            **common,
        },
        "gate_a": {
            **gate_payload,
            **common,
        },
    }
    filenames = {
        "proper_score_calibration": "OOF_PROPER_SCORE_CALIBRATION.json",
        "grading_bootstrap": "OOF_GRADING_CLUSTER_BOOTSTRAP.json",
        "intervention_bootstrap": "OOF_INTERVENTION_PAIRED_BOOTSTRAP.json",
        "rf_footprint_diagnostics": "OOF_RF_FOOTPRINT_DIAGNOSTICS.json",
        "gate_a": "OOF_GATE_A_ADJUDICATION.json",
    }
    artifacts: dict[str, Any] = {}
    for name, payload in payloads.items():
        payload["content_checksum_sha256"] = canonical_sha256(payload)
        path = output_dir / filenames[name]
        write_json_atomic(path, payload)
        record_count = (
            len(payload["records"])
            if "records" in payload
            else len(payload.get("clauses", {}))
        )
        artifacts[name] = {
            "path": str(path.resolve()),
            "sha256": file_sha256(path),
            "content_checksum_sha256": payload["content_checksum_sha256"],
            "records": record_count,
        }
        if name == "proper_score_calibration":
            artifacts[name]["reliability_records"] = len(payload["reliability_records"])
        elif name == "intervention_bootstrap":
            artifacts[name]["derived_curve_endpoint_records"] = len(
                payload["derived_curve_endpoint_records"]
            )
        elif name == "rf_footprint_diagnostics":
            artifacts[name]["effect_by_footprint_bin_records"] = len(
                payload["effect_by_maximum_single_rf_footprint_bin"]
            )
        elif name == "gate_a":
            artifacts[name]["gate_status"] = payload["status"]
            artifacts[name]["claim_authorized"] = payload["claim_authorized"]
    manifest: dict[str, Any] = {
        "schema": STATISTICS_MANIFEST_SCHEMA,
        **common,
        "analysis_implementation_sha256": runtime_source["analysis_implementation_sha256"],
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
