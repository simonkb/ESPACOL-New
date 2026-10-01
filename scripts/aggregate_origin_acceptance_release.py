#!/usr/bin/env python3
"""Audit and aggregate the coordinated outer release without reselection."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from statistics import mean, stdev
from typing import Any

import numpy as np

from scripts.origin_acceptance_baseline_common import (
    BASELINE_ORDER,
    BASELINE_SPECS,
    DATASET_FOLDS,
    PAIRED_BOOTSTRAP_SAMPLES,
    PAIRED_BOOTSTRAP_SEED,
    PROTOCOL_ID,
    PROTOCOL_SHA256,
    REPLICATION_SEEDS,
    canonical_sha256,
    file_sha256,
    full_tasks,
    release_identifier_policy,
)
from train_origin import write_json_atomic


_METRICS = (
    "acc",
    "qwk",
    "mae",
    "expected_grade_mae",
    "balanced_acc",
    "macro_f1",
    "ece",
    "nll",
    "rps",
    "multiclass_brier",
    "threshold_ece",
    "threshold_binary_brier",
    "classwise_ece",
)
_BOUNDARY_ECE_METRICS = tuple(f"threshold_ece_boundary_{index}" for index in range(4))
_BOUNDARY_BRIER_METRICS = tuple(
    f"threshold_binary_brier_boundary_{index}" for index in range(4)
)
_CLASSWISE_ECE_METRICS = tuple(f"classwise_ece_class_{index}" for index in range(5))
_PER_GRADE_RECALL_METRICS = tuple(f"per_grade_recall_{index}" for index in range(5))
_SUMMARY_METRICS = (
    _METRICS
    + _BOUNDARY_ECE_METRICS
    + _BOUNDARY_BRIER_METRICS
    + _CLASSWISE_ECE_METRICS
    + _PER_GRADE_RECALL_METRICS
)
_BOOTSTRAP_METRICS = (
    "acc",
    "qwk",
    "mae",
    "expected_grade_mae",
    "balanced_acc",
    "macro_f1",
    "nll",
    "rps",
    "multiclass_brier",
    "threshold_ece",
    "threshold_binary_brier",
    "classwise_ece",
) + (
    _BOUNDARY_ECE_METRICS
    + _BOUNDARY_BRIER_METRICS
    + _CLASSWISE_ECE_METRICS
    + _PER_GRADE_RECALL_METRICS
)
_HIGHER_IS_BETTER = {
    "acc": True,
    "qwk": True,
    "mae": False,
    "expected_grade_mae": False,
    "balanced_acc": True,
    "macro_f1": True,
    "nll": False,
    "rps": False,
    "multiclass_brier": False,
    "threshold_ece": False,
    "threshold_binary_brier": False,
    "classwise_ece": False,
    **{metric: False for metric in _BOUNDARY_ECE_METRICS},
    **{metric: False for metric in _BOUNDARY_BRIER_METRICS},
    **{metric: False for metric in _CLASSWISE_ECE_METRICS},
    **{metric: True for metric in _PER_GRADE_RECALL_METRICS},
}


def _load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise TypeError(f"expected object at {path}")
    return value


def _validate_sha256_identifiers(values: np.ndarray, *, field: str) -> None:
    values = np.asarray(values).astype(str)
    if values.ndim != 1 or len(values) < 1:
        raise ValueError(f"{field} must be a non-empty vector")
    hexadecimal = set("0123456789abcdef")
    if any(len(value) != 64 or not set(value) <= hexadecimal for value in values):
        raise ValueError(f"{field} must contain lowercase SHA-256 identifiers")


def _derived_seed(base: int, dataset: str, comparator: str) -> int:
    material = f"{base}:{dataset}:{comparator}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(material).digest()[:8], "big") % (2**32)


def _cluster_statistics(record: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Additive sufficient statistics for a patient/image cluster bootstrap."""

    probs = record["class_probs"].astype(np.float64, copy=False)
    cumulative = record["cumulative_probs"].astype(np.float64, copy=False)
    labels = record["label"].astype(np.int64, copy=False)
    predicted = record["prediction"].astype(np.int64, copy=False)
    expected_grade = record["expected_grade"].astype(np.float64, copy=False)
    cluster_ids = record["cluster_id"].astype(str)
    clusters = list(dict.fromkeys(cluster_ids.tolist()))
    lookup = {cluster: index for index, cluster in enumerate(clusters)}
    inverse = np.asarray([lookup[value] for value in cluster_ids], dtype=np.int64)
    cluster_count = len(clusters)
    classes = probs.shape[1]
    boundaries = classes - 1
    stats: dict[str, np.ndarray] = {
        "cluster_ids": np.asarray(clusters, dtype=str),
        "n": np.zeros(cluster_count),
        "correct": np.zeros(cluster_count),
        "absolute_error": np.zeros(cluster_count),
        "expected_absolute_error": np.zeros(cluster_count),
        "nll": np.zeros(cluster_count),
        "rps": np.zeros(cluster_count),
        "multiclass_brier": np.zeros(cluster_count),
        "threshold_binary_brier": np.zeros((cluster_count, boundaries)),
        "confusion": np.zeros((cluster_count, classes, classes)),
        "calibration_predicted": np.zeros((cluster_count, boundaries, 15)),
        "calibration_observed": np.zeros((cluster_count, boundaries, 15)),
        "classwise_calibration_predicted": np.zeros(
            (cluster_count, classes, 15)
        ),
        "classwise_calibration_observed": np.zeros(
            (cluster_count, classes, 15)
        ),
    }
    np.add.at(stats["n"], inverse, 1.0)
    np.add.at(stats["correct"], inverse, (predicted == labels).astype(np.float64))
    np.add.at(stats["absolute_error"], inverse, np.abs(predicted - labels))
    np.add.at(
        stats["expected_absolute_error"], inverse, np.abs(expected_grade - labels)
    )
    selected = np.clip(probs[np.arange(len(labels)), labels], np.finfo(np.float64).tiny, 1.0)
    np.add.at(stats["nll"], inverse, -np.log(selected))
    thresholds = np.arange(boundaries)[None, :]
    target_thresholds = labels[:, None] > thresholds
    np.add.at(
        stats["threshold_binary_brier"],
        inverse,
        (cumulative - target_thresholds) ** 2,
    )
    per_sample_rps = np.mean((cumulative - target_thresholds) ** 2, axis=1)
    np.add.at(stats["rps"], inverse, per_sample_rps)
    target_classes = np.eye(classes, dtype=np.float64)[labels]
    np.add.at(
        stats["multiclass_brier"],
        inverse,
        np.sum((probs - target_classes) ** 2, axis=1),
    )
    np.add.at(stats["confusion"], (inverse, labels, predicted), 1.0)
    assignments = np.minimum((np.clip(cumulative, 0.0, 1.0) * 15).astype(int), 14)
    for boundary in range(boundaries):
        np.add.at(
            stats["calibration_predicted"],
            (inverse, np.full(len(labels), boundary), assignments[:, boundary]),
            cumulative[:, boundary],
        )
        np.add.at(
            stats["calibration_observed"],
            (inverse, np.full(len(labels), boundary), assignments[:, boundary]),
            target_thresholds[:, boundary].astype(np.float64),
        )
    class_assignments = np.minimum(
        (np.clip(probs, 0.0, 1.0) * 15).astype(int), 14
    )
    for class_index in range(classes):
        np.add.at(
            stats["classwise_calibration_predicted"],
            (
                inverse,
                np.full(len(labels), class_index),
                class_assignments[:, class_index],
            ),
            probs[:, class_index],
        )
        np.add.at(
            stats["classwise_calibration_observed"],
            (
                inverse,
                np.full(len(labels), class_index),
                class_assignments[:, class_index],
            ),
            (labels == class_index).astype(np.float64),
        )
    return stats


def _weighted_metrics(
    weights: np.ndarray, stats: dict[str, np.ndarray]
) -> dict[str, np.ndarray]:
    """Vectorized metrics for rows of cluster multiplicity weights."""

    total = weights @ stats["n"]
    if np.any(total <= 0):
        raise AssertionError("bootstrap draw contains no observations")
    result = {
        "acc": 100.0 * (weights @ stats["correct"]) / total,
        "mae": (weights @ stats["absolute_error"]) / total,
        "expected_grade_mae": (weights @ stats["expected_absolute_error"]) / total,
        "nll": (weights @ stats["nll"]) / total,
        "rps": (weights @ stats["rps"]) / total,
        "multiclass_brier": (weights @ stats["multiclass_brier"]) / total,
    }
    confusion = np.tensordot(weights, stats["confusion"], axes=(1, 0))
    classes = confusion.shape[1]
    row = confusion.sum(axis=2)
    column = confusion.sum(axis=1)
    diagonal = np.diagonal(confusion, axis1=1, axis2=2)
    recalls = np.divide(
        diagonal,
        row,
        out=np.zeros_like(diagonal),
        where=row > 0.0,
    )
    precisions = np.divide(
        diagonal,
        column,
        out=np.zeros_like(diagonal),
        where=column > 0.0,
    )
    f1_denominator = recalls + precisions
    f1 = np.divide(
        2.0 * recalls * precisions,
        f1_denominator,
        out=np.zeros_like(recalls),
        where=f1_denominator > 0.0,
    )
    present = row > 0.0
    present_count = np.maximum(present.sum(axis=1), 1)
    result["balanced_acc"] = 100.0 * (recalls * present).sum(axis=1) / present_count
    result["macro_f1"] = (f1 * present).sum(axis=1) / present_count
    for grade in range(classes):
        result[f"per_grade_recall_{grade}"] = recalls[:, grade]
    expected = row[:, :, None] * column[:, None, :] / total[:, None, None]
    coordinates = np.arange(classes, dtype=np.float64)
    penalty = (coordinates[:, None] - coordinates[None, :]) ** 2
    penalty /= max(1, (classes - 1) ** 2)
    numerator = np.sum(confusion * penalty[None], axis=(1, 2))
    denominator = np.sum(expected * penalty[None], axis=(1, 2))
    qwk = np.zeros_like(denominator)
    valid_qwk = denominator > 0.0
    qwk[valid_qwk] = 1.0 - numerator[valid_qwk] / denominator[valid_qwk]
    result["qwk"] = qwk
    predicted_sum = np.tensordot(
        weights, stats["calibration_predicted"], axes=(1, 0)
    )
    observed_sum = np.tensordot(
        weights, stats["calibration_observed"], axes=(1, 0)
    )
    threshold_ece_by_boundary = (
        np.abs(predicted_sum - observed_sum).sum(axis=2) / total[:, None]
    )
    result["threshold_ece"] = threshold_ece_by_boundary.mean(axis=1)
    threshold_brier_by_boundary = (
        np.tensordot(weights, stats["threshold_binary_brier"], axes=(1, 0))
        / total[:, None]
    )
    result["threshold_binary_brier"] = threshold_brier_by_boundary.mean(axis=1)
    for boundary in range(threshold_ece_by_boundary.shape[1]):
        result[f"threshold_ece_boundary_{boundary}"] = threshold_ece_by_boundary[
            :, boundary
        ]
        result[f"threshold_binary_brier_boundary_{boundary}"] = (
            threshold_brier_by_boundary[:, boundary]
        )
    classwise_predicted_sum = np.tensordot(
        weights, stats["classwise_calibration_predicted"], axes=(1, 0)
    )
    classwise_observed_sum = np.tensordot(
        weights, stats["classwise_calibration_observed"], axes=(1, 0)
    )
    classwise_ece_by_class = (
        np.abs(classwise_predicted_sum - classwise_observed_sum).sum(axis=2)
        / total[:, None]
    )
    result["classwise_ece"] = classwise_ece_by_class.mean(axis=1)
    for class_index in range(classwise_ece_by_class.shape[1]):
        result[f"classwise_ece_class_{class_index}"] = classwise_ece_by_class[
            :, class_index
        ]
    return result


def _oof_threshold_reliability(
    records: dict[tuple[str, int, str, int], dict[str, np.ndarray]],
    *,
    dataset: str,
    variant: str,
    training_seed: int,
    bins: int = 15,
) -> dict[str, Any]:
    cumulative = np.concatenate(
        [
            records[(dataset, fold, variant, training_seed)]["cumulative_probs"]
            for fold in DATASET_FOLDS[dataset]
        ]
    ).astype(np.float64, copy=False)
    labels = np.concatenate(
        [
            records[(dataset, fold, variant, training_seed)]["label"]
            for fold in DATASET_FOLDS[dataset]
        ]
    ).astype(np.int64, copy=False)
    boundaries = cumulative.shape[1]
    payload = []
    eces = []
    briers = []
    for boundary in range(boundaries):
        predicted = np.clip(cumulative[:, boundary], 0.0, 1.0)
        observed = (labels > boundary).astype(np.float64)
        binary_brier = float(np.mean((predicted - observed) ** 2))
        assignment = np.minimum((predicted * bins).astype(int), bins - 1)
        bin_rows = []
        ece = 0.0
        for index in range(bins):
            selected = assignment == index
            count = int(selected.sum())
            if count:
                mean_predicted = float(predicted[selected].mean())
                empirical = float(observed[selected].mean())
                gap = abs(mean_predicted - empirical)
                ece += count / len(labels) * gap
            else:
                mean_predicted = empirical = gap = None
            bin_rows.append(
                {
                    "bin": index,
                    "lower": index / bins,
                    "upper": (index + 1) / bins,
                    "count": count,
                    "mean_predicted": mean_predicted,
                    "empirical_frequency": empirical,
                    "absolute_gap": gap,
                }
            )
        eces.append(ece)
        briers.append(binary_brier)
        payload.append(
            {
                "boundary": boundary,
                "event": f"Y>{boundary}",
                "ece": ece,
                "binary_brier": binary_brier,
                "bins": bin_rows,
            }
        )
    return {
        "scope": "pooled_out_of_fold_predictions_for_one_training_seed",
        "bin_count": bins,
        "n": int(len(labels)),
        "threshold_ece": float(np.mean(eces)),
        "threshold_ece_by_boundary": eces,
        "threshold_binary_brier": float(np.mean(briers)),
        "threshold_binary_brier_by_boundary": briers,
        "boundaries": payload,
    }


def _oof_classwise_reliability(
    records: dict[tuple[str, int, str, int], dict[str, np.ndarray]],
    *,
    dataset: str,
    variant: str,
    training_seed: int,
    bins: int = 15,
) -> dict[str, Any]:
    probabilities = np.concatenate(
        [
            records[(dataset, fold, variant, training_seed)]["class_probs"]
            for fold in DATASET_FOLDS[dataset]
        ]
    ).astype(np.float64, copy=False)
    labels = np.concatenate(
        [
            records[(dataset, fold, variant, training_seed)]["label"]
            for fold in DATASET_FOLDS[dataset]
        ]
    ).astype(np.int64, copy=False)
    classes = probabilities.shape[1]
    payload = []
    eces = []
    for class_index in range(classes):
        predicted = np.clip(probabilities[:, class_index], 0.0, 1.0)
        observed = (labels == class_index).astype(np.float64)
        assignment = np.minimum((predicted * bins).astype(int), bins - 1)
        bin_rows = []
        ece = 0.0
        for index in range(bins):
            selected = assignment == index
            count = int(selected.sum())
            if count:
                mean_predicted = float(predicted[selected].mean())
                empirical = float(observed[selected].mean())
                gap = abs(mean_predicted - empirical)
                ece += count / len(labels) * gap
            else:
                mean_predicted = empirical = gap = None
            bin_rows.append(
                {
                    "bin": index,
                    "lower": index / bins,
                    "upper": (index + 1) / bins,
                    "right_edge_inclusive": index == bins - 1,
                    "count": count,
                    "mean_predicted": mean_predicted,
                    "empirical_frequency": empirical,
                    "absolute_gap": gap,
                }
            )
        eces.append(ece)
        payload.append(
            {
                "class": class_index,
                "event": f"Y={class_index}",
                "prevalence": float(observed.mean()),
                "ece": ece,
                "bins": bin_rows,
            }
        )
    return {
        "scope": "pooled_out_of_fold_predictions_for_one_training_seed",
        "bin_count": bins,
        "n": int(len(labels)),
        "definition": "one_vs_rest_per_class",
        "aggregation": "unweighted_mean_across_classes",
        "classwise_ece": float(np.mean(eces)),
        "classwise_ece_by_class": eces,
        "classes": payload,
    }


def paired_fold_seed_cluster_bootstrap(
    records: dict[tuple[str, int, str, int], dict[str, np.ndarray]],
    *,
    dataset: str,
    comparator: str,
    samples: int,
    seed: int,
    chunk_size: int = 128,
) -> dict[str, Any]:
    """Paired clusters within fold, with each draw shared across three seeds."""

    if samples < 1:
        raise ValueError("bootstrap samples must be positive")
    folds = DATASET_FOLDS[dataset]
    statistics: dict[tuple[int, str, int], dict[str, np.ndarray]] = {}
    for fold in folds:
        reference_record = records[
            (dataset, fold, "origin_ctmc", REPLICATION_SEEDS[0])
        ]
        for training_seed in REPLICATION_SEEDS:
            for variant in ("origin_ctmc", comparator):
                key = (dataset, fold, variant, training_seed)
                record = records[key]
                for identity_field in ("image_id", "label", "cluster_id"):
                    if not np.array_equal(
                        record[identity_field], reference_record[identity_field]
                    ):
                        raise AssertionError(
                            f"unpaired {identity_field} for {dataset}/fold{fold}/"
                            f"{variant}/seed{training_seed}"
                        )
                statistics[(fold, variant, training_seed)] = _cluster_statistics(record)
        reference_clusters = statistics[(fold, "origin_ctmc", REPLICATION_SEEDS[0])][
            "cluster_ids"
        ]
        for training_seed in REPLICATION_SEEDS:
            for variant in ("origin_ctmc", comparator):
                observed = statistics[(fold, variant, training_seed)]["cluster_ids"]
                if not np.array_equal(observed, reference_clusters):
                    raise AssertionError(
                        f"cluster order is not paired for {dataset}/fold{fold}/{variant}"
                    )

    point = {metric: 0.0 for metric in _BOOTSTRAP_METRICS}
    cell_count = len(folds) * len(REPLICATION_SEEDS)
    for fold in folds:
        clusters = len(statistics[(fold, "origin_ctmc", REPLICATION_SEEDS[0])]["cluster_ids"])
        weights = np.ones((1, clusters), dtype=np.float64)
        for training_seed in REPLICATION_SEEDS:
            origin = _weighted_metrics(weights, statistics[(fold, "origin_ctmc", training_seed)])
            control = _weighted_metrics(weights, statistics[(fold, comparator, training_seed)])
            for metric in _BOOTSTRAP_METRICS:
                point[metric] += float(origin[metric][0] - control[metric][0]) / cell_count

    draws = {metric: np.empty(samples, dtype=np.float64) for metric in _BOOTSTRAP_METRICS}
    rng = np.random.default_rng(_derived_seed(seed, dataset, comparator))
    completed = 0
    while completed < samples:
        batch = min(chunk_size, samples - completed)
        accumulated = {
            metric: np.zeros(batch, dtype=np.float64)
            for metric in _BOOTSTRAP_METRICS
        }
        for fold in folds:
            clusters = len(
                statistics[(fold, "origin_ctmc", REPLICATION_SEEDS[0])]["cluster_ids"]
            )
            weights = rng.multinomial(
                clusters,
                np.full(clusters, 1.0 / clusters),
                size=batch,
            ).astype(np.float64, copy=False)
            for training_seed in REPLICATION_SEEDS:
                origin = _weighted_metrics(
                    weights, statistics[(fold, "origin_ctmc", training_seed)]
                )
                control = _weighted_metrics(
                    weights, statistics[(fold, comparator, training_seed)]
                )
                for metric in _BOOTSTRAP_METRICS:
                    accumulated[metric] += (
                        origin[metric] - control[metric]
                    ) / cell_count
        for metric in _BOOTSTRAP_METRICS:
            draws[metric][completed : completed + batch] = accumulated[metric]
        completed += batch

    comparison: dict[str, Any] = {}
    for metric in _BOOTSTRAP_METRICS:
        lower, upper = np.percentile(draws[metric], [2.5, 97.5])
        comparison[metric] = {
            "origin_minus_comparator": point[metric],
            "bootstrap_mean_delta": float(draws[metric].mean()),
            "ci95_percentile": [float(lower), float(upper)],
            "higher_is_better": _HIGHER_IS_BETTER[metric],
        }
    return {
        "method": "paired_within_fold_cluster_bootstrap_percentile",
        "reference_variant": "origin_ctmc",
        "comparator": comparator,
        "dataset": dataset,
        "cluster_unit": "patient_stem" if dataset == "dr" else "image",
        "fold_pairing": "same cluster multiplicities for both arms and all seeds within fold",
        "seed_pairing": "arm contrast within each training seed then mean across seeds",
        "fold_aggregation": "unweighted mean of paired fold-seed contrasts",
        "samples": samples,
        "base_seed": seed,
        "derived_seed": _derived_seed(seed, dataset, comparator),
        "delta_orientation": "origin_ctmc_minus_comparator",
        "comparisons": comparison,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment_root", required=True)
    parser.add_argument(
        "--bootstrap_samples", type=int, default=PAIRED_BOOTSTRAP_SAMPLES
    )
    parser.add_argument("--bootstrap_seed", type=int, default=PAIRED_BOOTSTRAP_SEED)
    args = parser.parse_args()
    if args.bootstrap_samples != PAIRED_BOOTSTRAP_SAMPLES:
        raise ValueError("bootstrap sample count differs from the registered protocol")
    if args.bootstrap_seed != PAIRED_BOOTSTRAP_SEED:
        raise ValueError("bootstrap seed differs from the registered protocol")
    root = Path(args.experiment_root)
    output_dir = root / "full" / "outer_release"
    aggregate_targets = (
        output_dir / "aggregate.json",
        output_dir / "fold_metrics.csv",
        output_dir / "OUTER_RELEASE_COMPLETE.json",
    )
    existing = [str(path) for path in aggregate_targets if path.exists()]
    if existing:
        raise FileExistsError(
            "aggregation refuses to overwrite authoritative release artifacts: "
            + ", ".join(existing)
        )
    rows: list[dict[str, Any]] = []
    records: dict[tuple[str, int, str, int], dict[str, np.ndarray]] = {}
    identity_by_fold: dict[tuple[str, int], tuple[tuple[str, ...], tuple[int, ...]]] = {}
    for task in full_tasks():
        directory = root / "full" / "outer_release" / task.key
        metrics_path = directory / "outer_metrics.json"
        predictions_path = directory / "outer_predictions.npz"
        complete_path = directory / "OUTER_COMPLETE.json"
        if not all(path.is_file() for path in (metrics_path, predictions_path, complete_path)):
            raise FileNotFoundError(f"incomplete outer release: {directory}")
        payload = _load_json(metrics_path)
        if payload.get("schema") != "origin-acceptance-outer-result-v2":
            raise ValueError(f"outer metrics schema mismatch: {metrics_path}")
        checksum = payload.get("content_checksum_sha256")
        unsigned = dict(payload)
        unsigned.pop("content_checksum_sha256", None)
        if checksum != canonical_sha256(unsigned):
            raise ValueError(f"outer metrics checksum mismatch: {metrics_path}")
        if payload.get("protocol_sha256") != PROTOCOL_SHA256:
            raise ValueError(f"protocol mismatch: {metrics_path}")
        if payload.get("test_evaluated") is not True or payload.get("selection_reopened") is not False:
            raise ValueError(f"invalid outer-release semantics: {metrics_path}")
        if payload.get("predictions_sha256") != file_sha256(predictions_path):
            raise ValueError(f"prediction hash mismatch: {predictions_path}")
        if payload.get("identifier_privacy") != release_identifier_policy(task.dataset):
            raise ValueError(f"identifier privacy contract mismatch: {metrics_path}")
        complete = _load_json(complete_path)
        if complete.get("schema") != "origin-acceptance-outer-complete-v2":
            raise ValueError(f"outer completion schema mismatch: {complete_path}")
        if complete.get("outer_metrics_sha256") != file_sha256(metrics_path):
            raise ValueError(f"completion hash mismatch: {complete_path}")
        with np.load(predictions_path, allow_pickle=False) as archive:
            required_arrays = {
                "image_id",
                "cluster_id",
                "label",
                "prediction",
                "expected_grade",
                "class_probs",
                "cumulative_probs",
            }
            missing = required_arrays - set(archive.files)
            if missing:
                raise ValueError(
                    f"outer prediction archive lacks {sorted(missing)}: {predictions_path}"
                )
            forbidden = {"image_path", "patient_id", "raw_cluster_id"} & set(
                archive.files
            )
            if forbidden:
                raise ValueError(
                    f"outer prediction archive exposes raw identifiers {sorted(forbidden)}: "
                    f"{predictions_path}"
                )
            image_ids = tuple(str(value) for value in archive["image_id"].tolist())
            labels = tuple(int(value) for value in archive["label"].tolist())
            record = {name: np.asarray(archive[name]).copy() for name in required_arrays}
        _validate_sha256_identifiers(record["image_id"], field="image_id")
        _validate_sha256_identifiers(record["cluster_id"], field="cluster_id")
        if len(set(record["image_id"].astype(str).tolist())) != len(record["image_id"]):
            raise ValueError(f"duplicate image_id in {predictions_path}")
        fold_key = (task.dataset, task.fold)
        identity = (image_ids, labels)
        previous = identity_by_fold.setdefault(fold_key, identity)
        if previous != identity:
            raise ValueError(f"outer item/order mismatch across comparators for {fold_key}")
        records[
            (task.dataset, task.fold, task.baseline_variant, task.training_seed)
        ] = record
        metrics = payload["metrics"]
        per_grade_recall = metrics.get("per_grade_recall")
        if not isinstance(per_grade_recall, list) or len(per_grade_recall) != 5:
            raise ValueError(
                f"outer metrics must contain recalls for grades 0--4: {metrics_path}"
            )
        threshold_ece_by_boundary = metrics.get("threshold_ece_by_boundary")
        threshold_brier_by_boundary = metrics.get(
            "threshold_binary_brier_by_boundary"
        )
        classwise_ece_by_class = metrics.get("classwise_ece_by_class")
        if not isinstance(threshold_ece_by_boundary, list) or len(
            threshold_ece_by_boundary
        ) != 4:
            raise ValueError(
                f"outer metrics must contain ECE for four boundaries: {metrics_path}"
            )
        if not isinstance(threshold_brier_by_boundary, list) or len(
            threshold_brier_by_boundary
        ) != 4:
            raise ValueError(
                f"outer metrics must contain binary Brier for four boundaries: "
                f"{metrics_path}"
            )
        if not isinstance(classwise_ece_by_class, list) or len(
            classwise_ece_by_class
        ) != 5:
            raise ValueError(
                f"outer metrics must contain one-vs-rest ECE for grades 0--4: "
                f"{metrics_path}"
            )
        rows.append(
            {
                "dataset": task.dataset,
                "fold": task.fold,
                "baseline_variant": task.baseline_variant,
                "training_seed": task.training_seed,
                **{name: float(metrics[name]) for name in _METRICS},
                **{
                    f"threshold_ece_boundary_{index}": float(value)
                    for index, value in enumerate(threshold_ece_by_boundary)
                },
                **{
                    f"threshold_binary_brier_boundary_{index}": float(value)
                    for index, value in enumerate(threshold_brier_by_boundary)
                },
                **{
                    f"classwise_ece_class_{index}": float(value)
                    for index, value in enumerate(classwise_ece_by_class)
                },
                **{
                    f"per_grade_recall_{index}": float(value)
                    for index, value in enumerate(per_grade_recall)
                },
                "n": int(metrics["n"]),
                "best_epoch": int(payload["best_epoch"]),
            }
        )

    summary: dict[str, Any] = {}
    for dataset, folds in DATASET_FOLDS.items():
        summary[dataset] = {}
        for variant in BASELINE_ORDER:
            seed_means: dict[str, dict[str, float]] = {}
            for seed in REPLICATION_SEEDS:
                subset = [
                    row
                    for row in rows
                    if row["dataset"] == dataset
                    and row["baseline_variant"] == variant
                    and row["training_seed"] == seed
                ]
                if len(subset) != len(folds):
                    raise ValueError(f"missing folds for {dataset}/{variant}/seed{seed}")
                seed_means[str(seed)] = {
                    metric: mean(float(row[metric]) for row in subset)
                    for metric in _SUMMARY_METRICS
                }
            replication = {}
            for metric in _SUMMARY_METRICS:
                values = [seed_means[str(seed)][metric] for seed in REPLICATION_SEEDS]
                replication[metric] = {
                    "mean_of_seed_cv_means": mean(values),
                    "sd_across_seed_cv_means": stdev(values),
                    "values": values,
                }
            summary[dataset][variant] = {
                "seed_cv_means": seed_means,
                "replication_summary": replication,
                "threshold_reliability_by_seed": {
                    str(seed): _oof_threshold_reliability(
                        records,
                        dataset=dataset,
                        variant=variant,
                        training_seed=seed,
                    )
                    for seed in REPLICATION_SEEDS
                },
                "classwise_reliability_by_seed": {
                    str(seed): _oof_classwise_reliability(
                        records,
                        dataset=dataset,
                        variant=variant,
                        training_seed=seed,
                    )
                    for seed in REPLICATION_SEEDS
                },
            }

    bootstrap = {
        dataset: {
            comparator: paired_fold_seed_cluster_bootstrap(
                records,
                dataset=dataset,
                comparator=comparator,
                samples=args.bootstrap_samples,
                seed=args.bootstrap_seed,
            )
            for comparator in BASELINE_ORDER
            if comparator != "origin_ctmc"
        }
        for dataset in DATASET_FOLDS
    }

    payload: dict[str, Any] = {
        "schema": "origin-acceptance-outer-aggregate-v2",
        "protocol_id": PROTOCOL_ID,
        "protocol_sha256": PROTOCOL_SHA256,
        "status": "complete",
        "release_worker_count": len(rows),
        "aggregation_unit": "fold_mean_with_replication_seed_as_repeat",
        "historical_origin_oof": {
            "role": "external_sanity_reference_only",
            "included_in_paired_contrasts": False,
            "included_in_model_selection": False,
            "note": "origin_ctmc is retrained in every registered fold/seed cell",
        },
        "comparator_implementation_scope": {
            "kind": "matched_in_repo_analogues",
            "official_author_implementations": False,
            "comparator_variants": [
                variant for variant in BASELINE_ORDER if variant != "origin_ctmc"
            ],
            "implementation_origin_by_variant": {
                variant: BASELINE_SPECS[variant]["implementation_origin"]
                for variant in BASELINE_ORDER
            },
            "claim_boundary": (
                "These are controlled matched analogues implemented in this repository, "
                "not results from official author implementations."
            ),
        },
        "posterior_quality_contract": {
            "multiclass_brier": "mean_sum_k_(p_k-onehot_k)^2",
            "threshold_binary_brier": (
                "per_boundary_mean_(P(Y>k)-1[Y>k])^2_then_unweighted_boundary_mean"
            ),
            "threshold_ece": (
                "15_equal_width_bins_per_boundary_then_unweighted_boundary_mean"
            ),
            "classwise_ece": (
                "15_equal_width_bins_one_vs_rest_per_class_then_unweighted_class_mean"
            ),
            "reliability_scope": "pooled_out_of_fold_predictions_per_training_seed",
        },
        "summary": summary,
        "paired_cluster_bootstrap": bootstrap,
    }
    payload["content_checksum_sha256"] = canonical_sha256(payload)
    write_json_atomic(output_dir / "aggregate.json", payload)
    with (output_dir / "fold_metrics.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    marker = {
        "schema": "origin-acceptance-outer-release-complete-v2",
        "protocol_sha256": PROTOCOL_SHA256,
        "aggregate_sha256": file_sha256(output_dir / "aggregate.json"),
        "fold_metrics_sha256": file_sha256(output_dir / "fold_metrics.csv"),
        "worker_count": len(rows),
    }
    write_json_atomic(output_dir / "OUTER_RELEASE_COMPLETE.json", marker)
    print(json.dumps({"status": "complete", "workers": len(rows), "path": str(output_dir)}))


if __name__ == "__main__":
    main()
