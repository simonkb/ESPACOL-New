#!/usr/bin/env python3
"""Audit and aggregate the coordinated outer release without reselection."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean, stdev
from typing import Any

import numpy as np

from scripts.origin_acceptance_baseline_common import (
    BASELINE_ORDER,
    DATASET_FOLDS,
    PROTOCOL_ID,
    PROTOCOL_SHA256,
    REPLICATION_SEEDS,
    canonical_sha256,
    file_sha256,
    full_tasks,
)
from train_origin import write_json_atomic


_METRICS = ("acc", "qwk", "mae", "balanced_acc", "macro_f1", "ece", "nll", "rps")


def _load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise TypeError(f"expected object at {path}")
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment_root", required=True)
    args = parser.parse_args()
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
    identity_by_fold: dict[tuple[str, int], tuple[tuple[str, ...], tuple[int, ...]]] = {}
    for task in full_tasks():
        directory = root / "full" / "outer_release" / task.key
        metrics_path = directory / "outer_metrics.json"
        predictions_path = directory / "outer_predictions.npz"
        complete_path = directory / "OUTER_COMPLETE.json"
        if not all(path.is_file() for path in (metrics_path, predictions_path, complete_path)):
            raise FileNotFoundError(f"incomplete outer release: {directory}")
        payload = _load_json(metrics_path)
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
        complete = _load_json(complete_path)
        if complete.get("outer_metrics_sha256") != file_sha256(metrics_path):
            raise ValueError(f"completion hash mismatch: {complete_path}")
        with np.load(predictions_path, allow_pickle=False) as archive:
            paths = tuple(str(value) for value in archive["image_path"].tolist())
            labels = tuple(int(value) for value in archive["label"].tolist())
        fold_key = (task.dataset, task.fold)
        identity = (paths, labels)
        previous = identity_by_fold.setdefault(fold_key, identity)
        if previous != identity:
            raise ValueError(f"outer item/order mismatch across comparators for {fold_key}")
        metrics = payload["metrics"]
        rows.append(
            {
                "dataset": task.dataset,
                "fold": task.fold,
                "baseline_variant": task.baseline_variant,
                "training_seed": task.training_seed,
                **{name: float(metrics[name]) for name in _METRICS},
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
                    for metric in _METRICS
                }
            replication = {}
            for metric in _METRICS:
                values = [seed_means[str(seed)][metric] for seed in REPLICATION_SEEDS]
                replication[metric] = {
                    "mean_of_seed_cv_means": mean(values),
                    "sd_across_seed_cv_means": stdev(values),
                    "values": values,
                }
            summary[dataset][variant] = {
                "seed_cv_means": seed_means,
                "replication_summary": replication,
            }

    payload: dict[str, Any] = {
        "schema": "origin-acceptance-outer-aggregate-v1",
        "protocol_id": PROTOCOL_ID,
        "protocol_sha256": PROTOCOL_SHA256,
        "status": "complete",
        "release_worker_count": len(rows),
        "aggregation_unit": "fold_mean_with_replication_seed_as_repeat",
        "summary": summary,
    }
    payload["content_checksum_sha256"] = canonical_sha256(payload)
    write_json_atomic(output_dir / "aggregate.json", payload)
    with (output_dir / "fold_metrics.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    marker = {
        "schema": "origin-acceptance-outer-release-complete-v1",
        "protocol_sha256": PROTOCOL_SHA256,
        "aggregate_sha256": file_sha256(output_dir / "aggregate.json"),
        "fold_metrics_sha256": file_sha256(output_dir / "fold_metrics.csv"),
        "worker_count": len(rows),
    }
    write_json_atomic(output_dir / "OUTER_RELEASE_COMPLETE.json", marker)
    print(json.dumps({"status": "complete", "workers": len(rows), "path": str(output_dir)}))


if __name__ == "__main__":
    main()
