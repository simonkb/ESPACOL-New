#!/usr/bin/env python3
"""Read-only audit for acquisition-format shortcuts in MOSAIC preprocessing.

The audit fits deliberately weak lookup baselines on the *training* partition
and reports their performance on the inner validation partition.  By default
it never opens or evaluates an outer-test image.  ``--include_test`` must be
provided explicitly to inspect that split after architecture selection.

Two shortcut channels are audited:

1. raw source width/height, before any preprocessing; and
2. the number of valid cells in MOSAIC's transformed proof lattice.

The second channel should be constant under
``canonical-square-fixed-ellipse-v1`` and therefore reproduce only the
training-set majority classifier.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Callable, Hashable, Sequence

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from Datasets.mosaic_data import (  # noqa: E402
    MOSAIC_PREPROCESSING_VERSION,
    Item,
    MosaicFundusTransform,
    aptos_fold,
    eyepacs_fold,
    load_aptos_items,
    load_eyepacs_items,
)
from models.local_efficientnet import downsample_retinal_field_mask  # noqa: E402


def _majority(labels: Sequence[int]) -> int:
    counts = Counter(int(label) for label in labels)
    if not counts:
        raise ValueError("cannot fit a lookup on an empty training split")
    # Stable tie break toward the less severe grade.
    return min(counts, key=lambda label: (-counts[label], label))


def _fit_lookup(
    keys: Sequence[Hashable], labels: Sequence[int]
) -> tuple[dict[Hashable, int], int]:
    grouped: dict[Hashable, list[int]] = defaultdict(list)
    for key, label in zip(keys, labels):
        grouped[key].append(int(label))
    return {key: _majority(group) for key, group in grouped.items()}, _majority(labels)


def _lookup_predict(
    mapping: dict[Hashable, int], fallback: int, keys: Sequence[Hashable]
) -> np.ndarray:
    return np.asarray([mapping.get(key, fallback) for key in keys], dtype=np.int64)


def _metrics(labels: Sequence[int], predictions: Sequence[int], classes: int = 5) -> dict:
    truth = np.asarray(labels, dtype=np.int64)
    pred = np.asarray(predictions, dtype=np.int64)
    if truth.shape != pred.shape or truth.ndim != 1 or truth.size == 0:
        raise ValueError("truth and predictions must be equally sized non-empty vectors")

    confusion = np.zeros((classes, classes), dtype=np.int64)
    np.add.at(confusion, (truth, pred), 1)
    support = confusion.sum(axis=1)
    predicted = confusion.sum(axis=0)
    diagonal = np.diag(confusion).astype(np.float64)
    recall = np.divide(diagonal, support, out=np.zeros(classes), where=support > 0)
    precision = np.divide(
        diagonal, predicted, out=np.zeros(classes), where=predicted > 0
    )
    f1 = np.divide(
        2.0 * precision * recall,
        precision + recall,
        out=np.zeros(classes),
        where=(precision + recall) > 0,
    )

    grid = np.arange(classes, dtype=np.float64)
    weights = ((grid[:, None] - grid[None, :]) / max(classes - 1, 1)) ** 2
    expected = np.outer(support, predicted) / truth.size
    denominator = float((weights * expected).sum())
    qwk = 0.0 if denominator == 0.0 else 1.0 - float((weights * confusion).sum()) / denominator
    return {
        "n": int(truth.size),
        "accuracy": float((truth == pred).mean()),
        "balanced_accuracy": float(recall[support > 0].mean()),
        "macro_f1": float(f1[support > 0].mean()),
        "mae": float(np.abs(truth - pred).mean()),
        "qwk": float(qwk),
        "per_grade_recall": recall.tolist(),
        "confusion": confusion.tolist(),
    }


def _source_dimensions(items: Sequence[Item]) -> list[tuple[int, int]]:
    dimensions: list[tuple[int, int]] = []
    for path, _ in items:
        # PIL reads only image headers for ``size``; image pixels are not decoded.
        with Image.open(path) as image:
            dimensions.append((int(image.width), int(image.height)))
    return dimensions


def _dimension_summary(dimensions: Sequence[tuple[int, int]]) -> dict:
    counts = Counter(dimensions)
    widths = np.asarray([width for width, _ in dimensions], dtype=np.int64)
    heights = np.asarray([height for _, height in dimensions], dtype=np.int64)
    return {
        "n": len(dimensions),
        "unique_width_height_pairs": len(counts),
        "square_fraction": float(np.mean(widths == heights)),
        "width_min_median_max": [
            int(widths.min()),
            float(np.median(widths)),
            int(widths.max()),
        ],
        "height_min_median_max": [
            int(heights.min()),
            float(np.median(heights)),
            int(heights.max()),
        ],
        "most_common_width_height": [
            {"width": int(key[0]), "height": int(key[1]), "count": int(count)}
            for key, count in counts.most_common(10)
        ],
    }


def _mask_count(image_size: int, output_stride: int) -> tuple[int, int, int]:
    transform = MosaicFundusTransform(image_size, augment=False)
    pixel_mask = transform.canonical_valid_mask().unsqueeze(0)
    lattice_size = (
        math.ceil(image_size / output_stride),
        math.ceil(image_size / output_stride),
    )
    lattice_mask = downsample_retinal_field_mask(pixel_mask, lattice_size)
    return int(pixel_mask.sum()), int(lattice_mask.sum()), int(lattice_mask.numel())


def _evaluate_lookup(
    train_items: Sequence[Item],
    target_items: Sequence[Item],
    train_keys: Sequence[Hashable],
    target_keys: Sequence[Hashable],
) -> dict:
    train_labels = [int(label) for _, label in train_items]
    target_labels = [int(label) for _, label in target_items]
    lookup, fallback = _fit_lookup(train_keys, train_labels)
    predictions = _lookup_predict(lookup, fallback, target_keys)
    return {
        "training_unique_keys": len(set(train_keys)),
        "target_seen_key_fraction": float(
            np.mean([key in lookup for key in target_keys])
        ),
        "fallback_grade": int(fallback),
        "metrics": _metrics(target_labels, predictions),
    }


def _audit_fold(
    *,
    fold: int,
    items: Sequence[Item],
    split_fn: Callable,
    n_folds: int,
    seed: int,
    image_size: int,
    output_stride: int,
    include_test: bool,
) -> dict:
    train_items, validation_items, test_items = split_fn(
        items,
        fold,
        n_folds=n_folds,
        val_fraction=0.1,
        seed=seed,
    )
    train_dimensions = _source_dimensions(train_items)
    validation_dimensions = _source_dimensions(validation_items)
    pixel_count, lattice_count, lattice_capacity = _mask_count(
        image_size, output_stride
    )

    train_mask_keys = [lattice_count] * len(train_items)
    validation_mask_keys = [lattice_count] * len(validation_items)
    report = {
        "fold": int(fold),
        "counts": {
            "train": len(train_items),
            "validation": len(validation_items),
            "outer_test": len(test_items),
        },
        "source_dimensions": {
            "train": _dimension_summary(train_dimensions),
            "validation": _dimension_summary(validation_dimensions),
            "exact_width_height_lookup_validation": _evaluate_lookup(
                train_items,
                validation_items,
                train_dimensions,
                validation_dimensions,
            ),
        },
        "transformed_mask_count": {
            "image_independent": True,
            "pixel_valid_count": pixel_count,
            "lattice_output_stride": int(output_stride),
            "lattice_valid_count": lattice_count,
            "lattice_capacity": lattice_capacity,
            "unique_counts_all_grades": [lattice_count],
            "lookup_validation": _evaluate_lookup(
                train_items,
                validation_items,
                train_mask_keys,
                validation_mask_keys,
            ),
        },
        "outer_test": {
            "evaluated": False,
            "image_reads": 0,
            "reason": "disabled by default; pass --include_test explicitly",
        },
    }

    if include_test:
        test_dimensions = _source_dimensions(test_items)
        report["source_dimensions"]["outer_test"] = _dimension_summary(
            test_dimensions
        )
        report["source_dimensions"]["exact_width_height_lookup_outer_test"] = (
            _evaluate_lookup(
                train_items, test_items, train_dimensions, test_dimensions
            )
        )
        report["transformed_mask_count"]["lookup_outer_test"] = _evaluate_lookup(
            train_items,
            test_items,
            train_mask_keys,
            [lattice_count] * len(test_items),
        )
        report["outer_test"] = {"evaluated": True, "image_reads": len(test_items)}
    return report


def _parse_folds(value: str, n_folds: int) -> list[int]:
    if value.lower() == "all":
        return list(range(n_folds))
    result = [int(part.strip()) for part in value.split(",") if part.strip()]
    if not result or len(result) != len(set(result)):
        raise ValueError("folds must be a non-empty comma-separated unique list")
    if any(fold < 0 or fold >= n_folds for fold in result):
        raise ValueError(f"folds must lie in [0, {n_folds - 1}]")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit raw geometry and transformed mask-count shortcuts",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dataset", choices=("aptos", "dr"), default="aptos")
    parser.add_argument("--data_root", default=None)
    parser.add_argument("--labels_csv", default=None)
    parser.add_argument("--folds", default="0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--image_size", type=int, default=896)
    parser.add_argument("--output_stride", type=int, default=8)
    parser.add_argument(
        "--json_out",
        default=None,
        help="Optional path for an atomic JSON copy of the printed report",
    )
    parser.add_argument(
        "--include_test",
        action="store_true",
        help="explicitly permit reading/evaluating outer-test image headers",
    )
    args = parser.parse_args()

    n_folds = 5 if args.dataset == "aptos" else 10
    root = args.data_root or (
        "Datasets/aptos2019-blindness-detection"
        if args.dataset == "aptos"
        else "Datasets/DR"
    )
    if args.dataset == "aptos":
        items = load_aptos_items(root, args.labels_csv)
        split_fn = aptos_fold
    else:
        items = load_eyepacs_items(root, args.labels_csv)
        split_fn = eyepacs_fold

    result = {
        "audit": "mosaic-acquisition-shortcuts-v1",
        "dataset": args.dataset,
        "preprocessing_version": MOSAIC_PREPROCESSING_VERSION,
        "image_size": int(args.image_size),
        "outer_test_opt_in": bool(args.include_test),
        "folds": [
            _audit_fold(
                fold=fold,
                items=items,
                split_fn=split_fn,
                n_folds=n_folds,
                seed=args.seed,
                image_size=args.image_size,
                output_stride=args.output_stride,
                include_test=args.include_test,
            )
            for fold in _parse_folds(args.folds, n_folds)
        ],
    }
    rendered = json.dumps(result, indent=2)
    print(rendered)
    if args.json_out:
        destination = Path(args.json_out)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
        try:
            temporary.write_text(rendered + "\n")
            os.replace(temporary, destination)
        finally:
            if temporary.exists():
                temporary.unlink()


if __name__ == "__main__":
    main()
