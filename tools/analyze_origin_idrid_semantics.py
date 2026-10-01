#!/usr/bin/env python3
"""Checksum-sealed, image-cluster inference for the ORIGIN IDRiD audit.

Ten independently selected EyePACS fold checkpoints evaluate the same 81 IDRiD
images.  This postprocessor first averages checkpoints *within each image* and
only then performs image-cluster inference.  It therefore never treats repeated
checkpoint evaluations as independent samples.  All deletion effects replay an
already stored internal rate ledger; they are explicitly not causal pixel
interventions.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.origin_v3_cv_common import (
    canonical_sha256,
    file_sha256,
    read_json,
    verify_checksummed_payload,
    write_json_atomic,
)


PROTOCOL_SCHEMA = "origin-idrid-semantic-statistics-protocol-v1"
ALIGNMENT_SCHEMA = "origin-idrid-semantic-alignment-statistics-v1"
DELETION_SCHEMA = "origin-idrid-semantic-deletion-statistics-v1"
UNIT_SCHEMA = "origin-idrid-semantic-image-unit-v1"
MANIFEST_SCHEMA = "origin-idrid-semantic-statistics-manifest-v1"
INPUT_SCHEMA = "origin-idrid-semantic-audit-v1"

SCALES = ("s4", "s8", "s16", "s32")
SCALE_GROUPS = {
    "overall": SCALES,
    "fine": ("s4", "s8"),
    "coarse": ("s16", "s32"),
}
LESIONS = ("union", "microaneurysm", "haemorrhage", "hard_exudate", "soft_exudate")
DELETION_GROUPS = ("all", "fine", "coarse")


class StatisticsConfig:
    def __init__(
        self,
        *,
        bootstrap_samples: int,
        bootstrap_seed: int,
        confidence_level: float,
        permutation_samples: int,
        permutation_seed: int,
        exact_sign_flip_max_n: int,
        zero_tolerance: float,
    ) -> None:
        self.bootstrap_samples = int(bootstrap_samples)
        self.bootstrap_seed = int(bootstrap_seed)
        self.confidence_level = float(confidence_level)
        self.permutation_samples = int(permutation_samples)
        self.permutation_seed = int(permutation_seed)
        self.exact_sign_flip_max_n = int(exact_sign_flip_max_n)
        self.zero_tolerance = float(zero_tolerance)
        if self.bootstrap_samples < 1000:
            raise ValueError("at least 1000 cluster bootstrap replicates are required")
        if not 0.0 < self.confidence_level < 1.0:
            raise ValueError("confidence level must lie in (0,1)")
        if self.permutation_samples < 999:
            raise ValueError("at least 999 sign-flip samples are required")
        if not 0 <= self.exact_sign_flip_max_n <= 20:
            raise ValueError("exact sign-flip threshold must lie in [0,20]")
        if not 0.0 <= self.zero_tolerance < 1e-3:
            raise ValueError("zero tolerance is invalid")


def _stable_seed(seed: int, *parts: Any) -> int:
    message = ":".join((str(seed), *(str(part) for part in parts)))
    return int.from_bytes(hashlib.sha256(message.encode()).digest()[:8], "little")


def load_protocol(path: str | Path) -> tuple[dict[str, Any], StatisticsConfig]:
    payload = read_json(path)
    verify_checksummed_payload(payload, schema=PROTOCOL_SCHEMA)
    if payload.get("inference_unit") != "idrid_image_after_within_image_checkpoint_aggregation":
        raise ValueError("IDRiD inference unit changed")
    if payload.get("interpretation_scope") != "stored_internal_ledger_not_causal_pixel_intervention":
        raise ValueError("IDRiD interpretation scope changed")
    config = payload.get("config")
    if not isinstance(config, Mapping):
        raise TypeError("IDRiD statistics config is missing")
    result = StatisticsConfig(
        bootstrap_samples=config["bootstrap_samples"],
        bootstrap_seed=config["bootstrap_seed"],
        confidence_level=config["confidence_level"],
        permutation_samples=config["permutation_samples"],
        permutation_seed=config["permutation_seed"],
        exact_sign_flip_max_n=config["exact_sign_flip_max_n"],
        zero_tolerance=config["zero_tolerance"],
    )
    return payload, result


def _finite(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _mean(values: Iterable[Any]) -> float | None:
    finite = [value for item in values if (value := _finite(item)) is not None]
    return None if not finite else float(np.mean(finite))


def _percentile_interval(values: np.ndarray, confidence: float) -> list[float]:
    alpha = (1.0 - confidence) / 2.0
    return [float(value) for value in np.quantile(values, [alpha, 1.0 - alpha])]


def cluster_bootstrap_mean(
    values: Sequence[float],
    *,
    samples: int,
    seed: int,
    confidence: float,
) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or len(array) < 1 or not np.all(np.isfinite(array)):
        raise ValueError("bootstrap values must be a non-empty finite vector")
    rng = np.random.default_rng(seed)
    draws = np.empty(samples, dtype=np.float64)
    for start in range(0, samples, 512):
        size = min(512, samples - start)
        indices = rng.integers(0, len(array), size=(size, len(array)))
        draws[start : start + size] = array[indices].mean(axis=1)
    return {
        "estimate": float(array.mean()),
        "bootstrap_mean": float(draws.mean()),
        "ci_percentile": _percentile_interval(draws, confidence),
        "n_image_clusters": len(array),
        "bootstrap_samples": samples,
        "bootstrap_seed_derived": int(seed),
    }


def exact_two_sided_sign_test(values: Sequence[float], *, zero_tolerance: float) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    nonzero = array[np.abs(array) > zero_tolerance]
    positive = int(np.sum(nonzero > 0))
    negative = int(np.sum(nonzero < 0))
    n = positive + negative
    if n == 0:
        p_value = 1.0
    else:
        lower = min(positive, negative)
        tail = sum(math.comb(n, index) for index in range(lower + 1)) / (2.0 ** n)
        p_value = min(1.0, 2.0 * tail)
    return {
        "method": "exact_two_sided_binomial_sign_test",
        "null": "equal_probability_of_positive_and_negative_nonzero_differences",
        "positive": positive,
        "negative": negative,
        "ties_excluded": int(len(array) - n),
        "p_value": float(p_value),
    }


def paired_sign_flip_test(
    values: Sequence[float],
    *,
    samples: int,
    seed: int,
    exact_max_n: int,
    zero_tolerance: float,
) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    array = array[np.abs(array) > zero_tolerance]
    n = len(array)
    observed = 0.0 if n == 0 else float(array.mean())
    if n == 0:
        return {
            "method": "degenerate_all_ties",
            "n_nonzero_pairs": 0,
            "observed_mean_difference": 0.0,
            "two_sided_p_value": 1.0,
            "samples": 0,
            "seed": None,
            "assumption": "none",
        }
    threshold = abs(observed) - 1e-15
    extreme = 0
    if n <= exact_max_n:
        total = 1 << n
        bit_positions = np.arange(n, dtype=np.uint64)
        for start in range(0, total, 8192):
            states = np.arange(start, min(total, start + 8192), dtype=np.uint64)[:, None]
            signs = 1.0 - 2.0 * ((states >> bit_positions[None]) & 1).astype(np.float64)
            extreme += int(np.sum(np.abs(signs @ array / n) >= threshold))
        p_value = extreme / total
        method = "exact_paired_sign_flip"
        reported_samples = total
        reported_seed: int | None = None
    else:
        rng = np.random.default_rng(seed)
        for start in range(0, samples, 4096):
            size = min(4096, samples - start)
            signs = rng.integers(0, 2, size=(size, n), dtype=np.int8) * 2 - 1
            extreme += int(np.sum(np.abs(signs @ array / n) >= threshold))
        p_value = (extreme + 1.0) / (samples + 1.0)
        method = "monte_carlo_paired_sign_flip"
        reported_samples = samples
        reported_seed = int(seed)
    return {
        "method": method,
        "n_nonzero_pairs": n,
        "observed_mean_difference": observed,
        "two_sided_p_value": float(p_value),
        "samples": reported_samples,
        "seed": reported_seed,
        "assumption": "exchangeable_difference_signs_under_a_symmetric_null",
    }


def _identity(record: Mapping[str, Any]) -> tuple[int, str]:
    return int(record["fold"]), str(record["image_id"])


def validate_input_records(
    summary: Mapping[str, Any],
    records: Sequence[Mapping[str, Any]],
    *,
    expected_folds: Sequence[int],
    expected_images: int,
    expected_random_repeats: int,
) -> dict[str, Any]:
    if summary.get("schema") != INPUT_SCHEMA:
        raise ValueError("unexpected IDRiD semantic input schema")
    if summary.get("scope") != "external_zero_shot_cell_bin_alignment_and_internal_ledger_deletion":
        raise ValueError("unexpected IDRiD semantic input scope")
    if int(summary.get("idrid_images", -1)) != expected_images:
        raise AssertionError("IDRiD image count changed")
    if int(summary.get("random_repeats", -1)) != expected_random_repeats:
        raise AssertionError("random-control repeat count changed")
    checkpoints = summary.get("checkpoints")
    if not isinstance(checkpoints, list):
        raise TypeError("checkpoint metadata are missing")
    checkpoint_folds = [int(item["fold"]) for item in checkpoints]
    if sorted(checkpoint_folds) != sorted(int(value) for value in expected_folds):
        raise AssertionError("checkpoint fold set changed")
    if len({str(item["checkpoint_sha256"]) for item in checkpoints}) != len(checkpoints):
        raise AssertionError("checkpoint hashes are not unique")

    alignment_keys: set[tuple[Any, ...]] = set()
    deletion_keys: set[tuple[Any, ...]] = set()
    image_metadata: dict[str, tuple[int, str, int]] = {}
    prevalence: dict[tuple[str, str, str], float] = {}
    deleted_counts: dict[tuple[str, str], int] = {}
    for record in records:
        kind = record.get("kind")
        fold, image = _identity(record)
        if fold not in expected_folds:
            raise AssertionError("record references an unexpected checkpoint fold")
        grade, split = int(record["grade"]), str(record["split"])
        target_boundary = int(record.get("target_boundary", min(max(grade - 1, 0), 3)))
        expected_target = min(max(grade - 1, 0), 3)
        if target_boundary != expected_target:
            raise AssertionError("target boundary no longer follows the declared grade rule")
        metadata = (grade, split, target_boundary)
        if image in image_metadata and image_metadata[image] != metadata:
            raise AssertionError("image grade/split metadata differ across checkpoints")
        image_metadata[image] = metadata
        if kind == "alignment":
            scale, lesion, boundary = (
                str(record["scale"]), str(record["lesion"]), int(record["boundary"])
            )
            if scale not in SCALES or lesion not in LESIONS or boundary not in range(4):
                raise AssertionError("invalid alignment scale, lesion, or boundary")
            expected_group = "fine" if scale in SCALE_GROUPS["fine"] else "coarse"
            if record.get("scale_group") != expected_group:
                raise AssertionError("alignment scale-group label changed")
            key = (fold, image, scale, lesion, boundary)
            if key in alignment_keys:
                raise AssertionError("duplicate raw alignment record")
            alignment_keys.add(key)
            value = _finite(record.get("prevalence"))
            if value is None or not 0.0 <= value <= 1.0:
                raise AssertionError("invalid lesion prevalence")
            ap = _finite(record.get("average_precision"))
            enrichment = _finite(record.get("ap_over_prevalence"))
            if ap is not None:
                if not 0.0 < value < 1.0 or not 0.0 <= ap <= 1.0:
                    raise AssertionError("defined AP requires both label classes")
                if enrichment is None or not math.isclose(
                    enrichment, ap / value, rel_tol=1e-6, abs_tol=1e-8
                ):
                    raise AssertionError("AP/prevalence enrichment does not replay")
            elif enrichment is not None:
                raise AssertionError("undefined AP cannot have a defined enrichment")
            prevalence_key = (image, scale, lesion)
            if prevalence_key in prevalence and not math.isclose(
                prevalence[prevalence_key], value, rel_tol=0.0, abs_tol=1e-12
            ):
                raise AssertionError("mask prevalence changed across folds or boundaries")
            prevalence[prevalence_key] = value
        elif kind == "external_mask_deletion":
            group = str(record["scale_group"])
            if group not in DELETION_GROUPS:
                raise AssertionError("invalid deletion scale group")
            key = (fold, image, group)
            if key in deletion_keys:
                raise AssertionError("duplicate raw deletion record")
            deletion_keys.add(key)
            count = int(record["deleted_cells"])
            count_key = (image, group)
            if count_key in deleted_counts and deleted_counts[count_key] != count:
                raise AssertionError("deleted lesion-cell count changed across checkpoints")
            deleted_counts[count_key] = count
            expected_lift = _finite(record.get("expected_grade_delta_lift"))
            tail_lift = _finite(record.get("target_tail_delta_lift"))
            expected_left = _finite(record.get("expected_grade_delta"))
            expected_right = _finite(record.get("random_expected_grade_delta_mean"))
            tail_left = _finite(record.get("target_tail_delta"))
            tail_right = _finite(record.get("random_target_tail_delta_mean"))
            if None in (expected_lift, tail_lift, expected_left, expected_right, tail_left, tail_right):
                raise AssertionError("deletion effects must be finite")
            if not math.isclose(expected_lift, expected_left - expected_right, abs_tol=2e-6):
                raise AssertionError("expected-grade lift does not replay")
            if not math.isclose(tail_lift, tail_left - tail_right, abs_tol=2e-6):
                raise AssertionError("tail lift does not replay")
            if int(record.get("random_repeats", -1)) != expected_random_repeats:
                raise AssertionError("deletion record random-repeat count changed")
        else:
            raise AssertionError(f"unknown semantic record kind: {kind!r}")
    images = sorted(image_metadata)
    if len(images) != expected_images:
        raise AssertionError("semantic records do not cover exactly 81 unique images")
    expected_alignment = len(expected_folds) * expected_images * len(SCALES) * len(LESIONS) * 4
    expected_deletion = len(expected_folds) * expected_images * len(DELETION_GROUPS)
    if len(alignment_keys) != expected_alignment or len(deletion_keys) != expected_deletion:
        raise AssertionError(
            "semantic row census is incomplete: "
            f"alignment={len(alignment_keys)}/{expected_alignment}, "
            f"deletion={len(deletion_keys)}/{expected_deletion}"
        )
    split_counts: dict[str, int] = defaultdict(int)
    for _, split, _ in image_metadata.values():
        split_counts[split] += 1
    return {
        "images": images,
        "image_metadata": image_metadata,
        "alignment_rows": len(alignment_keys),
        "deletion_rows": len(deletion_keys),
        "split_counts": dict(sorted(split_counts.items())),
    }


def aggregate_alignment_image_units(
    records: Sequence[Mapping[str, Any]],
    *,
    scale_group: str,
    lesion: str,
    boundary: int | None,
    grade: int | None = None,
    split: str | None = None,
    expected_folds: Sequence[int] = tuple(range(10)),
) -> list[dict[str, Any]]:
    if scale_group not in SCALE_GROUPS or lesion not in LESIONS:
        raise ValueError("unknown alignment scale group or lesion")
    scales = set(SCALE_GROUPS[scale_group])
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for record in records:
        if record.get("kind") != "alignment" or record.get("lesion") != lesion:
            continue
        if record.get("scale") not in scales:
            continue
        if boundary is None:
            if int(record["boundary"]) != int(record["target_boundary"]):
                continue
        elif int(record["boundary"]) != boundary:
            continue
        if grade is not None and int(record["grade"]) != grade:
            continue
        if split is not None and str(record["split"]) != split:
            continue
        grouped[str(record["image_id"])].append(record)
    units: list[dict[str, Any]] = []
    for image, image_records in sorted(grouped.items()):
        by_fold: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
        for record in image_records:
            by_fold[int(record["fold"])].append(record)
        if set(by_fold) != set(expected_folds):
            raise AssertionError("alignment image unit lacks all checkpoint folds")
        if any(len(rows) != len(scales) for rows in by_fold.values()):
            raise AssertionError("alignment fold does not contain every requested scale")
        fold_metrics: list[dict[str, Any]] = []
        for fold in expected_folds:
            rows = by_fold[int(fold)]
            evaluable = [
                row for row in rows
                if _finite(row.get("average_precision")) is not None
                and _finite(row.get("prevalence")) is not None
            ]
            # Chance AP is prevalence on the same binary ranking problem, so
            # prevalence is averaged only over scales on which AP is defined.
            ap = _mean(row.get("average_precision") for row in evaluable)
            prevalence = _mean(row.get("prevalence") for row in evaluable)
            ratio = _mean(row.get("ap_over_prevalence") for row in evaluable)
            rate_ratio = _mean(row.get("mean_rate_ratio") for row in rows)
            differences = [
                float(row["average_precision"]) - float(row["prevalence"])
                for row in evaluable
            ]
            fold_metrics.append(
                {
                    "average_precision": ap,
                    "prevalence": prevalence,
                    "mean_ap_over_prevalence": ratio,
                    "ap_minus_prevalence": _mean(differences),
                    "mean_rate_ratio": rate_ratio,
                }
            )
        first = image_records[0]
        image_ap = _mean(item["average_precision"] for item in fold_metrics)
        image_prevalence = _mean(item["prevalence"] for item in fold_metrics)
        units.append(
            {
                "schema": UNIT_SCHEMA,
                "row_type": "alignment_image_unit",
                "image_id": image,
                "grade": int(first["grade"]),
                "split": str(first["split"]),
                "target_boundary": int(first["target_boundary"]),
                "scale_group": scale_group,
                "lesion": lesion,
                "boundary_selection": "target" if boundary is None else f"Y>{boundary}",
                "checkpoints_aggregated": len(expected_folds),
                "scales_aggregated": len(scales),
                "average_precision": image_ap,
                "prevalence": image_prevalence,
                "mean_ap_over_prevalence": _mean(
                    item["mean_ap_over_prevalence"] for item in fold_metrics
                ),
                "ratio_of_mean_ap_to_mean_prevalence": (
                    None
                    if image_ap is None or image_prevalence is None or image_prevalence <= 0
                    else image_ap / image_prevalence
                ),
                "ap_minus_prevalence": _mean(
                    item["ap_minus_prevalence"] for item in fold_metrics
                ),
                "mean_rate_ratio": _mean(item["mean_rate_ratio"] for item in fold_metrics),
            }
        )
    return units


def aggregate_deletion_image_units(
    records: Sequence[Mapping[str, Any]],
    *,
    scale_group: str,
    grade: int | None = None,
    target_boundary: int | None = None,
    split: str | None = None,
    expected_folds: Sequence[int] = tuple(range(10)),
) -> list[dict[str, Any]]:
    raw_group = "all" if scale_group == "overall" else scale_group
    if raw_group not in DELETION_GROUPS:
        raise ValueError("unknown deletion scale group")
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for record in records:
        if record.get("kind") != "external_mask_deletion":
            continue
        if record.get("scale_group") != raw_group:
            continue
        if grade is not None and int(record["grade"]) != grade:
            continue
        if target_boundary is not None and int(record["target_boundary"]) != target_boundary:
            continue
        if split is not None and str(record["split"]) != split:
            continue
        grouped[str(record["image_id"])].append(record)
    units: list[dict[str, Any]] = []
    for image, rows in sorted(grouped.items()):
        if {int(row["fold"]) for row in rows} != set(expected_folds) or len(rows) != len(expected_folds):
            raise AssertionError("deletion image unit lacks exactly one row per checkpoint")
        first = rows[0]
        guided_expected = _mean(row["expected_grade_delta"] for row in rows)
        random_expected = _mean(row["random_expected_grade_delta_mean"] for row in rows)
        guided_tail = _mean(row["target_tail_delta"] for row in rows)
        random_tail = _mean(row["random_target_tail_delta_mean"] for row in rows)
        expected_lift = _mean(row["expected_grade_delta_lift"] for row in rows)
        tail_lift = _mean(row["target_tail_delta_lift"] for row in rows)
        if None in (
            guided_expected, random_expected, guided_tail, random_tail,
            expected_lift, tail_lift,
        ):
            raise AssertionError("fold-aggregated deletion effect is non-finite")
        if not math.isclose(expected_lift, guided_expected - random_expected, abs_tol=2e-6):
            raise AssertionError("fold-aggregated expected lift does not replay")
        if not math.isclose(tail_lift, guided_tail - random_tail, abs_tol=2e-6):
            raise AssertionError("fold-aggregated tail lift does not replay")
        units.append(
            {
                "schema": UNIT_SCHEMA,
                "row_type": "deletion_image_unit",
                "image_id": image,
                "grade": int(first["grade"]),
                "split": str(first["split"]),
                "target_boundary": int(first["target_boundary"]),
                "scale_group": scale_group,
                "checkpoints_aggregated": len(expected_folds),
                "deleted_cells": int(first["deleted_cells"]),
                "guided_expected_grade_delta": guided_expected,
                "random_expected_grade_delta": random_expected,
                "expected_grade_delta_lift": expected_lift,
                "guided_target_tail_delta": guided_tail,
                "random_target_tail_delta": random_tail,
                "target_tail_delta_lift": tail_lift,
            }
        )
    return units


def _metric_inference(
    units: Sequence[Mapping[str, Any]],
    metric: str,
    *,
    config: StatisticsConfig,
    seed_label: str,
) -> dict[str, Any] | None:
    values = [value for unit in units if (value := _finite(unit.get(metric))) is not None]
    if not values:
        return None
    return cluster_bootstrap_mean(
        values,
        samples=config.bootstrap_samples,
        seed=_stable_seed(config.bootstrap_seed, seed_label, metric),
        confidence=config.confidence_level,
    )


def _paired_tests(
    differences: Sequence[float], *, config: StatisticsConfig, seed_label: str
) -> dict[str, Any]:
    return {
        "sign_test": exact_two_sided_sign_test(
            differences, zero_tolerance=config.zero_tolerance
        ),
        "sign_flip_test": paired_sign_flip_test(
            differences,
            samples=config.permutation_samples,
            seed=_stable_seed(config.permutation_seed, seed_label),
            exact_max_n=config.exact_sign_flip_max_n,
            zero_tolerance=config.zero_tolerance,
        ),
    }


def _alignment_specs(records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grades = sorted({int(row["grade"]) for row in records if row.get("kind") == "alignment"})
    splits = sorted({str(row["split"]) for row in records if row.get("kind") == "alignment"})
    specs: dict[str, dict[str, Any]] = {}

    def add(**spec: Any) -> None:
        key = "|".join(f"{name}={spec[name]}" for name in sorted(spec) if name != "family")
        specs.setdefault(key, {"spec_id": key, **spec})

    for group in SCALE_GROUPS:
        add(family="primary_scale", scale_group=group, lesion="union", boundary=None)
        for boundary in range(4):
            add(
                family="boundary_stratum", scale_group=group,
                lesion="union", boundary=boundary,
            )
        for grade in grades:
            add(
                family="grade_stratum", scale_group=group, lesion="union",
                boundary=None, grade=grade,
            )
        for lesion in LESIONS[1:]:
            add(
                family="lesion_stratum", scale_group=group,
                lesion=lesion, boundary=None,
            )
    for split in splits:
        add(
            family="segmentation_split_stratum", scale_group="overall",
            lesion="union", boundary=None, split=split,
        )
    return list(specs.values())


def _deletion_specs(records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grades = sorted({int(row["grade"]) for row in records if row.get("kind") == "external_mask_deletion"})
    boundaries = sorted(
        {int(row["target_boundary"]) for row in records if row.get("kind") == "external_mask_deletion"}
    )
    splits = sorted({str(row["split"]) for row in records if row.get("kind") == "external_mask_deletion"})
    specs: dict[str, dict[str, Any]] = {}

    def add(**spec: Any) -> None:
        key = "|".join(f"{name}={spec[name]}" for name in sorted(spec) if name != "family")
        specs.setdefault(key, {"spec_id": key, **spec})

    for group in SCALE_GROUPS:
        add(family="primary_scale", scale_group=group)
        for grade in grades:
            add(family="grade_stratum", scale_group=group, grade=grade)
        for boundary in boundaries:
            add(
                family="target_boundary_stratum", scale_group=group,
                target_boundary=boundary,
            )
    for split in splits:
        add(family="segmentation_split_stratum", scale_group="overall", split=split)
    return list(specs.values())


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> int:
    temporary = path.with_name(f".{path.name}.tmp")
    count = 0
    try:
        with temporary.open("w", encoding="utf-8") as stream:
            for row in rows:
                stream.write(json.dumps(row, sort_keys=True, separators=(",", ":"), allow_nan=False))
                stream.write("\n")
                count += 1
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    return count


def analyze_idrid_semantics(
    *,
    summary_path: Path,
    protocol_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    if (output_dir / "idrid_statistics_manifest.json").exists():
        raise FileExistsError("refusing to overwrite completed IDRiD statistics")
    protocol, config = load_protocol(protocol_path)
    summary = read_json(summary_path)
    records_path = summary_path.parent / str(summary.get("records", ""))
    if not records_path.is_file():
        raise FileNotFoundError(records_path)
    try:
        records_path.resolve().relative_to(summary_path.parent.resolve())
    except ValueError:
        raise AssertionError("IDRiD summary points outside its audit directory") from None
    if file_sha256(records_path) != summary.get("records_sha256"):
        raise AssertionError("IDRiD semantic record checksum mismatch")
    records: list[dict[str, Any]] = []
    with records_path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid JSONL at line {line_number}") from error
            if not isinstance(row, dict):
                raise TypeError("IDRiD JSONL rows must be objects")
            records.append(row)
    expected_folds = tuple(int(value) for value in protocol["expected_checkpoint_folds"])
    census = validate_input_records(
        summary,
        records,
        expected_folds=expected_folds,
        expected_images=int(protocol["expected_images"]),
        expected_random_repeats=int(protocol["expected_random_control_repeats"]),
    )
    expected_split_counts = {
        str(key): int(value)
        for key, value in protocol["expected_segmentation_split_counts"].items()
    }
    if census["split_counts"] != expected_split_counts:
        raise AssertionError(
            "IDRiD segmentation split census changed: "
            f"observed={census['split_counts']}, expected={expected_split_counts}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    alignment_statistics: list[dict[str, Any]] = []
    deletion_statistics: list[dict[str, Any]] = []
    unit_rows: list[dict[str, Any]] = []
    for spec in _alignment_specs(records):
        units = aggregate_alignment_image_units(
            records,
            scale_group=spec["scale_group"],
            lesion=spec["lesion"],
            boundary=spec["boundary"],
            grade=spec.get("grade"),
            split=spec.get("split"),
            expected_folds=expected_folds,
        )
        for unit in units:
            unit_rows.append({"spec_id": spec["spec_id"], **unit})
        eligible = [
            unit for unit in units
            if _finite(unit.get("average_precision")) is not None
            and _finite(unit.get("prevalence")) is not None
            and _finite(unit.get("ap_minus_prevalence")) is not None
        ]
        metrics = {
            metric: _metric_inference(
                eligible, metric, config=config,
                seed_label=f"alignment:{spec['spec_id']}",
            )
            for metric in (
                "average_precision", "prevalence", "mean_ap_over_prevalence",
                "ratio_of_mean_ap_to_mean_prevalence", "ap_minus_prevalence",
                "mean_rate_ratio",
            )
        }
        differences = [float(unit["ap_minus_prevalence"]) for unit in eligible]
        alignment_statistics.append(
            {
                **spec,
                "n_images_in_stratum": len(units),
                "n_images_with_defined_ap": len(eligible),
                "metrics": metrics,
                "enrichment_tests_against_ap_equals_prevalence": (
                    None if not differences else _paired_tests(
                        differences, config=config,
                        seed_label=f"alignment:{spec['spec_id']}:enrichment",
                    )
                ),
            }
        )

    for spec in _deletion_specs(records):
        units = aggregate_deletion_image_units(
            records,
            scale_group=spec["scale_group"],
            grade=spec.get("grade"),
            target_boundary=spec.get("target_boundary"),
            split=spec.get("split"),
            expected_folds=expected_folds,
        )
        for unit in units:
            unit_rows.append({"spec_id": spec["spec_id"], **unit})
        eligible = [unit for unit in units if int(unit["deleted_cells"]) > 0]
        metrics = {
            metric: _metric_inference(
                eligible, metric, config=config,
                seed_label=f"deletion:{spec['spec_id']}",
            )
            for metric in (
                "guided_expected_grade_delta", "random_expected_grade_delta",
                "expected_grade_delta_lift", "guided_target_tail_delta",
                "random_target_tail_delta", "target_tail_delta_lift",
            )
        }
        expected_lifts = [float(unit["expected_grade_delta_lift"]) for unit in eligible]
        tail_lifts = [float(unit["target_tail_delta_lift"]) for unit in eligible]
        deletion_statistics.append(
            {
                **spec,
                "n_images_in_stratum": len(units),
                "n_images_with_nonzero_deleted_lesion_cells": len(eligible),
                "n_zero_cell_images_excluded": len(units) - len(eligible),
                "metrics": metrics,
                "expected_grade_lift_tests": (
                    None if not expected_lifts else _paired_tests(
                        expected_lifts, config=config,
                        seed_label=f"deletion:{spec['spec_id']}:expected",
                    )
                ),
                "target_tail_lift_tests": (
                    None if not tail_lifts else _paired_tests(
                        tail_lifts, config=config,
                        seed_label=f"deletion:{spec['spec_id']}:tail",
                    )
                ),
            }
        )

    common = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol_checksum_sha256": protocol["content_checksum_sha256"],
        "input_summary_path": str(summary_path.resolve()),
        "input_summary_sha256": file_sha256(summary_path),
        "input_records_path": str(records_path.resolve()),
        "input_records_sha256": file_sha256(records_path),
        "input_census": census,
        "inference_unit": "IDRiD image after averaging all 10 checkpoint evaluations",
        "cluster_limit": "IDRiD provides no patient linkage; the image is the available cluster",
        "interpretation_scope": "stored internal ledger intervention, not a causal pixel intervention",
    }
    alignment_payload: dict[str, Any] = {
        "schema": ALIGNMENT_SCHEMA,
        **common,
        "estimand": "image-weighted mean after within-image checkpoint and scale aggregation",
        "records": alignment_statistics,
        "multiplicity": "no adjustment; stratum intervals/tests are descriptive",
    }
    deletion_payload: dict[str, Any] = {
        "schema": DELETION_SCHEMA,
        **common,
        "estimand": "image-weighted paired guided-minus-random stored-ledger deletion effect",
        "records": deletion_statistics,
        "multiplicity": "no adjustment; stratum intervals/tests are descriptive",
    }
    for payload in (alignment_payload, deletion_payload):
        payload["content_checksum_sha256"] = canonical_sha256(payload)
    alignment_path = output_dir / "IDRID_ALIGNMENT_CLUSTER_STATISTICS.json"
    deletion_path = output_dir / "IDRID_DELETION_CLUSTER_STATISTICS.json"
    unit_path = output_dir / "IDRID_IMAGE_LEVEL_UNITS.jsonl"
    write_json_atomic(alignment_path, alignment_payload)
    write_json_atomic(deletion_path, deletion_payload)
    unit_count = _write_jsonl(unit_path, unit_rows)
    artifacts = {
        "alignment_statistics": {
            "path": str(alignment_path.resolve()),
            "sha256": file_sha256(alignment_path),
            "content_checksum_sha256": alignment_payload["content_checksum_sha256"],
            "records": len(alignment_statistics),
        },
        "deletion_statistics": {
            "path": str(deletion_path.resolve()),
            "sha256": file_sha256(deletion_path),
            "content_checksum_sha256": deletion_payload["content_checksum_sha256"],
            "records": len(deletion_statistics),
        },
        "image_level_units": {
            "path": str(unit_path.resolve()),
            "sha256": file_sha256(unit_path),
            "row_schema": UNIT_SCHEMA,
            "rows": unit_count,
        },
    }
    manifest: dict[str, Any] = {
        "schema": MANIFEST_SCHEMA,
        **common,
        "analysis_implementation_sha256": file_sha256(Path(__file__).resolve()),
        "artifacts": artifacts,
    }
    manifest["content_checksum_sha256"] = canonical_sha256(manifest)
    write_json_atomic(output_dir / "idrid_statistics_manifest.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument(
        "--protocol", type=Path,
        default=REPO_ROOT / "scripts" / "protocols" / "origin_idrid_semantic_statistics_protocol.json",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    manifest = analyze_idrid_semantics(
        summary_path=args.summary,
        protocol_path=args.protocol,
        output_dir=args.output_dir,
    )
    print(
        "ORIGIN IDRiD semantic statistics complete: "
        f"manifest={args.output_dir / 'idrid_statistics_manifest.json'} "
        f"checksum={manifest['content_checksum_sha256']}"
    )


if __name__ == "__main__":
    main()
