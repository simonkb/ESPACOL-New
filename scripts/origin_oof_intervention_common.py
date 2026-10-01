#!/usr/bin/env python3
"""Shared, deterministic machinery for the ORIGIN OOF intervention census.

This module deliberately operates on an already-computed local rate ledger.
Selection never calls the image encoder and never uses an intervention effect:
cells are ranked only by their native contribution to the target boundary.
Every delete/retain result is obtained by replaying :func:`decode_pure_birth_rates`.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from models.origin import decode_pure_birth_rates
from scripts.origin_v3_cv_common import canonical_sha256, read_json, verify_checksummed_payload


PROTOCOL_SCHEMA = "origin-oof-intervention-protocol-v1"
FOLD_SCHEMA = "origin-oof-intervention-fold-v1"
AGGREGATE_SCHEMA = "origin-oof-intervention-aggregate-v1"
ROW_SCHEMA = "origin-oof-intervention-row-v1"

CONTROL_METHODS = (
    "coordinate_permuted",
    "random_scale_count_stride_area",
    "random_scale_footprint",
    "random_boundary_mass",
)
METHODS = ("ranked_native", "least_evidential", *CONTROL_METHODS)
CURVE_ELIGIBILITY_RULE = "true_grade_gt_boundary_and_class_map_gt_boundary"
_THRESHOLD_SPECS = (
    ("min_delete_map_change", "deletion", "map", 0.0),
    ("min_delete_expected_drop_0p25", "deletion", "expected", 0.25),
    ("min_retain_map_preservation", "retention", "map", 0.0),
    *((f"min_delete_tail_effect_{int(level * 100)}pct", "deletion", "tail", level)
      for level in (0.5, 0.8, 0.9)),
    *((f"min_retain_tail_effect_{int(level * 100)}pct", "retention", "tail", level)
      for level in (0.5, 0.8, 0.9)),
)


@dataclass(frozen=True)
class InterventionAuditConfig:
    """Scientific choices frozen before looking at aggregate OOF results."""

    cell_budget_fractions: tuple[float, ...]
    random_repeats: int
    random_seed: int
    denominator_epsilon: float
    footprint_bin_fraction: float
    mass_match_relative_tolerance: float
    decode_chunk_size: int

    def validated(self, *, require_production_repeats: bool = False) -> "InterventionAuditConfig":
        budgets = tuple(float(value) for value in self.cell_budget_fractions)
        if not budgets or budgets[0] != 0.0 or budgets[-1] != 1.0:
            raise ValueError("cell budgets must start at 0 and end at 1")
        if any(not math.isfinite(value) or not 0.0 <= value <= 1.0 for value in budgets):
            raise ValueError("cell budgets must be finite fractions in [0,1]")
        if any(left >= right for left, right in zip(budgets, budgets[1:])):
            raise ValueError("cell budgets must be strictly increasing")
        if self.random_repeats < (50 if require_production_repeats else 1):
            minimum = 50 if require_production_repeats else 1
            raise ValueError(f"random_repeats must be at least {minimum}")
        for name, value in {
            "denominator_epsilon": self.denominator_epsilon,
            "footprint_bin_fraction": self.footprint_bin_fraction,
            "mass_match_relative_tolerance": self.mass_match_relative_tolerance,
        }.items():
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        if self.decode_chunk_size < 1:
            raise ValueError("decode_chunk_size must be positive")
        return self

    def record(self) -> dict[str, Any]:
        result = asdict(self)
        result["cell_budget_fractions"] = list(self.cell_budget_fractions)
        return result


def load_audit_protocol(path: str | Path) -> tuple[dict[str, Any], InterventionAuditConfig]:
    payload = read_json(path)
    verify_checksummed_payload(payload, schema=PROTOCOL_SCHEMA)
    if payload.get("ranking") != "descending_native_local_rate_at_target_boundary":
        raise ValueError("audit protocol changed the non-circular native ranking")
    if payload.get("interventions") != ["deletion", "retention"]:
        raise ValueError("audit protocol must contain deletion and retention")
    if payload.get("curve_eligibility") != CURVE_ELIGIBILITY_RULE:
        raise ValueError("audit protocol curve eligibility rule changed")
    if tuple(payload.get("controls", ())) != CONTROL_METHODS:
        raise ValueError("audit protocol control set or ordering changed")
    config_values = payload.get("config")
    if not isinstance(config_values, Mapping):
        raise TypeError("audit protocol config is missing")
    config = InterventionAuditConfig(
        cell_budget_fractions=tuple(config_values["cell_budget_fractions"]),
        random_repeats=int(config_values["random_repeats"]),
        random_seed=int(config_values["random_seed"]),
        denominator_epsilon=float(config_values["denominator_epsilon"]),
        footprint_bin_fraction=float(config_values["footprint_bin_fraction"]),
        mass_match_relative_tolerance=float(
            config_values["mass_match_relative_tolerance"]
        ),
        decode_chunk_size=int(config_values["decode_chunk_size"]),
    ).validated(require_production_repeats=True)
    if payload.get("config_sha256") != canonical_sha256(config.record()):
        raise ValueError("audit protocol config checksum mismatch")
    return payload, config


def _value(metadata: Any, name: str) -> Any:
    if isinstance(metadata, Mapping):
        return metadata.get(name)
    return getattr(metadata, name, None)


def _as_numpy(value: Any, *, dtype: np.dtype[Any] | None = None) -> np.ndarray:
    if torch.is_tensor(value):
        value = value.detach().cpu().numpy()
    result = np.asarray(value)
    return result.astype(dtype, copy=False) if dtype is not None else result


def _stable_seed(base_seed: int, *parts: Any) -> int:
    message = ":".join((str(int(base_seed)), *(str(part) for part in parts)))
    digest = hashlib.sha256(message.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little", signed=False)


def _decode(rates: np.ndarray, chunk_size: int) -> dict[str, np.ndarray]:
    rates = np.asarray(rates, dtype=np.float64)
    if rates.ndim == 1:
        rates = rates[None]
    if rates.ndim != 2 or rates.shape[1] < 1:
        raise ValueError("rates must have shape (N,K-1)")
    if not np.isfinite(rates).all() or (rates < -2e-10).any():
        raise FloatingPointError("intervention rates are invalid")
    rates = np.maximum(rates, 0.0)
    probabilities: list[np.ndarray] = []
    tails: list[np.ndarray] = []
    expected: list[np.ndarray] = []
    maps: list[np.ndarray] = []
    for start in range(0, len(rates), chunk_size):
        tensor = torch.from_numpy(rates[start : start + chunk_size])
        decoded = decode_pure_birth_rates(tensor, force_fp64=True)
        probabilities.append(decoded.class_probs.detach().cpu().numpy())
        tails.append(decoded.cumulative_probs.detach().cpu().numpy())
        expected.append(decoded.expected_grade.detach().cpu().numpy())
        maps.append(decoded.class_map.detach().cpu().numpy())
    return {
        "probabilities": np.concatenate(probabilities),
        "tails": np.concatenate(tails),
        "expected": np.concatenate(expected),
        "map": np.concatenate(maps),
    }


def _cell_geometry(
    height: int,
    width: int,
    metadata: Any,
) -> tuple[np.ndarray, np.ndarray, float, int, int]:
    stride = int(_value(metadata, "output_stride"))
    receptive_field = float(_value(metadata, "receptive_field"))
    center_offset = float(_value(metadata, "center_offset"))
    input_size = _value(metadata, "input_size")
    if stride < 1 or receptive_field <= 0.0 or input_size is None:
        raise ValueError("complete stride/RF/input metadata is required")
    input_height, input_width = (int(input_size[0]), int(input_size[1]))
    ys = center_offset + np.arange(height, dtype=np.float64) * stride
    xs = center_offset + np.arange(width, dtype=np.float64) * stride
    yy, xx = np.meshgrid(ys, xs, indexing="ij")
    half = receptive_field / 2.0
    rectangles = np.stack(
        (
            np.clip(yy - half, 0.0, input_height),
            np.clip(xx - half, 0.0, input_width),
            np.clip(yy + half, 0.0, input_height),
            np.clip(xx + half, 0.0, input_width),
        ),
        axis=-1,
    ).reshape(-1, 4)
    areas = (
        np.maximum(0.0, rectangles[:, 2] - rectangles[:, 0])
        * np.maximum(0.0, rectangles[:, 3] - rectangles[:, 1])
    )
    return rectangles, areas, float(stride * stride), input_height, input_width


def _rectangle_union_area(rectangles: np.ndarray, height: int, width: int) -> float:
    rectangles = np.asarray(rectangles, dtype=np.float64)
    if len(rectangles) == 0:
        return 0.0
    valid = (rectangles[:, 2] > rectangles[:, 0]) & (
        rectangles[:, 3] > rectangles[:, 1]
    )
    rectangles = rectangles[valid]
    if len(rectangles) == 0:
        return 0.0
    full = np.array([0.0, 0.0, float(height), float(width)])
    if np.any(np.all(np.isclose(rectangles, full[None], atol=1e-12), axis=1)):
        return float(height * width)
    ys = np.unique(np.concatenate((rectangles[:, 0], rectangles[:, 2])))
    xs = np.unique(np.concatenate((rectangles[:, 1], rectangles[:, 3])))
    # ConvNeXt centres and RF boundaries form a small regular coordinate set;
    # a difference grid is exact and much cheaper than pixel rasterisation.
    diff = np.zeros((len(ys), len(xs)), dtype=np.int32)
    y_lookup = {float(value): index for index, value in enumerate(ys)}
    x_lookup = {float(value): index for index, value in enumerate(xs)}
    for y0, x0, y1, x1 in rectangles:
        a, b = y_lookup[float(y0)], y_lookup[float(y1)]
        c, d = x_lookup[float(x0)], x_lookup[float(x1)]
        diff[a, c] += 1
        diff[b, c] -= 1
        diff[a, d] -= 1
        diff[b, d] += 1
    covered = diff.cumsum(0).cumsum(1)[:-1, :-1] > 0
    slab_areas = np.diff(ys)[:, None] * np.diff(xs)[None, :]
    return float(slab_areas[covered].sum())


def _prefix(array: np.ndarray) -> np.ndarray:
    return np.concatenate(
        (np.zeros((1,) + array.shape[1:], dtype=np.float64), array.cumsum(axis=0)),
        axis=0,
    )


def _prefix_max(array: np.ndarray) -> np.ndarray:
    if len(array) == 0:
        return np.zeros(1, dtype=np.float64)
    return np.concatenate(([0.0], np.maximum.accumulate(array.astype(np.float64))))


def _rank(scores: np.ndarray, indices: np.ndarray | None = None) -> np.ndarray:
    if indices is None:
        indices = np.arange(len(scores), dtype=np.int64)
    indices = np.asarray(indices, dtype=np.int64)
    local = np.lexsort((indices, -scores[indices]))
    return indices[local]


def _budget_index(fraction: float, count: int) -> int:
    if fraction <= 0.0:
        return 0
    if fraction >= 1.0:
        return count
    return min(count, max(1, int(math.ceil(float(fraction) * count))))


def _safe_normalize(value: float, denominator: float, epsilon: float) -> float | None:
    if not math.isfinite(value) or not math.isfinite(denominator) or abs(denominator) <= epsilon:
        return None
    return float(value / denominator)


def _json_compact(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _scenario_metrics(
    *,
    boundary: int,
    full: dict[str, np.ndarray],
    prior: dict[str, np.ndarray],
    deletion: dict[str, np.ndarray],
    retention: dict[str, np.ndarray],
    index: int,
    epsilon: float,
) -> dict[str, Any]:
    full_p = full["probabilities"][0]
    prior_p = prior["probabilities"][0]
    delete_p = deletion["probabilities"][index]
    retain_p = retention["probabilities"][index]
    full_tail = float(full["tails"][0, boundary])
    prior_tail = float(prior["tails"][0, boundary])
    delete_tail = float(deletion["tails"][index, boundary])
    retain_tail = float(retention["tails"][index, boundary])
    full_expected = float(full["expected"][0])
    prior_expected = float(prior["expected"][0])
    delete_expected = float(deletion["expected"][index])
    retain_expected = float(retention["expected"][index])
    full_map = int(full["map"][0])
    prior_map = int(prior["map"][0])
    delete_map = int(deletion["map"][index])
    retain_map = int(retention["map"][index])
    tail_range = full_tail - prior_tail
    expected_range = full_expected - prior_expected
    tv_range = 0.5 * float(np.abs(full_p - prior_p).sum())
    map_range = float(full_map - prior_map)
    delete_tail_raw = full_tail - delete_tail
    retain_tail_raw = retain_tail - prior_tail
    delete_expected_raw = full_expected - delete_expected
    retain_expected_raw = retain_expected - prior_expected
    delete_tv_raw = 0.5 * float(np.abs(full_p - delete_p).sum())
    retain_tv_raw = 0.5 * float(np.abs(retain_p - prior_p).sum())
    delete_map_raw = float(full_map - delete_map)
    retain_map_raw = float(retain_map - prior_map)
    return {
        "deletion_tail_raw": delete_tail_raw,
        "deletion_tail_normalized": _safe_normalize(delete_tail_raw, tail_range, epsilon),
        "retention_tail_raw": retain_tail_raw,
        "retention_tail_normalized": _safe_normalize(retain_tail_raw, tail_range, epsilon),
        "deletion_expected_grade_raw": delete_expected_raw,
        "deletion_expected_grade_normalized": _safe_normalize(
            delete_expected_raw, expected_range, epsilon
        ),
        "retention_expected_grade_raw": retain_expected_raw,
        "retention_expected_grade_normalized": _safe_normalize(
            retain_expected_raw, expected_range, epsilon
        ),
        "deletion_posterior_tv_raw": delete_tv_raw,
        "deletion_posterior_tv_normalized": _safe_normalize(
            delete_tv_raw, tv_range, epsilon
        ),
        "retention_posterior_tv_raw": retain_tv_raw,
        "retention_posterior_tv_normalized": _safe_normalize(
            retain_tv_raw, tv_range, epsilon
        ),
        "deletion_map_grade_drop_raw": delete_map_raw,
        "deletion_map_grade_drop_normalized": _safe_normalize(
            delete_map_raw, map_range, epsilon
        ),
        "retention_map_grade_gain_raw": retain_map_raw,
        "retention_map_grade_gain_normalized": _safe_normalize(
            retain_map_raw, map_range, epsilon
        ),
        "deletion_map_changed": float(delete_map != full_map),
        "retention_map_preserved": float(retain_map == full_map),
        "retention_tv_from_full": 0.5 * float(np.abs(retain_p - full_p).sum()),
        "retention_tail_gap_to_full": full_tail - retain_tail,
    }


_AGGREGATED_FIELDS = (
    "selected_count",
    "selected_cell_fraction",
    "selected_nominal_stride_area_px2",
    "selected_nominal_stride_area_fraction",
    "selected_nominal_stride_area_input_fraction",
    "selected_target_boundary_mass",
    "selected_target_boundary_mass_fraction",
    "selected_rate_vector_l1_fraction",
    "summed_clipped_footprint_area_input_fraction",
    "maximum_single_footprint_area_input_fraction",
    "matched_target_mass_relative_error",
    "matched_rate_vector_l1_relative_error",
    "deletion_tail_raw",
    "deletion_tail_normalized",
    "retention_tail_raw",
    "retention_tail_normalized",
    "deletion_expected_grade_raw",
    "deletion_expected_grade_normalized",
    "retention_expected_grade_raw",
    "retention_expected_grade_normalized",
    "deletion_posterior_tv_raw",
    "deletion_posterior_tv_normalized",
    "retention_posterior_tv_raw",
    "retention_posterior_tv_normalized",
    "deletion_map_grade_drop_raw",
    "deletion_map_grade_drop_normalized",
    "retention_map_grade_gain_raw",
    "retention_map_grade_gain_normalized",
    "deletion_map_changed",
    "retention_map_preserved",
    "retention_tv_from_full",
    "retention_tail_gap_to_full",
)


def _aggregate_samples(samples: Sequence[Mapping[str, Any]], scale_names: Sequence[str]) -> dict[str, Any]:
    if not samples:
        raise ValueError("cannot aggregate an empty intervention sample")
    result: dict[str, Any] = {"repeat_count": len(samples)}
    for field in (*_AGGREGATED_FIELDS, *(f"selected_count_{name}" for name in scale_names)):
        observed = [float(item[field]) for item in samples if item.get(field) is not None]
        if not observed:
            result[field] = None
            result[f"{field}_sd"] = None
            result[f"{field}_q025"] = None
            result[f"{field}_q975"] = None
            continue
        values = np.asarray(observed, dtype=np.float64)
        result[field] = float(values.mean())
        result[f"{field}_sd"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
        result[f"{field}_q025"] = float(np.quantile(values, 0.025))
        result[f"{field}_q975"] = float(np.quantile(values, 0.975))
    union_values = [
        float(item["clipped_union_footprint_area_input_fraction"])
        for item in samples
        if item.get("clipped_union_footprint_area_input_fraction") is not None
    ]
    result["clipped_union_footprint_area_input_fraction"] = (
        float(np.mean(union_values)) if union_values else None
    )
    result["mass_match_within_tolerance_rate"] = float(
        np.mean([float(item["mass_match_within_tolerance"]) for item in samples])
    )
    return result


def _auc(rows: Sequence[Mapping[str, Any]], x_field: str, y_field: str) -> float | None:
    pairs = [
        (float(row[x_field]), float(row[y_field]))
        for row in rows
        if row.get(x_field) is not None and row.get(y_field) is not None
    ]
    if len(pairs) < 2:
        return None
    by_x: dict[float, list[float]] = {}
    for x, y in pairs:
        by_x.setdefault(x, []).append(y)
    xs = np.asarray(sorted(by_x), dtype=np.float64)
    ys = np.asarray([np.mean(by_x[x]) for x in xs], dtype=np.float64)
    if len(xs) < 2 or xs[-1] <= xs[0]:
        return None
    integral = np.sum(0.5 * (ys[1:] + ys[:-1]) * np.diff(xs))
    return float(integral / (xs[-1] - xs[0]))


def _summarize_curves(
    identity: Mapping[str, Any], boundary: int, rows: Sequence[dict[str, Any]]
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    endpoint_fields = (
        "deletion_tail_raw",
        "deletion_tail_normalized",
        "retention_tail_raw",
        "retention_tail_normalized",
        "deletion_expected_grade_raw",
        "deletion_expected_grade_normalized",
        "retention_expected_grade_raw",
        "retention_expected_grade_normalized",
        "deletion_posterior_tv_raw",
        "deletion_posterior_tv_normalized",
        "retention_posterior_tv_raw",
        "retention_posterior_tv_normalized",
        "deletion_map_changed",
        "retention_map_preserved",
    )
    x_fields = {
        "cells": "selected_cell_fraction",
        "nominal_stride_area": "selected_nominal_stride_area_fraction",
        "target_evidence_mass": "selected_target_boundary_mass_fraction",
    }
    by_method = {
        method: [row for row in rows if row["method"] == method] for method in METHODS
    }
    for method, method_rows in by_method.items():
        if not method_rows:
            continue
        summary: dict[str, Any] = {
            "schema": ROW_SCHEMA,
            "row_type": "image_boundary_curve_summary",
            **identity,
            "boundary": boundary,
            "method": method,
        }
        for budget_name, x_field in x_fields.items():
            for endpoint in endpoint_fields:
                summary[f"auc_{endpoint}_by_{budget_name}"] = _auc(
                    method_rows, x_field, endpoint
                )
        result.append(summary)
    ranked = next((item for item in result if item["method"] == "ranked_native"), None)
    if ranked is not None:
        for summary in result:
            if summary["method"] in ("ranked_native", "least_evidential"):
                continue
            for key, ranked_value in ranked.items():
                if not key.startswith("auc_") or ranked_value is None or summary.get(key) is None:
                    continue
                summary[f"ranked_advantage_{key}"] = float(ranked_value) - float(summary[key])
                denominator = float(summary[key])
                summary[f"ranked_ratio_{key}"] = (
                    None if abs(denominator) <= 1e-12 else float(ranked_value) / denominator
                )
    return result


def _minimal_prefix(
    *,
    mode: str,
    prefix_rates: np.ndarray,
    prior_rates: np.ndarray,
    full_rates: np.ndarray,
    full: dict[str, np.ndarray],
    prior: dict[str, np.ndarray],
    boundary: int,
    predicate: str,
    threshold: float,
    epsilon: float,
    chunk_size: int,
) -> int | None:
    count = len(prefix_rates) - 1
    cache: dict[int, dict[str, np.ndarray]] = {}

    def decoded(k: int) -> dict[str, np.ndarray]:
        if k not in cache:
            if mode == "deletion":
                rates = full_rates - prefix_rates[k]
                if k == count:
                    rates = prior_rates
            else:
                rates = prior_rates + prefix_rates[k]
                if k == count:
                    rates = full_rates
            cache[k] = _decode(rates, chunk_size)
        return cache[k]

    full_tail = float(full["tails"][0, boundary])
    prior_tail = float(prior["tails"][0, boundary])
    tail_range = full_tail - prior_tail
    full_expected = float(full["expected"][0])
    full_map = int(full["map"][0])

    def passes(k: int) -> bool:
        value = decoded(k)
        if predicate == "map":
            return (
                int(value["map"][0]) != full_map
                if mode == "deletion"
                else int(value["map"][0]) == full_map
            )
        if predicate == "expected":
            return full_expected - float(value["expected"][0]) >= threshold - 1e-12
        tail = float(value["tails"][0, boundary])
        raw = full_tail - tail if mode == "deletion" else tail - prior_tail
        normalized = _safe_normalize(raw, tail_range, epsilon)
        return normalized is not None and normalized >= threshold - 1e-12

    if passes(0):
        return 0
    if not passes(count):
        return None
    low, high = 0, count
    while high - low > 1:
        middle = (low + high) // 2
        if passes(middle):
            high = middle
        else:
            low = middle
    return high


def audit_image_ledger(
    *,
    rate_maps: Mapping[str, Any],
    valid_masks: Mapping[str, Any],
    prior_rates: Any,
    metadata: Mapping[str, Any],
    identity: Mapping[str, Any],
    true_grade: int,
    config: InterventionAuditConfig,
    baseline_total_rates: Any | None = None,
    baseline_probabilities: Any | None = None,
    curve_eligible_boundaries: set[int] | None = None,
) -> dict[str, Any]:
    """Run the complete grouped census for one image ledger.

    Random repeats are summarized *within* this image.  Consequently each
    emitted curve/summary row remains an independent image-level bootstrap
    unit (or a patient-cluster unit through ``identity['cluster_id']``).
    """

    config.validated()
    if not rate_maps or tuple(rate_maps) != tuple(valid_masks):
        raise ValueError("rate maps and validity masks must have identical scales")
    scale_names = tuple(rate_maps)
    vectors_parts: list[np.ndarray] = []
    scale_parts: list[np.ndarray] = []
    stride_parts: list[np.ndarray] = []
    footprint_parts: list[np.ndarray] = []
    rectangle_parts: list[np.ndarray] = []
    image_shape: tuple[int, int] | None = None
    scale_geometry: dict[str, Any] = {}
    for scale_index, scale in enumerate(scale_names):
        rates = _as_numpy(rate_maps[scale], dtype=np.float64)
        valid = _as_numpy(valid_masks[scale]).astype(bool, copy=False)
        if rates.ndim != 3 or valid.shape != rates.shape[:2]:
            raise ValueError(f"scale {scale} must have rate map (H,W,B) and mask (H,W)")
        if not np.isfinite(rates).all() or (rates < 0.0).any():
            raise FloatingPointError(f"scale {scale} contains invalid local rates")
        rectangles, footprint_areas, stride_area, input_height, input_width = _cell_geometry(
            rates.shape[0], rates.shape[1], metadata[scale]
        )
        if image_shape is None:
            image_shape = (input_height, input_width)
        elif image_shape != (input_height, input_width):
            raise ValueError("scale metadata disagree on input image size")
        flat_valid = valid.reshape(-1)
        vectors_parts.append(rates.reshape(-1, rates.shape[-1])[flat_valid])
        scale_parts.append(np.full(int(flat_valid.sum()), scale_index, dtype=np.int16))
        stride_parts.append(np.full(int(flat_valid.sum()), stride_area, dtype=np.float64))
        footprint_parts.append(footprint_areas[flat_valid])
        rectangle_parts.append(rectangles[flat_valid])
        scale_geometry[scale] = {
            "output_stride_pixels": int(_value(metadata[scale], "output_stride")),
            "receptive_field_pixels": float(_value(metadata[scale], "receptive_field")),
            "input_size_pixels": [input_height, input_width],
            "theoretical_support_class": (
                "global_context"
                if float(_value(metadata[scale], "receptive_field")) >= max(input_height, input_width)
                else "finite_local"
            ),
            "valid_cell_count": int(flat_valid.sum()),
        }
    assert image_shape is not None
    vectors = np.concatenate(vectors_parts, axis=0)
    scales = np.concatenate(scale_parts)
    stride_costs = np.concatenate(stride_parts)
    footprint_areas = np.concatenate(footprint_parts)
    rectangles = np.concatenate(rectangle_parts)
    if len(vectors) < 1:
        raise ValueError("image ledger has no valid cells")
    boundaries = vectors.shape[1]
    prior_rates_np = _as_numpy(prior_rates, dtype=np.float64).reshape(-1)
    if prior_rates_np.shape != (boundaries,):
        raise ValueError("prior rate vector has the wrong shape")
    reconstructed_full_rates = prior_rates_np + vectors.sum(axis=0, dtype=np.float64)
    if baseline_total_rates is None:
        full_rates = reconstructed_full_rates
    else:
        full_rates = _as_numpy(baseline_total_rates, dtype=np.float64).reshape(-1)
        if full_rates.shape != (boundaries,):
            raise ValueError("baseline total-rate vector has the wrong shape")
    local_total = full_rates - prior_rates_np
    reconstruction_error = float(np.max(np.abs(full_rates - reconstructed_full_rates)))
    full = _decode(full_rates, config.decode_chunk_size)
    prior = _decode(prior_rates_np, config.decode_chunk_size)
    probability_error = 0.0
    if baseline_probabilities is not None:
        baseline = _as_numpy(baseline_probabilities, dtype=np.float64).reshape(-1)
        probability_error = float(np.max(np.abs(full["probabilities"][0] - baseline)))
        if probability_error > 2e-10:
            raise AssertionError(
                f"same-rate decoder replay changed baseline posterior: {probability_error:g}"
            )
    predicted_grade = int(full["map"][0])
    common_identity = {
        **dict(identity),
        "true_grade": int(true_grade),
        "predicted_grade": predicted_grade,
        "correct": bool(predicted_grade == int(true_grade)),
    }
    image_height, image_width = image_shape
    image_area = float(image_height * image_width)
    total_stride_area = float(stride_costs.sum())
    total_vector_l1 = float(np.abs(local_total).sum())
    global_indices = np.arange(len(vectors), dtype=np.int64)
    footprint_bins = np.floor(
        footprint_areas / max(config.footprint_bin_fraction * image_area, 1e-12) + 1e-12
    ).astype(np.int64)
    stratum_ids = scales.astype(np.int64) * (int(footprint_bins.max()) + 1) + footprint_bins
    unique_strata = np.unique(stratum_ids)
    all_union_area = _rectangle_union_area(rectangles, image_height, image_width)
    curve_rows: list[dict[str, Any]] = []
    image_rows: list[dict[str, Any]] = []

    for boundary in range(boundaries):
        scores = vectors[:, boundary]
        order = _rank(scores)
        reverse_order = np.lexsort((global_indices, scores))
        ordered_vectors = vectors[order]
        prefix_rates = _prefix(ordered_vectors)
        prefix_stride = np.concatenate(([0.0], stride_costs[order].cumsum()))
        prefix_mass = np.concatenate(([0.0], scores[order].cumsum()))
        prefix_footprint = np.concatenate(([0.0], footprint_areas[order].cumsum()))
        prefix_max_footprint = _prefix_max(footprint_areas[order])
        local_mass = float(local_total[boundary])
        raw_local_mass = float(scores.sum(dtype=np.float64))
        if abs(local_mass - raw_local_mass) > max(2e-5, 2e-6 * max(1.0, abs(local_mass))):
            raise AssertionError("target-boundary ledger no longer reconstructs total-minus-prior")
        score_square_sum = float(np.square(scores).sum(dtype=np.float64))
        effective_support = (
            0.0 if raw_local_mass <= 0.0 or score_square_sum <= 0.0
            else raw_local_mass * raw_local_mass / score_square_sum
        )
        positive = scores[scores > 0.0]
        if len(positive):
            weights = positive / positive.sum()
            entropy = float(-(weights * np.log(weights)).sum())
            entropy_normalized = float(entropy / math.log(len(positive))) if len(positive) > 1 else 0.0
        else:
            entropy = 0.0
            entropy_normalized = 0.0
        scale_mass = {
            scale: float(vectors[scales == index, boundary].sum(dtype=np.float64))
            for index, scale in enumerate(scale_names)
        }
        expected_range = float(full["expected"][0] - prior["expected"][0])
        posterior_tv_range = 0.5 * float(
            np.abs(full["probabilities"][0] - prior["probabilities"][0]).sum()
        )
        map_range = int(full["map"][0]) - int(prior["map"][0])
        true_boundary_positive = int(true_grade) > boundary
        predicted_boundary_positive = predicted_grade > boundary
        curve_eligible = (
            True
            if curve_eligible_boundaries is None
            else boundary in curve_eligible_boundaries
        )
        image_row: dict[str, Any] = {
            "schema": ROW_SCHEMA,
            "row_type": "image_boundary_census",
            **common_identity,
            "boundary": boundary,
            "true_boundary_positive": true_boundary_positive,
            "predicted_boundary_positive": predicted_boundary_positive,
            "curve_eligible": curve_eligible,
            "curve_eligibility_rule": CURVE_ELIGIBILITY_RULE,
            "curve_exclusion_reason": (
                None
                if curve_eligible
                else (
                    "true_and_predicted_boundary_not_positive"
                    if not true_boundary_positive and not predicted_boundary_positive
                    else "true_boundary_not_positive"
                    if not true_boundary_positive
                    else "predicted_boundary_not_positive"
                )
            ),
            "valid_cell_count": len(vectors),
            "full_class_probabilities": full["probabilities"][0].tolist(),
            "prior_class_probabilities": prior["probabilities"][0].tolist(),
            "full_tail_probability": float(full["tails"][0, boundary]),
            "prior_tail_probability": float(prior["tails"][0, boundary]),
            "locally_attributable_tail_range": float(
                full["tails"][0, boundary] - prior["tails"][0, boundary]
            ),
            "tail_range_near_zero": bool(
                abs(float(full["tails"][0, boundary] - prior["tails"][0, boundary]))
                <= config.denominator_epsilon
            ),
            "full_expected_grade": float(full["expected"][0]),
            "prior_expected_grade": float(prior["expected"][0]),
            "locally_attributable_expected_grade_range": expected_range,
            "expected_grade_range_near_zero": bool(
                abs(expected_range) <= config.denominator_epsilon
            ),
            "locally_attributable_posterior_tv_range": posterior_tv_range,
            "posterior_tv_range_near_zero": bool(
                abs(posterior_tv_range) <= config.denominator_epsilon
            ),
            "full_map_grade": int(full["map"][0]),
            "prior_map_grade": int(prior["map"][0]),
            "locally_attributable_map_grade_range": map_range,
            "map_grade_range_near_zero": bool(
                abs(map_range) <= config.denominator_epsilon
            ),
            "local_target_boundary_mass": local_mass,
            "target_mass_hhi": (
                None if raw_local_mass <= config.denominator_epsilon
                else float(score_square_sum / (raw_local_mass * raw_local_mass))
            ),
            "target_mass_effective_support": effective_support,
            "target_mass_entropy": entropy,
            "target_mass_entropy_normalized": entropy_normalized,
            "scale_target_mass": scale_mass,
            "scale_target_mass_json": _json_compact(scale_mass),
            "scale_geometry": scale_geometry,
            "scale_geometry_json": _json_compact(scale_geometry),
            "ledger_total_rate_reconstruction_max_abs_error": reconstruction_error,
            "baseline_decoder_replay_max_abs_probability_error": probability_error,
        }
        if not curve_eligible:
            for name, _, _, _ in _THRESHOLD_SPECS:
                image_row[f"{name}_cell_count"] = None
                image_row[f"{name}_cell_fraction"] = None
                image_row[f"{name}_nominal_stride_area_fraction"] = None
                image_row[f"{name}_target_mass_fraction"] = None
            image_rows.append(image_row)
            continue
        budget_ks = [_budget_index(value, len(vectors)) for value in config.cell_budget_fractions]
        references: list[dict[str, Any]] = []
        for budget_index, (budget, k) in enumerate(zip(config.cell_budget_fractions, budget_ks)):
            selected = order[:k]
            counts = np.bincount(scales[selected], minlength=len(scale_names)).astype(np.int64)
            strata_counts = {
                int(stratum): int(np.count_nonzero(stratum_ids[selected] == stratum))
                for stratum in unique_strata
            }
            selected_rates = prefix_rates[k].copy()
            if k == len(vectors):
                selected_rates = local_total.copy()
            union_area = (
                0.0 if k == 0 else all_union_area if k == len(vectors)
                else _rectangle_union_area(rectangles[selected], image_height, image_width)
            )
            references.append(
                {
                    "budget_index": budget_index,
                    "requested_cell_fraction": float(budget),
                    "k": k,
                    "indices": selected,
                    "rates": selected_rates,
                    "counts": counts,
                    "strata_counts": strata_counts,
                    "stride": float(prefix_stride[k]),
                    "footprint": float(prefix_footprint[k]),
                    "max_footprint": float(prefix_max_footprint[k]),
                    "union": float(union_area / image_area),
                }
            )

        scenarios: list[dict[str, Any]] = []

        def add_scenario(
            method: str,
            repeat: int,
            reference: Mapping[str, Any],
            selected_rates: np.ndarray,
            count: int,
            stride: float,
            footprint: float,
            max_footprint: float,
            counts: np.ndarray,
            union: float | None = None,
        ) -> None:
            selected_rates = np.asarray(selected_rates, dtype=np.float64).copy()
            if count == len(vectors):
                selected_rates = local_total.copy()
            target_mass = float(selected_rates[boundary])
            reference_rates = np.asarray(reference["rates"], dtype=np.float64)
            reference_mass = float(reference_rates[boundary])
            mass_denominator = max(abs(reference_mass), config.denominator_epsilon)
            vector_denominator = max(
                float(np.abs(reference_rates).sum()), config.denominator_epsilon
            )
            mass_error = abs(target_mass - reference_mass) / mass_denominator
            vector_error = float(np.abs(selected_rates - reference_rates).sum()) / vector_denominator
            scenario: dict[str, Any] = {
                "method": method,
                "repeat": repeat,
                "budget_index": int(reference["budget_index"]),
                "requested_cell_fraction": float(reference["requested_cell_fraction"]),
                "selected_rates": selected_rates,
                "selected_count": float(count),
                "selected_cell_fraction": float(count / len(vectors)),
                "selected_nominal_stride_area_px2": float(stride),
                "selected_nominal_stride_area_fraction": float(stride / total_stride_area),
                "selected_nominal_stride_area_input_fraction": float(stride / image_area),
                "selected_target_boundary_mass": target_mass,
                "selected_target_boundary_mass_fraction": _safe_normalize(
                    target_mass, local_mass, config.denominator_epsilon
                ),
                "selected_rate_vector_l1_fraction": _safe_normalize(
                    float(np.abs(selected_rates).sum()),
                    total_vector_l1,
                    config.denominator_epsilon,
                ),
                "summed_clipped_footprint_area_input_fraction": float(footprint / image_area),
                "maximum_single_footprint_area_input_fraction": float(max_footprint / image_area),
                "clipped_union_footprint_area_input_fraction": union,
                "matched_target_mass_relative_error": mass_error,
                "matched_rate_vector_l1_relative_error": vector_error,
                "mass_match_within_tolerance": float(
                    mass_error <= config.mass_match_relative_tolerance
                ),
            }
            for scale_index, scale in enumerate(scale_names):
                scenario[f"selected_count_{scale}"] = float(counts[scale_index])
            scenarios.append(scenario)

        for reference in references:
            k = int(reference["k"])
            add_scenario(
                "ranked_native", 0, reference, reference["rates"], k,
                reference["stride"], reference["footprint"], reference["max_footprint"],
                reference["counts"], reference["union"],
            )
            selected = reverse_order[:k]
            rates = vectors[selected].sum(axis=0, dtype=np.float64) if k else np.zeros(boundaries)
            if k == len(vectors):
                rates = local_total.copy()
            counts = np.bincount(scales[selected], minlength=len(scale_names))
            add_scenario(
                "least_evidential", 0, reference, rates, k,
                float(stride_costs[selected].sum()), float(footprint_areas[selected].sum()),
                float(footprint_areas[selected].max()) if k else 0.0, counts, None,
            )

        # Each repeat supplies one nested random curve per control.  This is
        # both faster and statistically cleaner than redrawing every budget.
        for repeat in range(config.random_repeats):
            rng = np.random.default_rng(
                _stable_seed(config.random_seed, identity.get("image_key"), boundary, "coordinate", repeat)
            )
            pseudo = np.empty(len(vectors), dtype=np.float64)
            for scale_index in range(len(scale_names)):
                members = np.flatnonzero(scales == scale_index)
                pseudo[members] = scores[members][rng.permutation(len(members))]
            random_order = _rank(pseudo)
            random_prefix_rates = _prefix(vectors[random_order])
            random_prefix_stride = np.concatenate(([0.0], stride_costs[random_order].cumsum()))
            random_prefix_foot = np.concatenate(([0.0], footprint_areas[random_order].cumsum()))
            random_prefix_max = _prefix_max(footprint_areas[random_order])
            for reference in references:
                k = int(reference["k"])
                counts = np.bincount(scales[random_order[:k]], minlength=len(scale_names))
                add_scenario(
                    "coordinate_permuted", repeat, reference, random_prefix_rates[k], k,
                    random_prefix_stride[k], random_prefix_foot[k], random_prefix_max[k], counts,
                )

            rng = np.random.default_rng(
                _stable_seed(config.random_seed, identity.get("image_key"), boundary, "scale", repeat)
            )
            scale_orders: dict[int, np.ndarray] = {}
            scale_prefix_rates: dict[int, np.ndarray] = {}
            scale_prefix_foot: dict[int, np.ndarray] = {}
            scale_prefix_max: dict[int, np.ndarray] = {}
            for scale_index in range(len(scale_names)):
                members = np.flatnonzero(scales == scale_index)
                chosen_order = members[rng.permutation(len(members))]
                scale_orders[scale_index] = chosen_order
                scale_prefix_rates[scale_index] = _prefix(vectors[chosen_order])
                scale_prefix_foot[scale_index] = np.concatenate(
                    ([0.0], footprint_areas[chosen_order].cumsum())
                )
                scale_prefix_max[scale_index] = _prefix_max(footprint_areas[chosen_order])
            for reference in references:
                counts = np.asarray(reference["counts"], dtype=np.int64)
                rates = sum(
                    (scale_prefix_rates[index][int(counts[index])] for index in range(len(scale_names))),
                    np.zeros(boundaries, dtype=np.float64),
                )
                footprint = sum(
                    float(scale_prefix_foot[index][int(counts[index])])
                    for index in range(len(scale_names))
                )
                max_footprint = max(
                    float(scale_prefix_max[index][int(counts[index])])
                    for index in range(len(scale_names))
                )
                add_scenario(
                    "random_scale_count_stride_area", repeat, reference, rates,
                    int(counts.sum()), float(reference["stride"]), footprint,
                    max_footprint, counts,
                )

            rng = np.random.default_rng(
                _stable_seed(config.random_seed, identity.get("image_key"), boundary, "footprint", repeat)
            )
            stratum_prefix_rates: dict[int, np.ndarray] = {}
            stratum_prefix_foot: dict[int, np.ndarray] = {}
            stratum_prefix_max: dict[int, np.ndarray] = {}
            stratum_scale: dict[int, int] = {}
            for stratum in unique_strata:
                members = np.flatnonzero(stratum_ids == stratum)
                chosen_order = members[rng.permutation(len(members))]
                stratum_prefix_rates[int(stratum)] = _prefix(vectors[chosen_order])
                stratum_prefix_foot[int(stratum)] = np.concatenate(
                    ([0.0], footprint_areas[chosen_order].cumsum())
                )
                stratum_prefix_max[int(stratum)] = _prefix_max(footprint_areas[chosen_order])
                stratum_scale[int(stratum)] = int(scales[members[0]])
            for reference in references:
                rates = np.zeros(boundaries, dtype=np.float64)
                footprint = 0.0
                max_footprint = 0.0
                counts = np.zeros(len(scale_names), dtype=np.int64)
                for stratum, count in reference["strata_counts"].items():
                    count = int(count)
                    rates += stratum_prefix_rates[stratum][count]
                    footprint += float(stratum_prefix_foot[stratum][count])
                    max_footprint = max(max_footprint, float(stratum_prefix_max[stratum][count]))
                    counts[stratum_scale[stratum]] += count
                add_scenario(
                    "random_scale_footprint", repeat, reference, rates,
                    int(counts.sum()), float(reference["stride"]), footprint,
                    max_footprint, counts,
                )

            rng = np.random.default_rng(
                _stable_seed(config.random_seed, identity.get("image_key"), boundary, "mass", repeat)
            )
            mass_order = rng.permutation(len(vectors))
            mass_prefix_rates = _prefix(vectors[mass_order])
            mass_prefix = np.concatenate(([0.0], scores[mass_order].cumsum()))
            mass_prefix_stride = np.concatenate(([0.0], stride_costs[mass_order].cumsum()))
            mass_prefix_foot = np.concatenate(([0.0], footprint_areas[mass_order].cumsum()))
            mass_prefix_max = _prefix_max(footprint_areas[mass_order])
            for reference in references:
                target = float(np.asarray(reference["rates"])[boundary])
                insertion = int(np.searchsorted(mass_prefix, target, side="left"))
                candidates = [max(0, min(len(vectors), insertion))]
                if insertion > 0:
                    candidates.append(insertion - 1)
                k = min(candidates, key=lambda value: (abs(float(mass_prefix[value]) - target), value))
                counts = np.bincount(scales[mass_order[:k]], minlength=len(scale_names))
                add_scenario(
                    "random_boundary_mass", repeat, reference, mass_prefix_rates[k], k,
                    mass_prefix_stride[k], mass_prefix_foot[k], mass_prefix_max[k], counts,
                )

        selected_matrix = np.stack([item.pop("selected_rates") for item in scenarios])
        deletion_rates = full_rates[None] - selected_matrix
        retention_rates = prior_rates_np[None] + selected_matrix
        for scenario_index, scenario in enumerate(scenarios):
            if int(scenario["selected_count"]) == len(vectors):
                deletion_rates[scenario_index] = prior_rates_np
                retention_rates[scenario_index] = full_rates
        deletion = _decode(deletion_rates, config.decode_chunk_size)
        retention = _decode(retention_rates, config.decode_chunk_size)
        for scenario_index, scenario in enumerate(scenarios):
            scenario.update(
                _scenario_metrics(
                    boundary=boundary,
                    full=full,
                    prior=prior,
                    deletion=deletion,
                    retention=retention,
                    index=scenario_index,
                    epsilon=config.denominator_epsilon,
                )
            )
        boundary_rows: list[dict[str, Any]] = []
        for method in METHODS:
            for budget_index, budget in enumerate(config.cell_budget_fractions):
                samples = [
                    item for item in scenarios
                    if item["method"] == method and item["budget_index"] == budget_index
                ]
                aggregated = _aggregate_samples(samples, scale_names)
                row = {
                    "schema": ROW_SCHEMA,
                    "row_type": "image_boundary_curve_point",
                    **common_identity,
                    "boundary": boundary,
                    "method": method,
                    "budget_index": budget_index,
                    "requested_cell_fraction": float(budget),
                    **aggregated,
                }
                boundary_rows.append(row)
        ranked_by_budget = {
            int(row["budget_index"]): row
            for row in boundary_rows if row["method"] == "ranked_native"
        }
        for row in boundary_rows:
            ranked = ranked_by_budget[int(row["budget_index"])]
            if row["method"] in CONTROL_METHODS:
                for endpoint in (
                    "deletion_tail_raw", "deletion_tail_normalized",
                    "retention_tail_raw", "retention_tail_normalized",
                    "deletion_expected_grade_raw", "retention_expected_grade_raw",
                    "deletion_posterior_tv_raw", "retention_posterior_tv_raw",
                ):
                    left, right = ranked.get(endpoint), row.get(endpoint)
                    row[f"ranked_reference_{endpoint}"] = left
                    row[f"ranked_advantage_{endpoint}"] = (
                        None if left is None or right is None else float(left) - float(right)
                    )
                    row[f"ranked_ratio_{endpoint}"] = (
                        None if left is None or right is None or abs(float(right)) <= 1e-12
                        else float(left) / float(right)
                    )
        curve_rows.extend(boundary_rows)
        for name, mode, predicate, threshold in _THRESHOLD_SPECS:
            k = _minimal_prefix(
                mode=mode,
                prefix_rates=prefix_rates,
                prior_rates=prior_rates_np,
                full_rates=full_rates,
                full=full,
                prior=prior,
                boundary=boundary,
                predicate=predicate,
                threshold=float(threshold),
                epsilon=config.denominator_epsilon,
                chunk_size=config.decode_chunk_size,
            )
            image_row[f"{name}_cell_count"] = k
            image_row[f"{name}_cell_fraction"] = None if k is None else float(k / len(vectors))
            image_row[f"{name}_nominal_stride_area_fraction"] = (
                None if k is None else float(prefix_stride[k] / total_stride_area)
            )
            image_row[f"{name}_target_mass_fraction"] = (
                None if k is None else _safe_normalize(
                    float(prefix_mass[k]), local_mass, config.denominator_epsilon
                )
            )
        image_rows.append(image_row)
        # Curve summaries are appended after all boundary image rows below.

    summary_rows: list[dict[str, Any]] = []
    for boundary in range(boundaries):
        summary_rows.extend(
            _summarize_curves(
                common_identity,
                boundary,
                [row for row in curve_rows if int(row["boundary"]) == boundary],
            )
        )
    return {
        "image_rows": image_rows,
        "curve_rows": curve_rows,
        "summary_rows": summary_rows,
        "diagnostics": {
            "ledger_total_rate_reconstruction_max_abs_error": reconstruction_error,
            "baseline_decoder_replay_max_abs_probability_error": probability_error,
            "valid_cell_count": len(vectors),
            "num_boundaries": boundaries,
            "eligible_boundary_count": sum(
                bool(row["curve_eligible"]) for row in image_rows
            ),
        },
    }


__all__ = [
    "AGGREGATE_SCHEMA",
    "CONTROL_METHODS",
    "FOLD_SCHEMA",
    "InterventionAuditConfig",
    "METHODS",
    "PROTOCOL_SCHEMA",
    "ROW_SCHEMA",
    "audit_image_ledger",
    "load_audit_protocol",
]
