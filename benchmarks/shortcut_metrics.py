"""Metrics and preregistered gates for the controlled ordinal shortcut audit.

The functions in this module are model-agnostic.  They accept already-rendered
maps or paired effects, which lets the same blind audit score ORIGIN, the
same-ledger hazard decoder, pooled ordinal models, and post-hoc baselines.
Ground-truth masks are used only inside metric functions, never to construct a
candidate evidence map.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch.nn import functional as F


def _float_array(value: Any, *, ndim: int | None = None) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if ndim is not None and result.ndim != ndim:
        raise ValueError(f"expected a {ndim}-dimensional array, got {result.shape}")
    if not np.all(np.isfinite(result)):
        raise ValueError("metric inputs must be finite")
    return result


def binary_average_precision(scores: Any, targets: Any) -> float:
    """Average precision with deterministic stable tie handling.

    A tied score block is evaluated at its final rank, matching the usual
    threshold interpretation rather than depending on input order.
    """

    scores_array = _float_array(scores).reshape(-1)
    target_array = np.asarray(targets, dtype=bool).reshape(-1)
    if scores_array.shape != target_array.shape:
        raise ValueError("scores and targets must have identical shapes")
    positives = int(target_array.sum())
    if positives == 0:
        return float("nan")
    order = np.argsort(-scores_array, kind="mergesort")
    sorted_scores = scores_array[order]
    sorted_targets = target_array[order]
    cumulative = np.cumsum(sorted_targets, dtype=np.float64)
    block_ends = np.r_[np.flatnonzero(np.diff(sorted_scores) != 0), len(order) - 1]
    previous_recall = 0.0
    average_precision = 0.0
    for end in block_ends:
        recall = cumulative[end] / positives
        precision = cumulative[end] / (end + 1)
        average_precision += (recall - previous_recall) * precision
        previous_recall = recall
    return float(average_precision)


def pointing_accuracy(
    evidence_maps: Any,
    target_masks: Any,
    valid_masks: Any | None = None,
) -> np.ndarray:
    """Return per-sample/per-boundary pointing-game successes."""

    evidence = _float_array(evidence_maps, ndim=4)
    targets = np.asarray(target_masks, dtype=bool)
    if targets.shape != evidence.shape:
        raise ValueError("target masks must match evidence map shape (N,B,H,W)")
    if valid_masks is None:
        valid = np.ones((evidence.shape[0], 1, *evidence.shape[-2:]), dtype=bool)
    else:
        valid = np.asarray(valid_masks, dtype=bool)
        if valid.ndim == 3:
            valid = valid[:, None]
        if valid.shape not in {
            (evidence.shape[0], 1, *evidence.shape[-2:]),
            evidence.shape,
        }:
            raise ValueError("valid masks must have shape (N,H,W), (N,1,H,W), or (N,B,H,W)")
    valid = np.broadcast_to(valid, evidence.shape)
    result = np.full(evidence.shape[:2], np.nan, dtype=np.float64)
    for sample in range(evidence.shape[0]):
        for boundary in range(evidence.shape[1]):
            usable = valid[sample, boundary]
            target = targets[sample, boundary] & usable
            if not usable.any() or not target.any():
                continue
            scores = np.where(usable, evidence[sample, boundary], -np.inf)
            maximum = scores == np.max(scores)
            # Fractional credit makes exact tied maxima order-independent.
            result[sample, boundary] = float((maximum & target).sum() / maximum.sum())
    return result


def localization_metrics(
    evidence_maps: Any,
    marker_masks: Any,
    valid_masks: Any | None = None,
    *,
    localization_applicable: bool = True,
) -> dict[str, Any]:
    """Blind AUPRC, pointing accuracy, and boundary-marker assignment."""

    evidence = _float_array(evidence_maps, ndim=4)
    masks = np.asarray(marker_masks, dtype=bool)
    if masks.shape != evidence.shape:
        raise ValueError("marker masks must match evidence maps")
    samples, boundaries, height, width = evidence.shape
    if valid_masks is None:
        valid = np.ones((samples, 1, height, width), dtype=bool)
    else:
        valid = np.asarray(valid_masks, dtype=bool)
        if valid.ndim == 3:
            valid = valid[:, None]
        if valid.shape not in {(samples, 1, height, width), evidence.shape}:
            raise ValueError("invalid valid-mask shape")
    valid = np.broadcast_to(valid, evidence.shape)

    per_sample_ap = np.full((samples, boundaries), np.nan, dtype=np.float64)
    prevalence = np.full_like(per_sample_ap, np.nan)
    for sample in range(samples):
        for boundary in range(boundaries):
            usable = valid[sample, boundary]
            if not usable.any():
                continue
            target = masks[sample, boundary, usable]
            per_sample_ap[sample, boundary] = binary_average_precision(
                evidence[sample, boundary, usable], target
            )
            prevalence[sample, boundary] = float(target.mean())
    points = pointing_accuracy(evidence, masks, valid)

    effective_support = np.full((samples, boundaries), np.nan, dtype=np.float64)
    for sample in range(samples):
        for boundary in range(boundaries):
            usable = valid[sample, boundary]
            values = np.maximum(evidence[sample, boundary, usable], 0.0)
            total = float(values.sum())
            if total > 0.0:
                effective_support[sample, boundary] = (
                    total**2 / max(float(np.square(values).sum()), 1e-30)
                ) / len(values)

    assignment = np.full((samples, boundaries), -1, dtype=np.int64)
    assignment_correct = np.full((samples, boundaries), np.nan, dtype=np.float64)
    if localization_applicable:
        for sample in range(samples):
            for evidence_boundary in range(boundaries):
                overlap = []
                for marker_boundary in range(boundaries):
                    marker = masks[sample, marker_boundary] & valid[sample, evidence_boundary]
                    overlap.append(
                        float(evidence[sample, evidence_boundary][marker].mean())
                        if marker.any()
                        else -math.inf
                    )
                winner = int(np.argmax(overlap))
                assignment[sample, evidence_boundary] = winner
                assignment_correct[sample, evidence_boundary] = float(
                    winner == evidence_boundary
                )

    return {
        "localization_applicable": bool(localization_applicable),
        "per_sample_boundary_auprc": per_sample_ap,
        "per_sample_boundary_prevalence": prevalence,
        "per_sample_boundary_pointing": points,
        "per_sample_boundary_effective_support_fraction": effective_support,
        "per_sample_boundary_assignment": assignment,
        "per_sample_boundary_assignment_correct": assignment_correct,
        "boundary_auprc": np.nanmean(per_sample_ap, axis=0),
        "boundary_prevalence": np.nanmean(prevalence, axis=0),
        "boundary_pointing_accuracy": np.nanmean(points, axis=0),
        "boundary_identification_accuracy": (
            np.nanmean(assignment_correct, axis=0)
            if localization_applicable
            else np.full(boundaries, np.nan)
        ),
        "macro_auprc": float(np.nanmean(per_sample_ap)),
        "macro_auprc_lift": float(np.nanmean(per_sample_ap - prevalence)),
        "macro_pointing_accuracy": float(np.nanmean(points)),
        "macro_effective_support_fraction": float(np.nanmean(effective_support)),
        "macro_boundary_identification_accuracy": (
            float(np.nanmean(assignment_correct)) if localization_applicable else float("nan")
        ),
    }


def familywise_localization_permutation_test(
    evidence_maps: Any,
    marker_masks: Any,
    valid_masks: Any | None = None,
    *,
    permutations: int = 999,
    seed: int = 7193,
) -> dict[str, Any]:
    """Sample-permutation localization test with max-statistic FWER control.

    Maps remain fixed and each boundary's procedural masks are reassigned
    between images.  This preserves marker size/shape and image-specific map
    distributions while destroying the learned location correspondence.  The
    test statistic is evidence mass concentration relative to mask prevalence;
    AUPRC and pointing accuracy remain separately reported descriptive metrics.
    """

    evidence = _float_array(evidence_maps, ndim=4)
    masks = np.asarray(marker_masks, dtype=bool)
    if masks.shape != evidence.shape:
        raise ValueError("marker masks must match evidence maps")
    if permutations < 1:
        raise ValueError("permutations must be positive")
    samples, boundaries = evidence.shape[:2]
    if samples < 2:
        raise ValueError("permutation localization requires at least two samples")
    valid = None if valid_masks is None else np.asarray(valid_masks, dtype=bool)
    if valid is None:
        usable = np.ones((samples, *evidence.shape[-2:]), dtype=bool)
    else:
        if valid.ndim == 4:
            usable = valid[:, 0]
        elif valid.ndim == 3:
            usable = valid
        else:
            raise ValueError("valid masks must have shape (N,H,W) or (N,1,H,W)")
    observed = np.empty(boundaries, dtype=np.float64)
    # Cache all unique procedural supports. Position generators intentionally
    # reuse a finite held-out lattice, making this both exact and inexpensive.
    score_tables: list[np.ndarray] = []
    mask_ids: list[np.ndarray] = []
    for boundary in range(boundaries):
        flat_masks = masks[:, boundary].reshape(samples, -1)
        unique_masks, inverse = np.unique(flat_masks, axis=0, return_inverse=True)
        flat_evidence = np.maximum(evidence[:, boundary].reshape(samples, -1), 0.0)
        flat_valid = usable.reshape(samples, -1)
        flat_evidence = np.where(flat_valid, flat_evidence, 0.0)
        totals = flat_evidence.sum(axis=1, keepdims=True)
        normalized = flat_evidence / np.maximum(totals, 1e-30)
        mass = normalized @ unique_masks.astype(np.float64).T
        prevalence = unique_masks.sum(axis=1, dtype=np.float64)[None, :] / np.maximum(
            flat_valid.sum(axis=1, keepdims=True), 1.0
        )
        table = mass - prevalence
        score_tables.append(table)
        mask_ids.append(inverse)
        observed[boundary] = float(np.mean(table[np.arange(samples), inverse]))
    generator = np.random.default_rng(int(seed))
    null = np.empty((permutations, boundaries), dtype=np.float64)
    for iteration in range(permutations):
        for boundary in range(boundaries):
            order = generator.permutation(samples)
            null[iteration, boundary] = float(
                np.mean(score_tables[boundary][np.arange(samples), mask_ids[boundary][order]])
            )
    max_null = np.nanmax(null, axis=1)
    corrected = np.asarray(
        [
            (1.0 + np.count_nonzero(max_null >= value)) / (permutations + 1.0)
            for value in observed
        ],
        dtype=np.float64,
    )
    uncorrected = np.asarray(
        [
            (1.0 + np.count_nonzero(null[:, boundary] >= observed[boundary]))
            / (permutations + 1.0)
            for boundary in range(boundaries)
        ],
        dtype=np.float64,
    )
    return {
        "permutations": int(permutations),
        "seed": int(seed),
        "statistic": "evidence_mass_minus_mask_prevalence",
        "observed_boundary_statistic": observed,
        "null_boundary_mean": np.nanmean(null, axis=0),
        "uncorrected_p": uncorrected,
        "fwer_p": corrected,
    }


def boundary_response_matrix(paired_tails: Any) -> np.ndarray:
    """Mean response of every ordinal tail to every independently toggled cue.

    Input shape is ``(sample, cue, bit={0,1}, response_boundary)``; output rows
    are response boundaries and columns are toggled marker identities.
    """

    values = _float_array(paired_tails, ndim=4)
    if values.shape[2] != 2:
        raise ValueError("paired_tails bit axis must have size two")
    changes = values[:, :, 1, :] - values[:, :, 0, :]
    return changes.mean(axis=0).T


def boundary_response_selectivity(matrix: Any) -> dict[str, float]:
    response = _float_array(matrix, ndim=2)
    if response.shape[0] != response.shape[1]:
        raise ValueError("boundary response matrix must be square")
    magnitude = np.abs(response)
    diagonal = np.diag(magnitude)
    off_diagonal = magnitude[~np.eye(len(response), dtype=bool)]
    diagonal_mean = float(diagonal.mean())
    off_mean = float(off_diagonal.mean()) if off_diagonal.size else 0.0
    return {
        "diagonal_mean_abs": diagonal_mean,
        "off_diagonal_mean_abs": off_mean,
        "diagonal_minus_off_diagonal": diagonal_mean - off_mean,
        "diagonal_to_off_diagonal_ratio": (
            diagonal_mean / off_mean if off_mean > 0.0 else float("inf")
        ),
        "dominant_diagonal_fraction": float(
            np.mean(np.argmax(magnitude, axis=0) == np.arange(response.shape[1]))
        ),
    }


def _average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        stop = start + 1
        while stop < len(values) and sorted_values[stop] == sorted_values[start]:
            stop += 1
        ranks[order[start:stop]] = (start + stop - 1) / 2.0 + 1.0
        start = stop
    return ranks


def spearman_correlation(left: Any, right: Any) -> float:
    x = _float_array(left).reshape(-1)
    y = _float_array(right).reshape(-1)
    if x.shape != y.shape or len(x) < 2:
        raise ValueError("Spearman inputs must have equal length >= 2")
    rx = _average_ranks(x)
    ry = _average_ranks(y)
    if rx.std() == 0.0 or ry.std() == 0.0:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


def bootstrap_spearman_interval(
    left: Any,
    right: Any,
    *,
    replicates: int = 2000,
    seed: int = 991,
    confidence: float = 0.95,
) -> tuple[float, float]:
    x = _float_array(left).reshape(-1)
    y = _float_array(right).reshape(-1)
    if x.shape != y.shape or len(x) < 3:
        raise ValueError("bootstrap inputs must have equal length >= 3")
    if replicates < 1 or not 0.0 < confidence < 1.0:
        raise ValueError("invalid bootstrap configuration")
    generator = np.random.default_rng(int(seed))
    estimates: list[float] = []
    for _ in range(replicates):
        indices = generator.integers(0, len(x), size=len(x))
        estimate = spearman_correlation(x[indices], y[indices])
        if math.isfinite(estimate):
            estimates.append(estimate)
    if not estimates:
        return float("nan"), float("nan")
    alpha = (1.0 - confidence) / 2.0
    return tuple(float(value) for value in np.quantile(estimates, [alpha, 1.0 - alpha]))


def internal_pixel_effect_audit(
    internal_effect: Any,
    random_internal_effect: Any,
    pixel_effect: Any,
    *,
    bootstrap_replicates: int = 2000,
    seed: int = 991,
) -> dict[str, Any]:
    internal = _float_array(internal_effect).reshape(-1)
    random_effect = _float_array(random_internal_effect).reshape(-1)
    pixel = _float_array(pixel_effect).reshape(-1)
    if internal.shape != random_effect.shape or internal.shape != pixel.shape:
        raise ValueError("internal, random, and pixel effects must have equal shape")
    ratio_values = internal / np.maximum(np.abs(random_effect), 1e-12)
    correlation = spearman_correlation(internal, pixel)
    interval = bootstrap_spearman_interval(
        internal,
        pixel,
        replicates=bootstrap_replicates,
        seed=seed,
    )
    return {
        "n_effects": int(len(internal)),
        "median_internal_effect": float(np.median(internal)),
        "median_random_internal_effect": float(np.median(random_effect)),
        "median_internal_minus_random": float(np.median(internal - random_effect)),
        "median_internal_to_random_ratio": float(np.median(ratio_values)),
        "internal_pixel_spearman": correlation,
        "internal_pixel_spearman_ci95": list(interval),
    }


def condition_performance(
    labels: Any,
    predictions_by_condition: Mapping[str, Any],
    groups_by_condition: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    labels_array = np.asarray(labels, dtype=np.int64).reshape(-1)
    result: dict[str, Any] = {}
    for condition, raw_predictions in predictions_by_condition.items():
        predictions = np.asarray(raw_predictions, dtype=np.int64).reshape(-1)
        if predictions.shape != labels_array.shape:
            raise ValueError(f"prediction shape mismatch for {condition}")
        record: dict[str, Any] = {
            "accuracy": float(np.mean(predictions == labels_array)),
            "mae": float(np.mean(np.abs(predictions - labels_array))),
        }
        if groups_by_condition is not None and condition in groups_by_condition:
            groups = np.asarray(groups_by_condition[condition]).reshape(-1)
            if groups.shape != labels_array.shape:
                raise ValueError(f"group shape mismatch for {condition}")
            group_accuracy = {
                str(group): float(np.mean(predictions[groups == group] == labels_array[groups == group]))
                for group in np.unique(groups)
            }
            record["group_accuracy"] = group_accuracy
            record["worst_group_accuracy"] = min(group_accuracy.values())
        result[str(condition)] = record
    if "aligned" in result:
        aligned = result["aligned"]["accuracy"]
        for comparator in ("neutral", "inverted", "clean"):
            if comparator in result:
                result[f"aligned_minus_{comparator}_accuracy"] = (
                    aligned - result[comparator]["accuracy"]
                )
    return result


@dataclass(frozen=True)
class GateBThresholds:
    """Numerical interpretation of the revision plan's Gate B."""

    cue_accuracy_gap_min: float = 0.05
    localization_fwer_alpha: float = 0.05
    localized_boundaries_required: int = 3
    localized_seeds_required: int = 2
    deletion_advantage_min: float = 0.20
    deletion_ratio_min: float = 2.0
    correlation_target: float = 0.50
    correlation_lower_bound_min: float = 0.30
    clean_auprc_lift_max: float = 0.02

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def assess_gate_b(
    shortcut_seed_reports: Sequence[Mapping[str, Any]],
    clean_seed_reports: Sequence[Mapping[str, Any]],
    *,
    thresholds: GateBThresholds | None = None,
) -> dict[str, Any]:
    """Apply the predeclared multi-seed pass/fail gate without silent skips."""

    gate = thresholds or GateBThresholds()
    if not shortcut_seed_reports:
        raise ValueError("Gate B requires shortcut seed reports")
    learned_cue: list[bool] = []
    localized_boundaries: list[int] = []
    deletion_pass: list[bool] = []
    correlation_pass: list[bool] = []
    for report in shortcut_seed_reports:
        performance = report["condition_performance"]
        gap = min(
            float(performance["aligned_minus_neutral_accuracy"]),
            float(performance["aligned_minus_inverted_accuracy"]),
        )
        learned_cue.append(gap >= gate.cue_accuracy_gap_min)
        p_values = np.asarray(report["localization_permutation"]["fwer_p"], dtype=float)
        localized_boundaries.append(int(np.count_nonzero(p_values <= gate.localization_fwer_alpha)))
        effects = report["internal_pixel_effects"]
        deletion_pass.append(
            float(effects["median_internal_minus_random"]) >= gate.deletion_advantage_min
            and float(effects["median_internal_to_random_ratio"]) >= gate.deletion_ratio_min
        )
        lower = float(effects["internal_pixel_spearman_ci95"][0])
        correlation_pass.append(
            float(effects["internal_pixel_spearman"]) >= gate.correlation_target
            and lower > gate.correlation_lower_bound_min
        )

    clean_lifts = [
        float(report["localization"]["macro_auprc_lift"])
        for report in clean_seed_reports
    ]
    checks = {
        "cue_learned_all_seeds": all(learned_cue),
        "localization_seed_requirement": (
            sum(value >= gate.localized_boundaries_required for value in localized_boundaries)
            >= gate.localized_seeds_required
        ),
        "deletion_all_seeds": all(deletion_pass),
        "internal_pixel_correlation_all_seeds": all(correlation_pass),
        "clean_negative_control": (
            bool(clean_lifts) and float(np.median(clean_lifts)) <= gate.clean_auprc_lift_max
        ),
    }
    return {
        "thresholds": gate.as_dict(),
        "per_seed": {
            "cue_learned": learned_cue,
            "localized_boundaries": localized_boundaries,
            "deletion_pass": deletion_pass,
            "correlation_pass": correlation_pass,
            "clean_auprc_lift": clean_lifts,
        },
        "checks": checks,
        "passed": all(checks.values()),
    }


def rasterize_native_ledger(
    local_rate_maps: Mapping[str, torch.Tensor],
    *,
    output_size: tuple[int, int],
) -> torch.Tensor:
    """Mass-preserving visualization of native ledger cells at pixel scale.

    Selection for intervention must still use native cells.  This raster is
    only for mask-based localization metrics and never for selecting a region.
    """

    if not local_rate_maps:
        raise ValueError("local_rate_maps is empty")
    height, width = (int(value) for value in output_size)
    if height < 1 or width < 1:
        raise ValueError("output_size must be positive")

    def overlap_transport(
        source_size: int,
        target_size: int,
        *,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        """Fraction of each source-cell mass assigned to every target cell.

        Source and target lattices partition the same unit interval.  Entry
        ``(o, i)`` is the exact fractional overlap of source cell ``i`` with
        target cell ``o``.  Consequently every source column sums to one and
        transport conserves ledger mass for arbitrary, including odd and
        non-divisible, lattice sizes.
        """

        # Work in FP64 when constructing geometric boundaries.  Casting only
        # after column normalization prevents cumulative coordinate roundoff
        # from creating scale-dependent mass drift in FP32 evidence maps.
        work_dtype = torch.float64
        source_left = torch.arange(
            source_size, dtype=work_dtype, device=device
        )[None, :]
        source_right = source_left + 1.0
        target_left = (
            torch.arange(target_size, dtype=work_dtype, device=device)[:, None]
            * (float(source_size) / float(target_size))
        )
        target_right = target_left + float(source_size) / float(target_size)
        overlap = (
            torch.minimum(target_right, source_right)
            - torch.maximum(target_left, source_left)
        ).clamp_min(0.0)
        overlap = overlap / overlap.sum(dim=0, keepdim=True).clamp_min(
            torch.finfo(work_dtype).tiny
        )
        return overlap.to(dtype=dtype)

    combined: torch.Tensor | None = None
    for name in sorted(local_rate_maps):
        rates = torch.as_tensor(local_rate_maps[name])
        if rates.ndim != 4:
            raise ValueError(f"ledger {name!r} must have shape (N,B,H,W)")
        if not rates.is_floating_point():
            raise TypeError(f"ledger {name!r} must be floating point")
        source_height, source_width = rates.shape[-2:]
        row_transport = overlap_transport(
            source_height,
            height,
            dtype=rates.dtype,
            device=rates.device,
        )
        column_transport = overlap_transport(
            source_width,
            width,
            dtype=rates.dtype,
            device=rates.device,
        )
        # Each operation distributes source-cell mass along one dimension;
        # unlike interpolation, no cell is sampled away and no density/mass
        # conversion factor is implicit.
        resized = torch.matmul(row_transport, rates)
        resized = torch.matmul(resized, column_transport.transpose(0, 1))
        combined = resized if combined is None else combined + resized
    assert combined is not None
    return combined


def json_ready(value: Any) -> Any:
    """Convert metric payloads to strict-JSON-compatible Python values."""

    if isinstance(value, Mapping):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return json_ready(value.tolist())
    if isinstance(value, np.generic):
        return json_ready(value.item())
    if torch.is_tensor(value):
        return json_ready(value.detach().cpu().tolist())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


__all__ = [
    "GateBThresholds",
    "assess_gate_b",
    "binary_average_precision",
    "bootstrap_spearman_interval",
    "boundary_response_matrix",
    "boundary_response_selectivity",
    "condition_performance",
    "familywise_localization_permutation_test",
    "internal_pixel_effect_audit",
    "json_ready",
    "localization_metrics",
    "pointing_accuracy",
    "rasterize_native_ledger",
    "spearman_correlation",
]
