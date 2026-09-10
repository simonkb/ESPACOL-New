#!/usr/bin/env python3
"""Read-only, full-inner-validation structural audit for ORIGIN-v3.

The audit encodes only the inner-validation images and never updates model
parameters. All deletion interventions then replay the stored local rate
ledger; images are not masked or re-encoded for an intervention. Locked outer
images are read only as opaque bytes to verify the pre-existing split hash;
they are never decoded, transformed, inferred on, or evaluated.
"""

from __future__ import annotations

import argparse
import fcntl
import gc
import hashlib
import json
import math
import os
from collections import defaultdict
from dataclasses import fields
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader

from configs.origin_config import OriginConfig
from Datasets.origin_data import (
    OriginFundusTransform,
    OriginGenericTransform,
    OriginImageDataset,
    class_histogram,
    load_origin_items,
    split_origin_items,
    split_paths_are_disjoint,
)
from models.origin import build_origin_model
from train_origin import split_signature
from training.origin_trainer import (
    _architecture_record,
    _canonical_sha256,
    _critical_config,
    evaluate_origin_predictions,
    origin_implementation_signature,
)


_DECISIONS = ("class_map", "posterior_median", "rounded_expected")


def _acquire_nonblocking_lock(path: Path, *, shared: bool) -> Any:
    """Return a held advisory lock or fail instead of waiting."""

    path.parent.mkdir(parents=True, exist_ok=True)
    stream = path.open("a+")
    operation = fcntl.LOCK_SH if shared else fcntl.LOCK_EX
    try:
        fcntl.flock(stream.fileno(), operation | fcntl.LOCK_NB)
    except BlockingIOError:
        stream.close()
        raise RuntimeError(f"audit lock is already held: {path}") from None
    return stream


def _verify_provenance(
    state: Mapping[str, Any],
    cfg: OriginConfig,
    model: torch.nn.Module,
    manifest: Mapping[str, Any],
    train_items: Sequence[tuple[str, int]],
    validation_items: Sequence[tuple[str, int]],
    locked_items: Sequence[tuple[str, int]],
) -> str:
    """Fail closed if code, model, config, or complete split identity changed."""

    if state.get("schema") != "origin-checkpoint-v3":
        raise ValueError("full validation audit requires an ORIGIN-v3 checkpoint")
    if manifest.get("schema") != "origin-split-v2":
        raise ValueError("full validation audit requires an origin-split-v2 manifest")
    if manifest.get("evaluation_scope") != "inner_validation_only":
        raise ValueError("full validation audit requires an inner-validation-only split")
    if state.get("implementation_signature") != origin_implementation_signature():
        raise ValueError("current ORIGIN implementation differs from checkpoint")
    architecture = _architecture_record(model)
    if state.get("architecture") != architecture:
        raise ValueError("reconstructed architecture record differs from checkpoint")
    if state.get("architecture_signature") != _canonical_sha256(architecture):
        raise ValueError("checkpoint architecture signature is invalid")
    critical_config = _critical_config(cfg)
    if state.get("critical_config") != critical_config:
        raise ValueError("reconstructed critical configuration differs from checkpoint")
    if state.get("config_signature") != _canonical_sha256(critical_config):
        raise ValueError("checkpoint configuration signature is invalid")
    if not split_paths_are_disjoint(train_items, validation_items, locked_items):
        raise ValueError("reconstructed ORIGIN splits overlap")
    signature = split_signature(
        ("train", train_items),
        ("validation", validation_items),
        ("locked_test", locked_items),
    )
    if signature != state.get("split_signature") or signature != manifest.get("signature"):
        raise ValueError("recomputed complete split signature differs from checkpoint/manifest")
    expected_counts = {
        "train": len(train_items),
        "validation": len(validation_items),
        "locked_test": len(locked_items),
    }
    expected_histograms = {
        "train": class_histogram(train_items, cfg.n_classes),
        "validation": class_histogram(validation_items, cfg.n_classes),
        "locked_test": class_histogram(locked_items, cfg.n_classes),
    }
    if manifest.get("counts") != expected_counts or manifest.get("histograms") != expected_histograms:
        raise ValueError("split manifest counts or histograms differ from reconstruction")
    if int(manifest.get("fold", -1)) != int(state.get("fold", -2)):
        raise ValueError("checkpoint and split manifest fold differ")
    if manifest.get("dataset") != cfg.dataset:
        raise ValueError("checkpoint configuration and split dataset differ")
    return signature


def _field(output: object, name: str) -> Any:
    return output.get(name) if isinstance(output, Mapping) else getattr(output, name)


def _decision(output: object, rule: str) -> torch.Tensor:
    if rule == "rounded_expected":
        return _field(output, "expected_grade").round().long()
    return _field(output, rule).long()


def _quantiles(
    values: Sequence[float], probabilities: Sequence[float]
) -> dict[str, float | None]:
    if not values:
        return {f"q{100 * value:g}": None for value in probabilities}
    array = np.asarray(values, dtype=np.float64)
    return {
        f"q{100 * probability:g}": float(np.quantile(array, probability))
        for probability in probabilities
    }


def _histogram(values: Iterable[int], size: int) -> list[int]:
    result = [0] * size
    for value in values:
        result[int(value)] += 1
    return result


def _effect_summary(records: Sequence[Mapping[str, Any]], quantiles: Sequence[float]) -> dict[str, Any]:
    if not records:
        return {"n": 0}
    expected = [float(item["expected_grade_delta"]) for item in records]
    grade_delta = [int(item["baseline_prediction"]) - int(item["replayed_prediction"]) for item in records]
    removed = np.asarray([item["removed_boundary_rates"] for item in records], dtype=np.float64)
    cumulative = np.asarray([item["cumulative_probability_delta"] for item in records], dtype=np.float64)
    return {
        "n": len(records),
        "prediction_change_count": sum(value != 0 for value in grade_delta),
        "prediction_change_rate": sum(value != 0 for value in grade_delta) / len(records),
        "mean_prediction_grade_drop": float(np.mean(grade_delta)),
        "mean_expected_grade_delta": float(np.mean(expected)),
        "expected_grade_delta_quantiles": _quantiles(expected, quantiles),
        "mean_removed_boundary_rates": removed.mean(axis=0).tolist(),
        "mean_cumulative_probability_delta": cumulative.mean(axis=0).tolist(),
    }


def _rf_metadata(scale_metadata: Any, spatial: Sequence[int]) -> dict[str, Any]:
    stride = getattr(scale_metadata, "output_stride", None)
    receptive_field = getattr(scale_metadata, "receptive_field", None)
    center_offset = getattr(scale_metadata, "center_offset", None)
    input_size = getattr(scale_metadata, "input_size", None)
    result = {
        "output_stride_pixels": None if stride is None else int(stride),
        "receptive_field_pixels": None if receptive_field is None else int(receptive_field),
        "input_center_yx": None,
        "unclipped_receptive_field_bbox_yxyx": None,
        "clipped_receptive_field_bbox_yxyx": None,
        "clipped_input_coverage_fraction": None,
        "global_input_support": None,
    }
    if (
        len(spatial) != 2
        or stride is None
        or receptive_field is None
        or center_offset is None
        or input_size is None
    ):
        return result
    center_y = float(center_offset + spatial[0] * stride)
    center_x = float(center_offset + spatial[1] * stride)
    half = float(receptive_field) / 2.0
    raw = [center_y - half, center_x - half, center_y + half, center_x + half]
    height, width = (float(input_size[0]), float(input_size[1]))
    clipped = [
        max(0.0, min(height, raw[0])),
        max(0.0, min(width, raw[1])),
        max(0.0, min(height, raw[2])),
        max(0.0, min(width, raw[3])),
    ]
    area = max(0.0, clipped[2] - clipped[0]) * max(0.0, clipped[3] - clipped[1])
    coverage = area / (height * width)
    result.update(
        {
            "input_center_yx": [center_y, center_x],
            "unclipped_receptive_field_bbox_yxyx": raw,
            "clipped_receptive_field_bbox_yxyx": clipped,
            "clipped_input_coverage_fraction": coverage,
            "global_input_support": bool(coverage >= 1.0 - 1e-12),
        }
    )
    return result


def _scale_geometry_metadata(scale_metadata: Any) -> dict[str, Any]:
    """Serialize the theoretical encoder support without implying causality."""

    input_size = getattr(scale_metadata, "input_size", None)
    lattice_size = getattr(scale_metadata, "lattice_size", None)
    receptive_field = getattr(scale_metadata, "receptive_field", None)
    result = {
        "output_stride_pixels": getattr(scale_metadata, "output_stride", None),
        "receptive_field_pixels": receptive_field,
        "center_offset_pixels": getattr(scale_metadata, "center_offset", None),
        "input_size_pixels": None if input_size is None else list(input_size),
        "lattice_size": None if lattice_size is None else list(lattice_size),
        "globally_mixed": bool(getattr(scale_metadata, "globally_mixed", False)),
    }
    if input_size is not None and receptive_field is not None:
        result["rf_extent_exceeds_both_input_dimensions"] = bool(
            receptive_field >= max(input_size)
        )
    else:
        result["rf_extent_exceeds_both_input_dimensions"] = None
    return result


def _assert_distribution(output: object, *, tolerance: float = 2e-10) -> None:
    probs = _field(output, "class_probs")
    cumulative = _field(output, "cumulative_probs")
    expected = _field(output, "expected_grade")
    tensors = (probs, cumulative, expected, _field(output, "total_rates"))
    if not all(bool(torch.isfinite(value).all()) for value in tensors):
        raise FloatingPointError("non-finite ORIGIN replay distribution")
    if bool((probs < -tolerance).any()) or bool((cumulative < -tolerance).any()):
        raise AssertionError("ORIGIN replay produced negative probabilities")
    if not torch.allclose(
        probs.sum(-1), torch.ones_like(probs.sum(-1)), atol=tolerance, rtol=tolerance
    ):
        raise AssertionError("ORIGIN replay class probabilities do not sum to one")
    if cumulative.shape[1] > 1 and bool(
        (cumulative[:, 1:] - cumulative[:, :-1] > tolerance).any()
    ):
        raise AssertionError("ORIGIN cumulative probabilities are not monotone")
    if not torch.allclose(expected, cumulative.sum(-1), atol=tolerance, rtol=tolerance):
        raise AssertionError("expected grade is not the cumulative-probability sum")


def _replay_records(
    model: torch.nn.Module,
    baseline: object,
    removals: Mapping[str, torch.Tensor],
    labels: torch.Tensor,
    sample_ids: torch.Tensor,
    decision_rule: str,
) -> tuple[list[dict[str, Any]], float]:
    intervention = model.replay_without(
        baseline, removals, force_decoder_fp64=True
    )
    replayed = getattr(intervention, "output", intervention)
    removed = getattr(intervention, "removed_rates")
    baseline_rates = _field(baseline, "total_rates")
    replayed_rates = _field(replayed, "total_rates")
    _assert_distribution(baseline)
    _assert_distribution(replayed)
    if not bool(torch.isfinite(removed).all()) or bool((removed < 0).any()):
        raise FloatingPointError("invalid removed ORIGIN rates")
    error = float((baseline_rates - removed - replayed_rates).abs().max().cpu())
    local_maps = _field(baseline, "local_rate_maps")
    ledger_dtype = next(iter(local_maps.values())).dtype
    rate_scale = max(
        1.0,
        float(baseline_rates.abs().max().cpu()),
        float(replayed_rates.abs().max().cpu()),
    )
    tolerance = max(2e-5, 16.0 * torch.finfo(ledger_dtype).eps * rate_scale)
    if not math.isfinite(error) or error > tolerance:
        raise AssertionError(
            f"exact total-rate replay failed: {error:.3g} > {tolerance:.3g}"
        )
    replayed_maps = _field(replayed, "local_rate_maps")
    for scale, original in local_maps.items():
        mask = intervention.removal_masks[scale].bool().unsqueeze(-1)
        removed_map = torch.where(mask, original, torch.zeros_like(original))
        if not torch.equal(original, replayed_maps[scale] + removed_map):
            raise AssertionError(f"exact ledger partition failed at {scale}")
    base_prediction = _decision(baseline, decision_rule).detach().cpu()
    new_prediction = _decision(replayed, decision_rule).detach().cpu()
    base_expected = _field(baseline, "expected_grade").detach().cpu()
    new_expected = _field(replayed, "expected_grade").detach().cpu()
    base_cumulative = _field(baseline, "cumulative_probs").detach().cpu()
    new_cumulative = _field(replayed, "cumulative_probs").detach().cpu()
    removed = removed.detach().cpu()
    cumulative_delta = base_cumulative - new_cumulative
    expected_delta = base_expected - new_expected
    identity_error = (expected_delta - cumulative_delta.sum(-1)).abs().max()
    if float(identity_error) > 2e-10:
        raise AssertionError("deletion expected-grade/cumulative-delta identity failed")
    if bool((cumulative_delta < -2e-10).any()) or bool((expected_delta < -2e-10).any()):
        raise AssertionError("deleting nonnegative evidence increased ordinal severity")
    records = []
    for row in range(len(labels)):
        records.append(
            {
                "sample_id": int(sample_ids[row]),
                "label": int(labels[row]),
                "baseline_prediction": int(base_prediction[row]),
                "replayed_prediction": int(new_prediction[row]),
                "expected_grade_delta": float(expected_delta[row]),
                "removed_boundary_rates": removed[row].tolist(),
                "cumulative_probability_delta": cumulative_delta[row].tolist(),
            }
        )
    return records, error


def _topk_masks(
    rate_maps: Mapping[str, torch.Tensor],
    valid_masks: Mapping[str, torch.Tensor],
    *,
    boundary: int,
    k: int,
) -> tuple[dict[str, torch.Tensor], dict[str, int]]:
    score_parts: list[torch.Tensor] = []
    offsets: dict[str, tuple[int, int]] = {}
    cursor = 0
    for scale, rates in rate_maps.items():
        scores = rates[..., boundary].flatten(1)
        scores = scores.masked_fill(~valid_masks[scale].flatten(1).bool(), -torch.inf)
        score_parts.append(scores)
        offsets[scale] = (cursor, cursor + scores.shape[1])
        cursor += scores.shape[1]
    scores = torch.cat(score_parts, dim=1)
    if bool((torch.isfinite(scores).sum(1) < k).any()):
        raise ValueError(f"top-k={k} exceeds the valid evidence-cell count")
    selected = scores.topk(k, dim=1).indices
    masks: dict[str, torch.Tensor] = {}
    counts: dict[str, int] = {}
    for scale, valid in valid_masks.items():
        start, stop = offsets[scale]
        within = (selected >= start) & (selected < stop)
        local = (selected - start).clamp(0, stop - start - 1)
        flat = torch.zeros_like(valid.flatten(1), dtype=torch.int64)
        flat.scatter_add_(1, local, within.to(torch.int64))
        masks[scale] = (flat > 0).reshape_as(valid)
        counts[scale] = int(within.sum().cpu())
    return masks, counts


def audit_origin_validation(
    model: torch.nn.Module,
    validation_loader: DataLoader,
    *,
    validation_items: Sequence[tuple[str, int]] | None = None,
    decision_rule: str = "class_map",
    top_ks: Sequence[int] = (1, 5, 10),
    quantiles: Sequence[float] = (0.0, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1.0),
    certificates_per_grade: int = 2,
    split_signature: str = "unknown",
    device: torch.device | str = "cpu",
    amp: bool = True,
) -> dict[str, Any]:
    """Audit every item in an inner-validation loader without model updates."""

    if decision_rule not in _DECISIONS:
        raise ValueError(f"decision_rule must be one of {_DECISIONS}")
    top_ks = tuple(sorted(set(int(value) for value in top_ks)))
    if not top_ks or top_ks[0] < 1:
        raise ValueError("top_ks must contain positive integers")
    if certificates_per_grade < 1:
        raise ValueError("certificates_per_grade must be positive")
    device = torch.device(device)
    model.to(device)
    model.eval()

    probabilities: list[torch.Tensor] = []
    cumulative_all: list[torch.Tensor] = []
    predictions: list[torch.Tensor] = []
    labels_all: list[torch.Tensor] = []
    expected_all: list[torch.Tensor] = []
    sample_ids_all: list[torch.Tensor] = []
    rates_all: list[torch.Tensor] = []
    prior_rates_all: list[torch.Tensor] = []
    local_boundary0_all: list[torch.Tensor] = []
    prior_only_grade0_probabilities: list[torch.Tensor] = []
    scale_names: list[str] | None = None
    scale_geometry: dict[str, dict[str, Any]] | None = None
    scale_sums: dict[str, torch.Tensor] = {}
    winning_total: dict[str, list[int]] = defaultdict(list)
    winning_cell: dict[str, list[int]] = defaultdict(list)
    grade_winning_cell: dict[int, dict[str, list[int]]] = defaultdict(lambda: defaultdict(list))
    strongest_effects: list[dict[str, Any]] = []
    all_local_effects: list[dict[str, Any]] = []
    per_scale_effects: dict[str, list[dict[str, Any]]] = defaultdict(list)
    per_scale_max_cell_effects: dict[str, list[dict[str, Any]]] = defaultdict(list)
    topk_effects: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    topk_scale_counts: dict[tuple[int, int], dict[str, int]] = defaultdict(lambda: defaultdict(int))
    argmax: dict[int, dict[str, Any]] = {}
    replay_error = 0.0
    prior_only_identity_error = 0.0
    grade0_factorization_error = 0.0

    with torch.inference_mode():
        for batch in validation_loader:
            if not isinstance(batch, (tuple, list)) or len(batch) != 4:
                raise ValueError("validation batches must be (image, mask, label, index)")
            images, pixel_masks, labels, sample_ids = batch
            images = images.to(device)
            pixel_masks = pixel_masks.to(device)
            labels = labels.long().to(device)
            sample_ids = torch.as_tensor(sample_ids, dtype=torch.long)
            with torch.autocast(
                device_type=device.type,
                enabled=bool(amp and device.type == "cuda"),
            ):
                output = model(
                    images,
                    pixel_valid_mask=pixel_masks,
                    force_decoder_fp64=True,
                )
            rate_maps = dict(_field(output, "local_rate_maps"))
            valid_masks = dict(_field(output, "valid_masks"))
            if scale_names is None:
                scale_names = list(rate_maps)
                metadata = getattr(output, "metadata", {})
                scale_geometry = {
                    scale: _scale_geometry_metadata(metadata.get(scale))
                    for scale in scale_names
                }
            elif list(rate_maps) != scale_names:
                raise ValueError("evidence scale ordering changed between batches")
            total_rates = _field(output, "total_rates")
            prior_rates = _field(output, "prior_rates")
            num_boundaries = total_rates.shape[1]
            for scale, rates in rate_maps.items():
                valid = valid_masks[scale].bool()
                if rates.shape[:-1] != valid.shape or rates.shape[-1] != num_boundaries:
                    raise ValueError(f"invalid local ledger shape at {scale}")
                if not bool(torch.isfinite(rates).all()) or bool((rates < 0).any()):
                    raise FloatingPointError(f"invalid local rates at {scale}")
                if bool((rates.masked_select(~valid.unsqueeze(-1)) != 0).any()):
                    raise AssertionError(f"invalid/background cells contribute at {scale}")

            batch_scale_totals = torch.stack(
                [rate_maps[name].sum(dim=(1, 2)) for name in scale_names], dim=1
            )
            local_boundary0 = batch_scale_totals[:, :, 0].sum(1)
            rate_tolerance = max(
                2e-5,
                16.0 * torch.finfo(next(iter(rate_maps.values())).dtype).eps
                * max(1.0, float(total_rates.abs().max().cpu())),
            )
            if float((
                total_rates[:, 0]
                - prior_rates[:, 0].to(total_rates.dtype)
                - local_boundary0.to(total_rates.dtype)
            ).abs().max()) > rate_tolerance:
                raise AssertionError("boundary-0 total != prior + local evidence")
            batch_scale_peaks = torch.stack(
                [
                    rate_maps[name].masked_fill(
                        ~valid_masks[name].bool().unsqueeze(-1), -torch.inf
                    ).flatten(1, 2).max(dim=1).values
                    for name in scale_names
                ],
                dim=1,
            )
            total_winners = batch_scale_totals.argmax(1).cpu()
            cell_winners = batch_scale_peaks.argmax(1).cpu()
            labels_cpu = labels.cpu()
            for boundary in range(num_boundaries):
                winning_total[str(boundary)].extend(total_winners[:, boundary].tolist())
                winning_cell[str(boundary)].extend(cell_winners[:, boundary].tolist())
                for row, grade in enumerate(labels_cpu.tolist()):
                    grade_winning_cell[grade][str(boundary)].append(
                        int(cell_winners[row, boundary])
                    )
            detached_scale = batch_scale_totals.detach().cpu().sum(dim=0)
            for index, scale in enumerate(scale_names):
                scale_sums[scale] = scale_sums.get(
                    scale, torch.zeros(num_boundaries, dtype=torch.float64)
                ) + detached_scale[index].double()

            # Highest all-boundary single-cell certificate for every sample.
            score_parts = []
            offsets: dict[str, tuple[int, int]] = {}
            cursor = 0
            for scale in scale_names:
                scores = rate_maps[scale].sum(-1).flatten(1)
                scores = scores.masked_fill(~valid_masks[scale].flatten(1).bool(), -torch.inf)
                score_parts.append(scores)
                offsets[scale] = (cursor, cursor + scores.shape[1])
                cursor += scores.shape[1]
            strongest = torch.cat(score_parts, dim=1).argmax(1)
            strongest_masks = {
                scale: torch.zeros_like(valid_masks[scale], dtype=torch.bool)
                for scale in scale_names
            }
            strongest_meta: list[tuple[str, list[int], float]] = []
            combined_scores = torch.cat(score_parts, dim=1)
            for row, selected in enumerate(strongest.tolist()):
                for scale in scale_names:
                    start, stop = offsets[scale]
                    if start <= selected < stop:
                        local = selected - start
                        spatial = list(np.unravel_index(local, valid_masks[scale].shape[1:]))
                        strongest_masks[scale][(row, *spatial)] = True
                        strongest_meta.append(
                            (scale, spatial, float(combined_scores[row, selected].cpu()))
                        )
                        break
            records, error = _replay_records(
                model, output, strongest_masks, labels_cpu, sample_ids, decision_rule
            )
            replay_error = max(replay_error, error)
            metadata = getattr(output, "metadata", {})
            base_probs = _field(output, "class_probs").detach().cpu()
            for row, record in enumerate(records):
                scale, spatial, value = strongest_meta[row]
                scale_metadata = metadata.get(scale) if isinstance(metadata, Mapping) else None
                record.update(
                    {
                        "selection": "largest_raw_single_cell_sum_across_boundaries",
                        "removed_scale": scale,
                        "removed_spatial_index": spatial,
                        "removed_local_rate_sum": value,
                        "baseline_class_probs": base_probs[row].tolist(),
                    }
                )
                record.update(_rf_metadata(scale_metadata, spatial))
                index = int(record["sample_id"])
                if validation_items is not None and 0 <= index < len(validation_items):
                    record["image_id"] = Path(validation_items[index][0]).name
                strongest_effects.append(record)

            # Delete each complete scale independently.
            for scale in scale_names:
                removals = {
                    name: (valid_masks[name].bool() if name == scale else torch.zeros_like(valid_masks[name], dtype=torch.bool))
                    for name in scale_names
                }
                records, error = _replay_records(
                    model, output, removals, labels_cpu, sample_ids, decision_rule
                )
                replay_error = max(replay_error, error)
                per_scale_effects[scale].extend(records)

                raw_scores = rate_maps[scale].sum(-1).masked_fill(
                    ~valid_masks[scale].bool(), -torch.inf
                )
                flat_scores = raw_scores.flatten(1)
                if bool((torch.isfinite(flat_scores).sum(1) == 0).any()):
                    raise AssertionError(f"scale {scale} has no valid evidence cell")
                selected = flat_scores.argmax(1)
                cell_mask = torch.zeros_like(valid_masks[scale], dtype=torch.bool)
                selected_meta: list[tuple[list[int], float]] = []
                for row, flat_index in enumerate(selected.tolist()):
                    spatial = list(
                        np.unravel_index(flat_index, valid_masks[scale].shape[1:])
                    )
                    cell_mask[(row, *spatial)] = True
                    selected_meta.append(
                        (spatial, float(flat_scores[row, flat_index].cpu()))
                    )
                cell_removals = {
                    name: (
                        cell_mask
                        if name == scale
                        else torch.zeros_like(valid_masks[name], dtype=torch.bool)
                    )
                    for name in scale_names
                }
                cell_records, error = _replay_records(
                    model, output, cell_removals, labels_cpu, sample_ids, decision_rule
                )
                replay_error = max(replay_error, error)
                scale_metadata = metadata.get(scale) if isinstance(metadata, Mapping) else None
                for row, record in enumerate(cell_records):
                    spatial, value = selected_meta[row]
                    record.update(
                        {
                            "selection": "largest_raw_single_cell_sum_across_boundaries_within_scale",
                            "removed_scale": scale,
                            "removed_spatial_index": spatial,
                            "removed_local_rate_sum": value,
                            "baseline_class_probs": base_probs[row].tolist(),
                        }
                    )
                    record.update(_rf_metadata(scale_metadata, spatial))
                    index = int(record["sample_id"])
                    if validation_items is not None and 0 <= index < len(validation_items):
                        record["image_id"] = Path(validation_items[index][0]).name
                    per_scale_max_cell_effects[scale].append(record)

            all_local_removals = {
                name: valid_masks[name].bool() for name in scale_names
            }
            all_local_records, error = _replay_records(
                model, output, all_local_removals, labels_cpu, sample_ids, decision_rule
            )
            all_local_effects.extend(all_local_records)
            replay_error = max(replay_error, error)
            prior_only = model.replay_without(
                output, all_local_removals, force_decoder_fp64=True
            ).output
            full_p0 = _field(output, "class_probs")[:, 0].double()
            prior_only_p0 = _field(prior_only, "class_probs")[:, 0].double()
            expected_local_factor = torch.exp(-local_boundary0.double())
            prior_only_identity_error = max(
                prior_only_identity_error,
                float((_field(prior_only, "total_rates") - prior_rates.to(
                    _field(prior_only, "total_rates").dtype
                )).abs().max().cpu()),
                float((_field(prior_only, "class_probs")[:, 0] - torch.exp(
                    -prior_rates[:, 0].double()
                )).abs().max().cpu()),
            )
            if prior_only_identity_error > 2e-10:
                raise AssertionError("prior-only boundary-0 identity failed")
            grade0_factorization_error = max(
                grade0_factorization_error,
                float((
                    full_p0 / prior_only_p0.clamp_min(torch.finfo(torch.float64).tiny)
                    - expected_local_factor
                ).abs().max().cpu()),
            )
            if grade0_factorization_error > 2e-10:
                raise AssertionError("grade-0 local-evidence factorization failed")
            prior_only_grade0_probabilities.append(prior_only_p0.detach().cpu())

            # Delete the k strongest witnesses separately for every boundary.
            for boundary in range(num_boundaries):
                for k in top_ks:
                    removals, selected_counts = _topk_masks(
                        rate_maps, valid_masks, boundary=boundary, k=k
                    )
                    records, error = _replay_records(
                        model, output, removals, labels_cpu, sample_ids, decision_rule
                    )
                    replay_error = max(replay_error, error)
                    topk_effects[(boundary, k)].extend(records)
                    for scale, count in selected_counts.items():
                        topk_scale_counts[(boundary, k)][scale] += count

            total_cpu = total_rates.detach().cpu()
            prediction_cpu = _decision(output, decision_rule).detach().cpu()
            expected_cpu = _field(output, "expected_grade").detach().cpu()
            for boundary in range(num_boundaries):
                value, row = total_cpu[:, boundary].max(dim=0)
                if boundary not in argmax or float(value) > argmax[boundary]["rate"]:
                    row = int(row)
                    index = int(sample_ids[row])
                    scale_decomposition = {
                        scale: float(batch_scale_totals[row, scale_index, boundary].cpu())
                        for scale_index, scale in enumerate(scale_names)
                    }
                    prior_value = float(prior_rates[row, boundary].cpu())
                    local_value = sum(scale_decomposition.values())
                    argmax[boundary] = {
                        "rate": float(value),
                        "sample_id": index,
                        "image_id": (
                            Path(validation_items[index][0]).name
                            if validation_items is not None and 0 <= index < len(validation_items)
                            else None
                        ),
                        "label": int(labels_cpu[row]),
                        "prediction": int(prediction_cpu[row]),
                        "expected_grade": float(expected_cpu[row]),
                        "class_probs": base_probs[row].tolist(),
                        "total_rates": total_cpu[row].tolist(),
                        "per_scale_raw_boundary_rates": scale_decomposition,
                        "largest_raw_rate_scale": max(
                            scale_decomposition, key=scale_decomposition.get
                        ),
                        "prior_boundary_rate": prior_value,
                        "summed_local_boundary_rate": local_value,
                        "rate_reconstruction_error": abs(
                            float(value) - prior_value - local_value
                        ),
                        "rate_cap_fraction": float(
                            value / float(getattr(output, "total_rate_cap", 64.0))
                        ),
                    }

            probabilities.append(_field(output, "class_probs").detach().cpu())
            cumulative_all.append(_field(output, "cumulative_probs").detach().cpu())
            predictions.append(prediction_cpu)
            labels_all.append(labels_cpu)
            expected_all.append(expected_cpu)
            sample_ids_all.append(sample_ids.cpu())
            rates_all.append(total_cpu)
            prior_rates_all.append(prior_rates.detach().cpu())
            local_boundary0_all.append(local_boundary0.detach().cpu())

    if scale_names is None:
        raise ValueError("validation loader is empty")
    if scale_geometry is None:  # pragma: no cover - coupled to scale_names
        raise AssertionError("missing ORIGIN scale geometry metadata")
    probs = torch.cat(probabilities)
    cumulative = torch.cat(cumulative_all)
    predicted = torch.cat(predictions)
    labels = torch.cat(labels_all)
    expected = torch.cat(expected_all)
    sample_ids = torch.cat(sample_ids_all)
    total_rates = torch.cat(rates_all).double()
    prior_rates = torch.cat(prior_rates_all).double()
    local_boundary0 = torch.cat(local_boundary0_all).double()
    prior_only_p0 = torch.cat(prior_only_grade0_probabilities).double()
    num_classes = probs.shape[1]
    num_boundaries = total_rates.shape[1]
    metrics = evaluate_origin_predictions(probs, predicted, labels)
    if validation_items is not None:
        expected_ids = list(range(len(validation_items)))
        observed_ids = sample_ids.tolist()
        if sorted(observed_ids) != expected_ids or len(set(observed_ids)) != len(expected_ids):
            raise AssertionError("validation loader sample IDs are incomplete or duplicated")
        for sample_id, label in zip(observed_ids, labels.tolist()):
            if int(validation_items[sample_id][1]) != int(label):
                raise AssertionError("validation loader label does not match split item")
    grade_metrics = {}
    for grade in range(num_classes):
        selected = labels == grade
        grade_metrics[str(grade)] = {
            "support": int(selected.sum()),
            "recall": float((predicted[selected] == grade).float().mean()),
            "prediction_histogram": _histogram(predicted[selected].tolist(), num_classes),
            "mean_absolute_error": float((predicted[selected] - grade).abs().float().mean()),
            "mean_expected_grade": float(expected[selected].mean()),
            "boundary_rate_quantiles": {
                str(boundary): _quantiles(total_rates[selected, boundary].tolist(), quantiles)
                for boundary in range(num_boundaries)
            },
            "winning_single_cell_scale_histograms": {
                str(boundary): {
                    scale: _histogram(
                        grade_winning_cell[grade][str(boundary)], len(scale_names)
                    )[index]
                    for index, scale in enumerate(scale_names)
                }
                for boundary in range(num_boundaries)
            },
        }

    local_total = sum(scale_sums.values())
    per_scale_rates = {}
    for scale in scale_names:
        per_scale_rates[scale] = {
            "sum_boundary_rates": scale_sums[scale].tolist(),
            "share_by_boundary": (
                scale_sums[scale] / local_total.clamp_min(torch.finfo(torch.float64).tiny)
            ).tolist(),
            "overall_share": float(scale_sums[scale].sum() / local_total.sum()),
        }
    winner_summary = {
        "largest_raw_rate_scale_total_by_boundary": {
            str(boundary): {
                scale: _histogram(winning_total[str(boundary)], len(scale_names))[index]
                for index, scale in enumerate(scale_names)
            }
            for boundary in range(num_boundaries)
        },
        "largest_raw_rate_single_cell_by_boundary": {
            str(boundary): {
                scale: _histogram(winning_cell[str(boundary)], len(scale_names))[index]
                for index, scale in enumerate(scale_names)
            }
            for boundary in range(num_boundaries)
        },
    }

    # Deterministic grade-stratified subset; ranking is unrelated to effect size.
    certificates = []
    for grade in range(num_classes):
        members = [item for item in strongest_effects if int(item["label"]) == grade]
        members.sort(
            key=lambda item: hashlib.sha256(
                f"{split_signature}:{grade}:{item['sample_id']}".encode()
            ).hexdigest()
        )
        certificates.extend(members[:certificates_per_grade])

    scale_certificates: dict[str, list[dict[str, Any]]] = {}
    for scale in scale_names:
        selected_certificates: list[dict[str, Any]] = []
        for grade in range(num_classes):
            members = [
                item
                for item in per_scale_max_cell_effects[scale]
                if int(item["label"]) == grade
            ]
            members.sort(
                key=lambda item: hashlib.sha256(
                    f"{split_signature}:{scale}:{grade}:{item['sample_id']}".encode()
                ).hexdigest()
            )
            selected_certificates.extend(members[:certificates_per_grade])
        scale_certificates[scale] = selected_certificates

    def boundary0_group(selected: torch.Tensor) -> dict[str, Any]:
        support = int(selected.sum())
        if support == 0:
            return {"support": 0}
        local_factor = torch.exp(-local_boundary0[selected])
        return {
            "support": support,
            "mean_total_boundary0_rate": float(total_rates[selected, 0].mean()),
            "mean_local_boundary0_evidence": float(local_boundary0[selected].mean()),
            "mean_prior_boundary0_rate": float(prior_rates[selected, 0].mean()),
            "local_boundary0_quantiles": _quantiles(local_boundary0[selected].tolist(), quantiles),
            "prior_boundary0_quantiles": _quantiles(prior_rates[selected, 0].tolist(), quantiles),
            "mean_grade0_probability": float(probs[selected, 0].mean()),
            "mean_prior_only_grade0_probability": float(prior_only_p0[selected].mean()),
            "mean_full_to_prior_grade0_probability_ratio": float(
                (probs[selected, 0].double() / prior_only_p0[selected]).mean()
            ),
            "mean_exp_negative_local_boundary0_evidence": float(local_factor.mean()),
        }

    grade0_boundary0 = {
        "identity": "total_boundary0_rate = prior_boundary0_rate + summed_local_boundary0_evidence",
        "prior_only_identity": "P(Y=0 | all local rates deleted) = exp(-prior_boundary0_rate)",
        "local_factorization_identity": (
            "P_full(Y=0) / P_prior_only(Y=0) = exp(-summed_local_boundary0_evidence)"
        ),
        "max_prior_only_identity_error": prior_only_identity_error,
        "max_local_factorization_identity_error": grade0_factorization_error,
        "true_grade0": boundary0_group(labels == 0),
        "predicted_grade0": boundary0_group(predicted == 0),
    }
    generator = getattr(model, "generator", None)
    cap = float(getattr(generator, "total_rate_cap", 64.0))
    boundary_scales = getattr(generator, "boundary_scales", None)
    scale_simplex = getattr(generator, "scale_simplex", None)
    if torch.is_tensor(boundary_scales):
        boundary_scales = boundary_scales.detach().double().cpu()
    if torch.is_tensor(scale_simplex):
        scale_simplex = scale_simplex.detach().double().cpu()
    generator_parameterization = {
        "total_rate_cap": cap,
        "prior_rate_cap": getattr(generator, "prior_rate_cap", None),
        "boundary_scale_cap": getattr(generator, "boundary_scale_cap", None),
        "atom_mass_cap": getattr(generator, "atom_mass_cap", None),
        "rate_roundoff_margin": getattr(generator, "rate_roundoff_margin", None),
        "reference_count": getattr(generator, "reference_count", None),
        "prior_rates_by_boundary": prior_rates[0].tolist(),
        "boundary_scales": (
            None if boundary_scales is None else boundary_scales.tolist()
        ),
        "scale_simplex_by_scale_and_boundary": (
            None
            if scale_simplex is None
            else {
                scale: scale_simplex[index].tolist()
                for index, scale in enumerate(scale_names)
            }
        ),
    }
    cap_fractions = total_rates / cap
    rate_cap_diagnostics = {
        "total_rate_cap": cap,
        "max_cap_fraction": float(cap_fractions.max()),
        "sample_count_any_boundary_at_or_above_80pct": int(
            cap_fractions.ge(0.8).any(1).sum()
        ),
        "fraction_any_boundary_at_or_above_80pct": float(
            cap_fractions.ge(0.8).any(1).double().mean()
        ),
        "per_boundary": {
            str(boundary): {
                "max_cap_fraction": float(cap_fractions[:, boundary].max()),
                "sample_count_at_or_above_80pct": int(
                    cap_fractions[:, boundary].ge(0.8).sum()
                ),
                "cumulative_probability_min": float(cumulative[:, boundary].min()),
                "cumulative_probability_max": float(cumulative[:, boundary].max()),
            }
            for boundary in range(num_boundaries)
        },
    }

    return {
        "schema": "origin-full-validation-audit-v1",
        "scope": "inner_validation_only",
        "n": len(labels),
        "decision_rule": decision_rule,
        "split_signature": split_signature,
        "metrics": metrics,
        "grade_stratified_metrics": grade_metrics,
        "grade0_boundary0_evidence_diagnostics": grade0_boundary0,
        "generator_parameterization": generator_parameterization,
        "scale_geometry_and_theoretical_support": scale_geometry,
        "rate_cap_and_saturation_diagnostics": rate_cap_diagnostics,
        "boundary_total_rate_quantiles": {
            str(boundary): _quantiles(total_rates[:, boundary].tolist(), quantiles)
            for boundary in range(num_boundaries)
        },
        "boundary_argmax_samples": {str(key): value for key, value in argmax.items()},
        "per_scale_raw_rate_contributions": per_scale_rates,
        "winning_scale_histograms": winner_summary,
        "largest_raw_rate_single_cell_deletion_effects": {
            "overall": _effect_summary(strongest_effects, quantiles),
            "by_grade": {
                str(grade): _effect_summary(
                    [item for item in strongest_effects if int(item["label"]) == grade],
                    quantiles,
                )
                for grade in range(num_classes)
            },
        },
        "all_local_evidence_deletion_effects": {
            "overall": _effect_summary(all_local_effects, quantiles),
            "by_grade": {
                str(grade): _effect_summary(
                    [item for item in all_local_effects if int(item["label"]) == grade],
                    quantiles,
                )
                for grade in range(num_classes)
            },
        },
        "per_scale_deletion_effects": {
            scale: {
                "overall": _effect_summary(records, quantiles),
                "by_grade": {
                    str(grade): _effect_summary(
                        [item for item in records if int(item["label"]) == grade],
                        quantiles,
                    )
                    for grade in range(num_classes)
                },
            }
            for scale, records in per_scale_effects.items()
        },
        "per_scale_largest_raw_rate_cell_deletion_effects": {
            scale: {
                "overall": _effect_summary(records, quantiles),
                "by_grade": {
                    str(grade): _effect_summary(
                        [item for item in records if int(item["label"]) == grade],
                        quantiles,
                    )
                    for grade in range(num_classes)
                },
            }
            for scale, records in per_scale_max_cell_effects.items()
        },
        "topk_deletion_effects": {
            str(boundary): {
                str(k): {
                    "ranking": "largest_raw_local_rate_for_this_boundary",
                    "effects": _effect_summary(topk_effects[(boundary, k)], quantiles),
                    "selected_scale_histogram": dict(topk_scale_counts[(boundary, k)]),
                }
                for k in top_ks
            }
            for boundary in range(num_boundaries)
        },
        "grade_stratified_certificates": certificates,
        "grade_and_scale_stratified_certificates": scale_certificates,
        "certificate_selection": (
            "deterministic_sha256_rank_within_true_grade_after_largest_raw_rate_cell_selection; "
            "certificate sampling is independent_of_intervention_effect_size"
        ),
        "max_exact_total_rate_replay_error": replay_error,
        "interpretation_scope": "exact_stored_ledger_deletion_not_causal_pixel_masking",
        "validation_sample_ids": sample_ids.tolist(),
    }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _assert_metric_reproduction(
    observed: Mapping[str, Any], expected: Mapping[str, Any], *, source: str
) -> None:
    scalar_keys = ("acc", "mae", "qwk", "ece", "balanced_acc", "macro_f1")
    for key in scalar_keys:
        if key not in expected or not math.isclose(
            float(observed[key]), float(expected[key]), rel_tol=1e-5, abs_tol=1e-5
        ):
            raise AssertionError(f"validation metric {key} does not reproduce {source}")
    for key in ("confusion", "per_grade_recall", "per_grade_support"):
        if key in expected and not np.allclose(
            np.asarray(observed[key]), np.asarray(expected[key]), rtol=1e-5, atol=1e-5
        ):
            raise AssertionError(f"validation metric {key} does not reproduce {source}")


def _parse_ints(value: str) -> tuple[int, ...]:
    result = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    if not result or min(result) < 1:
        raise argparse.ArgumentTypeError("expected comma-separated positive integers")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--output", default=None)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--top_ks", type=_parse_ints, default=(1, 5, 10))
    parser.add_argument("--certificates_per_grade", type=int, default=2)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("the production full-validation audit is CUDA-only")
    checkpoint_path = Path(args.checkpoint).resolve()
    fold_dir = checkpoint_path.parent
    destination = (
        Path(args.output)
        if args.output
        else fold_dir / "audits" / "full_validation_audit_v1.json"
    )
    if destination.exists() and not args.overwrite:
        raise FileExistsError(f"refusing to overwrite existing audit: {destination}")
    output_lock = _acquire_nonblocking_lock(
        destination.parent / ".full_validation_audit.lock", shared=False
    )
    writer_lock = _acquire_nonblocking_lock(fold_dir / ".writer.lock", shared=True)
    result_path = fold_dir / "result.json"
    if not result_path.is_file():
        raise FileNotFoundError("audit requires a completed fold result.json")
    with result_path.open() as stream:
        completed_result = json.load(stream)
    checkpoint_hash_before = _sha256_file(checkpoint_path)
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if state.get("schema") != "origin-checkpoint-v3":
        raise ValueError("full validation audit requires an ORIGIN-v3 checkpoint")
    config_values = dict(state["config"])
    allowed = {field.name for field in fields(OriginConfig)}
    cfg = OriginConfig(**{key: value for key, value in config_values.items() if key in allowed})
    if args.batch_size != cfg.batch_size:
        raise ValueError(
            "the production audit batch size must reproduce the checkpoint "
            f"validation batch size ({cfg.batch_size})"
        )
    fold = int(state["fold"])
    manifest_path = fold_dir / "split_manifest.json"
    with manifest_path.open() as stream:
        manifest = json.load(stream)
    if manifest.get("schema") != "origin-split-v2":
        raise ValueError("full validation audit requires an origin-split-v2 manifest")
    if int(completed_result.get("best_epoch", -1)) != int(state.get("epoch", -2)):
        raise ValueError("best checkpoint epoch differs from completed result")
    if int(completed_result.get("fold", -1)) != fold:
        raise ValueError("completed result fold differs from checkpoint")
    if bool(completed_result.get("test_evaluated", False)):
        raise ValueError("development audit refuses a result that evaluated the outer test")

    items = load_origin_items(
        cfg.dataset,
        args.data_root,
        labels_csv=cfg.labels_csv,
        image_column=cfg.image_column,
        label_column=cfg.label_column,
        image_dir=cfg.image_dir,
    )
    train_items, validation_items, locked_outer_items = split_origin_items(
        cfg.dataset,
        items,
        fold,
        n_folds=cfg.n_folds,
        val_fraction=cfg.val_fraction,
        seed=cfg.seed,
    )
    fundus = cfg.dataset in {"aptos", "dr"}
    transform = (
        OriginFundusTransform(cfg.img_size, augment=False)
        if fundus
        else OriginGenericTransform(cfg.img_size, augment=False)
    )
    loader = DataLoader(
        OriginImageDataset(validation_items, transform),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=False,
    )
    model = build_origin_model(
        num_classes=cfg.n_classes,
        encoder_name=cfg.encoder,
        pretrained=False,
        evidence_scales=cfg.evidence_scales,
        projection_dim=cfg.projection_dim,
        reference_count=cfg.reference_count,
        atom_rate_init=cfg.atom_rate_init,
        prior_rate_init=cfg.prior_rate_init,
        boundary_scale_init=cfg.boundary_scale_init,
        total_rate_cap=cfg.total_rate_cap,
        prior_rate_cap=cfg.prior_rate_cap,
        boundary_scale_cap=cfg.boundary_scale_cap,
        rate_roundoff_margin=cfg.rate_roundoff_margin,
        atom_mode=cfg.atom_mode,
        hybrid_cumulative_init=cfg.hybrid_cumulative_init,
        evidence_dropout=cfg.evidence_dropout,
        mask_valid_fraction=cfg.mask_valid_fraction,
        grad_checkpoint=False,
    )
    signature = _verify_provenance(
        state,
        cfg,
        model,
        manifest,
        train_items,
        validation_items,
        locked_outer_items,
    )
    # No Dataset/DataLoader is ever constructed for this list.
    del train_items, locked_outer_items
    model.load_state_dict(state["model_state"], strict=True)
    checkpoint_epoch = int(state["epoch"])
    architecture_signature = str(state["architecture_signature"])
    implementation_signature = str(state["implementation_signature"])
    checkpoint_config_signature = str(state["config_signature"])
    checkpoint_metrics = dict(state.get("metrics", {}))
    del state
    gc.collect()
    device = torch.device("cuda")
    payload = audit_origin_validation(
        model,
        loader,
        validation_items=validation_items,
        decision_rule=cfg.decision_rule,
        top_ks=args.top_ks,
        certificates_per_grade=args.certificates_per_grade,
        split_signature=signature,
        device=device,
        amp=cfg.amp,
    )
    payload.update(
        {
            "fold": fold,
            "checkpoint_epoch": checkpoint_epoch,
            "checkpoint_sha256": checkpoint_hash_before,
            "architecture_signature": architecture_signature,
            "implementation_signature": implementation_signature,
            "checkpoint_config_signature": checkpoint_config_signature,
            "audit_implementation_sha256": _sha256_file(Path(__file__).resolve()),
            "audit_settings": {
                "batch_size": args.batch_size,
                "num_workers": args.num_workers,
                "top_ks": list(args.top_ks),
                "certificates_per_grade": args.certificates_per_grade,
                "amp": bool(cfg.amp),
                "decoder_precision": "fp64",
            },
        }
    )
    result_metrics = completed_result.get(
        "best_validation", completed_result.get("best_validation_metrics", {})
    )
    _assert_metric_reproduction(payload["metrics"], checkpoint_metrics, source="checkpoint")
    _assert_metric_reproduction(payload["metrics"], result_metrics, source="completed result")
    checkpoint_hash_after = _sha256_file(checkpoint_path)
    if checkpoint_hash_after != checkpoint_hash_before:
        raise AssertionError("checkpoint changed while read-only audit was running")
    payload["metric_reproduction"] = {
        "checkpoint": True,
        "completed_result": True,
    }
    checksum_payload = dict(payload)
    encoded = json.dumps(checksum_payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    payload["content_checksum_sha256"] = hashlib.sha256(encoded).hexdigest()
    destination.parent.mkdir(parents=True, exist_ok=True)
    _write_json_atomic(destination, payload)
    print(json.dumps({"output": str(destination), "metrics": payload["metrics"]}, indent=2))
    # Keep both advisory locks alive through the atomic write.
    output_lock.close()
    writer_lock.close()


if __name__ == "__main__":
    main()
