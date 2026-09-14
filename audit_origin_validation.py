#!/usr/bin/env python3
"""Read-only, full-inner-validation structural audit for ORIGIN.

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
from models.origin import (
    aggregate_ordinal_pair_messages,
    bounded_rate_log_odds_merge,
    build_origin_model,
    reverse_cumulative_atoms,
)
from train_origin import split_signature
from training.origin_trainer import (
    _architecture_record,
    _canonical_sha256,
    _critical_config,
    evaluate_origin_predictions,
    origin_implementation_signature,
)


_DECISIONS = ("class_map", "posterior_median", "rounded_expected")
_V7_SPARSE_AGGREGATION = "fixed_budget_linear_conserved_v1"
_V8_WITNESS_AGGREGATION = "conserved_sparse_witness_allocation_v1"
_V8_RELATION_VARIANTS = {
    "identified_conserved_witness_v1",
    "additive_conserved_witness_control_v1",
    "shuffled_conserved_witness_control_v1",
}
_V8_CHECKPOINT_ROLES = {
    "hash_bound_v3_floor",
    "deployable_selected",
    "best_learned",
    "resume_state",
}


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

    relation_variant = str(getattr(cfg, "relation_variant", "dense_v1"))
    expected_schema = (
        (
            "origin-checkpoint-v6"
            if relation_variant == "dense_v1"
            else (
                "origin-checkpoint-v8"
                if relation_variant in _V8_RELATION_VARIANTS
                else "origin-checkpoint-v7"
            )
        )
        if cfg.relation_enabled
        else "origin-checkpoint-v3"
    )
    if state.get("schema") != expected_schema:
        raise ValueError(
            f"full validation audit requires checkpoint schema {expected_schema!r}"
        )
    if expected_schema == "origin-checkpoint-v8":
        if state.get("relation_protocol_version") != "v8":
            raise ValueError("v8 checkpoint omits its relation protocol identity")
        if state.get("relation_variant") != relation_variant:
            raise ValueError("v8 checkpoint relation variant differs from configuration")
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


def _relation_effect_summary(
    records: Sequence[Mapping[str, Any]],
    quantiles: Sequence[float],
) -> dict[str, Any]:
    if not records:
        return {"n": 0}
    expected = [float(item["expected_grade_delta"]) for item in records]
    grade_delta = [
        int(item["baseline_prediction"]) - int(item["replayed_prediction"])
        for item in records
    ]
    cumulative = np.asarray(
        [item["cumulative_probability_delta"] for item in records],
        dtype=np.float64,
    )
    absolute_expected = [abs(value) for value in expected]
    effect_tolerance = 1e-5
    result = {
        "n": len(records),
        "prediction_change_count": sum(value != 0 for value in grade_delta),
        "prediction_change_rate": sum(value != 0 for value in grade_delta) / len(records),
        "mean_prediction_grade_delta": float(np.mean(grade_delta)),
        "mean_expected_grade_delta": float(np.mean(expected)),
        "mean_absolute_expected_grade_delta": float(np.mean(np.abs(expected))),
        "expected_grade_delta_quantiles": _quantiles(expected, quantiles),
        "absolute_expected_grade_delta_quantiles": _quantiles(
            absolute_expected, quantiles
        ),
        "positive_effect_count_above_1e-5": sum(
            value > effect_tolerance for value in expected
        ),
        "negative_effect_count_below_minus_1e-5": sum(
            value < -effect_tolerance for value in expected
        ),
        "near_zero_effect_count_at_1e-5": sum(
            abs(value) <= effect_tolerance for value in expected
        ),
        "mean_removed_edge_message_l1": float(
            np.mean([item["removed_edge_message_l1"] for item in records])
        ),
        "mean_removed_edge_message_signed_sum": float(
            np.mean([item["removed_edge_message_signed_sum"] for item in records])
        ),
        "mean_cumulative_probability_delta": cumulative.mean(axis=0).tolist(),
    }
    if all("removed_edge_log_odds_l1" in item for item in records):
        result["mean_removed_edge_log_odds_l1"] = float(
            np.mean([item["removed_edge_log_odds_l1"] for item in records])
        )
        result["mean_removed_edge_log_odds_signed_sum"] = float(
            np.mean(
                [item["removed_edge_log_odds_signed_sum"] for item in records]
            )
        )
    return result


def _masked_directed_anova_projection(
    scores: torch.Tensor,
    candidate_edge_mask: torch.Tensor,
) -> torch.Tensor:
    """Independently project pair scores off endpoint-only main effects.

    The candidate graph must be the complete directed graph without self edges
    on each sample's valid regions.  The returned residual is orthogonal to an
    intercept, every target indicator, and every source indicator.  This
    audit implementation is intentionally separate from the model helper.
    """

    if scores.ndim != 4 or scores.shape[-1] != scores.shape[-2]:
        raise ValueError("pair scores must have shape (N,B,M,M)")
    batch, _, regions, _ = scores.shape
    if candidate_edge_mask.shape != (batch, regions, regions):
        raise ValueError("candidate edge mask must have shape (N,M,M)")
    candidate = candidate_edge_mask.bool()
    projected = torch.zeros_like(scores, dtype=torch.float64)
    working = scores.to(dtype=torch.float64)
    diagonal = torch.eye(regions, dtype=torch.bool, device=scores.device)

    for row in range(batch):
        region_valid = candidate[row].any(dim=-1) | candidate[row].any(dim=-2)
        expected = (
            region_valid[:, None]
            & region_valid[None, :]
            & ~diagonal
        )
        if not torch.equal(candidate[row], expected):
            raise AssertionError(
                "candidate relation graph is not complete directed off-diagonal"
            )
        indices = region_valid.nonzero(as_tuple=False).flatten()
        count = int(indices.numel())
        if count < 3:
            # With fewer than three endpoints the directed graph contains no
            # identifiable residual after target/source main effects.
            continue
        sub = working[row][:, indices][:, :, indices]
        row_sum = sub.sum(dim=-1)
        column_sum = sub.sum(dim=-2)
        total = row_sum.sum(dim=-1, keepdim=True)
        gauge = total / (2.0 * float(count - 1))
        denominator = float(count * (count - 2))
        target_main = (
            float(count - 1) * (row_sum - gauge)
            + (column_sum - gauge)
        ) / denominator
        source_main = (
            (row_sum - gauge)
            + float(count - 1) * (column_sum - gauge)
        ) / denominator
        residual = sub - target_main[..., :, None] - source_main[..., None, :]
        residual = residual.masked_fill(
            torch.eye(count, dtype=torch.bool, device=scores.device)[None], 0.0
        )
        for local_target, global_target in enumerate(indices.tolist()):
            projected[row, :, global_target, indices] = residual[:, local_target]
    return projected


def _audit_sparse_relation_trace(relation: object) -> dict[str, Any]:
    """Fail-closed structural audit of an identified sparse relation trace."""

    variant = str(getattr(relation, "variant", "dense_v1"))
    aggregation_kind = str(
        getattr(relation, "aggregation_kind", "geometry_normalized_tanh_v1")
    )
    if aggregation_kind != _V7_SPARSE_AGGREGATION:
        raise ValueError(
            "sparse relation trace has an unknown aggregation contract: "
            f"{aggregation_kind!r}"
        )
    candidate_value = getattr(relation, "candidate_edge_mask")
    selected_value = getattr(relation, "selected_edge_mask")
    edge_valid_value = getattr(relation, "edge_valid_mask")
    if (
        candidate_value.dtype != torch.bool
        or selected_value.dtype != torch.bool
        or edge_valid_value.dtype != torch.bool
    ):
        raise TypeError("sparse relation support masks must be boolean")
    candidate = candidate_value
    selected = selected_value
    edge_valid = edge_valid_value
    messages = getattr(relation, "edge_messages")
    contributions = getattr(relation, "edge_log_odds_contributions")
    budget = int(getattr(relation, "edge_budget"))
    raw_proposal = getattr(relation, "raw_proposal_scores")
    unidentified_proposal = getattr(relation, "unidentified_proposal_scores")
    pre_geometry_proposal = getattr(
        relation, "pre_geometry_proposal_residuals"
    )
    pre_geometry_additive_proposal = getattr(
        relation, "pre_geometry_additive_proposal_scores"
    )
    projected_proposal = getattr(relation, "projected_proposal_scores")
    additive_proposal = getattr(relation, "additive_proposal_scores")
    raw_interaction = getattr(relation, "raw_interaction_scores")
    unidentified_interaction = getattr(
        relation, "unidentified_interaction_scores"
    )
    pre_geometry_interaction = getattr(
        relation, "pre_geometry_interaction_residuals"
    )
    pre_geometry_additive_endpoints = getattr(
        relation, "pre_geometry_additive_endpoint_scores"
    )
    interaction_residuals = getattr(relation, "interaction_residuals")
    additive_endpoints = getattr(relation, "additive_endpoint_scores")
    geometry_modulation = getattr(relation, "geometry_modulation")
    region_valid = getattr(relation, "region_valid_mask").bool()

    if budget < 1:
        raise ValueError("sparse relation edge budget must be positive")
    if candidate.shape != edge_valid.shape or not torch.equal(candidate, edge_valid):
        raise AssertionError("candidate and valid relation masks differ")
    expected_candidate = (
        region_valid[:, :, None]
        & region_valid[:, None, :]
        & ~torch.eye(
            region_valid.shape[1], dtype=torch.bool, device=region_valid.device
        )[None]
    )
    if not torch.equal(candidate, expected_candidate):
        raise AssertionError("candidate relation graph is not valid directed pairs")
    expected_shape = messages.shape
    for name, value in {
        "selected edge mask": selected,
        "edge log-odds contributions": contributions,
        "raw proposal scores": raw_proposal,
        "unidentified proposal scores": unidentified_proposal,
        "pre-geometry proposal residuals": pre_geometry_proposal,
        "pre-geometry additive proposal scores": (
            pre_geometry_additive_proposal
        ),
        "projected proposal scores": projected_proposal,
        "additive proposal scores": additive_proposal,
        "raw interaction scores": raw_interaction,
        "unidentified interaction scores": unidentified_interaction,
        "pre-geometry interaction residuals": pre_geometry_interaction,
        "pre-geometry additive endpoint scores": (
            pre_geometry_additive_endpoints
        ),
        "interaction residuals": interaction_residuals,
        "additive endpoint scores": additive_endpoints,
        "geometry modulation": geometry_modulation,
    }.items():
        if value.shape != expected_shape:
            raise ValueError(f"{name} shape differs from edge-message shape")
    if bool((selected & ~candidate[:, None]).any()):
        raise AssertionError("selected relation support leaves candidate graph")
    for name, value in {
        "messages": messages,
        "contributions": contributions,
        "raw proposal": raw_proposal,
        "unidentified proposal": unidentified_proposal,
        "pre-geometry proposal": pre_geometry_proposal,
        "pre-geometry additive proposal": pre_geometry_additive_proposal,
        "projected proposal": projected_proposal,
        "additive proposal": additive_proposal,
        "raw interaction": raw_interaction,
        "unidentified interaction": unidentified_interaction,
        "pre-geometry interaction": pre_geometry_interaction,
        "pre-geometry additive endpoints": pre_geometry_additive_endpoints,
        "interaction residuals": interaction_residuals,
        "additive endpoints": additive_endpoints,
        "geometry modulation": geometry_modulation,
    }.items():
        if not bool(torch.isfinite(value).all()):
            raise FloatingPointError(f"sparse relation {name} are non-finite")
    if bool((messages.masked_select(~selected) != 0).any()):
        raise AssertionError("non-selected relation edges carry messages")
    if bool((contributions.masked_select(~selected) != 0).any()):
        raise AssertionError("non-selected relation edges carry contributions")

    candidate_count = candidate.sum(dim=(1, 2))
    selected_count = selected.sum(dim=(2, 3))
    expected_count = torch.minimum(
        candidate_count[:, None],
        torch.full_like(selected_count, budget),
    )
    if not torch.equal(selected_count, expected_count):
        raise AssertionError("sparse relation support violates its exact edge budget")

    # Verify that the stored support is a valid top-q support.  This is
    # independent of tie order: every selected score must be no smaller than
    # every eligible unselected score.
    proposal_for_selection = (
        additive_proposal
        if variant == "additive_endpoint_control_v1"
        else projected_proposal
    ).abs()
    selected_floor = proposal_for_selection.masked_fill(~selected, torch.inf).amin(
        dim=(2, 3)
    )
    unselected_candidates = candidate[:, None] & ~selected
    unselected_ceiling = proposal_for_selection.masked_fill(
        ~unselected_candidates, -torch.inf
    ).amax(dim=(2, 3))
    ranking_violation = (unselected_ceiling - selected_floor).clamp_min(0.0)
    ranking_violation = torch.where(
        torch.isfinite(ranking_violation), ranking_violation, torch.zeros_like(ranking_violation)
    )
    ranking_tolerance = 2e-6 * max(
        1.0, float(proposal_for_selection.detach().abs().max().cpu())
    )
    if float(ranking_violation.max().cpu()) > ranking_tolerance:
        raise AssertionError("selected relation support is not the declared top-q")

    independent_pre_geometry_proposal = _masked_directed_anova_projection(
        unidentified_proposal, candidate
    )
    independent_pre_geometry_interaction = _masked_directed_anova_projection(
        unidentified_interaction, candidate
    )
    expected_pre_geometry_additive_proposal = (
        unidentified_proposal - independent_pre_geometry_proposal
    )
    expected_pre_geometry_additive_interaction = (
        unidentified_interaction - independent_pre_geometry_interaction
    )
    independent_proposal = _masked_directed_anova_projection(
        raw_proposal, candidate
    )
    independent_interaction = _masked_directed_anova_projection(
        raw_interaction, candidate
    )
    projection_scale = max(
        1.0,
        float(unidentified_proposal.detach().abs().max().cpu()),
        float(unidentified_interaction.detach().abs().max().cpu()),
        float(raw_proposal.detach().abs().max().cpu()),
        float(raw_interaction.detach().abs().max().cpu()),
    )
    projection_tolerance = 5e-5 * projection_scale
    pre_geometry_proposal_projection_error = float(
        (
            pre_geometry_proposal.double()
            - independent_pre_geometry_proposal
        )
        .abs()
        .max()
        .cpu()
    )
    pre_geometry_interaction_projection_error = float(
        (
            pre_geometry_interaction.double()
            - independent_pre_geometry_interaction
        )
        .abs()
        .max()
        .cpu()
    )
    pre_geometry_additive_proposal_error = float(
        (
            pre_geometry_additive_proposal.double()
            - expected_pre_geometry_additive_proposal
        )
        .abs()
        .max()
        .cpu()
    )
    pre_geometry_additive_interaction_error = float(
        (
            pre_geometry_additive_endpoints.double()
            - expected_pre_geometry_additive_interaction
        )
        .abs()
        .max()
        .cpu()
    )
    geometry_proposal_source = (
        pre_geometry_additive_proposal
        if variant == "additive_endpoint_control_v1"
        else pre_geometry_proposal
    )
    geometry_interaction_source = (
        pre_geometry_additive_endpoints
        if variant == "additive_endpoint_control_v1"
        else pre_geometry_interaction
    )
    geometry_proposal_error = float(
        (
            raw_proposal
            - geometry_proposal_source * geometry_modulation
        )
        .abs()
        .max()
        .cpu()
    )
    geometry_interaction_error = float(
        (
            raw_interaction
            - geometry_interaction_source * geometry_modulation
        )
        .abs()
        .max()
        .cpu()
    )
    proposal_projection_error = float(
        (projected_proposal.double() - independent_proposal).abs().max().cpu()
    )
    interaction_projection_error = float(
        (interaction_residuals.double() - independent_interaction).abs().max().cpu()
    )
    projection_diagnostics = (
        pre_geometry_proposal_projection_error,
        pre_geometry_interaction_projection_error,
        pre_geometry_additive_proposal_error,
        pre_geometry_additive_interaction_error,
        geometry_proposal_error,
        geometry_interaction_error,
        proposal_projection_error,
        interaction_projection_error,
    )
    if not all(math.isfinite(value) for value in projection_diagnostics):
        raise FloatingPointError("sparse relation projection diagnostics are non-finite")
    if max(projection_diagnostics) > projection_tolerance:
        raise AssertionError("stored relation ANOVA projection does not replay")
    proposal_partition_error = float(
        (raw_proposal - (projected_proposal + additive_proposal)).abs().max().cpu()
    )
    interaction_partition_error = float(
        (raw_interaction - (interaction_residuals + additive_endpoints)).abs().max().cpu()
    )
    if not all(
        math.isfinite(value)
        for value in (proposal_partition_error, interaction_partition_error)
    ):
        raise FloatingPointError("sparse relation partition diagnostics are non-finite")
    if max(proposal_partition_error, interaction_partition_error) > projection_tolerance:
        raise AssertionError("relation interaction/main-effect partition is invalid")

    residual_mask = candidate[:, None]
    residual_fields = (
        pre_geometry_proposal,
        pre_geometry_interaction,
        projected_proposal,
        interaction_residuals,
    )
    row_column_terms = []
    for residual_field in residual_fields:
        masked_field = residual_field.masked_fill(~residual_mask, 0.0)
        row_column_terms.extend(
            (masked_field.sum(dim=-1), masked_field.sum(dim=-2))
        )
    row_column_error = max(
        float(value.abs().max().cpu()) for value in row_column_terms
    )
    sum_tolerance = max(
        2e-6,
        32.0
        * torch.finfo(torch.float32).eps
        * float(candidate.shape[-1])
        * projection_scale,
    )
    if not math.isfinite(row_column_error):
        raise FloatingPointError("sparse relation cancellation diagnostic is non-finite")
    if row_column_error > sum_tolerance:
        raise AssertionError("relation residual retains endpoint main effects")

    active_scores = (
        additive_endpoints
        if variant == "additive_endpoint_control_v1"
        else interaction_residuals
    )
    expected_messages = torch.where(
        selected, active_scores.tanh(), torch.zeros_like(active_scores)
    )
    message_error = float((messages - expected_messages).abs().max().cpu())
    boundaries = int(messages.shape[1])
    expected_contributions = (
        float(getattr(relation, "delta_cap"))
        / float(boundaries * budget)
        * expected_messages
    )
    contribution_error = float(
        (contributions - expected_contributions).abs().max().cpu()
    )
    target = contributions.sum(dim=-1)
    incremental = target.sum(dim=-1)
    cumulative = reverse_cumulative_atoms(incremental, dim=1)
    aggregation_error = max(
        float(
            (target - getattr(relation, "target_incremental_log_odds"))
            .abs()
            .max()
            .cpu()
        ),
        float(
            (incremental - getattr(relation, "incremental_log_odds"))
            .abs()
            .max()
            .cpu()
        ),
        float(
            (cumulative - getattr(relation, "cumulative_log_odds"))
            .abs()
            .max()
            .cpu()
        ),
    )
    if not all(
        math.isfinite(value)
        for value in (message_error, contribution_error, aggregation_error)
    ):
        raise FloatingPointError("sparse relation compilation diagnostics are non-finite")
    numeric_tolerance = max(2e-6, projection_tolerance)
    if max(message_error, contribution_error, aggregation_error) > numeric_tolerance:
        raise AssertionError("sparse relation contribution compilation is invalid")
    if bool((cumulative.abs() > float(getattr(relation, "delta_cap")) + 2e-6).any()):
        raise AssertionError("sparse relation cumulative budget is exceeded")

    nonzero_count = contributions.ne(0).sum(dim=(2, 3))
    density = selected_count.double() / candidate_count.clamp_min(1)[:, None].double()
    return {
        "variant": variant,
        "aggregation_kind": aggregation_kind,
        "edge_budget": budget,
        "candidate_count": candidate_count.detach().cpu(),
        "selected_count": selected_count.detach().cpu(),
        "nonzero_count": nonzero_count.detach().cpu(),
        "selected_density": density.detach().cpu(),
        "max_topq_ranking_violation": float(ranking_violation.max().cpu()),
        "max_pre_geometry_proposal_projection_replay_error": (
            pre_geometry_proposal_projection_error
        ),
        "max_pre_geometry_interaction_projection_replay_error": (
            pre_geometry_interaction_projection_error
        ),
        "max_pre_geometry_additive_proposal_identity_error": (
            pre_geometry_additive_proposal_error
        ),
        "max_pre_geometry_additive_interaction_identity_error": (
            pre_geometry_additive_interaction_error
        ),
        "max_geometry_proposal_identity_error": geometry_proposal_error,
        "max_geometry_interaction_identity_error": geometry_interaction_error,
        "max_proposal_projection_replay_error": proposal_projection_error,
        "max_interaction_projection_replay_error": interaction_projection_error,
        "max_proposal_partition_error": proposal_partition_error,
        "max_interaction_partition_error": interaction_partition_error,
        "max_residual_row_or_column_sum": row_column_error,
        "max_message_identity_error": message_error,
        "max_edge_contribution_identity_error": contribution_error,
        "max_sparse_aggregation_identity_error": aggregation_error,
    }


def _entmax15_reference(
    logits: torch.Tensor,
    support_mask: torch.Tensor,
) -> torch.Tensor:
    """Independent FP64 alpha=1.5 entmax reference on a masked support.

    This intentionally does not call the model implementation.  The loop is
    acceptable in the offline audit and makes the allocation contract
    independently replayable, including when shortlist sizes vary by sample.
    """

    if logits.shape != support_mask.shape or logits.ndim != 4:
        raise ValueError("masked entmax inputs must have matching (N,B,M,M) shapes")
    if support_mask.dtype != torch.bool:
        raise TypeError("masked entmax support must be boolean")
    result = torch.zeros_like(logits, dtype=torch.float64)
    working = logits.to(dtype=torch.float64)
    for sample in range(logits.shape[0]):
        for boundary in range(logits.shape[1]):
            support = support_mask[sample, boundary]
            values = working[sample, boundary][support]
            if values.numel() == 0:
                raise AssertionError("entmax shortlist is empty")

            # Peters et al.'s exact alpha=1.5 threshold algorithm.  Scaling by
            # 1/2 is part of the entmax15 transform, not a temperature choice.
            values = values / 2.0
            values = values - values.max()
            sorted_values = values.sort(descending=True).values
            rho = torch.arange(
                1,
                sorted_values.numel() + 1,
                device=values.device,
                dtype=torch.float64,
            )
            mean = sorted_values.cumsum(0) / rho
            mean_sq = sorted_values.square().cumsum(0) / rho
            variance_sum = rho * (mean_sq - mean.square())
            delta = (1.0 - variance_sum) / rho
            taus = mean - delta.clamp_min(0.0).sqrt()
            threshold_candidates = taus <= sorted_values
            support_size = int(threshold_candidates.sum().item())
            if support_size < 1:
                raise AssertionError("entmax threshold has empty support")
            threshold = taus[support_size - 1]
            probabilities = (values - threshold).clamp_min(0.0).square()
            normalizer = probabilities.sum()
            if not bool(torch.isfinite(normalizer)) or float(normalizer) <= 0.0:
                raise FloatingPointError("entmax reference has invalid normalization")
            probabilities = probabilities / normalizer
            result[sample, boundary][support] = probabilities
    return result


def _expanded_relation_strength(
    strength: torch.Tensor,
    reference: torch.Tensor,
) -> torch.Tensor:
    """Validate and broadcast a shared per-boundary V8 output gate."""

    batch, boundaries, regions, _ = reference.shape
    value = torch.as_tensor(strength, device=reference.device, dtype=reference.dtype)
    if value.shape == (boundaries,):
        value = value.reshape(1, boundaries, 1, 1)
    elif value.shape in {(1, boundaries), (batch, boundaries)}:
        value = value.reshape(value.shape[0], boundaries, 1, 1)
    elif value.shape not in {
        (1, boundaries, 1, 1),
        (batch, boundaries, 1, 1),
    }:
        raise ValueError(
            "relation strength must be shared per boundary, with shape (B,), "
            "(1,B), (N,B), (1,B,1,1), or (N,B,1,1)"
        )
    return value.expand(batch, boundaries, regions, regions)


def _audit_conserved_witness_relation_trace(
    relation: object,
) -> dict[str, Any]:
    """Fail-closed audit of a V8 conserved minimal-witness trace."""

    variant = str(getattr(relation, "variant", ""))
    aggregation_kind = str(getattr(relation, "aggregation_kind", ""))
    if variant not in _V8_RELATION_VARIANTS:
        raise ValueError(f"unknown V8 relation variant: {variant!r}")
    if aggregation_kind != _V8_WITNESS_AGGREGATION:
        raise ValueError(
            "V8 relation trace has an unknown aggregation contract: "
            f"{aggregation_kind!r}"
        )

    candidate = getattr(relation, "candidate_edge_mask")
    edge_valid = getattr(relation, "edge_valid_mask")
    shortlist = getattr(relation, "shortlist_edge_mask")
    active = getattr(relation, "active_witness_mask", None)
    selected_alias = getattr(relation, "selected_edge_mask", None)
    if active is None:
        if selected_alias is None:
            raise AttributeError("V8 relation trace omits its active witness support")
        active = selected_alias
    elif selected_alias is not None and not torch.equal(active, selected_alias):
        raise AssertionError("selected-edge alias differs from active witness support")
    for name, value in {
        "candidate edge mask": candidate,
        "edge-valid mask": edge_valid,
        "shortlist edge mask": shortlist,
        "active witness mask": active,
    }.items():
        if value.dtype != torch.bool:
            raise TypeError(f"{name} must be boolean")

    scores = getattr(relation, "identified_pair_scores")
    allocation = getattr(relation, "witness_allocation")
    messages = getattr(relation, "edge_messages")
    pair_gates = getattr(relation, "pair_gates")
    contributions = getattr(relation, "edge_log_odds_contributions")
    region_valid = getattr(relation, "region_valid_mask").bool()
    budget = int(getattr(relation, "edge_budget"))
    temperature = float(getattr(relation, "allocation_temperature"))
    delta_cap = float(getattr(relation, "delta_cap"))
    if budget < 1:
        raise ValueError("V8 maximum witness count must be positive")
    if not math.isfinite(temperature) or temperature <= 0.0:
        raise ValueError("V8 allocation temperature must be finite and positive")
    if not math.isfinite(delta_cap) or delta_cap <= 0.0:
        raise ValueError("V8 relation delta cap must be finite and positive")
    if candidate.shape != edge_valid.shape or not torch.equal(candidate, edge_valid):
        raise AssertionError("candidate and valid relation masks differ")
    expected_candidate = (
        region_valid[:, :, None]
        & region_valid[:, None, :]
        & ~torch.eye(
            region_valid.shape[1], dtype=torch.bool, device=region_valid.device
        )[None]
    )
    if not torch.equal(candidate, expected_candidate):
        raise AssertionError("candidate relation graph is not valid directed pairs")
    expected_shape = scores.shape
    if scores.ndim != 4 or scores.shape[-1] != scores.shape[-2]:
        raise ValueError("identified pair scores must have shape (N,B,M,M)")
    for name, value in {
        "shortlist edge mask": shortlist,
        "active witness mask": active,
        "witness allocation": allocation,
        "edge messages": messages,
        "pair gates": pair_gates,
        "edge log-odds contributions": contributions,
    }.items():
        if value.shape != expected_shape:
            raise ValueError(f"{name} shape differs from identified pair scores")
    if scores.shape[0] != candidate.shape[0] or scores.shape[-2:] != candidate.shape[-2:]:
        raise ValueError("V8 score and candidate graph shapes differ")
    if bool((shortlist & ~candidate[:, None]).any()):
        raise AssertionError("V8 shortlist leaves the candidate graph")
    if bool((active & ~shortlist).any()):
        raise AssertionError("active witnesses leave the V8 shortlist")
    for name, value in {
        "identified pair scores": scores,
        "witness allocation": allocation,
        "edge messages": messages,
        "pair gates": pair_gates,
        "edge log-odds contributions": contributions,
    }.items():
        if not bool(torch.isfinite(value).all()):
            raise FloatingPointError(f"V8 {name} are non-finite")
    if bool((allocation < 0).any()):
        raise AssertionError("V8 witness allocation contains negative mass")
    if bool((scores.masked_select(~candidate[:, None]) != 0).any()):
        raise AssertionError("invalid V8 relation edges carry identified scores")
    if bool((allocation.masked_select(~active) != 0).any()):
        raise AssertionError("inactive V8 witnesses carry allocation mass")
    if bool((allocation.masked_select(active) <= 0).any()):
        raise AssertionError("active V8 witnesses must carry positive allocation mass")
    if bool((contributions.masked_select(~active) != 0).any()):
        raise AssertionError("inactive V8 witnesses carry log-odds contributions")
    if not torch.equal(pair_gates, allocation):
        raise AssertionError("V8 pair gates do not expose the witness allocation")

    candidate_count = candidate.sum(dim=(1, 2))
    shortlist_count = shortlist.sum(dim=(2, 3))
    active_count = active.sum(dim=(2, 3))
    expected_shortlist_count = torch.minimum(
        candidate_count[:, None], torch.full_like(shortlist_count, budget)
    )
    if not torch.equal(shortlist_count, expected_shortlist_count):
        raise AssertionError("V8 shortlist violates its maximum-witness budget")
    if bool((active_count < 1).any()) or bool((active_count > shortlist_count).any()):
        raise AssertionError("V8 active witness count is outside [1, shortlist size]")

    # Ties may choose either stable edge, so audit the top-q set by its score
    # floor/ceiling rather than reproducing a device-specific tie order.
    magnitude = scores.abs()
    shortlist_floor = magnitude.masked_fill(~shortlist, torch.inf).amin(dim=(2, 3))
    unshortlisted = candidate[:, None] & ~shortlist
    unshortlisted_ceiling = magnitude.masked_fill(
        ~unshortlisted, -torch.inf
    ).amax(dim=(2, 3))
    ranking_violation = (unshortlisted_ceiling - shortlist_floor).clamp_min(0.0)
    ranking_violation = torch.where(
        torch.isfinite(ranking_violation),
        ranking_violation,
        torch.zeros_like(ranking_violation),
    )
    score_scale = max(1.0, float(scores.detach().abs().max().cpu()))
    numeric_tolerance = max(2e-6, 5e-6 * score_scale)
    if float(ranking_violation.max().cpu()) > numeric_tolerance:
        raise AssertionError("V8 shortlist is not a top-q identified-score support")

    allocation_sum = allocation.double().sum(dim=(2, 3))
    allocation_sum_error = float((allocation_sum - 1.0).abs().max().cpu())
    if allocation_sum_error > 2e-6:
        raise AssertionError("V8 witness allocation does not conserve unit capacity")
    reference_allocation = _entmax15_reference(
        magnitude.double() / temperature, shortlist
    )
    allocation_replay_error = float(
        (allocation.double() - reference_allocation).abs().max().cpu()
    )
    if allocation_replay_error > max(5e-6, numeric_tolerance):
        raise AssertionError("V8 entmax witness allocation does not replay")
    expected_active = allocation > 0
    if not torch.equal(active, expected_active):
        raise AssertionError("V8 active support is not the positive entmax support")

    effective = 1.0 / allocation.double().square().sum(dim=(2, 3))
    reported_effective = getattr(relation, "effective_witness_count")
    if reported_effective.shape != effective.shape:
        raise ValueError("V8 effective witness count has invalid shape")
    if not bool(torch.isfinite(reported_effective).all()):
        raise FloatingPointError("V8 effective witness counts are non-finite")
    effective_error = float(
        (reported_effective.double() - effective).abs().max().cpu()
    )
    if effective_error > 5e-6:
        raise AssertionError("V8 effective witness count does not replay")

    # The target and shuffled controls remain identified contrasts.  The
    # additive control intentionally preserves endpoint main effects.
    projection_error = 0.0
    row_column_error = 0.0
    if variant != "additive_conserved_witness_control_v1":
        independently_projected = _masked_directed_anova_projection(
            scores, candidate
        )
        projection_error = float(
            (scores.double() - independently_projected).abs().max().cpu()
        )
        masked_scores = scores.masked_fill(~candidate[:, None], 0.0)
        row_column_error = max(
            float(masked_scores.sum(-1).abs().max().cpu()),
            float(masked_scores.sum(-2).abs().max().cpu()),
        )
        projection_tolerance = max(
            2e-6,
            32.0
            * torch.finfo(torch.float32).eps
            * float(candidate.shape[-1])
            * score_scale,
        )
        if max(projection_error, row_column_error) > projection_tolerance:
            raise AssertionError("V8 identified score field retains endpoint main effects")

    expanded_strength = _expanded_relation_strength(
        getattr(relation, "relation_strength"), scores
    )
    if not bool(torch.isfinite(expanded_strength).all()):
        raise FloatingPointError("V8 relation strength is non-finite")
    if bool((expanded_strength.abs() > 1.0 + 2e-6).any()):
        raise AssertionError("V8 relation strength leaves its [-1,1] gate range")
    boundaries = int(scores.shape[1])
    scale = delta_cap / float(boundaries)
    expected_messages = torch.where(
        active,
        expanded_strength * scores.tanh(),
        torch.zeros_like(scores),
    )
    message_error = float((messages - expected_messages).abs().max().cpu())
    expected_contributions = (
        scale * expected_messages.double() * allocation.double()
    ).double()
    contribution_error = float(
        (contributions.double() - expected_contributions).abs().max().cpu()
    )
    target = contributions.double().sum(dim=-1)
    incremental = target.sum(dim=-1)
    cumulative = reverse_cumulative_atoms(incremental, dim=1)
    aggregation_error = max(
        float(
            (target - getattr(relation, "target_incremental_log_odds").double())
            .abs()
            .max()
            .cpu()
        ),
        float(
            (incremental - getattr(relation, "incremental_log_odds").double())
            .abs()
            .max()
            .cpu()
        ),
        float(
            (cumulative - getattr(relation, "cumulative_log_odds").double())
            .abs()
            .max()
            .cpu()
        ),
    )
    if max(message_error, contribution_error, aggregation_error) > max(
        5e-6, numeric_tolerance
    ):
        raise AssertionError("V8 conserved witness contribution compilation is invalid")
    per_boundary_l1 = contributions.double().abs().sum(dim=(2, 3))
    per_boundary_capacity = (
        scale * expanded_strength[:, :, 0, 0].double().abs()
    )
    reported_capacity = getattr(relation, "per_boundary_allocated_capacity")
    reported_l1 = getattr(relation, "per_boundary_l1_usage")
    if reported_capacity.shape != per_boundary_capacity.shape:
        raise ValueError("V8 allocated-capacity trace has invalid shape")
    if reported_l1.shape != per_boundary_l1.shape:
        raise ValueError("V8 L1-usage trace has invalid shape")
    if not bool(torch.isfinite(reported_capacity).all()) or not bool(
        torch.isfinite(reported_l1).all()
    ):
        raise FloatingPointError("V8 capacity-use traces are non-finite")
    capacity_identity_error = float(
        (reported_capacity.double() - per_boundary_capacity).abs().max().cpu()
    )
    l1_identity_error = float(
        (reported_l1.double() - per_boundary_l1).abs().max().cpu()
    )
    if max(capacity_identity_error, l1_identity_error) > 5e-6:
        raise AssertionError("V8 reported capacity use does not replay")
    budget_violation = float(
        (per_boundary_l1 - per_boundary_capacity).clamp_min(0.0).max().cpu()
    )
    cumulative_violation = float(
        (cumulative.abs() - delta_cap).clamp_min(0.0).max().cpu()
    )
    if max(budget_violation, cumulative_violation) > 5e-6:
        raise AssertionError("V8 conserved relation budget is exceeded")

    nonzero_count = contributions.ne(0).sum(dim=(2, 3))
    density = active_count.double() / candidate_count.clamp_min(1)[:, None].double()
    top_mass = allocation.double().flatten(2).amax(dim=-1)
    return {
        "variant": variant,
        "aggregation_kind": aggregation_kind,
        "edge_budget": budget,
        "allocation_temperature": temperature,
        "candidate_count": candidate_count.detach().cpu(),
        "shortlist_count": shortlist_count.detach().cpu(),
        "selected_count": active_count.detach().cpu(),
        "nonzero_count": nonzero_count.detach().cpu(),
        "selected_density": density.detach().cpu(),
        "effective_witness_count": effective.detach().cpu(),
        "top_witness_allocation": top_mass.detach().cpu(),
        "per_boundary_allocated_capacity": per_boundary_capacity.detach().cpu(),
        "per_boundary_l1_usage": per_boundary_l1.detach().cpu(),
        "max_topq_ranking_violation": float(ranking_violation.max().cpu()),
        "max_allocation_sum_error": allocation_sum_error,
        "max_entmax_allocation_replay_error": allocation_replay_error,
        "max_effective_witness_count_identity_error": effective_error,
        "max_identified_projection_replay_error": projection_error,
        "max_identified_residual_row_or_column_sum": row_column_error,
        "max_message_identity_error": message_error,
        "max_edge_contribution_identity_error": contribution_error,
        "max_sparse_aggregation_identity_error": aggregation_error,
        "max_allocated_capacity_identity_error": capacity_identity_error,
        "max_l1_usage_identity_error": l1_identity_error,
        "max_per_boundary_budget_violation": budget_violation,
        "max_cumulative_budget_violation": cumulative_violation,
    }


def _exact_two_sided_mcnemar_pvalue(helped: int, harmed: int) -> float:
    """Exact two-sided binomial McNemar p-value without SciPy."""

    discordant = int(helped) + int(harmed)
    if discordant == 0:
        return 1.0
    tail = min(int(helped), int(harmed))
    log_terms = [
        math.lgamma(discordant + 1)
        - math.lgamma(index + 1)
        - math.lgamma(discordant - index + 1)
        - discordant * math.log(2.0)
        for index in range(tail + 1)
    ]
    maximum = max(log_terms)
    probability = math.exp(maximum) * sum(
        math.exp(value - maximum) for value in log_terms
    )
    return min(1.0, 2.0 * probability)


def _relation_ablation_summary(
    records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Compare the full relation circuit with its exact no-relation replay."""

    if not records:
        return {"n": 0}
    labels = torch.tensor([int(item["label"]) for item in records])
    full_predictions = torch.tensor(
        [int(item["baseline_prediction"]) for item in records]
    )
    replayed_predictions = torch.tensor(
        [int(item["replayed_prediction"]) for item in records]
    )
    full_probabilities = torch.tensor(
        [item["baseline_class_probs"] for item in records], dtype=torch.float64
    )
    replayed_probabilities = torch.tensor(
        [item["replayed_class_probs"] for item in records], dtype=torch.float64
    )
    full_metrics = evaluate_origin_predictions(
        full_probabilities, full_predictions, labels
    )
    replayed_metrics = evaluate_origin_predictions(
        replayed_probabilities, replayed_predictions, labels
    )
    full_correct = full_predictions.eq(labels)
    replayed_correct = replayed_predictions.eq(labels)
    helped = int((full_correct & ~replayed_correct).sum())
    harmed = int((~full_correct & replayed_correct).sum())
    changed = full_predictions.ne(replayed_predictions)

    by_grade: dict[str, Any] = {}
    for grade in sorted(set(labels.tolist())):
        members = labels.eq(grade)
        by_grade[str(grade)] = {
            "support": int(members.sum()),
            "prediction_change_count": int((changed & members).sum()),
            "full_relation_only_correct": int(
                (full_correct & ~replayed_correct & members).sum()
            ),
            "no_relation_only_correct": int(
                (~full_correct & replayed_correct & members).sum()
            ),
        }
    metric_delta = {
        name: float(full_metrics[name]) - float(replayed_metrics[name])
        for name in ("acc", "qwk", "balanced_acc", "macro_f1", "mae", "ece")
    }
    return {
        "n": len(records),
        "full_relation_metrics": full_metrics,
        "no_relation_metrics": replayed_metrics,
        "full_minus_no_relation": metric_delta,
        "prediction_change_count": int(changed.sum()),
        "prediction_change_rate": float(changed.double().mean()),
        "full_relation_only_correct": helped,
        "no_relation_only_correct": harmed,
        "net_correct": helped - harmed,
        "exact_two_sided_mcnemar_pvalue_descriptive": (
            _exact_two_sided_mcnemar_pvalue(helped, harmed)
        ),
        "by_true_grade": by_grade,
    }


def _most_pivotal_sparse_edge_records(
    model: torch.nn.Module,
    output: object,
    relation: object,
    labels: torch.Tensor,
    sample_ids: torch.Tensor,
    decision_rule: str,
    collective_records: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], tuple[float, float, float, float]]:
    """Exhaustively choose each sample's strongest actual active-edge effect."""

    selected_mask = relation.selected_edge_mask.bool()
    contributions = relation.edge_log_odds_contributions
    flat_selected = selected_mask.flatten(1)
    flat_contributions = contributions.flatten(1)
    counts = flat_selected.sum(dim=1)
    if bool((counts <= 0).any()):
        raise AssertionError("sparse relation sample has no selected edge")
    # Contribution magnitude fixes only enumeration order.  Selection of the
    # final certificate below uses exact leave-one-edge-out expected-grade
    # effects, not this internal score.
    ranked = flat_contributions.abs().masked_fill(~flat_selected, -torch.inf)
    order = ranked.argsort(dim=1, descending=True)
    per_sample: list[list[tuple[dict[str, Any], int]]] = [
        [] for _ in range(len(labels))
    ]
    maxima = [0.0, 0.0, 0.0, 0.0]
    for rank in range(int(counts.max().item())):
        present = counts > rank
        flat_index = order[:, rank]
        removal = torch.zeros_like(selected_mask).flatten(1)
        removal.scatter_(1, flat_index[:, None], present[:, None])
        removal = removal.reshape_as(selected_mask)
        records, *errors = _relation_replay_records(
            model,
            output,
            removal,
            labels,
            sample_ids,
            decision_rule,
        )
        maxima = [max(old, new) for old, new in zip(maxima, errors)]
        for row, record in enumerate(records):
            if bool(present[row]):
                per_sample[row].append((record, int(flat_index[row])))

    regions = int(contributions.shape[-1])
    centers = relation.region_centers_yx.detach().cpu()
    result: list[dict[str, Any]] = []
    for row, candidates in enumerate(per_sample):
        candidates.sort(
            key=lambda item: abs(float(item[0]["expected_grade_delta"])),
            reverse=True,
        )
        record, flat_index = candidates[0]
        absolute_effects = [
            abs(float(item[0]["expected_grade_delta"])) for item in candidates
        ]
        effect_l1 = float(sum(absolute_effects))
        top4 = float(sum(sorted(absolute_effects, reverse=True)[:4]))
        boundary = flat_index // (regions * regions)
        endpoints = flat_index % (regions * regions)
        target = endpoints // regions
        source = endpoints % regions
        collective = abs(float(collective_records[row]["expected_grade_delta"]))
        record.update(
            {
                "selection": "most_pivotal_selected_pair_by_exact_leave_one_edge_replay",
                "boundary": int(boundary),
                "target_region": int(target),
                "source_region": int(source),
                "target_center_yx": centers[target].tolist(),
                "source_center_yx": centers[source].tolist(),
                "signed_pair_message": float(
                    relation.edge_messages.detach().cpu().flatten(1)[row, flat_index]
                ),
                "signed_edge_log_odds_contribution": float(
                    flat_contributions.detach().cpu()[row, flat_index]
                ),
                "selected_edge_count": int(counts[row]),
                "sum_absolute_individual_edge_effects": effect_l1,
                "top_edge_effect_share_of_individual_l1": (
                    abs(float(record["expected_grade_delta"])) / effect_l1
                    if effect_l1 > 0.0
                    else 0.0
                ),
                "top4_edge_effect_share_of_individual_l1": (
                    top4 / effect_l1 if effect_l1 > 0.0 else 0.0
                ),
                "absolute_collective_relation_effect": collective,
                "collective_to_individual_l1_ratio": (
                    collective / effect_l1 if effect_l1 > 0.0 else 0.0
                ),
                "region_receptive_field_pixels": int(
                    relation.region_receptive_field
                ),
                "region_output_stride_pixels": int(
                    relation.region_output_stride
                ),
            }
        )
        result.append(record)
    return result, tuple(maxima)  # type: ignore[return-value]


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
    baseline_source_rates = getattr(baseline, "base_total_rates", baseline_rates)
    replayed_source_rates = getattr(replayed, "base_total_rates", replayed_rates)
    _assert_distribution(baseline)
    _assert_distribution(replayed)
    if not bool(torch.isfinite(removed).all()) or bool((removed < 0).any()):
        raise FloatingPointError("invalid removed ORIGIN rates")
    error = float(
        (baseline_source_rates - removed - replayed_source_rates)
        .abs()
        .max()
        .cpu()
    )
    local_maps = _field(baseline, "local_rate_maps")
    ledger_dtype = next(iter(local_maps.values())).dtype
    rate_scale = max(
        1.0,
        float(baseline_rates.abs().max().cpu()),
        float(replayed_rates.abs().max().cpu()),
        float(baseline_source_rates.abs().max().cpu()),
        float(replayed_source_rates.abs().max().cpu()),
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


def _relation_replay_records(
    model: torch.nn.Module,
    baseline: object,
    removals: torch.Tensor,
    labels: torch.Tensor,
    sample_ids: torch.Tensor,
    decision_rule: str,
) -> tuple[list[dict[str, Any]], float, float, float, float]:
    """Delete stored pair messages and audit exact same-circuit replay."""

    replay = getattr(model, "replay_without_relations", None)
    if not callable(replay):
        raise TypeError("relation-enabled ORIGIN must expose replay_without_relations")
    intervention = replay(baseline, removals, force_decoder_fp64=True)
    replayed = getattr(intervention, "output", intervention)
    _assert_distribution(baseline)
    _assert_distribution(replayed)
    base_source = _field(baseline, "base_total_rates")
    replay_source = _field(replayed, "base_total_rates")
    source_error = float((base_source - replay_source).abs().max().cpu())
    if source_error != 0.0:
        raise AssertionError("relation deletion changed the conserved unary ledger")
    baseline_relation = _field(baseline, "relation_evidence")
    replayed_relation = _field(replayed, "relation_evidence")
    canonical = intervention.removal_mask.bool()
    removed = intervention.removed_edge_messages
    original_messages = baseline_relation.edge_messages
    requested = torch.as_tensor(removals, device=original_messages.device)
    batch, boundaries, regions, _ = original_messages.shape
    if requested.ndim == 2:
        requested = requested[None, None]
    elif requested.ndim == 3:
        if requested.shape[0] != batch:
            raise AssertionError("relation audit received an ambiguous removal mask")
        requested = requested[:, None]
    elif requested.ndim != 4:
        raise AssertionError("relation audit received an invalid removal-mask rank")
    try:
        expected_canonical = requested.bool().expand(
            batch, boundaries, regions, regions
        )
    except RuntimeError as error:
        raise AssertionError(
            "relation audit removal mask is not broadcastable to the edge ledger"
        ) from error
    expected_canonical = (
        expected_canonical & baseline_relation.edge_valid_mask[:, None]
    )
    baseline_selected = getattr(baseline_relation, "selected_edge_mask", None)
    if baseline_selected is not None:
        expected_canonical = expected_canonical & baseline_selected.bool()
    if not torch.equal(canonical, expected_canonical):
        raise AssertionError(
            "relation intervention changed or ignored the requested deletion mask"
        )
    for support_name in (
        "region_valid_mask",
        "edge_valid_mask",
        "candidate_edge_mask",
        "shortlist_edge_mask",
        "active_witness_mask",
        "selected_edge_mask",
    ):
        baseline_support = getattr(baseline_relation, support_name, None)
        replayed_support = getattr(replayed_relation, support_name, None)
        if baseline_support is None or replayed_support is None:
            if baseline_support is not replayed_support:
                raise AssertionError(
                    f"relation replay changed optional support {support_name}"
                )
        elif not torch.equal(baseline_support, replayed_support):
            raise AssertionError(
                f"relation replay reselected or changed support {support_name}"
            )
    for fixed_name in (
        "identified_pair_scores",
        "witness_allocation",
        "pair_gates",
        "effective_witness_count",
        "relation_strength",
        "allocation_temperature",
        "per_boundary_allocated_capacity",
    ):
        baseline_value = getattr(baseline_relation, fixed_name, None)
        replayed_value = getattr(replayed_relation, fixed_name, None)
        if baseline_value is None or replayed_value is None:
            if baseline_value is not replayed_value:
                raise AssertionError(
                    f"relation replay changed optional trace field {fixed_name}"
                )
        elif torch.is_tensor(baseline_value):
            if not torch.equal(baseline_value, replayed_value):
                raise AssertionError(
                    f"relation replay reallocated or changed trace field {fixed_name}"
                )
        elif baseline_value != replayed_value:
            raise AssertionError(
                f"relation replay changed optional trace field {fixed_name}"
            )
    partition_error = float(
        (original_messages - (replayed_relation.edge_messages + removed))
        .abs()
        .max()
        .cpu()
    )
    if partition_error != 0.0:
        raise AssertionError("relation replay did not exactly partition edge messages")
    if not torch.equal(
        removed,
        torch.where(canonical, original_messages, torch.zeros_like(original_messages)),
    ):
        raise AssertionError("relation intervention removal mask is not replayable")
    aggregation_kind = str(
        getattr(
            replayed_relation,
            "aggregation_kind",
            "geometry_normalized_tanh_v1",
        )
    )
    removed_contributions: torch.Tensor | None = None
    if aggregation_kind in {
        _V7_SPARSE_AGGREGATION,
        _V8_WITNESS_AGGREGATION,
    }:
        original_contributions = baseline_relation.edge_log_odds_contributions
        replayed_contributions = replayed_relation.edge_log_odds_contributions
        reported_removed_contributions = getattr(
            intervention, "removed_edge_log_odds_contributions", None
        )
        if reported_removed_contributions is None:
            raise AssertionError(
                "sparse relation replay omitted removed edge contributions"
            )
        removed_contributions = reported_removed_contributions
        contribution_partition_error = float(
            (
                original_contributions
                - (replayed_contributions + removed_contributions)
            )
            .abs()
            .max()
            .cpu()
        )
        if contribution_partition_error != 0.0:
            raise AssertionError(
                "relation replay did not exactly partition edge contributions"
            )
        partition_error = max(partition_error, contribution_partition_error)
        if bool((removed_contributions.masked_select(~canonical) != 0).any()):
            raise AssertionError(
                "relation replay changed non-removed edge contributions"
            )
        if not torch.equal(
            removed_contributions,
            torch.where(
                canonical,
                original_contributions,
                torch.zeros_like(original_contributions),
            ),
        ):
            raise AssertionError(
                "relation contribution removal mask is not replayable"
            )
        recomputed_target = replayed_contributions.sum(dim=-1)
        recomputed_incremental = recomputed_target.sum(dim=-1)
        recomputed_cumulative = reverse_cumulative_atoms(
            recomputed_incremental, dim=1
        )
        if aggregation_kind == _V8_WITNESS_AGGREGATION:
            replayed_l1 = getattr(replayed_relation, "per_boundary_l1_usage", None)
            if replayed_l1 is None:
                raise AssertionError("V8 relation replay omitted its realized L1 use")
            expected_l1 = replayed_contributions.abs().sum(dim=(2, 3))
            if not torch.equal(replayed_l1, expected_l1):
                raise AssertionError("V8 relation replay reports stale capacity use")
    else:
        _, recomputed_incremental, recomputed_cumulative = (
            aggregate_ordinal_pair_messages(
                replayed_relation.edge_messages,
                replayed_relation.edge_valid_mask,
                replayed_relation.region_valid_mask,
                delta_cap=replayed_relation.delta_cap,
            )
        )
    reaggregation_error = max(
        float(
            (recomputed_incremental - replayed_relation.incremental_log_odds)
            .abs()
            .max()
            .cpu()
        ),
        float(
            (recomputed_cumulative - replayed_relation.cumulative_log_odds)
            .abs()
            .max()
            .cpu()
        ),
    )
    recomputed_rates = bounded_rate_log_odds_merge(
        replay_source,
        recomputed_cumulative,
        total_rate_cap=float(getattr(replayed, "total_rate_cap")),
    )
    merge_error = float(
        (recomputed_rates - _field(replayed, "total_rates")).abs().max().cpu()
    )
    if max(reaggregation_error, merge_error) > 2e-10:
        raise AssertionError("relation replay does not reproduce its aggregation and merge")

    base_prediction = _decision(baseline, decision_rule).detach().cpu()
    new_prediction = _decision(replayed, decision_rule).detach().cpu()
    base_expected = _field(baseline, "expected_grade").detach().cpu()
    new_expected = _field(replayed, "expected_grade").detach().cpu()
    base_cumulative = _field(baseline, "cumulative_probs").detach().cpu()
    new_cumulative = _field(replayed, "cumulative_probs").detach().cpu()
    base_probabilities = _field(baseline, "class_probs").detach().cpu()
    new_probabilities = _field(replayed, "class_probs").detach().cpu()
    expected_delta = base_expected - new_expected
    cumulative_delta = base_cumulative - new_cumulative
    if float((expected_delta - cumulative_delta.sum(-1)).abs().max()) > 2e-10:
        raise AssertionError("relation deletion expected-grade identity failed")
    removed_cpu = removed.detach().cpu()
    removed_contributions_cpu = (
        None if removed_contributions is None else removed_contributions.detach().cpu()
    )
    records = []
    for row in range(len(labels)):
        records.append(
            {
                "sample_id": int(sample_ids[row]),
                "label": int(labels[row]),
                "baseline_prediction": int(base_prediction[row]),
                "replayed_prediction": int(new_prediction[row]),
                "baseline_expected_grade": float(base_expected[row]),
                "replayed_expected_grade": float(new_expected[row]),
                "expected_grade_delta": float(expected_delta[row]),
                "removed_edge_message_l1": float(removed_cpu[row].abs().sum()),
                "removed_edge_message_signed_sum": float(removed_cpu[row].sum()),
                "cumulative_probability_delta": cumulative_delta[row].tolist(),
                "baseline_class_probs": base_probabilities[row].tolist(),
                "replayed_class_probs": new_probabilities[row].tolist(),
            }
        )
        if removed_contributions_cpu is not None:
            records[-1].update(
                {
                    "removed_edge_log_odds_l1": float(
                        removed_contributions_cpu[row].abs().sum()
                    ),
                    "removed_edge_log_odds_signed_sum": float(
                        removed_contributions_cpu[row].sum()
                    ),
                }
            )
    return records, source_error, partition_error, reaggregation_error, merge_error


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
    base_rates_all: list[torch.Tensor] = []
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
    grade0_log_identity_error = 0.0
    boundary0_fp64_reconstruction_error = 0.0
    boundary0_fp32_reduction_difference = 0.0
    relation_presence: bool | None = None
    relation_variant: str | None = None
    relation_aggregation_kind: str | None = None
    relation_cumulative_all: list[torch.Tensor] = []
    relation_incremental_all: list[torch.Tensor] = []
    relation_edge_abs_sums: list[torch.Tensor] = []
    relation_valid_edge_counts: list[torch.Tensor] = []
    sparse_candidate_counts: list[torch.Tensor] = []
    sparse_selected_counts: list[torch.Tensor] = []
    sparse_nonzero_counts: list[torch.Tensor] = []
    sparse_selected_densities: list[torch.Tensor] = []
    sparse_shortlist_counts: list[torch.Tensor] = []
    sparse_effective_witness_counts: list[torch.Tensor] = []
    sparse_top_witness_allocations: list[torch.Tensor] = []
    sparse_allocated_capacities: list[torch.Tensor] = []
    sparse_l1_usages: list[torch.Tensor] = []
    sparse_trace_maxima: dict[str, float] = defaultdict(float)
    sparse_edge_budget: int | None = None
    relation_all_edge_effects: list[dict[str, Any]] = []
    relation_top_edge_effects: list[dict[str, Any]] = []
    relation_replay_source_error = 0.0
    relation_replay_partition_error = 0.0
    relation_reaggregation_error = 0.0
    relation_merge_error = 0.0
    relation_all_removed_to_base_error = 0.0

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
            base_total_rates = getattr(output, "base_total_rates", total_rates)
            if not torch.is_tensor(base_total_rates):
                raise TypeError("base_total_rates must be a tensor")
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

            # Match the model's conservation path: the public maps are NHWK
            # views of an NCHW FP32 ledger, which is cast to FP64 before each
            # spatial reduction. A direct FP32 NHWK reduction can differ by
            # several ulps over the 160x160 s4 lattice.
            batch_scale_totals = torch.stack(
                [
                    rate_maps[name]
                    .permute(0, 3, 1, 2)
                    .to(dtype=total_rates.dtype)
                    .sum(dim=(-2, -1))
                    for name in scale_names
                ],
                dim=1,
            )
            summed_local_boundary0 = batch_scale_totals[:, :, 0].sum(1)
            fp32_summed_local_boundary0 = torch.stack(
                [rate_maps[name].sum(dim=(1, 2))[:, 0] for name in scale_names],
                dim=1,
            ).sum(1)
            conserved_unary_boundary0 = (
                base_total_rates[:, 0] - prior_rates[:, 0].to(base_total_rates.dtype)
            )
            effective_local_boundary0 = (
                total_rates[:, 0] - prior_rates[:, 0].to(total_rates.dtype)
            )
            fp64_reconstruction_error = float((
                conserved_unary_boundary0
                - summed_local_boundary0.to(base_total_rates.dtype)
            ).abs().max())
            fp64_identity_tolerance = max(
                2e-10,
                128.0
                * torch.finfo(torch.float64).eps
                * max(1.0, float(total_rates[:, 0].abs().max().cpu())),
            )
            boundary0_fp64_reconstruction_error = max(
                boundary0_fp64_reconstruction_error, fp64_reconstruction_error
            )
            if fp64_reconstruction_error > fp64_identity_tolerance:
                raise AssertionError("FP64 boundary-0 ledger conservation failed")
            fp32_reduction_difference = float((
                conserved_unary_boundary0
                - fp32_summed_local_boundary0.to(base_total_rates.dtype)
            ).abs().max())
            boundary0_fp32_reduction_difference = max(
                boundary0_fp32_reduction_difference, fp32_reduction_difference
            )
            # This alternate reduction is deliberately non-canonical and is
            # retained only to quantify source-ledger roundoff. The exact
            # conservation assertion above is the one that may fail the audit.
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

            relation = getattr(output, "relation_evidence", None)
            has_relation = relation is not None
            if relation_presence is None:
                relation_presence = has_relation
            elif relation_presence != has_relation:
                raise AssertionError("relation evidence presence changed between batches")
            if relation is not None:
                batch_variant = str(getattr(relation, "variant", "dense_v1"))
                batch_aggregation = str(
                    getattr(
                        relation,
                        "aggregation_kind",
                        "geometry_normalized_tanh_v1",
                    )
                )
                if relation_variant is None:
                    relation_variant = batch_variant
                    relation_aggregation_kind = batch_aggregation
                elif (
                    relation_variant != batch_variant
                    or relation_aggregation_kind != batch_aggregation
                ):
                    raise AssertionError(
                        "relation variant or aggregation contract changed between batches"
                    )
                if relation.edge_messages.shape[:2] != (len(labels), num_boundaries):
                    raise ValueError("relation edge-message boundary shape is invalid")
                if relation.edge_valid_mask.shape != (
                    len(labels),
                    relation.edge_messages.shape[-2],
                    relation.edge_messages.shape[-1],
                ):
                    raise ValueError("relation edge-valid mask shape is invalid")
                if not bool(torch.isfinite(relation.edge_messages).all()):
                    raise FloatingPointError("relation edge messages are non-finite")
                invalid_messages = relation.edge_messages.masked_select(
                    ~relation.edge_valid_mask[:, None]
                )
                if invalid_messages.numel() and bool((invalid_messages != 0).any()):
                    raise AssertionError("invalid relation edges carry a message")
                relation_cumulative_all.append(
                    relation.cumulative_log_odds.detach().cpu()
                )
                relation_incremental_all.append(
                    relation.incremental_log_odds.detach().cpu()
                )
                relation_edge_abs_sums.append(
                    relation.edge_messages.detach().abs().sum(dim=(1, 2, 3)).cpu()
                )
                relation_valid_edge_counts.append(
                    relation.edge_valid_mask.detach().sum(dim=(1, 2)).cpu()
                )
                sparse_relation = batch_aggregation in {
                    _V7_SPARSE_AGGREGATION,
                    _V8_WITNESS_AGGREGATION,
                }
                if sparse_relation:
                    sparse_trace = (
                        _audit_conserved_witness_relation_trace(relation)
                        if batch_aggregation == _V8_WITNESS_AGGREGATION
                        else _audit_sparse_relation_trace(relation)
                    )
                    batch_budget = int(sparse_trace["edge_budget"])
                    if sparse_edge_budget is None:
                        sparse_edge_budget = batch_budget
                    elif sparse_edge_budget != batch_budget:
                        raise AssertionError(
                            "sparse relation edge budget changed between batches"
                        )
                    sparse_candidate_counts.append(sparse_trace["candidate_count"])
                    sparse_selected_counts.append(sparse_trace["selected_count"])
                    sparse_nonzero_counts.append(sparse_trace["nonzero_count"])
                    sparse_selected_densities.append(
                        sparse_trace["selected_density"]
                    )
                    if batch_aggregation == _V8_WITNESS_AGGREGATION:
                        sparse_shortlist_counts.append(
                            sparse_trace["shortlist_count"]
                        )
                        sparse_effective_witness_counts.append(
                            sparse_trace["effective_witness_count"]
                        )
                        sparse_top_witness_allocations.append(
                            sparse_trace["top_witness_allocation"]
                        )
                        sparse_allocated_capacities.append(
                            sparse_trace["per_boundary_allocated_capacity"]
                        )
                        sparse_l1_usages.append(
                            sparse_trace["per_boundary_l1_usage"]
                        )
                    for name, value in sparse_trace.items():
                        if name.startswith("max_"):
                            sparse_trace_maxima[name] = max(
                                sparse_trace_maxima[name], float(value)
                            )

                (
                    all_edge_records,
                    source_error,
                    partition_error,
                    reaggregation_error,
                    merge_error,
                ) = (
                    _relation_replay_records(
                        model,
                        output,
                        (
                            relation.selected_edge_mask
                            if sparse_relation
                            else relation.edge_valid_mask
                        ),
                        labels_cpu,
                        sample_ids,
                        decision_rule,
                    )
                )
                relation_all_edge_effects.extend(all_edge_records)
                relation_replay_source_error = max(
                    relation_replay_source_error, source_error
                )
                relation_replay_partition_error = max(
                    relation_replay_partition_error, partition_error
                )
                relation_reaggregation_error = max(
                    relation_reaggregation_error, reaggregation_error
                )
                relation_merge_error = max(relation_merge_error, merge_error)
                all_removed = model.replay_without_relations(
                    output,
                    (
                        relation.selected_edge_mask
                        if sparse_relation
                        else relation.edge_valid_mask
                    ),
                    force_decoder_fp64=True,
                ).output
                relation_all_removed_to_base_error = max(
                    relation_all_removed_to_base_error,
                    float(
                        (
                            _field(all_removed, "total_rates")
                            - base_total_rates
                        )
                        .abs()
                        .max()
                        .cpu()
                    ),
                )

                if sparse_relation:
                    top_records, errors = _most_pivotal_sparse_edge_records(
                        model,
                        output,
                        relation,
                        labels_cpu,
                        sample_ids,
                        decision_rule,
                        all_edge_records,
                    )
                    source_error, partition_error, reaggregation_error, merge_error = (
                        errors
                    )
                    relation_replay_source_error = max(
                        relation_replay_source_error, source_error
                    )
                    relation_replay_partition_error = max(
                        relation_replay_partition_error, partition_error
                    )
                    relation_reaggregation_error = max(
                        relation_reaggregation_error, reaggregation_error
                    )
                    relation_merge_error = max(relation_merge_error, merge_error)
                    for record in top_records:
                        index = int(record["sample_id"])
                        if (
                            validation_items is not None
                            and 0 <= index < len(validation_items)
                        ):
                            record["image_id"] = Path(
                                validation_items[index][0]
                            ).name
                        relation_top_edge_effects.append(record)
                else:
                    ranked = relation.edge_messages.abs().masked_fill(
                        ~relation.edge_valid_mask[:, None], -torch.inf
                    ).flatten(1)
                    if bool((torch.isfinite(ranked).sum(1) == 0).any()):
                        raise AssertionError(
                            "sample has no valid regional relation edge"
                        )
                    selected = ranked.argmax(1)
                    edge_mask = torch.zeros_like(
                        relation.edge_messages, dtype=torch.bool
                    ).flatten(1)
                    edge_mask.scatter_(1, selected[:, None], True)
                    edge_mask = edge_mask.reshape_as(relation.edge_messages)
                    (
                        top_records,
                        source_error,
                        partition_error,
                        reaggregation_error,
                        merge_error,
                    ) = _relation_replay_records(
                        model,
                        output,
                        edge_mask,
                        labels_cpu,
                        sample_ids,
                        decision_rule,
                    )
                    relation_replay_source_error = max(
                        relation_replay_source_error, source_error
                    )
                    relation_replay_partition_error = max(
                        relation_replay_partition_error, partition_error
                    )
                    relation_reaggregation_error = max(
                        relation_reaggregation_error, reaggregation_error
                    )
                    relation_merge_error = max(
                        relation_merge_error, merge_error
                    )
                    regions = relation.edge_messages.shape[-1]
                    centers = relation.region_centers_yx.detach().cpu()
                    flat_message = relation.edge_messages.detach().cpu().flatten(1)
                    for row, record in enumerate(top_records):
                        flat_index = int(selected[row])
                        boundary = flat_index // (regions * regions)
                        endpoints = flat_index % (regions * regions)
                        target = endpoints // regions
                        source = endpoints % regions
                        record.update(
                            {
                                "selection": "largest_absolute_stored_pair_message",
                                "boundary": boundary,
                                "target_region": target,
                                "source_region": source,
                                "target_center_yx": centers[target].tolist(),
                                "source_center_yx": centers[source].tolist(),
                                "signed_pair_message": float(
                                    flat_message[row, flat_index]
                                ),
                                "region_receptive_field_pixels": int(
                                    relation.region_receptive_field
                                ),
                                "region_output_stride_pixels": int(
                                    relation.region_output_stride
                                ),
                            }
                        )
                        index = int(record["sample_id"])
                        if (
                            validation_items is not None
                            and 0 <= index < len(validation_items)
                        ):
                            record["image_id"] = Path(
                                validation_items[index][0]
                            ).name
                        relation_top_edge_effects.append(record)

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
                        spatial = [
                            int(value)
                            for value in np.unravel_index(
                                int(local),
                                tuple(int(size) for size in valid_masks[scale].shape[1:]),
                            )
                        ]
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
                    spatial = [
                        int(value)
                        for value in np.unravel_index(
                            int(flat_index),
                            tuple(int(size) for size in valid_masks[scale].shape[1:]),
                        )
                    ]
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
            relation_trace = getattr(prior_only, "relation_evidence", None)
            if relation_trace is not None:
                replay_relations = getattr(model, "replay_without_relations", None)
                if not callable(replay_relations):
                    raise TypeError(
                        "relation-enabled ORIGIN must expose replay_without_relations"
                    )
                prior_only = replay_relations(
                    prior_only,
                    relation_trace.edge_valid_mask,
                    force_decoder_fp64=True,
                ).output
            prior_only_p0 = _field(prior_only, "class_probs")[:, 0].double()
            full_log_p0 = _field(output, "log_class_probs")[:, 0].double()
            prior_only_log_p0 = _field(prior_only, "log_class_probs")[:, 0].double()
            # Use the conserved FP64 total-minus-prior value in this exact
            # generator identity. Re-reducing the exposed FP32 maps in a
            # different order is equivalent mathematically but can differ by
            # several source-ledger ulps; that discrepancy is audited above.
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
            log_identity_tolerance = max(
                2e-10,
                128.0
                * torch.finfo(torch.float64).eps
                * max(1.0, float(total_rates[:, 0].abs().max().cpu())),
            )
            batch_log_identity_error = max(
                float((full_log_p0 + total_rates[:, 0].double()).abs().max().cpu()),
                float((prior_only_log_p0 + prior_rates[:, 0].double()).abs().max().cpu()),
                float((
                    full_log_p0
                    - prior_only_log_p0
                    + effective_local_boundary0.double()
                ).abs().max().cpu()),
            )
            grade0_log_identity_error = max(
                grade0_log_identity_error, batch_log_identity_error
            )
            if batch_log_identity_error > log_identity_tolerance:
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
                    base_rate_value = float(base_total_rates[row, boundary].cpu())
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
                            base_rate_value - prior_value - local_value
                        ),
                        "base_pre_relation_rate": base_rate_value,
                        "relation_rate_delta": float(value) - base_rate_value,
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
            base_rates_all.append(base_total_rates.detach().cpu())
            prior_rates_all.append(prior_rates.detach().cpu())
            local_boundary0_all.append(effective_local_boundary0.detach().cpu())

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
    base_total_rates = torch.cat(base_rates_all).double()
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
        "identity": (
            "base_pre_relation_boundary0_rate = prior_boundary0_rate + "
            "summed_unary_local_boundary0_evidence"
            if relation_presence
            else "total_boundary0_rate = prior_boundary0_rate + "
            "summed_local_boundary0_evidence"
        ),
        "prior_only_identity": "P(Y=0 | all local rates deleted) = exp(-prior_boundary0_rate)",
        "local_factorization_identity": (
            "P_full(Y=0) / P_prior_only(Y=0) = "
            "exp(-(final_boundary0_rate-prior_boundary0_rate))"
        ),
        "max_prior_only_identity_error": prior_only_identity_error,
        "max_log_space_local_factorization_identity_error": (
            grade0_log_identity_error
        ),
        "max_fp64_ledger_reconstruction_error": (
            boundary0_fp64_reconstruction_error
        ),
        "max_alternative_fp32_reduction_difference": (
            boundary0_fp32_reduction_difference
        ),
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

    relation_diagnostics: dict[str, Any] | None = None
    relation_certificates: list[dict[str, Any]] = []
    if relation_presence:
        relation_cumulative = torch.cat(relation_cumulative_all).double()
        relation_incremental = torch.cat(relation_incremental_all).double()
        relation_edge_l1 = torch.cat(relation_edge_abs_sums).double()
        relation_edge_counts = torch.cat(relation_valid_edge_counts).long()
        if relation_all_removed_to_base_error > 2e-10:
            raise AssertionError(
                "removing every regional relation did not recover the v3 base rates"
            )
        for grade in range(num_classes):
            members = [
                item
                for item in relation_top_edge_effects
                if int(item["label"]) == grade
            ]
            members.sort(
                key=lambda item: hashlib.sha256(
                    f"{split_signature}:relation:{grade}:{item['sample_id']}".encode()
                ).hexdigest()
            )
            relation_certificates.extend(members[:certificates_per_grade])
        all_edge_summary = _relation_effect_summary(
            relation_all_edge_effects, quantiles
        )
        top_edge_summary = _relation_effect_summary(
            relation_top_edge_effects, quantiles
        )
        relation_diagnostics = {
            "contract": (
                "stored_edge_log_odds_contributions_to_cumulative_transition_rate_log_odds"
                if relation_aggregation_kind == _V8_WITNESS_AGGREGATION
                else "stored_pair_messages_to_cumulative_transition_rate_log_odds"
            ),
            "variant": relation_variant,
            "aggregation_kind": relation_aggregation_kind,
            "cumulative_log_odds_bound": float(
                getattr(
                    getattr(getattr(model, "generator", None), "relation_field", None),
                    "delta_cap",
                    0.0,
                )
            ),
            "cumulative_log_odds_quantiles_by_boundary": {
                str(boundary): _quantiles(
                    relation_cumulative[:, boundary].tolist(), quantiles
                )
                for boundary in range(num_boundaries)
            },
            "incremental_log_odds_quantiles_by_atom": {
                str(boundary): _quantiles(
                    relation_incremental[:, boundary].tolist(), quantiles
                )
                for boundary in range(num_boundaries)
            },
            "edge_message_l1_per_image_quantiles": _quantiles(
                relation_edge_l1.tolist(), quantiles
            ),
            "valid_directed_edge_count_quantiles": _quantiles(
                relation_edge_counts.tolist(), quantiles
            ),
            "all_edge_deletion_effects": all_edge_summary,
            (
                "most_pivotal_selected_edge_deletion_effects"
                if relation_aggregation_kind
                in {_V7_SPARSE_AGGREGATION, _V8_WITNESS_AGGREGATION}
                else "largest_absolute_edge_deletion_effects"
            ): top_edge_summary,
            "full_vs_no_relation_ablation": _relation_ablation_summary(
                relation_all_edge_effects
            ),
            "max_exact_source_ledger_invariance_error": relation_replay_source_error,
            "max_exact_edge_partition_error": relation_replay_partition_error,
            "max_exact_relation_reaggregation_error": relation_reaggregation_error,
            "max_exact_relation_merge_error": relation_merge_error,
            "max_all_edges_removed_to_base_rate_error": (
                relation_all_removed_to_base_error
            ),
            "grade_stratified_relation_certificates": relation_certificates,
        }
        if relation_aggregation_kind in {
            _V7_SPARSE_AGGREGATION,
            _V8_WITNESS_AGGREGATION,
        }:
            candidate_counts = torch.cat(sparse_candidate_counts).double()
            selected_counts = torch.cat(sparse_selected_counts).double()
            nonzero_counts = torch.cat(sparse_nonzero_counts).double()
            selected_densities = torch.cat(sparse_selected_densities).double()
            relation_diagnostics["sparse_support_and_identifiability"] = {
                "edge_budget_per_boundary": sparse_edge_budget,
                "candidate_edge_count_quantiles": _quantiles(
                    candidate_counts.tolist(), quantiles
                ),
                "selected_edge_count_quantiles_by_boundary": {
                    str(boundary): _quantiles(
                        selected_counts[:, boundary].tolist(), quantiles
                    )
                    for boundary in range(num_boundaries)
                },
                "nonzero_edge_count_quantiles_by_boundary": {
                    str(boundary): _quantiles(
                        nonzero_counts[:, boundary].tolist(), quantiles
                    )
                    for boundary in range(num_boundaries)
                },
                "selected_candidate_density_quantiles_by_boundary": {
                    str(boundary): _quantiles(
                        selected_densities[:, boundary].tolist(), quantiles
                    )
                    for boundary in range(num_boundaries)
                },
                **dict(sparse_trace_maxima),
            }
            if relation_aggregation_kind == _V8_WITNESS_AGGREGATION:
                shortlist_counts = torch.cat(sparse_shortlist_counts).double()
                effective_counts = torch.cat(
                    sparse_effective_witness_counts
                ).double()
                top_allocations = torch.cat(
                    sparse_top_witness_allocations
                ).double()
                allocated_capacities = torch.cat(
                    sparse_allocated_capacities
                ).double()
                l1_usages = torch.cat(sparse_l1_usages).double()
                capacity_fraction = torch.where(
                    allocated_capacities > 0,
                    l1_usages / allocated_capacities,
                    torch.zeros_like(l1_usages),
                )
                relation_diagnostics["conserved_witness_allocation"] = {
                    "semantics": (
                        "unit allocation conserves available per-boundary capacity; "
                        "realized L1 use may be lower because the signed score and "
                        "relation-strength gates are bounded"
                    ),
                    "shortlist_count_quantiles_by_boundary": {
                        str(boundary): _quantiles(
                            shortlist_counts[:, boundary].tolist(), quantiles
                        )
                        for boundary in range(num_boundaries)
                    },
                    "effective_witness_count_quantiles_by_boundary": {
                        str(boundary): _quantiles(
                            effective_counts[:, boundary].tolist(), quantiles
                        )
                        for boundary in range(num_boundaries)
                    },
                    "top_witness_allocation_quantiles_by_boundary": {
                        str(boundary): _quantiles(
                            top_allocations[:, boundary].tolist(), quantiles
                        )
                        for boundary in range(num_boundaries)
                    },
                    "allocated_capacity_quantiles_by_boundary": {
                        str(boundary): _quantiles(
                            allocated_capacities[:, boundary].tolist(), quantiles
                        )
                        for boundary in range(num_boundaries)
                    },
                    "realized_l1_usage_quantiles_by_boundary": {
                        str(boundary): _quantiles(
                            l1_usages[:, boundary].tolist(), quantiles
                        )
                        for boundary in range(num_boundaries)
                    },
                    "realized_to_allocated_capacity_fraction_quantiles_by_boundary": {
                        str(boundary): _quantiles(
                            capacity_fraction[:, boundary].tolist(), quantiles
                        )
                        for boundary in range(num_boundaries)
                    },
                }
            pivotal_absolute = np.asarray(
                [
                    abs(float(item["expected_grade_delta"]))
                    for item in relation_top_edge_effects
                ],
                dtype=np.float64,
            )
            positive_fraction = float(
                all_edge_summary["positive_effect_count_above_1e-5"]
                / max(1, all_edge_summary["n"])
            )
            negative_fraction = float(
                all_edge_summary["negative_effect_count_below_minus_1e-5"]
                / max(1, all_edge_summary["n"])
            )
            strength_checks = {
                "at_least_one_top_edge_changes_a_prediction": bool(
                    top_edge_summary["prediction_change_count"] > 0
                ),
                "median_top_edge_absolute_expected_grade_effect_at_least_1e-3": bool(
                    float(np.quantile(pivotal_absolute, 0.5)) >= 1e-3
                ),
                "q90_top_edge_absolute_expected_grade_effect_at_least_1e-2": bool(
                    float(np.quantile(pivotal_absolute, 0.9)) >= 1e-2
                ),
                "at_least_5pct_collective_positive_effects": bool(
                    positive_fraction >= 0.05
                ),
                "at_least_5pct_collective_negative_effects": bool(
                    negative_fraction >= 0.05
                ),
            }
            relation_diagnostics["individual_certificate_strength_gate"] = {
                "scope": "pre_registered_engineering_gate_not_a_mathematical_guarantee",
                "positive_collective_effect_fraction_above_1e-5": positive_fraction,
                "negative_collective_effect_fraction_below_minus_1e-5": negative_fraction,
                "median_top_edge_absolute_expected_grade_effect": float(
                    np.quantile(pivotal_absolute, 0.5)
                ),
                "q90_top_edge_absolute_expected_grade_effect": float(
                    np.quantile(pivotal_absolute, 0.9)
                ),
                "checks": strength_checks,
                "passed": all(strength_checks.values()),
            }

    return {
        "schema": (
            "origin-full-validation-audit-v8"
            if relation_aggregation_kind == _V8_WITNESS_AGGREGATION
            else (
                "origin-full-validation-audit-v7"
                if relation_aggregation_kind == _V7_SPARSE_AGGREGATION
                else (
                    "origin-full-validation-audit-v6"
                    if relation_presence
                    else "origin-full-validation-audit-v2"
                )
            )
        ),
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
        "relation_interaction_diagnostics": relation_diagnostics,
        "interpretation_scope": (
            "exact_stored_unary_and_pair_ledger_deletion_not_causal_pixel_masking"
            if relation_presence
            else "exact_stored_ledger_deletion_not_causal_pixel_masking"
        ),
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


def _checkpoint_role_result_contract(
    state: Mapping[str, Any],
    completed_result: Mapping[str, Any],
    *,
    checkpoint_sha256: str | None = None,
) -> tuple[str, int, Mapping[str, Any]]:
    """Resolve the selected-result record for legacy and role-tagged checkpoints.

    V8 deliberately writes both a deployable safety-floor-selected checkpoint
    and a best learned checkpoint.  Treating either one as the other can hide
    a failed learned relation behind the epoch-0 V3 floor, so the production
    audit requires an explicit role and binds it to a distinct result record.
    """

    schema = str(state.get("schema", ""))
    if schema != "origin-checkpoint-v8":
        metrics = completed_result.get(
            "best_validation", completed_result.get("best_validation_metrics", {})
        )
        if not isinstance(metrics, Mapping):
            raise ValueError("completed result has invalid best-validation metrics")
        return "legacy_selected", int(completed_result.get("best_epoch", -1)), metrics

    role = str(state.get("checkpoint_role", ""))
    if role not in _V8_CHECKPOINT_ROLES:
        raise ValueError(f"v8 checkpoint has an unknown checkpoint role: {role!r}")
    if role == "resume_state":
        raise ValueError("full validation audit refuses a resume-state checkpoint")

    checkpoint_epoch = int(state.get("epoch", -1))
    phase = str(state.get("training_phase", ""))
    if role == "hash_bound_v3_floor":
        if checkpoint_epoch != 0 or phase != "hash_bound_v3_floor":
            raise ValueError("v8 floor checkpoint role is inconsistent with epoch/phase")
        if not bool(
            completed_result.get("best_checkpoint_is_hash_bound_v3_floor", False)
        ):
            raise ValueError("v8 result does not select the declared V3 floor")
        epoch = int(completed_result.get("best_epoch", -1))
        metrics = completed_result.get(
            "best_validation", completed_result.get("best_validation_metrics", {})
        )
    elif role == "deployable_selected":
        if checkpoint_epoch < 1 or phase == "hash_bound_v3_floor":
            raise ValueError("deployable V8 checkpoint is not a learned candidate")
        if bool(completed_result.get("best_checkpoint_is_hash_bound_v3_floor", False)):
            raise ValueError("deployable V8 role conflicts with a selected floor result")
        epoch = int(completed_result.get("best_epoch", -1))
        metrics = completed_result.get(
            "best_validation", completed_result.get("best_validation_metrics", {})
        )
    else:  # best_learned
        if checkpoint_epoch < 1 or phase == "hash_bound_v3_floor":
            raise ValueError("best-learned V8 checkpoint is not a learned candidate")
        if "best_learned_epoch" not in completed_result:
            raise ValueError("v8 result omits best_learned_epoch")
        epoch = int(completed_result["best_learned_epoch"])
        metrics = completed_result.get(
            "best_learned_validation",
            completed_result.get("best_learned_validation_metrics", {}),
        )
    if not isinstance(metrics, Mapping) or not metrics:
        raise ValueError(f"v8 result omits metrics for checkpoint role {role!r}")
    if epoch != checkpoint_epoch:
        raise ValueError(
            f"v8 checkpoint epoch differs from its {role!r} result record"
        )
    if checkpoint_sha256 is not None:
        if role == "best_learned":
            result_role = completed_result.get("best_learned_checkpoint_role")
            result_hash = completed_result.get("best_learned_checkpoint_sha256")
        else:
            result_role = completed_result.get("selected_checkpoint_role")
            result_hash = completed_result.get(
                "deployable_best_checkpoint_sha256",
                completed_result.get("best_checkpoint_sha256"),
            )
        if result_role != role:
            raise ValueError("v8 result checkpoint role differs from checkpoint")
        if result_hash != checkpoint_sha256:
            raise ValueError("v8 result checkpoint hash differs from audited artifact")
    return role, epoch, metrics


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
        else fold_dir / "audits" / "full_validation_audit_v2.json"
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
    if state.get("schema") not in {
        "origin-checkpoint-v3",
        "origin-checkpoint-v6",
        "origin-checkpoint-v7",
        "origin-checkpoint-v8",
    }:
        raise ValueError("full validation audit requires an ORIGIN checkpoint")
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
    checkpoint_role, result_checkpoint_epoch, result_metrics = (
        _checkpoint_role_result_contract(
            state,
            completed_result,
            checkpoint_sha256=checkpoint_hash_before,
        )
    )
    if result_checkpoint_epoch != int(state.get("epoch", -2)):
        raise ValueError("checkpoint epoch differs from its completed-result record")
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
        relation_enabled=cfg.relation_enabled,
        relation_source_scale=cfg.relation_source_scale,
        relation_grid_size=cfg.relation_grid_size,
        relation_dim=cfg.relation_dim,
        relation_head_dim=cfg.relation_head_dim,
        relation_delta_cap=cfg.relation_delta_cap,
        relation_variant=cfg.relation_variant,
        relation_edge_budget=cfg.relation_edge_budget,
        relation_allocation_temperature=cfg.relation_allocation_temperature,
        relation_permutation_seed=cfg.relation_permutation_seed,
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
    warm_start_provenance = state.get("warm_start_provenance")
    warm_start_metric_safety_floor = state.get("warm_start_metric_safety_floor")
    selected_metric_safety_floor_evaluation = state.get(
        "candidate_metric_safety_floor_evaluation"
    )
    selected_training_phase = state.get("training_phase")
    if cfg.relation_enabled:
        if not isinstance(warm_start_provenance, Mapping):
            raise ValueError("relation checkpoint omits hash-bound v3 provenance")
        if not isinstance(warm_start_metric_safety_floor, Mapping):
            raise ValueError("relation checkpoint omits its v3 multi-metric safety floor")
        if not isinstance(selected_metric_safety_floor_evaluation, Mapping):
            raise ValueError("relation checkpoint omits candidate safety-floor evaluation")
        if checkpoint_role != "best_learned" and not bool(
            selected_metric_safety_floor_evaluation.get("checkpoint_eligible", False)
        ):
            raise ValueError("deployable relation checkpoint was not safety-floor eligible")
        expected_phase = (
            "hash_bound_v3_floor"
            if checkpoint_epoch == 0
            else (
                "relation_only"
                if checkpoint_epoch <= cfg.relation_only_epochs
                else "joint"
            )
        )
        if selected_training_phase != expected_phase:
            raise ValueError(
                "relation checkpoint training phase is inconsistent with its epoch"
            )
        if completed_result.get("warm_start_metric_safety_floor") != dict(
            warm_start_metric_safety_floor
        ):
            raise ValueError(
                "completed result safety-floor thresholds differ from checkpoint"
            )
        result_safety_evaluation = (
            completed_result.get("best_learned_metric_safety_floor_evaluation")
            if checkpoint_role == "best_learned"
            else completed_result.get(
                "selected_checkpoint_metric_safety_floor_evaluation"
            )
        )
        if result_safety_evaluation != dict(selected_metric_safety_floor_evaluation):
            raise ValueError(
                "completed result role-specific safety-floor evaluation differs from checkpoint"
            )
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
            "checkpoint_role": checkpoint_role,
            "checkpoint_sha256": checkpoint_hash_before,
            "architecture_signature": architecture_signature,
            "implementation_signature": implementation_signature,
            "checkpoint_config_signature": checkpoint_config_signature,
            "warm_start_provenance": warm_start_provenance,
            "warm_start_metric_safety_floor": warm_start_metric_safety_floor,
            "selected_checkpoint_metric_safety_floor_evaluation": (
                selected_metric_safety_floor_evaluation
            ),
            "selected_checkpoint_training_phase": selected_training_phase,
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
    _assert_metric_reproduction(payload["metrics"], checkpoint_metrics, source="checkpoint")
    _assert_metric_reproduction(
        payload["metrics"], result_metrics, source=f"completed result ({checkpoint_role})"
    )
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
