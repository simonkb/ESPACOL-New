"""Fail-closed validation-audit checks for ORIGIN-v7 sparse relations."""

from __future__ import annotations

import copy
from types import SimpleNamespace

import pytest
import torch

from audit_origin_validation import (
    _audit_sparse_relation_trace,
    _masked_directed_anova_projection,
    _relation_ablation_summary,
)
from models.origin import (
    _hard_stable_pair_budget,
    aggregate_sparse_ordinal_pair_messages,
    masked_offdiagonal_pair_anova,
)


def _trace(*, batch: int = 2, boundaries: int = 4, regions: int = 5):
    torch.manual_seed(913)
    region_valid = torch.ones(batch, regions, dtype=torch.bool)
    diagonal = torch.eye(regions, dtype=torch.bool)
    candidate = region_valid[:, :, None] & region_valid[:, None, :] & ~diagonal[None]
    unidentified_proposal = torch.randn(batch, boundaries, regions, regions)
    unidentified_interaction = torch.randn(batch, boundaries, regions, regions)
    unidentified_proposal = torch.where(
        candidate[:, None], unidentified_proposal, 0.0
    )
    unidentified_interaction = torch.where(
        candidate[:, None], unidentified_interaction, 0.0
    )
    pre_geometry_proposal, pre_geometry_additive_proposal = (
        masked_offdiagonal_pair_anova(
            unidentified_proposal, candidate, region_valid
        )
    )
    pre_geometry_interaction, pre_geometry_additive_interaction = (
        masked_offdiagonal_pair_anova(
            unidentified_interaction, candidate, region_valid
        )
    )
    geometry_modulation = 1.0 + 0.25 * torch.tanh(
        torch.randn(batch, boundaries, regions, regions)
    )
    raw_proposal = pre_geometry_proposal * geometry_modulation
    raw_interaction = pre_geometry_interaction * geometry_modulation
    projected_proposal, additive_proposal = masked_offdiagonal_pair_anova(
        raw_proposal, candidate, region_valid
    )
    interaction, additive_endpoint = masked_offdiagonal_pair_anova(
        raw_interaction, candidate, region_valid
    )
    budget = 3
    selected = _hard_stable_pair_budget(
        projected_proposal, candidate, edge_budget=budget
    )
    messages = torch.where(selected, interaction.tanh(), torch.zeros_like(interaction))
    contributions, target, incremental, cumulative = (
        aggregate_sparse_ordinal_pair_messages(
            messages,
            selected,
            candidate,
            region_valid,
            delta_cap=2.0,
            edge_budget=budget,
        )
    )
    return SimpleNamespace(
        variant="identified_sparse_v1",
        aggregation_kind="fixed_budget_linear_conserved_v1",
        raw_proposal_scores=raw_proposal,
        unidentified_proposal_scores=unidentified_proposal,
        pre_geometry_proposal_residuals=pre_geometry_proposal,
        pre_geometry_additive_proposal_scores=(
            pre_geometry_additive_proposal
        ),
        projected_proposal_scores=projected_proposal,
        additive_proposal_scores=additive_proposal,
        raw_interaction_scores=raw_interaction,
        unidentified_interaction_scores=unidentified_interaction,
        pre_geometry_interaction_residuals=pre_geometry_interaction,
        pre_geometry_additive_endpoint_scores=(
            pre_geometry_additive_interaction
        ),
        interaction_residuals=interaction,
        additive_endpoint_scores=additive_endpoint,
        geometry_modulation=geometry_modulation,
        candidate_edge_mask=candidate,
        edge_valid_mask=candidate,
        selected_edge_mask=selected,
        pair_gates=selected.float(),
        edge_messages=messages,
        edge_log_odds_contributions=contributions,
        target_incremental_log_odds=target,
        incremental_log_odds=incremental,
        cumulative_log_odds=cumulative,
        region_valid_mask=region_valid,
        edge_budget=budget,
        delta_cap=2.0,
    )


def test_independent_masked_anova_annihilates_endpoint_main_effects() -> None:
    torch.manual_seed(13)
    scores = torch.randn(2, 3, 6, 6, dtype=torch.float64)
    valid = torch.tensor(
        [[True, True, True, True, False, False], [True] * 6]
    )
    candidate = (
        valid[:, :, None]
        & valid[:, None, :]
        & ~torch.eye(6, dtype=torch.bool)[None]
    )
    scores = torch.where(candidate[:, None], scores, 0.0)
    target = torch.randn(2, 3, 6, 1, dtype=torch.float64)
    source = torch.randn(2, 3, 1, 6, dtype=torch.float64)
    constant = torch.randn(2, 3, 1, 1, dtype=torch.float64)
    nuisance = torch.where(
        candidate[:, None], target + source + constant, 0.0
    )

    projected = _masked_directed_anova_projection(scores, candidate)
    projected_with_nuisance = _masked_directed_anova_projection(
        scores + nuisance, candidate
    )
    torch.testing.assert_close(
        projected_with_nuisance, projected, atol=2e-12, rtol=2e-12
    )
    torch.testing.assert_close(
        _masked_directed_anova_projection(projected, candidate),
        projected,
        atol=2e-12,
        rtol=2e-12,
    )
    masked = projected.masked_fill(~candidate[:, None], 0.0)
    torch.testing.assert_close(
        masked.sum(-1), torch.zeros_like(masked.sum(-1)), atol=2e-12, rtol=0
    )
    torch.testing.assert_close(
        masked.sum(-2), torch.zeros_like(masked.sum(-2)), atol=2e-12, rtol=0
    )


def test_sparse_trace_audit_reconstructs_budget_projection_and_contributions() -> None:
    trace = _trace()
    report = _audit_sparse_relation_trace(trace)

    assert report["variant"] == "identified_sparse_v1"
    assert report["aggregation_kind"] == "fixed_budget_linear_conserved_v1"
    assert report["edge_budget"] == 3
    assert torch.equal(report["selected_count"], torch.full((2, 4), 3))
    for name, value in report.items():
        if name.startswith("max_"):
            assert float(value) < 1e-4


@pytest.mark.parametrize(
    ("corruption", "message"),
    [
        ("support", "leaves candidate graph"),
        ("budget", "exact edge budget"),
        ("projection", "ANOVA projection"),
        ("contribution", "contribution compilation"),
    ],
)
def test_sparse_trace_audit_fails_closed_on_corruption(
    corruption: str, message: str
) -> None:
    trace = copy.deepcopy(_trace(batch=1))
    if corruption == "support":
        trace.selected_edge_mask[0, 0, 0, 0] = True
    elif corruption == "budget":
        index = trace.selected_edge_mask[0, 0].nonzero()[0]
        trace.selected_edge_mask[0, 0, index[0], index[1]] = False
        trace.edge_messages[0, 0, index[0], index[1]] = 0.0
        trace.edge_log_odds_contributions[0, 0, index[0], index[1]] = 0.0
    elif corruption == "projection":
        index = trace.candidate_edge_mask[0].nonzero()[0]
        trace.projected_proposal_scores[0, 0, index[0], index[1]] += 0.25
    elif corruption == "contribution":
        index = trace.selected_edge_mask[0, 0].nonzero()[0]
        trace.edge_log_odds_contributions[0, 0, index[0], index[1]] += 0.25
    else:  # pragma: no cover
        raise AssertionError(corruption)

    with pytest.raises(AssertionError, match=message):
        _audit_sparse_relation_trace(trace)


@pytest.mark.parametrize(
    "field",
    ("additive_proposal_scores", "additive_endpoint_scores"),
)
def test_sparse_trace_audit_fails_closed_on_nonfinite_additive_fields(
    field: str,
) -> None:
    trace = _trace(batch=1)
    value = getattr(trace, field)
    value[0, 0, 0, 1] = torch.nan
    with pytest.raises(FloatingPointError, match="non-finite"):
        _audit_sparse_relation_trace(trace)


def test_relation_ablation_reports_paired_help_harm_and_signs() -> None:
    records = [
        {
            "label": 1,
            "baseline_prediction": 1,
            "replayed_prediction": 0,
            "baseline_class_probs": [0.1, 0.8, 0.1],
            "replayed_class_probs": [0.8, 0.1, 0.1],
        },
        {
            "label": 2,
            "baseline_prediction": 1,
            "replayed_prediction": 2,
            "baseline_class_probs": [0.1, 0.8, 0.1],
            "replayed_class_probs": [0.1, 0.1, 0.8],
        },
        {
            "label": 0,
            "baseline_prediction": 0,
            "replayed_prediction": 0,
            "baseline_class_probs": [0.8, 0.1, 0.1],
            "replayed_class_probs": [0.8, 0.1, 0.1],
        },
    ]
    report = _relation_ablation_summary(records)
    assert report["prediction_change_count"] == 2
    assert report["full_relation_only_correct"] == 1
    assert report["no_relation_only_correct"] == 1
    assert report["net_correct"] == 0
    assert report["exact_two_sided_mcnemar_pvalue_descriptive"] == 1.0
    assert report["by_true_grade"]["1"]["full_relation_only_correct"] == 1
    assert report["by_true_grade"]["2"]["no_relation_only_correct"] == 1
