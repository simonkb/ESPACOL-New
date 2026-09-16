"""Fail-closed audit checks for ORIGIN-v8 conserved witness traces."""

from __future__ import annotations

import copy
from types import SimpleNamespace

import pytest
import torch

from audit_origin_validation import (
    _audit_conserved_witness_relation_trace,
    _checkpoint_role_result_contract,
    _entmax15_reference,
    _masked_directed_anova_projection,
)
from models.origin import reverse_cumulative_atoms


def _trace(
    *,
    variant: str = "identified_conserved_witness_v1",
    batch: int = 2,
    boundaries: int = 4,
    regions: int = 5,
) -> SimpleNamespace:
    torch.manual_seed(811)
    region_valid = torch.ones(batch, regions, dtype=torch.bool)
    diagonal = torch.eye(regions, dtype=torch.bool)
    candidate = region_valid[:, :, None] & region_valid[:, None, :] & ~diagonal[None]
    raw = torch.randn(batch, boundaries, regions, regions)
    raw = torch.where(candidate[:, None], raw, 0.0)
    scores = (
        raw
        if variant == "additive_conserved_witness_control_v1"
        else _masked_directed_anova_projection(raw, candidate).float()
    )
    budget = 4
    flat_candidate = candidate[:, None].expand_as(scores).flatten(2)
    ranked = scores.abs().masked_fill(~candidate[:, None], -torch.inf).flatten(2)
    shortlist_flat = torch.zeros_like(flat_candidate)
    shortlist_flat.scatter_(2, ranked.topk(budget, dim=2).indices, True)
    shortlist = shortlist_flat.reshape_as(scores)
    temperature = 0.7
    allocation = _entmax15_reference(
        scores.double().abs() / temperature, shortlist
    )
    active = allocation > 0
    strength = torch.tensor([0.8, -0.6, 0.35, -0.2])[:boundaries]
    messages = torch.where(
        active,
        strength.reshape(1, boundaries, 1, 1) * scores.tanh(),
        torch.zeros_like(scores),
    )
    contributions = (
        2.0
        / float(boundaries)
        * messages.double()
        * allocation.double()
    ).double()
    target = contributions.sum(-1)
    incremental = target.sum(-1)
    cumulative = reverse_cumulative_atoms(incremental, dim=1)
    return SimpleNamespace(
        variant=variant,
        aggregation_kind="conserved_sparse_witness_allocation_v1",
        identified_pair_scores=scores,
        candidate_edge_mask=candidate,
        edge_valid_mask=candidate,
        shortlist_edge_mask=shortlist,
        active_witness_mask=active,
        selected_edge_mask=active.clone(),
        witness_allocation=allocation,
        pair_gates=allocation.clone(),
        relation_strength=strength,
        edge_messages=messages,
        edge_log_odds_contributions=contributions,
        effective_witness_count=(
            1.0 / allocation.double().square().sum(dim=(2, 3))
        ),
        target_incremental_log_odds=target,
        incremental_log_odds=incremental,
        cumulative_log_odds=cumulative,
        region_valid_mask=region_valid,
        edge_budget=budget,
        allocation_temperature=temperature,
        delta_cap=2.0,
        per_boundary_allocated_capacity=(
            strength.abs().reshape(1, boundaries).expand(batch, -1)
            * (2.0 / float(boundaries))
        ),
        per_boundary_l1_usage=contributions.abs().sum(dim=(2, 3)),
    )


@pytest.mark.parametrize(
    "variant",
    (
        "identified_conserved_witness_v1",
        "additive_conserved_witness_control_v1",
        "shuffled_conserved_witness_control_v1",
    ),
)
def test_v8_trace_audit_replays_allocation_compiler_and_budget(variant: str) -> None:
    report = _audit_conserved_witness_relation_trace(_trace(variant=variant))

    assert report["variant"] == variant
    assert report["aggregation_kind"] == "conserved_sparse_witness_allocation_v1"
    assert report["edge_budget"] == 4
    assert torch.equal(report["shortlist_count"], torch.full((2, 4), 4))
    assert bool((report["selected_count"] <= 4).all())
    assert bool((report["selected_count"] >= 1).all())
    assert bool((report["effective_witness_count"] >= 1.0).all())
    for name, value in report.items():
        if name.startswith("max_"):
            assert float(value) < 1e-4


def test_v8_entmax_reference_is_sparse_normalized_and_deterministic() -> None:
    logits = torch.tensor([[[[8.0, 1.0], [0.0, -4.0]]]], dtype=torch.float64)
    support = torch.ones_like(logits, dtype=torch.bool)
    first = _entmax15_reference(logits, support)
    second = _entmax15_reference(logits, support)

    torch.testing.assert_close(first, second, atol=0, rtol=0)
    torch.testing.assert_close(first.sum((2, 3)), torch.ones(1, 1, dtype=torch.float64))
    assert int(first.ne(0).sum()) == 1


@pytest.mark.parametrize(
    ("corruption", "message"),
    (
        ("alias", "alias differs"),
        ("shortlist", "shortlist leaves"),
        ("allocation", "does not conserve"),
        ("entmax", "entmax witness allocation"),
        ("pair_gates", "pair gates"),
        ("effective", "effective witness count"),
        ("projection", "retains endpoint main effects"),
        ("message", "contribution compilation"),
        ("contribution", "contribution compilation"),
        ("capacity", "capacity use"),
        ("strength", "gate range"),
    ),
)
def test_v8_trace_audit_fails_closed_on_corruption(
    corruption: str, message: str
) -> None:
    trace = copy.deepcopy(_trace(batch=1))
    if corruption == "alias":
        index = trace.active_witness_mask[0, 0].nonzero()[0]
        trace.selected_edge_mask[0, 0, index[0], index[1]] = False
    elif corruption == "shortlist":
        trace.shortlist_edge_mask[0, 0, 0, 0] = True
    elif corruption == "allocation":
        index = trace.active_witness_mask[0, 0].nonzero()[0]
        trace.witness_allocation[0, 0, index[0], index[1]] += 0.1
        trace.pair_gates.copy_(trace.witness_allocation)
    elif corruption == "entmax":
        indices = trace.active_witness_mask[0, 0].nonzero()
        assert len(indices) >= 2
        first, second = indices[:2]
        amount = min(
            0.01,
            float(trace.witness_allocation[0, 0, second[0], second[1]]) / 2.0,
        )
        trace.witness_allocation[0, 0, first[0], first[1]] += amount
        trace.witness_allocation[0, 0, second[0], second[1]] -= amount
        trace.pair_gates.copy_(trace.witness_allocation)
    elif corruption == "pair_gates":
        index = trace.active_witness_mask[0, 0].nonzero()[0]
        trace.pair_gates[0, 0, index[0], index[1]] += 0.1
    elif corruption == "effective":
        trace.effective_witness_count[0, 0] += 0.1
    elif corruption == "projection":
        index = trace.candidate_edge_mask[0].nonzero()[0]
        trace.identified_pair_scores[0, 0, index[0], index[1]] += 0.25
    elif corruption == "message":
        index = trace.active_witness_mask[0, 0].nonzero()[0]
        trace.edge_messages[0, 0, index[0], index[1]] += 0.1
    elif corruption == "contribution":
        index = trace.active_witness_mask[0, 0].nonzero()[0]
        trace.edge_log_odds_contributions[0, 0, index[0], index[1]] += 0.1
    elif corruption == "capacity":
        trace.per_boundary_allocated_capacity[0, 0] += 0.1
    elif corruption == "strength":
        trace.relation_strength[0] = 1.1
    else:  # pragma: no cover
        raise AssertionError(corruption)

    with pytest.raises(AssertionError, match=message):
        _audit_conserved_witness_relation_trace(trace)


def _metrics() -> dict[str, float]:
    return {
        "acc": 85.0,
        "mae": 0.2,
        "qwk": 0.82,
        "ece": 0.03,
        "balanced_acc": 60.0,
        "macro_f1": 0.61,
    }


@pytest.mark.parametrize(
    ("role", "epoch", "phase", "result_key"),
    (
        ("deployable_selected", 4, "relation_only", "best_validation"),
        ("best_learned", 7, "relation_only", "best_learned_validation"),
    ),
)
def test_v8_checkpoint_roles_bind_to_distinct_result_records(
    role: str, epoch: int, phase: str, result_key: str
) -> None:
    state = {
        "schema": "origin-checkpoint-v8",
        "checkpoint_role": role,
        "epoch": epoch,
        "training_phase": phase,
    }
    result = {
        "best_epoch": 4,
        "best_validation": _metrics(),
        "best_checkpoint_is_hash_bound_v3_floor": False,
        "best_learned_epoch": 7,
        "best_learned_validation": {**_metrics(), "acc": 84.0},
        "selected_checkpoint_role": "deployable_selected",
        "deployable_best_checkpoint_sha256": "deployable-hash",
        "best_learned_checkpoint_role": "best_learned",
        "best_learned_checkpoint_sha256": "learned-hash",
    }
    resolved_role, resolved_epoch, metrics = _checkpoint_role_result_contract(
        state,
        result,
        checkpoint_sha256=(
            "learned-hash" if role == "best_learned" else "deployable-hash"
        ),
    )
    assert resolved_role == role
    assert resolved_epoch == epoch
    assert metrics == result[result_key]


def test_v8_checkpoint_floor_role_is_explicit_and_epoch_zero() -> None:
    state = {
        "schema": "origin-checkpoint-v8",
        "checkpoint_role": "hash_bound_v3_floor",
        "epoch": 0,
        "training_phase": "hash_bound_v3_floor",
    }
    result = {
        "best_epoch": 0,
        "best_validation": _metrics(),
        "best_checkpoint_is_hash_bound_v3_floor": True,
        "selected_checkpoint_role": "hash_bound_v3_floor",
        "deployable_best_checkpoint_sha256": "floor-hash",
    }
    assert _checkpoint_role_result_contract(
        state, result, checkpoint_sha256="floor-hash"
    )[:2] == (
        "hash_bound_v3_floor",
        0,
    )


@pytest.mark.parametrize("role", ("resume_state", "unknown", ""))
def test_v8_checkpoint_role_audit_rejects_non_auditable_roles(role: str) -> None:
    state = {
        "schema": "origin-checkpoint-v8",
        "checkpoint_role": role,
        "epoch": 3,
        "training_phase": "relation_only",
    }
    with pytest.raises(ValueError):
        _checkpoint_role_result_contract(state, {})


def test_v8_checkpoint_role_audit_rejects_mismatched_artifact_hash() -> None:
    state = {
        "schema": "origin-checkpoint-v8",
        "checkpoint_role": "best_learned",
        "epoch": 3,
        "training_phase": "relation_only",
    }
    result = {
        "best_learned_epoch": 3,
        "best_learned_validation": _metrics(),
        "best_learned_checkpoint_role": "best_learned",
        "best_learned_checkpoint_sha256": "expected",
    }
    with pytest.raises(ValueError, match="hash differs"):
        _checkpoint_role_result_contract(
            state, result, checkpoint_sha256="different"
        )
