"""Focused regression tests for the validation-only decoder audit."""

import json
import math

import pytest
import torch

from tools.audit_mosaic_decoders import (
    _alpha_diagnostics,
    _batch_witness_concentration,
    _checkpoint_decision_rule,
    _merge_witness_concentration_batches,
    _proof_diagnostics,
)


def test_legacy_checkpoint_uses_historical_rounded_expected_rule() -> None:
    assert _checkpoint_decision_rule({}) == "rounded_expected"


def test_explicit_checkpoint_decision_rule_is_preserved() -> None:
    assert (
        _checkpoint_decision_rule({"decision_rule": "deweighted_class_map"})
        == "deweighted_class_map"
    )


def test_unknown_checkpoint_decision_rule_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown MOSAIC decision rule"):
        _checkpoint_decision_rule({"decision_rule": "validation_tuned"})


def test_witness_concentration_reduces_dense_ledger_and_clamps_top_k() -> None:
    witnesses = torch.tensor(
        [
            [[1.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
            [[0.5, 0.6], [0.5, 0.3], [0.0, 0.1]],
            [[0.0, 0.2], [0.0, 0.2], [0.0, 0.2]],
        ]
    )

    concentration = _batch_witness_concentration(witnesses)

    assert concentration["spatial_cell_count"] == 3
    assert concentration["clamped_top_k"] == {
        1: 1,
        4: 3,
        16: 3,
        32: 3,
        64: 3,
    }
    torch.testing.assert_close(
        concentration["witness_count"],
        torch.tensor([[1.0, 0.0], [1.0, 1.0], [0.0, 0.6]]),
    )
    torch.testing.assert_close(
        concentration["effective_support"],
        torch.tensor([[1.0, 0.0], [2.0, 1.0 / 0.46], [0.0, 3.0]]),
    )
    torch.testing.assert_close(
        concentration["max_witness_probability"],
        torch.tensor([[1.0, 0.0], [0.5, 0.6], [0.0, 0.2]]),
    )
    torch.testing.assert_close(
        concentration["top_k_witness_mass_fraction"][1],
        torch.tensor([[1.0, 0.0], [0.5, 0.6], [0.0, 1.0 / 3.0]]),
    )
    torch.testing.assert_close(
        concentration["top_k_witness_mass_fraction"][64],
        torch.tensor([[1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]),
    )


def test_witness_concentration_batches_merge_without_dense_probabilities() -> None:
    witnesses = torch.tensor(
        [
            [[0.7], [0.2], [0.1]],
            [[0.4], [0.3], [0.2]],
            [[0.0], [0.0], [0.0]],
        ]
    )
    whole = _batch_witness_concentration(witnesses)
    merged = _merge_witness_concentration_batches(
        [
            _batch_witness_concentration(witnesses[:1]),
            _batch_witness_concentration(witnesses[1:]),
        ]
    )

    assert "witness_probabilities" not in merged
    for name in (
        "witness_count",
        "effective_support",
        "max_witness_probability",
    ):
        torch.testing.assert_close(merged[name], whole[name])
    for requested_k in (1, 4, 16, 32, 64):
        torch.testing.assert_close(
            merged["top_k_witness_mass_fraction"][requested_k],
            whole["top_k_witness_mass_fraction"][requested_k],
        )


def test_proof_diagnostics_condition_on_boundary_risk_and_summarize_alpha() -> None:
    labels = torch.tensor([0, 1, 2, 3])
    proof_sizes = torch.tensor([[0, 10], [1, 20], [2, 30], [3, 40]])
    transitions = torch.tensor(
        [[0.0, 0.1], [0.2, 0.0], [0.4, 0.5], [0.7, 0.8]]
    )
    witnesses = torch.tensor(
        [
            [[0.0, 1.0], [0.0, 0.0], [0.0, 0.0]],
            [[0.5, 0.5], [0.5, 0.5], [0.0, 0.0]],
            [[1.0, 0.4], [1.0, 0.3], [1.0, 0.3]],
            [[1.0, 0.2], [1.0, 0.2], [1.0, 0.2]],
        ]
    )
    concentration = _batch_witness_concentration(witnesses)
    retained_overflow = torch.zeros(4, 2)
    alpha = torch.tensor([[0.25, 0.75], [1.0, 0.0]])

    diagnostics = _proof_diagnostics(
        proof_sizes,
        transitions,
        labels,
        concentration,
        retained_overflow,
        alpha,
    )
    json.dumps(diagnostics, allow_nan=False)

    assert diagnostics["witness_spatial_cell_count"] == 3
    boundary = diagnostics["boundaries"][1]
    assert boundary["overall_count"] == 4
    assert boundary["at_risk_count"] == 3
    assert boundary["advance_count"] == 2
    assert boundary["stop_count"] == 1
    assert boundary["below_risk_count"] == 1
    groups = boundary["conditional_concentration"]
    assert {name: group["sample_count"] for name, group in groups.items()} == {
        "overall": 4,
        "at_risk": 3,
        "advance": 2,
        "stop": 1,
        "below_risk": 1,
    }
    assert groups["at_risk"]["proof_size"] == {
        "mean": 30.0,
        "median": 30.0,
        "p90": 38.0,
        "zero_rate": 0.0,
    }
    assert groups["below_risk"]["witness_count"] == {
        "mean": 1.0,
        "median": 1.0,
        "p90": 1.0,
    }
    assert groups["overall"]["top_k_witness_mass_fraction"]["64"][
        "clamped_k"
    ] == 3
    assert boundary["alpha"] == {
        "expected_threshold": 1.0,
        "mode_threshold": 1,
        "entropy_nats": 0.0,
        "weights": [1.0, 0.0],
    }

    empty_group = diagnostics["boundaries"][0]["conditional_concentration"][
        "below_risk"
    ]
    assert empty_group["sample_count"] == 0
    assert empty_group["proof_size"] == {
        "mean": None,
        "median": None,
        "p90": None,
        "zero_rate": None,
    }


def test_alpha_diagnostics_use_one_based_thresholds_and_normalized_weights() -> None:
    diagnostics = _alpha_diagnostics(torch.tensor([1.0, 2.0, 1.0]))

    assert diagnostics["expected_threshold"] == 2.0
    assert diagnostics["mode_threshold"] == 2
    assert diagnostics["entropy_nats"] == pytest.approx(
        -(0.25 * math.log(0.25) + 0.5 * math.log(0.5) + 0.25 * math.log(0.25))
    )
    assert diagnostics["weights"] == [0.25, 0.5, 0.25]
