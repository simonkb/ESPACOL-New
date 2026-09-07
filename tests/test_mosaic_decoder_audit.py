"""Focused regression tests for the validation-only decoder audit."""

import json
import math

import pytest
import torch

from models.mosaic_decoder import proof_only_decisions
from tools.audit_mosaic_decoders import (
    _alpha_diagnostics,
    _batch_witness_concentration,
    _batch_regional_lme_attenuation,
    _binary_auroc,
    _checkpoint_decision_rule,
    _declared_decision_replay_mismatches,
    _fixed_region_valid_counts,
    _log_probability_invariants,
    _merge_witness_concentration_batches,
    _proof_diagnostics,
    _proof_only_decisions_from_log_pairs,
    _regional_lme_attenuation_summary,
    _transition_path_diagnostics,
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


def test_model_replay_uses_declared_checkpoint_rule_not_historical_default() -> None:
    specs = {
        "rounded_expected": (torch.tensor([0, 1]), torch.ones(2, 1)),
        "posterior_median": (torch.tensor([1, 1]), torch.ones(2, 1)),
    }
    actual = torch.tensor([1, 1])
    assert (
        _declared_decision_replay_mismatches(
            actual, specs, "posterior_median"
        )
        == 0
    )
    assert (
        _declared_decision_replay_mismatches(
            actual, specs, "rounded_expected"
        )
        == 1
    )


def test_log_probability_invariants_distinguish_underflow_and_exact_zero() -> None:
    probabilities = torch.tensor([0.5, 0.0, 0.0])
    logs = torch.tensor([math.log(0.5), -1000.0, -torch.inf])
    audit = _log_probability_invariants(probabilities, logs)

    assert audit["all_valid"] is True
    assert audit["probability_zero_finite_log_count_underflow_preserved"] == 1
    assert (
        audit["probability_zero_negative_infinite_log_count_exact_endpoint"]
        == 1
    )
    assert audit["probability_positive_negative_infinite_log_count_violation"] == 0


def test_exact_log_decoder_preserves_conditional_mass_after_probability_underflow() -> None:
    # Both unnormalized outcomes are below FP32 probability range. Their log
    # ratio is nevertheless ordinary: advance is e times as likely as stop.
    log_advance = torch.tensor([[-1000.0]])
    log_stop = torch.tensor([[-1001.0]])
    historical_probability = log_advance.exp()
    weights = torch.ones(1, 2)

    historical = proof_only_decisions(
        historical_probability,
        log_stop,
        weights,
    )
    exact = _proof_only_decisions_from_log_pairs(
        log_advance,
        log_stop,
        weights,
    )

    assert float(historical.raw_cumulative_probabilities) == 0.0
    assert float(exact.raw_cumulative_probabilities) == pytest.approx(
        0.7310585786
    )
    assert int(historical.raw_argmax) == 0
    assert int(exact.raw_argmax) == 1
    assert int(exact.raw_posterior_median) == 1


def test_exact_log_decoder_preserves_structural_zero_endpoint() -> None:
    exact = _proof_only_decisions_from_log_pairs(
        torch.tensor([[-torch.inf, 0.0]], dtype=torch.float64),
        torch.tensor([[0.0, -torch.inf]], dtype=torch.float64),
        torch.ones(2, 2, dtype=torch.float64),
    )

    torch.testing.assert_close(
        exact.raw_class_probabilities,
        torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float64),
    )
    assert int(exact.raw_mean_round) == 0


def test_binary_auroc_is_exact_with_ties_and_reports_missing_outcome() -> None:
    # Positives earn 3.5 of the four pairwise credits against the two
    # negatives (including one half-credit tie), so AUROC = 7/8.
    scores = torch.tensor([0.1, 0.5, 0.5, 0.9])
    targets = torch.tensor([0, 0, 1, 1])
    assert _binary_auroc(scores, targets) == pytest.approx(0.875)
    assert _binary_auroc(scores, torch.ones(4, dtype=torch.bool)) is None


def test_boundary_auroc_uses_only_the_continuation_risk_set() -> None:
    # At boundary 2 the two Y<2 samples have deliberately extreme scores that
    # would change an all-sample AUROC.  They must not enter the clinical
    # stop-vs-advance comparison (Y=2 versus Y>2).
    labels = torch.tensor([0, 1, 2, 3, 4])
    projected = torch.tensor(
        [
            [0.1, 0.1, 1.0, 0.1],
            [0.2, 0.2, 0.9, 0.2],
            [0.3, 0.3, 0.1, 0.3],
            [0.4, 0.4, 0.7, 0.4],
            [0.5, 0.5, 0.8, 0.5],
        ]
    )
    dense = projected + torch.tensor(
        [[0.0, 0.0, -0.1, 0.0]] * 5
    )

    audit = _transition_path_diagnostics(projected, dense, labels)
    boundary = audit["boundaries"][2]

    assert boundary["risk_set"] == "Y>=2"
    assert boundary["at_risk_count"] == 3
    assert boundary["advance_count"] == 2
    assert boundary["stop_count"] == 1
    assert boundary["below_risk_count"] == 2
    assert boundary["projected_auroc"] == pytest.approx(1.0)
    assert boundary["dense_pre_projection_auroc"] == pytest.approx(1.0)
    assert boundary["dense_minus_projected"]["at_risk"]["count"] == 3
    assert boundary["dense_minus_projected"]["overall"]["count"] == 5


def test_boundary_auroc_is_null_when_risk_set_has_one_outcome() -> None:
    labels = torch.tensor([0, 1, 1])
    transitions = torch.tensor([[0.1, 0.2], [0.7, 0.3], [0.8, 0.4]])
    audit = _transition_path_diagnostics(transitions, transitions, labels)

    assert audit["boundaries"][1]["at_risk_count"] == 2
    assert audit["boundaries"][1]["advance_count"] == 0
    assert audit["boundaries"][1]["projected_auroc"] is None
    assert audit["boundaries"][1]["dense_pre_projection_auroc"] is None


def test_boundary_auroc_ranks_exact_log_odds_after_probability_underflow() -> None:
    labels = torch.tensor([0, 1])
    underflowed = torch.zeros(2, 1)
    log_advance = torch.tensor([[-1000.0], [-900.0]])
    log_stop = torch.zeros(2, 1)
    audit = _transition_path_diagnostics(
        underflowed,
        underflowed,
        labels,
        projected_log_transitions=log_advance,
        projected_log_stops=log_stop,
        dense_log_transitions=log_advance,
        dense_log_stops=log_stop,
    )
    boundary = audit["boundaries"][0]

    assert boundary["auroc_score_definition"] == (
        "exact_log_advance_minus_log_stop"
    )
    assert boundary["projected_auroc"] == pytest.approx(1.0)
    assert boundary["dense_pre_projection_auroc"] == pytest.approx(1.0)


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


def test_fixed_region_counts_and_lme_attenuation_are_exact() -> None:
    source_valid = torch.tensor(
        [[
            True, True, False, False,
            True, False, True, True,
            False, False, True, True,
            True, True, True, False,
        ]]
    )
    counts = _fixed_region_valid_counts(source_valid, (4, 4), (2, 2))
    torch.testing.assert_close(counts, torch.tensor([[3, 2, 2, 3]]))

    source_logits = torch.tensor([[[-2.0], [0.0], [2.0], [1.0]]])
    temperature = 0.5
    region_zero = temperature * (
        torch.logsumexp(source_logits[:, :2] / temperature, dim=1)
        - math.log(2.0)
    )
    region_one = temperature * (
        torch.logsumexp(source_logits[:, 2:] / temperature, dim=1)
        - math.log(2.0)
    )
    regional_logits = torch.stack((region_zero, region_one), dim=1)
    source_log_witness = torch.nn.functional.logsigmoid(source_logits)
    source_log_nonwitness = torch.nn.functional.logsigmoid(-source_logits)
    regional_log_witness = torch.nn.functional.logsigmoid(regional_logits)
    regional_log_nonwitness = torch.nn.functional.logsigmoid(-regional_logits)

    attenuation = _batch_regional_lme_attenuation(
        source_log_witness_probabilities=source_log_witness,
        source_log_nonwitness_probabilities=source_log_nonwitness,
        regional_log_witness_probabilities=regional_log_witness,
        regional_log_nonwitness_probabilities=regional_log_nonwitness,
        peak_source_indices=torch.tensor([[[1], [2]]]),
        regional_valid_mask=torch.tensor([[True, True]]),
        valid_source_counts=torch.tensor([[2, 2]]),
    )
    expected_peak_logits = torch.tensor([[[0.0], [2.0]]])
    torch.testing.assert_close(
        attenuation["source_peak_logit"], expected_peak_logits
    )
    torch.testing.assert_close(
        attenuation["regional_lme_logit"], regional_logits
    )
    assert bool((attenuation["logit_attenuation"] >= 0.0).all())

    summary = _regional_lme_attenuation_summary(
        attenuation,
        temperature=temperature,
        source_lattice_size=(1, 4),
        regional_block_size=(1, 2),
        replay_max_probability_error=0.0,
        replay_peak_index_mismatches=0,
    )
    json.dumps(summary, allow_nan=False)
    assert summary["available"] is True
    assert summary["regional_event_count"] == 2
    assert summary["boundaries"][0]["valid_region_events"] == 2
    assert (
        summary["boundaries"][0]["lme_above_source_max_count_tolerance_1e-6"]
        == 0
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
