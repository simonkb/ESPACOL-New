"""Focused tests for MOSAIC's proof-only decision audit."""

import pytest
import torch

from models.mosaic_decoder import (
    cascade_class_probabilities,
    inverse_outcome_weighting,
    proof_only_decisions,
)


def _log_stop(transitions: torch.Tensor) -> torch.Tensor:
    return torch.log1p(-transitions)


def test_unit_weights_are_identity() -> None:
    transitions = torch.tensor(
        [[0.15, 0.45, 0.72, 0.31], [0.91, 0.67, 0.22, 0.08]],
        dtype=torch.float64,
    )
    weights = torch.ones(4, 2, dtype=torch.float64)
    corrected = inverse_outcome_weighting(transitions, _log_stop(transitions), weights)
    torch.testing.assert_close(corrected.transitions, transitions)
    torch.testing.assert_close(corrected.stop_probabilities, 1.0 - transitions)

    bundle = proof_only_decisions(transitions, _log_stop(transitions), weights)
    torch.testing.assert_close(
        bundle.deweighted_class_probabilities,
        bundle.raw_class_probabilities,
    )
    assert torch.equal(bundle.deweighted_argmax, bundle.raw_argmax)


def test_inverse_exactly_recovers_synthetically_weighted_posterior() -> None:
    natural = torch.tensor(
        [[0.05, 0.30, 0.65, 0.95], [0.80, 0.42, 0.17, 0.73]],
        dtype=torch.float64,
    )
    # Columns are [stop, advance], exactly as in the MOSAIC loss.
    weights = torch.tensor(
        [[0.7, 2.4], [1.8, 0.6], [0.9, 4.1], [3.0, 1.2]],
        dtype=torch.float64,
    )
    weighted = (
        weights[:, 1] * natural
        / (weights[:, 1] * natural + weights[:, 0] * (1.0 - natural))
    )

    recovered = inverse_outcome_weighting(weighted, _log_stop(weighted), weights)
    torch.testing.assert_close(recovered.transitions, natural, atol=1e-12, rtol=1e-12)
    torch.testing.assert_close(
        recovered.stop_probabilities,
        1.0 - natural,
        atol=1e-12,
        rtol=1e-12,
    )

    expected_cumulative, expected_classes = cascade_class_probabilities(
        natural,
        _log_stop(natural),
    )
    actual = proof_only_decisions(weighted, _log_stop(weighted), weights)
    torch.testing.assert_close(
        actual.deweighted_cumulative_probabilities,
        expected_cumulative,
        atol=1e-12,
        rtol=1e-12,
    )
    torch.testing.assert_close(
        actual.deweighted_class_probabilities,
        expected_classes,
        atol=1e-12,
        rtol=1e-12,
    )


def test_row_scaling_of_outcome_weights_cancels() -> None:
    transitions = torch.tensor([[0.13, 0.51, 0.89]], dtype=torch.float64)
    weights = torch.tensor(
        [[0.4, 2.0], [1.2, 0.7], [4.0, 1.1]], dtype=torch.float64
    )
    scales = torch.tensor([[0.01], [8.0], [1000.0]], dtype=torch.float64)
    first = inverse_outcome_weighting(transitions, _log_stop(transitions), weights)
    second = inverse_outcome_weighting(
        transitions,
        _log_stop(transitions),
        weights * scales,
    )
    torch.testing.assert_close(first.transitions, second.transitions)
    torch.testing.assert_close(first.log_stop_probabilities, second.log_stop_probabilities)


def test_endpoints_and_extreme_log_stops_stay_finite_and_normalized() -> None:
    transitions = torch.tensor(
        [[0.0, 1.0, 1.0], [1.0, 1.0, 0.0]], dtype=torch.float64
    )
    log_stops = torch.tensor(
        [[0.0, -1000.0, -torch.inf], [-1000.0, -torch.inf, 0.0]],
        dtype=torch.float64,
    )
    weights = torch.tensor(
        [[0.2, 7.0], [5.0, 0.4], [1.3, 2.1]], dtype=torch.float64
    )
    bundle = proof_only_decisions(transitions, log_stops, weights)

    for probabilities in (
        bundle.raw_class_probabilities,
        bundle.deweighted_class_probabilities,
    ):
        assert torch.isfinite(probabilities).all()
        assert torch.all(probabilities >= 0.0)
        torch.testing.assert_close(
            probabilities.sum(dim=-1),
            torch.ones(2, dtype=torch.float64),
            atol=1e-14,
            rtol=0.0,
        )
    assert torch.isfinite(bundle.deweighted_transitions).all()
    assert not torch.isnan(bundle.deweighted_log_stop_probabilities).any()
    assert not torch.isposinf(bundle.deweighted_log_stop_probabilities).any()
    assert int(bundle.raw_argmax[0]) == 0
    assert int(bundle.raw_argmax[1]) == 2


@pytest.mark.parametrize(
    "weights,exception",
    [
        (torch.ones(2, 2), ValueError),
        (torch.tensor([[1.0, 0.0], [1.0, 1.0], [1.0, 1.0]]), ValueError),
        (torch.tensor([[1.0, -1.0], [1.0, 1.0], [1.0, 1.0]]), ValueError),
        (torch.tensor([[1.0, float("nan")], [1.0, 1.0], [1.0, 1.0]]), ValueError),
        (torch.ones(3, 2, dtype=torch.long), TypeError),
    ],
)
def test_malformed_or_noninvertible_weights_are_rejected(
    weights: torch.Tensor,
    exception: type[Exception],
) -> None:
    transitions = torch.tensor([[0.2, 0.4, 0.6]])
    with pytest.raises(exception):
        inverse_outcome_weighting(transitions, _log_stop(transitions), weights)


def test_mean_round_and_class_argmax_are_distinct_decision_rules() -> None:
    # This cascade represents class probabilities [0.40, 0.35, 0.25].
    # Its posterior mean is 0.85 (rounds to grade 1), while the maximum-
    # probability class is grade 0.
    transitions = torch.tensor([[0.60, 5.0 / 12.0]], dtype=torch.float64)
    bundle = proof_only_decisions(
        transitions,
        _log_stop(transitions),
        torch.ones(2, 2, dtype=torch.float64),
    )
    torch.testing.assert_close(
        bundle.raw_class_probabilities,
        torch.tensor([[0.40, 0.35, 0.25]], dtype=torch.float64),
        atol=1e-14,
        rtol=1e-14,
    )
    torch.testing.assert_close(
        bundle.raw_expected_grade,
        torch.tensor([0.85], dtype=torch.float64),
    )
    assert int(bundle.raw_mean_round) == 1
    assert int(bundle.raw_argmax) == 0
    assert int(bundle.deweighted_argmax) == 0


def test_all_point_decisions_have_batch_shape_and_valid_grade_range() -> None:
    transitions = torch.tensor(
        [
            [0.05, 0.15, 0.25, 0.35],
            [0.80, 0.70, 0.60, 0.50],
            [1.00, 1.00, 1.00, 1.00],
        ]
    )
    log_stops = _log_stop(transitions)
    weights = torch.tensor(
        [[0.8, 1.4], [1.3, 0.7], [0.6, 2.0], [1.1, 0.9]]
    )
    bundle = proof_only_decisions(transitions, log_stops, weights)
    decisions = (
        bundle.raw_mean_round,
        bundle.raw_argmax,
        bundle.raw_posterior_median,
        bundle.deweighted_mean_round,
        bundle.deweighted_argmax,
        bundle.deweighted_posterior_median,
    )
    for decision in decisions:
        assert decision.shape == (3,)
        assert decision.dtype == torch.long
        assert torch.all((decision >= 0) & (decision <= 4))


def test_posterior_median_counts_continuations_with_upper_tie_convention() -> None:
    # Cumulative probabilities are respectively:
    # [0.49, ...]       -> median grade 0
    # [0.50, 0.25, ...] -> median grade 1 (advance on the exact tie)
    # [0.90, 0.72, 0.504, 0.2016] -> median grade 3
    transitions = torch.tensor(
        [
            [0.49, 0.90, 0.90, 0.90],
            [0.50, 0.50, 0.50, 0.50],
            [0.90, 0.80, 0.70, 0.40],
        ],
        dtype=torch.float64,
    )
    bundle = proof_only_decisions(
        transitions,
        _log_stop(transitions),
        torch.ones(4, 2, dtype=torch.float64),
    )
    expected = torch.tensor([0, 1, 3])
    assert torch.equal(bundle.raw_posterior_median, expected)
    assert torch.equal(bundle.deweighted_posterior_median, expected)
    assert torch.equal(bundle.deweighted_mean_round, bundle.raw_mean_round)
