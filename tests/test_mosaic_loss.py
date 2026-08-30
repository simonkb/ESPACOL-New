"""Focused deterministic tests for the MOSAIC loss functions."""

from __future__ import annotations

import math
import unittest

import torch

from losses.mosaic import (
    MosaicLoss,
    balanced_continuation_nll,
    build_at_risk_transition_weights,
    transition_outcome_counts,
    witness_js_stability,
)


class AtRiskWeightTests(unittest.TestCase):
    def test_transition_counts_exclude_samples_below_risk_set(self) -> None:
        labels = [0, 0, 1, 2, 4]
        expected = torch.tensor(
            [
                [2, 3],  # Y=0 versus Y>0
                [1, 2],  # among Y>=1: Y=1 versus Y>1
                [1, 1],  # among Y>=2: Y=2 versus Y>2
                [0, 1],  # among Y>=3: no grade 3, one grade 4
            ]
        )
        self.assertTrue(torch.equal(transition_outcome_counts(labels, 5), expected))

    def test_inverse_frequency_weights_are_normalised_and_hard_capped(self) -> None:
        # At k=0 this produces 1000 stop examples and one advance example.
        labels = [0] * 1000 + [4]
        weights, counts = build_at_risk_transition_weights(
            labels,
            5,
            method="inverse_frequency",
            max_weight=3.0,
            return_counts=True,
            dtype=torch.float64,
        )
        self.assertLessEqual(float(weights.max()), 3.0 + 1e-10)
        for k in range(4):
            n = counts[k].to(torch.float64)
            if n.sum() > 0:
                weighted_mean = (n * weights[k]).sum() / n.sum()
                self.assertAlmostEqual(float(weighted_mean), 1.0, places=10)

    def test_missing_outcomes_have_zero_weight_without_nan(self) -> None:
        weights, counts = build_at_risk_transition_weights(
            [0, 0, 0], 5, return_counts=True
        )
        self.assertTrue(torch.isfinite(weights).all())
        self.assertEqual(float(weights[0, 0]), 1.0)
        self.assertEqual(float(weights[0, 1]), 0.0)
        self.assertTrue(torch.equal(weights[1:], torch.zeros_like(weights[1:])))
        self.assertTrue(torch.equal(counts[1:], torch.zeros_like(counts[1:])))

    def test_effective_number_is_less_extreme_than_inverse_frequency(self) -> None:
        labels = [0] * 100 + [1] * 10 + [2] * 3 + [3] * 2 + [4]
        effective = build_at_risk_transition_weights(
            labels, 5, method="effective_num", beta=0.9, max_weight=None
        )
        inverse = build_at_risk_transition_weights(
            labels, 5, method="inverse_frequency", max_weight=None
        )
        self.assertLess(float(effective[0, 1] / effective[0, 0]), float(inverse[0, 1] / inverse[0, 0]))


class ContinuationLikelihoodTests(unittest.TestCase):
    def test_matches_manual_continuation_likelihood(self) -> None:
        transitions = torch.tensor(
            [
                [0.2, 0.3, 0.4, 0.5],
                [0.7, 0.6, 0.4, 0.3],
                [0.9, 0.8, 0.7, 0.6],
            ],
            dtype=torch.float64,
        )
        labels = torch.tensor([0, 1, 4])
        got = balanced_continuation_nll(transitions, labels)
        expected_per_sample = [
            -math.log(1.0 - 0.2),
            -math.log(0.7) - math.log(1.0 - 0.6),
            -sum(math.log(x) for x in (0.9, 0.8, 0.7, 0.6)),
        ]
        self.assertAlmostEqual(float(got), sum(expected_per_sample) / 3.0, places=12)

    def test_direct_log_stop_is_used_without_probability_round_trip(self) -> None:
        transitions = torch.ones(1, 4, requires_grad=True)
        log_stops = torch.tensor([[-180.0, -90.0, -30.0, -4.0]], requires_grad=True)
        loss = balanced_continuation_nll(
            transitions,
            torch.tensor([0]),
            log_stop_probabilities=log_stops,
        )
        self.assertAlmostEqual(float(loss.detach()), 180.0, places=5)
        loss.backward()
        self.assertEqual(float(log_stops.grad[0, 0]), -1.0)
        self.assertTrue(torch.isfinite(log_stops.grad).all())

    def test_negative_infinite_stop_outside_risk_set_does_not_create_nan(self) -> None:
        transitions = torch.tensor([[0.8, 0.7, 0.6, 0.5]], requires_grad=True)
        log_stops = torch.tensor(
            [[math.log(0.2), math.log(0.3), -torch.inf, -torch.inf]],
            requires_grad=True,
        )
        loss = balanced_continuation_nll(
            transitions,
            torch.tensor([1]),
            log_stop_probabilities=log_stops,
        )
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertTrue(torch.isfinite(transitions.grad).all())
        self.assertTrue(torch.isfinite(log_stops.grad).all())

    def test_weights_select_correct_boundary_outcome(self) -> None:
        c = torch.full((2, 2), 0.5, dtype=torch.float64)
        y = torch.tensor([0, 2])
        weights = torch.tensor([[2.0, 3.0], [5.0, 7.0]], dtype=torch.float64)
        per_sample = balanced_continuation_nll(c, y, weights, reduction="none")
        log2 = math.log(2.0)
        self.assertAlmostEqual(float(per_sample[0]), 2.0 * log2, places=12)
        self.assertAlmostEqual(float(per_sample[1]), (3.0 + 7.0) * log2, places=12)

    def test_below_risk_boundaries_have_zero_gradient(self) -> None:
        c = torch.tensor([[0.4, 0.3, 0.2, 0.1]], requires_grad=True)
        loss = balanced_continuation_nll(c, torch.tensor([1]))
        loss.backward()
        self.assertNotEqual(float(c.grad[0, 0]), 0.0)
        self.assertNotEqual(float(c.grad[0, 1]), 0.0)
        self.assertTrue(torch.equal(c.grad[0, 2:], torch.zeros_like(c.grad[0, 2:])))

    def test_gradients_are_finite_near_probability_boundaries(self) -> None:
        c = torch.tensor(
            [[1e-12, 1.0 - 1e-12, 0.5, 0.9]], dtype=torch.float64, requires_grad=True
        )
        loss = balanced_continuation_nll(c, torch.tensor([2]), eps=1e-9)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(torch.isfinite(c.grad).all())


class WitnessStabilityTests(unittest.TestCase):
    def test_js_is_zero_for_identical_states_and_symmetric(self) -> None:
        torch.manual_seed(7)
        p = torch.softmax(torch.randn(2, 5, 5), dim=-1)
        q = torch.softmax(torch.randn(2, 5, 5), dim=-1)
        self.assertLess(float(witness_js_stability(p, p)), 1e-7)
        self.assertAlmostEqual(
            float(witness_js_stability(p, q)),
            float(witness_js_stability(q, p)),
            places=7,
        )
        self.assertGreater(float(witness_js_stability(p, q)), 0.0)

    def test_mask_excludes_invalid_cells_and_empty_mask_is_differentiable(self) -> None:
        p = torch.tensor([[[1.0, 0.0], [1.0, 0.0]]], requires_grad=True)
        q = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]], requires_grad=True)
        mask = torch.tensor([[True, False]])
        self.assertAlmostEqual(
            float(witness_js_stability(p, q, mask).detach()), 0.0, places=7
        )

        empty = witness_js_stability(p, q, torch.zeros_like(mask))
        empty.backward()
        self.assertEqual(float(empty.detach()), 0.0)
        self.assertIsNotNone(p.grad)
        self.assertTrue(torch.isfinite(p.grad).all())


class CombinedMosaicLossTests(unittest.TestCase):
    def test_combined_objective_and_diagnostics(self) -> None:
        projected = torch.tensor(
            [[0.2, 0.3, 0.4, 0.5], [0.8, 0.7, 0.6, 0.5]], requires_grad=True
        )
        dense = torch.tensor(
            [[0.25, 0.35, 0.45, 0.55], [0.85, 0.75, 0.65, 0.55]],
            requires_grad=True,
        )
        labels = torch.tensor([0, 4])
        p = torch.softmax(torch.randn(2, 3, 5), dim=-1)
        q = torch.softmax(torch.randn(2, 3, 5), dim=-1)
        criterion = MosaicLoss(5, dense_weight=0.2, stability_weight=0.3)
        total, diagnostics = criterion(
            projected,
            labels,
            dense_transitions=dense,
            witness_states_a=p,
            witness_states_b=q,
        )
        expected = (
            balanced_continuation_nll(projected, labels)
            + 0.2 * balanced_continuation_nll(dense, labels)
            + 0.3 * witness_js_stability(p, q)
        )
        self.assertTrue(torch.allclose(total, expected))
        self.assertEqual(
            set(("loss_total", "loss_ccl", "loss_dense", "loss_stability"))
            - diagnostics.keys(),
            set(),
        )
        total.backward()
        self.assertTrue(torch.isfinite(projected.grad).all())
        self.assertTrue(torch.isfinite(dense.grad).all())

    def test_dense_and_stability_inputs_are_required_when_enabled(self) -> None:
        transitions = torch.full((2, 4), 0.5)
        labels = torch.tensor([0, 4])
        with self.assertRaisesRegex(ValueError, "dense_transitions"):
            MosaicLoss(5, dense_weight=0.1)(transitions, labels)

        criterion = MosaicLoss(5, dense_weight=0.0, stability_weight=0.1)
        with self.assertRaisesRegex(ValueError, "witness-state views"):
            criterion(transitions, labels)

    def test_fold_weights_are_checkpointed_as_buffer(self) -> None:
        criterion = MosaicLoss.from_training_labels(
            [0, 0, 1, 2, 3, 4], 5, dense_weight=0.0
        )
        self.assertIn("transition_weights", criterion.state_dict())
        clone = MosaicLoss(5, dense_weight=0.0)
        clone.load_state_dict(criterion.state_dict())
        self.assertTrue(torch.equal(clone.transition_weights, criterion.transition_weights))


if __name__ == "__main__":
    unittest.main()
