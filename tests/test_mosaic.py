"""Focused mathematical tests for :mod:`models.mosaic`.

These tests intentionally use tiny witness sets so exact Bernoulli and subset
enumeration can serve as an independent oracle for the optimized code.
"""

import itertools
import math

import torch

from losses.mosaic import MosaicLoss
from models.mosaic import (
    DualProofProjection,
    LocalOrdinalStateHead,
    MOSAICOrdinalCore,
    OrdinalCardinalityCircuit,
    TruncatedPoissonBinomial,
    continuation_probabilities,
    fixed_proof_pivotality,
    nested_witness_probabilities,
)


def _brute_distribution(probabilities: torch.Tensor, max_count: int) -> torch.Tensor:
    result = torch.zeros(max_count + 1, dtype=torch.float64)
    values = probabilities.detach().double().tolist()
    for outcome in itertools.product((0, 1), repeat=len(values)):
        probability = 1.0
        for event, event_probability in zip(outcome, values):
            probability *= event_probability if event else 1.0 - event_probability
        result[min(sum(outcome), max_count)] += probability
    return result


def _score_subset(
    probabilities: torch.Tensor,
    selected: torch.Tensor,
    alpha: torch.Tensor,
    count: TruncatedPoissonBinomial,
) -> torch.Tensor:
    distribution = count(probabilities * selected.float(), implementation="serial")
    tails = count.tails(distribution)
    return (tails * alpha).sum()


def test_nested_local_witnesses_are_structural_and_masked() -> None:
    torch.manual_seed(1)
    logits = torch.randn(2, 7, 5, dtype=torch.float64)
    valid = torch.tensor(
        [[True, True, False, True, True, False, True], [False] * 7]
    )
    evidence = nested_witness_probabilities(logits, valid)
    witnesses = evidence.witness_probabilities

    assert witnesses.dtype == torch.float32
    assert torch.all(witnesses[..., :-1] >= witnesses[..., 1:])
    assert torch.equal(witnesses[~valid], torch.zeros_like(witnesses[~valid]))
    expected_normal = torch.zeros_like(evidence.state_probabilities[~valid])
    expected_normal[..., 0] = 1.0
    assert torch.equal(evidence.state_probabilities[~valid], expected_normal)

    extreme = nested_witness_probabilities(torch.randn(4, 25000, 5) * 80.0)
    assert torch.all((extreme.witness_probabilities >= 0.0) & (extreme.witness_probabilities <= 1.0))
    assert torch.all(
        extreme.witness_probabilities[..., :-1]
        >= extreme.witness_probabilities[..., 1:]
    )


def test_local_head_bias_matches_requested_initial_abnormal_count() -> None:
    head = LocalOrdinalStateHead(
        input_dim=3,
        num_classes=5,
        expected_num_cells=1000,
        initial_abnormal_count=0.4,
    )
    with torch.no_grad():
        _, evidence = head(torch.randn(2, 1000, 3) * 3.0)
    assert torch.equal(head.linear.weight, torch.zeros_like(head.linear.weight))
    expected_count = evidence.witness_probabilities[..., 0].sum(dim=1)
    torch.testing.assert_close(
        expected_count, torch.full_like(expected_count, 0.4), atol=2e-4, rtol=2e-4
    )


def test_serial_distribution_matches_brute_force_and_preserves_overflow() -> None:
    probabilities = torch.tensor([0.05, 0.25, 0.60, 0.95, 1.0, 0.0])
    for max_count in (1, 2, 4):
        layer = TruncatedPoissonBinomial(max_count, implementation="serial")
        actual = layer(probabilities)
        expected = _brute_distribution(probabilities, max_count).float()
        torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-6)
        torch.testing.assert_close(actual.sum(), torch.tensor(1.0), atol=2e-6, rtol=0)


def test_block_tree_matches_serial_for_awkward_shapes_and_masks() -> None:
    torch.manual_seed(2)
    probabilities = torch.rand(2, 3, 23)
    valid = torch.rand(2, 3, 23) > 0.25
    layer = TruncatedPoissonBinomial(
        max_count=7, implementation="block_tree", block_size=5
    )
    tree = layer(probabilities, valid)
    serial = layer(probabilities, valid, implementation="serial")
    torch.testing.assert_close(tree, serial, atol=3e-6, rtol=3e-6)
    torch.testing.assert_close(
        tree.sum(dim=-1), torch.ones_like(tree[..., 0]), atol=3e-6, rtol=0
    )


def test_full_lattice_sparse_and_saturated_probability_invariants() -> None:
    """Regression for mass drift that only appears at the intended P=12,544."""

    cells = 112 * 112
    layer = TruncatedPoissonBinomial(
        max_count=32, implementation="block_tree", block_size=64
    )
    for probability in (0.1 / cells, 0.5 / cells, 1.0 / cells, 12.5 / cells, 0.01):
        distribution = layer(torch.full((1, cells), probability))
        tails = layer.tails(distribution)
        assert torch.isfinite(distribution).all()
        assert torch.all(distribution >= 0.0)
        torch.testing.assert_close(
            distribution.sum(dim=-1), torch.ones(1), atol=2e-6, rtol=0
        )
        assert torch.all((tails >= 0.0) & (tails <= 1.0))


def test_saturated_grade_zero_keeps_nonzero_recovery_gradient() -> None:
    """The directly evaluated lower tail must survive c rounding to one."""

    cells = 112 * 112
    state_logits = torch.tensor([0.99, 1e-12, 1e-12, 1e-12, 0.01]).log()
    logits = (
        state_logits.view(1, 1, 5)
        .repeat(1, cells, 1)
        .clone()
        .requires_grad_()
    )
    core = MOSAICOrdinalCore(
        num_classes=5,
        max_count=32,
        implementation="block_tree",
        block_size=64,
    )
    output = core(logits, project=False)
    assert torch.all((output.class_probabilities >= 0.0) & (output.class_probabilities <= 1.0))
    torch.testing.assert_close(
        output.class_probabilities.sum(dim=-1), torch.ones(1), atol=2e-6, rtol=0
    )
    assert float(output.stop_probabilities[0, 0].detach()) > 0.0

    loss, _ = MosaicLoss(5, dense_weight=0.0)(
        output.transitions,
        torch.tensor([0]),
        projected_stop_probabilities=output.stop_probabilities,
    )
    loss.backward()
    assert torch.isfinite(loss)
    assert logits.grad is not None and float(logits.grad.abs().sum()) > 0.0
    assert core.circuit.alpha_logits.grad is not None
    assert float(core.circuit.alpha_logits.grad.abs().sum()) > 0.0


def test_scaled_log_lower_tail_matches_float64_brute_force() -> None:
    probabilities = torch.tensor([[0.07, 0.2, 0.41, 0.63]])
    count = TruncatedPoissonBinomial(3, implementation="serial")
    log_conditional, log_survival = count.scaled_log_lower_tail(probabilities)
    brute = _brute_distribution(probabilities[0].double(), 3)
    low_survival = brute[:3].sum()
    torch.testing.assert_close(
        log_survival.double(), low_survival.log().view(1), atol=2e-6, rtol=2e-6
    )
    torch.testing.assert_close(
        log_conditional.double(),
        (brute[:3] / low_survival).log().view(1, 3),
        atol=2e-6,
        rtol=2e-6,
    )


def test_scaled_log_lower_tail_block_tree_matches_serial() -> None:
    torch.manual_seed(91)
    probabilities = torch.rand(2, 3, 97) * 0.2
    count = TruncatedPoissonBinomial(12, implementation="block_tree", block_size=16)
    serial_conditional, serial_survival = count.scaled_log_lower_tail(
        probabilities, implementation="serial"
    )
    tree_conditional, tree_survival = count.scaled_log_lower_tail(
        probabilities, implementation="block_tree"
    )
    torch.testing.assert_close(tree_survival, serial_survival, atol=2e-5, rtol=2e-5)
    torch.testing.assert_close(
        tree_conditional, serial_conditional, atol=3e-5, rtol=3e-5
    )


def test_scaled_log_lower_tail_has_endpoint_safe_semantics_and_gradients() -> None:
    probabilities = torch.tensor([[0.0, 0.2, 1.0]], requires_grad=True)
    count = TruncatedPoissonBinomial(3, implementation="serial")
    log_conditional, log_survival = count.scaled_log_lower_tail(probabilities)
    # With one deterministic event and one Bernoulli(.2), C is 1 or 2 and
    # therefore always below R=3.
    torch.testing.assert_close(log_survival, torch.zeros_like(log_survival))
    objective = log_survival.sum() + log_conditional.sum()
    objective.backward()
    assert probabilities.grad is not None
    assert torch.isfinite(probabilities.grad).all()
    assert float(probabilities.grad[0, 0]) == 0.0
    assert float(probabilities.grad[0, 2]) == 0.0


def test_full_lattice_diffuse_evidence_has_finite_log_stop_and_gradients() -> None:
    cells = 112 * 112
    abnormal_probability = 0.015
    state = torch.tensor(
        [1.0 - abnormal_probability] + [abnormal_probability / 4.0] * 4
    )
    logits = state.log().view(1, 1, 5).repeat(1, cells, 1).clone().requires_grad_()
    core = MOSAICOrdinalCore(
        num_classes=5,
        max_count=32,
        implementation="block_tree",
        block_size=64,
    )
    output = core(logits, project=False)
    assert torch.isfinite(output.log_stop_probabilities).all()
    loss, _ = MosaicLoss(5, dense_weight=0.1)(
        output.transitions,
        torch.tensor([0]),
        projected_log_stop_probabilities=output.log_stop_probabilities,
        dense_transitions=output.dense_transitions,
        dense_log_stop_probabilities=output.dense_log_stop_probabilities,
    )
    loss.backward()
    assert torch.isfinite(loss)
    assert logits.grad is not None and torch.isfinite(logits.grad).all()
    assert float(logits.grad.abs().sum()) > 0.0
    assert core.circuit.alpha_logits.grad is not None
    assert torch.isfinite(core.circuit.alpha_logits.grad).all()
    assert float(core.circuit.alpha_logits.grad.abs().sum()) > 0.0


def test_log_stop_preserves_first_count_under_extreme_diffusion() -> None:
    cells = 112 * 112
    probability = torch.full((1, cells), 0.03, requires_grad=True)
    count = TruncatedPoissonBinomial(32, implementation="block_tree", block_size=64)
    log_conditional, log_survival = count.scaled_log_lower_tail(probability)
    alpha = torch.zeros(1, 32)
    alpha[0, 0] = 1.0
    log_stop = OrdinalCardinalityCircuit.score_log_stops(
        log_conditional.unsqueeze(1), log_survival.unsqueeze(1), alpha
    )
    expected = cells * math.log1p(-0.03)
    torch.testing.assert_close(
        log_stop.squeeze(), torch.tensor(expected), atol=2e-3, rtol=2e-5
    )
    (-log_stop.sum()).backward()
    assert probability.grad is not None and torch.isfinite(probability.grad).all()
    assert float(probability.grad.abs().sum()) > 0.0


def test_public_log_stop_mixture_accepts_exact_zero_weights() -> None:
    probabilities = torch.tensor(
        [[0.1, 0.2, 0.3]], dtype=torch.float32, requires_grad=True
    )
    count = TruncatedPoissonBinomial(3, implementation="serial")
    log_conditional, log_survival = count.scaled_log_lower_tail(probabilities)
    alpha = torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float32)
    log_stop = OrdinalCardinalityCircuit.score_log_stops(
        log_conditional.unsqueeze(1), log_survival.unsqueeze(1), alpha
    )
    expected = sum(math.log1p(-float(p)) for p in probabilities.detach()[0])
    torch.testing.assert_close(
        log_stop.squeeze(), torch.tensor(expected), atol=1e-6, rtol=1e-6
    )
    (-log_stop.sum()).backward()
    assert probabilities.grad is not None
    assert torch.isfinite(probabilities.grad).all()


def test_extreme_alpha_logits_keep_log_stop_gradients_finite() -> None:
    witnesses = torch.tensor(
        [[[0.1], [0.2], [0.3]]], dtype=torch.float32, requires_grad=True
    )
    circuit = OrdinalCardinalityCircuit(
        num_boundaries=1,
        max_count=3,
        implementation="serial",
        alpha_init_count=None,
    )
    with torch.no_grad():
        circuit.alpha_logits.copy_(torch.tensor([[1000.0, -1000.0, -1000.0]]))
    result = circuit(witnesses)
    loss = -result.log_stop_probabilities.sum()
    loss.backward()
    assert torch.isfinite(loss)
    assert witnesses.grad is not None and torch.isfinite(witnesses.grad).all()
    assert float(witnesses.grad.abs().sum()) > 0.0
    assert circuit.alpha_logits.grad is not None
    assert torch.isfinite(circuit.alpha_logits.grad).all()


def test_cardinality_score_bounds_zero_and_monotonicity() -> None:
    torch.manual_seed(3)
    circuit = OrdinalCardinalityCircuit(
        num_boundaries=2,
        max_count=5,
        implementation="serial",
        alpha_init_count=None,
    )
    low = torch.rand(3, 9, 2) * 0.5
    high = torch.clamp(low + torch.rand_like(low) * 0.5, max=1.0)
    zero = circuit(torch.zeros_like(low)).transitions
    low_score = circuit(low).transitions
    high_score = circuit(high).transitions

    torch.testing.assert_close(zero, torch.zeros_like(zero), atol=1e-7, rtol=0)
    assert torch.all((low_score >= 0.0) & (low_score <= 1.0 + 1e-6))
    assert torch.all(high_score + 1e-6 >= low_score)


def test_dual_proof_matches_exhaustive_minimum_subset() -> None:
    probabilities = torch.tensor([0.82, 0.61, 0.35, 0.18, 0.07])
    witnesses = probabilities[None, :, None]
    alpha = torch.softmax(torch.tensor([[1.4, 0.8, -0.2]]), dim=-1)
    epsilon = 0.04
    suppression = 0.55
    projector = DualProofProjection(
        num_boundaries=1,
        max_count=3,
        sufficiency_tolerance=epsilon,
        complement_suppression=suppression,
        implementation="block_tree",
        block_size=2,
    )
    proof = projector(witnesses, alpha)

    count = TruncatedPoissonBinomial(3, implementation="serial")
    all_selected = torch.ones_like(probabilities, dtype=torch.bool)
    dense = _score_subset(probabilities, all_selected, alpha[0], count)
    target = max(float(dense - epsilon), 0.0)
    brute_minimum = len(probabilities)
    for subset_bits in itertools.product((False, True), repeat=len(probabilities)):
        selected = torch.tensor(subset_bits)
        retained = _score_subset(probabilities, selected, alpha[0], count)
        complement = _score_subset(probabilities, ~selected, alpha[0], count)
        if (
            float(retained) + 1e-7 >= target
            and float(dense - complement) + 1e-7 >= suppression * target
        ):
            brute_minimum = min(brute_minimum, int(selected.sum()))

    assert int(proof.proof_size.item()) == brute_minimum
    selected = proof.selected_mask[0, :, 0]
    # Canonical proof is the top-m prefix.
    order = torch.argsort(probabilities, descending=True)
    expected = torch.zeros_like(selected).scatter(
        0, order[:brute_minimum], torch.ones(brute_minimum, dtype=torch.bool)
    )
    assert torch.equal(selected, expected)
    assert float(proof.sufficiency_gap) <= epsilon + 2e-6
    assert float(proof.complement_drop) + 2e-6 >= suppression * target


def test_negligible_transition_selects_empty_proof_and_full_is_always_feasible() -> None:
    alpha = torch.tensor([[1.0, 0.0]])
    projector = DualProofProjection(
        num_boundaries=1,
        max_count=2,
        sufficiency_tolerance=0.02,
        complement_suppression=1.0,
        implementation="serial",
        block_size=3,
    )
    empty = projector(torch.zeros(1, 4, 1), alpha)
    assert int(empty.proof_size.item()) == 0
    assert not empty.selected_mask.any()

    nonempty = projector(torch.tensor([[[0.9], [0.8], [0.4], [0.2]]]), alpha)
    assert 0 <= int(nonempty.proof_size.item()) <= 4
    assert float(nonempty.sufficiency_gap) <= 0.02 + 2e-6


def test_empty_proof_endpoint_is_exact_under_merge_order_roundoff() -> None:
    """Regression for an independently recomputed suffix rejecting m=0."""

    probabilities = torch.tensor([0.2058061361, 0.2031814456, 0.1970242858])
    alpha = torch.tensor(
        [[0.09631055, 0.17498745, 0.37786454, 0.08739246, 0.26344502]]
    )
    projector = DualProofProjection(
        num_boundaries=1,
        max_count=5,
        sufficiency_tolerance=0.1876596446,
        complement_suppression=0.2870812860,
        implementation="block_tree",
        block_size=1,
    )
    proof = projector(probabilities[None, :, None], alpha)
    assert float(proof.dense_transition) <= projector.sufficiency_tolerance
    assert int(proof.proof_size) == 0
    assert not proof.selected_mask.any()


def test_tied_witnesses_use_original_index_as_canonical_tie_break() -> None:
    witnesses = torch.tensor([[[0.6], [0.6], [0.2]]])
    projector = DualProofProjection(
        num_boundaries=1,
        max_count=1,
        sufficiency_tolerance=0.4,
        complement_suppression=0.0,
        implementation="block_tree",
        block_size=2,
    )
    proof = projector(witnesses, torch.ones(1, 1))
    assert int(proof.proof_size) == 1
    assert proof.selected_mask[0, 0, 0]
    assert not proof.selected_mask[0, 1, 0]


def test_replay_and_analytic_pivotality_equal_direct_intervention() -> None:
    witnesses = torch.tensor(
        [[[0.86], [0.72], [0.41], [0.17], [0.03]]], requires_grad=True
    )
    alpha = torch.softmax(torch.tensor([[1.0, 0.2, -0.4]]), dim=-1)
    projector = DualProofProjection(
        num_boundaries=1,
        max_count=3,
        sufficiency_tolerance=0.01,
        complement_suppression=0.6,
        implementation="block_tree",
        block_size=2,
    )
    proof = projector(witnesses, alpha)
    delta = fixed_proof_pivotality(witnesses, proof, alpha, max_count=3)
    count = TruncatedPoissonBinomial(3, implementation="serial")
    selected = proof.selected_mask[0, :, 0]
    retained = witnesses[0, :, 0] * selected.float()
    base = _score_subset(
        witnesses[0, :, 0], selected, alpha[0], count
    )
    torch.testing.assert_close(
        base, proof.projected_transition.squeeze(), atol=2e-6, rtol=2e-6
    )
    for index in range(witnesses.shape[1]):
        if selected[index]:
            intervened = retained.clone()
            intervened[index] = 0.0
            after = (count.tails(count(intervened)) * alpha[0]).sum()
            torch.testing.assert_close(
                delta[0, index, 0], base - after, atol=3e-6, rtol=3e-6
            )
        else:
            assert float(delta[0, index, 0].detach()) == 0.0


def test_continuation_distribution_is_ordered_and_normalized() -> None:
    transitions = torch.tensor(
        [[0.8, 0.7, 0.4, 0.2], [0.1, 0.9, 0.6, 0.5]]
    )
    cumulative, classes = continuation_probabilities(transitions)
    assert torch.all(cumulative[..., :-1] >= cumulative[..., 1:])
    assert torch.all(classes >= 0.0)
    torch.testing.assert_close(
        classes.sum(dim=-1), torch.ones(2), atol=1e-7, rtol=0
    )
    torch.testing.assert_close(cumulative.sum(dim=-1), (classes * torch.arange(5)).sum(dim=-1))


def test_invalid_cells_never_selected_and_permutation_does_not_change_prediction() -> None:
    torch.manual_seed(4)
    logits = torch.randn(2, 8, 4)
    valid = torch.tensor(
        [[True, False, True, True, False, True, True, False], [True] * 8]
    )
    core = MOSAICOrdinalCore(
        num_classes=4,
        max_count=4,
        sufficiency_tolerance=0.02,
        complement_suppression=0.5,
        implementation="block_tree",
        block_size=3,
    )
    output = core(logits, valid, return_pivotality=True)
    assert not output.proof.selected_mask[~valid].any()
    assert torch.all(output.witness_probabilities[~valid] == 0)

    permutation = torch.tensor([5, 1, 7, 0, 3, 6, 2, 4])
    permuted = core(logits[:, permutation], valid[:, permutation])
    torch.testing.assert_close(
        output.class_probabilities,
        permuted.class_probabilities,
        atol=5e-6,
        rtol=5e-6,
    )


def test_count_math_is_fp32_and_gradients_are_finite_near_saturation() -> None:
    raw = torch.tensor(
        [[-16.0, -8.0, 0.0, 8.0, 16.0, 3.0, -3.0]], requires_grad=True
    )
    probabilities = torch.sigmoid(raw).to(torch.float16)
    layer = TruncatedPoissonBinomial(
        max_count=4, implementation="block_tree", block_size=3
    )
    distribution = layer(probabilities)
    assert distribution.dtype == torch.float32
    loss = layer.tails(distribution).sum()
    loss.backward()
    assert raw.grad is not None
    assert torch.isfinite(raw.grad).all()


def test_projected_forward_has_no_global_bypass_and_finite_parameter_gradients() -> None:
    torch.manual_seed(5)
    logits = torch.randn(2, 10, 5, requires_grad=True)
    core = MOSAICOrdinalCore(
        num_classes=5,
        max_count=4,
        sufficiency_tolerance=0.0,
        complement_suppression=0.4,
        implementation="block_tree",
        block_size=4,
    )
    output = core(logits)
    loss = -torch.log(output.class_probabilities[:, 2] + 1e-7).mean()
    loss.backward()

    assert logits.grad is not None and torch.isfinite(logits.grad).all()
    assert core.circuit.alpha_logits.grad is not None
    assert torch.isfinite(core.circuit.alpha_logits.grad).all()
    # The encoder-independent core has only the global, dataset-level alpha
    # parameters.  There is no pooled-feature or residual classification path.
    assert [name for name, _ in core.named_parameters()] == ["circuit.alpha_logits"]
