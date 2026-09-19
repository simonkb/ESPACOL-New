"""Mathematical invariants for ordinal-shell warranted transport (OSWT)."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from models.paths_oswt import (
    PathsOSWTRefiner,
    apply_shell_warranted_transport,
    ordinal_shells_from_cumulative_atoms,
)


def test_ordinal_shells_form_exact_nonnegative_simplex_and_mask_invalid_cells() -> None:
    atoms = torch.tensor(
        [
            [
                [[0.90, 0.75], [0.60, 0.40]],
                [[0.60, 0.50], [0.40, 0.30]],
                [[0.40, 0.30], [0.20, 0.20]],
                [[0.10, 0.10], [0.05, 0.02]],
            ]
        ],
        dtype=torch.float64,
    )
    valid = torch.tensor([[[True, False], [True, True]]])
    shells = ordinal_shells_from_cumulative_atoms(atoms, valid)

    expected_first = torch.tensor(
        [0.10, 0.30, 0.20, 0.30, 0.10], dtype=torch.float64
    )
    torch.testing.assert_close(shells[0, :, 0, 0], expected_first)
    assert torch.all(shells >= 0.0)
    torch.testing.assert_close(
        shells.sum(dim=1)[valid],
        torch.ones(int(valid.sum()), dtype=torch.float64),
        atol=2e-15,
        rtol=0.0,
    )
    assert torch.equal(shells[0, :, 0, 1], torch.zeros(5, dtype=torch.float64))


def test_zero_strength_refiner_is_exact_origin_object_identity() -> None:
    sentinel = object()
    refiner = PathsOSWTRefiner(5, ("s4",), strength=0.0)
    assert refiner(sentinel) is sentinel  # type: ignore[arg-type]


def test_grade_three_shell_attracts_probability_from_both_adjacent_grades() -> None:
    # Row 0 is predominantly grade 2; row 1 is predominantly grade 4.
    base = torch.tensor(
        [
            [0.02, 0.03, 0.70, 0.20, 0.05],
            [0.02, 0.03, 0.05, 0.20, 0.70],
        ],
        dtype=torch.float64,
    )
    left = torch.tensor([[0.5, 0.5, 0.05, 0.80]] * 2, dtype=torch.float64)
    right = torch.tensor([[0.5, 0.5, 0.80, 0.05]] * 2, dtype=torch.float64)
    result = apply_shell_warranted_transport(
        base,
        left,
        right,
        beta=torch.full((4,), 3.0, dtype=torch.float64),
        tau=torch.full((4,), 0.05, dtype=torch.float64),
    )

    # Boundary 2 pushes 2 -> 3; boundary 3 pushes 4 -> 3.
    assert result.log_odds_increment[0, 2] > 0.0
    assert result.log_odds_increment[1, 3] < 0.0
    assert result.class_probs[0, 3] > base[0, 3]
    assert result.class_probs[1, 3] > base[1, 3]
    assert result.class_probs[0, 2] < base[0, 2]
    assert result.class_probs[1, 4] < base[1, 4]


def test_posterior_normalization_adjacent_odds_and_signed_flow_are_exact() -> None:
    base = torch.tensor(
        [[0.13, 0.21, 0.17, 0.29, 0.20], [0.55, 0.15, 0.12, 0.10, 0.08]],
        dtype=torch.float64,
    )
    left = torch.tensor(
        [[0.10, 0.40, 0.20, 0.80], [0.65, 0.20, 0.35, 0.15]],
        dtype=torch.float64,
    )
    right = torch.tensor(
        [[0.70, 0.30, 0.75, 0.15], [0.20, 0.80, 0.25, 0.70]],
        dtype=torch.float64,
    )
    result = apply_shell_warranted_transport(
        base,
        left,
        right,
        beta=torch.tensor([0.7, 1.1, 0.9, 1.3], dtype=torch.float64),
        tau=torch.tensor([0.03, 0.07, 0.05, 0.09], dtype=torch.float64),
    )

    torch.testing.assert_close(
        result.class_probs.sum(dim=-1),
        torch.ones(2, dtype=torch.float64),
        atol=2e-15,
        rtol=0.0,
    )
    torch.testing.assert_close(
        result.refined_adjacent_log_odds,
        result.base_adjacent_log_odds + result.log_odds_increment,
        atol=2e-14,
        rtol=2e-14,
    )
    expected_flow = torch.cumsum(base - result.class_probs, dim=-1)[..., :-1]
    expected_w1 = expected_flow.abs().sum(dim=-1)
    torch.testing.assert_close(result.signed_boundary_flow, expected_flow)
    torch.testing.assert_close(result.wasserstein1, expected_w1)

    replayed_delta = torch.cat(
        (
            -result.signed_boundary_flow[..., :1],
            result.signed_boundary_flow[..., :-1]
            - result.signed_boundary_flow[..., 1:],
            result.signed_boundary_flow[..., -1:],
        ),
        dim=-1,
    )
    torch.testing.assert_close(result.class_probs - base, replayed_delta)


def test_transport_has_finite_nonzero_gradients_and_passes_gradcheck() -> None:
    logits = torch.tensor(
        [[-0.8, -0.2, 0.1, 0.5, -0.1]], dtype=torch.float64, requires_grad=True
    )
    left_logits = torch.tensor(
        [[-1.0, 0.2, -0.4, 1.1]], dtype=torch.float64, requires_grad=True
    )
    right_logits = torch.tensor(
        [[0.7, -0.5, 0.6, -0.8]], dtype=torch.float64, requires_grad=True
    )
    raw_beta = torch.tensor(
        [-0.2, 0.1, 0.3, -0.1], dtype=torch.float64, requires_grad=True
    )
    raw_tau = torch.tensor(
        [-2.0, -1.7, -1.9, -1.6], dtype=torch.float64, requires_grad=True
    )

    def function(l, left, right, b, t):
        return apply_shell_warranted_transport(
            l.softmax(dim=-1),
            left.sigmoid(),
            right.sigmoid(),
            F.softplus(b),
            F.softplus(t),
        ).class_probs

    assert torch.autograd.gradcheck(
        function,
        (logits, left_logits, right_logits, raw_beta, raw_tau),
        eps=1e-6,
        atol=2e-5,
        rtol=2e-4,
    )
    loss = -function(
        logits, left_logits, right_logits, raw_beta, raw_tau
    )[0, 3].log()
    loss.backward()
    for tensor in (logits, left_logits, right_logits, raw_beta, raw_tau):
        assert tensor.grad is not None
        assert torch.isfinite(tensor.grad).all()
        assert float(tensor.grad.abs().sum()) > 0.0


def test_boundary_zero_is_coverage_aware_two_sided_and_identity_without_warrant() -> None:
    base = torch.tensor(
        [[0.60, 0.15, 0.12, 0.08, 0.05]] * 3,
        dtype=torch.float64,
    )
    left = torch.tensor(
        [
            [0.00, 0.00, 0.00, 0.00],
            [0.70, 0.40, 0.40, 0.40],
            [0.02, 0.40, 0.40, 0.40],
        ],
        dtype=torch.float64,
    )
    right = torch.tensor(
        [
            [0.00, 0.00, 0.00, 0.00],
            [0.02, 0.40, 0.40, 0.40],
            [0.70, 0.40, 0.40, 0.40],
        ],
        dtype=torch.float64,
    )
    result = apply_shell_warranted_transport(
        base,
        left,
        right,
        beta=torch.ones(4, dtype=torch.float64),
        tau=torch.full((4,), 0.05, dtype=torch.float64),
    )

    assert result.shell_contrast[0, 0] == 0.0
    assert result.boundary_coverage[0, 0] == 0.0
    torch.testing.assert_close(result.class_probs[0], base[0], atol=2e-15, rtol=0.0)
    assert result.log_odds_increment[1, 0] < 0.0
    assert result.log_odds_increment[2, 0] > 0.0
    assert result.left_boundary_warrant[1, 0] > result.right_boundary_warrant[1, 0]
    assert result.right_boundary_warrant[2, 0] > result.left_boundary_warrant[2, 0]


def test_cut_aligned_gate_is_exact_and_ungated_control_is_one() -> None:
    base = torch.tensor(
        [[0.10, 0.20, 0.30, 0.25, 0.15]], dtype=torch.float64
    )
    left = torch.full((1, 4), 0.2, dtype=torch.float64)
    right = torch.full((1, 4), 0.8, dtype=torch.float64)
    beta = torch.ones(4, dtype=torch.float64)
    tau = torch.full((4,), 0.05, dtype=torch.float64)
    gated = apply_shell_warranted_transport(base, left, right, beta, tau)
    ungated = apply_shell_warranted_transport(
        base, left, right, beta, tau, risk_gate_enabled=False
    )
    q = torch.tensor([[0.90, 0.70, 0.40, 0.15]], dtype=torch.float64)
    torch.testing.assert_close(gated.base_boundary_probability, q)
    torch.testing.assert_close(gated.boundary_gate, 4.0 * q * (1.0 - q))
    torch.testing.assert_close(ungated.boundary_gate, torch.ones_like(q))


def test_a_boundary_increment_does_not_depend_on_number_of_grades() -> None:
    three = apply_shell_warranted_transport(
        torch.tensor([[0.4, 0.3, 0.3]], dtype=torch.float64),
        torch.tensor([[0.2, 0.5]], dtype=torch.float64),
        torch.tensor([[0.8, 0.5]], dtype=torch.float64),
        torch.ones(2, dtype=torch.float64),
        torch.full((2,), 0.1, dtype=torch.float64),
    )
    five = apply_shell_warranted_transport(
        torch.tensor([[0.4, 0.1, 0.1, 0.1, 0.3]], dtype=torch.float64),
        torch.tensor([[0.2, 0.5, 0.5, 0.5]], dtype=torch.float64),
        torch.tensor([[0.8, 0.5, 0.5, 0.5]], dtype=torch.float64),
        torch.ones(4, dtype=torch.float64),
        torch.full((4,), 0.1, dtype=torch.float64),
    )
    torch.testing.assert_close(
        three.log_odds_increment[:, 0], five.log_odds_increment[:, 0]
    )


def test_refiner_beta_and_tau_are_explicitly_bounded() -> None:
    refiner = PathsOSWTRefiner(
        5,
        ("s4",),
        beta_init=0.1,
        tau_init=0.05,
        beta_cap=3.0,
        tau_floor=1e-3,
        tau_cap=0.5,
    )
    torch.testing.assert_close(refiner.beta, torch.full((4,), 0.1, dtype=torch.float64))
    torch.testing.assert_close(refiner.tau, torch.full((4,), 0.05, dtype=torch.float64))
    with torch.no_grad():
        refiner.raw_beta.fill_(100.0)
        refiner.raw_tau.fill_(-100.0)
    assert torch.all(refiner.beta <= 3.0)
    assert torch.all(refiner.tau >= 1e-3)
