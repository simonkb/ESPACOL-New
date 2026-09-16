from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from losses.origin import categorical_nll
from losses.paths import (
    PathsLoss,
    risk_set_boundary_weights,
    risk_set_log_score,
)


def _output_from_continuation_logits(logits: torch.Tensor) -> SimpleNamespace:
    continuation = logits.sigmoid()
    cumulative = continuation.cumprod(dim=1)
    leading_survival = torch.cat(
        (torch.ones_like(cumulative[:, :1]), cumulative[:, :-1]),
        dim=1,
    )
    stopped = leading_survival * (1.0 - continuation)
    class_probs = torch.cat((stopped, cumulative[:, -1:]), dim=1)
    return SimpleNamespace(
        continuation_logits=logits,
        cumulative_probs=cumulative,
        class_probs=class_probs,
        log_class_probs=class_probs.log(),
    )


def _continuation_logits_from_probs(probabilities: torch.Tensor) -> torch.Tensor:
    tail = probabilities.flip((0,)).cumsum(dim=0).flip((0,))
    continuation = tail[1:] / tail[:-1]
    return torch.logit(continuation)


def test_risk_set_weights_are_inverse_exposure_and_mean_one() -> None:
    counts = torch.tensor([70, 20, 8, 2])
    actual = risk_set_boundary_weights(counts, power=1.0)
    risk_counts = torch.tensor([100.0, 30.0, 10.0], dtype=torch.float64)
    expected = (risk_counts / risk_counts[0]).reciprocal()
    expected = expected / expected.mean()
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(actual.mean(), torch.tensor(1.0, dtype=actual.dtype))
    assert bool((actual > 0).all())


def test_zero_risk_weight_power_recovers_unit_boundary_weights() -> None:
    actual = risk_set_boundary_weights([70, 20, 8, 2], power=0.0)
    torch.testing.assert_close(actual, torch.ones(3, dtype=torch.float64))


def test_risk_set_weights_reject_an_unobserved_late_boundary() -> None:
    with pytest.raises(ValueError, match="non-empty risk set"):
        risk_set_boundary_weights([10, 0, 0], power=1.0)


def test_unit_risk_set_score_is_exact_categorical_nll() -> None:
    logits = torch.tensor(
        [
            [-1.2, 0.7, -0.4, 1.1],
            [2.0, -0.3, 0.8, -1.5],
            [-0.6, -0.2, 1.3, 0.4],
            [0.5, 0.9, -1.0, -0.8],
            [1.0, 0.2, 0.3, 1.4],
        ],
        dtype=torch.float64,
    )
    labels = torch.arange(5)
    output = _output_from_continuation_logits(logits)
    sequential = risk_set_log_score(logits, labels, torch.ones(4))
    categorical = categorical_nll(output.log_class_probs, labels)
    torch.testing.assert_close(sequential, categorical, atol=1e-12, rtol=1e-12)


def test_weighted_score_has_strictly_proper_population_target() -> None:
    true_probs = torch.tensor([0.52, 0.18, 0.20, 0.08, 0.02], dtype=torch.float64)
    true_logits = _continuation_logits_from_probs(true_probs)
    weights = risk_set_boundary_weights([520, 180, 200, 80, 20], power=1.0)

    def expected_score(candidate: torch.Tensor) -> torch.Tensor:
        losses = []
        for grade in range(true_probs.numel()):
            loss = risk_set_log_score(
                candidate.unsqueeze(0),
                torch.tensor([grade]),
                weights,
            )
            losses.append(true_probs[grade] * loss)
        return torch.stack(losses).sum()

    optimum = expected_score(true_logits)
    perturbations = (
        torch.tensor([0.4, 0.0, 0.0, 0.0], dtype=torch.float64),
        torch.tensor([0.0, -0.5, 0.0, 0.0], dtype=torch.float64),
        torch.tensor([0.0, 0.0, 0.7, 0.0], dtype=torch.float64),
        torch.tensor([0.0, 0.0, 0.0, -0.6], dtype=torch.float64),
        torch.tensor([0.2, -0.3, 0.4, -0.5], dtype=torch.float64),
    )
    for delta in perturbations:
        assert float(expected_score(true_logits + delta)) > float(optimum)

    differentiable = true_logits.clone().requires_grad_(True)
    expected_score(differentiable).backward()
    assert differentiable.grad is not None
    torch.testing.assert_close(
        differentiable.grad,
        torch.zeros_like(differentiable.grad),
        atol=1e-12,
        rtol=0.0,
    )


def test_paths_loss_is_finite_for_extreme_continuation_logits() -> None:
    logits = torch.tensor(
        [
            [1000.0, -1000.0, 1000.0, -1000.0],
            [-1000.0, 1000.0, -1000.0, 1000.0],
        ],
        requires_grad=True,
    )
    output = _output_from_continuation_logits(logits)
    criterion = PathsLoss(
        5,
        label_counts=[100, 30, 20, 8, 2],
        risk_set_power=1.0,
        rps_weight=0.25,
    )
    loss, diagnostics = criterion(output, torch.tensor([4, 0]), epoch=7)
    assert torch.isfinite(loss)
    assert torch.isfinite(diagnostics["nll"])
    assert torch.isfinite(diagnostics["rps"])
    loss.backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()


def test_paths_loss_exposes_origin_trainer_contract() -> None:
    criterion = PathsLoss(
        3,
        label_counts=[80, 15, 5],
        risk_set_power=0.5,
        rps_weight=0.4,
    )
    logits = torch.tensor([[-0.2, 0.7], [1.1, -0.8]], requires_grad=True)
    output = _output_from_continuation_logits(logits)
    loss, diagnostics = criterion(output, torch.tensor([0, 2]), epoch=3)

    assert criterion.likelihood_component_unweighted
    assert criterion.configured_objective_is_proper
    assert criterion.budget_is_active(3) is False
    assert diagnostics["budget_active"] is False
    assert diagnostics["weighted_likelihood"] is False
    assert diagnostics["likelihood_component_unweighted"] is True
    assert diagnostics["configured_objective_is_proper"] is True
    torch.testing.assert_close(diagnostics["evidence_budget"], torch.tensor(0.0))
    expected = diagnostics["nll"] + 0.4 * diagnostics["rps"]
    torch.testing.assert_close(loss.detach(), expected)


def test_reduction_does_not_renormalize_by_active_risk_terms() -> None:
    logits = torch.zeros(2, 3)
    labels = torch.tensor([0, 3])
    weights = torch.tensor([0.5, 1.0, 2.0])
    actual = risk_set_log_score(logits, labels, weights, reduction="none")
    log_two = torch.log(torch.tensor(2.0))
    expected = torch.tensor(
        [0.5 * log_two, (0.5 + 1.0 + 2.0) * log_two]
    )
    torch.testing.assert_close(actual, expected)


def test_risk_set_score_matches_direct_bce_masking() -> None:
    logits = torch.tensor([[0.3, -0.4, 1.2]], dtype=torch.float64)
    label = torch.tensor([1])
    weights = torch.tensor([0.2, 1.1, 1.7], dtype=torch.float64)
    actual = risk_set_log_score(logits, label, weights)
    direct = (
        0.2
        * F.binary_cross_entropy_with_logits(
            logits[:, 0], torch.ones(1, dtype=logits.dtype)
        )
        + 1.1
        * F.binary_cross_entropy_with_logits(
            logits[:, 1], torch.zeros(1, dtype=logits.dtype)
        )
    )
    torch.testing.assert_close(actual, direct)
