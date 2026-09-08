from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from losses.origin import (
    OriginLoss,
    categorical_nll,
    evidence_budget,
    ranked_probability_score,
)


def _output_from_logits(logits: torch.Tensor) -> SimpleNamespace:
    probabilities = logits.softmax(dim=1)
    cumulative = torch.stack(
        [probabilities[:, boundary + 1 :].sum(dim=1) for boundary in range(logits.shape[1] - 1)],
        dim=1,
    )
    rates = torch.nn.functional.softplus(logits[:, :-1])
    local = rates[:, None, None, :]
    return SimpleNamespace(
        class_probs=probabilities,
        log_class_probs=probabilities.log(),
        cumulative_probs=cumulative,
        total_rates=rates,
        local_rate_maps={"s8": local},
    )


def test_categorical_nll_is_exact_unweighted_log_likelihood() -> None:
    probabilities = torch.tensor([[0.7, 0.2, 0.1], [0.1, 0.3, 0.6]])
    labels = torch.tensor([0, 2])
    actual = categorical_nll(probabilities.log(), labels)
    expected = -(torch.log(torch.tensor(0.7)) + torch.log(torch.tensor(0.6))) / 2
    torch.testing.assert_close(actual, expected)


def test_ranked_probability_score_uses_all_ordinal_boundaries() -> None:
    cumulative = torch.tensor([[0.8, 0.3], [0.4, 0.1]])
    labels = torch.tensor([2, 0])
    actual = ranked_probability_score(cumulative, labels)
    expected = torch.tensor(
        (
            ((0.8 - 1.0) ** 2 + (0.3 - 1.0) ** 2) / 2
            + ((0.4 - 0.0) ** 2 + (0.1 - 0.0) ** 2) / 2
        )
        / 2
    )
    torch.testing.assert_close(actual, expected)


def test_ranked_probability_score_rejects_non_nested_cumulatives() -> None:
    with pytest.raises(ValueError, match="nested"):
        ranked_probability_score(
            torch.tensor([[0.2, 0.4]]),
            torch.tensor([1]),
        )


def test_evidence_budget_uses_local_ledger_not_total_rates() -> None:
    output = SimpleNamespace(
        log_class_probs=torch.zeros(2, 3),
        total_rates=torch.full((2, 2), 1000.0),
        local_rate_maps={
            "s8": torch.tensor(
                [
                    [[[1.0, 2.0]]],
                    [[[3.0, 4.0]]],
                ]
            )
        },
    )
    expected = (torch.log1p(torch.tensor(3.0)) + torch.log1p(torch.tensor(7.0))) / 2
    torch.testing.assert_close(evidence_budget(output), expected)


def test_origin_loss_delays_budget_until_after_dense_warmup() -> None:
    logits = torch.tensor(
        [[2.0, 0.0, -1.0], [-1.0, 0.0, 2.0]],
        requires_grad=True,
    )
    output = _output_from_logits(logits)
    labels = torch.tensor([0, 2])
    criterion = OriginLoss(
        3,
        rps_weight=0.4,
        evidence_budget_weight=0.2,
        evidence_budget_delay_epochs=2,
    )
    warmup_loss, warmup = criterion(output, labels, epoch=2)
    active_loss, active = criterion(output, labels, epoch=3)
    assert warmup["budget_active"] is False
    assert active["budget_active"] is True
    torch.testing.assert_close(warmup["evidence_budget"], torch.tensor(0.0))
    torch.testing.assert_close(
        active_loss - warmup_loss,
        0.2 * evidence_budget(output),
    )
    active_loss.backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()
    assert float(logits.grad.abs().sum()) > 0


def test_unweighted_unregularized_loss_is_flagged_as_proper_objective() -> None:
    criterion = OriginLoss(3)
    assert not criterion.uses_weighted_likelihood
    assert criterion.likelihood_component_unweighted
    assert criterion.configured_objective_is_proper


def test_weighted_likelihood_is_explicitly_flagged_as_changed_target() -> None:
    criterion = OriginLoss(3, class_weights=[1.0, 2.0, 3.0])
    assert criterion.uses_weighted_likelihood
    assert not criterion.likelihood_component_unweighted
    assert not criterion.configured_objective_is_proper


def test_rate_budget_is_explicitly_not_a_proper_scoring_objective() -> None:
    criterion = OriginLoss(3, evidence_budget_weight=1e-3)
    assert criterion.likelihood_component_unweighted
    assert not criterion.configured_objective_is_proper


def test_origin_loss_rejects_posterior_that_is_not_normalized() -> None:
    output = _output_from_logits(torch.tensor([[0.0, 0.0, 0.0]]))
    output.class_probs = output.class_probs * 0.9
    with pytest.raises(ValueError, match="sum to one"):
        OriginLoss(3)(output, torch.tensor([1]), epoch=1)


def test_target_negative_infinity_fails_loudly() -> None:
    log_probs = torch.tensor([[0.0, -torch.inf]])
    with pytest.raises(FloatingPointError, match="target class"):
        categorical_nll(log_probs, torch.tensor([1]))
