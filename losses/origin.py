"""Training objectives for ORIGIN's generator-exclusive ordinal posterior.

ORIGIN exposes a normalized categorical posterior produced by its spatial
generator.  The primary objective in this module is therefore ordinary
categorical negative log likelihood, not an independently learned classifier
or a collection of binary surrogates.  A ranked probability score (RPS) adds
an ordinal distance signal while remaining a proper scoring rule.  An optional
delayed evidence budget is an explicitly regularizing ablation and is disabled
by default.

Class weighting is supported for controlled imbalance experiments.  It changes
the population target of the likelihood, so callers must opt in explicitly.
Neither an unweighted loss nor a proper scoring rule, by itself, proves that a
finite fitted model is empirically calibrated; calibration is measured
separately by the trainer.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch
import torch.nn as nn


def _field(output: object, name: str) -> Any:
    if isinstance(output, Mapping):
        if name not in output:
            raise KeyError(f"ORIGIN output has no {name!r} field")
        return output[name]
    if not hasattr(output, name):
        raise AttributeError(f"ORIGIN output has no {name!r} attribute")
    return getattr(output, name)


def _validate_labels(labels: torch.Tensor, n: int, num_classes: int) -> torch.Tensor:
    labels = labels.to(dtype=torch.long)
    if labels.ndim != 1 or labels.numel() != n:
        raise ValueError(f"labels must have shape ({n},), got {tuple(labels.shape)}")
    invalid = (labels < 0) | (labels >= num_classes)
    if bool(invalid.any()):
        bad = labels[invalid][:8].detach().cpu().tolist()
        raise ValueError(
            f"labels must lie in [0, {num_classes - 1}]; invalid values: {bad}"
        )
    return labels


def ranked_probability_score(
    cumulative_probs: torch.Tensor,
    labels: torch.Tensor,
    *,
    reduction: str = "mean",
) -> torch.Tensor:
    """Return the ordinal ranked probability score.

    ``cumulative_probs[:, k]`` is ``P(Y > k)``.  The returned per-sample score
    is the mean squared cumulative-distribution error across the ``K-1``
    boundaries.  Both this score and categorical NLL are proper scoring rules.
    """

    if cumulative_probs.ndim != 2 or cumulative_probs.shape[1] < 1:
        raise ValueError(
            "cumulative_probs must have shape (N, K-1) with K >= 2; got "
            f"{tuple(cumulative_probs.shape)}"
        )
    if not torch.is_floating_point(cumulative_probs):
        raise TypeError("cumulative_probs must be floating point")
    if not bool(torch.isfinite(cumulative_probs).all()):
        raise ValueError("cumulative_probs must be finite")
    tolerance = 2e-5 if cumulative_probs.dtype in (torch.float16, torch.bfloat16) else 1e-6
    if bool(((cumulative_probs < -tolerance) | (cumulative_probs > 1.0 + tolerance)).any()):
        raise ValueError("cumulative_probs must lie in [0, 1]")
    if cumulative_probs.shape[1] > 1 and bool(
        (cumulative_probs[:, 1:] > cumulative_probs[:, :-1] + tolerance).any()
    ):
        raise ValueError("ORIGIN cumulative probabilities must be nested")

    n, boundaries = cumulative_probs.shape
    labels = _validate_labels(labels.to(cumulative_probs.device), n, boundaries + 1)
    threshold = torch.arange(boundaries, device=cumulative_probs.device)
    target = (labels.unsqueeze(1) > threshold.unsqueeze(0)).to(cumulative_probs.dtype)
    per_sample = (cumulative_probs - target).square().mean(dim=1)
    if reduction == "none":
        return per_sample
    if reduction == "sum":
        return per_sample.sum()
    if reduction == "mean":
        return per_sample.mean()
    raise ValueError("reduction must be 'none', 'mean', or 'sum'")


def categorical_nll(
    log_class_probs: torch.Tensor,
    labels: torch.Tensor,
    *,
    class_weights: torch.Tensor | None = None,
    reduction: str = "mean",
) -> torch.Tensor:
    """Negative log likelihood on an already normalized class posterior.

    With class weights, ``mean`` divides by the sum of selected sample weights,
    matching PyTorch's weighted NLL convention.  Unweighted NLL is the default
    unweighted empirical-population likelihood used by ORIGIN.
    """

    if log_class_probs.ndim != 2 or log_class_probs.shape[1] < 2:
        raise ValueError(
            "log_class_probs must have shape (N, K) with K >= 2; got "
            f"{tuple(log_class_probs.shape)}"
        )
    if not torch.is_floating_point(log_class_probs):
        raise TypeError("log_class_probs must be floating point")
    if bool(torch.isnan(log_class_probs).any()) or bool(torch.isposinf(log_class_probs).any()):
        raise ValueError("log_class_probs must not contain NaN or +inf")
    n, num_classes = log_class_probs.shape
    labels = _validate_labels(labels.to(log_class_probs.device), n, num_classes)
    selected = log_class_probs.gather(1, labels.unsqueeze(1)).squeeze(1)
    if not bool(torch.isfinite(selected).all()):
        raise FloatingPointError("target class has non-finite log probability")
    per_sample = -selected

    selected_weights: torch.Tensor | None = None
    if class_weights is not None:
        class_weights = torch.as_tensor(
            class_weights,
            device=log_class_probs.device,
            dtype=log_class_probs.dtype,
        )
        if class_weights.shape != (num_classes,):
            raise ValueError(
                f"class_weights must have shape ({num_classes},), got "
                f"{tuple(class_weights.shape)}"
            )
        if not bool(torch.isfinite(class_weights).all()) or bool((class_weights <= 0).any()):
            raise ValueError("class_weights must be finite and strictly positive")
        selected_weights = class_weights[labels]
        per_sample = per_sample * selected_weights

    if reduction == "none":
        return per_sample
    if reduction == "sum":
        return per_sample.sum()
    if reduction == "mean":
        if selected_weights is None:
            return per_sample.mean()
        return per_sample.sum() / selected_weights.sum().clamp_min(
            torch.finfo(per_sample.dtype).tiny
        )
    raise ValueError("reduction must be 'none', 'mean', or 'sum'")


def _rate_map_sample_sums(
    local_rate_maps: Mapping[str, torch.Tensor],
    *,
    batch_size: int,
) -> torch.Tensor:
    totals: torch.Tensor | None = None
    for name, rate_map in local_rate_maps.items():
        if not torch.is_tensor(rate_map):
            raise TypeError(f"local_rate_maps[{name!r}] must be a tensor")
        if rate_map.ndim < 2 or rate_map.shape[0] != batch_size:
            raise ValueError(
                f"local_rate_maps[{name!r}] must start with batch dimension "
                f"{batch_size}; got {tuple(rate_map.shape)}"
            )
        if not bool(torch.isfinite(rate_map).all()) or bool((rate_map < 0).any()):
            raise ValueError(f"local_rate_maps[{name!r}] must be finite and nonnegative")
        sample_sum = rate_map.reshape(batch_size, -1).sum(dim=1)
        totals = sample_sum if totals is None else totals + sample_sum
    if totals is None:
        raise ValueError("local_rate_maps must contain at least one scale")
    return totals


def evidence_budget(output: object) -> torch.Tensor:
    """Return ``mean(log1p(sum local rates))`` for an ORIGIN output.

    The local ledger is preferred because it excludes an optional null-prior
    generator.  ``total_rates`` is accepted as a conservative fallback for
    synthetic heads and older checkpoints.
    """

    log_probs = _field(output, "log_class_probs")
    batch_size = int(log_probs.shape[0])
    local_maps = (
        output.get("local_rate_maps")
        if isinstance(output, Mapping)
        else getattr(output, "local_rate_maps", None)
    )
    if isinstance(local_maps, Mapping) and local_maps:
        sample_totals = _rate_map_sample_sums(local_maps, batch_size=batch_size)
    else:
        total_rates = _field(output, "total_rates")
        if total_rates.ndim != 2 or total_rates.shape[0] != batch_size:
            raise ValueError(
                "total_rates must have shape (N, K-1), got "
                f"{tuple(total_rates.shape)}"
            )
        if not bool(torch.isfinite(total_rates).all()) or bool((total_rates < 0).any()):
            raise ValueError("total_rates must be finite and nonnegative")
        sample_totals = total_rates.sum(dim=1)
    return torch.log1p(sample_totals).mean()


class OriginLoss(nn.Module):
    """Ordinal posterior loss with an optional total-rate penalty ablation."""

    def __init__(
        self,
        num_classes: int,
        *,
        rps_weight: float = 0.25,
        evidence_budget_weight: float = 0.0,
        evidence_budget_delay_epochs: int = 0,
        class_weights: Sequence[float] | torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        if num_classes < 2:
            raise ValueError("num_classes must be at least 2")
        if rps_weight < 0 or evidence_budget_weight < 0:
            raise ValueError("loss weights must be nonnegative")
        if evidence_budget_delay_epochs < 0:
            raise ValueError("evidence_budget_delay_epochs must be nonnegative")
        self.num_classes = int(num_classes)
        self.rps_weight = float(rps_weight)
        self.evidence_budget_weight = float(evidence_budget_weight)
        self.evidence_budget_delay_epochs = int(evidence_budget_delay_epochs)
        if class_weights is None:
            weights = torch.empty(0, dtype=torch.float32)
        else:
            weights = torch.as_tensor(class_weights, dtype=torch.float32).flatten()
            if weights.shape != (num_classes,):
                raise ValueError(
                    f"class_weights must have shape ({num_classes},), got "
                    f"{tuple(weights.shape)}"
                )
            if not bool(torch.isfinite(weights).all()) or bool((weights <= 0).any()):
                raise ValueError("class_weights must be finite and strictly positive")
        self.register_buffer("class_weights", weights)

    @property
    def uses_weighted_likelihood(self) -> bool:
        return self.class_weights.numel() > 0

    @property
    def likelihood_component_unweighted(self) -> bool:
        """Whether the NLL component uses the empirical sampling weights."""

        return not self.uses_weighted_likelihood

    @property
    def configured_objective_is_proper(self) -> bool:
        """Whether configured loss terms preserve a proper posterior target.

        NLL plus RPS is proper. The optional rate-magnitude regularizer is a
        parameter penalty, so activating it deliberately gives up this claim.
        Sampling-policy changes are owned by the trainer and are not visible
        to this loss object.
        """

        return self.likelihood_component_unweighted and self.evidence_budget_weight == 0.0

    def budget_is_active(self, epoch: int | None) -> bool:
        if self.evidence_budget_weight == 0.0:
            return False
        if epoch is None:
            return self.evidence_budget_delay_epochs == 0
        # Epochs are one-indexed in the trainer.  A delay of d means epochs
        # 1,...,d are dense warm-up and the budget starts at d+1.
        return int(epoch) > self.evidence_budget_delay_epochs

    def forward(
        self,
        output: object,
        labels: torch.Tensor,
        *,
        epoch: int | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor | bool]]:
        log_probs = _field(output, "log_class_probs")
        probs = _field(output, "class_probs")
        cumulative = _field(output, "cumulative_probs")
        if log_probs.shape != probs.shape or log_probs.shape[1] != self.num_classes:
            raise ValueError(
                f"ORIGIN posterior must have shape (N, {self.num_classes})"
            )
        if cumulative.shape != (log_probs.shape[0], self.num_classes - 1):
            raise ValueError(
                "ORIGIN cumulative posterior has the wrong shape: "
                f"{tuple(cumulative.shape)}"
            )
        if not bool(torch.isfinite(probs).all()) or bool((probs < 0).any()):
            raise ValueError("class_probs must be finite and nonnegative")
        probability_error = (probs.sum(dim=1) - 1.0).abs().max()
        tolerance = 5e-5 if probs.dtype in (torch.float16, torch.bfloat16) else 2e-5
        if float(probability_error.detach()) > tolerance:
            raise ValueError(
                "class_probs must sum to one; maximum absolute error is "
                f"{float(probability_error.detach()):.3g}"
            )

        weights = self.class_weights if self.uses_weighted_likelihood else None
        nll = categorical_nll(log_probs, labels, class_weights=weights)
        rps = ranked_probability_score(cumulative, labels)
        active = self.budget_is_active(epoch)
        if active:
            budget = evidence_budget(output)
        else:
            # Preserve device/dtype without touching the ledger during warm-up.
            budget = log_probs.new_zeros(())
        total = nll + self.rps_weight * rps
        if active:
            total = total + self.evidence_budget_weight * budget
        if not bool(torch.isfinite(total)):
            raise FloatingPointError("non-finite ORIGIN loss")
        return total, {
            "nll": nll.detach(),
            "rps": rps.detach(),
            "evidence_budget": budget.detach(),
            "budget_active": active,
            "weighted_likelihood": self.uses_weighted_likelihood,
            "likelihood_component_unweighted": self.likelihood_component_unweighted,
            "configured_objective_is_proper": self.configured_objective_is_proper,
        }


__all__ = [
    "OriginLoss",
    "categorical_nll",
    "evidence_budget",
    "ranked_probability_score",
]
