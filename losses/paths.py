"""Strictly proper risk-set training objective for PATHS.

PATHS factorises an ordinal posterior into continuation decisions.  At
boundary ``k`` its logit represents

``logit P(Y > k | Y >= k)``.

The categorical log score therefore decomposes into binary log scores on the
successive risk sets.  Multiplying *both* outcomes at a boundary by the same
fixed positive constant preserves strict propriety while allowing scarce late
boundaries to receive a comparable optimisation budget.  This is materially
different from class weighting: class weighting changes the population
posterior, whereas fixed boundary weights do not change the optimum of any
conditional continuation probability.

Boundary weights are computed once from training-fold counts only and stored
in the criterion state.  Validation/test labels must never be used to build
them.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from .origin import ranked_probability_score


def _field(output: object, name: str) -> Any:
    if isinstance(output, Mapping):
        if name not in output:
            raise KeyError(f"PATHS output has no {name!r} field")
        return output[name]
    if not hasattr(output, name):
        raise AttributeError(f"PATHS output has no {name!r} attribute")
    return getattr(output, name)


def _validate_labels(
    labels: torch.Tensor,
    *,
    batch_size: int,
    num_classes: int,
    device: torch.device,
) -> torch.Tensor:
    labels = labels.to(device=device, dtype=torch.long)
    if labels.ndim != 1 or labels.numel() != batch_size:
        raise ValueError(
            f"labels must have shape ({batch_size},), got {tuple(labels.shape)}"
        )
    invalid = (labels < 0) | (labels >= num_classes)
    if bool(invalid.any()):
        bad = labels[invalid][:8].detach().cpu().tolist()
        raise ValueError(
            f"labels must lie in [0, {num_classes - 1}]; invalid values: {bad}"
        )
    return labels


def risk_set_boundary_weights(
    label_counts: Sequence[int] | torch.Tensor,
    *,
    power: float = 1.0,
) -> torch.Tensor:
    """Build normalized inverse-risk-set weights from training counts.

    For class counts ``n_0, ..., n_{K-1}``, boundary ``k`` is observed for
    samples with ``Y >= k`` and therefore has exposure

    ``N_k = sum_{y=k}^{K-1} n_y``.

    The returned weights are proportional to ``(N_k / N_0) ** (-power)`` and
    normalized to arithmetic mean one.  The common normalization has no
    effect on the population minimizer, but keeps loss and learning-rate
    scales comparable across datasets.  ``power=0`` exactly recovers the
    ordinary categorical log score.
    """

    if not math.isfinite(power) or power < 0.0:
        raise ValueError("risk-set power must be finite and nonnegative")
    counts = torch.as_tensor(label_counts, dtype=torch.float64)
    if counts.ndim != 1 or counts.numel() < 2:
        raise ValueError("label_counts must be a one-dimensional K>=2 vector")
    if not bool(torch.isfinite(counts).all()):
        raise ValueError("label_counts must be finite")
    if bool((counts < 0).any()):
        raise ValueError("label_counts must be nonnegative")
    if bool((counts != counts.round()).any()):
        raise ValueError("label_counts must contain integer counts")
    if float(counts.sum()) <= 0.0:
        raise ValueError("label_counts must contain at least one sample")

    # Reverse cumulative counts give N_k for k=0,...,K-2.  A zero late risk
    # set means that its continuation probability is not estimable and cannot
    # define a strictly proper K-class score on this fold, so fail explicitly.
    risk_counts = counts.flip((0,)).cumsum(dim=0).flip((0,))[:-1]
    if bool((risk_counts <= 0).any()):
        raise ValueError("every ordinal boundary must have a non-empty risk set")
    relative_exposure = risk_counts / risk_counts[0]
    weights = relative_exposure.pow(-float(power))
    weights = weights / weights.mean()
    if not bool(torch.isfinite(weights).all()) or bool((weights <= 0).any()):
        raise FloatingPointError("risk-set boundary weights are not finite and positive")
    return weights


def risk_set_log_score(
    continuation_logits: torch.Tensor,
    labels: torch.Tensor,
    boundary_weights: torch.Tensor,
    *,
    reduction: str = "mean",
) -> torch.Tensor:
    """Evaluate the boundary-weighted continuation log score.

    For a sample of grade ``y``, boundaries below ``y`` have continuation
    target one, boundary ``y`` has stop target zero (unless ``y`` is the final
    grade), and later boundaries are outside the sample's risk set.  The
    per-sample loss is

    ``sum_k c_k 1[y>=k] BCEWithLogits(eta_k, 1[y>k])``.

    Reduction is over original samples, never over the number or total weight
    of active boundary terms.  A label-dependent denominator would change the
    scoring rule.
    """

    if continuation_logits.ndim != 2 or continuation_logits.shape[1] < 1:
        raise ValueError(
            "continuation_logits must have shape (N, K-1) with K>=2; got "
            f"{tuple(continuation_logits.shape)}"
        )
    if not continuation_logits.is_floating_point():
        raise TypeError("continuation_logits must be floating point")
    if not bool(torch.isfinite(continuation_logits).all()):
        raise ValueError("continuation_logits must be finite")
    batch_size, num_boundaries = continuation_logits.shape
    labels = _validate_labels(
        labels,
        batch_size=batch_size,
        num_classes=num_boundaries + 1,
        device=continuation_logits.device,
    )
    weights = torch.as_tensor(
        boundary_weights,
        device=continuation_logits.device,
        dtype=continuation_logits.dtype,
    )
    if weights.shape != (num_boundaries,):
        raise ValueError(
            f"boundary_weights must have shape ({num_boundaries},), got "
            f"{tuple(weights.shape)}"
        )
    if not bool(torch.isfinite(weights).all()) or bool((weights <= 0).any()):
        raise ValueError("boundary_weights must be finite and strictly positive")

    boundaries = torch.arange(num_boundaries, device=continuation_logits.device)
    at_risk = labels[:, None] >= boundaries[None, :]
    targets = (labels[:, None] > boundaries[None, :]).to(continuation_logits.dtype)
    terms = F.binary_cross_entropy_with_logits(
        continuation_logits,
        targets,
        reduction="none",
    )
    per_sample = (
        terms
        * at_risk.to(dtype=continuation_logits.dtype)
        * weights.unsqueeze(0)
    ).sum(dim=1)

    if reduction == "none":
        return per_sample
    if reduction == "sum":
        return per_sample.sum()
    if reduction == "mean":
        return per_sample.mean()
    raise ValueError("reduction must be 'none', 'mean', or 'sum'")


class PathsLoss(nn.Module):
    """PATHS proper ordinal score with training-fold risk equalization.

    The interface deliberately mirrors :class:`losses.origin.OriginLoss` so
    the existing trainer can consume its scalar diagnostics.  PATHS has no
    evidence-magnitude regularizer: such a penalty would give up the strict
    proper-scoring claim and is therefore kept out of the primary objective.
    """

    def __init__(
        self,
        num_classes: int,
        *,
        label_counts: Sequence[int] | torch.Tensor,
        risk_set_power: float = 1.0,
        rps_weight: float = 0.25,
    ) -> None:
        super().__init__()
        if num_classes < 2:
            raise ValueError("num_classes must be at least two")
        if not math.isfinite(rps_weight) or rps_weight < 0.0:
            raise ValueError("rps_weight must be finite and nonnegative")
        counts = torch.as_tensor(label_counts)
        if counts.ndim != 1 or counts.numel() != num_classes:
            raise ValueError(
                f"label_counts must have shape ({num_classes},), got "
                f"{tuple(counts.shape)}"
            )
        self.num_classes = int(num_classes)
        self.risk_set_power = float(risk_set_power)
        self.rps_weight = float(rps_weight)
        weights = risk_set_boundary_weights(counts, power=self.risk_set_power)
        self.register_buffer("boundary_weights", weights)

    @property
    def likelihood_component_unweighted(self) -> bool:
        """The score does not apply outcome/class importance weights.

        Boundary coefficients weight a set of proper conditional scores and
        leave every continuation target unchanged.
        """

        return True

    @property
    def configured_objective_is_proper(self) -> bool:
        return True

    def budget_is_active(self, epoch: int | None) -> bool:
        del epoch
        return False

    def forward(
        self,
        output: object,
        labels: torch.Tensor,
        *,
        epoch: int | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor | bool]]:
        del epoch
        logits = _field(output, "continuation_logits")
        cumulative = _field(output, "cumulative_probs")
        if logits.ndim != 2 or logits.shape != (
            logits.shape[0],
            self.num_classes - 1,
        ):
            raise ValueError(
                "PATHS continuation logits must have shape "
                f"(N, {self.num_classes - 1}); got {tuple(logits.shape)}"
            )
        if cumulative.shape != logits.shape:
            raise ValueError(
                "PATHS cumulative probabilities must match continuation logits; "
                f"got {tuple(cumulative.shape)} and {tuple(logits.shape)}"
            )
        primary = risk_set_log_score(
            logits,
            labels,
            self.boundary_weights,
        )
        rps = ranked_probability_score(cumulative, labels)
        total = primary + self.rps_weight * rps
        if not bool(torch.isfinite(total)):
            raise FloatingPointError("non-finite PATHS loss")
        zero = total.detach().new_zeros(())
        return total, {
            # OriginTrainer names its primary likelihood diagnostic ``nll``.
            # Keep that public contract even when non-unit proper boundary
            # coefficients make the score differ numerically from plain NLL.
            "nll": primary.detach(),
            "rps": rps.detach(),
            "evidence_budget": zero,
            "budget_active": False,
            "weighted_likelihood": False,
            "likelihood_component_unweighted": True,
            "configured_objective_is_proper": True,
        }


# Capitalized alias is convenient in paper-facing code without forcing callers
# to guess the project's preferred acronym casing.
PATHSLoss = PathsLoss


__all__ = [
    "PATHSLoss",
    "PathsLoss",
    "risk_set_boundary_weights",
    "risk_set_log_score",
]
