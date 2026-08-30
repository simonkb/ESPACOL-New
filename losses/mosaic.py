"""Losses for the MOSAIC ordinal proof model.

MOSAIC predicts conditional continuation probabilities ``c[:, k]`` for the
event that an image which reached grade rung ``k`` advances to rung ``k + 1``.
Consequently, a grade-``y`` image contributes positive observations at every
boundary below ``y`` and (unless it is the last grade) one negative observation
at boundary ``y``.  Later boundaries are undefined for that image and must not
be treated as negatives.

This module intentionally contains only the three losses in the MOSAIC plan:

* balanced at-risk continuation negative log likelihood;
* the same likelihood on the training-only dense transition; and
* optional Jensen--Shannon consistency of local categorical witness states.

It does not add a global cross-entropy or any other classifier bypass.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import torch
import torch.nn as nn


WeightMethod = Literal["effective_num", "inverse_frequency", "none"]
Reduction = Literal["none", "mean", "sum"]


def _labels_tensor(labels: Sequence[int] | torch.Tensor) -> torch.Tensor:
    """Return a detached, one-dimensional CPU long tensor of labels."""

    values = torch.as_tensor(labels, dtype=torch.long, device="cpu").flatten()
    if values.numel() == 0:
        raise ValueError("training labels must not be empty")
    return values


def _validate_labels(labels: torch.Tensor, num_classes: int) -> None:
    if num_classes < 2:
        raise ValueError(f"num_classes must be at least 2, got {num_classes}")
    if labels.ndim != 1:
        raise ValueError(f"labels must be one-dimensional, got {labels.shape}")
    invalid = (labels < 0) | (labels >= num_classes)
    if invalid.any():
        bad = labels[invalid][:8].tolist()
        raise ValueError(
            f"labels must lie in [0, {num_classes - 1}]; invalid values: {bad}"
        )


def transition_outcome_counts(
    labels: Sequence[int] | torch.Tensor,
    num_classes: int,
) -> torch.Tensor:
    """Count negative/positive outcomes inside each boundary's risk set.

    Returns a ``(K - 1, 2)`` long tensor.  At boundary ``k``:

    * column 0 counts ``Y == k`` (stop at this rung), and
    * column 1 counts ``Y > k`` (advance to the next rung).

    Samples with ``Y < k`` are not at risk and are excluded from both columns.
    Counts are computed from the complete training fold, never per minibatch.
    """

    values = _labels_tensor(labels)
    _validate_labels(values, num_classes)
    boundaries = torch.arange(num_classes - 1, dtype=torch.long).unsqueeze(1)
    y = values.unsqueeze(0)
    negatives = (y == boundaries).sum(dim=1)
    positives = (y > boundaries).sum(dim=1)
    return torch.stack((negatives, positives), dim=1)


def _normalise_weights_with_cap(
    raw: torch.Tensor,
    counts: torch.Tensor,
    max_weight: float | None,
) -> torch.Tensor:
    """Give observed at-risk examples unit mean weight, respecting a hard cap.

    A small bisection solves ``sum n_j min(cap, scale * raw_j) = sum n_j``.
    This avoids the common clamp-then-renormalise bug, which can silently break
    the requested rare-boundary cap.  With ``max_weight >= 1`` a solution
    exists whenever the boundary has at least one observation.
    """

    observed = counts > 0
    if not observed.any():
        return torch.zeros_like(raw)

    raw = torch.where(observed, raw, torch.zeros_like(raw))
    total = counts.sum()
    if max_weight is None:
        scale = total / (counts * raw).sum().clamp_min(torch.finfo(raw.dtype).tiny)
        return raw * scale

    cap = float(max_weight)
    if cap < 1.0:
        raise ValueError("max_weight must be at least 1.0 when weights are normalised")

    # The objective is monotone in scale.  Grow the upper bracket until every
    # observed outcome is capped, then bisect.  Work in float64 on CPU because
    # this runs once per fold, not in the training loop.
    lo = torch.tensor(0.0, dtype=raw.dtype)
    hi = torch.tensor(1.0, dtype=raw.dtype)

    def weighted_sum(scale: torch.Tensor) -> torch.Tensor:
        return (counts * torch.clamp(raw * scale, max=cap)).sum()

    while weighted_sum(hi) < total:
        hi = hi * 2.0
    for _ in range(80):
        mid = (lo + hi) * 0.5
        if weighted_sum(mid) < total:
            lo = mid
        else:
            hi = mid
    return torch.where(observed, torch.clamp(raw * hi, max=cap), torch.zeros_like(raw))


def build_at_risk_transition_weights(
    labels: Sequence[int] | torch.Tensor,
    num_classes: int,
    *,
    method: WeightMethod = "effective_num",
    beta: float = 0.999,
    max_weight: float | None = 10.0,
    normalise: bool = True,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
    return_counts: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Construct fold-level weights for continuation outcomes.

    Args:
        labels: All integer labels in the *training fold*.
        num_classes: Number of ordered grades ``K``.
        method: ``"effective_num"`` (recommended), ``"inverse_frequency"``,
            or ``"none"``.
        beta: Effective-number decay in ``[0, 1)``.  Ignored by other methods.
        max_weight: Hard cap applied after per-boundary normalisation.  A cap
            controls noisy rare late-boundary outcomes.  Set ``None`` to disable.
        normalise: If true, observed examples at each boundary have mean weight
            one (subject to the hard cap, which is solved exactly).
        return_counts: Also return the CPU ``(K-1, 2)`` outcome-count tensor.

    Missing outcomes receive weight zero.  This is safe because no training
    term can select that weight; assigning a fabricated large value would make
    fold statistics misleading.
    """

    counts_long = transition_outcome_counts(labels, num_classes)
    counts = counts_long.to(torch.float64)
    observed = counts > 0

    if method == "effective_num":
        if not 0.0 <= beta < 1.0:
            raise ValueError(f"beta must lie in [0, 1), got {beta}")
        if beta == 0.0:
            raw = observed.to(torch.float64)
        else:
            # effective_n = (1 - beta**n) / (1 - beta).  expm1 is stable for
            # beta close to one and small n.
            log_beta = torch.log(torch.tensor(beta, dtype=torch.float64))
            effective_n = -torch.expm1(counts * log_beta) / (1.0 - beta)
            raw = torch.where(observed, effective_n.reciprocal(), torch.zeros_like(counts))
    elif method == "inverse_frequency":
        raw = torch.where(observed, counts.reciprocal(), torch.zeros_like(counts))
    elif method == "none":
        raw = observed.to(torch.float64)
    else:
        raise ValueError(
            "method must be 'effective_num', 'inverse_frequency', or 'none', "
            f"got {method!r}"
        )

    if normalise:
        rows = [
            _normalise_weights_with_cap(raw[k], counts[k], max_weight)
            for k in range(num_classes - 1)
        ]
        weights = torch.stack(rows, dim=0)
    else:
        weights = raw
        if max_weight is not None:
            if max_weight <= 0:
                raise ValueError("max_weight must be positive")
            weights = weights.clamp(max=float(max_weight))

    weights = weights.to(device=device, dtype=dtype)
    if return_counts:
        return weights, counts_long
    return weights


def _validate_transition_inputs(
    transitions: torch.Tensor,
    labels: torch.Tensor,
    transition_weights: torch.Tensor | None,
) -> tuple[int, torch.Tensor]:
    if transitions.ndim != 2:
        raise ValueError(
            f"transitions must have shape (N, K-1), got {tuple(transitions.shape)}"
        )
    if not torch.is_floating_point(transitions):
        raise TypeError("transitions must be a floating-point tensor")
    n, num_boundaries = transitions.shape
    num_classes = num_boundaries + 1
    labels = labels.to(device=transitions.device, dtype=torch.long)
    if labels.ndim != 1 or labels.numel() != n:
        raise ValueError(f"labels must have shape ({n},), got {tuple(labels.shape)}")
    _validate_labels(labels, num_classes)
    if transition_weights is not None and tuple(transition_weights.shape) != (
        num_boundaries,
        2,
    ):
        raise ValueError(
            "transition_weights must have shape "
            f"({num_boundaries}, 2), got {tuple(transition_weights.shape)}"
        )
    return num_classes, labels


def balanced_continuation_nll(
    transitions: torch.Tensor,
    labels: torch.Tensor,
    transition_weights: torch.Tensor | None = None,
    *,
    stop_probabilities: torch.Tensor | None = None,
    log_stop_probabilities: torch.Tensor | None = None,
    eps: float = 1e-7,
    reduction: Reduction = "mean",
) -> torch.Tensor:
    """Balanced continuation likelihood over boundary-specific risk sets.

    ``transitions[n, k]`` must be the conditional probability of advancing
    from rung ``k`` to ``k+1``.  For sample ``n`` this implements exactly

    ``-sum_{k < y_n} w[k,1] log(c[n,k])``
    ``-1[y_n < K-1] w[y_n,0] log(1-c[n,y_n])``.

    The ``mean`` reduction averages these per-image sums, matching the plan's
    sample-level objective.  All logarithms run in float32 (or float64 when the
    input is float64) even under AMP.
    """

    if eps <= 0.0 or eps >= 0.5:
        raise ValueError(f"eps must lie in (0, 0.5), got {eps}")
    if reduction not in ("none", "mean", "sum"):
        raise ValueError(f"unknown reduction {reduction!r}")

    _, labels = _validate_transition_inputs(transitions, labels, transition_weights)
    work_dtype = torch.float64 if transitions.dtype == torch.float64 else torch.float32
    c = transitions.to(dtype=work_dtype).clamp(min=0.0, max=1.0)
    if log_stop_probabilities is not None:
        if log_stop_probabilities.shape != transitions.shape:
            raise ValueError("log_stop_probabilities must match transitions")
        log_stop = log_stop_probabilities.to(dtype=work_dtype)
        if torch.isnan(log_stop).any() or torch.isposinf(log_stop).any():
            raise ValueError("log_stop_probabilities must not contain NaN or +inf")
        # Do not exponentiate or clamp this value: it is the stable lower-tail
        # likelihood precisely when its probability is unrepresentable.
        stop = None
    elif stop_probabilities is None:
        stop = 1.0 - c
    else:
        if stop_probabilities.shape != transitions.shape:
            raise ValueError("stop_probabilities must match transitions")
        stop = stop_probabilities.to(dtype=work_dtype).clamp_min(0.0)
        pair_total = (c + stop).clamp_min(torch.finfo(work_dtype).tiny)
        c = c / pair_total
        stop = stop / pair_total
    num_boundaries = c.shape[1]
    boundaries = torch.arange(num_boundaries, device=c.device).unsqueeze(0)
    y = labels.unsqueeze(1)
    at_risk = y >= boundaries
    advance = y > boundaries

    if transition_weights is None:
        weights = torch.ones((num_boundaries, 2), device=c.device, dtype=work_dtype)
    else:
        weights = transition_weights.to(device=c.device, dtype=work_dtype)
        if not torch.isfinite(weights).all() or (weights < 0).any():
            raise ValueError("transition_weights must be finite and non-negative")

    outcome_weights = torch.where(
        advance,
        weights[:, 1].unsqueeze(0),
        weights[:, 0].unsqueeze(0),
    )
    # Direct lower-tail probabilities avoid the catastrophic cancellation in
    # ``1-c`` when a 12,544-cell continuation score is extremely close to one.
    # Do not clamp at a conventional 1e-7: that would deliberately zero the
    # recovery gradient for a confidently wrong grade-0 image.
    tiny = torch.finfo(work_dtype).tiny
    if log_stop_probabilities is None:
        assert stop is not None
        log_stop = torch.log(stop.clamp_min(tiny))
    log_likelihood = torch.where(advance, torch.log(c.clamp_min(tiny)), log_stop)
    per_sample = -torch.where(
        at_risk,
        outcome_weights * log_likelihood,
        torch.zeros_like(log_likelihood),
    ).sum(dim=1)

    if reduction == "none":
        return per_sample
    if reduction == "sum":
        return per_sample.sum()
    return per_sample.mean()


def witness_js_stability(
    witness_states_a: torch.Tensor,
    witness_states_b: torch.Tensor,
    valid_mask: torch.Tensor | None = None,
    *,
    eps: float = 1e-7,
    reduction: Reduction = "mean",
) -> torch.Tensor:
    """Jensen--Shannon consistency for geometry-aligned local states.

    Inputs are categorical local-state probabilities with shape ``(..., K)``
    from two photometric views that have identical geometry.  The final axis is
    normalised defensively, so small numeric drift from a softmax is harmless.
    ``valid_mask`` must broadcast to the leading dimensions and excludes
    padded/background cells from the mean.
    """

    if witness_states_a.shape != witness_states_b.shape:
        raise ValueError(
            "witness views must have the same shape, got "
            f"{tuple(witness_states_a.shape)} and {tuple(witness_states_b.shape)}"
        )
    if witness_states_a.ndim < 2:
        raise ValueError("witness states require at least a cell and state dimension")
    if reduction not in ("none", "mean", "sum"):
        raise ValueError(f"unknown reduction {reduction!r}")
    if eps <= 0.0:
        raise ValueError("eps must be positive")

    work_dtype = (
        torch.float64
        if witness_states_a.dtype == torch.float64
        and witness_states_b.dtype == torch.float64
        else torch.float32
    )
    p = witness_states_a.to(dtype=work_dtype).clamp_min(0.0)
    q = witness_states_b.to(dtype=work_dtype).clamp_min(0.0)
    p = p / p.sum(dim=-1, keepdim=True).clamp_min(eps)
    q = q / q.sum(dim=-1, keepdim=True).clamp_min(eps)
    midpoint = 0.5 * (p + q)

    # x * log(x / m), with the mathematically correct limit 0*log(0)=0.
    kl_pm = torch.where(p > 0, p * (torch.log(p.clamp_min(eps)) - torch.log(midpoint.clamp_min(eps))), 0.0).sum(dim=-1)
    kl_qm = torch.where(q > 0, q * (torch.log(q.clamp_min(eps)) - torch.log(midpoint.clamp_min(eps))), 0.0).sum(dim=-1)
    js = 0.5 * (kl_pm + kl_qm)

    if valid_mask is not None:
        try:
            mask = torch.broadcast_to(valid_mask.to(device=js.device, dtype=torch.bool), js.shape)
        except RuntimeError as exc:
            raise ValueError(
                f"valid_mask shape {tuple(valid_mask.shape)} cannot broadcast to {tuple(js.shape)}"
            ) from exc
        if reduction == "none":
            return torch.where(mask, js, torch.zeros_like(js))
        if not mask.any():
            return js.sum() * 0.0
        selected = js[mask]
    else:
        selected = js

    if reduction == "none":
        return selected
    if reduction == "sum":
        return selected.sum()
    return selected.mean()


class MosaicLoss(nn.Module):
    """The deliberately small MOSAIC training objective.

    ``projected_transitions`` are always the primary prediction path.  Dense
    transitions are optional only when ``dense_weight == 0``; otherwise their
    absence is an error rather than a silently disabled stabilizer.
    """

    def __init__(
        self,
        num_classes: int,
        transition_weights: torch.Tensor | None = None,
        *,
        dense_weight: float = 0.1,
        stability_weight: float = 0.0,
        eps: float = 1e-7,
    ) -> None:
        super().__init__()
        if num_classes < 2:
            raise ValueError("num_classes must be at least 2")
        if dense_weight < 0.0 or stability_weight < 0.0:
            raise ValueError("loss weights must be non-negative")
        if transition_weights is None:
            transition_weights = torch.ones(num_classes - 1, 2)
        if tuple(transition_weights.shape) != (num_classes - 1, 2):
            raise ValueError(
                f"transition_weights must have shape ({num_classes - 1}, 2)"
            )
        self.num_classes = int(num_classes)
        self.dense_weight = float(dense_weight)
        self.stability_weight = float(stability_weight)
        self.eps = float(eps)
        self.register_buffer("transition_weights", transition_weights.detach().float().clone())

    @classmethod
    def from_training_labels(
        cls,
        labels: Sequence[int] | torch.Tensor,
        num_classes: int,
        *,
        weight_method: WeightMethod = "effective_num",
        weight_beta: float = 0.999,
        max_transition_weight: float | None = 10.0,
        dense_weight: float = 0.1,
        stability_weight: float = 0.0,
        eps: float = 1e-7,
    ) -> "MosaicLoss":
        """Build the criterion and its persistent weights from one train fold."""

        weights = build_at_risk_transition_weights(
            labels,
            num_classes,
            method=weight_method,
            beta=weight_beta,
            max_weight=max_transition_weight,
        )
        return cls(
            num_classes,
            weights,
            dense_weight=dense_weight,
            stability_weight=stability_weight,
            eps=eps,
        )

    def forward(
        self,
        projected_transitions: torch.Tensor,
        labels: torch.Tensor,
        *,
        projected_stop_probabilities: torch.Tensor | None = None,
        projected_log_stop_probabilities: torch.Tensor | None = None,
        dense_transitions: torch.Tensor | None = None,
        dense_stop_probabilities: torch.Tensor | None = None,
        dense_log_stop_probabilities: torch.Tensor | None = None,
        witness_states_a: torch.Tensor | None = None,
        witness_states_b: torch.Tensor | None = None,
        valid_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if projected_transitions.shape[-1] != self.num_classes - 1:
            raise ValueError(
                f"expected {self.num_classes - 1} transitions, got "
                f"{projected_transitions.shape[-1]}"
            )

        projected = balanced_continuation_nll(
            projected_transitions,
            labels,
            self.transition_weights,
            stop_probabilities=projected_stop_probabilities,
            log_stop_probabilities=projected_log_stop_probabilities,
            eps=self.eps,
        )

        zero = projected_transitions.float().sum() * 0.0
        dense = zero
        if self.dense_weight > 0.0:
            if dense_transitions is None:
                raise ValueError("dense_transitions are required when dense_weight > 0")
            dense = balanced_continuation_nll(
                dense_transitions,
                labels,
                self.transition_weights,
                stop_probabilities=dense_stop_probabilities,
                log_stop_probabilities=dense_log_stop_probabilities,
                eps=self.eps,
            )

        stability = zero
        if self.stability_weight > 0.0:
            if witness_states_a is None or witness_states_b is None:
                raise ValueError(
                    "both witness-state views are required when stability_weight > 0"
                )
            stability = witness_js_stability(
                witness_states_a,
                witness_states_b,
                valid_mask,
                eps=self.eps,
            )

        total = projected + self.dense_weight * dense + self.stability_weight * stability
        # Keep diagnostics as detached device scalars.  Converting each one to
        # a Python float here would force a dozen CUDA synchronizations for
        # every minibatch; the trainer transfers them once at epoch end.
        diagnostics: dict[str, torch.Tensor] = {
            "loss_total": total.detach(),
            "loss_ccl": projected.detach(),
            "loss_dense": dense.detach(),
            "loss_stability": stability.detach(),
            "mean_projected_transition": projected_transitions.detach().float().mean(),
        }
        if dense_transitions is not None:
            diagnostics["mean_dense_transition"] = (
                dense_transitions.detach().float().mean()
            )

        # Flat scalar diagnostics remain compatible with the repository's
        # existing metric accumulation and expose late-boundary failure modes.
        labels_on_device = labels.to(projected_transitions.device)
        for k in range(self.num_classes - 1):
            risk = labels_on_device >= k
            risk_count = risk.sum().detach().float()
            advance_count = ((labels_on_device > k) & risk).sum().detach().float()
            diagnostics[f"at_risk_boundary_{k}"] = risk_count
            diagnostics[f"advance_rate_boundary_{k}"] = (
                advance_count / risk_count.clamp_min(1.0)
            )

        return total, diagnostics


__all__ = [
    "MosaicLoss",
    "balanced_continuation_nll",
    "build_at_risk_transition_weights",
    "transition_outcome_counts",
    "witness_js_stability",
]
