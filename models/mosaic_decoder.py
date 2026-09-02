"""Proof-only decision rules for MOSAIC ordinal continuation circuits.

The MOSAIC training loss may weight the two outcomes at each ordinal boundary
differently.  In that case the learned continuation probability is the
cost-sensitive posterior rather than the unweighted posterior.  This module
provides the analytic inverse of that distortion and converts the resulting
conditional decisions into a class distribution.

Importantly, every quantity here is a deterministic function of the MOSAIC
proof transitions.  There is no feature input, learned parameter, validation
fit, or auxiliary classification path.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


PROOF_DECISION_RULES = (
    "rounded_expected",
    "class_map",
    "posterior_median",
    "deweighted_mean_round",
    "deweighted_class_map",
    "deweighted_posterior_median",
)


@dataclass(frozen=True)
class DeweightedContinuation:
    """Unweighted boundary probabilities recovered from weighted training."""

    transitions: torch.Tensor
    stop_probabilities: torch.Tensor
    log_transitions: torch.Tensor
    log_stop_probabilities: torch.Tensor


@dataclass(frozen=True)
class ProofOnlyDecisionBundle:
    """Auditable grade decisions derived exclusively from proof transitions.

    ``raw_mean_round`` reproduces MOSAIC's original point decision.
    ``raw_argmax`` isolates the effect of using the accuracy-optimal decision
    rule without probability correction.  The deweighted variants additionally
    remove the known outcome-weight distortion. Posterior medians are included
    as pre-specified absolute-error decisions, not fitted alternatives.

    The posterior median is the number of cumulative continuation
    probabilities greater than or equal to ``0.5``.  Thus an exact ``0.5``
    tie advances to the higher grade (the upper-median convention).
    """

    raw_cumulative_probabilities: torch.Tensor
    raw_class_probabilities: torch.Tensor
    deweighted_transitions: torch.Tensor
    deweighted_stop_probabilities: torch.Tensor
    deweighted_log_stop_probabilities: torch.Tensor
    deweighted_cumulative_probabilities: torch.Tensor
    deweighted_class_probabilities: torch.Tensor
    raw_expected_grade: torch.Tensor
    deweighted_expected_grade: torch.Tensor
    raw_mean_round: torch.Tensor
    raw_argmax: torch.Tensor
    raw_posterior_median: torch.Tensor
    deweighted_mean_round: torch.Tensor
    deweighted_argmax: torch.Tensor
    deweighted_posterior_median: torch.Tensor


def _working_dtype(*tensors: torch.Tensor) -> torch.dtype:
    dtype = torch.float32
    for tensor in tensors:
        if torch.is_floating_point(tensor):
            dtype = torch.promote_types(dtype, tensor.dtype)
    return dtype


def _validate_boundary_inputs(
    transitions: torch.Tensor,
    log_stop_probabilities: torch.Tensor,
) -> None:
    if not torch.is_floating_point(transitions):
        raise TypeError("transitions must be a floating-point tensor")
    if not torch.is_floating_point(log_stop_probabilities):
        raise TypeError("log_stop_probabilities must be a floating-point tensor")
    if transitions.ndim < 1 or transitions.shape[-1] < 1:
        raise ValueError("at least one continuation boundary is required")
    if log_stop_probabilities.shape != transitions.shape:
        raise ValueError("log_stop_probabilities must match transitions")
    if not bool(torch.isfinite(transitions).all()):
        raise ValueError("transitions must contain only finite values")
    if bool(((transitions < 0.0) | (transitions > 1.0)).any()):
        raise ValueError("transitions must lie in [0, 1]")
    if bool(torch.isnan(log_stop_probabilities).any()) or bool(
        torch.isposinf(log_stop_probabilities).any()
    ):
        raise ValueError("log_stop_probabilities must not contain NaN or +inf")


def _normalised_boundary_logs(
    transitions: torch.Tensor,
    log_stop_probabilities: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return normalized ``(log advance, log stop)`` boundary pairs."""

    _validate_boundary_inputs(transitions, log_stop_probabilities)
    dtype = _working_dtype(transitions, log_stop_probabilities)
    continuation = transitions.to(dtype=dtype)
    log_stop = log_stop_probabilities.to(
        device=transitions.device,
        dtype=dtype,
    )
    # ``torch.log(0) == -inf`` is the desired exact endpoint semantics.
    log_continuation = torch.log(continuation)
    log_total = torch.logaddexp(log_continuation, log_stop)
    if bool(torch.isneginf(log_total).any()):
        raise ValueError(
            "each boundary needs non-zero advance or stop probability"
        )
    return log_continuation - log_total, log_stop - log_total


def cascade_class_probabilities(
    transitions: torch.Tensor,
    log_stop_probabilities: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert conditional proof transitions into cumulative and class laws.

    The calculation is performed in log space, so exact endpoints and stop
    probabilities far below the floating-point normal range remain well
    defined.  For ``B`` boundaries, the returned shapes are ``(..., B)`` and
    ``(..., B + 1)`` respectively.
    """

    log_continue, log_stop = _normalised_boundary_logs(
        transitions,
        log_stop_probabilities,
    )
    log_cumulative = torch.cumsum(log_continue, dim=-1)

    first = log_stop[..., :1]
    if transitions.shape[-1] > 1:
        middle = log_cumulative[..., :-1] + log_stop[..., 1:]
        log_classes = torch.cat((first, middle, log_cumulative[..., -1:]), dim=-1)
    else:
        log_classes = torch.cat((first, log_cumulative), dim=-1)

    # The cascade is analytically normalized.  A final log normalization only
    # removes accumulated roundoff and is harmless at exact 0/1 endpoints.
    log_classes = log_classes - torch.logsumexp(log_classes, dim=-1, keepdim=True)
    return log_cumulative.exp(), log_classes.exp()


def inverse_outcome_weighting(
    transitions: torch.Tensor,
    log_stop_probabilities: torch.Tensor,
    outcome_weights: torch.Tensor,
) -> DeweightedContinuation:
    r"""Analytically recover unweighted boundary posteriors.

    ``outcome_weights[k] = (w_stop, w_advance)`` must match the weights used in
    the boundary likelihood.  If the weighted optimum is

    .. math::

       q = \frac{w_{advance}p}
                {w_{advance}p + w_{stop}(1-p)},

    then the natural posterior is recovered exactly as

    .. math::

       p = \frac{w_{stop}q}
                {w_{stop}q + w_{advance}(1-q)}.

    Row-wise scaling of the weights cancels.  Strictly positive weights are
    required because an outcome assigned zero training weight is not
    identifiable and therefore cannot be inverted honestly.
    """

    log_continue, log_stop = _normalised_boundary_logs(
        transitions,
        log_stop_probabilities,
    )
    boundaries = transitions.shape[-1]
    if outcome_weights.ndim != 2 or tuple(outcome_weights.shape) != (boundaries, 2):
        raise ValueError(
            "outcome_weights must have shape "
            f"({boundaries}, 2) ordered as [stop, advance]; got "
            f"{tuple(outcome_weights.shape)}"
        )
    if not torch.is_floating_point(outcome_weights):
        raise TypeError("outcome_weights must be a floating-point tensor")
    if not bool(torch.isfinite(outcome_weights).all()):
        raise ValueError("outcome_weights must contain only finite values")
    if bool((outcome_weights <= 0.0).any()):
        raise ValueError(
            "outcome_weights must be strictly positive for analytic inversion"
        )

    dtype = _working_dtype(log_continue, log_stop, outcome_weights)
    log_continue = log_continue.to(dtype=dtype)
    log_stop = log_stop.to(dtype=dtype)
    weights = outcome_weights.to(device=transitions.device, dtype=dtype)

    # Recover p from the cost-sensitive q.  Stop weight multiplies the advance
    # numerator and advance weight multiplies the stop numerator.
    log_advance_numerator = log_continue + weights[:, 0].log()
    log_stop_numerator = log_stop + weights[:, 1].log()
    log_total = torch.logaddexp(log_advance_numerator, log_stop_numerator)
    corrected_log_continue = log_advance_numerator - log_total
    corrected_log_stop = log_stop_numerator - log_total
    return DeweightedContinuation(
        transitions=corrected_log_continue.exp(),
        stop_probabilities=corrected_log_stop.exp(),
        log_transitions=corrected_log_continue,
        log_stop_probabilities=corrected_log_stop,
    )


def proof_only_decisions(
    transitions: torch.Tensor,
    log_stop_probabilities: torch.Tensor,
    outcome_weights: torch.Tensor,
) -> ProofOnlyDecisionBundle:
    """Return all pre-specified decoder decisions from one proof circuit."""

    raw_cumulative, raw_classes = cascade_class_probabilities(
        transitions,
        log_stop_probabilities,
    )
    corrected = inverse_outcome_weighting(
        transitions,
        log_stop_probabilities,
        outcome_weights,
    )
    corrected_cumulative, corrected_classes = cascade_class_probabilities(
        corrected.transitions,
        corrected.log_stop_probabilities,
    )

    axis = torch.arange(
        raw_classes.shape[-1],
        device=raw_classes.device,
        dtype=raw_classes.dtype,
    )
    raw_expected = (raw_classes * axis).sum(dim=-1)
    corrected_expected = (corrected_classes * axis).sum(dim=-1)
    max_grade = raw_classes.shape[-1] - 1
    return ProofOnlyDecisionBundle(
        raw_cumulative_probabilities=raw_cumulative,
        raw_class_probabilities=raw_classes,
        deweighted_transitions=corrected.transitions,
        deweighted_stop_probabilities=corrected.stop_probabilities,
        deweighted_log_stop_probabilities=corrected.log_stop_probabilities,
        deweighted_cumulative_probabilities=corrected_cumulative,
        deweighted_class_probabilities=corrected_classes,
        raw_expected_grade=raw_expected,
        deweighted_expected_grade=corrected_expected,
        raw_mean_round=raw_expected.round().long().clamp(0, max_grade),
        raw_argmax=raw_classes.argmax(dim=-1),
        raw_posterior_median=(raw_cumulative >= 0.5).sum(dim=-1),
        deweighted_mean_round=corrected_expected.round()
        .long()
        .clamp(0, max_grade),
        deweighted_argmax=corrected_classes.argmax(dim=-1),
        deweighted_posterior_median=(corrected_cumulative >= 0.5).sum(dim=-1),
    )


def decision_rule_outputs(
    decision: ProofOnlyDecisionBundle,
) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    """Map every pre-specified rule to its prediction and cumulative law."""

    return {
        "rounded_expected": (
            decision.raw_mean_round,
            decision.raw_cumulative_probabilities,
        ),
        "class_map": (
            decision.raw_argmax,
            decision.raw_cumulative_probabilities,
        ),
        "posterior_median": (
            decision.raw_posterior_median,
            decision.raw_cumulative_probabilities,
        ),
        "deweighted_mean_round": (
            decision.deweighted_mean_round,
            decision.deweighted_cumulative_probabilities,
        ),
        "deweighted_class_map": (
            decision.deweighted_argmax,
            decision.deweighted_cumulative_probabilities,
        ),
        "deweighted_posterior_median": (
            decision.deweighted_posterior_median,
            decision.deweighted_cumulative_probabilities,
        ),
    }


__all__ = [
    "PROOF_DECISION_RULES",
    "DeweightedContinuation",
    "ProofOnlyDecisionBundle",
    "cascade_class_probabilities",
    "decision_rule_outputs",
    "inverse_outcome_weighting",
    "proof_only_decisions",
]
