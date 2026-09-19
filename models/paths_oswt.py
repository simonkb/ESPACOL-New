"""PATHS-OSWT: ordinal-shell warranted posterior transport.

This module deliberately lives beside, rather than replacing, the audited
SAPT implementation in :mod:`models.paths`.  It keeps the ORIGIN-v3 image to
posterior path intact and compiles a second, replayable spatial ledger from
the *exclusive* ordinal shells implied by ORIGIN's nested cumulative atoms.

For ``K-1`` nested cumulative atoms ``u_0 >= ... >= u_(K-2)``, the exclusive
grade shells are the exact simplex

``v_0 = 1-u_0``, ``v_j = u_(j-1)-u_j``, ``v_(K-1) = u_(K-2)``.

At every ordinal cut the stored shells are first partitioned into complementary
low- and high-side ledgers.  Each side is divided by its original per-scale,
per-boundary mass before the nonlinear transform, producing a spatial share
distribution.  Each share is passed through the tangent-removed PGF transform and
multiplied by its original partition mass.  This amount--concentration product
removes inverse-cell-count shrinkage, remains in ``[0,1]``, and suppresses a
numerically tiny but accidentally focal partition.  Both factors are fixed by
the original forward pass and are never renormalized after intervention.  The
resulting boundary warrants define bounded increments to adjacent posterior
log odds.  No classifier or unconstrained residual bypasses that computation.

Cell deletion masks the *stored* shell and ORIGIN-rate ledgers, then replays
both decoders.  In particular, deletion never recomputes ``v_0`` as
``1-u_0``; doing so would silently turn absence created by intervention into
new grade-0 evidence.

The internal atoms and warrants are computational evidence, not lesion
counts, diagnoses, or causal pixel effects without independent validation.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import math
from typing import Dict, Mapping, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from .origin import (
    OriginInterventionOutput,
    OriginModel,
    OriginOutput,
    OriginScaleEvidence,
    replay_without as replay_origin_without,
)
from .paths import (
    PathsContinuationDistribution,
    continuation_logits_from_log_probs,
    decode_continuation_logits,
    tangent_removed_pgf_probe,
)


_OSWT_DTYPE = torch.float64
_DEFAULT_PROBES = (0.05, 0.20, 0.50, 0.80)


def _require_finite(tensor: torch.Tensor, name: str) -> None:
    if not bool(torch.isfinite(tensor).all()):
        nan = int(torch.isnan(tensor).sum().item())
        pos = int(torch.isposinf(tensor).sum().item())
        neg = int(torch.isneginf(tensor).sum().item())
        raise FloatingPointError(
            f"{name} contains non-finite values "
            f"(nan={nan}, +inf={pos}, -inf={neg})"
        )


def _canonical_probe_values(values: Sequence[float]) -> Tuple[float, ...]:
    probes = tuple(float(value) for value in values)
    if not probes:
        raise ValueError("OSWT requires at least one PGF probe")
    if not all(math.isfinite(value) and 0.0 < value < 1.0 for value in probes):
        raise ValueError("every OSWT probe must lie strictly in (0, 1)")
    if tuple(sorted(probes)) != probes or len(set(probes)) != len(probes):
        raise ValueError("OSWT probes must be unique and strictly increasing")
    return probes


def _inverse_bounded_sigmoid(
    value: float,
    *,
    lower: float,
    upper: float,
    name: str,
) -> float:
    value = float(value)
    lower = float(lower)
    upper = float(upper)
    if not all(math.isfinite(item) for item in (value, lower, upper)):
        raise ValueError(f"{name} bounds and initialization must be finite")
    if not lower < value < upper:
        raise ValueError(
            f"{name} initialization must lie strictly inside ({lower:g}, {upper:g})"
        )
    probability = (value - lower) / (upper - lower)
    return math.log(probability) - math.log1p(-probability)


def _canonical_unit_interval(tensor: torch.Tensor, *, name: str) -> torch.Tensor:
    if not torch.is_tensor(tensor) or not tensor.is_floating_point():
        raise TypeError(f"{name} must be a floating-point tensor")
    _require_finite(tensor, name)
    tolerance = 16.0 * torch.finfo(tensor.dtype).eps
    outside = (tensor < -tolerance) | (tensor > 1.0 + tolerance)
    if bool(outside.any()):
        minimum = float(tensor.detach().amin().item())
        maximum = float(tensor.detach().amax().item())
        raise ValueError(
            f"{name} must lie in [0,1] up to roundoff; "
            f"observed [{minimum:g}, {maximum:g}]"
        )
    return tensor.clamp(0.0, 1.0)


def ordinal_shells_from_cumulative_atoms(
    cumulative_atoms: torch.Tensor,
    valid_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    r"""Compile nested cumulative atoms into an exclusive grade simplex.

    Parameters
    ----------
    cumulative_atoms:
        Tensor ``(N,K-1,H,W)`` with values in ``[0,1]`` and non-increasing
        channels.
    valid_mask:
        Optional ``(N,H,W)`` mask.  Invalid cells receive an all-zero shell
        ledger rather than a synthetic grade-0 shell.
    """

    if not torch.is_tensor(cumulative_atoms) or not cumulative_atoms.is_floating_point():
        raise TypeError("cumulative_atoms must be a floating-point tensor")
    if cumulative_atoms.ndim != 4 or cumulative_atoms.shape[1] < 1:
        raise ValueError("cumulative_atoms must have shape (N,K-1,H,W)")
    atoms = _canonical_unit_interval(
        cumulative_atoms,
        name="OSWT normalized cumulative atoms",
    ).to(dtype=_OSWT_DTYPE)
    tolerance = 32.0 * torch.finfo(atoms.dtype).eps
    if atoms.shape[1] > 1:
        nesting_gap = atoms[:, 1:] - atoms[:, :-1]
        if bool((nesting_gap > tolerance).any()):
            maximum = float(nesting_gap.detach().amax().item())
            raise ValueError(
                "OSWT cumulative atoms are not nested "
                f"(maximum u_(k+1)-u_k={maximum:g})"
            )

    first = 1.0 - atoms[:, :1]
    middle = atoms[:, :-1] - atoms[:, 1:]
    last = atoms[:, -1:]
    shells = torch.cat((first, middle, last), dim=1)
    shells = _canonical_unit_interval(shells, name="OSWT exclusive ordinal shells")

    if valid_mask is None:
        valid = torch.ones(
            atoms.shape[0], atoms.shape[2], atoms.shape[3],
            dtype=torch.bool, device=atoms.device,
        )
    else:
        valid = torch.as_tensor(valid_mask, device=atoms.device).bool()
        if tuple(valid.shape) != (atoms.shape[0], atoms.shape[2], atoms.shape[3]):
            raise ValueError("valid_mask shape does not match cumulative_atoms")

    shell_sum = shells.sum(dim=1)
    simplex_error = (shell_sum[valid] - 1.0).abs()
    if simplex_error.numel() and bool((simplex_error > tolerance).any()):
        raise FloatingPointError(
            "OSWT shell compilation did not conserve unit mass "
            f"(maximum error={float(simplex_error.max().item()):g})"
        )
    return torch.where(valid[:, None], shells, torch.zeros_like(shells))


def canonical_boundary_scale_weights(
    boundary_scale_simplex: torch.Tensor,
) -> torch.Tensor:
    """Validate and FP64-canonicalize V3's native boundary-scale simplex.

    Both sides of boundary ``k`` must use the *same* V3 scale distribution
    ``w[:, k]``.  Mixing a different scale distribution on each side would
    allow the contrast to change because of scale calibration rather than
    ordinal evidence.
    """

    if not torch.is_tensor(boundary_scale_simplex) or not (
        boundary_scale_simplex.is_floating_point()
    ):
        raise TypeError("boundary_scale_simplex must be floating point")
    if boundary_scale_simplex.ndim != 2 or boundary_scale_simplex.shape[1] < 1:
        raise ValueError("boundary_scale_simplex must have shape (S,K-1)")
    source_dtype = boundary_scale_simplex.dtype
    source_tolerance = 64.0 * torch.finfo(source_dtype).eps
    weights = boundary_scale_simplex.to(dtype=_OSWT_DTYPE)
    _require_finite(weights, "V3 boundary scale simplex")
    if bool((weights < -source_tolerance).any()):
        raise ValueError("V3 boundary scale simplex contains negative weights")
    column_mass = weights.sum(dim=0)
    boundary_error = (column_mass - 1.0).abs()
    if bool((boundary_error > source_tolerance).any()) or bool(
        (column_mass <= 0.0).any()
    ):
        raise ValueError(
            "V3 boundary scale weights must sum to one per boundary "
            f"(maximum error={float(boundary_error.max().item()):g})"
        )
    # The source is normally float32.  Normalize once in FP64 so subsequent
    # exact-ledger audits do not confuse source roundoff with a violation.
    return weights / column_mass


@dataclass
class PathsOSWTRefinementDistribution:
    """Exact trace of shell-warranted adjacent log-odds refinement."""

    class_probs: torch.Tensor
    log_class_probs: torch.Tensor
    cumulative_probs: torch.Tensor
    expected_grade: torch.Tensor
    posterior_median: torch.Tensor
    class_map: torch.Tensor
    continuation_logits: torch.Tensor
    continuation_probs: torch.Tensor
    stop_probs: torch.Tensor
    base_adjacent_log_odds: torch.Tensor
    refined_adjacent_log_odds: torch.Tensor
    left_boundary_warrant: torch.Tensor
    right_boundary_warrant: torch.Tensor
    boundary_coverage: torch.Tensor
    shell_contrast: torch.Tensor
    base_boundary_probability: torch.Tensor
    boundary_gate: torch.Tensor
    log_odds_increment: torch.Tensor
    grade_log_correction: torch.Tensor
    signed_boundary_flow: torch.Tensor
    wasserstein1: torch.Tensor

    @property
    def predicted_grade(self) -> torch.Tensor:
        return self.class_map

    @property
    def monotone_grade(self) -> torch.Tensor:
        return self.posterior_median

    @property
    def final_class_probs(self) -> torch.Tensor:
        return self.class_probs

    @property
    def contrast(self) -> torch.Tensor:
        return self.shell_contrast

    @property
    def transport_increment(self) -> torch.Tensor:
        return self.log_odds_increment

    @property
    def flows(self) -> torch.Tensor:
        return self.signed_boundary_flow

    @property
    def w1(self) -> torch.Tensor:
        return self.wasserstein1


def _boundary_parameter(
    value: torch.Tensor,
    *,
    num_boundaries: int,
    device: torch.device,
    name: str,
    positive: bool,
) -> torch.Tensor:
    if not torch.is_tensor(value) or not value.is_floating_point():
        raise TypeError(f"{name} must be a floating-point tensor")
    result = value.to(device=device, dtype=_OSWT_DTYPE)
    _require_finite(result, name)
    if tuple(result.shape) != (num_boundaries,):
        raise ValueError(
            f"{name} must have shape ({num_boundaries},), "
            f"observed {tuple(result.shape)}"
        )
    if positive and bool((result <= 0.0).any()):
        raise ValueError(f"{name} must be strictly positive")
    return result


def apply_shell_warranted_transport(
    base_class_probs: torch.Tensor,
    left_boundary_warrant: torch.Tensor,
    right_boundary_warrant: torch.Tensor,
    beta: torch.Tensor,
    tau: torch.Tensor,
    *,
    strength: float = 1.0,
    rho: Optional[float] = None,
    risk_gate_enabled: bool = True,
    probability_tolerance: float = 5e-12,
) -> PathsOSWTRefinementDistribution:
    r"""Update every adjacent grade log-odds by a warranted increment.

    At ordinal boundary ``k``, the left warrant measures the complementary
    shell partition ``Y<=k`` and the right warrant measures ``Y>k``.  Both
    were compiled with the same PGF functional and the same native V3
    boundary-scale simplex.  Their signed, coverage-aware contrast is

    ``D_k=(R_k-L_k)/(R_k+L_k+tau_k)``.

    It is bounded in ``(-1,1)`` and exactly zero when no stored warrant
    remains.  The definition is invariant to the number of ordinal grades.

    Let ``q_k=P_base(Y>k)``.  The cut-aligned uncertainty gate is

    ``G_k=4*q_k*(1-q_k)`` (or one for the registered ungated control), and

    ``T_k=rho*beta_k*G_k*D_k``.

    Adding ``T_k`` to adjacent log odds has a unique normalized categorical
    solution.  Prefix sums of the increments construct it without solving
    a numerical system.  There is no alternate classifier output.
    """

    if not torch.is_tensor(base_class_probs) or not base_class_probs.is_floating_point():
        raise TypeError("base_class_probs must be a floating-point tensor")
    if base_class_probs.ndim < 2 or base_class_probs.shape[-1] < 2:
        raise ValueError("base_class_probs must have shape (...,K), K>=2")
    if rho is not None:
        strength = float(rho)
    if not math.isfinite(strength) or float(strength) < 0.0:
        raise ValueError("strength/rho must be finite and non-negative")
    if not math.isfinite(probability_tolerance) or probability_tolerance < 0.0:
        raise ValueError("probability_tolerance must be finite and non-negative")

    tolerance = float(probability_tolerance)
    source_probability_tolerance = max(
        tolerance,
        64.0 * torch.finfo(base_class_probs.dtype).eps,
    )
    probabilities = base_class_probs.to(dtype=_OSWT_DTYPE)
    _require_finite(probabilities, "OSWT base class probabilities")
    if bool((probabilities <= 0.0).any()):
        raise ValueError("OSWT requires strictly positive base class probabilities")
    row_mass = probabilities.sum(dim=-1, keepdim=True)
    row_error = (row_mass.squeeze(-1) - 1.0).abs()
    if bool((row_error > source_probability_tolerance).any()):
        raise ValueError(
            "base_class_probs must sum to one "
            f"(maximum error={float(row_error.max().item()):g})"
        )
    probabilities = probabilities / row_mass

    num_grades = probabilities.shape[-1]
    num_boundaries = num_grades - 1
    expected_warrant_shape = probabilities.shape[:-1] + (num_boundaries,)

    def canonical_warrant(value: torch.Tensor, name: str) -> torch.Tensor:
        if not torch.is_tensor(value) or not value.is_floating_point():
            raise TypeError(f"{name} must be a floating-point tensor")
        result = _canonical_unit_interval(
            value.to(device=probabilities.device), name=name
        ).to(dtype=_OSWT_DTYPE)
        if tuple(result.shape) != tuple(expected_warrant_shape):
            raise ValueError(
                f"{name} must have shape {tuple(expected_warrant_shape)}, "
                f"observed {tuple(result.shape)}"
            )
        return result

    left = canonical_warrant(
        left_boundary_warrant, "OSWT left-boundary warrants"
    )
    right = canonical_warrant(
        right_boundary_warrant, "OSWT right-boundary warrants"
    )
    beta = _boundary_parameter(
        beta, num_boundaries=num_boundaries, device=probabilities.device,
        name="OSWT beta", positive=True,
    )
    tau = _boundary_parameter(
        tau, num_boundaries=num_boundaries, device=probabilities.device,
        name="OSWT tau", positive=True,
    )
    total_warrant = left + right
    coverage = total_warrant / (total_warrant + tau)
    contrast = (right - left) / (total_warrant + tau)
    # q_k is the probability mass to the right of ordinal cut k.  Unlike an
    # adjacent-class mass gate, 4*q*(1-q) is aligned with the exact cut whose
    # log odds are changed: it is maximal at an uncertain cut, symmetric, and
    # tends to zero when the base posterior has already resolved that cut.
    boundary_probability = torch.flip(
        torch.cumsum(torch.flip(probabilities[..., 1:], dims=(-1,)), dim=-1),
        dims=(-1,),
    )
    uncertainty_gate = 4.0 * boundary_probability * (1.0 - boundary_probability)
    gate = uncertainty_gate if risk_gate_enabled else torch.ones_like(uncertainty_gate)
    increment = float(strength) * beta * gate * contrast
    _require_finite(contrast, "OSWT shell contrast")
    _require_finite(boundary_probability, "OSWT base boundary probabilities")
    _require_finite(gate, "OSWT boundary gate")
    _require_finite(increment, "OSWT adjacent log-odds increments")

    base_log = probabilities.log()
    base_adjacent = base_log[..., 1:] - base_log[..., :-1]
    zero = torch.zeros_like(increment[..., :1])
    grade_correction = torch.cat(
        (zero, torch.cumsum(increment, dim=-1)), dim=-1
    )
    proposed_log = F.log_softmax(base_log + grade_correction, dim=-1)
    proposed_probs = proposed_log.exp()
    _require_finite(proposed_log, "OSWT normalized posterior log probabilities")

    # Canonical continuation fields are needed by the strictly proper ordinal
    # loss.  Factorization must replay the same categorical posterior.
    continuation_logits = continuation_logits_from_log_probs(proposed_log)
    decoded: PathsContinuationDistribution = decode_continuation_logits(
        continuation_logits, probability_tolerance=tolerance
    )
    factorization_error = (decoded.class_probs - proposed_probs).abs()
    factorization_tolerance = tolerance + tolerance * proposed_probs.abs()
    if bool((factorization_error > factorization_tolerance).any()):
        raise FloatingPointError(
            "OSWT continuation factorization did not replay the posterior "
            f"(maximum error={float(factorization_error.max().item()):g})"
        )

    refined_adjacent = (
        decoded.log_class_probs[..., 1:] - decoded.log_class_probs[..., :-1]
    )
    odds_error = (refined_adjacent - (base_adjacent + increment)).abs()
    odds_tolerance = 8.0 * tolerance + 8.0 * tolerance * refined_adjacent.abs()
    if bool((odds_error > odds_tolerance).any()):
        raise FloatingPointError(
            "OSWT adjacent log-odds update failed exact replay "
            f"(maximum error={float(odds_error.max().item()):g})"
        )

    # This is the unique signed flow across ordinal cuts in one dimension.
    flow = torch.cumsum(probabilities - decoded.class_probs, dim=-1)[..., :-1]
    reconstructed_delta = torch.cat(
        (
            -flow[..., :1],
            flow[..., :-1] - flow[..., 1:],
            flow[..., -1:],
        ),
        dim=-1,
    )
    flow_error = (decoded.class_probs - probabilities - reconstructed_delta).abs()
    if bool((flow_error > factorization_tolerance).any()):
        raise FloatingPointError(
            "OSWT signed boundary flows do not replay the posterior "
            f"(maximum error={float(flow_error.max().item()):g})"
        )
    wasserstein = flow.abs().sum(dim=-1)
    _require_finite(flow, "OSWT signed boundary flows")
    _require_finite(wasserstein, "OSWT ordinal Wasserstein-1 transport")

    return PathsOSWTRefinementDistribution(
        class_probs=decoded.class_probs,
        log_class_probs=decoded.log_class_probs,
        cumulative_probs=decoded.cumulative_probs,
        expected_grade=decoded.expected_grade,
        posterior_median=decoded.posterior_median,
        class_map=decoded.class_map,
        continuation_logits=decoded.continuation_logits,
        continuation_probs=decoded.continuation_probs,
        stop_probs=decoded.stop_probs,
        base_adjacent_log_odds=base_adjacent,
        refined_adjacent_log_odds=refined_adjacent,
        left_boundary_warrant=left,
        right_boundary_warrant=right,
        boundary_coverage=coverage,
        shell_contrast=contrast,
        base_boundary_probability=boundary_probability,
        boundary_gate=gate,
        log_odds_increment=increment,
        grade_log_correction=grade_correction,
        signed_boundary_flow=flow,
        wasserstein1=wasserstein,
    )


def apply_shell_warranted_transport_from_log_probs(
    base_log_probs: torch.Tensor,
    left_boundary_warrant: torch.Tensor,
    right_boundary_warrant: torch.Tensor,
    beta: torch.Tensor,
    tau: torch.Tensor,
    *,
    strength: float = 1.0,
    rho: Optional[float] = None,
    risk_gate_enabled: bool = True,
    probability_tolerance: float = 5e-12,
) -> PathsOSWTRefinementDistribution:
    """Log-probability entry point for :func:`apply_shell_warranted_transport`.

    The input must already be a normalized log categorical distribution; it
    is not treated as an arbitrary classifier-logit bypass.
    """

    if not torch.is_tensor(base_log_probs) or not base_log_probs.is_floating_point():
        raise TypeError("base_log_probs must be a floating-point tensor")
    if not math.isfinite(probability_tolerance) or probability_tolerance < 0.0:
        raise ValueError("probability_tolerance must be finite and non-negative")
    logs = base_log_probs.to(dtype=_OSWT_DTYPE)
    _require_finite(logs, "OSWT base log probabilities")
    log_normalizer = torch.logsumexp(logs, dim=-1, keepdim=True)
    log_normalization_error = log_normalizer.squeeze(-1).abs()
    source_tolerance = max(
        float(probability_tolerance),
        64.0 * torch.finfo(base_log_probs.dtype).eps,
    )
    if bool((log_normalization_error > source_tolerance).any()):
        raise ValueError(
            "base_log_probs must be normalized "
            f"(maximum logsumexp error={float(log_normalization_error.max().item()):g})"
        )
    logs = logs - log_normalizer
    return apply_shell_warranted_transport(
        logs.exp(),
        left_boundary_warrant,
        right_boundary_warrant,
        beta,
        tau,
        strength=strength,
        rho=rho,
        risk_gate_enabled=risk_gate_enabled,
        probability_tolerance=probability_tolerance,
    )


@dataclass
class PathsOSWTScaleEvidence:
    """Frozen complementary-partition warrant ledger for one V3 scale."""

    name: str
    normalized_cumulative_atoms: torch.Tensor
    ordinal_shells: torch.Tensor
    original_valid_mask: torch.Tensor
    active_mask: torch.Tensor
    original_valid_count: torch.Tensor
    local_left_partition_mass_map: torch.Tensor
    local_right_partition_mass_map: torch.Tensor
    baseline_left_mass: torch.Tensor
    baseline_right_mass: torch.Tensor
    surviving_left_mass: torch.Tensor
    surviving_right_mass: torch.Tensor
    local_left_share_map: torch.Tensor
    local_right_share_map: torch.Tensor
    local_left_phi_spectrum: torch.Tensor
    local_right_phi_spectrum: torch.Tensor
    local_left_warrant_spectrum: torch.Tensor
    local_right_warrant_spectrum: torch.Tensor
    left_warrant_spectrum: torch.Tensor
    right_warrant_spectrum: torch.Tensor
    mixed_left_warrant: torch.Tensor
    mixed_right_warrant: torch.Tensor
    local_left_boundary_warrant_map: torch.Tensor
    local_right_boundary_warrant_map: torch.Tensor
    metadata: object

    @property
    def local_partition_mass_maps(self) -> tuple[torch.Tensor, torch.Tensor]:
        return self.local_left_partition_mass_map, self.local_right_partition_mass_map

    @property
    def baseline_partition_masses(self) -> tuple[torch.Tensor, torch.Tensor]:
        return self.baseline_left_mass, self.baseline_right_mass

    @property
    def surviving_partition_masses(self) -> tuple[torch.Tensor, torch.Tensor]:
        return self.surviving_left_mass, self.surviving_right_mass

    @property
    def spatial_partition_shares(self) -> tuple[torch.Tensor, torch.Tensor]:
        return self.local_left_share_map, self.local_right_share_map

    @property
    def concentration_spectra(self) -> tuple[torch.Tensor, torch.Tensor]:
        return self.left_warrant_spectrum, self.right_warrant_spectrum

    @property
    def local_concentration_spectra(self) -> tuple[torch.Tensor, torch.Tensor]:
        return self.local_left_warrant_spectrum, self.local_right_warrant_spectrum


@dataclass
class PathsOSWTOutput:
    """Complete OSWT trace; the refined posterior is the only grade path."""

    base_output: OriginOutput
    shell_evidence: Dict[str, PathsOSWTScaleEvidence]
    probe_z: torch.Tensor
    probe_weights: torch.Tensor
    boundary_scale_weights: torch.Tensor
    beta: torch.Tensor
    tau: torch.Tensor
    strength: float
    risk_gate_enabled: bool
    left_boundary_warrant: torch.Tensor
    right_boundary_warrant: torch.Tensor
    boundary_coverage: torch.Tensor
    shell_contrast: torch.Tensor
    base_boundary_probability: torch.Tensor
    boundary_gate: torch.Tensor
    log_odds_increment: torch.Tensor
    base_adjacent_log_odds: torch.Tensor
    refined_adjacent_log_odds: torch.Tensor
    grade_log_correction: torch.Tensor
    signed_boundary_flow: torch.Tensor
    wasserstein1: torch.Tensor
    boundary_correction: torch.Tensor
    base_continuation_logits: torch.Tensor
    continuation_logits: torch.Tensor
    continuation_probs: torch.Tensor
    stop_probs: torch.Tensor
    class_probs: torch.Tensor
    log_class_probs: torch.Tensor
    cumulative_probs: torch.Tensor
    expected_grade: torch.Tensor
    posterior_median: torch.Tensor
    class_map: torch.Tensor

    @property
    def base_class_probs(self) -> torch.Tensor:
        return self.base_output.class_probs

    @property
    def base_log_class_probs(self) -> torch.Tensor:
        return self.base_output.log_class_probs

    @property
    def base_cumulative_probs(self) -> torch.Tensor:
        return self.base_output.cumulative_probs

    @property
    def base_expected_grade(self) -> torch.Tensor:
        return self.base_output.expected_grade

    @property
    def scale_evidence(self) -> Dict[str, OriginScaleEvidence]:
        return self.base_output.scale_evidence

    @property
    def spectrum_evidence(self) -> Dict[str, PathsOSWTScaleEvidence]:
        """Compatibility alias for certificate serializers."""

        return self.shell_evidence

    @property
    def prior_rates(self) -> torch.Tensor:
        return self.base_output.prior_rates

    @property
    def total_rates(self) -> torch.Tensor:
        return self.base_output.total_rates

    @property
    def generator(self) -> torch.Tensor:
        return self.base_output.generator

    @property
    def transition_matrix(self) -> torch.Tensor:
        return self.base_output.transition_matrix

    @property
    def atom_mode(self) -> str:
        return self.base_output.atom_mode

    @property
    def scale_simplex(self) -> torch.Tensor:
        return self.base_output.scale_simplex

    @property
    def boundary_scales(self) -> torch.Tensor:
        return self.base_output.boundary_scales

    @property
    def total_rate_cap(self) -> float:
        return self.base_output.total_rate_cap

    @property
    def prior_rate_cap(self) -> float:
        return self.base_output.prior_rate_cap

    @property
    def boundary_scale_cap(self) -> float:
        return self.base_output.boundary_scale_cap

    @property
    def atom_mass_cap(self) -> float:
        return self.base_output.atom_mass_cap

    @property
    def rate_roundoff_margin(self) -> float:
        return self.base_output.rate_roundoff_margin

    @property
    def encoder_output(self):
        return self.base_output.encoder_output

    @property
    def local_rate_maps(self) -> Dict[str, torch.Tensor]:
        return self.base_output.local_rate_maps

    @property
    def local_rate_maps_channels_first(self) -> Dict[str, torch.Tensor]:
        return self.base_output.local_rate_maps_channels_first

    @property
    def valid_masks(self) -> Dict[str, torch.Tensor]:
        return self.base_output.valid_masks

    @property
    def active_valid_masks(self) -> Dict[str, torch.Tensor]:
        return {name: item.active_mask for name, item in self.shell_evidence.items()}

    @property
    def metadata(self):
        return self.base_output.metadata

    @property
    def predicted_grade(self) -> torch.Tensor:
        return self.class_map

    @property
    def monotone_grade(self) -> torch.Tensor:
        return self.posterior_median

    @property
    def class_probabilities(self) -> torch.Tensor:
        return self.class_probs

    @property
    def log_class_probabilities(self) -> torch.Tensor:
        return self.log_class_probs

    @property
    def cumulative_probabilities(self) -> torch.Tensor:
        return self.cumulative_probs

    @property
    def rho(self) -> float:
        return self.strength

    @property
    def gains(self) -> torch.Tensor:
        return self.beta

    @property
    def transport_gains(self) -> torch.Tensor:
        return self.beta

    @property
    def transport_direction(self) -> torch.Tensor:
        return self.shell_contrast

    @property
    def net_boundary_flow(self) -> torch.Tensor:
        return self.signed_boundary_flow

    @property
    def local_warrant_maps(self) -> Dict[str, torch.Tensor]:
        """Additive grade-support maps built only from causal boundary ledgers.

        Grade 0 uses the left side of boundary 0, grade K-1 uses the right
        side of boundary K-2, and each interior grade uses the two partition
        warrants that bracket it.  Every term enters the prediction path.
        """

        result: Dict[str, torch.Tensor] = {}
        for name, item in self.shell_evidence.items():
            left = item.local_left_boundary_warrant_map
            right = item.local_right_boundary_warrant_map
            if left.shape[1] == 1:
                grade_support = torch.cat((left, right), dim=1)
            else:
                grade_support = torch.cat(
                    (left[:, :1], right[:, :-1] + left[:, 1:], right[:, -1:]),
                    dim=1,
                )
            result[name] = grade_support.permute(0, 2, 3, 1)
        return result

    @property
    def local_left_warrant_maps(self) -> Dict[str, torch.Tensor]:
        return {
            name: item.local_left_boundary_warrant_map.permute(0, 2, 3, 1)
            for name, item in self.shell_evidence.items()
        }

    @property
    def local_right_warrant_maps(self) -> Dict[str, torch.Tensor]:
        return {
            name: item.local_right_boundary_warrant_map.permute(0, 2, 3, 1)
            for name, item in self.shell_evidence.items()
        }

    @property
    def local_transport_maps(self) -> Dict[str, torch.Tensor]:
        """Compatibility alias; maps are warrants, not realized mass flow."""

        return self.local_warrant_maps

    @property
    def local_intervention_scores(self) -> Dict[str, torch.Tensor]:
        return {
            name: (
                item.local_left_boundary_warrant_map
                + item.local_right_boundary_warrant_map
            ).sum(dim=1)
            for name, item in self.shell_evidence.items()
        }

    @property
    def advance_correction(self) -> torch.Tensor:
        return self.log_odds_increment.clamp_min(0.0)

    @property
    def stop_correction(self) -> torch.Tensor:
        return (-self.log_odds_increment).clamp_min(0.0)


def _compile_scale_boundary_warrant(
    evidence: PathsOSWTScaleEvidence,
    *,
    boundary_scale_weight: torch.Tensor,
    probe_weights: torch.Tensor,
) -> PathsOSWTScaleEvidence:
    left_local = evidence.local_left_warrant_spectrum
    right_local = evidence.local_right_warrant_spectrum
    left_spectrum = left_local.sum(dim=(2, 3))
    right_spectrum = right_local.sum(dim=(2, 3))
    mixed_left = torch.einsum("nbl,l->nb", left_spectrum, probe_weights)
    mixed_right = torch.einsum("nbl,l->nb", right_spectrum, probe_weights)
    left_per_cell = torch.einsum("nbhwl,l->nbhw", left_local, probe_weights)
    right_per_cell = torch.einsum("nbhwl,l->nbhw", right_local, probe_weights)
    scale = boundary_scale_weight[None, :, None, None]
    local_left = left_per_cell * scale
    local_right = right_per_cell * scale
    _require_finite(left_spectrum, f"OSWT left spectrum at {evidence.name}")
    _require_finite(right_spectrum, f"OSWT right spectrum at {evidence.name}")
    _require_finite(local_left, f"OSWT local left warrant at {evidence.name}")
    _require_finite(local_right, f"OSWT local right warrant at {evidence.name}")
    return replace(
        evidence,
        surviving_left_mass=evidence.local_left_partition_mass_map.sum(dim=(-2, -1)),
        surviving_right_mass=evidence.local_right_partition_mass_map.sum(dim=(-2, -1)),
        left_warrant_spectrum=left_spectrum,
        right_warrant_spectrum=right_spectrum,
        mixed_left_warrant=mixed_left,
        mixed_right_warrant=mixed_right,
        local_left_boundary_warrant_map=local_left,
        local_right_boundary_warrant_map=local_right,
    )


def _oswt_output_from_ledgers(
    base_output: OriginOutput,
    shell_evidence: Dict[str, PathsOSWTScaleEvidence],
    *,
    probe_z: torch.Tensor,
    probe_weights: torch.Tensor,
    boundary_scale_weights: torch.Tensor,
    beta: torch.Tensor,
    tau: torch.Tensor,
    strength: float,
    risk_gate_enabled: bool,
) -> PathsOSWTOutput:
    if not shell_evidence:
        raise ValueError("OSWT requires at least one shell evidence scale")
    left_warrant = sum(
        item.local_left_boundary_warrant_map.sum(dim=(-2, -1))
        for item in shell_evidence.values()
    )
    right_warrant = sum(
        item.local_right_boundary_warrant_map.sum(dim=(-2, -1))
        for item in shell_evidence.values()
    )
    left_warrant = _canonical_unit_interval(
        left_warrant, name="OSWT aggregate left-boundary warrant"
    )
    right_warrant = _canonical_unit_interval(
        right_warrant, name="OSWT aggregate right-boundary warrant"
    )
    refined = apply_shell_warranted_transport(
        base_output.class_probs,
        left_warrant,
        right_warrant,
        beta,
        tau,
        strength=strength,
        risk_gate_enabled=risk_gate_enabled,
    )
    base_continuation = continuation_logits_from_log_probs(
        base_output.log_class_probs
    )
    boundary_correction = refined.continuation_logits - base_continuation
    return PathsOSWTOutput(
        base_output=base_output,
        shell_evidence=shell_evidence,
        probe_z=probe_z,
        probe_weights=probe_weights,
        boundary_scale_weights=boundary_scale_weights,
        beta=beta,
        tau=tau,
        strength=float(strength),
        risk_gate_enabled=bool(risk_gate_enabled),
        left_boundary_warrant=refined.left_boundary_warrant,
        right_boundary_warrant=refined.right_boundary_warrant,
        boundary_coverage=refined.boundary_coverage,
        shell_contrast=refined.shell_contrast,
        base_boundary_probability=refined.base_boundary_probability,
        boundary_gate=refined.boundary_gate,
        log_odds_increment=refined.log_odds_increment,
        base_adjacent_log_odds=refined.base_adjacent_log_odds,
        refined_adjacent_log_odds=refined.refined_adjacent_log_odds,
        grade_log_correction=refined.grade_log_correction,
        signed_boundary_flow=refined.signed_boundary_flow,
        wasserstein1=refined.wasserstein1,
        boundary_correction=boundary_correction,
        base_continuation_logits=base_continuation,
        continuation_logits=refined.continuation_logits,
        continuation_probs=refined.continuation_probs,
        stop_probs=refined.stop_probs,
        class_probs=refined.class_probs,
        log_class_probs=refined.log_class_probs,
        cumulative_probs=refined.cumulative_probs,
        expected_grade=refined.expected_grade,
        posterior_median=refined.posterior_median,
        class_map=refined.class_map,
    )


class PathsOSWTRefiner(nn.Module):
    """Compile V3 atoms into complementary ordinal-boundary warrants."""

    def __init__(
        self,
        num_grades: int,
        evidence_scales: Sequence[str],
        *,
        probe_z: Sequence[float] = _DEFAULT_PROBES,
        beta_init: float = 0.1,
        tau_init: float = 0.05,
        beta_cap: float = 3.0,
        tau_floor: float = 1e-3,
        tau_cap: float = 0.5,
        strength: float = 1.0,
        rho: Optional[float] = None,
        risk_gate_enabled: bool = True,
    ) -> None:
        super().__init__()
        if num_grades < 2:
            raise ValueError("num_grades must be at least two")
        scales = tuple(str(value) for value in evidence_scales)
        if not scales or len(scales) != len(set(scales)):
            raise ValueError("evidence_scales must be non-empty and unique")
        probes = _canonical_probe_values(probe_z)
        if rho is not None:
            strength = float(rho)
        if not math.isfinite(strength) or strength < 0.0:
            raise ValueError("strength/rho must be finite and non-negative")
        beta_cap = float(beta_cap)
        tau_floor = float(tau_floor)
        tau_cap = float(tau_cap)
        beta_raw_init = _inverse_bounded_sigmoid(
            beta_init, lower=0.0, upper=beta_cap, name="OSWT beta"
        )
        tau_raw_init = _inverse_bounded_sigmoid(
            tau_init, lower=tau_floor, upper=tau_cap, name="OSWT tau"
        )

        self.num_grades = int(num_grades)
        self.num_boundaries = self.num_grades - 1
        self.evidence_scales = scales
        self.beta_init = float(beta_init)
        self.tau_init = float(tau_init)
        self.beta_cap = beta_cap
        self.tau_floor = tau_floor
        self.tau_cap = tau_cap
        self.strength = float(strength)
        self.risk_gate_enabled = bool(risk_gate_enabled)
        self.register_buffer("probe_z", torch.tensor(probes, dtype=torch.float64))
        # One shared measurement functional makes adjacent shell warrants
        # directly comparable; grade-specific probe functions would create a
        # hidden calibration bypass inside the signed contrast.
        self.probe_logits = nn.Parameter(torch.zeros(len(probes)))
        self.raw_beta = nn.Parameter(
            torch.full(
                (self.num_boundaries,),
                beta_raw_init,
            )
        )
        self.raw_tau = nn.Parameter(
            torch.full(
                (self.num_boundaries,),
                tau_raw_init,
            )
        )

    @property
    def probe_weights(self) -> torch.Tensor:
        return self.probe_logits.double().softmax(dim=-1)

    @property
    def beta(self) -> torch.Tensor:
        return self.beta_cap * torch.sigmoid(self.raw_beta.double())

    @property
    def tau(self) -> torch.Tensor:
        span = self.tau_cap - self.tau_floor
        return self.tau_floor + span * torch.sigmoid(self.raw_tau.double())

    @property
    def rho(self) -> float:
        return self.strength

    @property
    def gains(self) -> torch.Tensor:
        return self.beta

    def forward(self, base_output: OriginOutput) -> Union[OriginOutput, PathsOSWTOutput]:
        # This object-identity branch is the registered exact V3 control.
        if self.strength == 0.0:
            return base_output
        if base_output.atom_mode != "cumulative":
            raise ValueError("OSWT requires cumulative ORIGIN-v3 atoms")
        if tuple(base_output.scale_evidence) != self.evidence_scales:
            raise ValueError(
                "OSWT scale order differs from the V3 ledger: "
                f"{tuple(base_output.scale_evidence)} vs {self.evidence_scales}"
            )
        if base_output.class_probs.shape[-1] != self.num_grades:
            raise ValueError("V3 grade count differs from OSWT")

        probe_weights = self.probe_weights
        beta = self.beta
        tau = self.tau
        boundary_scale_weights = canonical_boundary_scale_weights(
            base_output.scale_simplex
        )
        if tuple(boundary_scale_weights.shape) != (
            len(self.evidence_scales), self.num_boundaries
        ):
            raise ValueError("V3 boundary scale simplex has incompatible shape")

        ledgers: Dict[str, PathsOSWTScaleEvidence] = {}
        for scale_index, name in enumerate(self.evidence_scales):
            source = base_output.scale_evidence[name]
            valid = source.valid_mask.bool()
            normalized = _canonical_unit_interval(
                source.compiled_atoms / float(base_output.atom_mass_cap),
                name=f"OSWT normalized cumulative atoms at {name}",
            ).to(dtype=_OSWT_DTYPE)
            shells = ordinal_shells_from_cumulative_atoms(normalized, valid)
            # Compile exact complementary partitions from the stored shell
            # simplex *before* any nonlinear spatial functional.  Therefore
            # the two sides of every boundary contain the same cells, use the
            # same measurement functional, and differ only in ordinal side.
            left_mass_map = torch.cumsum(shells, dim=1)[:, :-1]
            reverse_shell_mass = torch.flip(
                torch.cumsum(torch.flip(shells, dims=(1,)), dim=1), dims=(1,)
            )
            right_mass_map = reverse_shell_mass[:, 1:]
            partition_error = (
                left_mass_map + right_mass_map - valid[:, None].to(_OSWT_DTYPE)
            ).abs()
            if bool((partition_error > 5e-12).any()):
                raise FloatingPointError(
                    "OSWT complementary boundary partitions do not conserve "
                    f"unit mass (maximum error={float(partition_error.max().item()):g})"
                )

            baseline_left_mass = left_mass_map.sum(dim=(-2, -1))
            baseline_right_mass = right_mass_map.sum(dim=(-2, -1))
            original_valid_count = valid.sum(dim=(-2, -1))

            def compile_side(
                mass_map: torch.Tensor,
                baseline_mass: torch.Tensor,
            ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                positive = baseline_mass > 0.0
                safe_mass = torch.where(
                    positive, baseline_mass, torch.ones_like(baseline_mass)
                )
                share = mass_map / safe_mass[:, :, None, None]
                share = torch.where(
                    positive[:, :, None, None], share, torch.zeros_like(share)
                )
                phi = tangent_removed_pgf_probe(share, self.probe_z)
                phi = torch.where(
                    valid[:, None, :, :, None], phi, torch.zeros_like(phi)
                )
                # Multiplication by the original partition mass, rather than
                # mean cell mass, removes the 1/M lattice-size shrinkage.
                # Because phi_z(u)<=u^2 for u in [0,1], the
                # aggregate warrant remains <=1 while retaining both amount
                # and focal concentration of evidence.
                local_warrant = phi * baseline_mass[:, :, None, None, None]
                return share, phi, local_warrant

            (
                local_left_share,
                local_left_phi,
                local_left_warrant,
            ) = compile_side(left_mass_map, baseline_left_mass)
            (
                local_right_share,
                local_right_phi,
                local_right_warrant,
            ) = compile_side(right_mass_map, baseline_right_mass)
            initial = PathsOSWTScaleEvidence(
                name=name,
                normalized_cumulative_atoms=normalized,
                ordinal_shells=shells,
                original_valid_mask=valid,
                active_mask=valid,
                original_valid_count=original_valid_count,
                local_left_partition_mass_map=left_mass_map,
                local_right_partition_mass_map=right_mass_map,
                baseline_left_mass=baseline_left_mass,
                baseline_right_mass=baseline_right_mass,
                surviving_left_mass=baseline_left_mass,
                surviving_right_mass=baseline_right_mass,
                local_left_share_map=local_left_share,
                local_right_share_map=local_right_share,
                local_left_phi_spectrum=local_left_phi,
                local_right_phi_spectrum=local_right_phi,
                local_left_warrant_spectrum=local_left_warrant,
                local_right_warrant_spectrum=local_right_warrant,
                left_warrant_spectrum=torch.empty(0, device=normalized.device),
                right_warrant_spectrum=torch.empty(0, device=normalized.device),
                mixed_left_warrant=torch.empty(0, device=normalized.device),
                mixed_right_warrant=torch.empty(0, device=normalized.device),
                local_left_boundary_warrant_map=torch.empty(
                    0, device=normalized.device
                ),
                local_right_boundary_warrant_map=torch.empty(
                    0, device=normalized.device
                ),
                metadata=source.metadata,
            )
            ledgers[name] = _compile_scale_boundary_warrant(
                initial,
                boundary_scale_weight=boundary_scale_weights[scale_index],
                probe_weights=probe_weights,
            )

        return _oswt_output_from_ledgers(
            base_output,
            ledgers,
            probe_z=self.probe_z,
            probe_weights=probe_weights,
            boundary_scale_weights=boundary_scale_weights,
            beta=beta,
            tau=tau,
            strength=self.strength,
            risk_gate_enabled=self.risk_gate_enabled,
        )


@dataclass
class PathsOSWTInterventionOutput:
    """Exact joint deletion of stored ORIGIN-rate and OSWT-shell ledgers."""

    baseline: PathsOSWTOutput
    output: PathsOSWTOutput
    removal_masks: Dict[str, torch.Tensor]
    removed_rates: torch.Tensor
    removed_left_boundary_warrant: torch.Tensor
    removed_right_boundary_warrant: torch.Tensor
    removed_signed_boundary_flow: torch.Tensor
    removed_log_odds_increment: torch.Tensor
    removed_boundary_correction: torch.Tensor
    removed_left_phi_sums: Dict[str, torch.Tensor]
    removed_right_phi_sums: Dict[str, torch.Tensor]
    removed_left_mass_sums: Dict[str, torch.Tensor]
    removed_right_mass_sums: Dict[str, torch.Tensor]
    base_intervention: OriginInterventionOutput

    @property
    def delta_expected_grade(self) -> torch.Tensor:
        return self.baseline.expected_grade - self.output.expected_grade

    @property
    def class_probs(self) -> torch.Tensor:
        return self.output.class_probs

    @property
    def cumulative_probs(self) -> torch.Tensor:
        return self.output.cumulative_probs

    @property
    def predicted_grade(self) -> torch.Tensor:
        return self.output.predicted_grade

    @property
    def monotone_grade(self) -> torch.Tensor:
        return self.output.posterior_median

    @property
    def removed_net_boundary_flow(self) -> torch.Tensor:
        return self.removed_signed_boundary_flow

    @property
    def removed_signed_correction(self) -> torch.Tensor:
        return self.removed_boundary_correction


def _canonical_removal_mask(
    removal: torch.Tensor,
    target: torch.Tensor,
    *,
    name: str,
) -> torch.Tensor:
    removal = torch.as_tensor(removal, device=target.device)
    if removal.ndim == 2:
        removal = removal.unsqueeze(0)
    if removal.ndim == 4 and removal.shape[1] == 1:
        removal = removal[:, 0]
    if removal.ndim != 3:
        raise ValueError(
            f"removal mask for {name!r} must have shape (H,W), (N,H,W), "
            "or (N,1,H,W)"
        )
    if removal.shape[0] == 1 and target.shape[0] > 1:
        removal = removal.expand(target.shape[0], -1, -1)
    if tuple(removal.shape) != tuple(target.shape):
        raise ValueError(
            f"removal mask for {name!r} has shape {tuple(removal.shape)}; "
            f"expected {tuple(target.shape)}"
        )
    return removal.bool()


def replay_paths_oswt_without(
    output: PathsOSWTOutput,
    removal_masks: Mapping[str, torch.Tensor],
    *,
    force_decoder_fp64: bool = True,
) -> PathsOSWTInterventionOutput:
    """Delete stored shell/rate cells and replay the only prediction path."""

    if not force_decoder_fp64:
        raise ValueError("OSWT structural decoders must run in FP64")
    unknown = set(removal_masks) - set(output.shell_evidence)
    if unknown:
        raise ValueError(f"removal masks contain unknown scales {sorted(unknown)}")

    canonical: Dict[str, torch.Tensor] = {}
    for name, evidence in output.shell_evidence.items():
        requested = removal_masks.get(name)
        mask = (
            torch.zeros_like(evidence.active_mask)
            if requested is None
            else _canonical_removal_mask(requested, evidence.active_mask, name=name)
        )
        canonical[name] = mask & evidence.active_mask

    base_intervention = replay_origin_without(
        output.base_output, canonical, force_decoder_fp64=True
    )

    replayed: Dict[str, PathsOSWTScaleEvidence] = {}
    removed_left_phi: Dict[str, torch.Tensor] = {}
    removed_right_phi: Dict[str, torch.Tensor] = {}
    removed_left_mass: Dict[str, torch.Tensor] = {}
    removed_right_mass: Dict[str, torch.Tensor] = {}
    removed_left_warrant_by_scale: list[torch.Tensor] = []
    removed_right_warrant_by_scale: list[torch.Tensor] = []
    for scale_index, (name, evidence) in enumerate(output.shell_evidence.items()):
        mask = canonical[name]
        boundary_mask = mask[:, None]
        spectrum_mask = mask[:, None, :, :, None]
        removed_left_phi[name] = torch.where(
            spectrum_mask,
            evidence.local_left_phi_spectrum,
            torch.zeros_like(evidence.local_left_phi_spectrum),
        ).sum(dim=(2, 3))
        removed_right_phi[name] = torch.where(
            spectrum_mask,
            evidence.local_right_phi_spectrum,
            torch.zeros_like(evidence.local_right_phi_spectrum),
        ).sum(dim=(2, 3))
        removed_left_mass[name] = torch.where(
            boundary_mask,
            evidence.local_left_partition_mass_map,
            torch.zeros_like(evidence.local_left_partition_mass_map),
        ).sum(dim=(-2, -1))
        removed_right_mass[name] = torch.where(
            boundary_mask,
            evidence.local_right_partition_mass_map,
            torch.zeros_like(evidence.local_right_partition_mass_map),
        ).sum(dim=(-2, -1))
        removed_left_warrant_by_scale.append(
            torch.where(
                boundary_mask,
                evidence.local_left_boundary_warrant_map,
                torch.zeros_like(evidence.local_left_boundary_warrant_map),
            ).sum(dim=(-2, -1))
        )
        removed_right_warrant_by_scale.append(
            torch.where(
                boundary_mask,
                evidence.local_right_boundary_warrant_map,
                torch.zeros_like(evidence.local_right_boundary_warrant_map),
            ).sum(dim=(-2, -1))
        )

        # All kept quantities come from the stored forward ledger.  Most
        # importantly, ordinal_shells (including v0) are masked directly and
        # never reconstructed from the post-deletion cumulative atom field.
        kept_normalized = torch.where(
            boundary_mask,
            torch.zeros_like(evidence.normalized_cumulative_atoms),
            evidence.normalized_cumulative_atoms,
        )
        kept_shells = torch.where(
            boundary_mask,
            torch.zeros_like(evidence.ordinal_shells),
            evidence.ordinal_shells,
        )
        kept_left_mass = torch.where(
            boundary_mask,
            torch.zeros_like(evidence.local_left_partition_mass_map),
            evidence.local_left_partition_mass_map,
        )
        kept_right_mass = torch.where(
            boundary_mask,
            torch.zeros_like(evidence.local_right_partition_mass_map),
            evidence.local_right_partition_mass_map,
        )
        kept_left_share = torch.where(
            boundary_mask,
            torch.zeros_like(evidence.local_left_share_map),
            evidence.local_left_share_map,
        )
        kept_right_share = torch.where(
            boundary_mask,
            torch.zeros_like(evidence.local_right_share_map),
            evidence.local_right_share_map,
        )
        kept_left_phi = torch.where(
            spectrum_mask,
            torch.zeros_like(evidence.local_left_phi_spectrum),
            evidence.local_left_phi_spectrum,
        )
        kept_right_phi = torch.where(
            spectrum_mask,
            torch.zeros_like(evidence.local_right_phi_spectrum),
            evidence.local_right_phi_spectrum,
        )
        kept_left_warrant = torch.where(
            spectrum_mask,
            torch.zeros_like(evidence.local_left_warrant_spectrum),
            evidence.local_left_warrant_spectrum,
        )
        kept_right_warrant = torch.where(
            spectrum_mask,
            torch.zeros_like(evidence.local_right_warrant_spectrum),
            evidence.local_right_warrant_spectrum,
        )
        provisional = replace(
            evidence,
            normalized_cumulative_atoms=kept_normalized,
            ordinal_shells=kept_shells,
            active_mask=evidence.active_mask & ~mask,
            local_left_partition_mass_map=kept_left_mass,
            local_right_partition_mass_map=kept_right_mass,
            # Baseline masses and denominators are intentionally
            # unchanged: an intervention may delete evidence but cannot
            # renormalize the remaining cells into a stronger explanation.
            local_left_share_map=kept_left_share,
            local_right_share_map=kept_right_share,
            local_left_phi_spectrum=kept_left_phi,
            local_right_phi_spectrum=kept_right_phi,
            local_left_warrant_spectrum=kept_left_warrant,
            local_right_warrant_spectrum=kept_right_warrant,
        )
        replayed[name] = _compile_scale_boundary_warrant(
            provisional,
            boundary_scale_weight=output.boundary_scale_weights[scale_index],
            probe_weights=output.probe_weights,
        )

    replay_output = _oswt_output_from_ledgers(
        base_intervention.output,
        replayed,
        probe_z=output.probe_z,
        probe_weights=output.probe_weights,
        boundary_scale_weights=output.boundary_scale_weights,
        beta=output.beta,
        tau=output.tau,
        strength=output.strength,
        risk_gate_enabled=output.risk_gate_enabled,
    )
    removed_left_warrant = sum(removed_left_warrant_by_scale)
    removed_right_warrant = sum(removed_right_warrant_by_scale)
    for side, baseline, replayed_warrant, removed_warrant in (
        (
            "left",
            output.left_boundary_warrant,
            replay_output.left_boundary_warrant,
            removed_left_warrant,
        ),
        (
            "right",
            output.right_boundary_warrant,
            replay_output.right_boundary_warrant,
            removed_right_warrant,
        ),
    ):
        ledger_error = (baseline - replayed_warrant - removed_warrant).abs()
        ledger_tolerance = 5e-12 + 5e-12 * baseline.abs()
        if bool((ledger_error > ledger_tolerance).any()):
            raise FloatingPointError(
                f"OSWT stored {side} boundary-warrant deletion did not replay "
                f"additively (maximum error={float(ledger_error.max().item()):g})"
            )

    return PathsOSWTInterventionOutput(
        baseline=output,
        output=replay_output,
        removal_masks=canonical,
        removed_rates=base_intervention.removed_rates,
        removed_left_boundary_warrant=removed_left_warrant,
        removed_right_boundary_warrant=removed_right_warrant,
        removed_signed_boundary_flow=(
            output.signed_boundary_flow - replay_output.signed_boundary_flow
        ),
        removed_log_odds_increment=(
            output.log_odds_increment - replay_output.log_odds_increment
        ),
        removed_boundary_correction=(
            output.boundary_correction - replay_output.boundary_correction
        ),
        removed_left_phi_sums=removed_left_phi,
        removed_right_phi_sums=removed_right_phi,
        removed_left_mass_sums=removed_left_mass,
        removed_right_mass_sums=removed_right_mass,
        base_intervention=base_intervention,
    )


class PathsOSWTModel(OriginModel):
    """ORIGIN-v3 followed exclusively by ordinal-shell warranted transport."""

    def __init__(
        self,
        *,
        paths_oswt_probe_z: Sequence[float] = _DEFAULT_PROBES,
        paths_oswt_beta_init: float = 0.1,
        paths_oswt_tau_init: float = 0.05,
        paths_oswt_beta_cap: float = 3.0,
        paths_oswt_tau_floor: float = 1e-3,
        paths_oswt_tau_cap: float = 0.5,
        paths_oswt_strength: float = 1.0,
        paths_oswt_rho: Optional[float] = None,
        paths_oswt_risk_gate: bool = True,
        **origin_kwargs,
    ) -> None:
        super().__init__(**origin_kwargs)
        self.paths_refiner = PathsOSWTRefiner(
            self.num_classes,
            self.evidence_scales,
            probe_z=paths_oswt_probe_z,
            beta_init=paths_oswt_beta_init,
            tau_init=paths_oswt_tau_init,
            beta_cap=paths_oswt_beta_cap,
            tau_floor=paths_oswt_tau_floor,
            tau_cap=paths_oswt_tau_cap,
            strength=paths_oswt_strength,
            rho=paths_oswt_rho,
            risk_gate_enabled=paths_oswt_risk_gate,
        )

    @property
    def paths_oswt_refiner(self) -> PathsOSWTRefiner:
        return self.paths_refiner

    def load_origin_v3_state_dict(
        self,
        source_state: Mapping[str, torch.Tensor],
    ) -> None:
        """Strictly load only an exact V3 base, leaving OSWT at init."""

        current = self.state_dict()
        refiner_prefix = "paths_refiner."
        expected_base = {key for key in current if not key.startswith(refiner_prefix)}
        supplied = set(source_state)
        missing = expected_base - supplied
        unexpected = supplied - expected_base
        if missing or unexpected:
            raise ValueError(
                "source is not an exact ORIGIN-v3 state dict: "
                f"missing={sorted(missing)}, unexpected={sorted(unexpected)}"
            )
        shape_errors = {
            key: (tuple(source_state[key].shape), tuple(current[key].shape))
            for key in expected_base
            if tuple(source_state[key].shape) != tuple(current[key].shape)
        }
        if shape_errors:
            raise ValueError(f"ORIGIN-v3 warm-start tensor shapes differ: {shape_errors}")
        incompatible = self.load_state_dict(dict(source_state), strict=False)
        expected_refiner = {
            key for key in current if key.startswith(refiner_prefix)
        }
        if (
            set(incompatible.missing_keys) != expected_refiner
            or incompatible.unexpected_keys
        ):
            raise RuntimeError(
                "internal OSWT warm-start invariant failed: "
                f"missing={incompatible.missing_keys}, "
                f"unexpected={incompatible.unexpected_keys}"
            )

    def architecture_metadata(self) -> Dict[str, object]:
        base = super().architecture_metadata()
        return {
            **base,
            "name": "PATHS-OSWT",
            "architecture_schema": "paths-ordinal-shell-warranted-transport-v1",
            "base_architecture": "ORIGIN-v3-bounded-rate",
            "posterior_path": (
                "nested_local_atoms_to_v3_ctmc_then_exclusive_grade_shells_"
                "to_amount_concentration_warrants_to_adjacent_log_odds_"
                "increments_to_unique_normalized_posterior"
            ),
            "shell_compilation": "v0=1-u0;vj=u(j-1)-uj;v(K-1)=u(K-2)",
            "shell_simplex": "exact_per_valid_cell_exclusive_grade_partition",
            "pgf_probes": self.paths_refiner.probe_z.detach().cpu().tolist(),
            "probe_parameterization": "fixed_z_shared_simplex_v1",
            "focality_transform": "tangent_removed_bernoulli_log_pgf_v2",
            "boundary_partitions": "L[k]=sum(g<=k)v[g];R[k]=sum(g>k)v[g]",
            "partition_semantics": (
                "residual_cumulative_atom_capacity_is_low_side_null_internal_evidence"
            ),
            "normalization": (
                "frozen_spatial_partition_share_times_original_partition_mass"
            ),
            "boundary_scale_weights": (
                "same_native_v3_boundary_simplex_on_both_sides_of_each_cut"
            ),
            "shell_contrast": "(R[k]-L[k])/(R[k]+L[k]+tau[k])",
            "coverage_aware_absence": "zero_warrant_gives_exact_identity",
            "risk_gate": (
                "cut_aligned_4q(1-q)_where_q=Pbase(Y>k)"
                if self.paths_refiner.risk_gate_enabled
                else "disabled_registered_control"
            ),
            "transport_parameterization": (
                "rho*bounded_beta*boundary_gate*shell_contrast"
            ),
            "strength": self.paths_refiner.strength,
            "beta_init": self.paths_refiner.beta_init,
            "beta_cap": self.paths_refiner.beta_cap,
            "tau_init": self.paths_refiner.tau_init,
            "tau_floor": self.paths_refiner.tau_floor,
            "tau_cap": self.paths_refiner.tau_cap,
            "decoder": "fp64_adjacent_log_odds_increment_normalization_v1",
            "flow_certificate": (
                "signed_ordinal_cut_flow_cumsum(base_p-refined_p)"
            ),
            "transport_cost": "exact_discrete_ordinal_wasserstein1=sum_abs_flow",
            "intervention": (
                "joint_stored_origin_rate_and_shell_warrant_deletion_without_"
                "shell_recomputation_or_denominator_renormalization"
            ),
            "deletion_semantics": (
                "exact_internal_stored_ledger_intervention_not_pixel_causality"
            ),
            "zero_strength_control": "exact_origin_output_object_identity",
            "no_classifier_bypass": True,
        }

    def forward(
        self,
        images: torch.Tensor,
        pixel_valid_mask: Optional[torch.Tensor] = None,
        *,
        force_decoder_fp64: bool = True,
        return_encoder_maps: bool = False,
    ) -> Union[OriginOutput, PathsOSWTOutput]:
        if not force_decoder_fp64:
            raise ValueError("OSWT structural decoders must run in FP64")
        encoded = self.encoder(images, pixel_valid_mask)
        base = self.generator(
            encoded,
            force_decoder_fp64=True,
            retain_encoder_output=return_encoder_maps,
        )
        return self.paths_refiner(base)

    def replay_without(
        self,
        output: Union[OriginOutput, PathsOSWTOutput],
        removal_masks: Mapping[str, torch.Tensor],
        *,
        force_decoder_fp64: bool = True,
    ) -> Union[OriginInterventionOutput, PathsOSWTInterventionOutput]:
        if isinstance(output, PathsOSWTOutput):
            return replay_paths_oswt_without(
                output, removal_masks, force_decoder_fp64=force_decoder_fp64
            )
        return replay_origin_without(
            output, removal_masks, force_decoder_fp64=force_decoder_fp64
        )

    def topk_intervention(self, *args, **kwargs):
        raise NotImplementedError(
            "OSWT requires joint rate-and-stored-shell selection; use "
            "replay_without with an explicit spatial mask"
        )


def build_paths_oswt_model(**kwargs) -> PathsOSWTModel:
    return PathsOSWTModel(**kwargs)


# Short functional alias used by mathematical tests and external consumers.
ordinal_shells = ordinal_shells_from_cumulative_atoms


__all__ = [
    "PathsOSWTInterventionOutput",
    "PathsOSWTModel",
    "PathsOSWTOutput",
    "PathsOSWTRefinementDistribution",
    "PathsOSWTRefiner",
    "PathsOSWTScaleEvidence",
    "apply_shell_warranted_transport",
    "apply_shell_warranted_transport_from_log_probs",
    "build_paths_oswt_model",
    "canonical_boundary_scale_weights",
    "ordinal_shells",
    "ordinal_shells_from_cumulative_atoms",
    "replay_paths_oswt_without",
]
