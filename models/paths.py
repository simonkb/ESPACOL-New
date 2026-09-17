"""PATHS-v3: spectral evidence with signed adjacent probability transport.

PATHS retains the audited ORIGIN-v3 cumulative-atom ledger and pure-birth
posterior as its safety path. It asks one additional structural question: for
a fixed amount of ordinal evidence, is that evidence concentrated in a few
spatial cells or diffusely spread across the image?

For normalized cumulative atom ``u`` and fixed probe ``z`` the local focality
numerator is

    phi_z(u) = [-log(1 - (1-z)u) - (1-z)u] / [-log(z) - (1-z)].

The removed tangent gives ``phi_z(0)=phi'_z(0)=0``, ``phi_z(1)=1``, and
``0 <= phi_z(u) <= u``. Dividing its sum by the stored baseline atom mass
produces a concentration spectrum in ``[0,1]``. The baseline denominator is
never renormalized during intervention, so every correction is an exact sum
of stored local terms. Deleting a cell removes both its ORIGIN rate and its
PATHS focality contribution before replaying the same decoders.

PATHS-v3 converts each boundary's concentration ``q`` into two non-negative
odds: one transports probability one grade upward and the other transports it
one grade downward.  Their balance is controlled by a bounded signed direction
``tanh(slope * (q - threshold))``.  The resulting tridiagonal Markov kernel is
row stochastic, so refinement conserves probability mass, never skips a grade,
and remains exactly replayable after deleting stored spatial evidence.  Unlike
the previous one-sided continuation correction, it can repair both under- and
over-grading.  ``strength=0`` still returns the actual
:class:`~models.origin.OriginOutput` object, making the V3 control exact.

Atoms and focality values are model-internal evidence quantities, not lesion
counts, biological progression times, or causal pixel effects without
independent clinical validation.
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


_PATHS_DTYPE = torch.float64
_DEFAULT_PROBES = (0.1, 0.25, 0.5, 0.75, 0.9)


def _require_finite(tensor: torch.Tensor, name: str) -> None:
    if not bool(torch.isfinite(tensor).all()):
        nan = int(torch.isnan(tensor).sum().item())
        pos = int(torch.isposinf(tensor).sum().item())
        neg = int(torch.isneginf(tensor).sum().item())
        raise FloatingPointError(
            f"{name} contains non-finite values "
            f"(nan={nan}, +inf={pos}, -inf={neg})"
        )


def _inverse_bounded_sigmoid(value: float, maximum: float, *, name: str) -> float:
    value = float(value)
    maximum = float(maximum)
    if not math.isfinite(maximum) or maximum <= 0.0:
        raise ValueError(f"{name} maximum must be finite and positive")
    if not math.isfinite(value) or not 0.0 < value < maximum:
        raise ValueError(f"{name} initialization must lie strictly in (0, maximum)")
    return math.log(value) - math.log(maximum - value)


def _canonical_probe_values(values: Sequence[float]) -> Tuple[float, ...]:
    probes = tuple(float(value) for value in values)
    if not probes:
        raise ValueError("PATHS requires at least one focality probe")
    if not all(math.isfinite(value) and 0.0 < value < 1.0 for value in probes):
        raise ValueError("every PATHS probe must lie strictly in (0, 1)")
    if tuple(sorted(probes)) != probes or len(set(probes)) != len(probes):
        raise ValueError("PATHS probes must be unique and strictly increasing")
    return probes


def _canonical_unit_interval(tensor: torch.Tensor, *, name: str) -> torch.Tensor:
    """Fail closed outside ``[0,1]`` and correct endpoint roundoff only."""

    if not torch.is_tensor(tensor) or not tensor.is_floating_point():
        raise TypeError(f"{name} must be a floating-point tensor")
    _require_finite(tensor, name)
    tolerance = 8.0 * torch.finfo(tensor.dtype).eps
    outside = (tensor < -tolerance) | (tensor > 1.0 + tolerance)
    if bool(outside.any()):
        minimum = float(tensor.detach().amin().item())
        maximum = float(tensor.detach().amax().item())
        raise ValueError(
            f"{name} must lie in [0, 1] up to roundoff; "
            f"observed [{minimum:g}, {maximum:g}]"
        )
    return tensor.clamp(0.0, 1.0)


def _require_nested_cumulative_atoms(
    atoms: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    name: str,
) -> None:
    """Audit the defining ordinal invariant ``u_k >= u_{k+1}``."""

    if atoms.ndim != 4 or atoms.shape[1] < 1:
        raise ValueError(f"{name} must have shape (N,K-1,H,W)")
    if valid_mask.shape != atoms.shape[:1] + atoms.shape[-2:]:
        raise ValueError(f"valid mask shape does not match {name}")
    if atoms.shape[1] == 1:
        return
    tolerance = 16.0 * torch.finfo(atoms.dtype).eps
    gap = atoms[:, 1:] - atoms[:, :-1]
    violation = (gap > tolerance) & valid_mask[:, None]
    if bool(violation.any()):
        maximum = float(gap[violation].detach().amax().item())
        raise ValueError(
            f"{name} violates cumulative ordinal nesting "
            f"u_k >= u_(k+1) (maximum violation={maximum:g})"
        )


def tangent_removed_pgf_probe(
    normalized_atoms: torch.Tensor,
    probe_z: torch.Tensor,
) -> torch.Tensor:
    r"""Evaluate the tangent-removed focality numerator in FP64.

    For :math:`u\in[0,1]`, :math:`z\in(0,1)`, :math:`a=1-z`, and
    :math:`L=-\log z`, this returns

    .. math::

       \phi_z(u)=\frac{-\log(1-au)-au}{L-a}.

    The output's final dimension indexes probes.
    """

    if not torch.is_tensor(normalized_atoms) or not normalized_atoms.is_floating_point():
        raise TypeError("normalized_atoms must be a floating-point tensor")
    if normalized_atoms.ndim < 1:
        raise ValueError("normalized_atoms must have at least one dimension")
    if not torch.is_tensor(probe_z) or probe_z.ndim != 1 or probe_z.numel() < 1:
        raise ValueError("probe_z must be a non-empty one-dimensional tensor")
    if not probe_z.is_floating_point():
        raise TypeError("probe_z must be floating point")

    atoms = _canonical_unit_interval(
        normalized_atoms, name="normalized cumulative atoms"
    ).to(dtype=_PATHS_DTYPE)
    probes = probe_z.to(device=atoms.device, dtype=_PATHS_DTYPE)
    _require_finite(probes, "PATHS probe values")
    if bool(((probes <= 0.0) | (probes >= 1.0)).any()):
        raise ValueError("PATHS probe values must lie strictly in (0, 1)")

    atoms = atoms.unsqueeze(-1)
    view = (1,) * (atoms.ndim - 1) + (probes.numel(),)
    probes = probes.reshape(view)
    one_minus_z = 1.0 - probes
    denominator = -torch.log(probes) - one_minus_z
    numerator = -torch.log1p(-one_minus_z * atoms) - one_minus_z * atoms
    result = numerator / denominator
    _require_finite(result, "tangent-removed PGF focality numerator")

    tolerance = 64.0 * torch.finfo(result.dtype).eps
    atom_bound = atoms.expand_as(result)
    if bool(((result < -tolerance) | (result > atom_bound + tolerance)).any()):
        raise FloatingPointError("PATHS focality numerator violated 0 <= phi_z(u) <= u")
    # Do not clamp a valid result: clamp/minimum subgradients at u=0 would
    # destroy the defining zero-tangent property even though the value stayed
    # in range. The fail-closed audit above already rejects a real violation.
    return result


# Import-compatible alias. Its semantics are the tangent-removed focality
# numerator, not the earlier raw negative-log PGF.
pgf_probe = tangent_removed_pgf_probe


@dataclass
class PathsContinuationDistribution:
    """Normalized ordinal law produced by conditional continuations."""

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
    def predicted_grade(self) -> torch.Tensor:
        return self.class_map

    @property
    def monotone_grade(self) -> torch.Tensor:
        return self.posterior_median


@dataclass
class PathsAdjacentTransportDistribution:
    """Trace of a signed, mass-conserving adjacent-grade transport step."""

    class_probs: torch.Tensor
    transport_matrix: torch.Tensor
    direction: torch.Tensor
    upward_odds: torch.Tensor
    downward_odds: torch.Tensor
    net_boundary_flow: torch.Tensor


def apply_signed_adjacent_transport(
    base_class_probs: torch.Tensor,
    boundary_concentration: torch.Tensor,
    thresholds: torch.Tensor,
    slopes: torch.Tensor,
    gains: torch.Tensor,
    *,
    strength: float = 1.0,
    probability_tolerance: float = 5e-12,
) -> PathsAdjacentTransportDistribution:
    r"""Transport posterior mass only between adjacent ordinal grades.

    For boundary concentration :math:`q_k`, the signed direction and transport
    odds are

    .. math::

       d_k &= \tanh(s_k(q_k-\theta_k)),\\
       o_k^+ &= \rho g_k q_k(1+d_k)/2,\\
       o_k^- &= \rho g_k q_k(1-d_k)/2.

    Row ``j`` of the tridiagonal kernel has unnormalised weights ``1`` for
    staying, ``o_j^+`` for moving to ``j+1``, and ``o_{j-1}^-`` for moving to
    ``j-1``.  Normalising each row makes the operation a Markov transport, not
    an unconstrained logit residual.  It therefore conserves total posterior
    mass exactly up to audited FP64 roundoff and cannot jump over a grade.
    """

    if not torch.is_tensor(base_class_probs) or not base_class_probs.is_floating_point():
        raise TypeError("base_class_probs must be a floating-point tensor")
    if base_class_probs.ndim < 2 or base_class_probs.shape[-1] < 2:
        raise ValueError("base_class_probs must have shape (..., K), K >= 2")
    if not math.isfinite(strength) or not 0.0 <= float(strength) <= 1.0:
        raise ValueError("strength must lie in [0, 1]")
    if not math.isfinite(probability_tolerance) or probability_tolerance < 0.0:
        raise ValueError("probability_tolerance must be finite and non-negative")

    probabilities = base_class_probs.to(dtype=_PATHS_DTYPE)
    _require_finite(probabilities, "PATHS base class probabilities")
    probability_tolerance = float(probability_tolerance)
    if bool((probabilities < -probability_tolerance).any()):
        raise ValueError("base_class_probs must be non-negative")
    row_error = (probabilities.sum(dim=-1) - 1.0).abs()
    if bool((row_error > probability_tolerance).any()):
        raise ValueError(
            "base_class_probs must sum to one "
            f"(maximum error={float(row_error.max().item()):g})"
        )

    num_boundaries = probabilities.shape[-1] - 1
    expected_shape = probabilities.shape[:-1] + (num_boundaries,)
    if not torch.is_tensor(boundary_concentration) or not (
        boundary_concentration.is_floating_point()
    ):
        raise TypeError("boundary_concentration must be a floating-point tensor")
    concentration = _canonical_unit_interval(
        boundary_concentration.to(device=probabilities.device, dtype=_PATHS_DTYPE),
        name="PATHS boundary concentration",
    )
    if concentration.shape != expected_shape:
        raise ValueError(
            "boundary_concentration must have shape "
            f"{tuple(expected_shape)}, observed {tuple(concentration.shape)}"
        )

    def _boundary_parameter(value: torch.Tensor, name: str) -> torch.Tensor:
        if not torch.is_tensor(value) or not value.is_floating_point():
            raise TypeError(f"{name} must be a floating-point tensor")
        result = value.to(device=probabilities.device, dtype=_PATHS_DTYPE)
        _require_finite(result, name)
        if result.shape != (num_boundaries,):
            raise ValueError(
                f"{name} must have shape ({num_boundaries},), observed "
                f"{tuple(result.shape)}"
            )
        return result

    thresholds = _boundary_parameter(thresholds, "PATHS transport thresholds")
    slopes = _boundary_parameter(slopes, "PATHS transport slopes")
    gains = _boundary_parameter(gains, "PATHS transport gains")
    if bool(((thresholds < 0.0) | (thresholds > 1.0)).any()):
        raise ValueError("PATHS transport thresholds must lie in [0, 1]")
    if bool((slopes <= 0.0).any()):
        raise ValueError("PATHS transport slopes must be positive")
    if bool((gains < 0.0).any()):
        raise ValueError("PATHS transport gains must be non-negative")

    direction = torch.tanh(slopes * (concentration - thresholds))
    transport_budget = float(strength) * gains * concentration
    upward_odds = transport_budget * (1.0 + direction) * 0.5
    downward_odds = transport_budget * (1.0 - direction) * 0.5
    _require_finite(direction, "PATHS transport direction")
    _require_finite(upward_odds, "PATHS upward transport odds")
    _require_finite(downward_odds, "PATHS downward transport odds")

    zero = torch.zeros_like(upward_odds[..., :1])
    row_up = torch.cat((upward_odds, zero), dim=-1)
    row_down = torch.cat((zero, downward_odds), dim=-1)
    normalizer = 1.0 + row_up + row_down
    diagonal = 1.0 / normalizer
    upper = row_up[..., :-1] / normalizer[..., :-1]
    lower = row_down[..., 1:] / normalizer[..., 1:]
    transport = (
        torch.diag_embed(diagonal)
        + torch.diag_embed(upper, offset=1)
        + torch.diag_embed(lower, offset=-1)
    )
    _require_finite(transport, "PATHS adjacent transport matrix")
    transport_row_error = (transport.sum(dim=-1) - 1.0).abs()
    if bool((transport_row_error > probability_tolerance).any()):
        raise FloatingPointError(
            "PATHS transport matrix is not row stochastic "
            f"(maximum error={float(transport_row_error.max().item()):g})"
        )
    if bool((transport < -probability_tolerance).any()):
        raise FloatingPointError("PATHS transport matrix contains negative mass")
    grade_axis = torch.arange(
        probabilities.shape[-1], device=transport.device
    )
    non_adjacent = (
        grade_axis[:, None] - grade_axis[None, :]
    ).abs() > 1
    if bool((transport[..., non_adjacent] != 0.0).any()):
        raise FloatingPointError("PATHS transport matrix contains a non-adjacent edge")

    transported = torch.matmul(probabilities.unsqueeze(-2), transport).squeeze(-2)
    _require_finite(transported, "PATHS transported class probabilities")
    transported_error = (transported.sum(dim=-1) - 1.0).abs()
    if bool((transported_error > probability_tolerance).any()):
        raise FloatingPointError(
            "PATHS adjacent transport did not conserve posterior mass "
            f"(maximum error={float(transported_error.max().item()):g})"
        )
    if bool((transported < -probability_tolerance).any()):
        raise FloatingPointError("PATHS adjacent transport produced negative mass")

    upper_probability = transport.diagonal(offset=1, dim1=-2, dim2=-1)
    lower_probability = transport.diagonal(offset=-1, dim1=-2, dim2=-1)
    net_flow = (
        probabilities[..., :-1] * upper_probability
        - probabilities[..., 1:] * lower_probability
    )
    _require_finite(net_flow, "PATHS signed net boundary flow")
    reconstructed_delta = torch.cat(
        (
            -net_flow[..., :1],
            net_flow[..., :-1] - net_flow[..., 1:],
            net_flow[..., -1:],
        ),
        dim=-1,
    )
    flow_error = (
        transported - probabilities - reconstructed_delta
    ).abs()
    if bool((flow_error > probability_tolerance).any()):
        raise FloatingPointError(
            "PATHS boundary flows do not replay the transported posterior "
            f"(maximum error={float(flow_error.max().item()):g})"
        )
    return PathsAdjacentTransportDistribution(
        class_probs=transported,
        transport_matrix=transport,
        direction=direction,
        upward_odds=upward_odds,
        downward_odds=downward_odds,
        net_boundary_flow=net_flow,
    )


def continuation_logits_from_log_probs(log_class_probs: torch.Tensor) -> torch.Tensor:
    """Convert a categorical law into at-risk continuation log-odds."""

    if log_class_probs.ndim < 2 or log_class_probs.shape[-1] < 2:
        raise ValueError("log_class_probs must have shape (..., K) with K >= 2")
    if not log_class_probs.is_floating_point():
        raise TypeError("log_class_probs must be floating point")
    if bool(torch.isnan(log_class_probs).any()) or bool(
        torch.isposinf(log_class_probs).any()
    ):
        raise ValueError("log_class_probs must not contain NaN or +inf")

    log_probs = log_class_probs.to(dtype=_PATHS_DTYPE)
    log_tails = torch.flip(
        torch.logcumsumexp(torch.flip(log_probs[..., 1:], dims=(-1,)), dim=-1),
        dims=(-1,),
    )
    logits = log_tails - log_probs[..., :-1]
    _require_finite(logits, "base continuation logits")
    return logits


def decode_continuation_logits(
    continuation_logits: torch.Tensor,
    *,
    probability_tolerance: float = 5e-12,
) -> PathsContinuationDistribution:
    """Decode conditional continuation logits into an FP64 ordinal posterior."""

    if continuation_logits.ndim < 1 or continuation_logits.shape[-1] < 1:
        raise ValueError("continuation_logits must have shape (..., K-1), K >= 2")
    if not continuation_logits.is_floating_point():
        raise TypeError("continuation_logits must be floating point")
    if not math.isfinite(probability_tolerance) or probability_tolerance < 0.0:
        raise ValueError("probability_tolerance must be finite and non-negative")

    logits = continuation_logits.to(dtype=_PATHS_DTYPE)
    _require_finite(logits, "continuation logits")
    log_continue = F.logsigmoid(logits)
    log_stop = F.logsigmoid(-logits)
    cumulative_log = torch.cumsum(log_continue, dim=-1)
    zero = torch.zeros_like(cumulative_log[..., :1])
    reach_log = torch.cat((zero, cumulative_log[..., :-1]), dim=-1)
    log_classes = torch.cat(
        (reach_log + log_stop, cumulative_log[..., -1:]), dim=-1
    )
    probabilities = log_classes.exp()
    row_error = (probabilities.sum(dim=-1) - 1.0).abs()
    if bool((row_error > probability_tolerance).any()):
        raise FloatingPointError(
            "continuation stick-breaking violated probability conservation "
            f"(maximum row-sum error={float(row_error.max().item()):g})"
        )
    _require_finite(probabilities, "PATHS class probabilities")

    cumulative = cumulative_log.exp()
    grade_axis = torch.arange(
        probabilities.shape[-1],
        device=probabilities.device,
        dtype=probabilities.dtype,
    )
    expected = (probabilities * grade_axis).sum(dim=-1)
    median = (cumulative >= 0.5).sum(dim=-1)
    class_map = probabilities.argmax(dim=-1)
    return PathsContinuationDistribution(
        continuation_logits=logits,
        continuation_probs=log_continue.exp(),
        stop_probs=log_stop.exp(),
        class_probs=probabilities,
        log_class_probs=log_classes,
        cumulative_probs=cumulative,
        expected_grade=expected,
        posterior_median=median,
        class_map=class_map,
    )


@dataclass
class PathsScaleSpectrum:
    """Replayable focality ledger for one ORIGIN feature scale."""

    name: str
    normalized_cumulative_atoms: torch.Tensor
    original_valid_mask: torch.Tensor
    active_mask: torch.Tensor
    original_valid_count: torch.Tensor
    local_mass_map: torch.Tensor
    baseline_mass: torch.Tensor
    surviving_mass: torch.Tensor
    local_phi_spectrum: torch.Tensor
    local_concentration_spectrum: torch.Tensor
    concentration_spectrum: torch.Tensor
    mixed_concentration: torch.Tensor
    local_transport_map: torch.Tensor
    metadata: object

    @property
    def local_correction_map(self) -> torch.Tensor:
        """Compatibility alias for the former one-sided implementation."""

        return self.local_transport_map


@dataclass
class PathsOutput:
    """Full PATHS-v3 trace with an unchanged, replayable V3 base ledger."""

    base_output: OriginOutput
    spectrum_evidence: Dict[str, PathsScaleSpectrum]
    probe_z: torch.Tensor
    probe_weights: torch.Tensor
    transport_gains: torch.Tensor
    transport_thresholds: torch.Tensor
    transport_slopes: torch.Tensor
    strength: float
    transport_gain_cap: float
    transport_slope_cap: float
    boundary_concentration: torch.Tensor
    transport_direction: torch.Tensor
    upward_odds: torch.Tensor
    downward_odds: torch.Tensor
    transport_matrix: torch.Tensor
    net_boundary_flow: torch.Tensor
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
    def metadata(self):
        return self.base_output.metadata

    @property
    def local_correction_maps(self) -> Dict[str, torch.Tensor]:
        """Compatibility alias for local concentration-transport maps."""

        return self.local_transport_maps

    @property
    def local_transport_maps(self) -> Dict[str, torch.Tensor]:
        """Additive concentration ledger in ``(N,H,W,K-1)`` layout.

        Summing these maps over scales and spatial cells exactly recovers
        :attr:`boundary_concentration`.  Transport itself is deliberately
        compiled only after this additive evidence ledger is complete.
        """

        return {
            name: evidence.local_transport_map.permute(0, 2, 3, 1)
            for name, evidence in self.spectrum_evidence.items()
        }

    @property
    def local_intervention_scores(self) -> Dict[str, torch.Tensor]:
        return {
            name: evidence.local_transport_map.sum(dim=1)
            for name, evidence in self.spectrum_evidence.items()
        }

    @property
    def concentration_spectra(self) -> Dict[str, torch.Tensor]:
        return {
            name: evidence.concentration_spectrum
            for name, evidence in self.spectrum_evidence.items()
        }

    @property
    def local_concentration_spectra(self) -> Dict[str, torch.Tensor]:
        return {
            name: evidence.local_concentration_spectrum
            for name, evidence in self.spectrum_evidence.items()
        }

    @property
    def local_mass_maps(self) -> Dict[str, torch.Tensor]:
        return {
            name: evidence.local_mass_map
            for name, evidence in self.spectrum_evidence.items()
        }

    @property
    def active_valid_masks(self) -> Dict[str, torch.Tensor]:
        return {name: evidence.active_mask for name, evidence in self.spectrum_evidence.items()}

    @property
    def original_valid_counts(self) -> Dict[str, torch.Tensor]:
        return {
            name: evidence.original_valid_count
            for name, evidence in self.spectrum_evidence.items()
        }

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

    # Compatibility aliases for earlier trainer/audit code.  They do not
    # change the v3 signed-transport semantics.
    @property
    def gains(self) -> torch.Tensor:
        return self.transport_gains

    @property
    def correction_cap(self) -> float:
        return self.transport_gain_cap

    @property
    def advance_gains(self) -> torch.Tensor:
        return self.transport_gains

    @property
    def stop_gains(self) -> torch.Tensor:
        return self.transport_gains

    @property
    def advance_correction(self) -> torch.Tensor:
        return self.boundary_correction.clamp_min(0.0)

    @property
    def stop_correction(self) -> torch.Tensor:
        return (-self.boundary_correction).clamp_min(0.0)


def _compile_scale_spectrum(
    evidence: PathsScaleSpectrum,
    *,
    scale_weights: torch.Tensor,
    probe_weights: torch.Tensor,
) -> PathsScaleSpectrum:
    """Reaggregate a stored local ledger without changing its denominator."""

    local = evidence.local_concentration_spectrum
    concentration = local.sum(dim=(2, 3))
    mixed = torch.einsum("nbl,bl->nb", concentration, probe_weights)
    per_cell_mixed = torch.einsum("nbhwl,bl->nbhw", local, probe_weights)
    local_transport = per_cell_mixed * scale_weights[None, :, None, None]
    _require_finite(concentration, f"PATHS concentration spectrum at {evidence.name}")
    _require_finite(local_transport, f"PATHS local transport at {evidence.name}")
    return replace(
        evidence,
        surviving_mass=evidence.local_mass_map.sum(dim=(-2, -1)),
        concentration_spectrum=concentration,
        mixed_concentration=mixed,
        local_transport_map=local_transport,
    )


def _paths_output_from_ledgers(
    base_output: OriginOutput,
    spectrum_evidence: Dict[str, PathsScaleSpectrum],
    *,
    probe_z: torch.Tensor,
    probe_weights: torch.Tensor,
    transport_gains: torch.Tensor,
    transport_thresholds: torch.Tensor,
    transport_slopes: torch.Tensor,
    strength: float,
    transport_gain_cap: float,
    transport_slope_cap: float,
) -> PathsOutput:
    if not spectrum_evidence:
        raise ValueError("PATHS requires at least one scale spectrum")
    concentration = sum(
        item.local_transport_map.sum(dim=(-2, -1))
        for item in spectrum_evidence.values()
    )
    tolerance = 128.0 * torch.finfo(concentration.dtype).eps
    if bool((concentration < -tolerance).any()) or bool(
        (concentration > 1.0 + tolerance).any()
    ):
        raise FloatingPointError("PATHS concentration left [0,1]")

    base_logits = continuation_logits_from_log_probs(base_output.log_class_probs)
    transported = apply_signed_adjacent_transport(
        base_output.class_probs,
        concentration,
        transport_thresholds,
        transport_slopes,
        transport_gains,
        strength=strength,
    )
    if bool((transported.class_probs <= 0.0).any()):
        raise FloatingPointError(
            "PATHS transported posterior must be strictly positive for exact "
            "continuation factorization"
        )
    final_logits = continuation_logits_from_log_probs(transported.class_probs.log())
    decoded = decode_continuation_logits(final_logits)
    factorization_error = (decoded.class_probs - transported.class_probs).abs()
    factorization_tolerance = 5e-12 + 5e-12 * transported.class_probs.abs()
    if bool((factorization_error > factorization_tolerance).any()):
        raise FloatingPointError(
            "PATHS continuation factorization did not replay transported posterior "
            f"(maximum error={float(factorization_error.max().item()):g})"
        )
    correction = final_logits - base_logits
    return PathsOutput(
        base_output=base_output,
        spectrum_evidence=spectrum_evidence,
        probe_z=probe_z,
        probe_weights=probe_weights,
        transport_gains=transport_gains,
        transport_thresholds=transport_thresholds,
        transport_slopes=transport_slopes,
        strength=float(strength),
        transport_gain_cap=float(transport_gain_cap),
        transport_slope_cap=float(transport_slope_cap),
        boundary_concentration=concentration,
        transport_direction=transported.direction,
        upward_odds=transported.upward_odds,
        downward_odds=transported.downward_odds,
        transport_matrix=transported.transport_matrix,
        net_boundary_flow=transported.net_boundary_flow,
        boundary_correction=correction,
        base_continuation_logits=base_logits,
        continuation_logits=decoded.continuation_logits,
        continuation_probs=decoded.continuation_probs,
        stop_probs=decoded.stop_probs,
        class_probs=decoded.class_probs,
        log_class_probs=decoded.log_class_probs,
        cumulative_probs=decoded.cumulative_probs,
        expected_grade=decoded.expected_grade,
        posterior_median=decoded.posterior_median,
        class_map=decoded.class_map,
    )


class PathsContinuationRefiner(nn.Module):
    """Compile normalized ORIGIN atoms into signed adjacent transport."""

    def __init__(
        self,
        num_boundaries: int,
        evidence_scales: Sequence[str],
        *,
        probe_z: Sequence[float] = _DEFAULT_PROBES,
        transport_gain_cap: float = 1.0,
        transport_gain_init: float = 0.05,
        transport_threshold_init: float = 0.5,
        transport_slope_init: float = 2.0,
        transport_slope_cap: float = 8.0,
        strength: float = 1.0,
        correction_cap: Optional[float] = None,
        gain_init: Optional[float] = None,
    ) -> None:
        super().__init__()
        if num_boundaries < 1:
            raise ValueError("num_boundaries must be positive")
        scales = tuple(str(value) for value in evidence_scales)
        if not scales or len(scales) != len(set(scales)):
            raise ValueError("evidence_scales must be non-empty and unique")
        probes = _canonical_probe_values(probe_z)
        # Transitional aliases keep old checkpoints/tools readable while all
        # v3 metadata and public model kwargs use the transport terminology.
        if correction_cap is not None:
            transport_gain_cap = float(correction_cap)
        if gain_init is not None:
            transport_gain_init = float(gain_init)
        if not math.isfinite(transport_gain_cap) or transport_gain_cap <= 0.0:
            raise ValueError("transport_gain_cap must be finite and positive")
        if not math.isfinite(transport_slope_cap) or transport_slope_cap <= 0.0:
            raise ValueError("transport_slope_cap must be finite and positive")
        if not math.isfinite(transport_threshold_init) or not (
            0.0 < transport_threshold_init < 1.0
        ):
            raise ValueError("transport_threshold_init must lie strictly in (0, 1)")
        if not math.isfinite(strength) or not 0.0 <= strength <= 1.0:
            raise ValueError("strength must lie in [0, 1]")
        gain_raw = _inverse_bounded_sigmoid(
            transport_gain_init, transport_gain_cap, name="PATHS transport gain"
        )
        threshold_raw = _inverse_bounded_sigmoid(
            transport_threshold_init, 1.0, name="PATHS transport threshold"
        )
        slope_raw = _inverse_bounded_sigmoid(
            transport_slope_init, transport_slope_cap, name="PATHS transport slope"
        )

        self.num_boundaries = int(num_boundaries)
        self.evidence_scales = scales
        self.transport_gain_cap = float(transport_gain_cap)
        self.transport_gain_init = float(transport_gain_init)
        self.transport_threshold_init = float(transport_threshold_init)
        self.transport_slope_init = float(transport_slope_init)
        self.transport_slope_cap = float(transport_slope_cap)
        self.strength = float(strength)
        self.register_buffer("probe_z", torch.tensor(probes, dtype=torch.float64))
        self.probe_logits = nn.Parameter(torch.zeros(self.num_boundaries, len(probes)))
        self.raw_transport_gains = nn.Parameter(
            torch.full((self.num_boundaries,), gain_raw)
        )
        self.raw_transport_thresholds = nn.Parameter(
            torch.full((self.num_boundaries,), threshold_raw)
        )
        self.raw_transport_slopes = nn.Parameter(
            torch.full((self.num_boundaries,), slope_raw)
        )

    @property
    def probe_weights(self) -> torch.Tensor:
        return self.probe_logits.double().softmax(dim=-1)

    @property
    def transport_gains(self) -> torch.Tensor:
        return self.transport_gain_cap * self.raw_transport_gains.double().sigmoid()

    @property
    def transport_thresholds(self) -> torch.Tensor:
        return self.raw_transport_thresholds.double().sigmoid()

    @property
    def transport_slopes(self) -> torch.Tensor:
        return self.transport_slope_cap * self.raw_transport_slopes.double().sigmoid()

    @property
    def raw_gains(self) -> torch.Tensor:
        return self.raw_transport_gains

    @property
    def gains(self) -> torch.Tensor:
        return self.transport_gains

    @property
    def correction_cap(self) -> float:
        return self.transport_gain_cap

    @property
    def gain_init(self) -> float:
        return self.transport_gain_init

    @property
    def advance_gains(self) -> torch.Tensor:
        return self.transport_gains

    @property
    def stop_gains(self) -> torch.Tensor:
        return self.transport_gains

    def forward(self, base_output: OriginOutput) -> Union[OriginOutput, PathsOutput]:
        if self.strength == 0.0:
            return base_output
        if base_output.atom_mode != "cumulative":
            raise ValueError("PATHS primary architecture requires cumulative V3 atoms")
        if tuple(base_output.scale_evidence) != self.evidence_scales:
            raise ValueError(
                "PATHS scale order differs from the V3 ledger: "
                f"{tuple(base_output.scale_evidence)} vs {self.evidence_scales}"
            )
        if base_output.total_rates.shape[-1] != self.num_boundaries:
            raise ValueError("V3 boundary count differs from PATHS")

        probe_weights = self.probe_weights
        transport_gains = self.transport_gains
        transport_thresholds = self.transport_thresholds
        transport_slopes = self.transport_slopes
        scale_weights = base_output.scale_simplex.to(dtype=_PATHS_DTYPE)
        _require_finite(scale_weights, "V3 scale simplex")
        if scale_weights.shape != (len(self.evidence_scales), self.num_boundaries):
            raise ValueError("V3 scale simplex has an incompatible shape")

        spectra: Dict[str, PathsScaleSpectrum] = {}
        for scale_index, name in enumerate(self.evidence_scales):
            source = base_output.scale_evidence[name]
            valid = source.valid_mask.bool()
            normalized = _canonical_unit_interval(
                source.compiled_atoms / float(base_output.atom_mass_cap),
                name=f"normalized cumulative atoms at {name}",
            ).to(dtype=_PATHS_DTYPE)
            _require_nested_cumulative_atoms(
                normalized, valid, name=f"normalized cumulative atoms at {name}"
            )
            local_phi = tangent_removed_pgf_probe(normalized, self.probe_z)
            valid_atoms = valid[:, None]
            valid_spectrum = valid[:, None, :, :, None]
            local_mass = torch.where(valid_atoms, normalized, torch.zeros_like(normalized))
            local_phi = torch.where(
                valid_spectrum, local_phi, torch.zeros_like(local_phi)
            )
            baseline_mass = local_mass.sum(dim=(-2, -1))
            # A zero-evidence boundary has an exactly zero spectrum.  Do not
            # divide that inactive branch by ``finfo(float64).tiny`` and mask
            # it afterwards: the division backward contains ``tiny**2``,
            # which underflows to zero and can create a hidden 0/0 NaN before
            # ``where`` suppresses its value.  A unit denominator is the
            # exact neutral extension on the zero-mass branch: both its value
            # and gradient remain zero, while positive masses retain the
            # defining mass normalization unchanged.
            positive_mass = baseline_mass > 0.0
            safe_mass = torch.where(
                positive_mass,
                baseline_mass,
                torch.ones_like(baseline_mass),
            )
            local_concentration = local_phi / safe_mass[:, :, None, None, None]
            local_concentration = torch.where(
                positive_mass[:, :, None, None, None],
                local_concentration,
                torch.zeros_like(local_concentration),
            )
            initial = PathsScaleSpectrum(
                name=name,
                normalized_cumulative_atoms=normalized,
                original_valid_mask=valid,
                active_mask=valid,
                original_valid_count=valid.sum(dim=(-2, -1)),
                local_mass_map=local_mass,
                baseline_mass=baseline_mass,
                surviving_mass=baseline_mass,
                local_phi_spectrum=local_phi,
                local_concentration_spectrum=local_concentration,
                concentration_spectrum=torch.empty(0, device=normalized.device),
                mixed_concentration=torch.empty(0, device=normalized.device),
                local_transport_map=torch.empty(0, device=normalized.device),
                metadata=source.metadata,
            )
            spectra[name] = _compile_scale_spectrum(
                initial,
                scale_weights=scale_weights[scale_index],
                probe_weights=probe_weights,
            )

        return _paths_output_from_ledgers(
            base_output,
            spectra,
            probe_z=self.probe_z,
            probe_weights=probe_weights,
            transport_gains=transport_gains,
            transport_thresholds=transport_thresholds,
            transport_slopes=transport_slopes,
            strength=self.strength,
            transport_gain_cap=self.transport_gain_cap,
            transport_slope_cap=self.transport_slope_cap,
        )


@dataclass
class PathsInterventionOutput:
    """Exact joint deletion of an ORIGIN rate and PATHS focality ledger."""

    baseline: PathsOutput
    output: PathsOutput
    removal_masks: Dict[str, torch.Tensor]
    removed_rates: torch.Tensor
    removed_boundary_concentration: torch.Tensor
    removed_net_boundary_flow: torch.Tensor
    removed_boundary_correction: torch.Tensor
    removed_phi_sums: Dict[str, torch.Tensor]
    removed_mass_sums: Dict[str, torch.Tensor]
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
    def removed_signed_correction(self) -> torch.Tensor:
        return self.removed_boundary_correction

    @property
    def removed_signed_boundary_flow(self) -> torch.Tensor:
        return self.removed_net_boundary_flow

    @property
    def removed_advance_correction(self) -> torch.Tensor:
        return self.removed_boundary_correction

    @property
    def removed_stop_correction(self) -> torch.Tensor:
        return torch.zeros_like(self.removed_boundary_correction)


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


def replay_paths_without(
    output: PathsOutput,
    removal_masks: Mapping[str, torch.Tensor],
    *,
    force_decoder_fp64: bool = True,
) -> PathsInterventionOutput:
    """Delete stored cells and exactly replay the complete PATHS classifier."""

    if not force_decoder_fp64:
        raise ValueError("PATHS structural decoders must run in FP64")
    unknown = set(removal_masks) - set(output.spectrum_evidence)
    if unknown:
        raise ValueError(f"removal masks contain unknown scales {sorted(unknown)}")

    canonical: Dict[str, torch.Tensor] = {}
    for name, evidence in output.spectrum_evidence.items():
        requested = removal_masks.get(name)
        if requested is None:
            mask = torch.zeros_like(evidence.active_mask)
        else:
            mask = _canonical_removal_mask(requested, evidence.active_mask, name=name)
        canonical[name] = mask & evidence.active_mask

    base_intervention = replay_origin_without(
        output.base_output, canonical, force_decoder_fp64=True
    )

    replayed: Dict[str, PathsScaleSpectrum] = {}
    removed_phi: Dict[str, torch.Tensor] = {}
    removed_mass: Dict[str, torch.Tensor] = {}
    removed_concentration_by_scale: list[torch.Tensor] = []
    for scale_index, (name, evidence) in enumerate(output.spectrum_evidence.items()):
        mask = canonical[name]
        spectrum_mask = mask[:, None, :, :, None]
        atom_mask = mask[:, None]
        transport_mask = mask[:, None]
        removed_phi[name] = torch.where(
            spectrum_mask,
            evidence.local_phi_spectrum,
            torch.zeros_like(evidence.local_phi_spectrum),
        ).sum(dim=(2, 3))
        removed_mass[name] = torch.where(
            atom_mask,
            evidence.local_mass_map,
            torch.zeros_like(evidence.local_mass_map),
        ).sum(dim=(-2, -1))
        # This is the independently accumulated additive ledger removed by
        # the intervention.  Do not define it as ``baseline - replay``: that
        # would make the replay certificate tautological.
        removed_concentration_by_scale.append(
            torch.where(
                transport_mask,
                evidence.local_transport_map,
                torch.zeros_like(evidence.local_transport_map),
            ).sum(dim=(-2, -1))
        )

        kept_phi = torch.where(
            spectrum_mask,
            torch.zeros_like(evidence.local_phi_spectrum),
            evidence.local_phi_spectrum,
        )
        kept_concentration = torch.where(
            spectrum_mask,
            torch.zeros_like(evidence.local_concentration_spectrum),
            evidence.local_concentration_spectrum,
        )
        kept_mass = torch.where(
            atom_mask,
            torch.zeros_like(evidence.local_mass_map),
            evidence.local_mass_map,
        )
        active = evidence.active_mask & ~mask
        provisional = replace(
            evidence,
            active_mask=active,
            local_mass_map=kept_mass,
            local_phi_spectrum=kept_phi,
            local_concentration_spectrum=kept_concentration,
        )
        replayed[name] = _compile_scale_spectrum(
            provisional,
            scale_weights=output.scale_simplex[scale_index].to(dtype=_PATHS_DTYPE),
            probe_weights=output.probe_weights,
        )

    replay_output = _paths_output_from_ledgers(
        base_intervention.output,
        replayed,
        probe_z=output.probe_z,
        probe_weights=output.probe_weights,
        transport_gains=output.transport_gains,
        transport_thresholds=output.transport_thresholds,
        transport_slopes=output.transport_slopes,
        strength=output.strength,
        transport_gain_cap=output.transport_gain_cap,
        transport_slope_cap=output.transport_slope_cap,
    )
    removed_correction = output.boundary_correction - replay_output.boundary_correction
    removed_concentration = sum(removed_concentration_by_scale)
    return PathsInterventionOutput(
        baseline=output,
        output=replay_output,
        removal_masks=canonical,
        removed_rates=base_intervention.removed_rates,
        removed_boundary_concentration=removed_concentration,
        removed_net_boundary_flow=(
            output.net_boundary_flow - replay_output.net_boundary_flow
        ),
        removed_boundary_correction=removed_correction,
        removed_phi_sums=removed_phi,
        removed_mass_sums=removed_mass,
        base_intervention=base_intervention,
    )


class PathsModel(OriginModel):
    """ORIGIN-v3 with signed adjacent probability transport (SAPT)."""

    def __init__(
        self,
        *,
        paths_probe_z: Sequence[float] = _DEFAULT_PROBES,
        paths_transport_gain_cap: float = 1.0,
        paths_transport_gain_init: float = 0.05,
        paths_transport_threshold_init: float = 0.5,
        paths_transport_slope_init: float = 2.0,
        paths_transport_slope_cap: float = 8.0,
        paths_strength: float = 1.0,
        paths_correction_cap: Optional[float] = None,
        paths_gain_init: Optional[float] = None,
        **origin_kwargs,
    ) -> None:
        super().__init__(**origin_kwargs)
        self.paths_refiner = PathsContinuationRefiner(
            self.num_classes - 1,
            self.evidence_scales,
            probe_z=paths_probe_z,
            transport_gain_cap=paths_transport_gain_cap,
            transport_gain_init=paths_transport_gain_init,
            transport_threshold_init=paths_transport_threshold_init,
            transport_slope_init=paths_transport_slope_init,
            transport_slope_cap=paths_transport_slope_cap,
            strength=paths_strength,
            correction_cap=paths_correction_cap,
            gain_init=paths_gain_init,
        )

    def load_origin_v3_state_dict(
        self,
        source_state: Mapping[str, torch.Tensor],
    ) -> None:
        """Strictly load an audited V3 state while leaving PATHS at init."""

        current = self.state_dict()
        paths_prefix = "paths_refiner."
        expected_base = {key for key in current if not key.startswith(paths_prefix)}
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
        expected_paths = {key for key in current if key.startswith(paths_prefix)}
        if set(incompatible.missing_keys) != expected_paths or incompatible.unexpected_keys:
            raise RuntimeError(
                "internal PATHS warm-start invariant failed: "
                f"missing={incompatible.missing_keys}, "
                f"unexpected={incompatible.unexpected_keys}"
            )

    def architecture_metadata(self) -> Dict[str, object]:
        base = super().architecture_metadata()
        return {
            **base,
            "name": "PATHS-v3-SAPT",
            "architecture_schema": "paths-signed-adjacent-transport-v3",
            "base_architecture": "ORIGIN-v3-bounded-rate",
            "posterior_path": (
                "nested_local_atoms_to_v3_ctmc_then_mass_normalized_"
                "tangent_removed_pgf_focality_to_signed_adjacent_markov_transport"
            ),
            "pgf_probes": self.paths_refiner.probe_z.detach().cpu().tolist(),
            "probe_parameterization": "fixed_z_boundary_simplex_v3",
            "focality_transform": "tangent_removed_bernoulli_log_pgf_v2",
            "normalization": "frozen_baseline_atom_mass_per_scale_boundary",
            "transport_parameterization": (
                "bounded_gain_threshold_slope_signed_adjacent_odds_v3"
            ),
            "transport_gain_cap": self.paths_refiner.transport_gain_cap,
            "transport_gain_init": self.paths_refiner.transport_gain_init,
            "transport_threshold_init": (
                self.paths_refiner.transport_threshold_init
            ),
            "transport_slope_init": self.paths_refiner.transport_slope_init,
            "transport_slope_cap": self.paths_refiner.transport_slope_cap,
            "strength": self.paths_refiner.strength,
            "transport_decoder": "fp64_row_stochastic_tridiagonal_markov_v1",
            "continuation_decoder": "fp64_replay_of_transported_posterior_v1",
            "local_transport_layout": "NHW(K-1)",
            "mass_conservation": "exact_row_stochastic_posterior_transport",
            "maximum_grade_jump": 1,
            "intervention": (
                "joint_stored_rate_and_frozen_mass_focality_ledger_deletion_"
                "with_exact_transport_recompilation"
            ),
            "no_classifier_bypass": True,
        }

    def forward(
        self,
        images: torch.Tensor,
        pixel_valid_mask: Optional[torch.Tensor] = None,
        *,
        force_decoder_fp64: bool = True,
        return_encoder_maps: bool = False,
    ) -> Union[OriginOutput, PathsOutput]:
        encoded = self.encoder(images, pixel_valid_mask)
        base = self.generator(
            encoded,
            force_decoder_fp64=force_decoder_fp64,
            retain_encoder_output=return_encoder_maps,
        )
        return self.paths_refiner(base)

    def replay_without(
        self,
        output: Union[OriginOutput, PathsOutput],
        removal_masks: Mapping[str, torch.Tensor],
        *,
        force_decoder_fp64: bool = True,
    ) -> Union[OriginInterventionOutput, PathsInterventionOutput]:
        if isinstance(output, PathsOutput):
            return replay_paths_without(
                output, removal_masks, force_decoder_fp64=force_decoder_fp64
            )
        return replay_origin_without(
            output, removal_masks, force_decoder_fp64=force_decoder_fp64
        )

    def topk_intervention(self, *args, **kwargs):
        raise NotImplementedError(
            "PATHS requires joint rate-and-focality selection; use replay_without "
            "with an explicit spatial mask"
        )


def build_paths_model(**kwargs) -> PathsModel:
    return PathsModel(**kwargs)


__all__ = [
    "PathsAdjacentTransportDistribution",
    "PathsContinuationDistribution",
    "PathsContinuationRefiner",
    "PathsInterventionOutput",
    "PathsModel",
    "PathsOutput",
    "PathsScaleSpectrum",
    "apply_signed_adjacent_transport",
    "build_paths_model",
    "continuation_logits_from_log_probs",
    "decode_continuation_logits",
    "pgf_probe",
    "replay_paths_without",
    "tangent_removed_pgf_probe",
]
