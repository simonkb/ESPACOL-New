"""Matched prediction-path controls for the frozen ORIGIN-v3 ablation.

The controls deliberately reuse the same ConvNeXt pyramid, preprocessing and
training objective.  They differ only in the named mechanism under test:

* three conventional globally pooled ordinal heads;
* a sequential-hazard decoder applied to the *identical* ORIGIN local ledger;
* structural ORIGIN variants configured by :class:`OriginAblationConfig`.

Every posterior exposed to the loss is FP64.  Pooled controls explicitly do
not claim a replayable local evidence ledger; the sequential-hazard control
does, and its deletion replay reuses ORIGIN's stored local contributions.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from .origin import (
    OriginInterventionOutput,
    OriginModel,
    OriginOutput,
    decode_pure_birth_rates,
)
from .origin_encoder import ConvNeXtTinyPyramidEncoder, OriginEncoderOutput


_ALL_SCALES = ("s4", "s8", "s16", "s32")
_POOLED_VARIANTS = {
    "pooled_softmax",
    "pooled_cumulative_logit",
    "pooled_conditional",
}


def _cfg_get(cfg: object, name: str, default: Any = None) -> Any:
    if isinstance(cfg, Mapping):
        return cfg.get(name, default)
    return getattr(cfg, name, default)


def _require_finite(tensor: torch.Tensor, name: str) -> None:
    if not bool(torch.isfinite(tensor).all()):
        raise FloatingPointError(f"{name} contains non-finite values")


def _posterior_statistics(
    class_probs: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Derive all public posterior fields without changing the distribution."""

    if class_probs.ndim != 2 or class_probs.shape[1] < 2:
        raise ValueError("class_probs must have shape (N,K), K >= 2")
    if class_probs.dtype != torch.float64:
        raise TypeError("ablation posterior must be FP64")
    _require_finite(class_probs, "ablation class posterior")
    if bool((class_probs < 0).any()):
        raise FloatingPointError("ablation class posterior contains negative mass")
    error = (class_probs.sum(dim=-1) - 1.0).abs().max()
    if float(error.detach()) > 5e-12:
        raise FloatingPointError(
            "ablation class posterior is not normalized: "
            f"maximum row error={float(error.detach()):.3g}"
        )
    cumulative = class_probs[:, 1:].flip((-1,)).cumsum(dim=-1).flip((-1,))
    grades = torch.arange(
        class_probs.shape[-1], device=class_probs.device, dtype=torch.float64
    )
    expected = (class_probs * grades).sum(dim=-1)
    median = (cumulative >= 0.5).sum(dim=-1)
    class_map = class_probs.argmax(dim=-1)
    log_probs = class_probs.clamp_min(torch.finfo(torch.float64).tiny).log()
    return log_probs, cumulative, expected, median, class_map


def _classes_from_cumulative(cumulative: torch.Tensor) -> torch.Tensor:
    """Convert nested ``P(Y>k)`` values into an exact categorical posterior."""

    if cumulative.ndim != 2 or cumulative.shape[1] < 1:
        raise ValueError("cumulative must have shape (N,K-1)")
    if cumulative.dtype != torch.float64:
        raise TypeError("cumulative posterior must be FP64")
    _require_finite(cumulative, "cumulative posterior")
    if bool(((cumulative < 0.0) | (cumulative > 1.0)).any()):
        raise FloatingPointError("cumulative posterior lies outside [0,1]")
    if cumulative.shape[1] > 1 and bool(
        (cumulative[:, 1:] > cumulative[:, :-1]).any()
    ):
        raise FloatingPointError("cumulative posterior is not nested")
    return torch.cat(
        (
            1.0 - cumulative[:, :1],
            cumulative[:, :-1] - cumulative[:, 1:],
            cumulative[:, -1:],
        ),
        dim=-1,
    )


def _equivalent_boundary_rates(cumulative: torch.Tensor) -> torch.Tensor:
    """Nonnegative telemetry only; never used to form a pooled prediction."""

    one_minus = (1.0 - cumulative).clamp_min(torch.finfo(torch.float64).tiny)
    return -one_minus.log()


def nominal_cumulative_origin_posterior(
    *,
    num_classes: int,
    reference_count: float,
    atom_rate_init: float,
    prior_rate_init: float,
    boundary_scale_init: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the nominal full-ORIGIN initial rates and endpoint posterior.

    At initialization every incremental severity atom has mass
    ``atom_rate_init``. Reverse-cumulative compilation therefore gives
    multipliers ``[K-1, ..., 1]``. Scale-simplex weights sum to one and each
    scale's conserved geometry sums to ``reference_count``, so this expression
    is independent of the number and resolution of the selected scales.
    """

    boundaries = int(num_classes) - 1
    if boundaries < 1:
        raise ValueError("num_classes must be at least two")
    values = (
        float(reference_count),
        float(atom_rate_init),
        float(prior_rate_init),
        float(boundary_scale_init),
    )
    if not all(math.isfinite(value) and value > 0.0 for value in values):
        raise ValueError("nominal ORIGIN initialization values must be positive")
    multiplicity = torch.arange(
        boundaries, 0, -1, dtype=torch.float64
    )
    rates = (
        float(prior_rate_init)
        + float(reference_count)
        * float(boundary_scale_init)
        * float(atom_rate_init)
        * multiplicity
    )
    decoded = decode_pure_birth_rates(rates.unsqueeze(0), force_fp64=True)
    return rates, decoded.class_probs[0]


def _inverse_softplus(value: torch.Tensor) -> torch.Tensor:
    if bool((value <= 0).any()):
        raise ValueError("inverse-softplus input must be positive")
    return value + torch.log(-torch.expm1(-value))


@dataclass
class AblationPosteriorOutput:
    """Uniform output contract for controls without a spatial rate ledger."""

    class_probs: torch.Tensor
    log_class_probs: torch.Tensor
    cumulative_probs: torch.Tensor
    expected_grade: torch.Tensor
    posterior_median: torch.Tensor
    class_map: torch.Tensor
    total_rates: torch.Tensor
    pooled_logits: torch.Tensor
    decoder_aux: Dict[str, torch.Tensor]

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
    def local_rate_maps(self) -> Dict[str, torch.Tensor]:
        return {}

    @property
    def valid_masks(self) -> Dict[str, torch.Tensor]:
        return {}

    @property
    def metadata(self) -> Dict[str, object]:
        return {}


class MultiScaleMaskedPoolingModel(nn.Module):
    """Matched global-pooling controls over all four native encoder stages."""

    supports_exact_replay = False

    def __init__(
        self,
        *,
        variant: str,
        num_classes: int = 5,
        projection_dim: int = 128,
        pretrained: bool = True,
        dropout: float = 0.0,
        reference_count: float = 4096.0,
        atom_rate_init: float = 1e-6,
        prior_rate_init: float = 1e-4,
        boundary_scale_init: float = 1.0,
        mask_valid_fraction: float = 0.5,
        grad_checkpoint: bool = False,
        encoder: Optional[nn.Module] = None,
    ) -> None:
        super().__init__()
        if variant not in _POOLED_VARIANTS:
            raise ValueError(f"unknown pooled control: {variant!r}")
        if num_classes < 2 or projection_dim < 1:
            raise ValueError("num_classes and projection_dim must be positive")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must lie in [0,1)")
        if encoder is None:
            encoder = ConvNeXtTinyPyramidEncoder(
                pretrained=pretrained,
                grad_checkpoint=grad_checkpoint,
                mask_valid_fraction=mask_valid_fraction,
            )
        if not hasattr(encoder, "STAGE_CHANNELS"):
            raise TypeError("encoder must expose STAGE_CHANNELS")
        missing = set(_ALL_SCALES) - set(encoder.STAGE_CHANNELS)
        if missing:
            raise ValueError(f"encoder is missing pooled-control scales {sorted(missing)}")

        self.encoder = encoder
        self.variant = variant
        self.num_classes = int(num_classes)
        self.num_boundaries = self.num_classes - 1
        self.evidence_scales = _ALL_SCALES
        self.projection_dim = int(projection_dim)
        self.scale_projections = nn.ModuleDict(
            {
                name: nn.Sequential(
                    nn.LayerNorm(int(encoder.STAGE_CHANNELS[name])),
                    nn.Linear(int(encoder.STAGE_CHANNELS[name]), projection_dim),
                    nn.GELU(),
                )
                for name in _ALL_SCALES
            }
        )
        self.fusion = nn.Sequential(
            nn.LayerNorm(projection_dim * len(_ALL_SCALES)),
            nn.Linear(projection_dim * len(_ALL_SCALES), projection_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        output_dim = self.num_classes if variant == "pooled_softmax" else self.num_boundaries
        self.posterior_head = nn.Linear(projection_dim, output_dim)

        if variant == "pooled_cumulative_logit":
            # One shared severity coordinate and strictly ordered thresholds.
            self.posterior_head = nn.Linear(projection_dim, 1)
            self.threshold_start = nn.Parameter(torch.tensor(-1.5))
            if self.num_boundaries > 1:
                inverse_softplus_one = math.log(math.expm1(1.0))
                self.threshold_gap_logits = nn.Parameter(
                    torch.full((self.num_boundaries - 1,), inverse_softplus_one)
                )
            else:
                self.register_parameter("threshold_gap_logits", None)
        else:
            self.register_parameter("threshold_start", None)
            self.register_parameter("threshold_gap_logits", None)

        nominal_rates, nominal_probs = nominal_cumulative_origin_posterior(
            num_classes=self.num_classes,
            reference_count=reference_count,
            atom_rate_init=atom_rate_init,
            prior_rate_init=prior_rate_init,
            boundary_scale_init=boundary_scale_init,
        )
        self.register_buffer("nominal_origin_total_rates", nominal_rates)
        self.register_buffer("nominal_origin_class_probs", nominal_probs)
        self._initialize_posterior_head_to_origin_prior()

    def _initialize_posterior_head_to_origin_prior(self) -> None:
        """Make every pooled control emit the same nominal law as ORIGIN."""

        probs = self.nominal_origin_class_probs.to(torch.float64)
        cumulative = probs[1:].flip((0,)).cumsum(dim=0).flip((0,))
        with torch.no_grad():
            self.posterior_head.weight.zero_()
            if self.variant == "pooled_softmax":
                self.posterior_head.bias.copy_(probs.log().to(self.posterior_head.bias))
            elif self.variant == "pooled_cumulative_logit":
                self.posterior_head.bias.zero_()
                thresholds = -torch.logit(cumulative)
                if self.threshold_start is None:
                    raise RuntimeError("cumulative-logit head has no threshold start")
                self.threshold_start.copy_(thresholds[0].to(self.threshold_start))
                if self.threshold_gap_logits is not None:
                    gaps = thresholds[1:] - thresholds[:-1]
                    self.threshold_gap_logits.copy_(
                        _inverse_softplus(gaps).to(self.threshold_gap_logits)
                    )
            else:
                continuation = torch.cat(
                    (cumulative[:1], cumulative[1:] / cumulative[:-1])
                )
                self.posterior_head.bias.copy_(
                    torch.logit(continuation).to(self.posterior_head.bias)
                )

    @staticmethod
    def _masked_mean(features: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        if valid_mask.dtype != torch.bool:
            valid_mask = valid_mask.bool()
        if valid_mask.shape != features.shape[:1] + features.shape[-2:]:
            raise ValueError("pooled-control valid mask shape mismatch")
        weights = valid_mask[:, None].to(features.dtype)
        count = weights.sum(dim=(-2, -1)).clamp_min(1.0)
        return (features * weights).sum(dim=(-2, -1)) / count

    def _decode(self, fused: torch.Tensor) -> AblationPosteriorOutput:
        # Keep the conventional head under the surrounding precision policy,
        # then cast logits once before all posterior arithmetic.
        raw = self.posterior_head(fused)
        logits = raw.to(torch.float64)
        aux: Dict[str, torch.Tensor] = {}
        if self.variant == "pooled_softmax":
            class_probs = logits.softmax(dim=-1)
        elif self.variant == "pooled_cumulative_logit":
            if self.threshold_start is None:
                raise RuntimeError("cumulative-logit head has no threshold start")
            first = self.threshold_start.to(torch.float64).reshape(1)
            if self.threshold_gap_logits is None:
                thresholds = first
            else:
                gaps = torch.nn.functional.softplus(
                    self.threshold_gap_logits.to(torch.float64)
                )
                thresholds = torch.cat((first, first + gaps.cumsum(dim=0)))
            cumulative = torch.sigmoid(logits - thresholds[None])
            class_probs = _classes_from_cumulative(cumulative)
            aux["ordered_thresholds"] = thresholds
        else:
            continuation = torch.sigmoid(logits)
            cumulative = continuation.cumprod(dim=-1)
            class_probs = _classes_from_cumulative(cumulative)
            aux["continuation_probabilities"] = continuation

        log_probs, cumulative, expected, median, class_map = _posterior_statistics(
            class_probs
        )
        return AblationPosteriorOutput(
            class_probs=class_probs,
            log_class_probs=log_probs,
            cumulative_probs=cumulative,
            expected_grade=expected,
            posterior_median=median,
            class_map=class_map,
            total_rates=_equivalent_boundary_rates(cumulative),
            pooled_logits=logits,
            decoder_aux=aux,
        )

    def architecture_metadata(self) -> Dict[str, object]:
        decoder = {
            "pooled_softmax": "categorical_softmax",
            "pooled_cumulative_logit": "shared_score_ordered_cumulative_logits",
            "pooled_conditional": "sequential_conditional_continuation",
        }[self.variant]
        return {
            "name": "ORIGIN-fold9-ablation-control",
            "ablation_variant": self.variant,
            "control_family": "multiscale_masked_global_pooling",
            "encoder": "convnext_tiny",
            "num_classes": self.num_classes,
            "consumed_scales": list(_ALL_SCALES),
            "pooling": "valid_masked_arithmetic_mean_per_scale",
            "fusion": "concat_projected_scales",
            "posterior_path": f"masked_pool_fusion_to_{decoder}",
            "posterior_dtype": "float64",
            "sensitive_head_compute_dtype": "float32_autocast_disabled",
            "initialization": "matched_nominal_cumulative_origin_posterior_v1",
            "nominal_origin_total_rates": self.nominal_origin_total_rates.detach().cpu().tolist(),
            "nominal_origin_class_probs": self.nominal_origin_class_probs.detach().cpu().tolist(),
            "total_rates_semantics": "hazard_equivalent_telemetry_not_prediction_path",
            "no_classifier_bypass": False,
            "supports_exact_replay": False,
            "interpretability_claim": "none_control_only",
        }

    def forward(
        self,
        images: torch.Tensor,
        pixel_valid_mask: Optional[torch.Tensor] = None,
        *,
        force_decoder_fp64: bool = True,
        **_: object,
    ) -> AblationPosteriorOutput:
        if not force_decoder_fp64:
            raise ValueError("ablation posterior arithmetic must run in FP64")
        encoded: OriginEncoderOutput = self.encoder(images, pixel_valid_mask)
        # Match ORIGIN's numerical policy: the image trunk may use AMP, but
        # the sensitive projections, fusion, and final posterior head execute
        # in FP32 before posterior arithmetic is promoted to FP64.
        with torch.autocast(device_type=images.device.type, enabled=False):
            pooled = []
            for name in _ALL_SCALES:
                scale = encoded.scales[name]
                vector = self._masked_mean(
                    scale.features.float(), scale.valid_mask
                )
                pooled.append(self.scale_projections[name](vector))
            fused = self.fusion(torch.cat(pooled, dim=-1))
            return self._decode(fused)


class OriginVariantModel(OriginModel):
    """Ordinary ORIGIN model with the tested variant in checkpoint metadata."""

    supports_exact_replay = True

    def __init__(self, *, ablation_variant: str, **kwargs: object) -> None:
        self.ablation_variant = str(ablation_variant)
        super().__init__(**kwargs)

    def architecture_metadata(self) -> Dict[str, object]:
        metadata = super().architecture_metadata()
        metadata.update(
            {
                "name": "ORIGIN-v3-fold9-ablation",
                "ablation_variant": self.ablation_variant,
                "control_family": "conserved_local_ledger",
                "posterior_path": "conserved_local_rates_to_pure_birth_matrix_exponential",
                "posterior_dtype": "float64",
                "initialization": getattr(
                    self,
                    "ablation_initialization",
                    "native_origin_parameterization",
                ),
                "supports_exact_replay": True,
            }
        )
        return metadata


@dataclass
class SequentialHazardOutput:
    """Sequential decoder posterior backed by an unchanged ORIGIN ledger."""

    ledger_output: OriginOutput
    class_probs: torch.Tensor
    log_class_probs: torch.Tensor
    cumulative_probs: torch.Tensor
    expected_grade: torch.Tensor
    posterior_median: torch.Tensor
    class_map: torch.Tensor
    total_rates: torch.Tensor
    continuation_probs: torch.Tensor

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
    def local_rate_maps(self) -> Dict[str, torch.Tensor]:
        return self.ledger_output.local_rate_maps

    @property
    def valid_masks(self) -> Dict[str, torch.Tensor]:
        return self.ledger_output.valid_masks

    @property
    def metadata(self) -> Dict[str, object]:
        return self.ledger_output.metadata


def decode_sequential_hazard_ledger(output: OriginOutput) -> SequentialHazardOutput:
    """Decode the stored rates as adjacent conditional advance hazards."""

    rates = output.total_rates.to(torch.float64)
    _require_finite(rates, "sequential-hazard ledger rates")
    if bool((rates < 0).any()):
        raise ValueError("sequential-hazard ledger rates must be nonnegative")
    # expm1 retains gradients for near-null rates; each boundary is crossed
    # conditionally on reaching the preceding grade.
    continuation = -torch.expm1(-rates)
    cumulative = continuation.cumprod(dim=-1)
    class_probs = _classes_from_cumulative(cumulative)
    log_probs, cumulative, expected, median, class_map = _posterior_statistics(
        class_probs
    )
    return SequentialHazardOutput(
        ledger_output=output,
        class_probs=class_probs,
        log_class_probs=log_probs,
        cumulative_probs=cumulative,
        expected_grade=expected,
        posterior_median=median,
        class_map=class_map,
        total_rates=output.total_rates,
        continuation_probs=continuation,
    )


@dataclass
class SequentialHazardInterventionOutput:
    baseline: SequentialHazardOutput
    output: SequentialHazardOutput
    removal_masks: Dict[str, torch.Tensor]
    removed_rates: torch.Tensor

    @property
    def delta_expected_grade(self) -> torch.Tensor:
        return self.baseline.expected_grade - self.output.expected_grade


class LedgerSequentialHazardModel(nn.Module):
    """Matched decoder control with the exact ORIGIN encoder and rate ledger."""

    supports_exact_replay = True

    def __init__(self, origin: OriginVariantModel) -> None:
        super().__init__()
        self.origin = origin
        self.num_classes = origin.num_classes
        self.encoder_name = origin.encoder_name

    @property
    def encoder(self) -> nn.Module:
        return self.origin.encoder

    @property
    def evidence_scales(self) -> Tuple[str, ...]:
        return self.origin.evidence_scales

    def architecture_metadata(self) -> Dict[str, object]:
        base = self.origin.architecture_metadata()
        base.update(
            {
                "name": "ORIGIN-v3-ledger-sequential-hazard-control",
                "ablation_variant": "ledger_sequential_hazard",
                "control_family": "identical_conserved_local_ledger_decoder_control",
                "posterior_path": "conserved_local_rates_to_sequential_conditional_hazards",
                "decoder": "fp64_adjacent_exponential_hazard_product",
                "source_ledger_identical_to_origin_full": True,
                "supports_exact_replay": True,
                "no_classifier_bypass": True,
            }
        )
        return base

    def forward(
        self,
        images: torch.Tensor,
        pixel_valid_mask: Optional[torch.Tensor] = None,
        *,
        force_decoder_fp64: bool = True,
        return_encoder_maps: bool = False,
    ) -> SequentialHazardOutput:
        base = self.origin(
            images,
            pixel_valid_mask,
            force_decoder_fp64=force_decoder_fp64,
            return_encoder_maps=return_encoder_maps,
        )
        return decode_sequential_hazard_ledger(base)

    def replay_without(
        self,
        output: SequentialHazardOutput,
        removal_masks: Mapping[str, torch.Tensor],
        *,
        force_decoder_fp64: bool = True,
    ) -> SequentialHazardInterventionOutput:
        if not isinstance(output, SequentialHazardOutput):
            raise TypeError("sequential-hazard replay requires its own baseline output")
        replayed: OriginInterventionOutput = self.origin.replay_without(
            output.ledger_output,
            removal_masks,
            force_decoder_fp64=force_decoder_fp64,
        )
        return SequentialHazardInterventionOutput(
            baseline=decode_sequential_hazard_ledger(replayed.baseline),
            output=decode_sequential_hazard_ledger(replayed.output),
            removal_masks=replayed.removal_masks,
            removed_rates=replayed.removed_rates,
        )


def _origin_kwargs(cfg: object, *, pretrained: bool) -> Dict[str, object]:
    return {
        "num_classes": int(_cfg_get(cfg, "n_classes", 5)),
        "encoder_name": str(_cfg_get(cfg, "encoder", "convnext_tiny")),
        "pretrained": bool(pretrained),
        "evidence_scales": tuple(_cfg_get(cfg, "evidence_scales", _ALL_SCALES)),
        "projection_dim": int(_cfg_get(cfg, "projection_dim", 128)),
        "reference_count": float(_cfg_get(cfg, "reference_count", 4096.0)),
        "atom_mode": str(_cfg_get(cfg, "atom_mode", "cumulative")),
        "hybrid_cumulative_init": float(_cfg_get(cfg, "hybrid_cumulative_init", 0.9)),
        "atom_rate_init": float(_cfg_get(cfg, "atom_rate_init", 1e-6)),
        "prior_rate_init": float(_cfg_get(cfg, "prior_rate_init", 1e-4)),
        "boundary_scale_init": float(_cfg_get(cfg, "boundary_scale_init", 1.0)),
        "total_rate_cap": float(_cfg_get(cfg, "total_rate_cap", 64.0)),
        "prior_rate_cap": float(_cfg_get(cfg, "prior_rate_cap", 1.0)),
        "boundary_scale_cap": float(_cfg_get(cfg, "boundary_scale_cap", 2.0)),
        "rate_roundoff_margin": float(_cfg_get(cfg, "rate_roundoff_margin", 1.0)),
        "evidence_dropout": float(_cfg_get(cfg, "evidence_dropout", 0.0)),
        "mask_valid_fraction": float(_cfg_get(cfg, "mask_valid_fraction", 0.5)),
        "grad_checkpoint": bool(_cfg_get(cfg, "grad_checkpoint", False)),
    }


def _match_direct_atom_initialization_to_cumulative(
    origin: OriginVariantModel,
    *,
    atom_rate_init: float,
) -> None:
    """Match direct/independent ablations to full ORIGIN's nominal rates.

    Removing reverse-cumulative compilation would otherwise also change the
    initial boundary posterior: equal direct atoms yield ``[1,1,...]`` while
    the cumulative model yields ``[B,B-1,...,1]``. Boundary-specific biases
    remove that optimization confound without changing either forward rule.
    """

    generator = origin.generator
    if generator.atom_mode not in {"simplex_direct", "independent"}:
        raise ValueError("prior matching applies only to direct atom controls")
    multiplicity = torch.arange(
        generator.num_boundaries,
        0,
        -1,
        dtype=torch.float64,
    )
    target_atoms = multiplicity * float(atom_rate_init)
    fractions = target_atoms / float(generator.atom_mass_cap)
    if bool((fractions <= 0.0).any()) or bool((fractions >= 1.0).any()):
        raise ValueError("matched direct atom initialization lies outside its cap")
    if generator.atom_mode == "simplex_direct":
        null_fraction = 1.0 - fractions.sum()
        if float(null_fraction) <= 0.0:
            raise ValueError("matched direct simplex leaves no null mass")
        biases = torch.log(fractions / null_fraction)
    else:
        biases = torch.logit(fractions)
    with torch.no_grad():
        for head in generator.heads.values():
            head.atom_logits.bias.copy_(biases.to(head.atom_logits.bias))
    origin.ablation_initialization = (
        "matched_nominal_cumulative_origin_boundary_rates_v1"
    )


def _build_scale_ablation_with_matched_full_initialization(
    cfg: object,
    *,
    variant: str,
    pretrained: bool,
) -> OriginVariantModel:
    """Build a scale subset with its shared modules initialized as in full ORIGIN.

    Constructing only coarse heads changes the order in which random numbers
    are consumed: an s16 head would otherwise receive the weights assigned to
    s4 in the full model.  Build a temporary full reference from the same RNG
    state, restore that state for the target construction, then copy the
    corresponding named heads.  This leaves scale removal as the controlled
    change without altering ``origin_full`` itself.
    """

    initial_cpu_rng = torch.get_rng_state()
    full_kwargs = _origin_kwargs(cfg, pretrained=pretrained)
    full_kwargs["evidence_scales"] = _ALL_SCALES
    reference = OriginVariantModel(
        ablation_variant="origin_full",
        **full_kwargs,
    )
    torch.set_rng_state(initial_cpu_rng)
    target = OriginVariantModel(
        ablation_variant=variant,
        **_origin_kwargs(cfg, pretrained=pretrained),
    )
    with torch.no_grad():
        for scale in target.evidence_scales:
            target.generator.heads[scale].load_state_dict(
                reference.generator.heads[scale].state_dict(), strict=True
            )
    target.ablation_initialization = (
        "matched_corresponding_full_origin_scale_heads_v1"
    )
    return target


def build_origin_ablation_model(
    cfg: object,
    *,
    pretrained: bool | None = None,
) -> nn.Module:
    """Construct one named control entirely from checkpointed config fields."""

    variant = str(_cfg_get(cfg, "ablation_variant", "origin_full")).lower()
    use_pretrained = bool(_cfg_get(cfg, "pretrained", True)) if pretrained is None else bool(pretrained)
    if variant in _POOLED_VARIANTS:
        scales = tuple(_cfg_get(cfg, "evidence_scales", _ALL_SCALES))
        if scales != _ALL_SCALES:
            raise ValueError(
                f"{variant} must consume all scales {_ALL_SCALES}; got {scales}"
            )
        return MultiScaleMaskedPoolingModel(
            variant=variant,
            num_classes=int(_cfg_get(cfg, "n_classes", 5)),
            projection_dim=int(_cfg_get(cfg, "projection_dim", 128)),
            pretrained=use_pretrained,
            dropout=float(_cfg_get(cfg, "evidence_dropout", 0.0)),
            reference_count=float(_cfg_get(cfg, "reference_count", 4096.0)),
            atom_rate_init=float(_cfg_get(cfg, "atom_rate_init", 1e-6)),
            prior_rate_init=float(_cfg_get(cfg, "prior_rate_init", 1e-4)),
            boundary_scale_init=float(_cfg_get(cfg, "boundary_scale_init", 1.0)),
            mask_valid_fraction=float(_cfg_get(cfg, "mask_valid_fraction", 0.5)),
            grad_checkpoint=bool(_cfg_get(cfg, "grad_checkpoint", False)),
        )

    known_origin = {
        "origin_full",
        "ledger_sequential_hazard",
        "origin_simplex_direct",
        "origin_independent_sigmoid",
        "origin_fine_only",
        "origin_coarse_only",
        "origin_nll_only",
    }
    if variant not in known_origin:
        raise ValueError(f"unknown ORIGIN ablation variant: {variant!r}")
    if variant in {"origin_fine_only", "origin_coarse_only"}:
        origin = _build_scale_ablation_with_matched_full_initialization(
            cfg,
            variant=variant,
            pretrained=use_pretrained,
        )
    else:
        origin = OriginVariantModel(
            ablation_variant=variant,
            **_origin_kwargs(cfg, pretrained=use_pretrained),
        )
    if variant in {"origin_simplex_direct", "origin_independent_sigmoid"}:
        _match_direct_atom_initialization_to_cumulative(
            origin,
            atom_rate_init=float(_cfg_get(cfg, "atom_rate_init", 1e-6)),
        )
    if variant == "ledger_sequential_hazard":
        return LedgerSequentialHazardModel(origin)
    return origin


__all__ = [
    "AblationPosteriorOutput",
    "LedgerSequentialHazardModel",
    "MultiScaleMaskedPoolingModel",
    "OriginVariantModel",
    "SequentialHazardInterventionOutput",
    "SequentialHazardOutput",
    "build_origin_ablation_model",
    "decode_sequential_hazard_ledger",
    "nominal_cumulative_origin_posterior",
]
