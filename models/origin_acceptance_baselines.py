"""Pre-registered matched baselines for the ORIGIN acceptance revision.

This module adds two genuinely local, generator-exclusive comparators while
reusing the already audited ORIGIN controls for the other two comparisons.
The local comparators deliberately expose *effective* per-cell ledger entries:
the original-support valid-count denominator and the equal scale weight are
baked into every stored entry.  Consequently an intervention is an exact
partition of a frozen ledger; it never re-encodes an image or renormalizes the
surviving cells.

``ordinal_additive_mil``
    Boundary-specific signed local contributions are added to an intercept,
    decoded as sequential continuation probabilities, and hence form a valid
    ordinal posterior.

``sparse_bagnet``
    Non-negative class-specific local activations are masked-mean pooled and
    decoded by a categorical softmax.  Sparsity is imposed by the companion
    loss on the unpooled valid activations, not on their image-level mean.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .origin import ChannelLayerNorm2d
from .origin_ablation import (
    _classes_from_cumulative,
    _equivalent_boundary_rates,
    _posterior_statistics,
    build_origin_ablation_model,
    nominal_cumulative_origin_posterior,
)
from .origin_encoder import ConvNeXtTinyPyramidEncoder, OriginEncoderOutput


ACCEPTANCE_BASELINE_VARIANTS = (
    "ledger_sequential_hazard",
    "pooled_conditional",
    "ordinal_additive_mil",
    "sparse_bagnet",
)

_ALL_SCALES = ("s4", "s8", "s16", "s32")
_LOCAL_VARIANTS = {"ordinal_additive_mil", "sparse_bagnet"}


def _cfg_get(cfg: object, name: str, default: Any = None) -> Any:
    if isinstance(cfg, Mapping):
        return cfg.get(name, default)
    return getattr(cfg, name, default)


def _inverse_softplus_scalar(value: float) -> float:
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError("softplus target must be finite and positive")
    return value + math.log(-math.expm1(-value))


class _PointwiseLocalMapHead(nn.Module):
    """A channel-only MLP; no operation mixes spatial sites."""

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        *,
        dropout: float,
    ) -> None:
        super().__init__()
        if min(in_channels, hidden_channels, out_channels) < 1:
            raise ValueError("local head dimensions must be positive")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must lie in [0,1)")
        self.projection = nn.Conv2d(in_channels, hidden_channels, 1)
        self.norm = ChannelLayerNorm2d(hidden_channels)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout2d(dropout)
        self.local_logits = nn.Conv2d(hidden_channels, out_channels, 1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 4:
            raise ValueError("features must have shape (N,C,H,W)")
        value = self.projection(features)
        value = self.dropout(self.activation(self.norm(value)))
        value = self.local_logits(value)
        if not bool(torch.isfinite(value).all()):
            raise FloatingPointError("local evidence head emitted non-finite values")
        return value


@dataclass
class AcceptanceLocalOutput:
    """Posterior and the complete fixed-denominator local evidence trace."""

    baseline_variant: str
    class_probs: torch.Tensor
    log_class_probs: torch.Tensor
    cumulative_probs: torch.Tensor
    expected_grade: torch.Tensor
    posterior_median: torch.Tensor
    class_map: torch.Tensor
    total_rates: torch.Tensor
    aggregated_logits: torch.Tensor
    effective_local_maps: Dict[str, torch.Tensor]
    raw_local_maps: Dict[str, torch.Tensor]
    original_valid_masks: Dict[str, torch.Tensor]
    scale_metadata: Dict[str, object]
    decoder_aux: Dict[str, torch.Tensor]
    sparse_activation_l1: torch.Tensor
    local_map_semantics: str

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
        """Compatibility alias; metadata gives the non-rate semantics."""

        return self.effective_local_maps

    @property
    def local_contribution_maps(self) -> Dict[str, torch.Tensor]:
        return self.effective_local_maps

    @property
    def local_evidence_maps(self) -> Dict[str, torch.Tensor]:
        return self.effective_local_maps

    @property
    def valid_masks(self) -> Dict[str, torch.Tensor]:
        return self.original_valid_masks

    @property
    def metadata(self) -> Dict[str, object]:
        return self.scale_metadata


@dataclass
class AcceptanceLocalInterventionOutput:
    baseline: AcceptanceLocalOutput
    output: AcceptanceLocalOutput
    removal_masks: Dict[str, torch.Tensor]
    removed_contributions: torch.Tensor

    @property
    def removed_rates(self) -> torch.Tensor:
        """Compatibility alias; values are logits, never CTMC rates."""

        return self.removed_contributions

    @property
    def delta_expected_grade(self) -> torch.Tensor:
        return self.baseline.expected_grade - self.output.expected_grade


class FixedCountLocalEvidenceModel(nn.Module):
    """Matched Additive-MIL or sparse BagNet local-evidence comparator."""

    supports_exact_replay = True

    def __init__(
        self,
        *,
        variant: str,
        num_classes: int = 5,
        projection_dim: int = 128,
        pretrained: bool = True,
        dropout: float = 0.0,
        evidence_scales: Sequence[str] = _ALL_SCALES,
        reference_count: float = 4096.0,
        atom_rate_init: float = 1e-6,
        prior_rate_init: float = 1e-4,
        boundary_scale_init: float = 1.0,
        sparse_activation_init: float = 1e-6,
        mask_valid_fraction: float = 0.5,
        grad_checkpoint: bool = False,
        encoder: Optional[nn.Module] = None,
    ) -> None:
        super().__init__()
        variant = str(variant).lower()
        if variant not in _LOCAL_VARIANTS:
            raise ValueError(f"unknown local acceptance baseline {variant!r}")
        if num_classes < 2:
            raise ValueError("num_classes must be at least two")
        scales = tuple(str(name) for name in evidence_scales)
        if scales != _ALL_SCALES:
            raise ValueError(
                f"acceptance baselines must consume all scales {_ALL_SCALES}; got {scales}"
            )
        if encoder is None:
            encoder = ConvNeXtTinyPyramidEncoder(
                pretrained=pretrained,
                grad_checkpoint=grad_checkpoint,
                mask_valid_fraction=mask_valid_fraction,
            )
        if not hasattr(encoder, "STAGE_CHANNELS"):
            raise TypeError("encoder must expose STAGE_CHANNELS")
        missing = set(scales) - set(encoder.STAGE_CHANNELS)
        if missing:
            raise ValueError(f"encoder is missing evidence scales {sorted(missing)}")

        self.encoder = encoder
        self.variant = variant
        self.num_classes = int(num_classes)
        self.num_boundaries = self.num_classes - 1
        self.evidence_scales = scales
        output_channels = (
            self.num_boundaries
            if variant == "ordinal_additive_mil"
            else self.num_classes
        )
        self.heads = nn.ModuleDict(
            {
                name: _PointwiseLocalMapHead(
                    int(encoder.STAGE_CHANNELS[name]),
                    int(projection_dim),
                    output_channels,
                    dropout=dropout,
                )
                for name in scales
            }
        )
        self.intercept = nn.Parameter(torch.zeros(output_channels))
        nominal_rates, nominal_probs = nominal_cumulative_origin_posterior(
            num_classes=self.num_classes,
            reference_count=reference_count,
            atom_rate_init=atom_rate_init,
            prior_rate_init=prior_rate_init,
            boundary_scale_init=boundary_scale_init,
        )
        self.register_buffer("nominal_origin_total_rates", nominal_rates)
        self.register_buffer("nominal_origin_class_probs", nominal_probs)
        self.sparse_activation_init = float(sparse_activation_init)
        self._initialize_matched_floor()

    def _initialize_matched_floor(self) -> None:
        probs = self.nominal_origin_class_probs.to(torch.float64)
        cumulative = probs[1:].flip((0,)).cumsum(dim=0).flip((0,))
        with torch.no_grad():
            if self.variant == "ordinal_additive_mil":
                continuation = torch.cat(
                    (cumulative[:1], cumulative[1:] / cumulative[:-1])
                )
                self.intercept.copy_(
                    torch.logit(continuation).to(self.intercept)
                )
                for head in self.heads.values():
                    head.local_logits.weight.zero_()
                    head.local_logits.bias.zero_()
            else:
                activation = self.sparse_activation_init
                raw_bias = _inverse_softplus_scalar(activation)
                self.intercept.copy_(
                    (probs.log() - activation).to(self.intercept)
                )
                for head in self.heads.values():
                    head.local_logits.weight.zero_()
                    nn.init.constant_(head.local_logits.bias, raw_bias)

    @staticmethod
    def _validated_mask(
        raw_map: torch.Tensor, valid_mask: torch.Tensor, scale: str
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if valid_mask.dtype != torch.bool:
            valid_mask = valid_mask.bool()
        expected = raw_map.shape[:1] + raw_map.shape[-2:]
        if tuple(valid_mask.shape) != tuple(expected):
            raise ValueError(
                f"valid mask shape mismatch at {scale}: {tuple(valid_mask.shape)} != {expected}"
            )
        counts = valid_mask.sum(dim=(-2, -1))
        if bool((counts == 0).any()):
            raise ValueError(f"every sample needs an original valid cell at scale {scale}")
        return valid_mask, counts

    def _decode(
        self,
        *,
        effective_maps: Dict[str, torch.Tensor],
        raw_maps: Dict[str, torch.Tensor],
        masks: Dict[str, torch.Tensor],
        metadata: Dict[str, object],
        sparse_l1: torch.Tensor,
    ) -> AcceptanceLocalOutput:
        logits = self.intercept.to(torch.float64)[None].expand(
            next(iter(effective_maps.values())).shape[0], -1
        )
        for local_map in effective_maps.values():
            logits = logits + local_map.to(torch.float64).sum(dim=(1, 2))
        if not bool(torch.isfinite(logits).all()):
            raise FloatingPointError("aggregated local-evidence logits are non-finite")

        aux: Dict[str, torch.Tensor] = {}
        if self.variant == "ordinal_additive_mil":
            continuation = torch.sigmoid(logits)
            cumulative = continuation.cumprod(dim=-1)
            class_probs = _classes_from_cumulative(cumulative)
            aux["continuation_probabilities"] = continuation
        else:
            class_probs = logits.softmax(dim=-1)
        log_probs, cumulative, expected, median, class_map = _posterior_statistics(
            class_probs
        )
        return AcceptanceLocalOutput(
            baseline_variant=self.variant,
            class_probs=class_probs,
            log_class_probs=log_probs,
            cumulative_probs=cumulative,
            expected_grade=expected,
            posterior_median=median,
            class_map=class_map,
            total_rates=_equivalent_boundary_rates(cumulative),
            aggregated_logits=logits,
            effective_local_maps=effective_maps,
            raw_local_maps=raw_maps,
            original_valid_masks=masks,
            scale_metadata=metadata,
            decoder_aux=aux,
            sparse_activation_l1=sparse_l1,
            local_map_semantics=(
                "signed_boundary_logit_contribution_fixed_original_valid_count"
                if self.variant == "ordinal_additive_mil"
                else "nonnegative_class_logit_evidence_fixed_original_valid_count"
            ),
        )

    def architecture_metadata(self) -> Dict[str, object]:
        additive = self.variant == "ordinal_additive_mil"
        return {
            "name": "ORIGIN-acceptance-revision-matched-baseline",
            "ablation_variant": self.variant,
            "baseline_variant": self.variant,
            "control_family": (
                "fixed_count_ordinal_additive_mil"
                if additive
                else "fixed_count_sparse_bagnet"
            ),
            "encoder": "convnext_tiny",
            "num_classes": self.num_classes,
            "consumed_scales": list(self.evidence_scales),
            "posterior_path": (
                "signed_local_boundary_maps_to_fixed_mean_to_sequential_continuation"
                if additive
                else "nonnegative_local_class_maps_to_fixed_mean_to_softmax"
            ),
            "pooling": "equal_scale_sum_of_original_valid_count_means",
            "intercept": "decoder_specific_trainable_intercept",
            "posterior_dtype": "float64",
            "sensitive_head_compute_dtype": "float32_autocast_disabled",
            "local_map_semantics": (
                "signed_boundary_logit_contribution"
                if additive
                else "nonnegative_multiclass_activation"
            ),
            "invalid_cells_contribute": False,
            "deletion_denominator": "frozen_original_valid_count",
            "no_classifier_bypass": True,
            "supports_exact_replay": True,
            "rate_telemetry_applicable": False,
            "total_rates_semantics": "posterior_equivalent_hazard_telemetry_only",
            "initialization": "matched_nominal_cumulative_origin_posterior_v1",
            "nominal_origin_total_rates": self.nominal_origin_total_rates.detach().cpu().tolist(),
            "nominal_origin_class_probs": self.nominal_origin_class_probs.detach().cpu().tolist(),
            "sparsity_regularizer": (
                "none"
                if additive
                else "mean_absolute_unpooled_valid_local_activation"
            ),
            "interpretation_scope": "exact_stored_ledger_not_causal_pixel_effect",
        }

    def forward(
        self,
        images: torch.Tensor,
        pixel_valid_mask: Optional[torch.Tensor] = None,
        *,
        force_decoder_fp64: bool = True,
        **_: object,
    ) -> AcceptanceLocalOutput:
        if not force_decoder_fp64:
            raise ValueError("acceptance-baseline posterior arithmetic must be FP64")
        encoded: OriginEncoderOutput = self.encoder(images, pixel_valid_mask)
        scale_weight = 1.0 / len(self.evidence_scales)
        effective_maps: Dict[str, torch.Tensor] = {}
        raw_maps: Dict[str, torch.Tensor] = {}
        masks: Dict[str, torch.Tensor] = {}
        metadata: Dict[str, object] = {}
        l1_terms: list[torch.Tensor] = []
        with torch.autocast(device_type=images.device.type, enabled=False):
            for name in self.evidence_scales:
                scale = encoded.scales[name]
                raw = self.heads[name](scale.features.float())
                if self.variant == "sparse_bagnet":
                    raw = F.softplus(raw)
                mask, count = self._validated_mask(raw, scale.valid_mask, name)
                mask_cf = mask[:, None].to(raw.dtype)
                valid_raw = raw * mask_cf
                denominator = count.to(raw.dtype)[:, None, None, None]
                # Spatial-last effective entries sum exactly to the scale's
                # original-support mean times the pre-registered equal weight.
                effective = (
                    valid_raw * (scale_weight / denominator)
                ).permute(0, 2, 3, 1).contiguous()
                raw_maps[name] = raw.permute(0, 2, 3, 1).contiguous()
                effective_maps[name] = effective
                masks[name] = mask
                metadata[name] = scale.metadata
                if self.variant == "sparse_bagnet":
                    per_sample = valid_raw.sum(dim=(1, 2, 3)) / (
                        count.to(raw.dtype) * self.num_classes
                    )
                    l1_terms.append(per_sample.mean())
        sparse_l1 = (
            torch.stack(l1_terms).mean()
            if l1_terms
            else next(iter(effective_maps.values())).new_zeros(())
        )
        return self._decode(
            effective_maps=effective_maps,
            raw_maps=raw_maps,
            masks=masks,
            metadata=metadata,
            sparse_l1=sparse_l1,
        )

    def replay_without(
        self,
        output: AcceptanceLocalOutput,
        removal_masks: Mapping[str, torch.Tensor],
        *,
        force_decoder_fp64: bool = True,
    ) -> AcceptanceLocalInterventionOutput:
        if not force_decoder_fp64:
            raise ValueError("acceptance-baseline replay arithmetic must be FP64")
        if not isinstance(output, AcceptanceLocalOutput):
            raise TypeError("replay requires an AcceptanceLocalOutput")
        if output.baseline_variant != self.variant:
            raise ValueError("cannot replay output from a different baseline variant")
        if set(removal_masks) != set(self.evidence_scales):
            raise ValueError("removal masks must name every evidence scale exactly once")
        replay_maps: Dict[str, torch.Tensor] = {}
        normalized_masks: Dict[str, torch.Tensor] = {}
        removed: torch.Tensor | None = None
        for name in self.evidence_scales:
            local_map = output.effective_local_maps[name]
            removal = removal_masks[name].to(device=local_map.device, dtype=torch.bool)
            valid = output.original_valid_masks[name].to(local_map.device).bool()
            if tuple(removal.shape) != tuple(local_map.shape[:-1]):
                raise ValueError(f"removal mask shape mismatch at {name}")
            removal = removal & valid
            expanded = removal.unsqueeze(-1)
            removed_map = torch.where(expanded, local_map, torch.zeros_like(local_map))
            replay_maps[name] = torch.where(
                expanded, torch.zeros_like(local_map), local_map
            )
            normalized_masks[name] = removal
            scale_removed = removed_map.to(torch.float64).sum(dim=(1, 2))
            removed = scale_removed if removed is None else removed + scale_removed
        if removed is None:
            raise RuntimeError("local acceptance ledger is empty")
        replayed = self._decode(
            effective_maps=replay_maps,
            raw_maps=output.raw_local_maps,
            masks=output.original_valid_masks,
            metadata=output.scale_metadata,
            sparse_l1=output.sparse_activation_l1,
        )
        return AcceptanceLocalInterventionOutput(
            baseline=output,
            output=replayed,
            removal_masks=normalized_masks,
            removed_contributions=removed,
        )


def build_origin_acceptance_baseline(
    cfg: object,
    *,
    pretrained: bool | None = None,
    encoder: Optional[nn.Module] = None,
) -> nn.Module:
    """Build one of the four pre-registered matched comparators."""

    variant = str(
        _cfg_get(cfg, "baseline_variant", _cfg_get(cfg, "ablation_variant", ""))
    ).lower()
    if variant not in ACCEPTANCE_BASELINE_VARIANTS:
        raise ValueError(
            f"unknown acceptance baseline {variant!r}; expected {ACCEPTANCE_BASELINE_VARIANTS}"
        )
    use_pretrained = (
        bool(_cfg_get(cfg, "pretrained", True))
        if pretrained is None
        else bool(pretrained)
    )
    if variant in {"ledger_sequential_hazard", "pooled_conditional"}:
        if encoder is not None:
            raise ValueError("encoder injection is supported only by the new local heads")
        return build_origin_ablation_model(cfg, pretrained=use_pretrained)
    return FixedCountLocalEvidenceModel(
        variant=variant,
        num_classes=int(_cfg_get(cfg, "n_classes", 5)),
        projection_dim=int(_cfg_get(cfg, "projection_dim", 128)),
        pretrained=use_pretrained,
        dropout=float(_cfg_get(cfg, "evidence_dropout", 0.0)),
        evidence_scales=tuple(_cfg_get(cfg, "evidence_scales", _ALL_SCALES)),
        reference_count=float(_cfg_get(cfg, "reference_count", 4096.0)),
        atom_rate_init=float(_cfg_get(cfg, "atom_rate_init", 1e-6)),
        prior_rate_init=float(_cfg_get(cfg, "prior_rate_init", 1e-4)),
        boundary_scale_init=float(_cfg_get(cfg, "boundary_scale_init", 1.0)),
        sparse_activation_init=float(
            _cfg_get(cfg, "sparse_activation_init", 1e-6)
        ),
        mask_valid_fraction=float(_cfg_get(cfg, "mask_valid_fraction", 0.5)),
        grad_checkpoint=bool(_cfg_get(cfg, "grad_checkpoint", False)),
        encoder=encoder,
    )


__all__ = [
    "ACCEPTANCE_BASELINE_VARIANTS",
    "AcceptanceLocalInterventionOutput",
    "AcceptanceLocalOutput",
    "FixedCountLocalEvidenceModel",
    "build_origin_acceptance_baseline",
]
