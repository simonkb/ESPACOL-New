"""Frozen one-fold ablation configurations for ORIGIN-v3.

The ablation launcher resolves every variant into the ordinary ORIGIN fields
(``evidence_scales``, ``atom_mode`` and ``rps_weight``) before constructing
this dataclass.  Keeping those resolved values in the checkpoint makes each
control reconstructable without depending on a mutable name-to-override map.
"""

from __future__ import annotations

from dataclasses import dataclass

from .origin_config import OriginConfig


ABLATION_VARIANTS = (
    "origin_full",
    "pooled_softmax",
    "pooled_cumulative_logit",
    "pooled_conditional",
    "ledger_sequential_hazard",
    "origin_simplex_direct",
    "origin_independent_sigmoid",
    "origin_fine_only",
    "origin_coarse_only",
    "origin_nll_only",
)

_ALL_SCALES = ("s4", "s8", "s16", "s32")


@dataclass
class OriginAblationConfig(OriginConfig):
    """ORIGIN configuration with an explicit, reconstructable control name."""

    ablation_variant: str = "origin_full"
    ablation_description: str = ""

    def __post_init__(self) -> None:
        self.ablation_variant = str(self.ablation_variant).lower()
        self.ablation_description = str(self.ablation_description)
        super().__post_init__()
        if self.ablation_variant not in ABLATION_VARIANTS:
            raise ValueError(
                "unsupported ORIGIN ablation_variant: "
                f"{self.ablation_variant!r}; expected one of {ABLATION_VARIANTS}"
            )

        variant = self.ablation_variant
        if variant.startswith("pooled_") and self.evidence_scales != _ALL_SCALES:
            raise ValueError(
                f"{variant} must consume all native scales {_ALL_SCALES}; "
                f"got {self.evidence_scales}"
            )
        expected_modes = {
            "origin_full": "cumulative",
            "ledger_sequential_hazard": "cumulative",
            "origin_simplex_direct": "simplex_direct",
            "origin_independent_sigmoid": "independent",
            "origin_fine_only": "cumulative",
            "origin_coarse_only": "cumulative",
            "origin_nll_only": "cumulative",
        }
        expected_mode = expected_modes.get(variant)
        if expected_mode is not None and self.atom_mode != expected_mode:
            raise ValueError(
                f"{variant} requires atom_mode={expected_mode!r}; "
                f"got {self.atom_mode!r}"
            )
        expected_scales = {
            "origin_fine_only": ("s4", "s8"),
            "origin_coarse_only": ("s16", "s32"),
        }.get(variant)
        if expected_scales is not None and self.evidence_scales != expected_scales:
            raise ValueError(
                f"{variant} requires evidence_scales={expected_scales}; "
                f"got {self.evidence_scales}"
            )
        if variant == "origin_nll_only" and self.rps_weight != 0.0:
            raise ValueError("origin_nll_only requires rps_weight=0.0")


__all__ = ["ABLATION_VARIANTS", "OriginAblationConfig"]
