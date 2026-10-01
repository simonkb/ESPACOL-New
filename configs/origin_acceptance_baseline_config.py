"""Frozen configuration identity for the acceptance-revision baselines."""

from __future__ import annotations

from dataclasses import dataclass
import math

from .origin_config import OriginConfig
from models.origin_acceptance_baselines import ACCEPTANCE_BASELINE_VARIANTS


_ALL_SCALES = ("s4", "s8", "s16", "s32")


@dataclass
class OriginAcceptanceBaselineConfig(OriginConfig):
    """One matched baseline run with split and optimization seed separated.

    ``seed`` remains the outer/inner split seed. ``training_seed`` changes only
    initialization, loader order, augmentation, dropout, and stochastic depth,
    so replication seeds cannot silently change the patients in a fold.
    """

    baseline_variant: str = "ledger_sequential_hazard"
    # Kept explicitly because the two reused controls reconstruct through the
    # existing ablation builder and checkpoints must be self-contained.
    ablation_variant: str = "ledger_sequential_hazard"
    baseline_description: str = ""
    training_seed: int = 42
    sparse_l1_weight: float = 0.0
    sparse_l1_delay_epochs: int = 0
    sparse_activation_init: float = 1e-6
    certificate_samples: int = 4
    certificate_replay_tolerance: float = 2e-6
    protocol_id: str = "origin-acceptance-baselines-v1"

    def __post_init__(self) -> None:
        self.baseline_variant = str(self.baseline_variant).lower()
        self.ablation_variant = str(self.ablation_variant).lower()
        if self.ablation_variant != self.baseline_variant:
            raise ValueError(
                "baseline_variant and reconstructable ablation_variant must match"
            )
        self.baseline_description = str(self.baseline_description)
        self.protocol_id = str(self.protocol_id)
        super().__post_init__()
        if self.baseline_variant not in ACCEPTANCE_BASELINE_VARIANTS:
            raise ValueError(
                f"unsupported baseline_variant {self.baseline_variant!r}; "
                f"expected one of {ACCEPTANCE_BASELINE_VARIANTS}"
            )
        if self.evidence_scales != _ALL_SCALES:
            raise ValueError(
                "matched acceptance baselines must use all native encoder scales "
                f"{_ALL_SCALES}"
            )
        if self.atom_mode != "cumulative":
            raise ValueError("matched baselines fix atom_mode='cumulative'")
        if self.evidence_budget_weight != 0.0:
            raise ValueError(
                "acceptance baselines disable ORIGIN's rate-budget regularizer; "
                "Sparse-BagNet uses sparse_l1_weight instead"
            )
        if not isinstance(self.training_seed, int) or self.training_seed < 0:
            raise ValueError("training_seed must be a non-negative integer")
        if not math.isfinite(self.sparse_l1_weight) or self.sparse_l1_weight < 0.0:
            raise ValueError("sparse_l1_weight must be finite and non-negative")
        if self.sparse_l1_delay_epochs < 0:
            raise ValueError("sparse_l1_delay_epochs must be non-negative")
        if (
            not math.isfinite(self.sparse_activation_init)
            or self.sparse_activation_init <= 0.0
        ):
            raise ValueError("sparse_activation_init must be finite and positive")
        if self.baseline_variant != "sparse_bagnet" and self.sparse_l1_weight != 0.0:
            raise ValueError(
                "sparse_l1_weight must be zero outside the sparse_bagnet baseline"
            )
        if self.certificate_samples < 1:
            raise ValueError("certificate_samples must be positive")
        if (
            not math.isfinite(self.certificate_replay_tolerance)
            or self.certificate_replay_tolerance < 0.0
        ):
            raise ValueError(
                "certificate_replay_tolerance must be finite and non-negative"
            )
        if self.protocol_id != "origin-acceptance-baselines-v1":
            raise ValueError("protocol_id is frozen for this experiment suite")


__all__ = ["OriginAcceptanceBaselineConfig"]
