"""Prospective configuration for PATHS Ordinal Shell-Warranted Transport.

OSWT is an isolated successor to SAPT.  It keeps the hash-bound ORIGIN-v3
predictor immutable and learns only a small refiner whose prediction is the
transported ordinal posterior itself.  Keeping this configuration separate is
intentional: an OSWT run must never resume from, or be mistaken for, a SAPT
checkpoint.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Optional, Tuple

from .origin_config import OriginConfig


PATHS_OSWT_PROTOCOL_VERSION = "paths-v8-oswt-complementary-v2"
PATHS_OSWT_CHECKPOINT_SCHEMA = "paths-oswt-checkpoint-v2"
PATHS_OSWT_VARIANTS = (
    "shell_warranted",
    "shell_warranted_ungated",
    "identity_v3",
)


@dataclass
class PathsOSWTConfig(OriginConfig):
    """One registered OSWT fold with an immutable, hash-bound V3 base."""

    run_dir: str = "runs/paths_oswt_aptos_f0"
    oswt_variant: str = "shell_warranted"

    # Fixed PGF coordinates used to compile local shell evidence into a grade
    # warrant.  The probes are architectural coordinates rather than fitted
    # thresholds and therefore remain fixed across folds and datasets.
    pgf_probes: Tuple[float, ...] = (0.05, 0.20, 0.50, 0.80)
    oswt_beta_init: float = 0.1
    oswt_tau_init: float = 0.05
    oswt_beta_cap: float = 3.0
    oswt_tau_floor: float = 1e-3
    oswt_tau_cap: float = 0.5
    oswt_strength: float = 1.0

    # The registered first experiment freezes every V3 tensor for every
    # epoch.  Only ``paths_refiner.*`` is optimized.
    paths_refiner_lr: float = 5e-4
    weight_decay: float = 0.0
    risk_set_alpha: float = 0.5

    # The path provides invocation provenance; the checksum provides content
    # identity.  Both are mandatory for every fresh run.
    warm_start_checkpoint: Optional[str] = None
    warm_start_sha256: Optional[str] = None
    warm_start_metric_tolerance: float = 1e-6

    certificate_samples: int = 8
    certificate_shortlist_size: int = 8
    certificate_replay_tolerance: float = 2e-5

    def __post_init__(self) -> None:
        super().__post_init__()
        self.oswt_variant = str(self.oswt_variant).lower()
        if self.oswt_variant not in PATHS_OSWT_VARIANTS:
            raise ValueError(
                "oswt_variant must be one of " + ", ".join(PATHS_OSWT_VARIANTS)
            )
        if self.atom_mode != "cumulative":
            raise ValueError("OSWT requires cumulative ORIGIN-v3 atoms")
        if self.encoder != "convnext_tiny" or self.evidence_scales != (
            "s4",
            "s8",
            "s16",
            "s32",
        ):
            raise ValueError("OSWT fixes the audited four-scale ConvNeXt-tiny V3 base")
        if self.relation_enabled:
            raise ValueError("OSWT cannot be combined with an ORIGIN relation field")
        if self.class_weighting != "none" or self.stratified_batches:
            raise ValueError(
                "OSWT uses proper risk-set scoring, not outcome weighting or oversampling"
            )
        if self.evidence_budget_weight != 0.0:
            raise ValueError("the registered OSWT objective excludes evidence penalties")

        self.pgf_probes = tuple(float(value) for value in self.pgf_probes)
        if not self.pgf_probes or any(
            not math.isfinite(value) or not 0.0 < value < 1.0
            for value in self.pgf_probes
        ):
            raise ValueError("every PGF probe must lie strictly in (0, 1)")
        if tuple(sorted(set(self.pgf_probes))) != self.pgf_probes:
            raise ValueError("pgf_probes must be unique and strictly increasing")
        if not math.isfinite(self.oswt_beta_init) or self.oswt_beta_init <= 0.0:
            raise ValueError("oswt_beta_init must be finite and positive")
        if not math.isfinite(self.oswt_tau_init) or self.oswt_tau_init <= 0.0:
            raise ValueError("oswt_tau_init must be finite and positive")
        if not math.isfinite(self.oswt_beta_cap) or not (
            0.0 < self.oswt_beta_init < self.oswt_beta_cap
        ):
            raise ValueError("oswt_beta_init must lie strictly inside (0, oswt_beta_cap)")
        if not all(
            math.isfinite(value)
            for value in (self.oswt_tau_floor, self.oswt_tau_cap)
        ) or not (
            0.0 < self.oswt_tau_floor < self.oswt_tau_init < self.oswt_tau_cap
        ):
            raise ValueError(
                "OSWT tau parameters must satisfy 0 < tau_floor < tau_init < tau_cap"
            )
        if (
            not math.isfinite(self.oswt_strength)
            or not 0.0 <= self.oswt_strength <= 1.0
        ):
            raise ValueError("oswt_strength must lie in [0, 1]")
        if self.oswt_variant == "identity_v3" and self.oswt_strength != 0.0:
            raise ValueError("identity_v3 requires oswt_strength=0")
        if self.oswt_variant != "identity_v3" and self.oswt_strength <= 0.0:
            raise ValueError("a learned OSWT variant requires positive oswt_strength")
        if not math.isfinite(self.paths_refiner_lr) or self.paths_refiner_lr <= 0.0:
            raise ValueError("paths_refiner_lr must be finite and positive")
        if self.weight_decay != 0.0:
            raise ValueError(
                "OSWT refiner scalars/simplex logits require weight_decay=0; "
                "raw-parameter decay changes the effective beta/tau prior"
            )
        if (
            not math.isfinite(self.risk_set_alpha)
            or not 0.0 <= self.risk_set_alpha <= 1.0
        ):
            raise ValueError("risk_set_alpha must lie in [0, 1]")
        if (self.warm_start_checkpoint is None) != (self.warm_start_sha256 is None):
            raise ValueError(
                "warm_start_checkpoint and warm_start_sha256 must be supplied together"
            )
        if self.warm_start_sha256 is not None:
            checksum = self.warm_start_sha256.lower()
            if len(checksum) != 64 or any(
                character not in "0123456789abcdef" for character in checksum
            ):
                raise ValueError("warm_start_sha256 must be 64 hexadecimal characters")
            self.warm_start_sha256 = checksum
        if (
            not math.isfinite(self.warm_start_metric_tolerance)
            or self.warm_start_metric_tolerance < 0.0
        ):
            raise ValueError("warm_start_metric_tolerance must be non-negative")
        if self.certificate_samples < 1:
            raise ValueError("certificate_samples must be positive")
        if self.certificate_shortlist_size < 1:
            raise ValueError("certificate_shortlist_size must be positive")
        if (
            not math.isfinite(self.certificate_replay_tolerance)
            or self.certificate_replay_tolerance < 0.0
        ):
            raise ValueError("certificate_replay_tolerance must be non-negative")


__all__ = [
    "PATHS_OSWT_CHECKPOINT_SCHEMA",
    "PATHS_OSWT_PROTOCOL_VERSION",
    "PATHS_OSWT_VARIANTS",
    "PathsOSWTConfig",
]
