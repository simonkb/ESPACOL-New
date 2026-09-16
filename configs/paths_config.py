"""Prospective configuration for PATHS severity grading.

PATHS is intentionally configured separately from ORIGIN.  The inherited
fields describe the immutable ORIGIN-v3 base; the fields declared here define
the PGF continuation correction and its validation-only development protocol.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Optional, Tuple

from .origin_config import OriginConfig


PATHS_PROTOCOL_VERSION = "paths-v2"


@dataclass
class PathsConfig(OriginConfig):
    """One PATHS fold with a hash-bound ORIGIN-v3 warm start."""

    run_dir: str = "runs/paths_aptos_f0_v2"

    # Fixed probability-generating-function probes.  They are architectural
    # coordinates, not fitted thresholds, and therefore remain identical
    # across datasets and folds.
    pgf_probes: Tuple[float, ...] = (0.05, 0.20, 0.50, 0.80)
    correction_cap: float = 3.0
    correction_gain_init: float = 0.05
    correction_strength: float = 1.0
    correction_only_epochs: int = 3
    encoder_freeze_epochs: int = 3

    # A warm-started model should learn the new focality law rapidly without
    # erasing the audited V3 representation.  These are deliberately separate
    # optimizer groups: the encoder moves least, the V3 generator may adapt a
    # little, and the new PATHS refiner receives the full learning rate.
    paths_encoder_lr: float = 1e-5
    paths_base_lr: float = 5e-5
    paths_refiner_lr: float = 5e-4

    # Boundary weights depend only on complete training-fold risk-set sizes.
    # They are fixed before optimization and remain positive, preserving the
    # conditional Bernoulli target at every ordinal boundary.
    risk_set_alpha: float = 0.5

    # An immutable, externally supplied SHA-256 is mandatory for every fresh
    # run.  A path alone is not an experiment identity.
    warm_start_checkpoint: Optional[str] = None
    warm_start_sha256: Optional[str] = None
    warm_start_metric_tolerance: float = 1e-6

    certificate_samples: int = 8
    certificate_replay_tolerance: float = 2e-5

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.atom_mode != "cumulative":
            raise ValueError("PATHS primary architecture requires cumulative V3 atoms")
        if self.evidence_scales != ("s4", "s8", "s16", "s32"):
            raise ValueError("PATHS-v2 fixes the audited four-scale V3 base")
        if self.class_weighting != "none" or self.stratified_batches:
            raise ValueError(
                "PATHS uses boundary-risk weighting, not outcome weighting or oversampling"
            )
        if self.evidence_budget_weight != 0.0:
            raise ValueError("PATHS primary proper objective excludes evidence penalties")
        self.pgf_probes = tuple(float(value) for value in self.pgf_probes)
        if not self.pgf_probes:
            raise ValueError("pgf_probes must contain at least one fixed probe")
        if any(
            not math.isfinite(value) or not 0.0 < value < 1.0
            for value in self.pgf_probes
        ):
            raise ValueError("every PGF probe must lie strictly in (0, 1)")
        if tuple(sorted(set(self.pgf_probes))) != self.pgf_probes:
            raise ValueError("pgf_probes must be unique and strictly increasing")
        if not math.isfinite(self.correction_cap) or self.correction_cap <= 0.0:
            raise ValueError("correction_cap must be finite and positive")
        if (
            not math.isfinite(self.correction_gain_init)
            or not 0.0 < self.correction_gain_init < self.correction_cap
        ):
            raise ValueError(
                "correction_gain_init must lie in (0, correction_cap)"
            )
        if (
            not math.isfinite(self.correction_strength)
            or not 0.0 < self.correction_strength <= 1.0
        ):
            raise ValueError("correction_strength must lie in (0, 1]")
        if self.correction_only_epochs < 0:
            raise ValueError("correction_only_epochs must be non-negative")
        if self.encoder_freeze_epochs < self.correction_only_epochs:
            raise ValueError(
                "encoder_freeze_epochs must cover the correction-only phase"
            )
        path_lrs = (
            self.paths_encoder_lr,
            self.paths_base_lr,
            self.paths_refiner_lr,
        )
        if not all(math.isfinite(value) and value > 0.0 for value in path_lrs):
            raise ValueError("all PATHS optimizer-group learning rates must be positive")
        if not self.paths_encoder_lr <= self.paths_base_lr <= self.paths_refiner_lr:
            raise ValueError(
                "PATHS learning rates must satisfy encoder <= base <= refiner"
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
        if (
            not math.isfinite(self.certificate_replay_tolerance)
            or self.certificate_replay_tolerance < 0.0
        ):
            raise ValueError("certificate_replay_tolerance must be non-negative")


__all__ = ["PATHS_PROTOCOL_VERSION", "PathsConfig"]
