"""Configuration for the ORIGIN severity generator.

ORIGIN is deliberately isolated from both OPTIC-C and MOSAIC.  Keeping its
configuration in a separate module prevents a new experiment from silently
inheriting legacy feature flags or checkpoint semantics.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Optional, Tuple

from Datasets.mosaic_data import MOSAIC_PREPROCESSING_VERSION


# ORIGIN calls the audited implementation directly, so its data-arithmetic
# identity is exactly the MOSAIC canonical full-canvas version.
ORIGIN_PREPROCESSING_VERSION = MOSAIC_PREPROCESSING_VERSION


@dataclass
class OriginConfig:
    """One fully specified ORIGIN cross-validation experiment.

    The default objective is an unweighted proper categorical likelihood.
    Class weighting is an explicit opt-in because weighted likelihoods are not
    calibrated estimates of the deployment population posterior.
    """

    # Data identity and split policy.
    dataset: str = "aptos"
    n_classes: int = 5
    n_folds: int = 5
    val_fraction: float = 0.1
    run_dir: str = "runs/origin_aptos"
    preprocessing_version: str = ORIGIN_PREPROCESSING_VERSION
    seed: int = 42

    # Input and local encoder.  Multi-scale values are output strides, not
    # image pyramid resize factors.
    img_size: int = 640
    encoder: str = "convnext_tiny"
    pretrained: bool = True
    evidence_scales: Tuple[str, ...] = ("s4", "s8", "s16", "s32")
    projection_dim: int = 128
    grad_checkpoint: bool = False
    mask_valid_fraction: float = 0.5

    # Conserved generator parameterisation.
    # A 640-pixel s4 retinal lattice has roughly 20k valid sites.  Scaling the
    # conserved sum to 4096 reference sites keeps a focal cell's gradient
    # measurable; the matching 1e-6 atom initialization keeps the initial
    # image-wide rate small.
    reference_count: float = 4096.0
    atom_rate_init: float = 1e-6
    prior_rate_init: float = 1e-4
    boundary_scale_init: float = 1.0
    # ORIGIN-v3 uses bounded null, atom-mass, and boundary-scale
    # parameterizations.  These bounds are part of the architecture identity,
    # not an optimizer-side clamp.
    total_rate_cap: float = 64.0
    prior_rate_cap: float = 1.0
    boundary_scale_cap: float = 2.0
    rate_roundoff_margin: float = 1.0
    atom_mode: str = "cumulative"
    hybrid_cumulative_init: float = 0.9
    evidence_dropout: float = 0.0
    force_decoder_fp64: bool = True
    decision_rule: str = "class_map"

    # Proper ordinal objective and optional evidence budget regularizer.
    rps_weight: float = 0.25
    evidence_budget_weight: float = 0.0
    evidence_budget_delay_epochs: int = 5
    class_weighting: str = "none"
    effective_num_beta: float = 0.999
    class_weight_cap: float = 10.0
    allow_weighted_likelihood: bool = False

    # Optimisation.
    epochs: int = 35
    batch_size: int = 8
    num_workers: int = 4
    pin_memory: bool = True
    lr: float = 1e-4
    head_lr: float = 5e-4
    weight_decay: float = 1e-5
    grad_clip_norm: float = 5.0
    encoder_freeze_epochs: int = 2
    scheduler: str = "plateau"
    lr_factor: float = 0.2
    lr_patience: int = 5
    lr_min: float = 1e-6
    early_stopping_patience: int = 12
    checkpoint_selection: str = "acc_then_qwk"
    selection_qwk_weight: float = 0.10
    amp: bool = True
    amp_init_scale: float = 4096.0
    amp_unfreeze_scale: float = 256.0
    amp_growth_interval: int = 2000
    amp_max_consecutive_skips: int = 8
    resume: bool = False

    # Loader policy. Oversampling is an explicit imbalance ablation because it
    # changes the effective training distribution even with unweighted NLL.
    stratified_batches: bool = False

    # Generic-dataset provenance (unused for the built-in datasets).
    labels_csv: Optional[str] = None
    image_column: Optional[str] = None
    label_column: Optional[str] = None
    image_dir: Optional[str] = None

    def __post_init__(self) -> None:
        self.dataset = self.dataset.lower()
        self.encoder = self.encoder.lower()
        self.scheduler = self.scheduler.lower()
        self.class_weighting = self.class_weighting.lower()
        self.decision_rule = self.decision_rule.lower()
        self.atom_mode = self.atom_mode.lower()
        self.evidence_scales = tuple(
            str(value).lower()
            if str(value).lower().startswith("s")
            else f"s{value}"
            for value in self.evidence_scales
        )

        if self.dataset not in {"aptos", "dr", "busi", "generic"}:
            raise ValueError(f"unsupported ORIGIN dataset: {self.dataset!r}")
        if self.n_classes < 2:
            raise ValueError("n_classes must be at least 2")
        if self.n_folds < 2:
            raise ValueError("n_folds must be at least 2")
        if not 0.0 < self.val_fraction < 1.0:
            raise ValueError("val_fraction must lie strictly between 0 and 1")
        if self.img_size <= 0 or self.batch_size <= 0 or self.epochs <= 0:
            raise ValueError("img_size, batch_size, and epochs must be positive")
        if self.num_workers < 0:
            raise ValueError("num_workers must be non-negative")
        if self.encoder != "convnext_tiny":
            raise ValueError("the initial ORIGIN implementation supports convnext_tiny")
        if self.projection_dim <= 0:
            raise ValueError("projection_dim must be positive")
        allowed_scales = {"s4", "s8", "s16", "s32"}
        if not self.evidence_scales or not set(self.evidence_scales).issubset(allowed_scales):
            raise ValueError(
                "evidence_scales must be selected from s4,s8,s16,s32"
            )
        if len(set(self.evidence_scales)) != len(self.evidence_scales):
            raise ValueError("evidence_scales must not contain duplicates")
        scale_stride = {"s4": 4, "s8": 8, "s16": 16, "s32": 32}
        if tuple(sorted(self.evidence_scales, key=scale_stride.__getitem__)) != self.evidence_scales:
            raise ValueError("evidence_scales must be strictly increasing")
        if not math.isfinite(self.reference_count) or self.reference_count <= 0.0:
            raise ValueError("reference_count must be finite and positive")
        if (
            not math.isfinite(self.atom_rate_init)
            or not math.isfinite(self.prior_rate_init)
            or self.atom_rate_init <= 0.0
            or self.prior_rate_init <= 0.0
        ):
            raise ValueError("initial rates must be positive")
        if not math.isfinite(self.boundary_scale_init) or self.boundary_scale_init <= 0.0:
            raise ValueError("boundary_scale_init must be positive")
        caps = (self.total_rate_cap, self.prior_rate_cap, self.boundary_scale_cap)
        if not all(math.isfinite(value) and value > 0.0 for value in caps):
            raise ValueError("ORIGIN-v3 rate caps must be finite and positive")
        if self.total_rate_cap > 700.0:
            raise ValueError(
                "total_rate_cap must not exceed the audited decoder limit 700"
            )
        if self.total_rate_cap <= self.prior_rate_cap:
            raise ValueError("total_rate_cap must exceed prior_rate_cap")
        if (
            not math.isfinite(self.rate_roundoff_margin)
            or self.rate_roundoff_margin <= 0.0
            or self.rate_roundoff_margin
            >= self.total_rate_cap - self.prior_rate_cap
        ):
            raise ValueError(
                "rate_roundoff_margin must lie strictly in "
                "(0, total_rate_cap - prior_rate_cap)"
            )
        if self.prior_rate_init >= self.prior_rate_cap:
            raise ValueError("prior_rate_init must be smaller than prior_rate_cap")
        if self.boundary_scale_init >= self.boundary_scale_cap:
            raise ValueError(
                "boundary_scale_init must be smaller than boundary_scale_cap"
            )
        atom_mass_cap = (
            (
                self.total_rate_cap
                - self.prior_rate_cap
                - self.rate_roundoff_margin
            )
            / (self.reference_count * self.boundary_scale_cap)
        )
        active_multiplicity = (
            1 if self.atom_mode == "independent" else self.n_classes - 1
        )
        if active_multiplicity * self.atom_rate_init >= atom_mass_cap:
            raise ValueError(
                "the initialized active atoms must leave non-zero null mass "
                "inside the derived ORIGIN-v3 atom_mass_cap "
                f"({atom_mass_cap:g})"
            )
        if self.atom_mode not in {"cumulative", "independent", "hybrid"}:
            raise ValueError(f"unsupported atom_mode: {self.atom_mode!r}")
        if not 0.0 < self.hybrid_cumulative_init < 1.0:
            raise ValueError("hybrid_cumulative_init must lie in (0, 1)")
        if not 0.0 <= self.evidence_dropout < 1.0:
            raise ValueError("evidence_dropout must lie in [0, 1)")
        if not 0.0 < self.mask_valid_fraction <= 1.0:
            raise ValueError("mask_valid_fraction must lie in (0, 1]")
        if self.rps_weight < 0.0 or self.evidence_budget_weight < 0.0:
            raise ValueError("auxiliary loss weights must be non-negative")
        if self.evidence_budget_delay_epochs < 0:
            raise ValueError("evidence_budget_delay_epochs must be non-negative")
        if self.class_weighting not in {"none", "inverse_frequency", "effective_num"}:
            raise ValueError(f"unsupported class_weighting: {self.class_weighting!r}")
        if self.class_weighting != "none" and not self.allow_weighted_likelihood:
            raise ValueError(
                "class weighting changes the fitted population posterior; pass "
                "allow_weighted_likelihood=True only for a declared ablation"
            )
        if not 0.0 <= self.effective_num_beta < 1.0:
            raise ValueError("effective_num_beta must lie in [0, 1)")
        if self.class_weight_cap < 1.0:
            raise ValueError("class_weight_cap must be at least 1")
        if self.scheduler != "plateau":
            raise ValueError(
                "the prospective ORIGIN protocol fixes scheduler='plateau' on "
                "validation loss"
            )
        if not self.force_decoder_fp64:
            raise ValueError("ORIGIN's structural decoder must run in FP64")
        if self.decision_rule not in {"posterior_median", "class_map", "rounded_expected"}:
            raise ValueError(f"unsupported decision_rule: {self.decision_rule!r}")
        if self.checkpoint_selection not in {"acc_then_qwk", "acc_qwk_score"}:
            raise ValueError(
                f"unsupported checkpoint_selection: {self.checkpoint_selection!r}"
            )
        if self.selection_qwk_weight < 0.0:
            raise ValueError("selection_qwk_weight must be non-negative")
        if self.lr <= 0.0 or self.head_lr <= 0.0 or self.lr_min < 0.0:
            raise ValueError("learning rates must be positive (lr_min may be zero)")
        if self.weight_decay < 0.0 or self.grad_clip_norm <= 0.0:
            raise ValueError("weight_decay must be non-negative and grad_clip_norm positive")
        if not 0.0 < self.lr_factor < 1.0:
            raise ValueError("lr_factor must lie in (0, 1)")
        if min(
            self.lr_patience,
            self.early_stopping_patience,
            self.encoder_freeze_epochs,
            self.amp_growth_interval,
            self.amp_max_consecutive_skips,
        ) < 0:
            raise ValueError("epoch/patience counters must be non-negative")
        if self.early_stopping_patience == 0 or self.amp_growth_interval == 0:
            raise ValueError("early stopping patience and AMP growth interval must be positive")
        if (
            not math.isfinite(self.amp_init_scale)
            or not math.isfinite(self.amp_unfreeze_scale)
            or self.amp_init_scale <= 0.0
            or self.amp_unfreeze_scale <= 0.0
        ):
            raise ValueError("AMP scales must be finite and positive")


__all__ = ["ORIGIN_PREPROCESSING_VERSION", "OriginConfig"]
