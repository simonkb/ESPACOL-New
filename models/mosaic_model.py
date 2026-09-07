"""Image-to-certificate MOSAIC model.

This wrapper is intentionally separate from ``models.framework``.  The final
grade has exactly one computational path:

``full canvas -> bounded local features -> local ordinal states -> exact
cardinality proof -> loss-aware proof decoder -> ordinal grade``.

There is no globally pooled feature, residual logit, text branch, or CORAL
head that can bypass the reported proof.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Optional, Sequence

import torch
import torch.nn as nn

from .local_efficientnet import (
    LatticeMetadata,
    LocalEfficientNetV2S,
    ReceptiveFieldMetadata,
    downsample_retinal_field_mask,
)
from .mosaic import (
    MOSAICOutput,
    MOSAICProofHead,
    normalize_region_pool_type,
    receptive_field_packing_mask,
)
from .mosaic_decoder import (
    PROOF_DECISION_RULES,
    ProofOnlyDecisionBundle,
    proof_only_decisions,
)
from utils.spatial_mask import centered_ellipse_mask


@dataclass
class MOSAICModelOutput:
    """Complete image forward trace used by training and certificate export."""

    evidence: MOSAICOutput
    valid_mask: torch.Tensor
    lattice: LatticeMetadata
    local_features: Optional[torch.Tensor] = None
    source_lattice: Optional[LatticeMetadata] = None
    source_valid_mask: Optional[torch.Tensor] = None
    decision_rule: str = "rounded_expected"
    decision_transition_weights: Optional[torch.Tensor] = None
    decisions: Optional[ProofOnlyDecisionBundle] = None

    # Short properties keep trainer code readable while retaining a clearly
    # separated encoder-independent mathematical core.
    @property
    def transitions(self) -> torch.Tensor:
        return self.evidence.transitions

    @property
    def dense_transitions(self) -> torch.Tensor:
        return self.evidence.dense_transitions

    @property
    def dense_stop_probabilities(self) -> torch.Tensor:
        return self.evidence.dense_stop_probabilities

    @property
    def dense_log_transition_probabilities(self) -> torch.Tensor:
        return self.evidence.dense_log_transition_probabilities

    @property
    def dense_log_stop_probabilities(self) -> torch.Tensor:
        return self.evidence.dense_log_stop_probabilities

    @property
    def stop_probabilities(self) -> torch.Tensor:
        return self.evidence.stop_probabilities

    @property
    def log_transition_probabilities(self) -> torch.Tensor:
        return self.evidence.log_transition_probabilities

    @property
    def log_stop_probabilities(self) -> torch.Tensor:
        return self.evidence.log_stop_probabilities

    @property
    def raw_cumulative_probabilities(self) -> torch.Tensor:
        """Unmodified cumulative law emitted by the cardinality core."""

        return self.evidence.cumulative_probabilities

    @property
    def raw_class_probabilities(self) -> torch.Tensor:
        """Unmodified class law emitted by the cardinality core."""

        return self.evidence.class_probabilities

    @property
    def raw_predicted_grade(self) -> torch.Tensor:
        """Legacy rounded-expected decision before outcome deweighting."""

        return self.evidence.predicted_grade

    @property
    def raw_argmax_grade(self) -> torch.Tensor:
        return self.evidence.argmax_grade

    @property
    def raw_expected_grade(self) -> torch.Tensor:
        return self.evidence.expected_grade

    def _require_deweighted_decisions(self) -> ProofOnlyDecisionBundle:
        if self.decisions is None:
            raise RuntimeError(
                "deweighted decisions are unavailable because at least one "
                "training outcome weight is zero"
            )
        return self.decisions

    @property
    def cumulative_probabilities(self) -> torch.Tensor:
        """Cumulative law selected by the configured proof-only decoder."""

        if self.decision_rule.startswith("deweighted_"):
            return self._require_deweighted_decisions().deweighted_cumulative_probabilities
        return self.raw_cumulative_probabilities

    @property
    def class_probabilities(self) -> torch.Tensor:
        """Class law selected by the configured proof-only decoder."""

        if self.decision_rule.startswith("deweighted_"):
            return self._require_deweighted_decisions().deweighted_class_probabilities
        return self.raw_class_probabilities

    @property
    def expected_grade(self) -> torch.Tensor:
        """Posterior mean under the selected raw or deweighted law."""

        if self.decision_rule.startswith("deweighted_"):
            return self._require_deweighted_decisions().deweighted_expected_grade
        return self.raw_expected_grade

    @property
    def predicted_grade(self) -> torch.Tensor:
        """Final grade selected by ``decision_rule`` from the reported proof."""

        if self.decision_rule == "rounded_expected":
            return self.raw_predicted_grade
        if self.decision_rule == "class_map":
            return self.raw_argmax_grade
        if self.decision_rule == "posterior_median":
            return (self.raw_cumulative_probabilities >= 0.5).sum(dim=-1)
        decisions = self._require_deweighted_decisions()
        if self.decision_rule == "deweighted_mean_round":
            return decisions.deweighted_mean_round
        if self.decision_rule == "deweighted_class_map":
            return decisions.deweighted_argmax
        if self.decision_rule == "deweighted_posterior_median":
            return decisions.deweighted_posterior_median
        raise RuntimeError(f"unconfigured decision rule {self.decision_rule!r}")

    @property
    def argmax_grade(self) -> torch.Tensor:
        """MAP grade under the selected raw or deweighted probability law."""

        return self.class_probabilities.argmax(dim=-1)

    @property
    def proof(self):
        return self.evidence.proof


class MOSAICModel(nn.Module):
    """Proof-exclusive ordinal disease severity grader."""

    def __init__(
        self,
        *,
        num_classes: int = 5,
        image_size: int = 896,
        local_stage: str = "rf_medium",
        local_dim: int = 128,
        pretrained: bool = True,
        grad_checkpoint: bool = False,
        initial_abnormal_count: float = 0.5,
        max_count: int = 32,
        sufficiency_tolerance: float = 0.02,
        complement_suppression: float = 0.5,
        count_implementation: str = "block_tree",
        count_block_size: int = 64,
        decision_rule: str = "rounded_expected",
        transition_weights: torch.Tensor | Sequence[Sequence[float]] | None = None,
        region_grid_size: int = 0,
        region_pool_temperature: Optional[float] = 0.25,
        region_pool_type: str = "normalized_logmeanexp",
        rf_packing_max_overlap: Optional[float] = None,
    ) -> None:
        super().__init__()
        self.num_classes = int(num_classes)
        self.image_size = int(image_size)
        if region_grid_size < 0:
            raise ValueError("region_grid_size must be non-negative")
        self.region_grid_size = int(region_grid_size)
        if rf_packing_max_overlap is not None:
            rf_packing_max_overlap = float(rf_packing_max_overlap)
            if (
                not math.isfinite(rf_packing_max_overlap)
                or not 0.0 <= rf_packing_max_overlap < 1.0
            ):
                raise ValueError("rf_packing_max_overlap must lie in [0, 1)")
            if self.region_grid_size:
                raise ValueError(
                    "RF packing and fixed regional compilation are mutually exclusive"
                )
        self.rf_packing_max_overlap = rf_packing_max_overlap
        self.region_pool_type = normalize_region_pool_type(region_pool_type)
        if self.region_pool_type == "normalized_logmeanexp":
            if (
                region_pool_temperature is None
                or not math.isfinite(region_pool_temperature)
                or region_pool_temperature <= 0.0
            ):
                raise ValueError("region_pool_temperature must be finite and positive")
            self.region_pool_temperature: Optional[float] = float(
                region_pool_temperature
            )
        else:
            self.region_pool_temperature = None
        self.encoder = LocalEfficientNetV2S(
            tap=local_stage,
            local_dim=local_dim,
            pretrained=pretrained,
            grad_checkpoint=grad_checkpoint,
            image_is_normalized=True,
        )
        expected_side = (
            self.image_size + self.encoder.output_stride - 1
        ) // self.encoder.output_stride
        canonical_mask = centered_ellipse_mask(self.image_size, self.image_size)
        canonical_lattice_mask = downsample_retinal_field_mask(
            canonical_mask,
            (expected_side, expected_side),
            min_valid_fraction=self.encoder.mask_valid_fraction,
        )
        self.expected_valid_cells = int(canonical_lattice_mask.sum())
        if self.expected_valid_cells <= 0:
            raise RuntimeError("canonical MOSAIC support contains no valid lattice cells")
        self.expected_valid_regions = self.expected_valid_cells
        if self.rf_packing_max_overlap is not None:
            canonical_flat = canonical_lattice_mask.reshape(1, -1)
            canonical_packing = receptive_field_packing_mask(
                torch.zeros_like(canonical_flat, dtype=torch.float32),
                canonical_flat,
                (expected_side, expected_side),
                output_stride=self.encoder.output_stride,
                receptive_field=self.encoder.receptive_field,
                max_overlap=self.rf_packing_max_overlap,
            )
            self.expected_valid_regions = int(canonical_packing.sum())
            if self.expected_valid_regions <= 0:
                raise RuntimeError("canonical MOSAIC RF packing contains no events")
        if self.region_grid_size:
            if expected_side % self.region_grid_size:
                raise ValueError(
                    "the source lattice must divide exactly into the configured "
                    f"regional grid ({expected_side} vs {self.region_grid_size})"
                )
            block = expected_side // self.region_grid_size
            regional_mask = (
                canonical_lattice_mask.reshape(
                    1,
                    self.region_grid_size,
                    block,
                    self.region_grid_size,
                    block,
                )
                .permute(0, 1, 3, 2, 4)
                .reshape(1, self.region_grid_size**2, block * block)
                .any(dim=-1)
            )
            self.expected_valid_regions = int(regional_mask.sum())
            if self.expected_valid_regions <= 0:
                raise RuntimeError("canonical MOSAIC regional support is empty")
        self.proof_head = MOSAICProofHead(
            input_dim=local_dim,
            num_classes=self.num_classes,
            # The head bias calibrates the canonical number of events consumed
            # by the count circuit. For fixed regions that is the number of
            # valid regions; for ORFP it is the deterministic row-major packing
            # capacity at the all-equal initialization. Using all 9,864 source
            # sites would make the intended initial evidence mass wrong by one
            # to two orders of magnitude.
            expected_num_cells=self.expected_valid_regions,
            initial_abnormal_count=initial_abnormal_count,
            max_count=max_count,
            sufficiency_tolerance=sufficiency_tolerance,
            complement_suppression=complement_suppression,
            implementation=count_implementation,
            block_size=count_block_size,
            region_grid_size=self.region_grid_size,
            region_pool_temperature=self.region_pool_temperature,
            region_pool_type=self.region_pool_type,
            rf_packing_max_overlap=self.rf_packing_max_overlap,
            source_output_stride=self.encoder.output_stride,
            source_receptive_field=self.encoder.receptive_field,
        )
        # Runtime decoder metadata is deliberately non-persistent.  It is
        # reconstructed from the training criterion/checkpoint, so legacy
        # model state dictionaries continue to load strictly and no learned
        # parameter can bypass the proof.
        self.register_buffer(
            "_decision_transition_weights",
            torch.ones(self.num_classes - 1, 2),
            persistent=False,
        )
        self._decision_rule = "rounded_expected"
        self.configure_proof_decoder(decision_rule, transition_weights)

    @property
    def decision_rule(self) -> str:
        return self._decision_rule

    @property
    def decision_transition_weights(self) -> torch.Tensor:
        return self._decision_transition_weights

    def configure_proof_decoder(
        self,
        decision_rule: str,
        transition_weights: torch.Tensor | Sequence[Sequence[float]] | None = None,
    ) -> None:
        """Configure the parameter-free final decision from criterion metadata.

        Outcome weights are ordered ``[stop, advance]``.  They are not saved in
        the model state because the criterion checkpoint is their authoritative
        source.  Every outcome must have positive training support: a fold that
        omits one side of a configured ordinal boundary is not a valid fold for
        the declared ``K``-grade model or its complete decoder audit.
        """

        if decision_rule not in PROOF_DECISION_RULES:
            raise ValueError(
                f"unknown MOSAIC decision rule {decision_rule!r}; expected one of "
                f"{PROOF_DECISION_RULES}"
            )
        if transition_weights is None:
            weights = torch.ones(
                self.num_classes - 1,
                2,
                device=self._decision_transition_weights.device,
            )
        else:
            weights = torch.as_tensor(
                transition_weights,
                dtype=torch.float32,
                device=self._decision_transition_weights.device,
            )
        if tuple(weights.shape) != (self.num_classes - 1, 2):
            raise ValueError(
                "transition_weights must have shape "
                f"({self.num_classes - 1}, 2) ordered as [stop, advance]"
            )
        if not bool(torch.isfinite(weights).all()) or bool((weights <= 0.0).any()):
            raise ValueError(
                "every ordinal boundary needs strictly positive stop and advance "
                "training weights; a zero-weight outcome makes the declared "
                "K-grade fold incomplete"
            )
        self._decision_transition_weights.copy_(weights)
        self._decision_rule = decision_rule

    @property
    def proof_tolerance(self) -> float:
        return float(self.proof_head.ordinal_core.projector.sufficiency_tolerance)

    def set_proof_tolerance(self, value: float) -> None:
        if value < 0.0:
            raise ValueError("proof tolerance must be non-negative")
        self.proof_head.ordinal_core.projector.sufficiency_tolerance = float(value)

    @property
    def rf_packing_enabled(self) -> bool:
        return self.rf_packing_max_overlap is not None

    @property
    def expected_valid_proof_events(self) -> int:
        return self.expected_valid_regions

    @property
    def receptive_field(self) -> int:
        return self.encoder.receptive_field

    @property
    def output_stride(self) -> int:
        return self.encoder.output_stride

    @property
    def proof_output_stride(self) -> int:
        if not self.region_grid_size:
            return self.encoder.output_stride
        source_side = (
            self.image_size + self.encoder.output_stride - 1
        ) // self.encoder.output_stride
        return self.encoder.output_stride * (source_side // self.region_grid_size)

    @property
    def proof_receptive_field(self) -> int:
        if not self.region_grid_size:
            return self.encoder.receptive_field
        source_side = (
            self.image_size + self.encoder.output_stride - 1
        ) // self.encoder.output_stride
        block = source_side // self.region_grid_size
        return self.encoder.receptive_field + (block - 1) * self.encoder.output_stride

    def _proof_lattice(self, source: LatticeMetadata) -> LatticeMetadata:
        """Return truthful geometry for the events consumed by the proof.

        A regional event can depend on any valid source RF inside its fixed
        block, so its theoretical support is the union of those RFs. The finer
        source-lattice peak remains available separately as provenance.
        """

        if not self.region_grid_size:
            return source
        source_h, source_w = source.lattice_size
        grid = self.region_grid_size
        if source_h != source_w or source_h % grid or source_w % grid:
            raise ValueError(
                "regional MOSAIC requires a square source lattice divisible "
                f"by {grid}; got {source.lattice_size}"
            )
        block = source_h // grid
        source_rf = source.receptive_field
        pool_tag = (
            "lme"
            if self.region_pool_type == "normalized_logmeanexp"
            else "existential_max"
        )
        regional_rf = ReceptiveFieldMetadata(
            tap=f"{source_rf.tap}_regional_{pool_tag}_{grid}x{grid}",
            feature_index=source_rf.feature_index,
            channels=source_rf.channels,
            output_stride=source_rf.output_stride * block,
            receptive_field=(
                source_rf.receptive_field
                + (block - 1) * source_rf.output_stride
            ),
            center_offset=(
                source_rf.center_offset
                + 0.5 * (block - 1) * source_rf.output_stride
            ),
            squeeze_excitation_removed=source_rf.squeeze_excitation_removed,
            globally_mixed=source_rf.globally_mixed,
        )
        return LatticeMetadata(
            input_size=source.input_size,
            lattice_size=(grid, grid),
            local_dim=source.local_dim,
            receptive_field=regional_rf,
        )

    def forward_features(
        self,
        image: torch.Tensor,
        pixel_valid_mask: Optional[torch.Tensor] = None,
        *,
        return_feature_map: bool = False,
    ):
        """Expose local features for leakage-safe head screening and caching."""
        return self.encoder(
            image,
            pixel_valid_mask=pixel_valid_mask,
            return_feature_map=return_feature_map,
        )

    def forward_from_features(
        self,
        local_features: torch.Tensor,
        valid_mask: torch.Tensor,
        *,
        lattice_size: Optional[tuple[int, int]] = None,
        project: bool = True,
        return_pivotality: bool = False,
    ) -> MOSAICOutput:
        return self.proof_head(
            local_features,
            valid_mask=valid_mask,
            lattice_size=lattice_size,
            project=project,
            return_pivotality=return_pivotality,
        )

    def forward(
        self,
        image: torch.Tensor,
        pixel_valid_mask: Optional[torch.Tensor] = None,
        *,
        project: bool = True,
        return_pivotality: bool = False,
        return_local_features: bool = False,
    ) -> MOSAICModelOutput:
        local = self.forward_features(image, pixel_valid_mask)
        evidence = self.forward_from_features(
            local.tokens,
            local.valid_mask,
            lattice_size=local.lattice.lattice_size,
            project=project,
            return_pivotality=return_pivotality,
        )
        proof_valid_mask = (
            local.valid_mask
            if evidence.evidence_valid_mask is None
            else evidence.evidence_valid_mask
        )
        proof_lattice = self._proof_lattice(local.lattice)
        decisions = proof_only_decisions(
            evidence.transitions,
            evidence.log_stop_probabilities,
            self._decision_transition_weights,
        )
        return MOSAICModelOutput(
            evidence=evidence,
            valid_mask=proof_valid_mask,
            lattice=proof_lattice,
            local_features=local.tokens if return_local_features else None,
            source_lattice=(
                local.lattice
                if self.region_grid_size or self.rf_packing_enabled
                else None
            ),
            source_valid_mask=(
                local.valid_mask
                if self.region_grid_size or self.rf_packing_enabled
                else None
            ),
            decision_rule=self._decision_rule,
            decision_transition_weights=self._decision_transition_weights,
            decisions=decisions,
        )


def build_mosaic_model(**kwargs) -> MOSAICModel:
    """Named builder mirroring the repository's existing model factory style."""
    return MOSAICModel(**kwargs)


__all__ = ["MOSAICModel", "MOSAICModelOutput", "build_mosaic_model"]
