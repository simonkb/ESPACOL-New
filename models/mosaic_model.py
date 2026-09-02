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
from typing import Optional, Sequence

import torch
import torch.nn as nn

from .local_efficientnet import (
    LatticeMetadata,
    LocalEfficientNetV2S,
    downsample_retinal_field_mask,
)
from .mosaic import MOSAICOutput, MOSAICProofHead
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
    def dense_log_stop_probabilities(self) -> torch.Tensor:
        return self.evidence.dense_log_stop_probabilities

    @property
    def stop_probabilities(self) -> torch.Tensor:
        return self.evidence.stop_probabilities

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
    ) -> None:
        super().__init__()
        self.num_classes = int(num_classes)
        self.image_size = int(image_size)
        self.encoder = LocalEfficientNetV2S(
            tap=local_stage,
            local_dim=local_dim,
            pretrained=pretrained,
            grad_checkpoint=grad_checkpoint,
            image_is_normalized=True,
        )
        expected_side = (self.image_size + self.encoder.output_stride - 1) // self.encoder.output_stride
        canonical_mask = centered_ellipse_mask(self.image_size, self.image_size)
        canonical_lattice_mask = downsample_retinal_field_mask(
            canonical_mask,
            (expected_side, expected_side),
            min_valid_fraction=self.encoder.mask_valid_fraction,
        )
        self.expected_valid_cells = int(canonical_lattice_mask.sum())
        if self.expected_valid_cells <= 0:
            raise RuntimeError("canonical MOSAIC support contains no valid lattice cells")
        self.proof_head = MOSAICProofHead(
            input_dim=local_dim,
            num_classes=self.num_classes,
            expected_num_cells=self.expected_valid_cells,
            initial_abnormal_count=initial_abnormal_count,
            max_count=max_count,
            sufficiency_tolerance=sufficiency_tolerance,
            complement_suppression=complement_suppression,
            implementation=count_implementation,
            block_size=count_block_size,
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
    def receptive_field(self) -> int:
        return self.encoder.receptive_field

    @property
    def output_stride(self) -> int:
        return self.encoder.output_stride

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
        project: bool = True,
        return_pivotality: bool = False,
    ) -> MOSAICOutput:
        return self.proof_head(
            local_features,
            valid_mask=valid_mask,
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
            project=project,
            return_pivotality=return_pivotality,
        )
        decisions = proof_only_decisions(
            evidence.transitions,
            evidence.log_stop_probabilities,
            self._decision_transition_weights,
        )
        return MOSAICModelOutput(
            evidence=evidence,
            valid_mask=local.valid_mask,
            lattice=local.lattice,
            local_features=local.tokens if return_local_features else None,
            decision_rule=self._decision_rule,
            decision_transition_weights=self._decision_transition_weights,
            decisions=decisions,
        )


def build_mosaic_model(**kwargs) -> MOSAICModel:
    """Named builder mirroring the repository's existing model factory style."""
    return MOSAICModel(**kwargs)


__all__ = ["MOSAICModel", "MOSAICModelOutput", "build_mosaic_model"]
