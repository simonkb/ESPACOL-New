"""Image-to-certificate MOSAIC model.

This wrapper is intentionally separate from ``models.framework``.  The final
grade has exactly one computational path:

``full canvas -> bounded local features -> local ordinal states -> exact
cardinality proof -> continuation probabilities``.

There is no globally pooled feature, residual logit, text branch, or CORAL
head that can bypass the reported proof.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn

from .local_efficientnet import (
    LatticeMetadata,
    LocalEfficientNetV2S,
    downsample_retinal_field_mask,
)
from .mosaic import MOSAICOutput, MOSAICProofHead
from utils.spatial_mask import centered_ellipse_mask


@dataclass
class MOSAICModelOutput:
    """Complete image forward trace used by training and certificate export."""

    evidence: MOSAICOutput
    valid_mask: torch.Tensor
    lattice: LatticeMetadata
    local_features: Optional[torch.Tensor] = None

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
    def class_probabilities(self) -> torch.Tensor:
        return self.evidence.class_probabilities

    @property
    def predicted_grade(self) -> torch.Tensor:
        return self.evidence.predicted_grade

    @property
    def argmax_grade(self) -> torch.Tensor:
        return self.evidence.argmax_grade

    @property
    def expected_grade(self) -> torch.Tensor:
        return self.evidence.expected_grade

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
        return MOSAICModelOutput(
            evidence=evidence,
            valid_mask=local.valid_mask,
            lattice=local.lattice,
            local_features=local.tokens if return_local_features else None,
        )


def build_mosaic_model(**kwargs) -> MOSAICModel:
    """Named builder mirroring the repository's existing model factory style."""
    return MOSAICModel(**kwargs)


__all__ = ["MOSAICModel", "MOSAICModelOutput", "build_mosaic_model"]
