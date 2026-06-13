"""
Full Hybrid Supervised Contrastive Ordinal Learning framework.

Default architecture (BiomedCLIP image backbone):
  Input (224x224) -> BiomedCLIP ViT-B/16 -> feat_dim features
              |
              |----> PCOL Head       (feat_dim -> feat_dim -> 128, L2-norm)
              |----> SCOLw Head      (feat_dim -> feat_dim -> 128, L2-norm)
              |----> Regression Head (feat_dim -> 1, scalar)

ESPAOCL extension (use_image_text=True):
              |----> Image-Text Head (feat_dim -> feat_dim -> 128, L2-norm)

TAMO extension (use_tamo=True, implies use_image_text=True):
              |----> TAMO Head       (feat_dim -> feat_dim -> 128, L2-norm)
                     Uses DeepProjectionHead for all heads when TAMO is active.

feat_dim is probed from the backbone at build time (OUT_DIM attribute).

When use_tamo=True, all projection heads are upgraded to DeepProjectionHead
(3-layer residual MLP with LayerNorm + GELU).  This provides higher-capacity
representations needed for the prototype geometry constraints in TAMO.
The regression head is also upgraded to DeepRegressionHead for consistency.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .backbone import BiomedCLIPImageBackbone, EfficientNetV2SBackbone
from .heads import (
    MLPProjectionHead,
    DeepProjectionHead,
    RegressionHead,
    DeepRegressionHead,
)

_DEFAULT_IMAGE_ENCODER = (
    "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
)


class HybridContrastiveOrdinalModel(nn.Module):

    def __init__(
        self,
        backbone: nn.Module,
        pcol_head: nn.Module,
        scolw_head: nn.Module,
        regression_head: nn.Module,
        image_text_head: nn.Module | None = None,
        tamo_head: nn.Module | None = None,
    ):
        super().__init__()
        self.backbone = backbone
        self.pcol_head = pcol_head
        self.scolw_head = scolw_head
        self.regression_head = regression_head
        self.image_text_head = image_text_head
        # Dedicated TAMO projection head — separate from the image-text head
        # to avoid conflicting gradient directions between sample-level IT
        # alignment and prototype-level geometry distillation.
        self.tamo_head = tamo_head

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor | None]:
        """
        Returns a dictionary for stable access in both baseline and TAMO modes.

        Returns:
            features : (N, feat_dim)
            z_pcol   : (N, 128)
            z_scolw  : (N, 128)
            z_it     : (N, 128) or None
            z_tamo   : (N, 128) or None
            pred     : (N,)
        """
        features = self.backbone(x)

        z_pcol = self.pcol_head(features)
        z_scolw = self.scolw_head(features)
        pred = self.regression_head(features)

        z_it = None
        if self.image_text_head is not None:
            z_it = self.image_text_head(features)

        z_tamo = None
        if self.tamo_head is not None:
            z_tamo = self.tamo_head(features)

        return {
            "features": features,
            "z_pcol": z_pcol,
            "z_scolw": z_scolw,
            "z_it": z_it,
            "z_tamo": z_tamo,
            "pred": pred,
        }

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Return backbone features (dim = backbone.OUT_DIM)."""
        return self.backbone(x)

    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Inference-only: returns continuous regression output."""
        features = self.backbone(x)
        return self.regression_head(features)


def build_model(
    n_classes: int,
    pretrained: bool = True,
    proj_hidden_dim: int = 0,
    proj_out_dim: int = 128,
    use_image_text: bool = False,
    use_tamo: bool = False,
    image_encoder_name: str = _DEFAULT_IMAGE_ENCODER,
) -> HybridContrastiveOrdinalModel:
    """
    Build the HybridContrastiveOrdinalModel with BiomedCLIP image backbone.

    proj_hidden_dim: hidden size of MLP projection heads.  0 (default) means
                     auto-set to backbone.OUT_DIM, following the paper's
                     principle of matching hidden dim to feature dim.

    use_tamo: when True, upgrades all projection and regression heads to the
              deeper variants (DeepProjectionHead, DeepRegressionHead) and
              adds a dedicated tamo_head.  This is the TAMO architecture.
              When False, uses the paper's original MLPProjectionHead.
    """
    backbone = BiomedCLIPImageBackbone(model_name=image_encoder_name, pretrained=pretrained)
    feat_dim = backbone.OUT_DIM
    hidden_dim = proj_hidden_dim if proj_hidden_dim > 0 else feat_dim

    if use_tamo:
        # Upgraded heads for TAMO: deeper capacity, LayerNorm-stable for
        # prototype computations, skip connections for fast convergence.
        pcol_head = DeepProjectionHead(feat_dim, hidden_dim, proj_out_dim)
        scolw_head = DeepProjectionHead(feat_dim, hidden_dim, proj_out_dim)
        reg_head = DeepRegressionHead(feat_dim, hidden_dim=min(hidden_dim, 256))
        # IT head kept as deep variant for consistency with TAMO embedding space.
        image_text_head = DeepProjectionHead(feat_dim, hidden_dim, proj_out_dim)
        # TAMO head is always present when use_tamo=True.
        tamo_head = DeepProjectionHead(feat_dim, hidden_dim, proj_out_dim)
    else:
        # Original paper heads — used for baseline and ablation sweep runs.
        pcol_head = MLPProjectionHead(feat_dim, hidden_dim, proj_out_dim)
        scolw_head = MLPProjectionHead(feat_dim, hidden_dim, proj_out_dim)
        reg_head = RegressionHead(feat_dim)
        image_text_head = MLPProjectionHead(feat_dim, hidden_dim, proj_out_dim) if use_image_text else None
        tamo_head = None

    return HybridContrastiveOrdinalModel(
        backbone=backbone,
        pcol_head=pcol_head,
        scolw_head=scolw_head,
        regression_head=reg_head,
        image_text_head=image_text_head,
        tamo_head=tamo_head,
    )
