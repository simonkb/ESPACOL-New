"""
Full Hybrid Supervised Contrastive Ordinal Learning framework.

Default architecture (BiomedCLIP image backbone):
  Input (224×224) -> BiomedCLIP ViT-B/16 -> feat_dim features
              |
              |----> PCOL Head       (feat_dim->feat_dim->128, L2-norm)
              |----> SCOLw Head      (feat_dim->feat_dim->128, L2-norm)
              |----> Regression Head (feat_dim->1, scalar)

ESPAOCL extension:
              |----> Image-Text Head (feat_dim->feat_dim->128, L2-norm)

feat_dim is probed from the backbone at build time (OUT_DIM attribute).
During training: all heads are jointly optimized.
During testing: only the regression head is used for inference.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .backbone import BiomedCLIPImageBackbone, EfficientNetV2SBackbone
from .heads import MLPProjectionHead, RegressionHead

_DEFAULT_IMAGE_ENCODER = (
    "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
)


class HybridContrastiveOrdinalModel(nn.Module):

    def __init__(
        self,
        backbone: nn.Module,
        pcol_head: MLPProjectionHead,
        scolw_head: MLPProjectionHead,
        regression_head: RegressionHead,
        image_text_head: MLPProjectionHead | None = None,
    ):
        super().__init__()
        self.backbone = backbone
        self.pcol_head = pcol_head
        self.scolw_head = scolw_head
        self.regression_head = regression_head
        self.image_text_head = image_text_head

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor | None]:
        """
        Returns a dictionary for stable access in both baseline and ESPAOCL modes.

        Returns:
            features : (N, feat_dim)
            z_pcol   : (N, 128)
            z_scolw  : (N, 128)
            z_it     : (N, 128) or None
            pred     : (N,)
        """
        features = self.backbone(x)

        z_pcol = self.pcol_head(features)
        z_scolw = self.scolw_head(features)
        pred = self.regression_head(features)

        z_it = None
        if self.image_text_head is not None:
            z_it = self.image_text_head(features)

        return {
            "features": features,
            "z_pcol": z_pcol,
            "z_scolw": z_scolw,
            "z_it": z_it,
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
    image_encoder_name: str = _DEFAULT_IMAGE_ENCODER,
) -> HybridContrastiveOrdinalModel:
    """
    Build the HybridContrastiveOrdinalModel with BiomedCLIP image backbone.

    proj_hidden_dim: hidden size of MLP projection heads.  0 (default) means
                     auto-set to backbone.OUT_DIM, following the paper's
                     principle of matching hidden dim to feature dim.
    """
    backbone = BiomedCLIPImageBackbone(model_name=image_encoder_name, pretrained=pretrained)
    feat_dim = backbone.OUT_DIM
    hidden_dim = proj_hidden_dim if proj_hidden_dim > 0 else feat_dim

    pcol_head = MLPProjectionHead(feat_dim, hidden_dim, proj_out_dim)
    scolw_head = MLPProjectionHead(feat_dim, hidden_dim, proj_out_dim)
    reg_head = RegressionHead(feat_dim)

    image_text_head = None
    if use_image_text:
        image_text_head = MLPProjectionHead(feat_dim, hidden_dim, proj_out_dim)

    return HybridContrastiveOrdinalModel(
        backbone=backbone,
        pcol_head=pcol_head,
        scolw_head=scolw_head,
        regression_head=reg_head,
        image_text_head=image_text_head,
    )