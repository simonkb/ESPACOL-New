"""
Full Hybrid Supervised Contrastive Ordinal Learning framework.

Baseline architecture:
  Input -> EfficientNet-V2S -> GAP -> 1280-dim features
              |
              |----> PCOL Head       (1280->1280->128, L2-norm)
              |----> SCOLw Head      (1280->1280->128, L2-norm)
              |----> Regression Head (1280->1, scalar)

ESPAOCL extension:
              |----> Image-Text Head (1280->1280->128, L2-norm)

During training: all heads are jointly optimized.
During testing: only the regression head is used for inference.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .backbone import EfficientNetV2SBackbone
from .heads import MLPProjectionHead, RegressionHead


class HybridContrastiveOrdinalModel(nn.Module):

    def __init__(
        self,
        backbone: EfficientNetV2SBackbone,
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
            features : (N, 1280)
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
        """Return 1280-dim GAP features from the EfficientNet-V2S backbone."""
        return self.backbone(x)

    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Inference-only: returns continuous regression output."""
        features = self.backbone(x)
        return self.regression_head(features)


def build_model(
    n_classes: int,
    pretrained: bool = True,
    proj_hidden_dim: int = 1280,
    proj_out_dim: int = 128,
    use_image_text: bool = False,
) -> HybridContrastiveOrdinalModel:
    backbone = EfficientNetV2SBackbone(pretrained=pretrained)
    feat_dim = backbone.OUT_DIM

    pcol_head = MLPProjectionHead(feat_dim, proj_hidden_dim, proj_out_dim)
    scolw_head = MLPProjectionHead(feat_dim, proj_hidden_dim, proj_out_dim)
    reg_head = RegressionHead(feat_dim)

    image_text_head = None
    if use_image_text:
        image_text_head = MLPProjectionHead(feat_dim, proj_hidden_dim, proj_out_dim)

    return HybridContrastiveOrdinalModel(
        backbone=backbone,
        pcol_head=pcol_head,
        scolw_head=scolw_head,
        regression_head=reg_head,
        image_text_head=image_text_head,
    )