"""
Full Hybrid Supervised Contrastive Ordinal Learning framework (Fig. 1 in paper).

Architecture:
  Input -> EfficientNet-V2S -> GAP -> 1280-dim features
              |                          |
              |----> PCOL Head  (1280->1280->128, L2-norm)
              |----> SCOLw Head (1280->1280->128, L2-norm)
              |----> Regression Head (1280->1, scalar)

During training: all three heads are jointly optimized (single-stage).
During testing: only the regression head is used for inference.
"""

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
    ):
        super().__init__()
        self.backbone = backbone
        self.pcol_head = pcol_head
        self.scolw_head = scolw_head
        self.regression_head = regression_head

    def forward(self, x: torch.Tensor):
        """
        Returns:
            z_pcol   : (N, 128)  L2-normalized embeddings for PCOL loss
            z_scolw  : (N, 128)  L2-normalized embeddings for SCOLw loss
            pred     : (N,)      continuous regression prediction
        """
        features = self.backbone(x)                  # (N, 1280)
        z_pcol = self.pcol_head(features)            # (N, 128)
        z_scolw = self.scolw_head(features)          # (N, 128)
        pred = self.regression_head(features)        # (N,)
        return z_pcol, z_scolw, pred

    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Inference-only: returns continuous regression output."""
        features = self.backbone(x)
        return self.regression_head(features)        # (N,)


def build_model(
    n_classes: int,                  # kept for future use / documentation
    pretrained: bool = True,
    proj_hidden_dim: int = 1280,
    proj_out_dim: int = 128,
) -> HybridContrastiveOrdinalModel:
    backbone = EfficientNetV2SBackbone(pretrained=pretrained)
    feat_dim = backbone.OUT_DIM       # 1280

    pcol_head = MLPProjectionHead(feat_dim, proj_hidden_dim, proj_out_dim)
    scolw_head = MLPProjectionHead(feat_dim, proj_hidden_dim, proj_out_dim)
    reg_head = RegressionHead(feat_dim)

    return HybridContrastiveOrdinalModel(backbone, pcol_head, scolw_head, reg_head)
