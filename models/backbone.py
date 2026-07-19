"""
EfficientNet-V2S backbone as specified in the paper (Section 3):
  "We use EfficientNet-V2S as the backbone encoder."
  "The feature map ψ from the encoder is passed through global average pooling
   (grey layer) to convert into feature-embeddings."

EfficientNet-V2S outputs 1280-dim features after GAP.
"""

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint_sequential
from torchvision.models import efficientnet_v2_s, EfficientNet_V2_S_Weights

from .heads import AttentionPool
from .tile_transformer import CrossTileOrdinalTransformer


class EfficientNetV2SBackbone(nn.Module):
    """
    EfficientNet-V2S with classifier removed.
    Returns 1280-dim GAP features.

    grad_checkpoint=True splits features into 4 segments and applies gradient
    checkpointing, trading ~60% of activation memory for an extra backward recompute.
    Useful when batch×tiles is large (multi-tile T=10), where the high-resolution
    early stages (150×150, 75×75) dominate GPU activation memory.
    """

    OUT_DIM = 1280

    def __init__(self, pretrained: bool = True, grad_checkpoint: bool = False):
        super().__init__()
        weights = EfficientNet_V2_S_Weights.DEFAULT if pretrained else None
        base = efficientnet_v2_s(weights=weights)
        self.features = base.features
        self.avgpool = base.avgpool   # AdaptiveAvgPool2d -> (N, 1280, 1, 1)
        self.grad_checkpoint = grad_checkpoint

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.grad_checkpoint and self.training and torch.is_grad_enabled():
            # use_reentrant=True re-triggers forward hooks during backward recompute,
            # which is required for LayerCAM to capture the correct activations.
            x = checkpoint_sequential(self.features, 4, x, use_reentrant=True)
        else:
            x = self.features(x)
        x = self.avgpool(x)
        return torch.flatten(x, 1)   # (N, 1280)


class TiledEfficientNetBackbone(nn.Module):
    """
    Multi-tile wrapper around EfficientNetV2SBackbone.
    Input : (N, T, C, H, W)  — T tiles per image
    Output: (N, 1280)         — pooled over tile features

    use_transformer=True replaces AttentionPool with CrossTileOrdinalTransformer,
    allowing tiles to exchange information via multi-head self-attention before
    aggregation. Output shape is identical (N, 1280) for drop-in compatibility.
    """

    OUT_DIM = 1280

    def __init__(
        self,
        pretrained: bool = True,
        grad_checkpoint: bool = False,
        use_transformer: bool = False,
        transformer_dim: int = 512,
        transformer_nhead: int = 8,
        transformer_layers: int = 2,
        transformer_dropout: float = 0.1,
    ):
        super().__init__()
        self.base = EfficientNetV2SBackbone(pretrained=pretrained, grad_checkpoint=grad_checkpoint)
        if use_transformer:
            self.pool = CrossTileOrdinalTransformer(
                feat_dim=self.OUT_DIM,
                dim=transformer_dim,
                nhead=transformer_nhead,
                n_layers=transformer_layers,
                dropout=transformer_dropout,
            )
        else:
            self.pool = AttentionPool(self.OUT_DIM)

    def forward(
        self, x: torch.Tensor, return_tile_feats: bool = False
    ) -> torch.Tensor:
        N, T, C, H, W = x.shape
        feats = self.base(x.view(N * T, C, H, W))   # (N*T, 1280)
        tile_feats = feats.view(N, T, -1)             # (N, T, 1280)
        pooled = self.pool(tile_feats)                 # (N, 1280)
        if return_tile_feats:
            return pooled, tile_feats
        return pooled
