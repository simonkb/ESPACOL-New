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


class EfficientNetV2SBackbone(nn.Module):
    """
    EfficientNet-V2S with classifier removed.
    Returns 1280-dim GAP features.
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
            # Split features into 4 segments — saves ~60% activation memory
            # by recomputing each segment's internals during backward pass
            x = checkpoint_sequential(self.features, 4, x, use_reentrant=True)
        else:
            x = self.features(x)
        x = self.avgpool(x)
        return torch.flatten(x, 1)   # (N, 1280)


class TiledEfficientNetBackbone(nn.Module):
    """
    Multi-tile wrapper around EfficientNetV2SBackbone.
    Input : (N, T, C, H, W)  — T tiles per image
    Output: (N, 1280)         — AttentionPool over tile features
    """

    OUT_DIM = 1280

    def __init__(self, pretrained: bool = True, grad_checkpoint: bool = False):
        super().__init__()
        self.base = EfficientNetV2SBackbone(pretrained=pretrained, grad_checkpoint=grad_checkpoint)
        self.pool = AttentionPool(self.OUT_DIM)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        N, T, C, H, W = x.shape
        feats = self.base(x.view(N * T, C, H, W))   # (N*T, 1280)
        feats = feats.view(N, T, -1)                  # (N, T, 1280)
        return self.pool(feats)                        # (N, 1280)
