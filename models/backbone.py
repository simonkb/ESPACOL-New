"""
EfficientNet-V2S backbone as specified in the paper (Section 3):
  "We use EfficientNet-V2S as the backbone encoder."
  "The feature map ψ from the encoder is passed through global average pooling
   (grey layer) to convert into feature-embeddings."

EfficientNet-V2S outputs 1280-dim features after GAP.
"""

import torch
import torch.nn as nn
from torchvision.models import efficientnet_v2_s, EfficientNet_V2_S_Weights


class EfficientNetV2SBackbone(nn.Module):
    """
    EfficientNet-V2S with classifier removed.
    Returns 1280-dim GAP features.
    """

    OUT_DIM = 1280

    def __init__(self, pretrained: bool = True):
        super().__init__()
        weights = EfficientNet_V2_S_Weights.DEFAULT if pretrained else None
        base = efficientnet_v2_s(weights=weights)
        self.features = base.features
        self.avgpool = base.avgpool   # AdaptiveAvgPool2d -> (N, 1280, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        return torch.flatten(x, 1)   # (N, 1280)
