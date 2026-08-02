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
            x = checkpoint_sequential(self.features, 4, x, use_reentrant=False)
        else:
            x = self.features(x)
        x = self.avgpool(x)
        return torch.flatten(x, 1)   # (N, 1280)


class TiledEfficientNetBackbone(nn.Module):
    """
    Multi-tile wrapper around EfficientNetV2SBackbone.
    Input : (N, T, C, H, W)  — T tiles per image
    Output: (N, 1280)         — AttentionPool or CTOT aggregated feature

    When use_transformer=True, a CrossTileOrdinalTransformer is used instead
    of AttentionPool. CTOT enables tile-to-tile communication and returns
    post-transformer tile features via forward_tiles() for use by GPA.
    """

    OUT_DIM = 1280

    def __init__(
        self,
        pretrained: bool = True,
        grad_checkpoint: bool = False,
        use_transformer: bool = False,
        tile_grid: int = 3,
        transformer_dim: int = 512,
        transformer_nhead: int = 8,
        transformer_layers: int = 2,
        transformer_dropout: float = 0.1,
    ):
        super().__init__()
        self.use_transformer = use_transformer
        n_tiles = tile_grid * tile_grid + 1   # local tiles + global
        self.base = EfficientNetV2SBackbone(pretrained=pretrained, grad_checkpoint=grad_checkpoint)

        if use_transformer:
            self.pool = CrossTileOrdinalTransformer(
                input_dim=self.OUT_DIM,
                d_model=transformer_dim,
                nhead=transformer_nhead,
                num_layers=transformer_layers,
                dropout=transformer_dropout,
                n_tiles=n_tiles,
            )
        else:
            self.pool = AttentionPool(self.OUT_DIM)

    def _encode_tiles(self, x: torch.Tensor) -> torch.Tensor:
        N, T, C, H, W = x.shape
        feats = self.base(x.view(N * T, C, H, W))   # (N*T, 1280)
        return feats.view(N, T, -1)                   # (N, T, 1280)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tile_feats = self._encode_tiles(x)
        return self.pool(tile_feats)   # (N, 1280) for both pool types

    def forward_tiles(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        Returns (aggregated_features, ctot_tile_features).

        ctot_tile_features is (N, T+1, d_model) when use_transformer=True;
        None otherwise (AttentionPool provides no per-tile post-attention features).
        """
        tile_feats = self._encode_tiles(x)
        if self.use_transformer:
            agg, ctot_feats = self.pool(tile_feats, return_tile_features=True)
            return agg, ctot_feats
        else:
            return self.pool(tile_feats), None
