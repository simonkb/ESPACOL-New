"""
Cross-Tile Ordinal Transformer (CTOT) — replaces AttentionPool.

Instead of a single learnable query averaging T tile features, a 2-layer
Transformer encoder with a learnable [GRADE] token lets tiles communicate
via multi-head self-attention. The [GRADE] token output becomes the image
representation, capturing spatial patterns across the 3x3 retinal grid.

Architecture:
    (N, T, 1280) -> Linear(1280, dim) -> prepend [GRADE] -> pos_emb
                 -> 2x pre-LN TransformerEncoderLayer -> [GRADE] output
                 -> Linear(dim, 1280) -> (N, 1280)

Tile layout for positional embeddings (T=10):
    idx 0:  [GRADE] token
    idx 1-9: 3x3 grid  [(0,0), (0,1), (0,2), (1,0), (1,1), (1,2), (2,0), (2,1), (2,2)]
    idx 10: global tile
"""

import math

import torch
import torch.nn as nn


class CrossTileOrdinalTransformer(nn.Module):
    """
    Replaces AttentionPool in TiledEfficientNetBackbone.
    Input : (N, T, feat_dim)
    Output: (N, feat_dim)    — same shape as AttentionPool for drop-in compatibility
    """

    def __init__(
        self,
        feat_dim: int = 1280,
        dim: int = 512,
        nhead: int = 8,
        n_layers: int = 2,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.feat_dim = feat_dim
        self.dim = dim

        self.input_proj = nn.Linear(feat_dim, dim)

        # Learnable [GRADE] token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        # Positional embeddings: max 32 positions to support any tile count + CLS
        self.pos_emb = nn.Parameter(torch.zeros(1, 32, dim))
        self._init_pos_emb()

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,  # pre-LayerNorm — more stable training
        )
        # enable_nested_tensor=False: pre-LN (norm_first=True) disables the nested-tensor
        # optimisation anyway, so disable explicitly to suppress the PyTorch UserWarning.
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=n_layers, enable_nested_tensor=False
        )

        self.output_proj = nn.Linear(dim, feat_dim)

        self._init_weights()

    def _init_pos_emb(self) -> None:
        # Sinusoidal initialization for positional embeddings
        pos = torch.arange(32).unsqueeze(1).float()  # (32, 1)
        dim_half = self.dim // 2
        div_term = torch.exp(
            torch.arange(0, dim_half, 1).float() * (-math.log(10000.0) / dim_half)
        )
        pe = torch.zeros(1, 32, self.dim)
        pe[0, :, 0:dim_half] = torch.sin(pos * div_term)
        pe[0, :, dim_half:dim_half * 2] = torch.cos(pos * div_term)
        self.pos_emb.data.copy_(pe)

    def _init_weights(self) -> None:
        nn.init.xavier_uniform_(self.input_proj.weight)
        nn.init.zeros_(self.input_proj.bias)
        nn.init.xavier_uniform_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (N, T, feat_dim)
        N, T, _ = x.shape

        x = self.input_proj(x)                              # (N, T, dim)
        cls = self.cls_token.expand(N, -1, -1)              # (N, 1, dim)
        x = torch.cat([cls, x], dim=1)                      # (N, T+1, dim)
        x = x + self.pos_emb[:, : T + 1, :]                # add positional bias

        x = self.transformer(x)                             # (N, T+1, dim)
        cls_out = x[:, 0]                                   # (N, dim) — [GRADE] token

        return self.output_proj(cls_out)                    # (N, feat_dim)
