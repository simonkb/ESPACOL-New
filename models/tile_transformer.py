"""
CrossTileOrdinalTransformer (CTOT): 2-layer pre-LN Transformer that replaces
AttentionPool when use_tile_transformer=True.

A learnable [GRADE] token accumulates information from all tiles via self-attention
(analogous to [CLS] in BERT). Its output is the aggregated image representation.

Spatial layout for T=10 (3×3 grid + global):
    Position 0: [GRADE] token
    Position 1–9: 3×3 local tiles in raster order
    Position 10: global tile

Tile-to-tile communication captures spatial disease patterns (e.g., DR spreads
from disc to arcades) that AttentionPool's per-tile independent scoring misses.
Literature support: TransMIL (Shao 2021), DSMIL (Li 2021), HIPT (Chen 2022).
"""

from __future__ import annotations
import math

import torch
import torch.nn as nn


class CrossTileOrdinalTransformer(nn.Module):
    """
    Input : (N, T, input_dim)
    Output: (N, input_dim)  when return_tile_features=False (default)
            (N, input_dim), (N, T+1, d_model)  when return_tile_features=True

    The second output includes [GRADE] at position 0 followed by T tile positions.
    GradePrototypeAttention operates on positions 1: (tiles only).
    """

    def __init__(
        self,
        input_dim: int = 1280,
        d_model: int = 512,
        nhead: int = 8,
        num_layers: int = 2,
        dropout: float = 0.1,
        n_tiles: int = 10,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_tiles = n_tiles
        seq_len = n_tiles + 1   # +1 for [GRADE] token

        self.input_proj = nn.Linear(input_dim, d_model)

        self.grade_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.trunc_normal_(self.grade_token, std=0.02)

        # Sinusoidal-initialized learnable positional embeddings
        self.pos_embed = nn.Parameter(torch.zeros(1, seq_len, d_model))
        self._init_pos_embed(seq_len, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 2,
            dropout=dropout,
            batch_first=True,
            norm_first=True,   # pre-LN: more stable training than post-LN
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)

        self.output_proj = nn.Linear(d_model, input_dim)

    def _init_pos_embed(self, seq_len: int, d_model: int) -> None:
        pe = torch.zeros(seq_len, d_model)
        position = torch.arange(seq_len).unsqueeze(1).float()
        half = d_model // 2
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[:half])
        with torch.no_grad():
            self.pos_embed.copy_(pe.unsqueeze(0))

    def forward(
        self,
        x: torch.Tensor,
        return_tile_features: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        N, T, _ = x.shape
        x = self.input_proj(x)                              # (N, T, d_model)
        grade_tok = self.grade_token.expand(N, 1, self.d_model)
        x = torch.cat([grade_tok, x], dim=1)               # (N, T+1, d_model)
        x = x + self.pos_embed[:, : T + 1, :]
        x = self.transformer(x)                             # (N, T+1, d_model)
        x = self.norm(x)
        grade_feat = x[:, 0, :]                             # (N, d_model)
        out = self.output_proj(grade_feat)                  # (N, input_dim)

        if return_tile_features:
            return out, x   # x: (N, T+1, d_model) — [GRADE] at pos 0
        return out
