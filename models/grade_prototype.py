"""
GradePrototypeAttention (GPA): K learnable prototype vectors (one per grade)
attend over post-CTOT tile features to produce per-tile grade evidence maps.

This provides intrinsic spatial explainability without post-hoc attribution:
    tile_weights[n, k, t]  = attention weight of grade k on tile t for image n
    tile_evidence[n, t, k] = evidence of tile t supporting grade k for image n

Both are produced via scaled dot-product attention between tiles and prototypes.

Inputs : tile_features (N, T+1, d_model) from CrossTileOrdinalTransformer
          — [GRADE] token at position 0 is skipped; GPA uses positions 1: only
Outputs:
    grade_features (N, K, d_model)  — grade-specific image representations
    tile_evidence  (N, T, K)        — softmax over K (which grade does this tile support?)
    tile_weights   (N, K, T)        — softmax over T (which tiles drive this grade?)
"""

from __future__ import annotations
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class GradePrototypeAttention(nn.Module):

    def __init__(self, d_model: int = 512, n_classes: int = 5):
        super().__init__()
        self.d_model = d_model
        self.n_classes = n_classes
        self.prototypes = nn.Parameter(torch.empty(n_classes, d_model))
        nn.init.trunc_normal_(self.prototypes, std=math.sqrt(2.0 / d_model))

    def forward(
        self,
        tile_features: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        tile_features: (N, T+1, d_model) — includes [GRADE] token at pos 0
        """
        tiles = tile_features[:, 1:, :]    # (N, T, d_model)  skip [GRADE]
        N, T, d = tiles.shape

        # Scaled dot-product affinity: tiles vs prototypes
        scale = d ** -0.5
        affinity = torch.einsum("ntd,kd->ntk", tiles, self.prototypes) * scale  # (N, T, K)

        # tile_evidence: softmax over K — "which grade does this tile support?"
        tile_evidence = F.softmax(affinity, dim=-1)          # (N, T, K)

        # tile_weights: softmax over T — "which tiles drive each grade?"
        tile_weights = F.softmax(affinity, dim=1)            # (N, T, K)
        tile_weights = tile_weights.permute(0, 2, 1)         # (N, K, T)

        # Grade-specific features: attention-weighted average of tile features
        grade_features = torch.einsum("nkt,ntd->nkd", tile_weights, tiles)  # (N, K, d)

        return grade_features, tile_evidence, tile_weights
