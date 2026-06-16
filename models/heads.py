"""
Projection heads and regression heads.

Original paper (Section 3):
  "The projection heads in contrastive learning blocks have 1280 and 128 neurons."
  Two identical MLP projection heads (one for PCOL, one for SCOLw):
    Linear(1280, 1280) -> BN -> ReLU -> Linear(1280, 128) -> L2-normalize

This file retains the original MLPProjectionHead for backward compatibility
and adds upgraded variants used when TAMO is enabled:

  DeepProjectionHead:
    3-layer residual MLP with LayerNorm + GELU + skip connection.
    Key improvements over MLPProjectionHead:
      - LayerNorm is stable when computing per-class prototypes (mean of
        embeddings) because it doesn't depend on batch statistics.
      - GELU provides smoother gradients than ReLU.
      - Skip connection from input to block-2 preserves the pretrained
        BiomedCLIP features, accelerating convergence.
      - Expand-compress design mixes features before prototype distance
        computation, enabling richer cross-class geometry learning.

  DeepRegressionHead:
    2-layer MLP regression head (vs. the paper's single linear layer).
    Captures non-linear severity relationships in the constrained TAMO space.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MLPProjectionHead(nn.Module):
    """
    2-layer MLP: input_dim -> hidden_dim -> out_dim, L2-normalized output.
    BatchNorm + ReLU between layers (standard contrastive learning design).
    Kept for backward compatibility with baseline and ablation sweep runs.
    """

    def __init__(
        self,
        input_dim: int = 1280,
        hidden_dim: int = 1280,
        out_dim: int = 128,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.net(x), dim=1)   # unit-sphere embedding


class DeepProjectionHead(nn.Module):
    """
    3-layer residual MLP projection head for the TAMO architecture.

    Architecture:
      x ──► FC(in, 2h) ─ LN ─ GELU ─ Dropout ──────────────────────────────────►
                                                                                  │
      x ──► skip_proj(in, h) ──────────────────────────────────────────►  (+) ─ LN ─ GELU ─ Dropout
                                               FC(2h, h) ───────────────────────►
                                                                                  │
                                                                             FC(h, out)
                                                                                  │
                                                                             L2-normalize

    Why this design for TAMO:
    - Expand-compress: the 2h bottleneck mixes features before prototype
      distance computation, enabling richer cross-class geometry learning.
    - LayerNorm (not BatchNorm): prototype c_k = mean(z | label==k) is a
      mean over a subset of the batch.  BatchNorm running statistics are
      unreliable for small within-class subsets; LayerNorm is instance-wise.
    - Skip from input: orthogonally initialized skip_proj preserves the
      pretrained BiomedCLIP features in early training, accelerating
      convergence from the strong pretrained initialization.
    - Dropout 0.1: mild regularization for small medical datasets where the
      effective per-class sample count per batch is ~4-6.
    """

    def __init__(
        self,
        input_dim: int = 512,
        hidden_dim: int = 512,
        out_dim: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()

        # Block 1: expand to 2x width for richer feature mixing
        self.fc1 = nn.Linear(input_dim, hidden_dim * 2)
        self.norm1 = nn.LayerNorm(hidden_dim * 2)

        # Block 2: compress back to hidden_dim, with skip from raw input
        self.fc2 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        # Orthogonal init ensures the backbone signal can bypass block 1
        # in early training without destructive interference.
        self.skip_proj = nn.Linear(input_dim, hidden_dim, bias=False)

        # Output block: project to embedding dimension
        self.fc3 = nn.Linear(hidden_dim, out_dim)

        self.dropout = nn.Dropout(dropout)
        self.act = nn.GELU()

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.orthogonal_(self.skip_proj.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Block 1: expand
        h = self.dropout(self.act(self.norm1(self.fc1(x))))
        # Block 2: compress with residual from input
        h = self.act(self.norm2(self.fc2(h) + self.skip_proj(x)))
        h = self.dropout(h)
        # Output projection
        out = self.fc3(h)
        return F.normalize(out, dim=1)   # unit-sphere embedding


class RegressionHead(nn.Module):
    """
    Linear regression head that predicts a continuous disease severity score.
    Optimized with RMSE loss; predictions are rounded to the nearest integer
    class at inference time for accuracy evaluation.
    """

    def __init__(self, input_dim: int = 1280):
        super().__init__()
        self.fc = nn.Linear(input_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x).squeeze(-1)    # (N,)


class DeepRegressionHead(nn.Module):
    """
    2-layer MLP regression head for the TAMO architecture.

    A single linear layer cannot capture the non-monotonic severity
    distribution that emerges once TAMO's geometry constraints reshape the
    embedding space.  The intermediate 256-dim hidden layer is deliberately
    smaller than the backbone feature dim to force compression rather than
    memorization.
    """

    def __init__(self, input_dim: int = 512, hidden_dim: int = 256, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)   # (N,)


class AttentionPool(nn.Module):
    """
    Single-query attention pooling over a bag of per-tile feature vectors.

    Used by TiledBiomedCLIPBackbone to aggregate features extracted from
    several high-resolution crops of one fundus image into a single
    feature vector, so a lesion that only appears in one tile (e.g. a
    microaneurysm visible in just one crop) can still dominate the pooled
    representation instead of being averaged away.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.query = nn.Parameter(torch.randn(dim) * 0.02)
        self.scale = dim ** -0.5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (N, T, D) -> (N, D)"""
        scores = (x @ self.query) * self.scale          # (N, T)
        weights = torch.softmax(scores, dim=1)          # (N, T)
        return (weights.unsqueeze(-1) * x).sum(dim=1)   # (N, D)


class CoralHead(nn.Module):
    """
    CORAL ordinal classification head (Cao et al., 2020).

    Replaces scalar RMSE regression with K-1 rank-consistent binary classifiers.
    All classifiers share the same feature-extraction weights; only the per-rank
    biases differ.  This shared-weight CORAL constraint ensures rank consistency:
    if the model predicts "grade >= 3" it will also predict "grade >= 1" and ">= 2".

    Training output : (N, K-1) raw logits — feed to CoralLoss.
    Inference       : predict() returns sum(sigmoid(logits)) ∈ [0, K-1],
                      which rounds to the predicted integer grade.
    """

    def __init__(
        self,
        input_dim: int = 512,
        n_classes: int = 5,
        hidden_dim: int = 256,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.extractor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        # Shared weight implements the CORAL constraint — same w for all K-1 ranks.
        self.weight = nn.Linear(hidden_dim, 1, bias=False)
        # Learnable per-rank biases: one threshold per grade boundary 0→1, 1→2, …
        self.bias = nn.Parameter(torch.zeros(n_classes - 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns (N, K-1) raw rank logits for training with CoralLoss."""
        h = self.extractor(x)
        # (N, 1) broadcasts with (K-1,) bias → (N, K-1)
        return self.weight(h) + self.bias

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Continuous grade prediction in [0, K-1]: sum of sigmoid activations.
        Round to nearest integer for accuracy; use as-is for MAE.
        """
        return torch.sigmoid(self.forward(x)).sum(dim=1)
