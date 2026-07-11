"""
Projection heads and regression head as described in the paper (Section 3):
  "The projection heads in contrastive learning blocks have 1280 and 128 neurons."
  "each consisting of two dense layers with 1280 and 128 neurons."
  "This final component is trained with a root-mean-squared loss (LRMSE)."

Two identical MLP projection heads (one for PCOL, one for SCOLw):
  Linear(1280, 1280) -> BN -> ReLU -> Linear(1280, 128) -> L2-normalize

Regression head:
  Linear(1280, 1)  (takes GAP features, predicts continuous disease grade)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionPool(nn.Module):
    """Softmax attention pooling of T tile embeddings into a single vector."""

    def __init__(self, dim: int):
        super().__init__()
        self.query = nn.Parameter(torch.randn(dim) * (dim ** -0.5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (N, T, D)
        scale = x.shape[-1] ** -0.5
        scores = torch.einsum("d,ntd->nt", self.query, x) * scale   # (N, T)
        weights = torch.softmax(scores, dim=-1)                       # (N, T)
        return torch.einsum("nt,ntd->nd", weights, x)                 # (N, D)


class MLPProjectionHead(nn.Module):
    """
    2-layer MLP: input_dim -> hidden_dim -> out_dim, L2-normalized output.
    BatchNorm + ReLU between layers (standard contrastive learning design).
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


class OrdinalDistributionHead(nn.Module):
    """
    CORAL ordinal regression head (Cao et al. 2020).

    Predicts K-1 threshold probabilities P(Y > k) for k = 0..K-2 using a
    shared linear weight with per-threshold bias offsets.  The expected grade
    (sum of probabilities) is used as the continuous prediction.

    Architecture:
        h = Linear(input_dim, 1)(x)          # (N, 1) shared score
        logits = h + bias                     # (N, K-1) per-threshold logits
        probs  = sigmoid(logits)              # (N, K-1) P(Y > k)
        pred   = probs.sum(dim=1)             # (N,)     expected grade in [0, K-1]
    """

    def __init__(self, input_dim: int = 1280, n_classes: int = 5):
        super().__init__()
        self.fc = nn.Linear(input_dim, 1)
        self.bias = nn.Parameter(torch.zeros(n_classes - 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.fc(x)                                # (N, 1)
        return h + self.bias.unsqueeze(0)             # (N, K-1)  raw logits

    def predict(self, logits: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(logits).sum(dim=1)       # (N,)  expected grade
