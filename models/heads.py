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
    CORAL-style ordinal distribution head (Cao 2020).

    Predicts P(Y > k) for k = 0 .. n_classes-2 via a shared linear weight
    and per-threshold bias terms. This encodes the ordinal constraint that
    P(Y > 0) >= P(Y > 1) >= ... >= P(Y > K-2) is encouraged but not enforced
    — the CORAL loss does the enforcement during training.

    Input : (N, input_dim)
    Output from forward(): (N, K-1)  — sigmoid probabilities P(Y > k)
    Output from predict(): (N,)      — expected grade = sum_k P(Y > k) ∈ [0, K-1]
    """

    def __init__(self, input_dim: int = 1280, n_classes: int = 5):
        super().__init__()
        self.n_classes = n_classes
        self.fc = nn.Linear(input_dim, 1, bias=False)
        # Initialise bias to approximate uniform class spacing
        bias_init = torch.arange(n_classes - 1).float() - (n_classes - 2) / 2.0
        self.bias = nn.Parameter(bias_init)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.fc(x)                           # (N, 1)
        return h + self.bias.unsqueeze(0)        # (N, K-1) — raw logits (no sigmoid)

    def probs(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.forward(x))    # (N, K-1) — P(Y > k), float32-safe

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        return self.probs(x).sum(dim=1)          # (N,) — expected grade ∈ [0, K-1]
