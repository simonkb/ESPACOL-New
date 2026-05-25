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
