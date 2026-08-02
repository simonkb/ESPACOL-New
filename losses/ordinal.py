"""
Ordinal regression losses for the OPTIC architecture.

CoralOrdinalLoss:
    Standard CORAL loss (Cao 2020) for ordinal classification.
    Predicts K-1 binary threshold probabilities P(Y > k) and trains with BCE.
    Naturally encodes ordinal structure: getting grade 4 wrong as grade 0 is
    penalised across all K-1 thresholds, not just one.

OrdinalStochasticDominanceLoss (OSD):
    Novel pairwise loss grounded in first-order stochastic dominance theory.
    For pairs (i, j) with y_i < y_j, enforces F_i(k) >= F_j(k) for all k,
    where F(k) = P(Y <= k) is the CDF.
    Interpretation: if y_i < y_j, image i should have more probability mass
    at lower grades (its CDF lies above j's CDF at every threshold).
    Unlike PCOL/SCOLw which operate on embedding distances, OSD enforces
    ordering in the output distribution space at every threshold level.

TileConsistencyLoss (TCL):
    Penalises tiles whose expected grade strongly disagrees with the
    image-level prediction. Uses differentiable expected-grade per tile
    (weighted sum over grade_evidence distribution). A soft margin of 1 allows
    adjacent-grade discrepancy — only large conflicts are penalised.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class CoralOrdinalLoss(nn.Module):
    """
    Input:
        logits (N, K-1) = raw threshold logits from OrdinalDistributionHead.forward()
        labels (N,)     = integer class labels in [0, K-1]
    Returns:
        Scalar mean BCE loss averaged over all N × (K-1) binary targets.

    Uses binary_cross_entropy_with_logits (numerically stable, AMP-safe).
    """

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        K = logits.shape[1] + 1   # n_classes
        k_range = torch.arange(K - 1, device=labels.device)   # (K-1,)
        targets = (labels.unsqueeze(1) > k_range.unsqueeze(0)).float()  # (N, K-1)
        return F.binary_cross_entropy_with_logits(logits, targets)


class OrdinalStochasticDominanceLoss(nn.Module):
    """
    For pairs (i, j) where y_i < y_j, enforces first-order stochastic dominance:
        F_j(k) - F_i(k) <= 0  for all k   (i.e., CDF of i lies above CDF of j)

    Loss = mean over valid pairs of [sum_k  max(0, F_j(k) - F_i(k) + margin)^2]

    Args:
        margin: slack before penalising — 0.0 means strict enforcement.
    """

    def __init__(self, margin: float = 0.0):
        super().__init__()
        self.margin = margin

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        logits : (N, K-1) = raw threshold logits from OrdinalDistributionHead.forward()
        labels : (N,) integer labels
        """
        probs = torch.sigmoid(logits.float())   # float32 for numerical stability
        cdf = 1.0 - probs   # F(k) = P(Y <= k) = 1 - P(Y > k), shape (N, K-1)

        labels_i = labels.unsqueeze(1)   # (N, 1)
        labels_j = labels.unsqueeze(0)   # (1, N)
        pair_mask = (labels_i < labels_j).float()   # (N, N): 1 iff y_i < y_j

        n_pairs = pair_mask.sum()
        if n_pairs == 0:
            return torch.tensor(0.0, device=probs.device, dtype=probs.dtype)

        # Compute CDF violation for all pairs
        cdf_i = cdf.unsqueeze(1)    # (N, 1, K-1)
        cdf_j = cdf.unsqueeze(0)    # (1, N, K-1)
        # Violation: F_j(k) > F_i(k) means j has unexpectedly more low-grade mass
        violations = F.relu(cdf_j - cdf_i + self.margin)   # (N, N, K-1)
        violation_sq = (violations ** 2).sum(dim=-1)         # (N, N) sum over thresholds

        loss = (pair_mask * violation_sq).sum() / n_pairs
        return loss


class TileConsistencyLoss(nn.Module):
    """
    Penalises tiles whose expected grade strongly disagrees with the image prediction.

    For each tile t, the expected grade is:
        tile_pred[n, t] = sum_k k * tile_evidence[n, t, k]
    where tile_evidence is the GPA softmax over K grades.

    Loss = mean over (n, t) of  relu(|tile_pred - pred_grade| - margin)^2
    The margin of 1 tolerates adjacent-grade discrepancy (expected for boundary tiles).

    Args:
        margin: allowable per-tile grade deviation before penalisation (default 1).
    """

    def __init__(self, margin: float = 1.0):
        super().__init__()
        self.margin = margin

    def forward(
        self,
        tile_evidence: torch.Tensor,  # (N, T, K)
        pred: torch.Tensor,           # (N,) continuous grade prediction
    ) -> torch.Tensor:
        K = tile_evidence.shape[-1]
        k_range = torch.arange(K, device=tile_evidence.device, dtype=tile_evidence.dtype)
        tile_pred = (tile_evidence * k_range).sum(dim=-1)   # (N, T)
        diff = (tile_pred - pred.unsqueeze(1)).abs()         # (N, T)
        return F.relu(diff - self.margin).pow(2).mean()
