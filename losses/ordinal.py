"""
CORAL ordinal regression loss — replaces L_RMSE.

Predicts K-1 binary threshold probabilities P(Y > k) for k = 0..K-2,
then optimises each with BCE. The expected grade is sum(P(Y > k)).

Reference: Cao et al. "Rank consistent ordinal regression for neural networks
with application to age estimation." Pattern Recognition Letters (2020).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CoralOrdinalLoss(nn.Module):
    """
    Binary cross-entropy summed over K-1 ordinal thresholds.

    Args:
        n_classes: number of ordinal classes (e.g. 5 for DR grades 0-4)
    """

    def __init__(self, n_classes: int = 5):
        super().__init__()
        self.n_classes = n_classes

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits : (N, n_classes-1) — raw threshold logits (NO sigmoid applied)
            labels : (N,) integer class labels in [0, n_classes-1]

        Returns:
            scalar loss
        """
        K = self.n_classes
        # Build binary targets: targets[n, k] = 1 if labels[n] > k
        thresholds = torch.arange(K - 1, device=labels.device)       # (K-1,)
        targets = (labels.unsqueeze(1) > thresholds).float()          # (N, K-1)

        # binary_cross_entropy_with_logits is AMP-safe (unlike binary_cross_entropy)
        loss = F.binary_cross_entropy_with_logits(logits, targets)
        return loss
