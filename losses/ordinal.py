"""
CORAL ordinal regression loss — replaces L_RMSE.

Predicts K-1 binary threshold probabilities P(Y > k) for k = 0..K-2,
then optimises each with balanced BCE. Without per-threshold pos_weight,
severe class imbalance (e.g. 73% grade-0 in DR) causes the model to predict
the majority class for all inputs — achieving a high "accuracy" that is
actually a degenerate no-op.

pos_weight for threshold k = sqrt(n_negative_k / n_positive_k), capped at 10.
Square-root smoothing avoids the instability of the raw ratio (which can reach
49× for the highest DR threshold).

Reference: Cao et al. "Rank consistent ordinal regression for neural networks
with application to age estimation." Pattern Recognition Letters (2020).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CoralOrdinalLoss(nn.Module):
    """
    Balanced binary cross-entropy over K-1 ordinal thresholds.

    Args:
        n_classes    : number of ordinal classes (e.g. 5 for DR grades 0-4)
        class_counts : list/tensor of per-class sample counts; when provided,
                       per-threshold pos_weight = sqrt(n_neg_k / n_pos_k)
                       capped at max_pos_weight to prevent gradient explosion.
        max_pos_weight: cap on per-threshold weight (default 10)
    """

    def __init__(
        self,
        n_classes: int = 5,
        class_counts=None,
        max_pos_weight: float = 10.0,
    ):
        super().__init__()
        self.n_classes = n_classes
        self.pos_weight = None

        if class_counts is not None:
            counts = torch.tensor(class_counts, dtype=torch.float)
            K = n_classes
            weights = []
            for k in range(K - 1):
                n_pos = counts[k + 1 :].sum().item()
                n_neg = counts[: k + 1].sum().item()
                # sqrt smoothing: gentler than raw ratio, prevents explosion
                w = min((n_neg / max(n_pos, 1)) ** 0.5, max_pos_weight)
                weights.append(w)
            self.register_buffer("pos_weight", torch.tensor(weights))

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits : (N, n_classes-1) — raw threshold logits (NO sigmoid applied)
            labels : (N,) integer class labels in [0, n_classes-1]

        Returns:
            scalar loss
        """
        K = self.n_classes
        thresholds = torch.arange(K - 1, device=labels.device)    # (K-1,)
        targets = (labels.unsqueeze(1) > thresholds).float()       # (N, K-1)

        pw = None
        if self.pos_weight is not None:
            pw = self.pos_weight.to(logits.device)

        # binary_cross_entropy_with_logits is AMP-safe
        return F.binary_cross_entropy_with_logits(logits, targets, pos_weight=pw)
