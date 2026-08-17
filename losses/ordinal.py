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


class GradePrototypeCELoss(nn.Module):
    """
    Direct supervision for GradePrototypeAttention (GPA).

    For each image, averages tile_evidence across tiles to get an image-level
    grade distribution, then applies NLL loss against the true label.

        mean_evidence[n, k] = mean_t tile_evidence[n, t, k]   # (N, K)
        L_gpa = NLL(log(mean_evidence), labels)

    This forces the K grade prototypes to become grade-discriminative:
    tiles in a grade-k image should collectively vote for grade k.
    Without this loss GPA prototypes receive only ~0.0003 weighted gradient
    from TCL and are essentially not trained.
    """

    def forward(
        self,
        tile_evidence: torch.Tensor,  # (N, T, K) — GPA softmax output
        labels: torch.Tensor,         # (N,) integer grade labels
    ) -> torch.Tensor:
        mean_evidence = tile_evidence.mean(dim=1)              # (N, K)
        log_evidence = torch.log(mean_evidence + 1e-8)         # (N, K)
        return F.nll_loss(log_evidence, labels)


class TileConsistencyLoss(nn.Module):
    """
    Penalises tiles whose expected grade disagrees with the image-level prediction.

    For each tile t, the expected grade is:
        tile_pred[n, t] = sum_k k * tile_evidence[n, t, k]
    where tile_evidence is the GPA softmax over K grades.

    `pred` is detached so gradients flow only through tile_evidence (GPA), not the
    ordinal head — this makes TCL a one-way teacher-student signal: ODH teaches GPA
    what grade each image is, without TCL interfering with the ordinal head's own loss.

    Loss = mean over (n, t) of  relu(|tile_pred - pred.detach()| - margin)^2

    Args:
        margin: allowable per-tile grade deviation before penalisation (default 0).
                margin=0 penalises any disagreement; margin=1 tolerates ±1 grade
                (not recommended — in practice tile_pred and pred stay within 1.0
                of each other so the loss never activates).
    """

    def __init__(self, margin: float = 0.0):
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
        # detach pred: only GPA (tile_evidence) receives gradient from TCL
        diff = (tile_pred - pred.detach().unsqueeze(1)).abs()  # (N, T)
        return F.relu(diff - self.margin).pow(2).mean()


class OrdinalPrototypeLoss(nn.Module):
    """
    Cross-entropy with ordinal distance penalty in the softmax denominator.

    For each sample, |y - k| is added to class k's logit before the softmax.
    Because |y - y| = 0 for the true class, the numerator is unchanged.
    Only wrong classes are penalised — and farther grades contribute more to the
    denominator, making the loss harder to minimise when the model assigns high
    similarity to far-away grade prototypes.

    This is the same denominator-penalty strategy used by PCOL and SCOLw
    (r_{a,n} = |y_a - y_n| added to negative logits), applied here to the
    cosine prototype classification loss.

    label_smoothing is applied after the ordinal penalty, exactly as in the
    original proto_CE cross-entropy.
    """

    def __init__(self, label_smoothing: float = 0.0):
        super().__init__()
        self.label_smoothing = label_smoothing

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        K = logits.shape[-1]
        k_range = torch.arange(K, device=labels.device, dtype=logits.dtype)
        ord_dist = (labels.float().unsqueeze(1) - k_range.unsqueeze(0)).abs()  # (N, K)
        penalized = logits + ord_dist   # true class unchanged (dist = 0)
        return F.cross_entropy(penalized, labels, label_smoothing=self.label_smoothing)
