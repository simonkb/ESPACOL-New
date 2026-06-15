"""
CORAL: Consistent Rank Logits for ordinal regression.

Reference: Cao et al., "Rank Consistent Ordinal Regression for Neural Networks",
           Pattern Recognition Letters, 2020.

For K ordinal classes, CORAL replaces the scalar RMSE regression with K-1
binary rank classifiers that share all weights except per-rank biases:

    logit_j(x) = w^T phi(x) + b_j,  j = 0..K-2

where phi(x) is a shared feature vector and b_j is a rank-specific bias.
The shared weight vector w ensures rank consistency:
if the model predicts "grade >= 3" it will also predict "grade >= 1" and ">= 2".

Loss: sum of K-1 binary cross-entropies
    L = (1/(N*(K-1))) * sum_i sum_j BCE(sigmoid(logit_j(x_i)), [labels_i > j])

Prediction (continuous, for MAE):
    pred_i = sum_j sigmoid(logit_j(x_i))   ∈ [0, K-1]

Prediction (discrete, for accuracy):
    grade_i = round(pred_i)
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class CoralLoss(nn.Module):
    """
    Binary cross-entropy summed over K-1 rank thresholds.
    Input:
        logits : (N, K-1) raw rank logits from CoralHead.forward()
        labels : (N,) integer grades 0..K-1
    """

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        K = logits.shape[1] + 1
        # rank_targets[n, j] = 1  iff  labels[n] > j
        rank_targets = torch.stack(
            [(labels > j).float() for j in range(K - 1)], dim=1
        )  # (N, K-1)
        return F.binary_cross_entropy_with_logits(logits, rank_targets, reduction="mean")


def coral_predict(logits: torch.Tensor) -> torch.Tensor:
    """
    Convert CORAL logits to a continuous grade prediction in [0, K-1].
    Prediction = number of activated rank thresholds (sum of sigmoid outputs).
    Round to nearest integer for accuracy evaluation.
    """
    return torch.sigmoid(logits).sum(dim=1)
