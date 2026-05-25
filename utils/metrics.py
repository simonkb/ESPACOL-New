"""
Evaluation metrics as used in the paper (Section 3 - Evaluation Metrics):
  - Accuracy (Acc.): percentage of correctly classified samples
  - Mean Absolute Error (MAE): mean of |predicted_class - true_class|

Predictions from the regression head are continuous scalars; they are
rounded to the nearest valid class index (clipped to [0, n_classes-1])
before computing accuracy and MAE.
"""

from __future__ import annotations

import numpy as np
import torch


def round_predictions(pred: torch.Tensor, n_classes: int) -> torch.Tensor:
    """Round continuous regression outputs to valid class integers."""
    return pred.round().long().clamp(0, n_classes - 1)


def compute_accuracy(
    pred: torch.Tensor,
    labels: torch.Tensor,
    n_classes: int,
) -> float:
    """Percentage accuracy after rounding predictions."""
    pred_cls = round_predictions(pred, n_classes)
    correct = (pred_cls == labels).float().sum().item()
    return 100.0 * correct / len(labels)


def compute_mae(
    pred: torch.Tensor,
    labels: torch.Tensor,
    n_classes: int,
) -> float:
    """Mean Absolute Error between rounded predictions and true labels."""
    pred_cls = round_predictions(pred, n_classes)
    return (pred_cls.float() - labels.float()).abs().mean().item()


def evaluate_predictions(
    pred: torch.Tensor,
    labels: torch.Tensor,
    n_classes: int,
) -> dict:
    """Return dict with acc and mae."""
    return {
        "acc": compute_accuracy(pred, labels, n_classes),
        "mae": compute_mae(pred, labels, n_classes),
    }


def per_class_accuracy(
    pred: torch.Tensor,
    labels: torch.Tensor,
    n_classes: int,
) -> dict[int, float]:
    """Class-wise accuracy (for the stacked bar analysis, Fig. 2 in paper)."""
    pred_cls = round_predictions(pred, n_classes)
    result = {}
    for c in range(n_classes):
        mask = labels == c
        if mask.sum() == 0:
            result[c] = float("nan")
        else:
            result[c] = 100.0 * (pred_cls[mask] == labels[mask]).float().mean().item()
    return result


def confusion_stats(
    pred: torch.Tensor,
    labels: torch.Tensor,
    n_classes: int,
) -> dict[int, dict]:
    """
    For each class, return proportions of:
      - correct predictions
      - adjacent-class errors  (|pred - label| == 1)
      - non-adjacent errors    (|pred - label| >= 2)
    Mirrors the stacked bar plot analysis in Fig. 2.
    """
    pred_cls = round_predictions(pred, n_classes)
    result = {}
    for c in range(n_classes):
        mask = labels == c
        if mask.sum() == 0:
            result[c] = {"correct": 0, "adjacent": 0, "other": 0, "n": 0}
            continue
        diff = (pred_cls[mask].float() - labels[mask].float()).abs()
        n = diff.shape[0]
        result[c] = {
            "correct": (diff == 0).float().mean().item(),
            "adjacent": (diff == 1).float().mean().item(),
            "other": (diff >= 2).float().mean().item(),
            "n": n,
        }
    return result
