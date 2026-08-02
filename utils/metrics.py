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
import torch.nn.functional as F


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


def compute_qwk(
    pred: torch.Tensor,
    labels: torch.Tensor,
    n_classes: int,
) -> float:
    """
    Quadratic Weighted Kappa — standard metric for ordinal medical image grading.

    QWK = 1 - (sum_{i,j} W[i,j] * O[i,j]) / (sum_{i,j} W[i,j] * E[i,j])

    where W[i,j] = (i-j)^2 / (K-1)^2  (quadratic disagreement weight),
    O is the observed confusion matrix, E is the expected confusion matrix
    under the assumption of independent label distributions.
    """
    pred_cls = round_predictions(pred, n_classes)
    y_true = labels.cpu().numpy()
    y_pred = pred_cls.cpu().numpy()

    K = n_classes
    O = np.zeros((K, K), dtype=np.float64)
    for yt, yp in zip(y_true, y_pred):
        O[int(yt), int(yp)] += 1

    w = np.zeros((K, K), dtype=np.float64)
    for i in range(K):
        for j in range(K):
            w[i, j] = (i - j) ** 2 / (K - 1) ** 2

    hist_true = O.sum(axis=1)
    hist_pred = O.sum(axis=0)
    E = np.outer(hist_true, hist_pred) / O.sum()

    num = (w * O).sum()
    den = (w * E).sum()
    if den == 0:
        return 0.0
    return float(1.0 - num / den)


def compute_ece(
    pred: torch.Tensor,
    labels: torch.Tensor,
    n_classes: int,
    n_bins: int = 15,
    ordinal_probs: torch.Tensor | None = None,
) -> float:
    """
    Expected Calibration Error — measures how well model confidence matches accuracy.

    When ordinal_probs (N, K-1) is provided, uses the predicted class probability
    P(Y = k) = P(Y > k-1) - P(Y > k) as the confidence for each prediction.
    Otherwise, uses a uniform confidence of 1/K (regression head has no calibration).

    ECE = sum_b (|B_b| / N) * |acc_b - conf_b|
    """
    pred_cls = round_predictions(pred, n_classes).numpy()
    y_true = labels.cpu().numpy()
    N = len(y_true)

    if ordinal_probs is not None:
        # Reconstruct full class probabilities from P(Y > k)
        probs = ordinal_probs.cpu().float()   # (N, K-1)
        # P(Y=0) = 1 - P(Y>0), P(Y=k) = P(Y>k-1) - P(Y>k), P(Y=K-1) = P(Y>K-2)
        p_gt = torch.zeros(N, n_classes)
        p_gt[:, 0] = 1.0 - probs[:, 0]
        for k in range(1, n_classes - 1):
            p_gt[:, k] = probs[:, k - 1] - probs[:, k]
        p_gt[:, -1] = probs[:, -1]
        p_gt = p_gt.clamp(min=0)
        p_gt = p_gt / p_gt.sum(dim=1, keepdim=True).clamp(min=1e-8)
        conf = p_gt[torch.arange(N), torch.tensor(pred_cls)].numpy()
    else:
        conf = np.full(N, 1.0 / n_classes)

    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (conf > lo) & (conf <= hi)
        if mask.sum() == 0:
            continue
        acc_b = (pred_cls[mask] == y_true[mask]).mean()
        conf_b = conf[mask].mean()
        ece += (mask.sum() / N) * abs(acc_b - conf_b)
    return float(ece)


def evaluate_predictions(
    pred: torch.Tensor,
    labels: torch.Tensor,
    n_classes: int,
    ordinal_probs: torch.Tensor | None = None,
) -> dict:
    """Return dict with acc, mae, qwk, and optionally ece."""
    result = {
        "acc": compute_accuracy(pred, labels, n_classes),
        "mae": compute_mae(pred, labels, n_classes),
        "qwk": compute_qwk(pred, labels, n_classes),
    }
    if ordinal_probs is not None:
        result["ece"] = compute_ece(pred, labels, n_classes, ordinal_probs=ordinal_probs)
    return result


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
