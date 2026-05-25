from __future__ import annotations

"""
Combined hybrid loss - Equation (3) in the paper:

  L_total = α * L_PCOL  +  β * L_SCOLw  +  L_RMSE

All three heads are optimized jointly in a single training stage.

class_weights tensor is pre-computed from the training set as:
  w[c] = N_total / (n_classes * n_c)
so that the average weight across all samples equals 1.0.
"""

import torch
import torch.nn as nn

from .pcol import PCOLLoss
from .scolw import SCOLwLoss


class HybridContrastiveOrdinalLoss(nn.Module):

    def __init__(
        self,
        alpha: float = 1.0,
        beta: float = 1.0,
        temperature: float = 0.07,
    ):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.pcol = PCOLLoss(temperature=temperature)
        self.scolw = SCOLwLoss(temperature=temperature)

    def forward(
        self,
        z_pcol: torch.Tensor,        # (N, D) L2-normed - from PCOL head
        z_scolw: torch.Tensor,       # (N, D) L2-normed - from SCOLw head
        pred: torch.Tensor,          # (N,)   regression output
        labels: torch.Tensor,        # (N,)   integer ordinal labels
        class_weights: torch.Tensor, # (n_classes,) inverse-freq weights
    ) -> tuple[torch.Tensor, dict]:
        """
        Returns:
            total_loss: scalar tensor
            components: dict with individual loss values for logging
        """
        l_pcol = self.pcol(z_pcol, labels)
        l_scolw = self.scolw(z_scolw, labels, class_weights)
        l_rmse = torch.sqrt(
            nn.functional.mse_loss(pred, labels.float())
        )

        total = self.alpha * l_pcol + self.beta * l_scolw + l_rmse

        return total, {
            "loss_total": total.item(),
            "loss_pcol": l_pcol.item(),
            "loss_scolw": l_scolw.item(),
            "loss_rmse": l_rmse.item(),
        }


def compute_class_weights(
    labels: list[int],
    n_classes: int,
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    """
    Inverse-frequency class weights as described in Section 2.3:
      w[c] = N_total / (n_classes * n_c)

    Args:
        labels:    list of integer labels for the training set
        n_classes: total number of classes
        device:    target device

    Returns:
        Tensor of shape (n_classes,) with weight for each class.
    """
    counts = torch.zeros(n_classes)
    for y in labels:
        counts[y] += 1

    # Replace zeros to avoid division by zero for missing classes
    counts = counts.clamp(min=1)
    n_total = counts.sum()
    weights = n_total / (n_classes * counts)
    return weights.to(device)
