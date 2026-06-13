from __future__ import annotations

"""
Combined hybrid loss with optional TAMO extension.

Baseline:
    L_total = alpha * L_PCOL + beta * L_SCOLw + L_RMSE

ESPAOCL extension (image-text):
    L_total = alpha * L_PCOL + beta * L_SCOLw + gamma_it * L_IT + L_RMSE

TAMO extension:
    L_total = alpha * L_PCOL + beta * L_SCOLw
              + gamma_it * L_IT
              + gamma_tamo * (L_PMD + lambda_orc * L_ORC)
              + L_RMSE

L_PMD and L_ORC are the two terms of TAMOLoss (see losses/tamo.py).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .pcol import PCOLLoss
from .scolw import SCOLwLoss
from .image_text import ImageTextOrdinalLoss
from .tamo import TAMOLoss


class HybridContrastiveOrdinalLoss(nn.Module):

    def __init__(
        self,
        alpha: float = 1.0,
        beta: float = 1.0,
        gamma: float = 0.0,
        temperature: float = 0.07,
        use_image_text: bool = False,
        lambda_ord_it: float = 1.0,
        # TAMO parameters
        use_tamo: bool = False,
        gamma_tamo: float = 0.1,
        lambda_orc: float = 1.0,
        huber_delta: float = 0.1,
    ):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.use_image_text = use_image_text
        self.use_tamo = use_tamo
        self.gamma_tamo = gamma_tamo

        self.pcol = PCOLLoss(temperature=temperature)
        self.scolw = SCOLwLoss(temperature=temperature)

        self.image_text = ImageTextOrdinalLoss(
            temperature=temperature,
            lambda_ord=lambda_ord_it,
        )

        # TAMOLoss is instantiated regardless of use_tamo so forward() can
        # be called uniformly; it returns zeros when use_tamo=False.
        self.tamo = TAMOLoss(
            lambda_orc=lambda_orc,
            huber_delta=huber_delta,
        )

    def forward(
        self,
        z_pcol: torch.Tensor,
        z_scolw: torch.Tensor,
        pred: torch.Tensor,
        labels: torch.Tensor,
        class_weights: torch.Tensor,
        z_it: torch.Tensor | None = None,
        text_prototypes: torch.Tensor | None = None,
        z_tamo: torch.Tensor | None = None,
        text_dist_matrix: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict]:

        l_pcol = self.pcol(z_pcol, labels)
        l_scolw = self.scolw(z_scolw, labels, class_weights)
        l_rmse = torch.sqrt(F.mse_loss(pred, labels.float()) + 1e-8)

        l_it = torch.tensor(0.0, device=pred.device)
        if self.use_image_text and self.gamma > 0.0:
            if z_it is None:
                raise ValueError("z_it must be provided when use_image_text=True.")
            if text_prototypes is None:
                raise ValueError("text_prototypes must be provided when use_image_text=True.")
            l_it = self.image_text(
                image_embeddings=z_it,
                labels=labels,
                text_prototypes=text_prototypes,
            )

        l_pmd = torch.tensor(0.0, device=pred.device)
        l_orc = torch.tensor(0.0, device=pred.device)
        if self.use_tamo and self.gamma_tamo > 0.0:
            if z_tamo is None:
                raise ValueError("z_tamo must be provided when use_tamo=True.")
            if text_dist_matrix is None:
                raise ValueError("text_dist_matrix must be provided when use_tamo=True.")
            l_pmd, l_orc = self.tamo(
                z_tamo=z_tamo,
                labels=labels,
                text_dist_matrix=text_dist_matrix,
            )

        l_tamo = l_pmd + self.tamo.lambda_orc * l_orc

        total = (
            self.alpha * l_pcol
            + self.beta * l_scolw
            + self.gamma * l_it
            + self.gamma_tamo * l_tamo
            + l_rmse
        )

        return total, {
            "loss_total": total.item(),
            "loss_pcol": l_pcol.item(),
            "loss_scolw": l_scolw.item(),
            "loss_it": l_it.item(),
            "loss_rmse": l_rmse.item(),
            "loss_tamo_pmd": l_pmd.item(),
            "loss_tamo_orc": l_orc.item(),
        }


def compute_class_weights(
    labels: list[int],
    n_classes: int,
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    """
    Batch-level inverse-frequency class weights for SCOLw:
        w[c] = N_batch / (n_classes * n_c)

    This follows the authors' clarification that SCOLw weights are computed
    dynamically per mini-batch using inverse class frequency.
    """
    counts = torch.zeros(n_classes)

    for y in labels:
        counts[int(y)] += 1

    counts = counts.clamp(min=1)
    n_total = counts.sum()
    weights = n_total / (n_classes * counts)

    return weights.to(device)
