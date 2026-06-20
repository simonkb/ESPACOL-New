from __future__ import annotations

"""
Combined hybrid loss:

Baseline:
    L_total = alpha * L_PCOL + beta * L_SCOLw + L_RMSE

ESPAOCL extension:
    L_total = alpha * L_PCOL + beta * L_SCOLw + gamma * L_IT + L_RMSE

where L_IT is the image-text ordinal alignment loss.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .pcol import PCOLLoss
from .scolw import SCOLwLoss
from .image_text import ImageTextOrdinalLoss


class HybridContrastiveOrdinalLoss(nn.Module):

    def __init__(
        self,
        alpha: float = 1.0,
        beta: float = 1.0,
        gamma: float = 0.0,
        temperature: float = 0.07,
        use_image_text: bool = False,
        lambda_ord_it: float = 1.0,
    ):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.use_image_text = use_image_text

        self.pcol = PCOLLoss(temperature=temperature)
        self.scolw = SCOLwLoss(temperature=temperature)

        self.image_text = ImageTextOrdinalLoss(
            temperature=temperature,
            lambda_ord=lambda_ord_it,
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

        total = (
            self.alpha * l_pcol
            + self.beta * l_scolw
            + self.gamma * l_it
            + l_rmse
        )

        return total, {
            "loss_total": total.item(),
            "loss_pcol": l_pcol.item(),
            "loss_scolw": l_scolw.item(),
            "loss_it": l_it.item(),
            "loss_rmse": l_rmse.item(),
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