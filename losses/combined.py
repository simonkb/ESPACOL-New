from __future__ import annotations

"""
Combined hybrid loss:

Baseline:
    L_total = alpha * L_PCOL + beta * L_SCOLw + L_RMSE

ESPAOCL extension:
    L_total = alpha * L_PCOL + beta * L_SCOLw + gamma * L_IT + L_RMSE

OPTIC extension (additive on top of ESPAOCL):
    L_total = alpha * L_PCOL + beta * L_SCOLw + gamma * L_IT
            + L_ord                           [replaces L_RMSE when ordinal_probs provided]
            + lambda_osd * L_OSD              [optional]
            + lambda_tcl * L_TCL              [optional]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .image_text import ImageTextOrdinalLoss
from .ordinal import CoralOrdinalLoss, GradePrototypeCELoss, OrdinalStochasticDominanceLoss, TileConsistencyLoss
from .pcol import PCOLLoss
from .scolw import SCOLwLoss


class HybridContrastiveOrdinalLoss(nn.Module):

    def __init__(
        self,
        alpha: float = 1.0,
        beta: float = 1.0,
        gamma: float = 0.0,
        temperature: float = 0.07,
        use_image_text: bool = False,
        lambda_ord_it: float = 1.0,
        # OPTIC loss components
        lambda_osd: float = 0.0,
        osd_margin: float = 0.0,
        lambda_tcl: float = 0.0,
        tcl_margin: float = 0.0,
        lambda_gpa: float = 0.0,
    ):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.use_image_text = use_image_text
        self.lambda_osd = lambda_osd
        self.lambda_tcl = lambda_tcl
        self.lambda_gpa = lambda_gpa

        self.pcol = PCOLLoss(temperature=temperature)
        self.scolw = SCOLwLoss(temperature=temperature)
        self.image_text = ImageTextOrdinalLoss(
            temperature=temperature,
            lambda_ord=lambda_ord_it,
        )
        self.coral = CoralOrdinalLoss()

        self.osd = OrdinalStochasticDominanceLoss(margin=osd_margin) if lambda_osd > 0 else None
        self.tcl = TileConsistencyLoss(margin=tcl_margin) if lambda_tcl > 0 else None
        self.gpa_ce = GradePrototypeCELoss() if lambda_gpa > 0 else None

    def forward(
        self,
        z_pcol: torch.Tensor,
        z_scolw: torch.Tensor,
        pred: torch.Tensor,
        labels: torch.Tensor,
        class_weights: torch.Tensor,
        z_it: torch.Tensor | None = None,
        text_prototypes: torch.Tensor | None = None,
        ordinal_logits: torch.Tensor | None = None,
        tile_evidence: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict]:

        l_pcol = self.pcol(z_pcol, labels)
        l_scolw = self.scolw(z_scolw, labels, class_weights)

        # Ordinal head active: use CORAL loss instead of RMSE
        if ordinal_logits is not None:
            l_ord = self.coral(ordinal_logits, labels)
            l_rmse = torch.tensor(0.0, device=pred.device)
        else:
            l_rmse = torch.sqrt(F.mse_loss(pred, labels.float()) + 1e-8)
            l_ord = torch.tensor(0.0, device=pred.device)

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

        l_osd = torch.tensor(0.0, device=pred.device)
        if self.osd is not None and ordinal_logits is not None and self.lambda_osd > 0:
            l_osd = self.osd(ordinal_logits, labels)

        l_tcl = torch.tensor(0.0, device=pred.device)
        if self.tcl is not None and tile_evidence is not None and self.lambda_tcl > 0:
            l_tcl = self.tcl(tile_evidence, pred)

        l_gpa = torch.tensor(0.0, device=pred.device)
        if self.gpa_ce is not None and tile_evidence is not None and self.lambda_gpa > 0:
            l_gpa = self.gpa_ce(tile_evidence, labels)

        total = (
            self.alpha * l_pcol
            + self.beta * l_scolw
            + self.gamma * l_it
            + (l_ord if ordinal_logits is not None else l_rmse)
            + self.lambda_osd * l_osd
            + self.lambda_tcl * l_tcl
            + self.lambda_gpa * l_gpa
        )

        return total, {
            "loss_total": total.item(),
            "loss_pcol": l_pcol.item(),
            "loss_scolw": l_scolw.item(),
            "loss_it": l_it.item(),
            "loss_rmse": l_rmse.item(),
            "loss_ord": l_ord.item(),
            "loss_osd": l_osd.item(),
            "loss_tcl": l_tcl.item(),
            "loss_gpa": l_gpa.item(),
        }


def compute_class_weights(
    labels: list[int],
    n_classes: int,
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    """
    Batch-level inverse-frequency class weights for SCOLw:
        w[c] = N_batch / (n_classes * n_c)
    """
    counts = torch.zeros(n_classes)
    for y in labels:
        counts[int(y)] += 1
    counts = counts.clamp(min=1)
    n_total = counts.sum()
    weights = n_total / (n_classes * counts)
    return weights.to(device)
