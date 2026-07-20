from __future__ import annotations

"""
Combined hybrid loss:

Baseline:
    L_total = alpha * L_PCOL + beta * L_SCOLw + L_RMSE

ESPAOCL-CT (novel architecture):
    L_total = alpha*L_PCOL + beta*L_SCOLw + gamma*L_IT + L_ord
            + delta*L_PIC + eta*L_cons

L_ord (CoralOrdinalLoss) replaces L_RMSE when use_ordinal_head=True.
L_faith has been removed — the ablation study showed it consistently harms accuracy.
"""

from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .pcol import PCOLLoss
from .scolw import SCOLwLoss
from .image_text import ImageTextOrdinalLoss
from .concept_losses import PerImageConceptLoss, GradeAgreementLoss
from .ordinal import CoralOrdinalLoss


class HybridContrastiveOrdinalLoss(nn.Module):

    def __init__(
        self,
        alpha: float = 1.0,
        beta: float = 1.0,
        gamma: float = 0.0,
        temperature: float = 0.07,
        use_image_text: bool = False,
        lambda_ord_it: float = 1.0,
        # Concept spine losses
        delta: float = 0.0,
        eta: float = 0.0,
        use_concept_spine: bool = False,
        concept_grade_mask: Optional[Dict[int, List[int]]] = None,
        n_concepts: int = 9,
        n_classes: int = 5,
        # Ordinal distribution head
        use_ordinal_head: bool = False,
        ordinal_class_counts=None,   # list of per-class counts for balanced CORAL
        # Grade-Coherent Tile Loss
        lambda_tile: float = 0.0,
    ):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta
        self.eta = eta
        self.use_image_text = use_image_text
        self.use_concept_spine = use_concept_spine
        self.use_ordinal_head = use_ordinal_head
        self.lambda_tile = lambda_tile

        self.pcol = PCOLLoss(temperature=temperature)
        self.scolw = SCOLwLoss(temperature=temperature)
        self.image_text = ImageTextOrdinalLoss(
            temperature=temperature,
            lambda_ord=lambda_ord_it,
        )

        if use_ordinal_head:
            self.ordinal_loss = CoralOrdinalLoss(
                n_classes=n_classes,
                class_counts=ordinal_class_counts,
            )

        if use_concept_spine:
            if concept_grade_mask is None:
                raise ValueError("concept_grade_mask must be provided when use_concept_spine=True.")
            self.pic_loss = PerImageConceptLoss(concept_grade_mask, n_concepts)
            self.cons_loss = GradeAgreementLoss(n_classes)

    def forward(
        self,
        z_pcol: torch.Tensor,
        z_scolw: torch.Tensor,
        pred: torch.Tensor,
        labels: torch.Tensor,
        class_weights: torch.Tensor,
        z_it: Optional[torch.Tensor] = None,
        text_prototypes: Optional[torch.Tensor] = None,
        c: Optional[torch.Tensor] = None,              # (N, M) concept activations
        E: Optional[torch.Tensor] = None,              # (N,) ordinal evidence
        ordinal_probs: Optional[torch.Tensor] = None,  # (N, K-1) from OrdinalDistHead
        tile_preds: Optional[torch.Tensor] = None,     # (N, T) per-tile scalar predictions
    ) -> tuple[torch.Tensor, dict]:

        l_pcol = self.pcol(z_pcol, labels)
        l_scolw = self.scolw(z_scolw, labels, class_weights)

        # Regression loss — CORAL ordinal or standard RMSE.
        # SCOLw already provides class-balanced gradients to the backbone via
        # class_weights; RMSE deliberately stays unweighted so the regression
        # head gets low-variance, grade-0-dominated gradient for stable convergence.
        if self.use_ordinal_head and ordinal_probs is not None:
            l_reg = self.ordinal_loss(ordinal_probs, labels)
        else:
            l_reg = torch.sqrt(F.mse_loss(pred, labels.float()) + 1e-8)

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

        l_pic = torch.tensor(0.0, device=pred.device)
        l_cons = torch.tensor(0.0, device=pred.device)
        if self.use_concept_spine and self.delta + self.eta > 0.0:
            if c is None or E is None:
                raise ValueError("c and E must be provided when use_concept_spine=True.")
            if self.delta > 0.0:
                l_pic = self.pic_loss(c, labels)
            if self.eta > 0.0:
                l_cons = self.cons_loss(E, pred)

        # Grade-Coherent Tile Loss (GCTL) — MIL semantics for DR grading:
        #   grade-0 images: mean tile prediction → 0 (all tiles look normal)
        #   grade k>0 images: max tile prediction → k (worst tile sets the grade)
        # Bypasses AttentionPool, giving direct ordinal gradient to the backbone.
        l_tile = pred.new_tensor(0.0)
        if tile_preds is not None and self.lambda_tile > 0.0:
            y_f = labels.float()
            g0 = labels == 0
            gp = ~g0
            terms = []
            if g0.any():
                terms.append(F.mse_loss(tile_preds[g0].mean(dim=1), y_f[g0]))
            if gp.any():
                terms.append(F.mse_loss(tile_preds[gp].max(dim=1).values, y_f[gp]))
            if terms:
                l_tile = sum(terms) / len(terms)

        total = (
            self.alpha * l_pcol
            + self.beta * l_scolw
            + self.gamma * l_it
            + l_reg
            + self.lambda_tile * l_tile
            + self.delta * l_pic
            + self.eta * l_cons
        )

        reg_key = "loss_ord" if self.use_ordinal_head else "loss_rmse"
        return total, {
            "loss_total": total.item(),
            "loss_pcol": l_pcol.item(),
            "loss_scolw": l_scolw.item(),
            "loss_it": l_it.item(),
            reg_key: l_reg.item(),
            "loss_tile": l_tile.item(),
            "loss_pic": l_pic.item(),
            "loss_cons": l_cons.item(),
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