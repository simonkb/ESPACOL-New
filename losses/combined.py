from __future__ import annotations

"""
Combined hybrid loss with optional extensions.

Baseline:
    L_total = alpha * L_PCOL + beta * L_SCOLw + L_RMSE

ESPAOCL (image-text):
    L_total = alpha * L_PCOL + beta * L_SCOLw + gamma_it * L_IT + L_RMSE

TAMO:
    L_total = alpha * L_PCOL + beta * L_SCOLw + gamma_it * L_IT
              + gamma_tamo * (L_PMD + lambda_orc * L_ORC) + L_RMSE

CORAL (use_coral=True):
    Replaces L_RMSE with CORAL binary cross-entropy over K-1 rank thresholds.

Two-Stage Hierarchical (use_two_stage=True):
    L_reg = L_screen + grading_lambda * L_grade
      L_screen : BCE(screen_logit, binary_label)         — all samples
      L_grade  : CORAL(grade_logits[dr+], grade-1)       — DR-positive only

    The binary screening loss forces early separation of grade-0 from grade-1+.
    The CORAL grading loss then focuses on ordinal discrimination within DR+,
    eliminating the "hedge toward grade-0" attractor that breaks RMSE regression
    when 73.5% of training data is grade-0.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .pcol import PCOLLoss
from .scolw import SCOLwLoss
from .image_text import ImageTextOrdinalLoss
from .tamo import TAMOLoss
from .coral import CoralLoss


class HybridContrastiveOrdinalLoss(nn.Module):

    def __init__(
        self,
        alpha: float = 1.0,
        beta: float = 1.0,
        gamma: float = 0.0,
        temperature: float = 0.07,
        use_image_text: bool = False,
        lambda_ord_it: float = 1.0,
        # Two-stage: screening (binary BCE) + grading (CORAL on DR+ only)
        use_two_stage: bool = False,
        grading_lambda: float = 2.0,
        # CORAL: replace RMSE with rank-consistent binary BCE over all K classes
        use_coral: bool = False,
        # TAMO parameters
        use_tamo: bool = False,
        gamma_tamo: float = 0.1,
        lambda_orc: float = 1.0,
        huber_delta: float = 0.1,
        temperature_pmd: float = 0.1,
        # Label smoothing for RMSE regression only (bypassed for CORAL / two-stage)
        label_smooth_sigma: float = 0.0,
        n_classes: int = 5,
    ):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.use_two_stage = use_two_stage
        self.grading_lambda = grading_lambda
        self.use_coral = use_coral
        self.label_smooth_sigma = label_smooth_sigma
        self.n_classes = n_classes
        self.gamma = gamma
        self.use_image_text = use_image_text
        self.use_tamo = use_tamo
        self.gamma_tamo = gamma_tamo

        self.pcol = PCOLLoss(temperature=temperature)
        self.scolw = SCOLwLoss(temperature=temperature)
        self.coral = CoralLoss()

        self.image_text = ImageTextOrdinalLoss(
            temperature=temperature,
            lambda_ord=lambda_ord_it,
        )

        self.tamo = TAMOLoss(
            lambda_orc=lambda_orc,
            huber_delta=huber_delta,
            temperature_pmd=temperature_pmd,
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
        pred_grading: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict]:

        l_pcol = self.pcol(z_pcol, labels)
        l_scolw = self.scolw(z_scolw, labels, class_weights)

        # ── Ordinal prediction loss ───────────────────────────────────────────
        if self.use_two_stage and pred_grading is not None:
            # Stage 1: binary screening — grade-0 (negative) vs grade-1+ (positive)
            # Use dataset-proportion pos_weight to compensate for stratified batch
            # oversampling of minority classes (batch has ~1:4 grade-0:DR+ ratio
            # vs 73.5:26.5 in the full dataset; pos_weight restores the correct
            # relative penalty without changing which samples are seen).
            binary_labels = (labels > 0).float()                 # (N,)
            pos_weight = torch.tensor(
                [self.n_classes / (self.n_classes - 1)],         # ≈1.25 for K=5
                device=pred.device,
            )
            l_screen = F.binary_cross_entropy_with_logits(
                pred, binary_labels, pos_weight=pos_weight
            )

            # Stage 2: CORAL ordinal grading — only on DR-positive samples.
            # Labels are shifted: grade k → k-1 so grades 1..K-1 map to 0..K-2.
            dr_mask = labels > 0
            if dr_mask.sum() >= 2:
                grading_labels = labels[dr_mask] - 1             # shift to 0-based
                l_grade = self.coral(pred_grading[dr_mask], grading_labels)
            else:
                l_grade = torch.tensor(0.0, device=pred.device)

            l_rmse = l_screen + self.grading_lambda * l_grade

        elif self.use_coral:
            # Direct K-class ordinal classification via shared-weight rank BCE.
            l_rmse = self.coral(pred, labels)

        else:
            # Standard RMSE with optional ordinal label smoothing.
            targets = labels.float()
            if self.label_smooth_sigma > 0.0 and self.training:
                noise = torch.randn_like(targets) * self.label_smooth_sigma
                targets = (targets + noise).clamp(0.0, float(self.n_classes - 1))
            l_rmse = torch.sqrt(F.mse_loss(pred, targets) + 1e-8)

        # ── Image-text ordinal loss ───────────────────────────────────────────
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

        # ── TAMO loss ─────────────────────────────────────────────────────────
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
    """
    counts = torch.zeros(n_classes)
    for y in labels:
        counts[int(y)] += 1
    counts = counts.clamp(min=1)
    n_total = counts.sum()
    weights = n_total / (n_classes * counts)
    return weights.to(device)
