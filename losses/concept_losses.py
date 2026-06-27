"""
losses/concept_losses.py — Per-Image Contrastive Concept Loss (L_PIC) and
Grade Agreement Loss (L_cons) for ESPAOCL.

L_PIC
-----
Multi-label binary cross-entropy between concept activations c (N, M) and
grade-specific binary concept targets. For each image with grade y_i:
  target[i, j] = 1  if concept j is in the cumulative concept set for grade y_i
  target[i, j] = 0  otherwise
Targets for grade 0 are all zeros (no pathological concepts expected).

L_cons
------
L1 distance between the concept-derived ordinal evidence E and the rounded
regression head prediction y_aux: L_cons = mean(|E - y_aux|)
Forces the concept spine's ordinal score to agree with the regression head.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List


def _build_concept_target(
    labels: torch.Tensor,             # (N,) integer grades
    concept_grade_mask: Dict[int, List[int]],
    n_concepts: int,
    device: torch.device,
) -> torch.Tensor:
    """Build binary concept target matrix (N, M) from per-grade concept index lists."""
    N = labels.shape[0]
    target = torch.zeros(N, n_concepts, dtype=torch.float32, device=device)
    for i, y in enumerate(labels.tolist()):
        indices = concept_grade_mask.get(int(y), [])
        if indices:
            target[i, indices] = 1.0
    return target


class PerImageConceptLoss(nn.Module):
    """
    L_PIC: Per-Image Contrastive Concept Loss.

    Parameters
    ----------
    concept_grade_mask : dict mapping grade → list of concept indices (cumulative)
    n_concepts         : M, total number of clinical concepts
    temperature        : τ for scaling logits before BCE (default 1.0 = no scaling)
    pos_weight         : optional per-concept positive weight tensor (M,) for class imbalance
    """

    def __init__(
        self,
        concept_grade_mask: Dict[int, List[int]],
        n_concepts: int,
        temperature: float = 1.0,
        pos_weight: torch.Tensor | None = None,
    ):
        super().__init__()
        self.concept_grade_mask = concept_grade_mask
        self.n_concepts = n_concepts
        self.temperature = temperature
        if pos_weight is not None:
            self.register_buffer("pos_weight", pos_weight)
        else:
            self.pos_weight = None

    def forward(
        self,
        c: torch.Tensor,        # (N, M) raw concept activations (cosine sims in [-1, 1])
        labels: torch.Tensor,   # (N,) integer grade labels
    ) -> torch.Tensor:
        target = _build_concept_target(
            labels, self.concept_grade_mask, self.n_concepts, c.device
        )
        # Soft targets: in-grade concepts get 1.0, others get 0.1
        # This prevents hard rejection of cross-grade concepts (e.g. Moderate
        # images having secondary activation of Severe concepts is penalised
        # less than zero, rather than treated as a fully wrong prediction).
        target_soft = torch.where(target > 0.5,
                                  torch.ones_like(target),
                                  torch.full_like(target, 0.1))

        return F.binary_cross_entropy_with_logits(
            c / self.temperature,
            target_soft,
            pos_weight=self.pos_weight,
        )


class GradeAgreementLoss(nn.Module):
    """
    L_cons = mean(|E - y_aux|)

    Pulls the concept spine's ordinal evidence E toward the regression head's
    rounded prediction y_aux. y_aux is detached (treated as target, not optimised here).

    Parameters
    ----------
    n_classes : K, used to clamp y_aux to [0, K-1]
    """

    def __init__(self, n_classes: int):
        super().__init__()
        self.n_classes = n_classes

    def forward(
        self,
        E: torch.Tensor,     # (N,) ordinal evidence from ConceptSpine
        pred: torch.Tensor,  # (N,) continuous regression output (detach internally)
    ) -> torch.Tensor:
        y_aux = pred.detach().round().clamp(0, self.n_classes - 1)
        return F.l1_loss(E, y_aux)
