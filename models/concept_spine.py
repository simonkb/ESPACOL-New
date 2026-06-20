"""
models/concept_spine.py — Concept Activation Spine for ESPAOCL.

Projects backbone features (N, 1280) into M clinical concept activations using
BiomedCLIP concept phrase text embeddings, then computes ordinal evidence E and
soft grade estimate y_soft via learned ordinal thresholds.

Forward outputs
---------------
c       : (N, M)  cosine similarity between image projection and each concept text vector
w       : (M,)    concept importance weights (softplus → w >= 0 for severity-increasing concepts)
E       : (N,)    ordinal evidence = c @ w
y_soft  : (N,)    soft ordinal grade = Σ σ((E - θ_k) / τ) summed over K-1 thresholds
z_c     : (N, 128) L2-normalized image concept embedding (needed by faithfulness loop for CAM)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional


class ConceptSpine(nn.Module):
    def __init__(
        self,
        in_dim: int = 1280,
        proj_dim: int = 128,
        n_concepts: int = 9,
        n_classes: int = 5,
        absence_indices: Optional[List[int]] = None,
        threshold_temp: float = 0.5,
    ):
        """
        Parameters
        ----------
        in_dim          : backbone feature dimension (1280 for EfficientNet-V2S)
        proj_dim        : concept embedding dimension (must match text encoder output)
        n_concepts      : M — number of clinical concepts
        n_classes       : K — number of ordinal grades
        absence_indices : concept indices that are "absence-of-disease" (w <= 0 for these).
                          None means all concepts are severity-increasing (w >= 0).
        threshold_temp  : τ for soft ordinal threshold sigmoid
        """
        super().__init__()

        # Concept projection head: same MLP structure as MLPProjectionHead
        self.concept_proj = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.BatchNorm1d(in_dim),
            nn.ReLU(inplace=True),
            nn.Linear(in_dim, proj_dim),
        )

        self.n_concepts = n_concepts
        self.n_classes = n_classes
        self.threshold_temp = threshold_temp
        self.absence_indices = absence_indices or []

        # Concept importance weights — raw params, constrained via softplus/negation
        self.raw_w = nn.Parameter(torch.zeros(n_concepts))

        # Ordinal thresholds θ_1 < θ_2 < ... < θ_{K-1}
        # Initialized so thresholds sit at 0.5, 1.5, ..., K-1.5 (midpoints between grades)
        self.raw_thresholds = nn.Parameter(
            torch.arange(n_classes - 1).float() + 0.5
        )

    def _weights(self) -> torch.Tensor:
        """Return constrained concept weights (M,).
        Severity-increasing concepts: w = softplus(raw_w) >= 0
        Absence-of-disease concepts: w = -softplus(raw_w) <= 0
        """
        w = F.softplus(self.raw_w)
        if self.absence_indices:
            idx = torch.tensor(self.absence_indices, device=w.device)
            w = w.clone()
            w[idx] = -F.softplus(self.raw_w[idx])
        return w

    def _sorted_thresholds(self) -> torch.Tensor:
        """Return sorted thresholds so ordinal ordering is enforced."""
        return torch.sort(self.raw_thresholds)[0]

    def forward(
        self,
        features: torch.Tensor,           # (N, in_dim)
        concept_text_emb: torch.Tensor,   # (M, proj_dim)  from text encoder, L2-normalised
    ):
        """
        Returns
        -------
        c       : (N, M)
        w       : (M,)
        E       : (N,)
        y_soft  : (N,)
        z_c     : (N, proj_dim)
        """
        z_c = F.normalize(self.concept_proj(features), dim=-1)  # (N, 128)
        c = z_c @ concept_text_emb.T                             # (N, M)

        w = self._weights()                                       # (M,)
        E = c @ w                                                 # (N,)

        thresholds = self._sorted_thresholds()                    # (K-1,)
        # Soft ordinal grade: sum of sigmoids over thresholds → range [0, K-1]
        y_soft = torch.sigmoid(
            (E.unsqueeze(1) - thresholds.unsqueeze(0)) / self.threshold_temp
        ).sum(dim=1)                                              # (N,)

        return c, w, E, y_soft, z_c

    def predict_grade(self, E: torch.Tensor) -> torch.Tensor:
        """Hard grade prediction from ordinal evidence. Returns (N,) long tensor."""
        thresholds = self._sorted_thresholds().detach()
        return (E.unsqueeze(1) > thresholds.unsqueeze(0)).sum(dim=1).long()
