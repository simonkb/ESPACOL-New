"""
Per-Image Contrastive Concept loss (L_pic) - the primary proposed novelty.

  L_pic = - (1/N) * sum_i  log[
      sum_{c in G(y_i)}  exp( s(z_i, v_c) / tau )
      ──────────────────────────────────────────────────────────────
      sum_{c}            exp( (s(z_i, v_c) + lambda * |y_i - g(c)|) / tau )
  ]

Notation:
  z_i        L2-normalized image embedding in the shared concept space
             (the output of the existing image->concept alignment projection,
             NOT the raw 1280-d backbone feature; the gradient must flow back
             through that projection into the backbone for this to reshape the
             embedding).
  v_c        L2-normalized concept vector for concept c (frozen BioMedCLIP text
             embedding, optionally push-apart separated). Reuses the existing
             text projection, so no new text-side parameters.
  s(z, v)    cosine similarity = z . v   (both unit norm).
  G(y_i)     own-grade concept group: all concepts whose grade equals y_i.
  g(c)       grade of concept c.
  |y_i-g(c)| ordinal distance; the additive margin lambda*|y_i-g(c)| only
             touches the denominator, so wrong-grade concepts must clear a
             larger gap to win, and a 1-grade error is penalized less than a
             2-grade error.
  tau        temperature.

Why this and not the per-class alignment (L_IT): L_IT pulls the image toward
its class prototype on average, which a Moderate image can satisfy while still
sitting next to a neighbor-grade concept. L_pic demands per-image supremacy:
the correct grade's concepts must win on THIS image, which is what forces the
encoder to move the bimodal Moderate points out of their neighbors' territory.

Vectorized, O(N*C), no Python loops. MPS-safe (-1e4 fill, no boolean indexing).
"""

import torch
import torch.nn as nn


class PerImageConceptLoss(nn.Module):

    def __init__(self, temperature: float = 0.1, lambda_ord: float = 0.5):
        super().__init__()
        self.temperature = temperature
        self.lambda_ord = lambda_ord

    def forward(
        self,
        z: torch.Tensor,                 # (N, D) L2-normalized image embeddings
        concept_vecs: torch.Tensor,      # (C, D) L2-normalized concept vectors
        concept_grades: torch.Tensor,    # (C,)   integer grade of each concept
        labels: torch.Tensor,            # (N,)   integer grade of each image
    ) -> torch.Tensor:
        """
        Returns a scalar loss averaged over the images that have at least one
        own-grade concept in the bank (all of them, in practice).
        """
        device = z.device
        tau = self.temperature
        lam = self.lambda_ord

        concept_grades = concept_grades.to(device)
        labels = labels.to(device)

        # Cosine similarity (both sides already unit norm).
        s = z @ concept_vecs.t()                                   # (N, C)

        # Ordinal distance between each image's grade and each concept's grade.
        ord_dist = (
            labels.float().unsqueeze(1) - concept_grades.float().unsqueeze(0)
        ).abs()                                                    # (N, C)

        # Positives: concepts whose grade matches the image grade.
        is_own = labels.unsqueeze(1) == concept_grades.unsqueeze(0)  # (N, C) bool

        # Numerator: logsumexp over OWN-grade concepts only.
        # Mask non-own concepts with -1e4 (MPS-stable substitute for -inf).
        num_logits = (s / tau).masked_fill(~is_own, -1e4)          # (N, C)
        log_num = torch.logsumexp(num_logits, dim=1)               # (N,)

        # Denominator: logsumexp over ALL concepts, with the ordinal margin
        # added to every concept (zero for own-grade, grows with grade gap).
        den_logits = (s + lam * ord_dist) / tau                    # (N, C)
        log_den = torch.logsumexp(den_logits, dim=1)               # (N,)

        loss = -(log_num - log_den)                                # (N,)  >= 0

        # Guard: drop any image with no own-grade concept (mult-mask, not index).
        has_own = is_own.any(dim=1).float()                        # (N,)
        denom = has_own.sum().clamp(min=1)
        return (loss * has_own).sum() / denom

