"""
Prototype-based Contrastive Ordinal Loss (PCOL) - Equation (1) in the paper.

  L_PCOL = - sum_{i in I}  log[
      exp( f_a^T · c_p  / τ )
      ─────────────────────────────────────────────────────────────
      sum_{c_n in N(i)}  exp( (f_a^T · c_n  +  r_{a,n}) / τ )
  ]

Notation:
  f_a     - L2-normalized embedding of anchor sample a
  c_p     - prototype of the positive class (mean of same-class embeddings in
            the current mini-batch, then L2-normalized)
  c_n     - prototype of a negative class
  N(i)    - set of all negative class prototypes (classes ≠ anchor class)
  r_{a,n} - ordinal distance: |y_a - y_n|  (Euclidean distance on label axis)
  τ       - temperature

The r_{a,n} term makes misalignment penalties proportional to ordinal distance,
preserving ordinal relationships in the latent space (Section 2.2).

Vectorized O(N·C) implementation; no Python loops over samples.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PCOLLoss(nn.Module):

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(
        self,
        embeddings: torch.Tensor,   # (N, D) L2-normalized
        labels: torch.Tensor,       # (N,)  integer class labels
    ) -> torch.Tensor:
        """
        Args:
            embeddings: unit-sphere projections from the PCOL head
            labels:     integer ordinal labels

        Returns:
            Scalar PCOL loss averaged over all anchors.
        """
        device = embeddings.device
        unique_classes, _ = torch.sort(torch.unique(labels))
        C = unique_classes.shape[0]

        if C < 2:
            # Need at least 2 classes in the batch for a contrastive signal
            return torch.tensor(0.0, device=device, requires_grad=True)

        # ── Build class prototypes ──────────────────────────────────────────
        # Prototype for class c = mean of class-c embeddings, then L2-normed.
        # Avoid boolean fancy-indexing (embeddings[mask]) which is unstable on
        # MPS; use float-weighted summation instead.
        protos = []              # will be (C, D)
        proto_labels = []        # (C,) - label for each prototype row
        for c in unique_classes:
            # (N, 1) float mask — avoids boolean indexing on MPS
            mask_f = (labels == c).float().unsqueeze(1)
            n_c = mask_f.sum().clamp(min=1)
            p = F.normalize((embeddings * mask_f).sum(0, keepdim=True) / n_c, dim=1)
            protos.append(p)
            proto_labels.append(c)

        protos = torch.cat(protos, dim=0)                  # (C, D)
        proto_labels = torch.stack(proto_labels).to(device) # (C,)

        # ── Similarity matrix: anchor embeddings vs prototypes ─────────────
        # (N, C) - cosine similarities (both sides L2-normed)
        sim = embeddings @ protos.t()

        # ── Ordinal distance matrix: |y_anchor - y_prototype| ──────────────
        # (N, C)
        ord_dist = (
            labels.float().unsqueeze(1) - proto_labels.float().unsqueeze(0)
        ).abs()

        # ── Positive mask: anchor i matches prototype j ─────────────────────
        # (N, C) bool
        is_pos = labels.unsqueeze(1) == proto_labels.unsqueeze(0)

        # ── Numerator: exp( sim[i, pos_j] / τ ) ─────────────────────────────
        # Each anchor has exactly one positive prototype.
        pos_sim = (sim * is_pos.float()).sum(dim=1)        # (N,)
        log_num = pos_sim / self.temperature               # (N,)

        # ── Denominator: sum over NEGATIVE prototypes ─────────────────────
        # (f_a · c_n + r_{a,n}) / τ  for negative entries
        neg_logits = (sim + ord_dist) / self.temperature   # (N, C)
        # Mask out positive prototype with a large negative value.
        # Use -1e9 instead of -inf: MPS logsumexp is unstable with -inf.
        neg_logits = neg_logits.masked_fill(is_pos, -1e9)
        log_denom = torch.logsumexp(neg_logits, dim=1)     # (N,) numerically stable

        # ── Per-anchor loss ──────────────────────────────────────────────────
        loss = -(log_num - log_denom)                      # (N,)

        return loss.mean()
