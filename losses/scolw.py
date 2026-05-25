"""
Weighted Supervised Contrastive Ordinal Loss (SCOLw) - Equation (2) in the paper.

  L_SCOLw = sum_{i in B}  ( -w_i / |P(i)| )  *  sum_{p in P(i)}  log[
      exp( f_a^T · f_p / τ )
      ─────────────────────────────────────────────────────────────────
      sum_{n in N(i)}  exp( (f_a^T · f_n  +  r_{a,n}) / τ )
  ]

Notation:
  f_a, f_p, f_n - L2-normalized embeddings of anchor, positive, negative samples
  P(i)          - set of positive samples (same class as anchor, different index)
  N(i)          - set of negative samples (different class from anchor)
  w_i           - per-sample class weight = N_total / (n_classes * n_class_i)
                  (inverse-frequency weighting; dynamically computed from batch
                  labels or pre-computed from the full training set)
  r_{a,n}       - ordinal distance |y_a - y_n|
  τ             - temperature

Vectorized O(N²) implementation; no Python loops over samples.
"""

import torch
import torch.nn as nn


class SCOLwLoss(nn.Module):

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(
        self,
        embeddings: torch.Tensor,       # (N, D) L2-normalized
        labels: torch.Tensor,           # (N,)  integer class labels
        class_weights: torch.Tensor,    # (n_classes,) pre-computed per-class weights
                                        # indexed by class integer value
    ) -> torch.Tensor:
        """
        Args:
            embeddings:    unit-sphere projections from the SCOLw head
            labels:        integer ordinal labels in [0, n_classes-1]
            class_weights: 1-D tensor of length n_classes; w[c] = weight for class c.
                           Typically N_total / (n_classes * n_c) so the mean weight = 1.

        Returns:
            Scalar SCOLw loss.
        """
        device = embeddings.device
        N = embeddings.shape[0]

        # Per-sample weights  w_i  drawn from the class-weight table.
        w = class_weights[labels]                          # (N,)

        # ── Pairwise similarity matrix ───────────────────────────────────────
        sim = embeddings @ embeddings.t()                  # (N, N)

        # ── Ordinal distance matrix  r_{a,n} = |y_a - y_n| ─────────────────
        ord_dist = (
            labels.float().unsqueeze(1) - labels.float().unsqueeze(0)
        ).abs()                                            # (N, N)

        # ── Boolean masks ───────────────────────────────────────────────────
        same_cls = labels.unsqueeze(1) == labels.unsqueeze(0)  # (N, N)
        self_mask = torch.eye(N, dtype=torch.bool, device=device)
        is_pos = same_cls & ~self_mask                         # (N, N)
        is_neg = ~same_cls                                     # (N, N)

        # ── Denominator: logsumexp over negative pairs ───────────────────────
        # neg_logit[i,j] = (sim[i,j] + r[i,j]) / τ  for j in N(i)
        neg_logits = (sim + ord_dist) / self.temperature      # (N, N)
        # Use -1e9 instead of -inf: MPS logsumexp is unstable with -inf.
        neg_logits = neg_logits.masked_fill(~is_neg, -1e9)
        log_denom = torch.logsumexp(neg_logits, dim=1)        # (N,)

        # ── Log-probability for each pair (i, p) ────────────────────────────
        # log p(i,p) = sim[i,p]/τ - log_denom[i]
        log_prob = sim / self.temperature - log_denom.unsqueeze(1)  # (N, N)

        # Zero out non-positive entries (we only sum over P(i))
        log_prob_pos = log_prob * is_pos.float()                # (N, N)

        # Number of positives per anchor (avoid div-by-zero)
        n_pos = is_pos.float().sum(dim=1)                       # (N,)

        # Per-anchor loss:  (-w_i / |P(i)|) * sum_{p in P(i)} log_prob
        per_anchor = -w * (log_prob_pos.sum(dim=1) / n_pos.clamp(min=1))  # (N,)

        # Mask anchors with no positives using multiplication instead of
        # boolean indexing (MPS has unstable behavior with fancy indexing).
        has_pos = (n_pos > 0).float()                           # (N,)
        n_valid = has_pos.sum().clamp(min=1)
        return (per_anchor * has_pos).sum() / n_valid
