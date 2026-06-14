"""
Text-Anchored Metric Ordinality (TAMO) Loss.

Novel contribution: constrains the pairwise distance geometry of image-class
prototypes to match the pairwise distance geometry of text prototypes from a
pretrained medical VLM (BiomedCLIP).  This enforces the clinically non-uniform
severity spacing encoded in the text space onto the visual embedding manifold.

Key insight:
  BiomedCLIP's text encoder has already learned that the semantic gap between
  "no DR" and "severe DR" is much larger than "no DR" and "mild DR".  TAMO
  transfers this non-uniform inter-class geometry as a free supervision signal
  for the image branch.

──────────────────────────────────────────────────────────────────────────────
PMD v2 — Soft Prototype InfoNCE (replaces v1 Huber on normalized distances)

v1 failure mode:
  Huber(D_img_norm - D_txt_norm) collapses to 0 within ~50 epochs because
  BiomedCLIP's shared embedding space already satisfies rough absolute distance
  matching from pretraining.  Once |D_img_norm - D_txt_norm| < huber_delta the
  loss goes to zero and stops providing gradient.

v2 fix:
  Replace absolute matching with distributional matching via KL divergence.
  For each anchor class j, form a distribution over all other classes by
  softmax-normalizing the negative distances:

    p_txt[j, k] = softmax(-D_txt[j, :] / τ)[k]   (teacher, fixed)
    p_img[j, k] = softmax(-D_img[j, :] / τ)[k]   (student, differentiable)

    L_PMD = mean_j  KL(p_txt_j  ||  p_img_j)

  KL divergence on softmax distributions:
  - Is scale-invariant (global scale of distances cancels in softmax)
  - Is sensitive to ORDERING, not absolute values
  - Stays non-zero whenever the distribution shape differs, even when
    absolute distances are already similar
  - Has temperature τ that controls how sharply "closer" classes are
    distinguished from "farther" ones

──────────────────────────────────────────────────────────────────────────────
ORC — Ordinal Rank-Consistency (unchanged from v1)

For every triplet (j,k,l) where D_txt[j,k] < D_txt[j,l]:
  violation = max(0, D_img_norm[j,k] - D_img_norm[j,l] + margin_jkl)
  margin_jkl = D_txt_norm[j,l] - D_txt_norm[j,k]

L_TAMO = L_PMD + lambda_orc * L_ORC
──────────────────────────────────────────────────────────────────────────────
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class TAMOLoss(nn.Module):
    """
    Text-Anchored Metric Ordinality (TAMO) loss.

    Args:
        lambda_orc:     weight for ORC term relative to PMD.
        temperature_pmd: softmax temperature for PMD-KL.  Lower = sharper
                         distribution = more sensitive to fine-grained ordering.
                         0.1 works well when D values are in the [0, 0.5] range.
        min_classes:    skip if fewer distinct classes appear in the batch.
    """

    def __init__(
        self,
        lambda_orc: float = 0.5,
        temperature_pmd: float = 0.1,
        min_classes: int = 3,
        # kept for API compatibility — no longer used in PMD
        huber_delta: float = 0.1,
    ):
        super().__init__()
        self.lambda_orc = lambda_orc
        self.temperature_pmd = temperature_pmd
        self.min_classes = min_classes

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _compute_prototypes(
        self,
        z: torch.Tensor,
        labels: torch.Tensor,
        present_classes: list[int],
    ) -> torch.Tensor:
        """
        Per-class mean embedding re-normalized to the unit sphere.
        Returns (K, d) where K = len(present_classes).
        """
        d = z.shape[1]
        protos = torch.stack([
            z[labels == k].mean(0) if (labels == k).any() else z.new_zeros(d)
            for k in present_classes
        ])
        return F.normalize(protos, dim=1)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        z_tamo: torch.Tensor,           # (N, d), L2-normalized
        labels: torch.Tensor,           # (N,) int64
        text_dist_matrix: torch.Tensor, # (C, C) precomputed text cosine distances
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns (L_PMD, L_ORC) as separate scalar tensors.
        Returns (0, 0) when fewer than min_classes appear in the batch.
        """
        device = z_tamo.device
        zero = z_tamo.new_zeros(())

        present_classes = sorted(labels.unique().tolist())
        if len(present_classes) < self.min_classes:
            return zero, zero

        K = len(present_classes)

        # ── Image-space prototype distance matrix ─────────────────────────
        img_protos = self._compute_prototypes(z_tamo, labels, present_classes)
        D_img = (1.0 - img_protos @ img_protos.t()).clamp(min=0.0)   # (K, K)

        # ── Text distance sub-matrix for present classes ───────────────────
        idx = torch.tensor(present_classes, device=device, dtype=torch.long)
        D_txt = text_dist_matrix[idx][:, idx].clamp(min=0.0).detach()  # (K, K)

        diag = torch.eye(K, dtype=torch.bool, device=device)
        off_diag = ~diag

        # ── PMD v2: KL divergence on off-diagonal softmax distributions ────
        #
        # Explicitly extract the K×(K-1) off-diagonal distances instead of
        # masking the diagonal with a large constant.  Masked-fill with 1e4
        # produces -1e5 logits that become -inf in float16, leading to
        # 0 * log(0/-inf) = 0 * NaN = NaN in kl_div (even though kl_div
        # backward is actually fine since it computes -target=0 for those
        # positions).  Explicit extraction avoids the issue entirely.
        #
        # off_diag is a (K, K) bool with True on all K*(K-1) off-diagonal
        # positions; indexing produces those elements in row-major order.
        tau = self.temperature_pmd

        D_img_od = D_img[off_diag].view(K, K - 1)           # (K, K-1) differentiable
        D_txt_od = D_txt[off_diag].view(K, K - 1).detach()  # (K, K-1) teacher, fixed

        # Teacher: text-derived distribution over the K-1 other classes
        txt_probs = F.softmax(-D_txt_od / tau, dim=1)        # (K, K-1)

        # Student: image-derived log-distribution
        img_log_probs = F.log_softmax(-D_img_od / tau, dim=1)  # (K, K-1)

        # KL(p_txt || p_img) — averaged over K anchor classes
        l_pmd = F.kl_div(
            img_log_probs,          # log Q (approximation)
            txt_probs,              # P    (target)
            reduction="batchmean",  # sum over K*(K-1), divide by K
            log_target=False,
        )

        # ── ORC: triplet rank-consistency with adaptive text margins ──────
        # Scale both matrices by their mean off-diagonal so margin units match.
        img_scale = D_img[off_diag].mean().clamp(min=1e-6)
        txt_scale = D_txt[off_diag].mean().clamp(min=1e-6)

        D_img_n = D_img / img_scale
        D_txt_n = D_txt / txt_scale   # still detached

        # Vectorized over all K^3 triplets
        D_img_jk = D_img_n.unsqueeze(2).expand(K, K, K)   # D_img[j, k, *]
        D_img_jl = D_img_n.unsqueeze(1).expand(K, K, K)   # D_img[j, *, l]
        D_txt_jk = D_txt_n.unsqueeze(2).expand(K, K, K)
        D_txt_jl = D_txt_n.unsqueeze(1).expand(K, K, K)

        margin = (D_txt_jl - D_txt_jk).clamp(min=0.0)
        orc_violations = F.relu(D_img_jk - D_img_jl + margin)

        i_range = torch.arange(K, device=device)
        distinct = (
            (i_range.view(K, 1, 1) != i_range.view(1, K, 1)) &
            (i_range.view(K, 1, 1) != i_range.view(1, 1, K)) &
            (i_range.view(1, K, 1) != i_range.view(1, 1, K))
        )
        valid = distinct & (D_txt_jk < D_txt_jl)

        l_orc = orc_violations[valid].mean() if valid.any() else zero

        return l_pmd, l_orc
