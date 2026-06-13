"""
Text-Anchored Metric Ordinality (TAMO) Loss.

Novel contribution: constrains the pairwise distance geometry of image-class
prototypes to match the pairwise distance geometry of text prototypes from a
pretrained medical VLM (BiomedCLIP).  This enforces the clinically non-uniform
severity spacing encoded in the text space onto the visual embedding manifold.

Key insight:
  BiomedCLIP's text encoder has already learned that the semantic gap between
  "no DR" and "severe DR" is much larger than "no DR" and "mild DR".  TAMO
  transfers this non-uniform inter-class geometry — which took large-scale
  pretraining to acquire — as a free supervision signal for the image branch,
  without requiring any additional labels.

Contrast with prior work:
  - OrdinalCLIP (NeurIPS 2022): sample-level classification logits, not
    prototype distance matrices.
  - L2RCLIP (NeurIPS 2023): cross-entropy bound reformulation, not scale-
    normalized Frobenius distillation.
  - CLIP-DR (MICCAI 2024): only sequential adjacent-grade ordering, not full
    C×C pairwise geometry.
  - CLOC (CVPR 2025): learned margins — TAMO margins are text-derived and
    data-free.
  - Current IT loss (this repo): sample-level pull, not prototype-level
    geometry constraint.

Two terms:
  L_TAMO = L_PMD + lambda_orc * L_ORC

─────────────────────────────────────────────────────────────────────────────
Term 1 — Prototype Metric Distillation (PMD)

For each batch, compute image-class prototypes c_k = mean(z_tamo | label==k),
then build pairwise cosine distance matrices:

  D_img[j,k] = 1 - c_j · c_k       (image prototype cosine distance)
  D_txt[j,k] = 1 - t_j · t_k       (text prototype cosine distance, fixed)

Scale-normalize both by their respective mean off-diagonal distances:

  D_img_norm = D_img / mean(D_img[off-diag])
  D_txt_norm = D_txt / mean(D_txt[off-diag])

The normalization removes global scale ambiguity and prevents the trivial
collapse solution (all embeddings identical → all D_img → 0 satisfies PMD
without normalization).

Minimize Huber loss between normalized matrices:
  L_PMD = mean over j≠k of huber_delta(D_img_norm[j,k] - D_txt_norm[j,k])

Huber (delta=0.1) is more robust than MSE to noisy batch prototypes from
small per-class sample counts.

─────────────────────────────────────────────────────────────────────────────
Term 2 — Ordinal Rank-Consistency (ORC)

For every ordered triplet (j, k, l) where D_txt[j,k] < D_txt[j,l]
(i.e., text says class k is semantically closer to j than class l is):

  margin_jkl = D_txt_norm[j,l] - D_txt_norm[j,k]   (adaptive text margin)
  violation  = max(0, D_img_norm[j,k] - D_img_norm[j,l] + margin_jkl)

  L_ORC = mean over all valid triplets of violation

The adaptive margin from text distances means harder ordinal constraints
(large semantic gaps) produce proportionally larger penalties, while nearby
grades (e.g. grade 1 vs grade 2) are treated more leniently.

For C=5 DR grades there are at most 60 directed triplets; the valid subset
(where D_txt[j,k] < D_txt[j,l]) is ~30 — negligible compute.
─────────────────────────────────────────────────────────────────────────────
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class TAMOLoss(nn.Module):
    """
    Text-Anchored Metric Ordinality (TAMO) loss.

    Args:
        lambda_orc: weight for ORC term relative to PMD (default 1.0 → equal)
        huber_delta: Huber loss threshold in scale-normalized distance units.
                     0.1 is robust to noisy batch prototypes from small classes.
        min_classes: skip both terms if fewer distinct classes appear in the
                     batch.  3 is the minimum for meaningful triplet geometry.
    """

    def __init__(
        self,
        lambda_orc: float = 1.0,
        huber_delta: float = 0.1,
        min_classes: int = 3,
    ):
        super().__init__()
        self.lambda_orc = lambda_orc
        self.huber_delta = huber_delta
        self.min_classes = min_classes

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _compute_prototypes(
        self,
        z: torch.Tensor,             # (N, d), L2-normalized
        labels: torch.Tensor,        # (N,) int64
        present_classes: list[int],
    ) -> torch.Tensor:
        """
        Per-class mean embedding for each class present in the batch.

        Returns shape (K, d) where K = len(present_classes).
        Each prototype is re-normalized to the unit sphere so cosine
        distance between prototypes is well-defined.
        """
        d = z.shape[1]
        protos = torch.stack([
            z[labels == k].mean(0) if (labels == k).any() else z.new_zeros(d)
            for k in present_classes
        ])  # (K, d)
        # Re-normalize: mean of unit vectors is not unit length itself.
        return F.normalize(protos, dim=1)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        z_tamo: torch.Tensor,           # (N, d), L2-normalized TAMO embeddings
        labels: torch.Tensor,           # (N,) int64 in [0, C-1]
        text_dist_matrix: torch.Tensor, # (C, C) precomputed text cosine distances
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns (L_PMD, L_ORC) as separate scalar tensors.
        The caller (HybridContrastiveOrdinalLoss) weights and sums them.

        Both terms return a zero tensor (no gradient) when the batch has
        fewer than min_classes distinct labels, avoiding NaN from empty
        prototype computations.
        """
        device = z_tamo.device
        zero = z_tamo.new_zeros(())

        present_classes = sorted(labels.unique().tolist())
        if len(present_classes) < self.min_classes:
            return zero, zero

        K = len(present_classes)

        # ── Step 1: image-space prototype distance matrix ─────────────────
        img_protos = self._compute_prototypes(z_tamo, labels, present_classes)
        # Cosine similarity -> cosine distance (both in [0, 2] theoretically,
        # but practically in [0, 1] for unit embeddings from positive space).
        sim_img = img_protos @ img_protos.t()          # (K, K)
        D_img = (1.0 - sim_img).clamp(min=0.0)        # (K, K)

        # ── Step 2: extract matching text sub-matrix ──────────────────────
        idx = torch.tensor(present_classes, device=device, dtype=torch.long)
        D_txt = text_dist_matrix[idx][:, idx].clamp(min=0.0)   # (K, K)

        # Off-diagonal mask (j != k)
        off_diag = ~torch.eye(K, dtype=torch.bool, device=device)

        # ── Step 3: scale normalization ───────────────────────────────────
        # Divide each matrix by its mean off-diagonal value so both are in
        # the same "unit" (mean distance = 1).  This prevents the collapse
        # solution and makes Huber delta=0.1 meaningful across datasets.
        img_scale = D_img[off_diag].mean().clamp(min=1e-6)
        txt_scale = D_txt[off_diag].mean().clamp(min=1e-6)

        D_img_norm = D_img / img_scale    # (K, K), gradients flow through D_img
        D_txt_norm = (D_txt / txt_scale).detach()   # (K, K), text is fixed supervision

        # ── PMD: scale-normalized Huber distillation ──────────────────────
        residuals = (D_img_norm - D_txt_norm)[off_diag]    # (K*(K-1),)
        abs_r = residuals.abs()
        l_pmd = torch.where(
            abs_r < self.huber_delta,
            0.5 * residuals ** 2,
            self.huber_delta * (abs_r - 0.5 * self.huber_delta),
        ).mean()

        # ── ORC: triplet rank-consistency with adaptive text margins ──────
        # Vectorized over all K^3 triplets (j, k, l) — for K=5 this is 125
        # elements, all computed in a single tensor op.
        #
        # D_img_jk[j, k, l] = D_img_norm[j, k]   (distance from j to k)
        # D_img_jl[j, k, l] = D_img_norm[j, l]   (distance from j to l)
        # valid[j, k, l] = True iff j≠k, j≠l, k≠l, and D_txt[j,k] < D_txt[j,l]
        D_img_jk = D_img_norm.unsqueeze(2).expand(K, K, K)   # D_img[j, k, *]
        D_img_jl = D_img_norm.unsqueeze(1).expand(K, K, K)   # D_img[j, *, l]

        D_txt_jk = D_txt_norm.unsqueeze(2).expand(K, K, K)   # D_txt[j, k, *]
        D_txt_jl = D_txt_norm.unsqueeze(1).expand(K, K, K)   # D_txt[j, *, l]

        # Adaptive margin = text distance gap (positive when k is closer to j)
        margin = (D_txt_jl - D_txt_jk).clamp(min=0.0)   # detached via D_txt_norm

        # Violation: image ordering should match text ordering up to margin
        orc_violations = F.relu(D_img_jk - D_img_jl + margin)   # (K, K, K)

        # Build valid-triplet mask (distinct indices + text ordering satisfied)
        i_range = torch.arange(K, device=device)
        j_idx = i_range.view(K, 1, 1)
        k_idx = i_range.view(1, K, 1)
        l_idx = i_range.view(1, 1, K)
        distinct = (j_idx != k_idx) & (j_idx != l_idx) & (k_idx != l_idx)
        txt_ordered = (D_txt_jk < D_txt_jl)   # D_txt is detached — no grad
        valid = distinct & txt_ordered

        if valid.any():
            l_orc = orc_violations[valid].mean()
        else:
            l_orc = zero

        return l_pmd, l_orc
