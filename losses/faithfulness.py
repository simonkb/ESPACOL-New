"""
losses/faithfulness.py — Perturbation-Consistency Faithfulness Loss (L_faith)
and soft occlusion utility for ESPAOCL.

L_faith = L_drop + L_dir + L_spec

L_drop  : ReLU(c'_k - (c_k - margin))
          After occluding concept k's heatmap region, c'_k should drop by at least `margin`.

L_dir   : |ΔE - ΔE_pred|  where ΔE = E - E'  (actual ordinal shift after occlusion)
          and ΔE_pred = w_k * c_k              (concept's predicted contribution)
          Ensures the concept's weight matches its actual ordinal influence.

L_spec  : mean ReLU(|c'_j - c_j| - tau) for j ≠ k
          Other concepts should be preserved after occluding only concept k's region.

soft_occlude
------------
Replaces the high-attention region of x (guided by heatmap H_k) with a Gaussian-blurred
version.  x_occ = mask * blur(x) + (1 - mask) * x
The mask is soft (sigmoid-thresholded) to allow differentiability if needed,
but during training we use hard threshold + detach to keep the heatmap as a fixed input.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
# Soft occlusion
# ─────────────────────────────────────────────────────────────────────────────

def _gaussian_kernel(kernel_size: int, sigma: float, device: torch.device) -> torch.Tensor:
    """Create a 2-D Gaussian kernel as a (1, 1, k, k) tensor."""
    coords = torch.arange(kernel_size, dtype=torch.float32, device=device)
    coords -= kernel_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    kernel = g.outer(g)                                  # (k, k)
    return kernel.unsqueeze(0).unsqueeze(0)              # (1, 1, k, k)


def soft_occlude(
    x: torch.Tensor,           # (N, 3, H, W)  float, [0, 1] range
    heatmap: torch.Tensor,     # (N, H, W)     float [0, 1], detached
    sigma: float = 7.0,
    kernel_size: int = 21,
    threshold: float = 0.5,
) -> torch.Tensor:
    """
    Replace pixels where heatmap > threshold with a Gaussian-blurred version.

    Returns occluded x with the same shape and dtype.
    heatmap must already be detached (it is treated as a fixed mask, not differentiable).

    A hard mask is used deliberately: the continuous variant blends only
    proportionally to the CAM value, which perturbs the image too weakly to move
    c'_k, leaving L_drop pinned at its constant floor.
    """
    N, C, H, W = x.shape
    # Build per-channel blur kernel
    kernel = _gaussian_kernel(kernel_size, sigma, x.device)
    kernel = kernel.expand(C, 1, kernel_size, kernel_size)   # (C, 1, k, k)
    pad = kernel_size // 2

    blurred = F.conv2d(x, kernel, padding=pad, groups=C)     # (N, C, H, W)

    mask = (heatmap > threshold).float()                      # (N, H, W) hard mask
    mask = mask.unsqueeze(1)                                  # (N, 1, H, W)

    return mask * blurred + (1.0 - mask) * x


# ─────────────────────────────────────────────────────────────────────────────
# Faithfulness loss
# ─────────────────────────────────────────────────────────────────────────────

class FaithfulnessLoss(nn.Module):
    """
    L_faith = L_drop + L_dir + L_spec

    Parameters
    ----------
    margin : minimum drop in c_k expected after occlusion (L_drop)
    tau    : tolerance for change in other concepts (L_spec)
    """

    def __init__(self, margin: float = 0.1, tau: float = 0.05):
        super().__init__()
        self.margin = margin
        self.tau = tau

    def forward(
        self,
        c: torch.Tensor,          # (N, M)  original concept activations (detach before passing)
        c_prime: torch.Tensor,    # (N, M)  concept activations after occlusion (differentiable)
        E: torch.Tensor,          # (N,)    original ordinal evidence (detach before passing)
        E_prime: torch.Tensor,    # (N,)    ordinal evidence after occlusion (differentiable)
        w: torch.Tensor,          # (M,)    concept weights (detach before passing)
        top_k_idx: torch.Tensor,  # (N,)    index of top concept per image
    ) -> tuple[torch.Tensor, dict]:
        N, M = c.shape
        batch_idx = torch.arange(N, device=c.device)

        # --- L_drop: after occluding concept k, its activation should drop ---
        c_k       = c[batch_idx, top_k_idx]              # (N,) detached
        c_prime_k = c_prime[batch_idx, top_k_idx]        # (N,) differentiable
        L_drop = F.relu(c_prime_k - (c_k - self.margin)).mean()

        # --- L_dir: ordinal shift should match concept's predicted contribution ---
        delta_E      = E - E_prime                        # (N,) diff through E_prime
        delta_E_pred = (w[top_k_idx] * c_k).detach()     # (N,) fixed target
        L_dir = torch.abs(delta_E - delta_E_pred).mean()

        # --- L_spec: other concepts should not change much ---
        one_hot = F.one_hot(top_k_idx, num_classes=M).bool()  # (N, M)
        c_diff = torch.abs(c_prime - c)                        # (N, M)
        # Mask out the top concept; average over remaining concepts
        c_diff_others = c_diff.masked_fill(one_hot, 0.0)       # zero out top concept
        n_others = M - 1
        L_spec = F.relu(c_diff_others - self.tau).sum(dim=1).mean() / n_others

        L_faith = L_drop + L_dir + L_spec
        components = {
            "loss_drop": L_drop.item(),
            "loss_dir":  L_dir.item(),
            "loss_spec": L_spec.item(),
            "loss_faith": L_faith.item(),
        }
        return L_faith, components
