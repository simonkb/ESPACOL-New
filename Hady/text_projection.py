"""
Recovered text projection ("Interpreter B") for gamma-aligned explainability.
============================================================================

Background
----------
The gamma image-text loss trains a per-image embedding ``z_it`` (produced by the
model's ``image_text_head``) to align with class *text prototypes*. Those
prototypes come from a BioMedCLIP text tower followed by a small projection head
(``models.text.TextProjectionHead``). The image side (``image_text_head``) lives
inside the model and IS saved in every checkpoint; the text-side projection head
was NOT saved. Without it we cannot map a clinical concept phrase into the same
128-d space as ``z_it``, so concept scores can't reflect the alignment.

This module defines the small projection we *recover* post-hoc (see
``recover_text_projection.py``): a single L2-normalized linear map
768 -> 128 from BioMedCLIP text features into the ``z_it`` space. We use a plain
linear map (not the original 768->768->128 MLP) because it is fit against only a
handful of class anchors and a linear, near-orthogonal map generalizes to unseen
concept phrases far better than an over-parameterized MLP.

The recovered map is saved as a small ``.pth`` bundle and loaded by
``explainability.py`` to embed concept phrases for the WHY scores.
"""

from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class RecoveredTextProjection(nn.Module):
    """Linear BioMedCLIP-text -> z_it-space projection, L2-normalized output.

    A single ``Linear(in_dim, out_dim)``. ``out_dim`` matches the model's
    ``proj_out_dim`` (128) so projected concept vectors live in the same space as
    ``z_it`` and cosine similarity is meaningful.
    """

    def __init__(self, in_dim: int = 768, out_dim: int = 128):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.proj = nn.Linear(in_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.proj(x), dim=-1)


def save_text_projection(
    path: str,
    projection: RecoveredTextProjection,
    centroids: torch.Tensor,
    class_prototypes: torch.Tensor,
    meta: Dict,
) -> None:
    """Persist the recovered projection plus the class anchors and metadata.

    centroids        : (n_classes, out_dim)  per-class z_it centroids (the exact,
                       checkpoint-faithful class anchors / validation oracle).
    class_prototypes : (n_classes, out_dim)  projection(class_description) — what
                       the fitted map produces for each class description.
    meta             : free-form dict (dataset, temperature, lambda_ord_it, ckpt).
    """
    bundle = {
        "state_dict": projection.state_dict(),
        "in_dim": projection.in_dim,
        "out_dim": projection.out_dim,
        "centroids": centroids.detach().cpu(),
        "class_prototypes": class_prototypes.detach().cpu(),
        "meta": meta,
    }
    torch.save(bundle, path)


def load_text_projection(
    path: str, device: torch.device | str = "cpu"
) -> Tuple[RecoveredTextProjection, Dict]:
    """Load a recovered projection saved by ``save_text_projection``.

    Returns ``(projection_module_in_eval_mode, bundle)`` where ``bundle`` also
    carries ``centroids``, ``class_prototypes`` and ``meta``.
    """
    bundle = torch.load(path, map_location=device)
    proj = RecoveredTextProjection(bundle["in_dim"], bundle["out_dim"])
    proj.load_state_dict(bundle["state_dict"])
    proj = proj.to(device).eval()
    for p in proj.parameters():
        p.requires_grad = False
    return proj, bundle
