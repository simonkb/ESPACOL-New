"""
Grade Prototype Memory — EMA per-grade prototype embeddings for stable
contrastive supervision with rare DR classes.

With batch_size=24 and 73% grade-0, minority grades (3: 2.5%, 4: 2.0%)
appear in only ~40% of batches. SCOLw then has no negative pairs for those
grades, making its gradient noisy. This module maintains a running mean
embedding per grade (updated via EMA after every batch) and computes a
prototype cross-entropy loss so that minority grades always have a stable
contrastive target regardless of their batch frequency.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class GradePrototypeMemory(nn.Module):
    """
    Non-parametric EMA memory of per-grade prototype embeddings.

    Prototypes live on the unit sphere (L2-normalised after each update).
    No learnable parameters — purely a training utility.
    """

    def __init__(
        self,
        feat_dim: int = 128,
        n_classes: int = 5,
        momentum: float = 0.9,
        temperature: float = 0.1,
    ):
        super().__init__()
        self.n_classes = n_classes
        self.momentum = momentum
        self.temperature = temperature

        # Initialised from data, not random — start as zeros and mark uninit
        self.register_buffer("prototypes", torch.zeros(n_classes, feat_dim))
        self.register_buffer("initialized", torch.zeros(n_classes, dtype=torch.bool))

    @property
    def is_ready(self) -> bool:
        return bool(self.initialized.all())

    @torch.no_grad()
    def update(self, embeddings: torch.Tensor, labels: torch.Tensor) -> None:
        """EMA update using current-batch mean embeddings per grade."""
        embeddings = embeddings.float()
        for k in range(self.n_classes):
            mask = (labels == k)
            if not mask.any():
                continue
            mean_emb = F.normalize(embeddings[mask].mean(0), dim=0)
            if not self.initialized[k]:
                self.prototypes[k] = mean_emb
                self.initialized[k] = True
            else:
                self.prototypes[k] = F.normalize(
                    self.momentum * self.prototypes[k] + (1.0 - self.momentum) * mean_emb,
                    dim=0,
                )

    def prototype_loss(
        self, embeddings: torch.Tensor, labels: torch.Tensor
    ) -> torch.Tensor:
        """
        Cross-entropy over cosine similarity to stored prototypes.
        Equivalent to InfoNCE against K fixed prototypes.
        Always computed in float32 for numerical stability.
        """
        z = embeddings.float()
        p = self.prototypes.float()          # (K, D)  unit vectors
        sims = z @ p.T / self.temperature    # (N, K)
        return F.cross_entropy(sims, labels)
