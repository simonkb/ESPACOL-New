"""
ConceptGradePrototypeModule: grade prototypes in concept space.

Provides three novel contributions:
  1. L_proto_CE  — cosine prototype matching with label supervision (dominant signal)
  2. L_concept   — aligns grade prototypes with clinical grade text embeddings
  3. L_tile      — supervises per-tile concept scores against grade-concept targets

Dual explainability outputs (spatial heatmap done by GPA; this adds concept axis):
  tile_concept_scores (N, T, C) — per-tile concept presence probability
  concept_report: readable from scores by caller (top-k per image)

Grade-concept soft targets are clinical priors: no learnable annotations required.
Tile-level zero-shot concept scoring is done via BiomedCLIP concept text embeddings.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# Clinical soft targets: grade_concept_targets[grade, concept]
# Concepts: microaneurysms, dot/blot hemorrhages, hard exudates, cotton wool spots,
#           venous beading, IRMA, neovascularization, preretinal hemorrhage, vitreous hemo
_GRADE_CONCEPT_TARGETS = [
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # Grade 0: no lesions
    [0.9, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # Grade 1: microaneurysms only
    [0.7, 0.7, 0.6, 0.3, 0.0, 0.0, 0.0, 0.0, 0.0],  # Grade 2: moderate NPDR
    [0.5, 0.9, 0.4, 0.5, 0.8, 0.7, 0.0, 0.0, 0.0],  # Grade 3: severe NPDR
    [0.3, 0.8, 0.2, 0.2, 0.5, 0.5, 0.9, 0.8, 0.7],  # Grade 4: PDR
]


class ConceptGradePrototypeModule(nn.Module):
    """
    Args:
        feat_dim:      backbone output dim (1280 for EfficientNetV2S)
        n_classes:     number of grade classes (5)
        n_concepts:    number of DR concepts (9)
        proj_dim:      concept projection output dim — must match text encoder proj_out_dim
        temperature:   cosine similarity scaling for proto_logits
    """

    def __init__(
        self,
        feat_dim: int = 1280,
        n_classes: int = 5,
        n_concepts: int = 9,
        proj_dim: int = 128,
        temperature: float = 0.1,
    ):
        super().__init__()
        self.temperature = temperature
        self.n_classes = n_classes
        self.n_concepts = n_concepts

        # Learnable grade prototype in backbone feature space
        self.grade_prototypes = nn.Parameter(torch.empty(n_classes, feat_dim))
        nn.init.trunc_normal_(self.grade_prototypes, std=0.02)

        # Projects backbone features into the same space as text embeddings
        self.concept_projection = nn.Sequential(
            nn.Linear(feat_dim, proj_dim),
        )

        # Clinical grade-concept soft supervision (no annotation required)
        targets = torch.tensor(_GRADE_CONCEPT_TARGETS, dtype=torch.float32)
        self.register_buffer("grade_concept_targets", targets)  # (n_classes, n_concepts)

    def forward(
        self,
        image_features: torch.Tensor,
        raw_tile_features: torch.Tensor,
        concept_embeds: torch.Tensor | None,
        grade_text_embeds: torch.Tensor | None,
        labels: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
        """
        Args:
            image_features:   (N, feat_dim)   — aggregated image representation
            raw_tile_features:(N, T, feat_dim) — per-tile backbone features
            concept_embeds:   (C, proj_dim)   — concept text embeddings (frozen BiomedCLIP)
            grade_text_embeds:(K, proj_dim)   — grade text embeddings (from ClinicalTextEncoder)
            labels:           (N,) long        — ground truth grade indices

        Returns:
            proto_logits:       (N, K)   — cosine prototype similarity / temperature
            concept_align_loss: scalar   — prototype-to-grade-text alignment (None if no text embeds)
            tile_concept_scores:(N, T, C) — per-tile per-concept cosine similarity (None if no concept embeds)
            tile_concept_loss:  scalar   — BCE tile concept vs clinical targets (None if no concept embeds)
        """
        # --- 1. Prototype logits (L_proto_CE supervision signal) ---
        img_norm = F.normalize(image_features, dim=-1)            # (N, D)
        proto_norm = F.normalize(self.grade_prototypes, dim=-1)   # (K, D)
        proto_logits = torch.matmul(img_norm, proto_norm.T) / self.temperature  # (N, K)

        # --- 2. Concept alignment loss: prototypes ↔ grade text embeddings ---
        concept_align_loss = None
        if grade_text_embeds is not None:
            proj_protos = F.normalize(self.concept_projection(self.grade_prototypes), dim=-1)  # (K, proj_dim)
            # Each grade prototype should match its grade text embedding
            cosine_align = (proj_protos * grade_text_embeds).sum(dim=-1)   # (K,)
            concept_align_loss = (1.0 - cosine_align).mean()

        # --- 3. Tile concept scores + loss ---
        tile_concept_scores = None
        tile_concept_loss = None
        if concept_embeds is not None:
            N, T, D = raw_tile_features.shape
            tiles_flat = raw_tile_features.view(N * T, D)
            proj_tiles = F.normalize(self.concept_projection(tiles_flat), dim=-1)  # (N*T, proj_dim)
            proj_tiles = proj_tiles.view(N, T, -1)                                 # (N, T, proj_dim)

            concept_norm = F.normalize(concept_embeds, dim=-1)                     # (C, proj_dim)
            # Cosine similarity: (N, T, C)
            tile_concept_scores = torch.einsum("ntd,cd->ntc", proj_tiles, concept_norm)

            # BCEWithLogits: scale cosine sims to sharper logits, aggregate over tiles,
            # supervise against clinical grade-concept soft targets. AMP-safe.
            scaled_logits = tile_concept_scores * 10.0                # (N, T, C)
            mean_logits = scaled_logits.mean(dim=1)                   # (N, C)
            targets = self.grade_concept_targets[labels]               # (N, C)
            tile_concept_loss = F.binary_cross_entropy_with_logits(mean_logits, targets)

        return proto_logits, concept_align_loss, tile_concept_scores, tile_concept_loss
