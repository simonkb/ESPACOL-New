"""
Full Hybrid Supervised Contrastive Ordinal Learning framework.

Baseline (HybridContrastiveOrdinalModel):
  Input -> EfficientNet-V2S -> GAP -> 1280-dim features
              |
              |----> PCOL Head       (1280->1280->128, L2-norm)
              |----> SCOLw Head      (1280->1280->128, L2-norm)
              |----> Regression Head (1280->1, scalar)
OPTIC extension (OPTICModel):
  Same as baseline but with optional components:
              |----> OrdinalDistributionHead   (CORAL; replaces RMSE regression)
              |
  Multi-tile path with CTOT:
    Tiled backbone -> CTOT -> [GRADE] token features (1280-dim)
              |----> GradePrototypeAttention   (grade-specific tile attribution maps)

All new components are opt-in via constructor flags — when all False, OPTICModel
behaves identically to HybridContrastiveOrdinalModel.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .backbone import EfficientNetV2SBackbone, TiledEfficientNetBackbone
from .concept_prototype import ConceptGradePrototypeModule
from .grade_prototype import GradePrototypeAttention
from .heads import MLPProjectionHead, OrdinalDistributionHead, RegressionHead


class HybridContrastiveOrdinalModel(nn.Module):

    def __init__(
        self,
        backbone: EfficientNetV2SBackbone,
        pcol_head: MLPProjectionHead,
        scolw_head: MLPProjectionHead,
        regression_head: RegressionHead,
    ):
        super().__init__()
        self.backbone = backbone
        self.pcol_head = pcol_head
        self.scolw_head = scolw_head
        self.regression_head = regression_head

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor | None]:
        features = self.backbone(x)

        z_pcol = self.pcol_head(features)
        z_scolw = self.scolw_head(features)
        pred = self.regression_head(features)

        return {
            "features": features,
            "z_pcol": z_pcol,
            "z_scolw": z_scolw,
            "pred": pred,
        }

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        return self.regression_head(features)


class OPTICModel(HybridContrastiveOrdinalModel):
    """
    OPTIC: Ordinal Prototype Tile Interaction for Clinical grading.

    Extends HybridContrastiveOrdinalModel with three independently toggleable components:
      1. CrossTileOrdinalTransformer — already embedded in TiledEfficientNetBackbone
         when use_transformer=True; no separate member here.
      2. GradePrototypeAttention (gpa) — per-tile grade evidence for explainability.
      3. OrdinalDistributionHead (ordinal_head) — CORAL-style distribution, replaces RMSE.

    forward() returns the same keys as the parent plus:
        ordinal_probs  (N, K-1)   — P(Y > k), present when ordinal_head is not None
        tile_evidence  (N, T, K)  — per-tile grade affinity, present when gpa is not None
        tile_weights   (N, K, T)  — per-grade tile attention, present when gpa is not None
        grade_features (N, K, d)  — grade-specific representations, present when gpa is not None
    """

    def __init__(
        self,
        backbone: TiledEfficientNetBackbone,
        pcol_head: MLPProjectionHead,
        scolw_head: MLPProjectionHead,
        regression_head: RegressionHead,
        gpa: GradePrototypeAttention | None = None,
        ordinal_head: OrdinalDistributionHead | None = None,
    ):
        super().__init__(
            backbone=backbone,
            pcol_head=pcol_head,
            scolw_head=scolw_head,
            regression_head=regression_head,
        )
        self.gpa = gpa
        self.ordinal_head = ordinal_head

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor | None]:
        tile_evidence = tile_weights = grade_features = None

        if self.gpa is not None:
            features, ctot_feats, _raw_tiles = self.backbone.forward_tiles(x)
            grade_features, tile_evidence, tile_weights = self.gpa(ctot_feats)
        else:
            features = self.backbone(x)

        z_pcol = self.pcol_head(features)
        z_scolw = self.scolw_head(features)

        ordinal_logits = None
        if self.ordinal_head is not None:
            ordinal_logits = self.ordinal_head(features)          # (N, K-1) raw logits
            pred = torch.sigmoid(ordinal_logits).sum(dim=1)       # (N,) expected grade
        else:
            pred = self.regression_head(features)

        return {
            "features": features,
            "z_pcol": z_pcol,
            "z_scolw": z_scolw,
            "pred": pred,
            "ordinal_logits": ordinal_logits,
            "tile_evidence": tile_evidence,
            "tile_weights": tile_weights,
            "grade_features": grade_features,
        }

    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        if self.ordinal_head is not None:
            return self.ordinal_head.predict(features)
        return self.regression_head(features)


class OPTICConceptModel(OPTICModel):
    """
    OPTIC-C: Concept-Grounded Grade Prototype extension.

    Adds ConceptGradePrototypeModule on top of OPTICModel.
    Novel losses dominate (~98% of gradient signal):
      - L_proto_CE:     cosine prototype CrossEntropy (image ↔ grade prototypes)
      - L_concept:      prototype ↔ grade text alignment
      - L_tile_concept: per-tile concept BCE vs clinical grade-concept targets
    SCOLw and PCOL are disabled (alpha=0, beta=0).

    Additional forward() keys:
        proto_logits        (N, K)     — cosine prototype similarity / temperature
        concept_align_loss  scalar     — prototype-text alignment loss (pre-computed)
        tile_concept_scores (N, T, C)  — per-tile concept probabilities (explainability)
        tile_concept_loss   scalar     — tile concept BCE loss (pre-computed)
        raw_tile_features   (N, T, D)  — raw backbone tile features for debugging
    """

    def __init__(
        self,
        backbone: TiledEfficientNetBackbone,
        pcol_head: MLPProjectionHead,
        scolw_head: MLPProjectionHead,
        regression_head: RegressionHead,
        concept_module: ConceptGradePrototypeModule,
        gpa: GradePrototypeAttention | None = None,
        ordinal_head: OrdinalDistributionHead | None = None,
    ):
        super().__init__(
            backbone=backbone,
            pcol_head=pcol_head,
            scolw_head=scolw_head,
            regression_head=regression_head,
            gpa=gpa,
            ordinal_head=ordinal_head,
        )
        self.concept_module = concept_module

    def forward(
        self,
        x: torch.Tensor,
        labels: torch.Tensor | None = None,
        concept_embeds: torch.Tensor | None = None,
        grade_text_embeds: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor | None]:
        tile_evidence = tile_weights = grade_features = None

        # Always run forward_tiles to get raw_tile_features for concept scoring
        features, ctot_feats, raw_tile_features = self.backbone.forward_tiles(x)

        if self.gpa is not None and ctot_feats is not None:
            grade_features, tile_evidence, tile_weights = self.gpa(ctot_feats)

        z_pcol = self.pcol_head(features)
        z_scolw = self.scolw_head(features)

        ordinal_logits = None
        if self.ordinal_head is not None:
            ordinal_logits = self.ordinal_head(features)
            pred = torch.sigmoid(ordinal_logits).sum(dim=1)
        else:
            pred = self.regression_head(features)

        # Concept prototype forward — losses pre-computed here
        proto_logits = concept_align_loss = tile_concept_scores = tile_concept_loss = None
        if labels is not None:
            proto_logits, concept_align_loss, tile_concept_scores, tile_concept_loss = (
                self.concept_module(
                    image_features=features,
                    raw_tile_features=raw_tile_features,
                    concept_embeds=concept_embeds,
                    grade_text_embeds=grade_text_embeds,
                    labels=labels,
                )
            )

        return {
            "features": features,
            "z_pcol": z_pcol,
            "z_scolw": z_scolw,
            "pred": pred,
            "ordinal_logits": ordinal_logits,
            "tile_evidence": tile_evidence,
            "tile_weights": tile_weights,
            "grade_features": grade_features,
            "raw_tile_features": raw_tile_features,
            # Concept outputs
            "proto_logits": proto_logits,
            "concept_align_loss": concept_align_loss,
            "tile_concept_scores": tile_concept_scores,
            "tile_concept_loss": tile_concept_loss,
        }


def build_model(
    n_classes: int,
    pretrained: bool = True,
    proj_hidden_dim: int = 1280,
    proj_out_dim: int = 128,
    use_multi_tile: bool = False,
    grad_checkpoint: bool = False,
    tile_grid: int = 3,
    # OPTIC architecture flags
    use_tile_transformer: bool = False,
    tile_transformer_dim: int = 512,
    tile_transformer_nhead: int = 8,
    tile_transformer_layers: int = 2,
    tile_transformer_dropout: float = 0.1,
    use_grade_prototypes: bool = False,
    use_ordinal_head: bool = False,
    # OPTIC-C: concept prototype flags
    use_concept_prototype: bool = False,
    n_concepts: int = 9,
    proto_temperature: float = 0.1,
) -> HybridContrastiveOrdinalModel:

    if use_tile_transformer and not use_multi_tile:
        raise ValueError("CrossTileOrdinalTransformer requires use_multi_tile=True")
    if use_grade_prototypes and not use_tile_transformer:
        raise ValueError("GradePrototypeAttention requires use_tile_transformer=True")
    if use_concept_prototype and not use_tile_transformer:
        raise ValueError("OPTICConceptModel requires use_tile_transformer=True")

    if use_multi_tile:
        backbone = TiledEfficientNetBackbone(
            pretrained=pretrained,
            grad_checkpoint=grad_checkpoint,
            use_transformer=use_tile_transformer,
            tile_grid=tile_grid,
            transformer_dim=tile_transformer_dim,
            transformer_nhead=tile_transformer_nhead,
            transformer_layers=tile_transformer_layers,
            transformer_dropout=tile_transformer_dropout,
        )
    else:
        backbone = EfficientNetV2SBackbone(
            pretrained=pretrained,
            grad_checkpoint=grad_checkpoint,
        )
    feat_dim = backbone.OUT_DIM

    pcol_head = MLPProjectionHead(feat_dim, proj_hidden_dim, proj_out_dim)
    scolw_head = MLPProjectionHead(feat_dim, proj_hidden_dim, proj_out_dim)
    reg_head = RegressionHead(feat_dim)

    gpa = None
    if use_grade_prototypes:
        gpa = GradePrototypeAttention(d_model=tile_transformer_dim, n_classes=n_classes)

    ordinal_head = None
    if use_ordinal_head:
        ordinal_head = OrdinalDistributionHead(input_dim=feat_dim, n_classes=n_classes)

    if use_concept_prototype:
        concept_module = ConceptGradePrototypeModule(
            feat_dim=feat_dim,
            n_classes=n_classes,
            n_concepts=n_concepts,
            proj_dim=proj_out_dim,
            temperature=proto_temperature,
        )
        return OPTICConceptModel(
            backbone=backbone,
            pcol_head=pcol_head,
            scolw_head=scolw_head,
            regression_head=reg_head,
            concept_module=concept_module,
            gpa=gpa,
            ordinal_head=ordinal_head,
        )

    use_optic = use_tile_transformer or use_grade_prototypes or use_ordinal_head
    if use_optic:
        return OPTICModel(
            backbone=backbone,
            pcol_head=pcol_head,
            scolw_head=scolw_head,
            regression_head=reg_head,
            gpa=gpa,
            ordinal_head=ordinal_head,
        )

    return HybridContrastiveOrdinalModel(
        backbone=backbone,
        pcol_head=pcol_head,
        scolw_head=scolw_head,
        regression_head=reg_head,
    )
