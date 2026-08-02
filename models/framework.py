"""
Full Hybrid Supervised Contrastive Ordinal Learning framework.

Baseline (HybridContrastiveOrdinalModel):
  Input -> EfficientNet-V2S -> GAP -> 1280-dim features
              |
              |----> PCOL Head       (1280->1280->128, L2-norm)
              |----> SCOLw Head      (1280->1280->128, L2-norm)
              |----> Regression Head (1280->1, scalar)
              |----> Image-Text Head (1280->1280->128, L2-norm)  [optional]

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
from .grade_prototype import GradePrototypeAttention
from .heads import MLPProjectionHead, OrdinalDistributionHead, RegressionHead


class HybridContrastiveOrdinalModel(nn.Module):

    def __init__(
        self,
        backbone: EfficientNetV2SBackbone,
        pcol_head: MLPProjectionHead,
        scolw_head: MLPProjectionHead,
        regression_head: RegressionHead,
        image_text_head: MLPProjectionHead | None = None,
    ):
        super().__init__()
        self.backbone = backbone
        self.pcol_head = pcol_head
        self.scolw_head = scolw_head
        self.regression_head = regression_head
        self.image_text_head = image_text_head

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor | None]:
        features = self.backbone(x)

        z_pcol = self.pcol_head(features)
        z_scolw = self.scolw_head(features)
        pred = self.regression_head(features)

        z_it = None
        if self.image_text_head is not None:
            z_it = self.image_text_head(features)

        return {
            "features": features,
            "z_pcol": z_pcol,
            "z_scolw": z_scolw,
            "z_it": z_it,
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
        image_text_head: MLPProjectionHead | None = None,
        gpa: GradePrototypeAttention | None = None,
        ordinal_head: OrdinalDistributionHead | None = None,
    ):
        super().__init__(
            backbone=backbone,
            pcol_head=pcol_head,
            scolw_head=scolw_head,
            regression_head=regression_head,
            image_text_head=image_text_head,
        )
        self.gpa = gpa
        self.ordinal_head = ordinal_head

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor | None]:
        tile_evidence = tile_weights = grade_features = None

        if self.gpa is not None:
            features, ctot_feats = self.backbone.forward_tiles(x)
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

        z_it = None
        if self.image_text_head is not None:
            z_it = self.image_text_head(features)

        return {
            "features": features,
            "z_pcol": z_pcol,
            "z_scolw": z_scolw,
            "z_it": z_it,
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


def build_model(
    n_classes: int,
    pretrained: bool = True,
    proj_hidden_dim: int = 1280,
    proj_out_dim: int = 128,
    use_image_text: bool = False,
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
) -> HybridContrastiveOrdinalModel:

    if use_tile_transformer and not use_multi_tile:
        raise ValueError("CrossTileOrdinalTransformer requires use_multi_tile=True")
    if use_grade_prototypes and not use_tile_transformer:
        raise ValueError("GradePrototypeAttention requires use_tile_transformer=True")

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

    image_text_head = None
    if use_image_text:
        image_text_head = MLPProjectionHead(feat_dim, proj_hidden_dim, proj_out_dim)

    # Use OPTICModel when any new component is enabled
    use_optic = use_tile_transformer or use_grade_prototypes or use_ordinal_head

    gpa = None
    if use_grade_prototypes:
        gpa = GradePrototypeAttention(d_model=tile_transformer_dim, n_classes=n_classes)

    ordinal_head = None
    if use_ordinal_head:
        ordinal_head = OrdinalDistributionHead(input_dim=feat_dim, n_classes=n_classes)

    if use_optic:
        return OPTICModel(
            backbone=backbone,
            pcol_head=pcol_head,
            scolw_head=scolw_head,
            regression_head=reg_head,
            image_text_head=image_text_head,
            gpa=gpa,
            ordinal_head=ordinal_head,
        )

    return HybridContrastiveOrdinalModel(
        backbone=backbone,
        pcol_head=pcol_head,
        scolw_head=scolw_head,
        regression_head=reg_head,
        image_text_head=image_text_head,
    )
