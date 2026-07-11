"""
Full Hybrid Supervised Contrastive Ordinal Learning framework.

Baseline architecture:
  Input -> EfficientNet-V2S -> GAP -> 1280-dim features
              |
              |----> PCOL Head       (1280->1280->128, L2-norm)
              |----> SCOLw Head      (1280->1280->128, L2-norm)
              |----> Regression Head (1280->1, scalar)

ESPAOCL extensions:
              |----> Image-Text Head  (1280->1280->128, L2-norm)
              |----> Concept Spine    (1280->128, cosine sim -> M concept activations)

During training: all heads are jointly optimized.
During testing: only the regression head is used for inference (+ concept spine if enabled).
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from .backbone import EfficientNetV2SBackbone, TiledEfficientNetBackbone
from .concept_spine import ConceptSpine
from .heads import MLPProjectionHead, OrdinalDistributionHead, RegressionHead


class HybridContrastiveOrdinalModel(nn.Module):

    def __init__(
        self,
        backbone: EfficientNetV2SBackbone,
        pcol_head: MLPProjectionHead,
        scolw_head: MLPProjectionHead,
        regression_head,   # RegressionHead or OrdinalDistributionHead
        image_text_head: Optional[MLPProjectionHead] = None,
        concept_spine: Optional[ConceptSpine] = None,
    ):
        super().__init__()
        self.backbone = backbone
        self.pcol_head = pcol_head
        self.scolw_head = scolw_head
        self.regression_head = regression_head
        self.image_text_head = image_text_head
        self.concept_spine = concept_spine
        self._use_ordinal_head = isinstance(regression_head, OrdinalDistributionHead)

    def forward(
        self,
        x: torch.Tensor,
        concept_text_emb: Optional[torch.Tensor] = None,  # (M, 128)
    ) -> dict:
        """
        Returns a dictionary for stable access in both baseline and ESPAOCL modes.

        Keys always present:
            features      : (N, 1280)
            z_pcol        : (N, 128)
            z_scolw       : (N, 128)
            z_it          : (N, 128) or None
            pred          : (N,)  — continuous grade (scalar for all head types)

        Key added when OrdinalDistributionHead is active:
            ordinal_probs : (N, K-1)  — P(Y > k) for k=0..K-2

        Keys added when concept_spine is enabled:
            c       : (N, M)   concept activations
            w       : (M,)     concept importance weights
            E       : (N,)     ordinal evidence
            y_soft  : (N,)     soft ordinal grade
            z_c     : (N, 128) L2-normalised concept image embedding
        """
        features = self.backbone(x)

        z_pcol = self.pcol_head(features)
        z_scolw = self.scolw_head(features)

        if self._use_ordinal_head:
            ordinal_probs = self.regression_head(features)          # (N, K-1)
            pred = self.regression_head.predict(ordinal_probs)      # (N,)
        else:
            ordinal_probs = None
            pred = self.regression_head(features)                   # (N,)

        z_it = None
        if self.image_text_head is not None:
            z_it = self.image_text_head(features)

        out = {
            "features": features,
            "z_pcol": z_pcol,
            "z_scolw": z_scolw,
            "z_it": z_it,
            "pred": pred,
        }

        if ordinal_probs is not None:
            out["ordinal_probs"] = ordinal_probs

        if self.concept_spine is not None and concept_text_emb is not None:
            c, w, E, y_soft, z_c = self.concept_spine(features, concept_text_emb)
            out.update({"c": c, "w": w, "E": E, "y_soft": y_soft, "z_c": z_c})

        return out

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Return 1280-dim GAP features from the EfficientNet-V2S backbone."""
        return self.backbone(x)

    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Inference-only: returns continuous grade (scalar)."""
        features = self.backbone(x)
        if self._use_ordinal_head:
            probs = self.regression_head(features)
            return self.regression_head.predict(probs)
        return self.regression_head(features)

    @torch.no_grad()
    def predict_with_concepts(
        self,
        x: torch.Tensor,
        concept_text_emb: torch.Tensor,  # (M, 128)
        top_k: int = 3,
        concept_names: Optional[list] = None,
    ) -> dict:
        """
        Single-pass inference returning grade + top-k WHY concepts.
        WHERE heatmaps are generated separately via LayerCAM (requires grad).
        """
        out = self.forward(x, concept_text_emb)
        pred_grade = out["pred"].round().clamp(0).long()

        result = {"pred_grade": pred_grade, "pred_continuous": out["pred"]}

        if "c" in out:
            scores = out["w"] * out["c"]              # (N, M)
            topk_vals, topk_idx = scores.topk(top_k, dim=1)
            result["concept_scores"] = topk_vals
            result["concept_indices"] = topk_idx
            result["E"] = out["E"]
            result["y_concept"] = self.concept_spine.predict_grade(out["E"])
            if concept_names is not None:
                result["concept_names"] = [
                    [concept_names[i] for i in row.tolist()]
                    for row in topk_idx
                ]

        return result


def build_model(
    n_classes: int,
    pretrained: bool = True,
    proj_hidden_dim: int = 1280,
    proj_out_dim: int = 128,
    use_image_text: bool = False,
    use_multi_tile: bool = False,
    use_concept_spine: bool = False,
    n_concepts: int = 9,
    absence_indices: Optional[list] = None,
    grad_checkpoint: bool = False,
    # CrossTileOrdinalTransformer options
    use_tile_transformer: bool = False,
    transformer_dim: int = 512,
    transformer_nhead: int = 8,
    transformer_layers: int = 2,
    transformer_dropout: float = 0.1,
    # OrdinalDistributionHead option
    use_ordinal_head: bool = False,
) -> HybridContrastiveOrdinalModel:
    if use_multi_tile:
        backbone = TiledEfficientNetBackbone(
            pretrained=pretrained,
            grad_checkpoint=grad_checkpoint,
            use_transformer=use_tile_transformer,
            transformer_dim=transformer_dim,
            transformer_nhead=transformer_nhead,
            transformer_layers=transformer_layers,
            transformer_dropout=transformer_dropout,
        )
    else:
        backbone = EfficientNetV2SBackbone(pretrained=pretrained, grad_checkpoint=grad_checkpoint)
    feat_dim = backbone.OUT_DIM

    pcol_head = MLPProjectionHead(feat_dim, proj_hidden_dim, proj_out_dim)
    scolw_head = MLPProjectionHead(feat_dim, proj_hidden_dim, proj_out_dim)

    if use_ordinal_head:
        reg_head = OrdinalDistributionHead(feat_dim, n_classes)
    else:
        reg_head = RegressionHead(feat_dim)

    image_text_head = None
    if use_image_text:
        image_text_head = MLPProjectionHead(feat_dim, proj_hidden_dim, proj_out_dim)

    spine = None
    if use_concept_spine:
        spine = ConceptSpine(
            in_dim=feat_dim,
            proj_dim=proj_out_dim,
            n_concepts=n_concepts,
            n_classes=n_classes,
            absence_indices=absence_indices,
        )

    return HybridContrastiveOrdinalModel(
        backbone=backbone,
        pcol_head=pcol_head,
        scolw_head=scolw_head,
        regression_head=reg_head,
        image_text_head=image_text_head,
        concept_spine=spine,
    )