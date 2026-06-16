"""
Full Hybrid Supervised Contrastive Ordinal Learning framework.

Default architecture (BiomedCLIP image backbone):
  Input (224x224) -> BiomedCLIP ViT-B/16 -> feat_dim features
              |
              |----> PCOL Head       (feat_dim -> feat_dim -> 128, L2-norm)
              |----> SCOLw Head      (feat_dim -> feat_dim -> 128, L2-norm)
              |----> Regression Head (feat_dim -> 1, scalar)

ESPAOCL extension (use_image_text=True):
              |----> Image-Text Head (feat_dim -> feat_dim -> 128, L2-norm)

TAMO extension (use_tamo=True):
              |----> TAMO Head       (feat_dim -> feat_dim -> 128, L2-norm)
                     Uses DeepProjectionHead for all heads.

Two-Stage Hierarchical extension (use_two_stage=True):
  Splits the ordinal problem into two specialised heads trained jointly:
    regression_head : ScreeningHead — binary grade-0 vs grade-1+
                      (RegressionHead scalar logit → BCE loss)
    grading_head    : CoralHead(n_classes-1) — ordinal grades 1..K-1
                      trained ONLY on DR-positive samples per batch.
  Inference: sigmoid(screening_logit) > 0.5 → use grading_head; else grade 0.

  Mathematical justification:
    With 73.5% grade-0, RMSE regression has a degenerate attractor at ŷ≈0.
    Separating screening from grading eliminates this attractor:
      Stage 1 (binary): BiomedCLIP trivially separates DR vs no-DR → 95%+
      Stage 2 (ordinal): balanced 4-class problem on DR+ only → 60-70%
    Combined expected accuracy: 0.735×0.97 + 0.265×0.65 ≈ 88.5%

feat_dim is probed from the backbone at build time (OUT_DIM attribute).
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .backbone import BiomedCLIPImageBackbone, TiledBiomedCLIPBackbone, EfficientNetV2SBackbone
from .heads import (
    MLPProjectionHead,
    DeepProjectionHead,
    RegressionHead,
    DeepRegressionHead,
    CoralHead,
)

_DEFAULT_IMAGE_ENCODER = (
    "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
)


class HybridContrastiveOrdinalModel(nn.Module):

    def __init__(
        self,
        backbone: nn.Module,
        pcol_head: nn.Module,
        scolw_head: nn.Module,
        regression_head: nn.Module,
        image_text_head: nn.Module | None = None,
        tamo_head: nn.Module | None = None,
        grading_head: nn.Module | None = None,
    ):
        super().__init__()
        self.backbone = backbone
        self.pcol_head = pcol_head
        self.scolw_head = scolw_head
        self.regression_head = regression_head
        self.image_text_head = image_text_head
        # Dedicated TAMO projection head — separate from the image-text head
        # to avoid conflicting gradient directions between sample-level IT
        # alignment and prototype-level geometry distillation.
        self.tamo_head = tamo_head
        # Two-stage grading head (CoralHead for grades 1..K-1).
        # Present only when use_two_stage=True.  regression_head then acts as
        # the binary screening head (grade-0 vs grade-1+).
        self.grading_head = grading_head

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor | None]:
        """
        Returns a dictionary for stable access in all modes.

        Returns:
            features      : (N, feat_dim)
            z_pcol        : (N, 128)
            z_scolw       : (N, 128)
            z_it          : (N, 128) or None
            z_tamo        : (N, 128) or None
            pred          : (N,) screening logit | (N,) regression | (N, K-1) coral
            pred_grading  : (N, K-2) coral logits for grades 1..K-1, or None
        """
        features = self.backbone(x)

        z_pcol = self.pcol_head(features)
        z_scolw = self.scolw_head(features)
        pred = self.regression_head(features)

        z_it = None
        if self.image_text_head is not None:
            z_it = self.image_text_head(features)

        z_tamo = None
        if self.tamo_head is not None:
            z_tamo = self.tamo_head(features)

        pred_grading = None
        if self.grading_head is not None:
            pred_grading = self.grading_head(features)

        return {
            "features": features,
            "z_pcol": z_pcol,
            "z_scolw": z_scolw,
            "z_it": z_it,
            "z_tamo": z_tamo,
            "pred": pred,
            "pred_grading": pred_grading,
        }

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Return backbone features (dim = backbone.OUT_DIM)."""
        return self.backbone(x)

    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Inference-only: returns continuous grade prediction in [0, K-1].

        Two-stage mode (grading_head is not None):
            screen_prob = sigmoid(regression_head(features))
            grade_cont  = coral_predict(grading_head(features)) + 1  in [1, K-1]
            prediction  = grade_cont  if screen_prob > 0.5
                          0.0         otherwise
            (continuous for MAE; round() for accuracy)

        CORAL single-head mode (regression_head is CoralHead):
            prediction = sum(sigmoid(coral_logits))   in [0, K-1]

        Regression mode:
            prediction = regression_head(features)    in ℝ (clamped externally)
        """
        features = self.backbone(x)

        if self.grading_head is not None:
            # Two-stage: screening then ordinal grading
            screen_logit = self.regression_head(features)      # (N,)
            p_dr = torch.sigmoid(screen_logit)                 # (N,) in [0, 1]
            grading_cont = self.grading_head.predict(features) # (N,) in [0, K-2]
            grade = torch.where(
                p_dr > 0.5,
                grading_cont + 1.0,                            # predicted grade 1..K-1
                torch.zeros_like(p_dr),                        # predicted grade 0
            )
            return grade

        if hasattr(self.regression_head, "predict"):
            return self.regression_head.predict(features)

        return self.regression_head(features)


def build_model(
    n_classes: int,
    pretrained: bool = True,
    proj_hidden_dim: int = 0,
    proj_out_dim: int = 128,
    use_image_text: bool = False,
    use_tamo: bool = False,
    use_coral: bool = False,
    use_two_stage: bool = False,
    use_multi_tile: bool = False,
    image_encoder_name: str = _DEFAULT_IMAGE_ENCODER,
) -> HybridContrastiveOrdinalModel:
    """
    Build the HybridContrastiveOrdinalModel with BiomedCLIP image backbone.

    use_multi_tile=True replaces the single-view backbone with
    TiledBiomedCLIPBackbone: the dataloader must then supply (N, T, C, H, W)
    batches (see Datasets.dataloaders.build_tile_transform). Orthogonal to
    use_tamo/use_coral/use_two_stage — it only changes how `features` are
    produced, not which head/loss consumes them.

    Head selection priority (mutually exclusive for the regression/grading slot):

        use_two_stage=True  →  regression_head = RegressionHead (binary screening)
                               grading_head    = CoralHead(n_classes-1) (grades 1..K-1)
        use_coral=True      →  regression_head = CoralHead(n_classes)  (all K classes)
        use_tamo=True       →  regression_head = DeepRegressionHead
        default             →  regression_head = RegressionHead

    Projection head selection (independent of above):
        use_tamo=True  →  DeepProjectionHead for pcol, scolw, it, tamo
        default        →  MLPProjectionHead  for pcol, scolw, it

    proj_hidden_dim: 0 (default) auto-sets to backbone.OUT_DIM.
    """
    if use_multi_tile:
        backbone = TiledBiomedCLIPBackbone(model_name=image_encoder_name, pretrained=pretrained)
    else:
        backbone = BiomedCLIPImageBackbone(model_name=image_encoder_name, pretrained=pretrained)
    feat_dim = backbone.OUT_DIM
    hidden_dim = proj_hidden_dim if proj_hidden_dim > 0 else feat_dim

    # ── Projection heads ──────────────────────────────────────────────────────
    if use_tamo:
        pcol_head = DeepProjectionHead(feat_dim, hidden_dim, proj_out_dim)
        scolw_head = DeepProjectionHead(feat_dim, hidden_dim, proj_out_dim)
        image_text_head = DeepProjectionHead(feat_dim, hidden_dim, proj_out_dim)
        tamo_head = DeepProjectionHead(feat_dim, hidden_dim, proj_out_dim)
    else:
        pcol_head = MLPProjectionHead(feat_dim, hidden_dim, proj_out_dim)
        scolw_head = MLPProjectionHead(feat_dim, hidden_dim, proj_out_dim)
        image_text_head = (
            MLPProjectionHead(feat_dim, hidden_dim, proj_out_dim)
            if use_image_text
            else None
        )
        tamo_head = None

    # ── Regression / ordinal / two-stage heads ────────────────────────────────
    if use_two_stage:
        # Stage 1: binary screening head (grade 0 vs 1+)
        reg_head = RegressionHead(feat_dim)
        # Stage 2: ordinal grading head (grades 1..K-1, mapped to 0..K-2 during loss)
        grading_head = CoralHead(feat_dim, n_classes=n_classes - 1)
    elif use_coral:
        reg_head = CoralHead(feat_dim, n_classes=n_classes)
        grading_head = None
    elif use_tamo:
        reg_head = DeepRegressionHead(feat_dim, hidden_dim=min(hidden_dim, 256))
        grading_head = None
    else:
        reg_head = RegressionHead(feat_dim)
        grading_head = None

    return HybridContrastiveOrdinalModel(
        backbone=backbone,
        pcol_head=pcol_head,
        scolw_head=scolw_head,
        regression_head=reg_head,
        image_text_head=image_text_head,
        tamo_head=tamo_head,
        grading_head=grading_head,
    )
