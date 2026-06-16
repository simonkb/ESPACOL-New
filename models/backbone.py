"""
Image backbone encoders.

BiomedCLIPImageBackbone (default):
  BiomedCLIP ViT-B/16 image encoder from the same pretrained model used
  for the text branch.  Input: 224×224, CLIP normalization.
  Output dim is probed dynamically at init (same pattern as ClinicalTextEncoder).

TiledBiomedCLIPBackbone:
  Wraps BiomedCLIPImageBackbone to process several high-resolution crops
  ("tiles") of one image instead of a single globally-downsampled view.
  Native DR fundus photos are ~2600-4900px; resizing the whole image to
  224x224 destroys lesion-scale detail (microaneurysms are only a handful
  of pixels even at native resolution — the resize filter averages them
  away before training starts).  Tiling preserves that detail in at least
  some crops; attention pooling aggregates the per-tile features into one
  vector so heads/losses downstream are unchanged.

EfficientNetV2SBackbone (kept for reference / BUSI comparisons):
  EfficientNet-V2S with classifier removed. 300×300, ImageNet normalization.
  Returns 1280-dim GAP features.
"""

import torch
import torch.nn as nn
from torchvision.models import efficientnet_v2_s, EfficientNet_V2_S_Weights

from .heads import AttentionPool


class BiomedCLIPImageBackbone(nn.Module):
    """
    BiomedCLIP ViT-B/16 image encoder as a drop-in backbone.

    Uses open_clip.create_model_and_transforms with the same model name as
    ClinicalTextEncoder so both branches share the same pretrained weights at
    initialization and can diverge during fine-tuning.

    Returns un-normalized projected CLIP image features (normalize=False),
    consistent with EfficientNet's un-normalized GAP output so projection
    heads behave the same way.
    """

    def __init__(
        self,
        model_name: str = "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224",
        pretrained: bool = True,
    ):
        super().__init__()
        try:
            import open_clip
        except ImportError as exc:
            raise ImportError(
                "BiomedCLIPImageBackbone requires open_clip_torch. "
                "Install it with: pip install open_clip_torch"
            ) from exc

        model, _, _ = open_clip.create_model_and_transforms(model_name)
        # Store only the visual encoder so the text half of the CLIP model is
        # not included in parameters() and not wastefully trained.
        self._visual = model.visual

        # Probe output dimension with a dummy forward pass (same pattern used
        # by ClinicalTextEncoder for the text branch).
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 224, 224)
            out = self._visual(dummy)
            self.OUT_DIM: int = out.shape[-1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._visual(x)   # (N, OUT_DIM), un-normalized


class TiledBiomedCLIPBackbone(nn.Module):
    """
    Runs BiomedCLIPImageBackbone on T tiles per image (shared weights across
    tiles) and aggregates with AttentionPool into a single feature vector.

    Input : (N, T, C, H, W) — T high-resolution crops per image.
    Output: (N, OUT_DIM) — pooled feature, same shape contract as
            BiomedCLIPImageBackbone, so heads/losses need no changes.
    """

    def __init__(
        self,
        model_name: str = "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224",
        pretrained: bool = True,
    ):
        super().__init__()
        self.encoder = BiomedCLIPImageBackbone(model_name=model_name, pretrained=pretrained)
        self.OUT_DIM: int = self.encoder.OUT_DIM
        self.pool = AttentionPool(self.OUT_DIM)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        N, T, C, H, W = x.shape
        feats = self.encoder(x.reshape(N * T, C, H, W))   # (N*T, OUT_DIM)
        feats = feats.view(N, T, -1)                      # (N, T, OUT_DIM)
        return self.pool(feats)                            # (N, OUT_DIM)


class EfficientNetV2SBackbone(nn.Module):
    """
    EfficientNet-V2S with classifier removed.
    Returns 1280-dim GAP features.
    """

    OUT_DIM = 1280

    def __init__(self, pretrained: bool = True):
        super().__init__()
        weights = EfficientNet_V2_S_Weights.DEFAULT if pretrained else None
        base = efficientnet_v2_s(weights=weights)
        self.features = base.features
        self.avgpool = base.avgpool   # AdaptiveAvgPool2d -> (N, 1280, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        return torch.flatten(x, 1)   # (N, 1280)
