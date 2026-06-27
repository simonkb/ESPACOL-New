"""
Auxiliary segmentation loss (L_seg) - targets the diffuse-attention problem.

  L_seg = BCE(sigmoid(s_hat), M) + (1 - Dice(sigmoid(s_hat), M))

  s_hat   predicted mask logits from SegmentationHead, (N, 1, H, W)
  M       ground-truth lesion mask in {0,1}, (N, 1, H', W')   (BUSI ships these)

The two terms do different jobs and both are needed:
  - BCE is per-pixel correctness, but on small lesions it is almost satisfied by
    predicting all-background (the lesion is a tiny fraction of pixels), so BCE
    alone collapses to an empty mask.
  - Soft Dice scores overlap and is insensitive to the huge background, so it
    forces the head to actually cover the lesion.

DR has no masks, so this term applies to BUSI only. To evaluate retinal
localization later, use an external set with masks (IDRiD).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SegLoss(nn.Module):

    def __init__(self, dice_eps: float = 1.0):
        super().__init__()
        self.dice_eps = dice_eps
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, logits: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
        """
        logits: (N, 1, H, W) raw mask logits (no sigmoid applied).
        masks:  (N, 1, H', W') or (N, H', W') float/byte mask in {0,1}.
        Masks are resized to the logits' resolution if they differ.
        """
        if masks.dim() == 3:
            masks = masks.unsqueeze(1)
        masks = masks.float()
        if masks.shape[-2:] != logits.shape[-2:]:
            masks = F.interpolate(masks, size=logits.shape[-2:], mode="nearest")

        bce = self.bce(logits, masks)

        probs = torch.sigmoid(logits)
        # soft Dice over each image, then averaged
        dims = (1, 2, 3)
        inter = (probs * masks).sum(dims)
        union = probs.sum(dims) + masks.sum(dims)
        dice = (2 * inter + self.dice_eps) / (union + self.dice_eps)
        dice_loss = (1.0 - dice).mean()

        return bce + dice_loss
