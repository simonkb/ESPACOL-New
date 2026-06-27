"""
BUSI dataset that ALSO returns the lesion mask, for the segmentation loss.

Your existing BUSIDataset deliberately skips every '*_mask*.png'. This variant
keeps the same (path, label) discovery but, for each image, finds its mask(s)
and returns (image, label, mask). BUSI masks live next to the image:

    benign (1).png
    benign (1)_mask.png        # primary
    benign (1)_mask_1.png      # a few images have multiple lesions -> union

Normal-class images have no lesion; they get an all-zero mask, which is correct
supervision (nothing to localize). Masks are resized to img_size with NEAREST
(no interpolation across the 0/1 boundary) and binarized.

Use this loader ONLY for the seg-loss training; the inherited ImageLabelDataset
is fine for everything else.
"""

import os
import glob
from typing import Callable, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD = [0.229, 0.224, 0.225]
CLASS_TO_LABEL = {"normal": 0, "benign": 1, "malignant": 2}


def _mask_paths_for(img_path: str) -> List[str]:
    """All mask files belonging to an image: '<stem>_mask*.png' beside it."""
    root, _ = os.path.splitext(img_path)
    return sorted(glob.glob(root + "_mask*"))


class BUSISegDataset(Dataset):
    def __init__(
        self,
        root_dir: str,
        items: Optional[List[Tuple[str, int]]] = None,
        img_size: int = 300,
        transform: Optional[Callable] = None,
    ):
        """
        root_dir : BUSI root (benign/ malignant/ normal/).
        items    : optional explicit (path,label) list (pass the fold's items so
                   train/test masks follow the same split as the grading model).
                   If None, discovers all non-mask images under root_dir.
        """
        self.img_size = img_size
        self.transform = transform or transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(_IMAGENET_MEAN, _IMAGENET_STD),
        ])
        self._mask_resize = transforms.Resize(
            (img_size, img_size), interpolation=transforms.InterpolationMode.NEAREST
        )

        if items is not None:
            self.items = list(items)
        else:
            self.items = []
            for cls, lab in CLASS_TO_LABEL.items():
                d = os.path.join(root_dir, cls)
                if not os.path.isdir(d):
                    continue
                for fn in os.listdir(d):
                    if "mask" in fn.lower():
                        continue
                    if fn.lower().endswith((".png", ".jpg", ".jpeg", ".bmp")):
                        self.items.append((os.path.join(d, fn), lab))

    def __len__(self) -> int:
        return len(self.items)

    def _load_mask(self, img_path: str) -> torch.Tensor:
        paths = _mask_paths_for(img_path)
        if not paths:
            # normal images (no lesion) -> empty mask
            return torch.zeros(1, self.img_size, self.img_size)
        acc = None
        for p in paths:
            m = Image.open(p).convert("L")
            m = self._mask_resize(m)
            arr = (np.array(m) > 127).astype(np.float32)   # binarize
            acc = arr if acc is None else np.maximum(acc, arr)  # union of lesions
        return torch.from_numpy(acc).unsqueeze(0)          # (1, H, W)

    def __getitem__(self, idx: int):
        img_path, label = self.items[idx]
        img = Image.open(img_path).convert("RGB")
        x = self.transform(img)
        mask = self._load_mask(img_path)
        y = torch.tensor(label, dtype=torch.long)
        return x, y, mask
