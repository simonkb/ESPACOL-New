"""
IDRiD dataset loaders for two tasks:

  1. Disease Grading (B. Disease Grading) — 5-fold stratified CV, 516 images.
  2. Segmentation eval (A. Segmentation) — inference only; returns images + lesion masks.

Expected root layout (Datasets/IDRiD/):
  A. Segmentation/
    1. Original Images/
      a. Training Set/*.jpg
      b. Testing Set/*.jpg
    2. All Segmentation Groundtruths/
      a. Training Set/
        1. Microaneurysms/*_MA.tif
        2. Haemorrhages/*_HE.tif
        3. Hard Exudates/*_EX.tif
        4. Soft Exudates/*_SE.tif
        5. Optic Disc/*_OD.tif
      b. Testing Set/  (same sub-structure)
  B. Disease Grading/
    1. Original Images/
      a. Training Set/*.jpg
      b. Testing Set/*.jpg
    2. Groundtruths/
      a. IDRiD_Disease Grading_Training Labels.csv
      b. IDRiD_Disease Grading_Testing Labels.csv

CSV columns: "Image name", "Retinopathy grade", "Risk of macular edema"
"""

from __future__ import annotations

import os
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from PIL import Image

import torch
from torch.utils.data import Dataset

from .dataloaders import (
    _pil_loader,
    build_transform,
    build_tile_transform,
)

# Mapping: grade -> list of lesion suffix keys present in that grade
# Used by the pointing game: for each grade, which mask types are relevant?
GRADE_TO_LESION_KEYS: Dict[int, List[str]] = {
    0: [],               # No apparent retinopathy
    1: ["MA"],           # Mild NPDR: microaneurysms only
    2: ["MA", "HE", "EX"],  # Moderate NPDR
    3: ["HE", "EX", "SE"],  # Severe NPDR
    4: ["HE", "EX", "SE"],  # PDR
}

# Lesion suffix -> subdirectory name under "2. All Segmentation Groundtruths"
LESION_DIR: Dict[str, str] = {
    "MA": "1. Microaneurysms",
    "HE": "2. Haemorrhages",
    "EX": "3. Hard Exudates",
    "SE": "4. Soft Exudates",
    "OD": "5. Optic Disc",
}


def load_all_idrid_items(idrid_root: str) -> List[Tuple[str, int]]:
    """
    Combine train + test grading images into one flat list.
    Returns list of (abs_image_path, grade) tuples.
    """
    grading_root = os.path.join(idrid_root, "B. Disease Grading")
    img_root = os.path.join(grading_root, "1. Original Images")
    gt_root = os.path.join(grading_root, "2. Groundtruths")

    splits = [
        (
            os.path.join(img_root, "a. Training Set"),
            os.path.join(gt_root, "a. IDRiD_Disease Grading_Training Labels.csv"),
        ),
        (
            os.path.join(img_root, "b. Testing Set"),
            os.path.join(gt_root, "b. IDRiD_Disease Grading_Testing Labels.csv"),
        ),
    ]

    items: List[Tuple[str, int]] = []
    for img_dir, csv_path in splits:
        if not os.path.isfile(csv_path):
            raise FileNotFoundError(f"IDRiD grading CSV not found: {csv_path}")
        df = pd.read_csv(csv_path)
        # Normalise column names (strip whitespace)
        df.columns = [c.strip() for c in df.columns]
        for _, row in df.iterrows():
            img_name = str(row["Image name"]).strip()
            grade = int(row["Retinopathy grade"])
            # Try with .jpg extension
            for ext in (".jpg", ".jpeg", ".JPG", ".png"):
                candidate = os.path.join(img_dir, img_name + ext)
                if os.path.isfile(candidate):
                    items.append((candidate, grade))
                    break
            else:
                # Try exact name
                candidate = os.path.join(img_dir, img_name)
                if os.path.isfile(candidate):
                    items.append((candidate, grade))
                else:
                    raise FileNotFoundError(
                        f"IDRiD image not found: {img_name} in {img_dir}"
                    )
    return items


class IDRiDGradingDataset(Dataset):
    """
    Minimal dataset wrapper for IDRiD grading items.
    Used with ImageLabelDataset in the CV loop — prefer that for cross-validation.
    This class is mainly for standalone use / sanity checks.
    """

    def __init__(
        self,
        items: List[Tuple[str, int]],
        transform: Optional[Callable] = None,
    ):
        self.items = items
        self.transform = transform or build_transform(300)

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        img_path, label = self.items[idx]
        img = _pil_loader(img_path, rgb=True)
        x = self.transform(img)
        y = torch.tensor(label, dtype=torch.long)
        return x, y


class IDRiDSegmentationDataset(Dataset):
    """
    Loads the 81 images that have pixel-level segmentation masks.
    Returns (image_tensor, label, image_id, masks_dict).

    masks_dict: {lesion_key: binary np.ndarray (H, W)} at original image
    resolution. Resize masks as needed in downstream eval.

    Lesion keys: MA, HE, EX, SE, OD.
    A mask is all-zeros (or absent) when that lesion is not annotated for a
    given image — check mask.any() before using.
    """

    def __init__(
        self,
        idrid_root: str,
        transform: Optional[Callable] = None,
        tile_transform: Optional[Callable] = None,
    ):
        seg_root = os.path.join(idrid_root, "A. Segmentation")
        img_root_seg = os.path.join(seg_root, "1. Original Images")
        gt_root_seg = os.path.join(seg_root, "2. All Segmentation Groundtruths")

        # Also need grading labels for the segmentation images
        grading_root = os.path.join(idrid_root, "B. Disease Grading")
        gt_root_grade = os.path.join(grading_root, "2. Groundtruths")

        grade_splits = {
            "a. Training Set": os.path.join(
                gt_root_grade,
                "a. IDRiD_Disease Grading_Training Labels.csv",
            ),
            "b. Testing Set": os.path.join(
                gt_root_grade,
                "b. IDRiD_Disease Grading_Testing Labels.csv",
            ),
        }

        # Build image_id -> grade mapping
        id_to_grade: Dict[str, int] = {}
        for _, csv_path in grade_splits.items():
            if not os.path.isfile(csv_path):
                continue
            df = pd.read_csv(csv_path)
            df.columns = [c.strip() for c in df.columns]
            for _, row in df.iterrows():
                img_id = str(row["Image name"]).strip()
                id_to_grade[img_id] = int(row["Retinopathy grade"])

        self.transform = transform or build_transform(300)
        self.tile_transform = tile_transform

        self.items: List[Tuple[str, int, str, str]] = []  # (img_path, grade, img_id, split_key)

        for split_key in ("a. Training Set", "b. Testing Set"):
            split_img_dir = os.path.join(img_root_seg, split_key)
            if not os.path.isdir(split_img_dir):
                continue
            for fname in sorted(os.listdir(split_img_dir)):
                if not fname.lower().endswith((".jpg", ".jpeg")):
                    continue
                img_id = os.path.splitext(fname)[0]  # e.g. "IDRiD_01"
                img_path = os.path.join(split_img_dir, fname)
                grade = id_to_grade.get(img_id, -1)
                self.items.append((img_path, grade, img_id, split_key))

        # Pre-build mask paths
        self._seg_gt_root = gt_root_seg
        self._lesion_keys = list(LESION_DIR.keys())

    def __len__(self) -> int:
        return len(self.items)

    def _load_mask(self, img_id: str, split_key: str, lesion_key: str) -> np.ndarray:
        """Load a .tif mask; return all-zeros if the file doesn't exist."""
        subdir = LESION_DIR[lesion_key]
        mask_fname = f"{img_id}_{lesion_key}.tif"
        mask_path = os.path.join(
            self._seg_gt_root, split_key, subdir, mask_fname
        )
        if not os.path.isfile(mask_path):
            return np.zeros((1, 1), dtype=np.uint8)  # sentinel — caller checks .any()
        with Image.open(mask_path) as m:
            arr = np.array(m.convert("L"))
        return (arr > 0).astype(np.uint8)

    def __getitem__(self, idx: int):
        img_path, grade, img_id, split_key = self.items[idx]
        img = _pil_loader(img_path, rgb=True)

        if self.tile_transform is not None:
            x = self.tile_transform(img)
        else:
            x = self.transform(img)

        masks = {
            k: self._load_mask(img_id, split_key, k)
            for k in self._lesion_keys
        }
        return x, grade, img_id, masks
