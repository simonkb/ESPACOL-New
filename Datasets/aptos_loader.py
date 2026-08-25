"""
APTOS 2019 Blindness Detection dataset loader — 5-fold stratified CV.

Expected root layout (Datasets/aptos2019-blindness-detection/):
  train.csv           — columns: id_code, diagnosis (0-4)
  train_images/
    *.png

3,662 training images. Test labels are not public; all CV is on the training set.
"""

from __future__ import annotations

import os
from typing import List, Tuple

import pandas as pd


def load_all_aptos_items(aptos_root: str) -> List[Tuple[str, int]]:
    """
    Returns list of (abs_image_path, grade) tuples from the training set.
    """
    csv_path = os.path.join(aptos_root, "train.csv")
    img_dir = os.path.join(aptos_root, "train_images")

    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"APTOS train.csv not found: {csv_path}")
    if not os.path.isdir(img_dir):
        raise FileNotFoundError(f"APTOS train_images/ not found: {img_dir}")

    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]

    items: List[Tuple[str, int]] = []
    for _, row in df.iterrows():
        img_id = str(row["id_code"]).strip()
        grade = int(row["diagnosis"])
        img_path = os.path.join(img_dir, img_id + ".png")
        if not os.path.isfile(img_path):
            raise FileNotFoundError(f"APTOS image not found: {img_path}")
        items.append((img_path, grade))

    return items
