"""
GPA Pointing Game + Tile IoU vs IDRiD segmentation masks.

For each image with a pixel-level lesion mask:
  1. Find the spatial tile with the highest GPA attention weight for the
     predicted grade (tile_weights[n, pred_grade, t] for t in 0..8 spatial).
  2. Map that tile index to image coordinates.
  3. Check whether the tile region overlaps with the relevant lesion mask.
  4. Also compare against GradCAM (not computed here — pass a GradCAM output
     file via --gradcam_npz if available).

Metrics reported:
  Pointing Game Accuracy  — fraction of images where top tile overlaps mask
  Mean Tile IoU           — intersection / union of tile region vs mask (tile-level binarised)

Grade → lesion key mapping (primary lesion per grade):
  0: skip (no lesions)
  1: MA
  2: HE + EX (union)
  3: HE + SE (union)
  4: HE (dominant in PDR)

Usage:
  python explainability/gpa_pointing_game.py \
      --idrid_root    Datasets/IDRiD \
      --inference_dir explainability/idrid_outputs \
      --output_csv    explainability/pointing_game_results.csv
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Datasets.idrid_loader import GRADE_TO_LESION_KEYS, LESION_DIR

TILE_GRID = 3
TILE_SIZE = 300  # tile size in the 900x900 canvas


def tile_to_bbox(tile_idx: int) -> Optional[Tuple[int, int, int, int]]:
    """
    Map tile index (0-8 spatial, 9 global) to (y1, x1, y2, x2) in the
    900x900 canvas. Returns None for the global tile (index 9).
    """
    if tile_idx >= TILE_GRID * TILE_GRID:
        return None  # global tile spans full image — not useful for pointing game
    row = tile_idx // TILE_GRID
    col = tile_idx % TILE_GRID
    y1, x1 = row * TILE_SIZE, col * TILE_SIZE
    return y1, x1, y1 + TILE_SIZE, x1 + TILE_SIZE


def load_mask_for_grade(
    idrid_root: str,
    img_id: str,
    grade: int,
    target_size: Tuple[int, int] = (900, 900),
) -> Optional[np.ndarray]:
    """
    Load and union the relevant lesion masks for a given grade.
    Returns binary (H, W) mask at target_size, or None if no masks available.
    """
    seg_root = os.path.join(idrid_root, "A. Segmentation", "2. All Segmentation Groundtruths")
    lesion_keys = GRADE_TO_LESION_KEYS.get(grade, [])
    if not lesion_keys:
        return None

    combined = np.zeros(target_size, dtype=np.uint8)
    found_any = False

    for split_key in ("a. Training Set", "b. Testing Set"):
        for lk in lesion_keys:
            subdir = LESION_DIR[lk]
            mask_path = os.path.join(seg_root, split_key, subdir, f"{img_id}_{lk}.tif")
            if os.path.isfile(mask_path):
                with Image.open(mask_path) as m:
                    arr = np.array(m.convert("L").resize(
                        (target_size[1], target_size[0]), Image.NEAREST
                    ))
                combined |= (arr > 0).astype(np.uint8)
                found_any = True

    return combined if found_any else None


def tile_iou(mask: np.ndarray, bbox: Tuple[int, int, int, int]) -> float:
    """Compute IoU between tile region and binary mask."""
    y1, x1, y2, x2 = bbox
    tile_mask = np.zeros_like(mask)
    tile_mask[y1:y2, x1:x2] = 1
    intersection = (mask & tile_mask).sum()
    union = (mask | tile_mask).sum()
    return float(intersection) / float(union) if union > 0 else 0.0


def tile_hits(mask: np.ndarray, bbox: Tuple[int, int, int, int]) -> bool:
    """True if any mask pixel falls inside the tile region."""
    y1, x1, y2, x2 = bbox
    return bool(mask[y1:y2, x1:x2].any())


def evaluate(args):
    all_files = sorted(f for f in os.listdir(args.inference_dir) if f.endswith("_outputs.npz"))
    if not all_files:
        raise FileNotFoundError(f"No fold_outputs.npz files in {args.inference_dir}")

    rows = []
    pg_hits, pg_total = 0, 0
    iou_vals: List[float] = []

    for fname in all_files:
        fold = fname.replace("_outputs.npz", "")
        data = np.load(os.path.join(args.inference_dir, fname), allow_pickle=True)
        img_ids = data["img_ids"]
        labels = data["labels"]
        pred_grades = data["pred_grades"]
        tile_weights = data["tile_weights"]   # (N, K, T)

        for i in range(len(img_ids)):
            img_id = str(img_ids[i])
            grade = int(labels[i])
            pred_g = int(pred_grades[i])
            use_grade = pred_g  # use predicted grade for pointing game (matching inference)

            if use_grade == 0:
                continue  # grade 0 has no lesions; skip

            mask = load_mask_for_grade(args.idrid_root, img_id, use_grade)
            if mask is None:
                continue  # no mask available for this image / grade

            # Highest-weight spatial tile for predicted grade
            tw = tile_weights[i, use_grade, :]  # (T,)
            spatial_tw = tw[:TILE_GRID * TILE_GRID]
            top_tile = int(np.argmax(spatial_tw))
            bbox = tile_to_bbox(top_tile)
            if bbox is None:
                continue

            hit = tile_hits(mask, bbox)
            iou = tile_iou(mask, bbox)

            pg_hits += int(hit)
            pg_total += 1
            iou_vals.append(iou)

            rows.append({
                "fold": fold,
                "img_id": img_id,
                "true_grade": grade,
                "pred_grade": pred_g,
                "top_tile": top_tile,
                "hit": int(hit),
                "iou": f"{iou:.4f}",
            })

    pg_acc = pg_hits / pg_total if pg_total > 0 else 0.0
    mean_iou = float(np.mean(iou_vals)) if iou_vals else 0.0

    print(f"\n=== GPA Pointing Game (N={pg_total}) ===")
    print(f"  Pointing Game Accuracy : {pg_acc * 100:.1f}%")
    print(f"  Mean Tile IoU          : {mean_iou:.4f}")

    if args.output_csv:
        os.makedirs(os.path.dirname(args.output_csv) or ".", exist_ok=True)
        with open(args.output_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"  Per-image results saved to {args.output_csv}")

    return pg_acc, mean_iou


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--idrid_root", default="Datasets/IDRiD")
    parser.add_argument("--inference_dir", default="explainability/idrid_outputs")
    parser.add_argument("--output_csv", default="explainability/pointing_game_results.csv")
    args = parser.parse_args()
    evaluate(args)
