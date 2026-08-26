"""
Qualitative figure: 4-panel comparison per grade.

For one representative IDRiD image per grade (grades 1-4):
  (a) Original fundus image
  (b) GPA tile heatmap for predicted grade (3x3 grid of attention weights)
  (c) IDRiD lesion mask overlay (union of relevant masks)
  (d) Tile concept score bar chart (top-3 concepts for top tile)

Outputs a high-res PDF panel saved to args.output_dir.

Usage:
  python explainability/make_qual_figure.py \
      --idrid_root    Datasets/IDRiD \
      --inference_dir explainability/idrid_outputs \
      --output_dir    explainability/figures
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Datasets.dataloaders import _pil_loader, crop_fundus_circle
from Datasets.idrid_loader import GRADE_TO_LESION_KEYS, LESION_DIR
from explainability.gpa_pointing_game import load_mask_for_grade

TILE_GRID = 3
TILE_SIZE = 300
CONCEPT_NAMES = [
    "Microaneurysms", "D/B Haemorrhages", "Hard Exudates",
    "Cotton Wool Spots", "Venous Beading", "IRMA",
    "Neovascularization", "Preretinal Haem.", "Vitreous Haem.",
]
GRADE_LABELS = {1: "Grade 1\n(Mild)", 2: "Grade 2\n(Moderate)",
                3: "Grade 3\n(Severe)", 4: "Grade 4\n(PDR)"}


def draw_tile_heatmap(ax, img: Image.Image, tile_weights_1d: np.ndarray, title: str):
    """Overlay 3x3 tile heatmap on fundus image."""
    ax.imshow(img)
    ax.set_title(title, fontsize=9, pad=3)
    ax.axis("off")

    spatial = tile_weights_1d[:TILE_GRID * TILE_GRID]
    vmin, vmax = spatial.min(), spatial.max()
    norm = Normalize(vmin=vmin, vmax=vmax)
    cmap = plt.cm.hot

    for t in range(TILE_GRID * TILE_GRID):
        row, col = divmod(t, TILE_GRID)
        # Normalise to fraction of display image size
        h, w = img.size[1], img.size[0]
        x = col * w / TILE_GRID
        y = row * h / TILE_GRID
        tw = w / TILE_GRID
        th = h / TILE_GRID
        color = cmap(norm(spatial[t]))
        rect = patches.Rectangle(
            (x, y), tw, th,
            linewidth=1, edgecolor="white",
            facecolor=color, alpha=0.45,
        )
        ax.add_patch(rect)


def draw_mask_overlay(ax, img: Image.Image, mask: Optional[np.ndarray], title: str):
    """Overlay lesion mask on fundus image."""
    ax.imshow(img)
    ax.set_title(title, fontsize=9, pad=3)
    ax.axis("off")
    if mask is not None and mask.any():
        mask_img = np.zeros((*mask.shape, 4), dtype=np.uint8)
        mask_img[mask > 0] = [255, 50, 50, 160]  # red, semi-transparent
        ax.imshow(mask_img)


def draw_concept_bar(ax, concept_scores_tile: np.ndarray, top_tile: int, title: str):
    """Bar chart of concept scores for the top-attention tile."""
    scores = concept_scores_tile  # (C,)
    top_k = 5
    top_idx = np.argsort(scores)[::-1][:top_k]
    names = [CONCEPT_NAMES[i][:14] for i in top_idx]
    vals = [scores[i] for i in top_idx]
    bars = ax.barh(names[::-1], vals[::-1], color="#E8704A")
    ax.set_xlim(0, 1.05)
    ax.set_title(f"{title}\n(tile {top_tile})", fontsize=9, pad=3)
    ax.tick_params(labelsize=7)
    ax.set_xlabel("Cosine similarity", fontsize=7)


def select_representative(
    img_ids: np.ndarray,
    labels: np.ndarray,
    pred_grades: np.ndarray,
    tile_weights: np.ndarray,
    tile_concept_scores: np.ndarray,
    target_grade: int,
    idrid_root: str,
) -> Optional[dict]:
    """Find first image where true_grade == target_grade and mask exists."""
    for i in range(len(img_ids)):
        if int(labels[i]) != target_grade:
            continue
        img_id = str(img_ids[i])
        mask = load_mask_for_grade(idrid_root, img_id, target_grade)
        if mask is None:
            continue

        # Find image path
        seg_img_root = os.path.join(idrid_root, "A. Segmentation", "1. Original Images")
        img_path = None
        for split_key in ("a. Training Set", "b. Testing Set"):
            for ext in (".jpg", ".jpeg"):
                p = os.path.join(seg_img_root, split_key, img_id + ext)
                if os.path.isfile(p):
                    img_path = p
                    break
            if img_path:
                break
        if img_path is None:
            continue

        pred_g = int(pred_grades[i])
        tw = tile_weights[i, pred_g, :]           # (T,)
        spatial_tw = tw[:TILE_GRID * TILE_GRID]
        top_tile = int(np.argmax(spatial_tw))
        tcs_top = tile_concept_scores[i, top_tile, :]  # (C,)

        return {
            "img_id": img_id,
            "img_path": img_path,
            "true_grade": int(labels[i]),
            "pred_grade": pred_g,
            "tile_weights": tw,
            "tcs_top_tile": tcs_top,
            "top_tile": top_tile,
            "mask": mask,
        }
    return None


def make_figure(args):
    os.makedirs(args.output_dir, exist_ok=True)

    # Load inference outputs from fold 0 (or first available)
    npz_files = sorted(f for f in os.listdir(args.inference_dir) if f.endswith("_outputs.npz"))
    if not npz_files:
        raise FileNotFoundError(f"No outputs in {args.inference_dir}")

    # Aggregate all folds for better representation
    all_ids, all_labels, all_preds, all_pred_grades, all_tw, all_tcs = [], [], [], [], [], []
    for fname in npz_files:
        d = np.load(os.path.join(args.inference_dir, fname), allow_pickle=True)
        all_ids.append(d["img_ids"])
        all_labels.append(d["labels"])
        all_preds.append(d["preds"])
        all_pred_grades.append(d["pred_grades"])
        all_tw.append(d["tile_weights"])
        all_tcs.append(d["tile_concept_scores"])

    img_ids = np.concatenate(all_ids)
    labels = np.concatenate(all_labels)
    preds = np.concatenate(all_preds)
    pred_grades = np.concatenate(all_pred_grades)
    tile_weights = np.concatenate(all_tw)
    tile_concept_scores = np.concatenate(all_tcs)

    target_grades = [1, 2, 3, 4]
    reps = {}
    for g in target_grades:
        rep = select_representative(
            img_ids, labels, pred_grades, tile_weights, tile_concept_scores, g, args.idrid_root
        )
        if rep is not None:
            reps[g] = rep
        else:
            print(f"Warning: no representative found for grade {g}")

    if not reps:
        print("No representatives found. Check inference outputs and IDRiD masks.")
        return

    n_grades = len(reps)
    fig, axes = plt.subplots(n_grades, 3, figsize=(12, 3.5 * n_grades))
    if n_grades == 1:
        axes = axes[np.newaxis, :]

    fig.suptitle("OPTIC-C Spatial Explanations vs IDRiD Lesion Masks", fontsize=12, y=1.01)

    for row_idx, g in enumerate(sorted(reps.keys())):
        rep = reps[g]
        img = _pil_loader(rep["img_path"], rgb=True)
        img_900 = img.resize((900, 900), Image.BILINEAR)

        grade_label = GRADE_LABELS.get(g, f"Grade {g}")
        ax_row = axes[row_idx]

        # Panel (a): original
        ax_row[0].imshow(img_900)
        ax_row[0].set_title(f"{grade_label}\nOriginal fundus", fontsize=9, pad=3)
        ax_row[0].axis("off")
        if row_idx == 0:
            ax_row[0].set_title(f"(a) Fundus\n{grade_label}", fontsize=9, pad=3)

        # Panel (b): GPA heatmap
        draw_tile_heatmap(ax_row[1], img_900, rep["tile_weights"], f"(b) GPA tile weights\n(predicted grade {rep['pred_grade']})")

        # Panel (c): mask overlay
        draw_mask_overlay(ax_row[2], img_900, rep["mask"], f"(c) IDRiD lesion mask\n({'/'.join(GRADE_TO_LESION_KEYS.get(g, ['none']))})")

    plt.tight_layout()

    out_pdf = os.path.join(args.output_dir, "optic_c_qualitative.pdf")
    out_png = os.path.join(args.output_dir, "optic_c_qualitative.png")
    fig.savefig(out_pdf, bbox_inches="tight", dpi=300)
    fig.savefig(out_png, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"Figure saved to {out_pdf}")
    print(f"Figure saved to {out_png}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--idrid_root", default="Datasets/IDRiD")
    parser.add_argument("--inference_dir", default="explainability/idrid_outputs")
    parser.add_argument("--output_dir", default="explainability/figures")
    args = parser.parse_args()
    make_figure(args)
