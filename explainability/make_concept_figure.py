"""
Qualitative concept heatmap figure for OPTIC-C CGPM validation.

For each DR grade (1–4), selects one representative IDRiD image and
renders a three-panel row:
  (a) Original fundus with 3×3 tile grid overlay
  (b) Tile heatmap for the grade-primary concept (e.g. microaneurysms for G1)
  (c) IDRiD ground-truth lesion mask (green) — where available

The figure visually validates that concept scores are:
  - Grade-appropriate (right concept lights up for each grade)
  - Spatially correct (high-scoring tile overlaps the lesion mask)

Usage (run from project root):
  python explainability/make_concept_figure.py \\
      --npz         explainability/idrid_outputs/official_outputs.npz \\
      --idrid_root  Datasets/IDRiD \\
      --out         paper/optic_c_concept_qual.png \\
      --dpi         300

Output: paper/optic_c_concept_qual.png
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ── Constants ─────────────────────────────────────────────────────────────────

CONCEPT_NAMES: List[str] = [
    "Microaneurysms",
    "Dot/Blot Haemorrhages",
    "Hard Exudates",
    "Cotton Wool Spots",
    "Venous Beading",
    "IRMA",
    "Neovascularisation",
    "Preretinal Haemorrhage",
    "Vitreous Haemorrhage",
]

MASK_SUBDIR: Dict[str, str] = {
    "MA": "1. Microaneurysms",
    "HE": "2. Haemorrhages",
    "EX": "3. Hard Exudates",
    "SE": "4. Soft Exudates",
}

# Per-grade: (primary concept index, mask key for that concept or None)
GRADE_PRIMARY: Dict[int, Tuple[int, Optional[str]]] = {
    1: (0, "MA"),   # microaneurysms — has mask
    2: (2, "EX"),   # hard exudates — has mask
    3: (1, "HE"),   # dot/blot haemorrhages — has mask
    4: (6, None),   # neovascularisation — no IDRiD mask
}

GRADE_LABELS = {
    0: "No DR",
    1: "Mild NPDR",
    2: "Moderate NPDR",
    3: "Severe NPDR",
    4: "Proliferative DR",
}

PANEL_LETTERS = ["(a)", "(b)", "(c)"]

# Design colours (matching paper palette)
PBLUE   = "#2B61A8"
DTXT    = "#1F2933"
MTXT    = "#667085"
GOLD    = "#D4A017"
TILE_EDGE = "white"

FONT_TITLE  = 18
FONT_LABEL  = 14
FONT_SMALL  = 11


# ── Helper utilities ──────────────────────────────────────────────────────────

def _img_id_to_seg_split(img_id: str) -> str:
    num = int(img_id.split("_")[-1])
    return "a. Training Set" if num <= 54 else "b. Testing Set"


def load_original_image(idrid_root: str, img_id: str) -> Optional[Image.Image]:
    split = _img_id_to_seg_split(img_id)
    path = os.path.join(
        idrid_root, "A. Segmentation", "1. Original Images", split,
        img_id + ".jpg",
    )
    if not os.path.isfile(path):
        return None
    return Image.open(path).convert("RGB")


def load_mask(idrid_root: str, img_id: str, mask_key: str) -> Optional[np.ndarray]:
    split = _img_id_to_seg_split(img_id)
    path = os.path.join(
        idrid_root, "A. Segmentation", "2. All Segmentation Groundtruths",
        split, MASK_SUBDIR[mask_key], f"{img_id}_{mask_key}.tif",
    )
    if not os.path.isfile(path):
        return None
    with Image.open(path) as m:
        arr = np.array(m.convert("L"))
    return (arr > 0).astype(np.uint8)


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x * 10.0, -20, 20)))


def select_best_image(
    img_ids: np.ndarray,
    labels: np.ndarray,
    pred_grades: np.ndarray,
    tcs: np.ndarray,
    target_grade: int,
    concept_idx: int,
    n_spatial: int,
    idrid_root: str,
) -> Optional[str]:
    """
    Among images with the target grade, pick the one where the max-tile
    concept score is highest and the image file is accessible.

    Preference order:
      1. True labels == target_grade  (if any labels != -1)
      2. Predicted grades == target_grade  (fallback when labels are all -1,
         which happens when segmentation IDs don't match the grading CSV)
    """
    # Try true labels first; fall back to predicted grades
    candidate_idxs = np.where(labels == target_grade)[0]
    if len(candidate_idxs) == 0:
        candidate_idxs = np.where(pred_grades == target_grade)[0]
    if len(candidate_idxs) == 0:
        return None

    scores = []
    for ci in candidate_idxs:
        s = tcs[ci, :n_spatial, concept_idx]
        scores.append(float(np.max(sigmoid(s))))

    order = np.argsort(scores)[::-1]
    for rank in order:
        img_id = str(img_ids[candidate_idxs[rank]])
        if load_original_image(idrid_root, img_id) is not None:
            return img_id
    return None


# ── Drawing helpers ───────────────────────────────────────────────────────────

def draw_original(ax, img: Image.Image, tile_grid: int, tile_size: int,
                  grade: int) -> None:
    img_sq = img.resize((tile_grid * tile_size, tile_grid * tile_size), Image.LANCZOS)
    ax.imshow(np.array(img_sq))

    # Tile grid lines
    for k in range(1, tile_grid):
        ax.axhline(k * tile_size, color=TILE_EDGE, linewidth=0.8, alpha=0.7)
        ax.axvline(k * tile_size, color=TILE_EDGE, linewidth=0.8, alpha=0.7)

    ax.set_title(
        f"{PANEL_LETTERS[0]}  Grade {grade} — {GRADE_LABELS[grade]}\nOriginal fundus",
        fontsize=FONT_TITLE, color=DTXT, pad=6,
    )
    ax.axis("off")


def draw_concept_heatmap(
    ax, img: Image.Image, scores_9: np.ndarray,
    concept_name: str, tile_grid: int, tile_size: int,
) -> None:
    """Tile heatmap: grayscale image + coloured tile overlay + gold max-tile border."""
    grid_px = tile_grid * tile_size
    img_sq = img.resize((grid_px, grid_px), Image.LANCZOS).convert("L")
    img_gray = np.array(img_sq)

    ax.imshow(img_gray, cmap="gray")

    probs = sigmoid(scores_9.reshape(tile_grid, tile_grid))
    norm = (probs - probs.min()) / (probs.max() - probs.min() + 1e-8)

    cmap = plt.cm.Blues
    for r in range(tile_grid):
        for c in range(tile_grid):
            colour = list(cmap(0.15 + 0.85 * float(norm[r, c])))
            colour[3] = 0.18 + 0.50 * float(norm[r, c])   # variable alpha
            rect = mpatches.FancyBboxPatch(
                (c * tile_size, r * tile_size),
                tile_size, tile_size,
                boxstyle="square,pad=0",
                facecolor=colour, edgecolor=TILE_EDGE, linewidth=0.6,
            )
            ax.add_patch(rect)

    # Gold border on max tile
    t_max = int(np.argmax(scores_9))
    mr, mc = t_max // tile_grid, t_max % tile_grid
    rect = mpatches.FancyBboxPatch(
        (mc * tile_size, mr * tile_size),
        tile_size, tile_size,
        boxstyle="square,pad=0",
        facecolor="none", edgecolor=GOLD, linewidth=3,
    )
    ax.add_patch(rect)

    ax.set_xlim(0, grid_px)
    ax.set_ylim(grid_px, 0)
    ax.set_title(
        f"{PANEL_LETTERS[1]}  Concept: {concept_name}\nTile activation (gold = max tile)",
        fontsize=FONT_TITLE, color=DTXT, pad=6,
    )
    ax.axis("off")


def draw_mask_panel(
    ax, img: Image.Image, mask: Optional[np.ndarray],
    mask_key: str, concept_name: str, tile_grid: int, tile_size: int,
) -> None:
    grid_px = tile_grid * tile_size
    img_sq = img.resize((grid_px, grid_px), Image.LANCZOS)
    ax.imshow(np.array(img_sq))

    if mask is not None and mask.any():
        # Resize mask to grid_px × grid_px
        mask_pil = Image.fromarray(mask * 255)
        mask_small = np.array(
            mask_pil.resize((grid_px, grid_px), Image.NEAREST)
        ) > 0

        overlay = np.zeros((grid_px, grid_px, 4), dtype=np.uint8)
        overlay[mask_small, 0] = 0
        overlay[mask_small, 1] = 200
        overlay[mask_small, 2] = 100
        overlay[mask_small, 3] = 180
        ax.imshow(overlay)
        title_suffix = f"IDRiD {mask_key} mask (green)"
    else:
        title_suffix = "No IDRiD mask available"

    # Tile grid lines
    for k in range(1, tile_grid):
        ax.axhline(k * tile_size, color=TILE_EDGE, linewidth=0.8, alpha=0.7)
        ax.axvline(k * tile_size, color=TILE_EDGE, linewidth=0.8, alpha=0.7)

    ax.set_title(
        f"{PANEL_LETTERS[2]}  Ground truth\n{title_suffix}",
        fontsize=FONT_TITLE, color=DTXT, pad=6,
    )
    ax.axis("off")


# ── Main ──────────────────────────────────────────────────────────────────────

def make_figure(args) -> None:
    if not os.path.isfile(args.npz):
        raise FileNotFoundError(
            f"Inference outputs not found: {args.npz}\n"
            "Run inference_idrid.py --seg_split all first."
        )

    data = np.load(args.npz, allow_pickle=True)
    img_ids     = np.array([str(x) for x in data["img_ids"]])   # (N,) clean Python str
    tcs         = data["tile_concept_scores"]                    # (N, T, C)
    labels      = data["labels"]                                 # (N,) — may be all -1
    pred_grades = data["pred_grades"]                            # (N,) — always valid

    n_spatial = args.tile_grid ** 2
    grades_to_show = [1, 2, 3, 4]
    n_grades = len(grades_to_show)

    fig, axes = plt.subplots(
        n_grades, 3,
        figsize=(16, 5.5 * n_grades),
        layout="constrained",
    )
    if n_grades == 1:
        axes = axes[np.newaxis, :]

    for row_idx, grade in enumerate(grades_to_show):
        c_idx, mask_key = GRADE_PRIMARY[grade]
        concept_name = CONCEPT_NAMES[c_idx]

        img_id = select_best_image(
            img_ids, labels, pred_grades, tcs, grade, c_idx, n_spatial, args.idrid_root
        )
        if img_id is None:
            print(f"  Warning: no image found for grade {grade} — skipping row")
            for ax in axes[row_idx]:
                ax.axis("off")
            continue

        print(f"  Grade {grade}: using {img_id}  (concept: {concept_name})")

        img = load_original_image(args.idrid_root, img_id)
        scores_all = tcs[np.where(img_ids == img_id)[0][0], :n_spatial, :]   # (9, C)
        scores_concept = scores_all[:, c_idx]   # (9,)

        mask = None
        if mask_key is not None:
            mask = load_mask(args.idrid_root, img_id, mask_key)

        draw_original(axes[row_idx, 0], img, args.tile_grid, args.tile_size, grade)
        draw_concept_heatmap(axes[row_idx, 1], img, scores_concept,
                             concept_name, args.tile_grid, args.tile_size)
        draw_mask_panel(axes[row_idx, 2], img, mask, mask_key or "—",
                        concept_name, args.tile_grid, args.tile_size)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    fig.savefig(args.out, dpi=args.dpi, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    print(f"\n  Concept figure saved → {args.out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Qualitative concept heatmap figure for OPTIC-C CGPM")
    parser.add_argument("--npz",
        default="explainability/idrid_outputs/official_outputs.npz")
    parser.add_argument("--idrid_root", default="Datasets/IDRiD")
    parser.add_argument("--out",        default="paper/optic_c_concept_qual.png")
    parser.add_argument("--tile_grid",  type=int, default=3)
    parser.add_argument("--tile_size",  type=int, default=300)
    parser.add_argument("--dpi",        type=int, default=300)
    args = parser.parse_args()
    make_figure(args)
