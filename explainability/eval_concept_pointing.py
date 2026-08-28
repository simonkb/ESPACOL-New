"""
Concept Pointing Game on IDRiD segmentation images (zero-shot).

For each IDRiD segmentation image × concept pair where a pixel mask exists,
checks whether the tile with the highest concept score overlaps the lesion mask.

Four DR concepts are evaluable against IDRiD pixel masks:
  Concept 0  microaneurysms              → IDRiD MA mask
  Concept 1  dot and blot hemorrhages    → IDRiD HE mask
  Concept 2  hard exudates               → IDRiD EX mask
  Concept 3  cotton wool spots           → IDRiD SE mask (soft exudates)

The remaining 5 concepts (venous beading, IRMA, neovascularisation,
preretinal/vitreous haemorrhage) have no IDRiD pixel-level annotation
and are excluded from the quantitative evaluation.

Usage (run from project root):
  python explainability/eval_concept_pointing.py \\
      --npz         explainability/idrid_outputs/official_outputs.npz \\
      --idrid_root  Datasets/IDRiD \\
      --tile_grid   3 \\
      --tile_size   300

Output: prints a table of per-concept and overall Pointing Game accuracy,
        and writes a JSON summary to <npz_dir>/concept_pointing_results.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ── Constants ─────────────────────────────────────────────────────────────────

CONCEPT_NAMES: List[str] = [
    "microaneurysms",
    "dot and blot hemorrhages",
    "hard exudates",
    "cotton wool spots",
    "venous beading",
    "intraretinal microvascular abnormalities",
    "neovascularization",
    "preretinal hemorrhage",
    "vitreous hemorrhage",
]

# Only concepts with IDRiD pixel masks can be evaluated
CONCEPT_TO_MASK_KEY: Dict[int, str] = {
    0: "MA",
    1: "HE",
    2: "EX",
    3: "SE",
}

MASK_SUBDIR: Dict[str, str] = {
    "MA": "1. Microaneurysms",
    "HE": "2. Haemorrhages",
    "EX": "3. Hard Exudates",
    "SE": "4. Soft Exudates",
}

# IDRiD segmentation split: IDs 01-54 = training, 55-81 = testing
def _img_id_to_seg_split(img_id: str) -> str:
    num = int(img_id.split("_")[-1])
    return "a. Training Set" if num <= 54 else "b. Testing Set"


# ── Mask loading ──────────────────────────────────────────────────────────────

def load_mask(idrid_root: str, img_id: str, mask_key: str) -> np.ndarray:
    """
    Load a binary IDRiD lesion mask at original image resolution.
    Returns all-zeros array if the mask file is absent (lesion not annotated).
    """
    split = _img_id_to_seg_split(img_id)
    mask_path = os.path.join(
        idrid_root,
        "A. Segmentation",
        "2. All Segmentation Groundtruths",
        split,
        MASK_SUBDIR[mask_key],
        f"{img_id}_{mask_key}.tif",
    )
    if not os.path.isfile(mask_path):
        return np.zeros((1, 1), dtype=np.uint8)
    with Image.open(mask_path) as m:
        arr = np.array(m.convert("L"))
    return (arr > 0).astype(np.uint8)


# ── Pointing game ─────────────────────────────────────────────────────────────

def pointing_game_hit(
    tile_scores: np.ndarray,   # (tile_grid²,) — spatial tiles only
    mask: np.ndarray,          # (H, W) binary, original image resolution
    tile_grid: int,
    tile_size: int,
) -> bool:
    """
    True if the tile with the highest concept score overlaps the lesion mask.

    The tile grid covers a square of side tile_grid × tile_size pixels.
    The mask is resized to this square via nearest-neighbour interpolation
    (preserves lesion presence; a lesion crossing a tile boundary counts
    for both tiles it overlaps).
    """
    t_star = int(np.argmax(tile_scores))
    row = t_star // tile_grid
    col = t_star % tile_grid

    grid_px = tile_grid * tile_size
    mask_pil = Image.fromarray(mask * 255)
    mask_small = np.array(
        mask_pil.resize((grid_px, grid_px), Image.NEAREST)
    ) > 0

    r0, r1 = row * tile_size, (row + 1) * tile_size
    c0, c1 = col * tile_size, (col + 1) * tile_size
    return bool(mask_small[r0:r1, c0:c1].any())


# ── Main evaluation ───────────────────────────────────────────────────────────

def evaluate(args) -> dict:
    if not os.path.isfile(args.npz):
        raise FileNotFoundError(
            f"Inference outputs not found: {args.npz}\n"
            "Run inference_idrid.py --seg_split all first."
        )

    data = np.load(args.npz, allow_pickle=True)
    img_ids      = data["img_ids"]             # (N,) str
    tcs          = data["tile_concept_scores"] # (N, T, C)
    pred_grades  = data["pred_grades"]         # (N,)
    labels       = data["labels"]              # (N,)

    N, T, C = tcs.shape
    n_spatial = args.tile_grid ** 2            # 9 for 3×3 grid

    print(f"\nLoaded {N} images, {T} tiles, {C} concepts from {args.npz}")

    # Per-concept results: list of {"hit": bool, "img_id": str, "grade": int}
    per_concept: Dict[int, List[dict]] = {c: [] for c in CONCEPT_TO_MASK_KEY}

    for i, img_id in enumerate(img_ids):
        spatial_scores = tcs[i, :n_spatial, :]   # (9, C)

        for c_idx, mask_key in CONCEPT_TO_MASK_KEY.items():
            mask = load_mask(args.idrid_root, img_id, mask_key)
            if not mask.any():
                continue   # lesion absent or not annotated for this image

            hit = pointing_game_hit(
                tile_scores=spatial_scores[:, c_idx],
                mask=mask,
                tile_grid=args.tile_grid,
                tile_size=args.tile_size,
            )
            per_concept[c_idx].append({
                "hit":   hit,
                "img_id": img_id,
                "true_grade": int(labels[i]),
                "pred_grade": int(pred_grades[i]),
            })

    # ── Print results table ────────────────────────────────────────────────
    w = 62
    print("\n" + "=" * w)
    print("  OPTIC-C  Concept Pointing Game  (IDRiD, zero-shot)")
    print("=" * w)
    print(f"  {'Concept':<42} {'N':>4}  {'Hits':>4}  {'PG %':>6}")
    print("-" * w)

    total_hits, total_n = 0, 0
    summary: Dict[str, dict] = {}

    for c_idx, mask_key in CONCEPT_TO_MASK_KEY.items():
        rows = per_concept[c_idx]
        n    = len(rows)
        hits = sum(r["hit"] for r in rows)
        pg   = 100.0 * hits / n if n > 0 else 0.0
        total_hits += hits
        total_n    += n
        name = CONCEPT_NAMES[c_idx]
        print(f"  {name:<42} {n:>4}  {hits:>4}  {pg:>5.1f}%")
        summary[name] = {"mask_key": mask_key, "n": n, "hits": hits, "pg": round(pg, 2)}

    print("-" * w)
    overall = 100.0 * total_hits / total_n if total_n > 0 else 0.0
    print(f"  {'Overall (all evaluable concept-image pairs)':<42} {total_n:>4}  "
          f"{total_hits:>4}  {overall:>5.1f}%")
    print("=" * w)

    # ── Per-grade breakdown ────────────────────────────────────────────────
    print("\n  Per-true-grade breakdown (averaged over evaluable concepts):")
    grade_summary: Dict[int, dict] = {}
    for grade in range(5):
        g_hits, g_n = 0, 0
        for c_idx in CONCEPT_TO_MASK_KEY:
            for r in per_concept[c_idx]:
                if r["true_grade"] == grade:
                    g_hits += r["hit"]
                    g_n    += 1
        if g_n > 0:
            pg_g = 100.0 * g_hits / g_n
            print(f"    Grade {grade}: {g_hits}/{g_n} = {pg_g:.1f}%")
            grade_summary[grade] = {"hits": g_hits, "n": g_n, "pg": round(pg_g, 2)}

    # ── Save JSON ──────────────────────────────────────────────────────────
    out_dir = os.path.dirname(args.npz)
    json_path = os.path.join(out_dir, "concept_pointing_results.json")
    result = {
        "overall_pg":   round(overall, 2),
        "total_n":      total_n,
        "total_hits":   total_hits,
        "per_concept":  summary,
        "per_grade":    {str(k): v for k, v in grade_summary.items()},
        "n_images":     int(N),
    }
    with open(json_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\n  Results saved → {json_path}\n")

    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Concept Pointing Game on IDRiD (zero-shot)")
    parser.add_argument("--npz",
        default="explainability/idrid_outputs/official_outputs.npz",
        help="Path to official_outputs.npz from inference_idrid.py")
    parser.add_argument("--idrid_root",
        default="Datasets/IDRiD",
        help="Root directory of the IDRiD dataset")
    parser.add_argument("--tile_grid", type=int, default=3,
        help="Tile grid size (default 3 → 3×3)")
    parser.add_argument("--tile_size", type=int, default=300,
        help="Tile size in pixels (default 300)")
    args = parser.parse_args()
    evaluate(args)
