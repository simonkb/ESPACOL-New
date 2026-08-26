"""
Faithfulness test: tile occlusion.

For each IDRiD segmentation image:
  1. Find the spatial tile with the highest GPA weight for the predicted grade.
  2. Zero that tile (replace with ImageNet mean pixel), re-run inference.
  3. Record whether the prediction changes and by how much.

Metrics:
  Accuracy drop     — (acc with occlusion) vs (acc without occlusion)
  Mean grade shift  — average absolute change in predicted grade after occlusion
  Faithfulness      — fraction of images where prediction changes after top-tile occlusion

Usage:
  python explainability/tile_occlusion.py \
      --idrid_root    Datasets/IDRiD \
      --model_dir     runs/optic_concept_idrid/official \
      --inference_dir explainability/idrid_outputs \
      --output_csv    explainability/tile_occlusion_results.csv
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from typing import List

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Datasets.dataloaders import build_tile_transform
from Datasets.idrid_loader import IDRiDSegmentationDataset
from models.framework import build_model
from configs.config import DRConfig

TILE_GRID = 3
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)


def occlude_tile(x: torch.Tensor, tile_idx: int) -> torch.Tensor:
    """
    x: (T, C, H, W) — tile tensor for one image.
    Zero the specified tile (replace with ImageNet mean, i.e. normalized zero).
    Returns a copy.
    """
    x = x.clone()
    x[tile_idx] = 0.0  # ImageNet-normalized zero ≈ mean pixel
    return x


def evaluate(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load inference outputs (for pre-computed tile_weights)
    npz_files = sorted(f for f in os.listdir(args.inference_dir) if f.endswith("_outputs.npz"))
    if not npz_files:
        raise FileNotFoundError(f"No outputs in {args.inference_dir}")

    # Aggregate tile_weights and pred_grades from all npz files
    img_id_to_tw: dict = {}
    img_id_to_pred: dict = {}
    for npz_fname in npz_files:
        data = np.load(os.path.join(args.inference_dir, npz_fname), allow_pickle=True)
        for i in range(len(data["img_ids"])):
            img_id_to_tw[str(data["img_ids"][i])] = data["tile_weights"][i]
            img_id_to_pred[str(data["img_ids"][i])] = int(data["pred_grades"][i])

    # Load the single model — find whichever fold*_best.pth exists
    import glob as _glob
    candidates = sorted(_glob.glob(os.path.join(args.model_dir, "fold*_best.pth")))
    if candidates:
        ckpt_path = candidates[0]
    else:
        ckpt_path = os.path.join(args.model_dir, "best_model.pt")
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"No checkpoint found in {args.model_dir}")
    print(f"Loading checkpoint: {ckpt_path}")
    cfg = DRConfig()
    model = build_model(
        n_classes=cfg.n_classes,
        pretrained=False,
        use_multi_tile=True,
        tile_grid=3,
        use_tile_transformer=True,
        use_grade_prototypes=True,
        use_ordinal_head=True,
        use_concept_prototype=True,
        proto_temperature=cfg.proto_temperature,
    )
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()

    # Use same split as inference to avoid leakage
    tile_tfm = build_tile_transform(tile_size=300, tile_grid=3, augment=False)
    ds = IDRiDSegmentationDataset(
        idrid_root=args.idrid_root,
        tile_transform=tile_tfm,
        split=args.seg_split,
    )

    rows: List[dict] = []
    correct_orig, correct_occ, n_total = 0, 0, 0
    grade_shifts: List[float] = []
    faithfulness_hits = 0

    from Datasets.dataloaders import _pil_loader

    with torch.no_grad():
        for img_path, grade, img_id, _ in ds.items:
            if img_id not in img_id_to_tw:
                continue

            tw = img_id_to_tw[img_id]   # (K, T)
            pred_g = img_id_to_pred[img_id]

            img = _pil_loader(img_path, rgb=True)
            x = tile_tfm(img).unsqueeze(0).to(device)  # (1, T, C, H, W)

            # Highest-weight spatial tile for predicted grade
            spatial_tw = tw[pred_g, :TILE_GRID * TILE_GRID]
            top_tile = int(np.argmax(spatial_tw))

            # Occlude top tile (replace with ImageNet mean = normalized zero)
            x_occ = x.clone()
            x_occ[0, top_tile] = 0.0

            pred_orig = model(x)["pred"].item()
            pred_occ = model(x_occ)["pred"].item()

            grade_orig = int(np.clip(round(pred_orig), 0, 4))
            grade_occ = int(np.clip(round(pred_occ), 0, 4))

            shift = abs(pred_occ - pred_orig)
            changed = int(grade_occ != grade_orig)

            correct_orig += int(grade_orig == grade)
            correct_occ += int(grade_occ == grade)
            n_total += 1
            grade_shifts.append(shift)
            faithfulness_hits += changed

            rows.append({
                "img_id": img_id,
                "true_grade": grade,
                "top_tile": top_tile,
                "pred_orig": f"{pred_orig:.3f}",
                "pred_occ": f"{pred_occ:.3f}",
                "grade_orig": grade_orig,
                "grade_occ": grade_occ,
                "grade_shift": f"{shift:.3f}",
                "changed": changed,
            })

    if n_total == 0:
        print("No images processed.")
        return

    acc_orig = correct_orig / n_total
    acc_occ = correct_occ / n_total
    acc_drop = acc_orig - acc_occ
    mean_shift = float(np.mean(grade_shifts))
    faithfulness = faithfulness_hits / n_total

    print(f"\n=== Tile Occlusion Faithfulness (N={n_total}) ===")
    print(f"  Accuracy (original)  : {acc_orig * 100:.1f}%")
    print(f"  Accuracy (occluded)  : {acc_occ * 100:.1f}%")
    print(f"  Accuracy drop        : {acc_drop * 100:.1f}pp")
    print(f"  Mean grade shift     : {mean_shift:.3f}")
    print(f"  Faithfulness (changed): {faithfulness * 100:.1f}%")

    if args.output_csv and rows:
        os.makedirs(os.path.dirname(args.output_csv) or ".", exist_ok=True)
        with open(args.output_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"  Per-image results saved to {args.output_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--idrid_root",    default="Datasets/IDRiD")
    parser.add_argument("--model_dir",     default="runs/optic_concept_idrid/official")
    parser.add_argument("--inference_dir", default="explainability/idrid_outputs")
    parser.add_argument("--seg_split",     default="test", choices=["test", "train", "all"])
    parser.add_argument("--output_csv",    default="explainability/tile_occlusion_results.csv")
    args = parser.parse_args()
    evaluate(args)
