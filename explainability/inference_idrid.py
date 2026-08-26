"""
Run inference on IDRiD segmentation images using a trained OPTIC-C checkpoint
and save tile_weights + tile_concept_scores for downstream explainability eval.

Usage:
  python explainability/inference_idrid.py \
      --idrid_root  Datasets/IDRiD \
      --model_dir   runs/optic_concept_idrid/official \
      --output_dir  explainability/idrid_outputs \
      --seg_split   test

  --seg_split test  (default): 27 segmentation test images — use this when the
                    model was trained on the official IDRiD grading split (413
                    images) to avoid evaluating on training data.
  --seg_split all : all 81 segmentation images (train + test).

Output: <output_dir>/official_outputs.npz with keys:
  img_ids             (N,) str   — IDRiD image IDs, e.g. "IDRiD_01"
  labels              (N,) int   — grade labels (-1 if not found)
  preds               (N,) float — continuous predicted grade
  pred_grades         (N,) int   — rounded predicted grade
  tile_weights        (N, K, T)  — grade-specific GPA attention weights
  tile_concept_scores (N, T, C)  — per-tile concept cosine similarities
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Datasets.dataloaders import build_tile_transform
from Datasets.idrid_loader import IDRiDSegmentationDataset
from models.framework import build_model
from configs.config import DRConfig


def _find_checkpoint(model_dir: str) -> str:
    """Find the best checkpoint in model_dir — handles any fold index."""
    import glob
    # Prefer explicit best checkpoints for any fold
    candidates = sorted(glob.glob(os.path.join(model_dir, "fold*_best.pth")))
    if candidates:
        return candidates[0]  # fold0_best.pth < fold1_best.pth etc.
    fallback = os.path.join(model_dir, "best_model.pt")
    if os.path.isfile(fallback):
        return fallback
    raise FileNotFoundError(
        f"No checkpoint found in {model_dir}. "
        "Expected fold<N>_best.pth or best_model.pt."
    )


def load_model(model_dir: str, device: torch.device):
    ckpt_path = _find_checkpoint(model_dir)
    print(f"Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device)

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
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()

    text_encoder = None
    try:
        from models.clinical_text import ClinicalTextEncoder
        text_encoder = ClinicalTextEncoder(
            model_name="hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224",
            proj_out_dim=128,
            n_grade_classes=5,
        ).to(device)
        if "text_encoder_state" in ckpt:
            text_encoder.load_state_dict(ckpt["text_encoder_state"])
        text_encoder.eval()
    except Exception as e:
        print(f"Warning: could not load text encoder ({e}). Concept scores will be zeros.")

    return model, text_encoder


def run_inference(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    tile_tfm = build_tile_transform(tile_size=300, tile_grid=3, augment=False)
    ds = IDRiDSegmentationDataset(
        idrid_root=args.idrid_root,
        tile_transform=tile_tfm,
        split=args.seg_split,
    )
    print(f"Segmentation images ({args.seg_split} split): {len(ds)}")

    def collate_skip_masks(batch):
        # Masks are variable-size (real masks vs [1,1] sentinels) — keep as list.
        xs      = torch.stack([b[0] for b in batch])
        labels  = torch.tensor([b[1] for b in batch])
        img_ids = [b[2] for b in batch]
        masks   = [b[3] for b in batch]   # list of dicts, not stacked
        return xs, labels, img_ids, masks

    loader = DataLoader(ds, batch_size=4, shuffle=False, num_workers=4,
                        pin_memory=True, collate_fn=collate_skip_masks)

    model, text_encoder = load_model(args.model_dir, device)

    all_img_ids, all_labels, all_preds, all_pred_grades = [], [], [], []
    all_tile_weights, all_tile_concept_scores = [], []

    with torch.no_grad():
        for x, labels, img_ids, _ in loader:
            x = x.to(device)

            concept_embeds = None
            if text_encoder is not None:
                concept_embeds = text_encoder.get_concept_embeds()

            out = model(x, concept_embeds=concept_embeds)

            preds = out["pred"].cpu().float().numpy()
            pred_grades = np.clip(np.round(preds).astype(int), 0, 4)
            tw = out.get("tile_weights")
            tcs = out.get("tile_concept_scores")

            all_img_ids.extend(list(img_ids))
            all_labels.extend(labels.numpy().tolist())
            all_preds.extend(preds.tolist())
            all_pred_grades.extend(pred_grades.tolist())
            all_tile_weights.append(tw.cpu().numpy() if tw is not None else None)
            all_tile_concept_scores.append(tcs.cpu().numpy() if tcs is not None else None)

    N = len(all_img_ids)
    tw_arr = (np.concatenate(all_tile_weights, axis=0) if all_tile_weights[0] is not None
              else np.zeros((N, 5, 10), dtype=np.float32))
    tcs_arr = (np.concatenate(all_tile_concept_scores, axis=0) if all_tile_concept_scores[0] is not None
               else np.zeros((N, 10, 9), dtype=np.float32))

    out_path = os.path.join(args.output_dir, "official_outputs.npz")
    np.savez(
        out_path,
        img_ids=np.array(all_img_ids),
        labels=np.array(all_labels, dtype=np.int32),
        preds=np.array(all_preds, dtype=np.float32),
        pred_grades=np.array(all_pred_grades, dtype=np.int32),
        tile_weights=tw_arr,
        tile_concept_scores=tcs_arr,
    )
    print(f"Saved {N} inference outputs → {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--idrid_root",  default="Datasets/IDRiD")
    parser.add_argument("--model_dir",   default="runs/optic_concept_idrid/official")
    parser.add_argument("--output_dir",  default="explainability/idrid_outputs")
    parser.add_argument("--seg_split",   default="test", choices=["test", "train", "all"],
                        help="Which segmentation images to run: 'test' (default, 27 images, "
                             "no leakage), 'all' (81 images).")
    args = parser.parse_args()
    run_inference(args)
