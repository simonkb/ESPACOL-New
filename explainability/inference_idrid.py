"""
Run inference on IDRiD segmentation set (81 images) using a trained OPTIC-C
checkpoint and save tile_weights + tile_concept_scores for downstream
explainability evaluation.

Usage (per fold checkpoint):
  python explainability/inference_idrid.py \
      --idrid_root Datasets/IDRiD \
      --run_dir    runs/optic_concept_idrid_cv \
      --output_dir explainability/idrid_outputs \
      --fold       0

Outputs (one .npz per fold in output_dir):
  fold{k}_outputs.npz with keys:
    img_ids          (N,) str — IDRiD image IDs, e.g. "IDRiD_01"
    labels           (N,) int — grade labels (-1 if not in grading set)
    preds            (N,) float — expected grade (continuous)
    pred_grades      (N,) int — rounded predicted grade
    tile_weights     (N, K, T) float32 — grade-specific tile attention weights
    tile_concept_scores (N, T, C) float32 — per-tile concept cosine similarities
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from Datasets.dataloaders import build_tile_transform
from Datasets.idrid_loader import IDRiDSegmentationDataset
from Models.framework import build_model
from configs.config import DRConfig


def load_checkpoint(fold_dir: str, device: torch.device):
    """Load best model checkpoint from a fold directory."""
    ckpt_path = os.path.join(fold_dir, "best_model.pt")
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device)
    return ckpt


def run_inference(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    tile_tfm = build_tile_transform(tile_size=300, tile_grid=3, augment=False)

    ds = IDRiDSegmentationDataset(
        idrid_root=args.idrid_root,
        tile_transform=tile_tfm,
    )
    loader = DataLoader(ds, batch_size=4, shuffle=False, num_workers=4, pin_memory=True)

    fold_dir = os.path.join(args.run_dir, f"fold{args.fold}")
    ckpt = load_checkpoint(fold_dir, device)

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
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()

    # Load text encoder for concept embeddings
    text_encoder = None
    try:
        from Models.clinical_text import ClinicalTextEncoder
        text_encoder = ClinicalTextEncoder(
            model_name="hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224",
            proj_out_dim=128,
            n_grade_classes=5,
        ).to(device)
        if "text_encoder_state_dict" in ckpt:
            text_encoder.load_state_dict(ckpt["text_encoder_state_dict"])
        text_encoder.eval()
    except Exception as e:
        print(f"Warning: could not load text encoder ({e}). Concept scores will be zeros.")

    all_img_ids = []
    all_labels = []
    all_preds = []
    all_pred_grades = []
    all_tile_weights = []
    all_tile_concept_scores = []

    with torch.no_grad():
        for batch in loader:
            x, labels, img_ids, _ = batch
            x = x.to(device)

            concept_embeds = None
            if text_encoder is not None:
                concept_embeds = text_encoder.get_concept_embeds()

            out = model(x, concept_embeds=concept_embeds)

            preds = out["pred"].cpu().float().numpy()
            pred_grades = np.clip(np.round(preds).astype(int), 0, 4)

            tw = out.get("tile_weights", None)
            tcs = out.get("tile_concept_scores", None)

            all_img_ids.extend(list(img_ids))
            all_labels.extend(labels.numpy().tolist())
            all_preds.extend(preds.tolist())
            all_pred_grades.extend(pred_grades.tolist())
            all_tile_weights.append(tw.cpu().numpy() if tw is not None else None)
            all_tile_concept_scores.append(tcs.cpu().numpy() if tcs is not None else None)

    out_path = os.path.join(args.output_dir, f"fold{args.fold}_outputs.npz")
    np.savez(
        out_path,
        img_ids=np.array(all_img_ids),
        labels=np.array(all_labels, dtype=np.int32),
        preds=np.array(all_preds, dtype=np.float32),
        pred_grades=np.array(all_pred_grades, dtype=np.int32),
        tile_weights=np.concatenate(all_tile_weights, axis=0) if all_tile_weights[0] is not None else np.zeros((len(all_img_ids), 5, 10)),
        tile_concept_scores=np.concatenate(all_tile_concept_scores, axis=0) if all_tile_concept_scores[0] is not None else np.zeros((len(all_img_ids), 10, 9)),
    )
    print(f"Saved {len(all_img_ids)} inference outputs to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--idrid_root", default="Datasets/IDRiD")
    parser.add_argument("--run_dir", default="runs/optic_concept_idrid_cv")
    parser.add_argument("--output_dir", default="explainability/idrid_outputs")
    parser.add_argument("--fold", type=int, required=True)
    args = parser.parse_args()
    run_inference(args)
