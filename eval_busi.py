"""
Measure the BUSI fixes on a trained checkpoint.

  Fix 2 (concept): per-class concept top-1 — fraction of images whose nearest
                   BioMedCLIP concept belongs to the true class. This is the
                   number that shows the per-image concept loss worked.
  Fix 1 (seg):     mean energy-in-mask and Dice on lesion images — shows the
                   seg head localizes (energy vs the ~0.29 diffuse baseline).

Run on a checkpoint trained with the matching flags:
  # both fixes in one run:
  python train_busi.py --busi_root Datasets/BUSI --folds 0 --use_concept --use_seg --run_dir runs/busi_both
  python eval_busi.py  --busi_root Datasets/BUSI --ckpt runs/busi_both/fold0/fold0_best.pth --has_concept --has_seg

  # or just the seg checkpoint you already have:
  python eval_busi.py  --busi_root Datasets/BUSI --ckpt runs/busi/fold0/fold0_best.pth --has_seg
"""

import argparse
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from configs.config import BUSIConfig
from Datasets.dataloaders import BUSIDataset, ImageLabelDataset, build_transform
from Datasets.busi_seg import BUSISegDataset
from training.cross_val import BUSICrossValidator
from models.framework import build_model
from models.concept_bank import (
    load_biomedclip, encode_concept_texts, BUSI_CONCEPTS, BUSI_CONCEPT_GRADES,
)


def load_state(path):
    """Robustly pull a state_dict out of a checkpoint regardless of key name."""
    obj = torch.load(path, map_location="cpu")
    if isinstance(obj, dict):
        for k in ("model_state", "model_state_dict", "state_dict", "model"):
            if k in obj and isinstance(obj[k], dict):
                return obj[k]
    return obj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--busi_root", default="Datasets/BUSI")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--has_concept", action="store_true",
                    help="checkpoint was trained with --use_concept")
    ap.add_argument("--has_seg", action="store_true",
                    help="checkpoint was trained with --use_seg")
    args = ap.parse_args()

    cfg = BUSIConfig()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Rebuild the SAME architecture the checkpoint was trained with, then load.
    model = build_model(
        n_classes=cfg.n_classes, pretrained=False,
        proj_hidden_dim=cfg.proj_hidden_dim, proj_out_dim=cfg.proj_out_dim,
        use_concept=args.has_concept, use_seg=args.has_seg,
    ).to(device).eval()
    sd = load_state(args.ckpt)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"loaded {len(sd)} tensors  (missing={len(missing)}, unexpected={len(unexpected)})")

    # Fold-0 test split, same splitter and seed as training.
    all_items = BUSIDataset(root_dir=args.busi_root, split="all").items
    cv = BUSICrossValidator(all_items, n_folds=cfg.n_folds,
                            val_fraction=cfg.val_fraction, seed=cfg.seed)
    _, _, test_items = cv.get_fold(0)
    tfm = build_transform(cfg.img_size)

    # ── Fix 2: concept top-1 by class ────────────────────────────────────────
    if args.has_concept:
        bmc, tok = load_biomedclip(
            getattr(cfg, "biomedclip_model",
                    "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"),
            device,
        )
        raw = encode_concept_texts(bmc, tok, BUSI_CONCEPTS, device)        # (C, 512)
        with torch.no_grad():
            V = F.normalize(model.project_concepts(raw), dim=-1)           # (C, 128)
        grades = BUSI_CONCEPT_GRADES.to(device)

        dl = DataLoader(ImageLabelDataset(test_items, transform=tfm),
                        batch_size=16, shuffle=False)
        hit, tot = {}, {}
        with torch.no_grad():
            for x, y in dl:
                z = F.normalize(model.forward_all(x.to(device))["z_align"], dim=-1)
                top_grade = grades[(z @ V.t()).argmax(1)]                  # nearest concept's grade
                for yi, ti in zip(y.tolist(), top_grade.tolist()):
                    tot[yi] = tot.get(yi, 0) + 1
                    hit[yi] = hit.get(yi, 0) + (1 if ti == yi else 0)
        print("\n== Fix 2: concept top-1 by class (nearest concept grade == true class) ==")
        for c in sorted(tot):
            print(f"  class {c}: {hit.get(c,0)}/{tot[c]} = {hit.get(c,0)/tot[c]:.3f}")
        n = sum(tot.values())
        print(f"  overall: {sum(hit.values())}/{n} = {sum(hit.values())/n:.3f}")

    # ── Fix 1: segmentation localization on lesion images ────────────────────
    if args.has_seg:
        dl = DataLoader(BUSISegDataset(args.busi_root, items=test_items, img_size=cfg.img_size),
                        batch_size=8, shuffle=False)
        energies, dices = [], []
        with torch.no_grad():
            for x, y, m in dl:
                x, m = x.to(device), m.to(device)
                prob = torch.sigmoid(model.forward_all(x, want_seg=True)["seg_logits"])
                for i in range(x.size(0)):
                    M = m[i]
                    if M.sum() < 1:
                        continue   # skip Normal (no lesion mask)
                    p = prob[i]
                    energies.append((p * M).sum().item() / (p.sum().item() + 1e-6))
                    pred = (p > 0.5).float()
                    inter = (pred * M).sum().item()
                    dices.append(2 * inter / (pred.sum().item() + M.sum().item() + 1e-6))
        if energies:
            print("\n== Fix 1: seg localization on lesion images ==")
            print(f"  mean energy-in-mask: {np.mean(energies):.3f}   (diffuse baseline ~0.29)")
            print(f"  mean Dice:           {np.mean(dices):.3f}")
        else:
            print("\n[Fix 1] no lesion masks found in the test fold")


if __name__ == "__main__":
    main()