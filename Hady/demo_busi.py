"""
Minimal inference demo for the trained BUSI model.

Loads BUSI_best_model.pth (EfficientNet-V2S + regression head, the ESPACOL
Hybrid Contrastive Ordinal model), picks one random image from the BUSI
dataset, and prints the model's prediction.

Run (from the repo root or anywhere):
    python Hady/demo.py                 # random image
    python Hady/demo.py --index 0       # deterministic single image
    python Hady/demo.py --busi_root /path/to/BUSI

Note: this is a regression-style ordinal model — it predicts a continuous
severity score that is rounded to the nearest class (no per-class probabilities).
"""

import argparse
import os
import random
import sys

import torch

# Make the repo root (parent of Hady/) importable regardless of CWD.
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _REPO_ROOT)

from Datasets.dataloaders import BUSIDataset
from models.framework import build_model
from utils.checkpoint import load_checkpoint
from utils.metrics import round_predictions

N_CLASSES = 3  # normal=0, benign=1, malignant=2
LABEL_TO_CLASS = {v: k for k, v in BUSIDataset.CLASS_TO_LABEL.items()}


def resolve_busi_root(explicit: str | None) -> str | None:
    """Return the first valid BUSI dataset root, or None if not found.

    A valid root contains benign/ malignant/ normal/ subfolders.
    """
    def _valid(p: str) -> bool:
        return all(os.path.isdir(os.path.join(p, c)) for c in BUSIDataset.CLASS_TO_LABEL)

    candidates = []
    if explicit:
        candidates.append(explicit)
    candidates += [
        os.path.join(_REPO_ROOT, "Datasets", "BUSI"),            # repo location (train_busi.py default)
        os.path.join(_REPO_ROOT, os.pardir, "Datasets", "BUSI"),  # legacy: alongside the repo
    ]
    for c in candidates:
        c = os.path.normpath(c)
        if _valid(c):
            return c
    return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="BUSI single-image inference demo")
    parser.add_argument(
        "--checkpoint",
        default=os.path.join(_HERE, "BUSI_best_model.pth"),
        help="Path to the trained BUSI checkpoint (default: next to this script).",
    )
    parser.add_argument(
        "--busi_root",
        default=None,
        help="BUSI dataset root containing benign/ malignant/ normal/ folders. "
        "If omitted, common locations are tried automatically.",
    )
    parser.add_argument(
        "--index",
        type=int,
        default=None,
        help="Image index to predict (default: random).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional seed for the random image choice (reproducible).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not os.path.isfile(args.checkpoint):
        sys.exit(f"Checkpoint not found: {args.checkpoint}")

    busi_root = resolve_busi_root(args.busi_root)
    if busi_root is None:
        sys.exit(
            "BUSI dataset not found. Pass its location with --busi_root "
            "(it should contain benign/ malignant/ normal/ folders)."
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Build model and load trained weights ────────────────────────────────
    model = build_model(n_classes=N_CLASSES, pretrained=False)
    metrics = load_checkpoint(args.checkpoint, model, optimizer=None, device=device)
    model.to(device).eval()
    if metrics:
        va, mae = metrics.get("val_acc"), metrics.get("val_mae")
        bits = []
        if va is not None:
            bits.append(f"val_acc={va:.2f}%")
        if mae is not None:
            bits.append(f"val_mae={mae:.4f}")
        print(f"Loaded checkpoint ({', '.join(bits) if bits else 'no metrics'})")

    # ── Pick one image (BUSIDataset applies the eval transform internally) ───
    ds = BUSIDataset(root_dir=busi_root, split="all")
    print(f"BUSI images found: {len(ds)}")

    if args.seed is not None:
        random.seed(args.seed)
    if args.index is not None:
        if not 0 <= args.index < len(ds):
            sys.exit(f"--index must be in [0, {len(ds) - 1}]")
        idx = args.index
    else:
        idx = random.randrange(len(ds))

    x, y = ds[idx]                       # x: (3,300,300) preprocessed, y: long tensor
    img_path = ds.items[idx][0]
    true_label = int(y)

    # ── Predict ─────────────────────────────────────────────────────────────
    with torch.no_grad():
        score = model.predict(x.unsqueeze(0).to(device))   # (1,) continuous
    pred_cls = int(round_predictions(score.cpu(), N_CLASSES).item())
    raw_score = float(score.item())

    # ── Report ──────────────────────────────────────────────────────────────
    correct = pred_cls == true_label
    print("\n" + "-" * 60)
    print(f"Image      : {os.path.basename(img_path)}  (index {idx})")
    print(f"True label : {LABEL_TO_CLASS[true_label]} ({true_label})")
    print(f"Reg. score : {raw_score:.4f}  (continuous ordinal output)")
    print(f"Prediction : {LABEL_TO_CLASS[pred_cls]} ({pred_cls})")
    print(f"Result     : {'CORRECT' if correct else 'WRONG'}")
    print("-" * 60)


if __name__ == "__main__":
    main()
