"""
Minimal inference demo for the trained DR (diabetic retinopathy) model.

Loads DR_best_model.pth (EfficientNet-V2S + regression head, the ESPACOL
Hybrid Contrastive Ordinal model), picks one random fundus image from the DR
dataset, and prints the model's predicted severity grade.

Run (from the repo root or anywhere):
    python Hady/demo_dr.py                 # random image (usually grade 0)
    python Hady/demo_dr.py --grade 4       # random Proliferative-DR image
    python Hady/demo_dr.py --index 0       # deterministic single image
    python Hady/demo_dr.py --dr_root /path/to/DR

Note: this is a regression-style ordinal model — it predicts a continuous
severity score that is rounded to the nearest grade (no per-class probabilities).
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

from Datasets.dataloaders import DRDataset
from models.framework import build_model
from utils.checkpoint import load_checkpoint
from utils.metrics import round_predictions

N_CLASSES = 5  # DR grades 0-4
GRADE_NAMES = {
    0: "No DR",
    1: "Mild",
    2: "Moderate",
    3: "Severe",
    4: "Proliferative DR",
}


def resolve_dr_root(explicit: str | None) -> str | None:
    """Return the first valid DR dataset root, or None if not found.

    A valid root contains a train/ subdir and a trainLabels.csv.
    """
    def _valid(p: str) -> bool:
        return os.path.isdir(os.path.join(p, "train")) and os.path.isfile(
            os.path.join(p, "trainLabels.csv")
        )

    candidates = []
    if explicit:
        candidates.append(explicit)
    candidates += [
        os.path.join(_REPO_ROOT, "Datasets", "DR"),            # repo location (train_dr.py default)
        os.path.join(_REPO_ROOT, os.pardir, "Datasets", "DR"),  # legacy: alongside the repo
    ]
    for c in candidates:
        c = os.path.normpath(c)
        if _valid(c):
            return c
    return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DR single-image inference demo")
    parser.add_argument(
        "--checkpoint",
        default=os.path.join(_HERE, "DR_best_model.pth"),
        help="Path to the trained DR checkpoint (default: next to this script).",
    )
    parser.add_argument(
        "--dr_root",
        default=None,
        help="DR dataset root (contains train/ and trainLabels.csv). "
        "If omitted, common locations are tried automatically.",
    )
    parser.add_argument(
        "--train_csv",
        default=None,
        help="Labels CSV (default: <dr_root>/trainLabels.csv).",
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
    parser.add_argument(
        "--grade",
        type=int,
        default=None,
        choices=range(N_CLASSES),
        help="Restrict the random pick to images with this true grade (0-4). "
        "Useful because ~74%% of DR images are grade 0.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not os.path.isfile(args.checkpoint):
        sys.exit(f"Checkpoint not found: {args.checkpoint}")

    dr_root = resolve_dr_root(args.dr_root)
    if dr_root is None:
        sys.exit(
            "DR dataset not found. Pass its location with --dr_root "
            "(it should contain a train/ folder and trainLabels.csv)."
        )
    train_csv = args.train_csv or os.path.join(dr_root, "trainLabels.csv")
    if not os.path.isfile(train_csv):
        sys.exit(f"Labels CSV not found: {train_csv}")

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

    # ── Pick one image (DRDataset applies the eval transform internally) ─────
    print("Indexing DR dataset (this can take a few seconds)...")
    ds = DRDataset(root_dir=dr_root, split="train", csv_path=train_csv)
    print(f"DR images found: {len(ds)}")

    # Candidate indices (optionally restricted to a specific true grade).
    if args.grade is not None:
        candidates = [i for i, (_, y) in enumerate(ds.items) if y == args.grade]
        if not candidates:
            sys.exit(f"No images with grade {args.grade} found in {train_csv}.")
    else:
        candidates = range(len(ds))

    if args.seed is not None:
        random.seed(args.seed)
    if args.index is not None:
        if not 0 <= args.index < len(ds):
            sys.exit(f"--index must be in [0, {len(ds) - 1}]")
        idx = args.index
    else:
        idx = random.choice(list(candidates))

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
    print(f"True grade  : {GRADE_NAMES[true_label]} ({true_label})")
    print(f"Reg. score : {raw_score:.4f}  (continuous ordinal output)")
    print(f"Prediction : {GRADE_NAMES[pred_cls]} ({pred_cls})")
    print(f"Result     : {'CORRECT' if correct else 'WRONG'}")
    print("-" * 60)


if __name__ == "__main__":
    main()
