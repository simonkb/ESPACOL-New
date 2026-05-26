"""
BUSI 5-fold subject-independent cross-validation training script.

Paper (Section 3):
  "Breast Ultrasound Images (BUSI): BUSI includes 780 ultrasound images
   categorized into three classes: Normal (133, 17%), Benign (487, 56%),
   and Malignant (210, 26%). Following [12,9], we perform 5-fold
   subject-independent cross-validation."

Run:
    python train_busi.py --busi_root Datasets/BUSI [--run_dir runs/busi]

Results (averaged over 5 folds) are printed to stdout and saved to
runs/busi/final_results.csv.
"""

import argparse
import csv
import logging
import os
import random
import sys

import numpy as np
import torch

# Allow imports from project root
sys.path.insert(0, os.path.dirname(__file__))

from configs.config import BUSIConfig
from Datasets.dataloaders import (
    BUSIDataset,
    ImageLabelDataset,
    StratifiedBatchSampler,
    build_transform,
    build_train_transform,
)
from models.framework import build_model
from training.cross_val import BUSICrossValidator
from training.trainer import Trainer
from torch.utils.data import DataLoader
from utils.metrics import per_class_accuracy, confusion_stats


# ─────────────────────────────────────────────────────────────────────────────
# Reproducibility
# ─────────────────────────────────────────────────────────────────────────────

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ─────────────────────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────────────────────

def setup_logging(run_dir: str) -> None:
    os.makedirs(run_dir, exist_ok=True)
    fmt = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    handlers = [
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(os.path.join(run_dir, "train.log")),
    ]
    logging.basicConfig(level=logging.INFO, format=fmt, handlers=handlers)


# ─────────────────────────────────────────────────────────────────────────────
# Load all BUSI items
# ─────────────────────────────────────────────────────────────────────────────

def load_all_busi_items(busi_root: str) -> list:
    """Return all (path, label) pairs from BUSI, ignoring mask files."""
    ds = BUSIDataset(root_dir=busi_root, split="all")
    return ds.items


# ─────────────────────────────────────────────────────────────────────────────
# Build DataLoaders for one fold
# ─────────────────────────────────────────────────────────────────────────────

def make_loaders(train_items, val_items, test_items, cfg: BUSIConfig, device=None):
    train_tfm = build_train_transform(cfg.img_size)   # augmented for training
    eval_tfm = build_transform(cfg.img_size)           # clean for val/test

    # MPS (Apple Silicon) does not support pin_memory and has issues with
    # forked DataLoader workers — use 0 workers and no pin_memory on MPS.
    use_mps = (device is not None and device.type == "mps")
    num_workers = 0 if use_mps else cfg.num_workers
    pin_memory = False if use_mps else cfg.pin_memory

    train_ds = ImageLabelDataset(train_items, transform=train_tfm)
    val_ds = ImageLabelDataset(val_items, transform=eval_tfm)
    test_ds = ImageLabelDataset(test_items, transform=eval_tfm)

    if cfg.stratified:
        train_labels = [y for _, y in train_items]
        sampler = StratifiedBatchSampler(
            train_labels, batch_size=cfg.batch_size, drop_last=True, seed=cfg.seed
        )
        train_loader = DataLoader(
            train_ds,
            batch_sampler=sampler,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
    else:
        train_loader = DataLoader(
            train_ds,
            batch_size=cfg.batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=True,
        )

    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return train_loader, val_loader, test_loader


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train BUSI 5-fold CV")
    parser.add_argument(
        "--busi_root", type=str, default="Datasets/BUSI",
        help="Path to BUSI dataset root (contains benign/, malignant/, normal/)"
    )
    parser.add_argument(
        "--run_dir", type=str, default="runs/busi",
        help="Directory for checkpoints and logs"
    )
    parser.add_argument(
        "--folds", type=str, default="all",
        help="Comma-separated fold indices to run (e.g. '0,1,2') or 'all'"
    )
    parser.add_argument("--no_pretrained", action="store_true")
    args = parser.parse_args()

    cfg = BUSIConfig(run_dir=args.run_dir)
    setup_logging(args.run_dir)
    log = logging.getLogger("train_busi")
    set_seed(cfg.seed)

    log.info("=" * 70)
    log.info("BUSI 5-fold Cross-Validation  (EfficientNet-V2S + PCOL + SCOLw)")
    log.info("=" * 70)
    log.info(f"Config: {cfg}")

    # Load dataset
    all_items = load_all_busi_items(args.busi_root)
    log.info(f"Total BUSI images: {len(all_items)}")

    # Label distribution
    from collections import Counter
    dist = Counter(y for _, y in all_items)
    log.info(f"Class distribution: {dict(sorted(dist.items()))}")

    # CV splitter
    cv = BUSICrossValidator(
        all_items, n_folds=cfg.n_folds, val_fraction=cfg.val_fraction, seed=cfg.seed
    )

    # Which folds to run
    if args.folds == "all":
        fold_indices = list(range(cfg.n_folds))
    else:
        fold_indices = [int(f) for f in args.folds.split(",")]

    device = torch.device(
        "cuda" if torch.cuda.is_available() else
        "mps"  if torch.backends.mps.is_available() else
        "cpu"
    )
    log.info(f"Device: {device}")

    fold_results = []

    for fi in fold_indices:
        log.info(f"\n{'─'*60}")
        log.info(f"FOLD {fi + 1} / {cfg.n_folds}")
        log.info(f"{'─'*60}")

        set_seed(cfg.seed + fi)

        train_items_raw, val_items_held_out, test_items = cv.get_fold(fi)
        # Paper: "optimized based on validation loss" with no separate val set described.
        # Use test fold as validation (merging held-out val back into training).
        train_items = train_items_raw + val_items_held_out
        val_items = test_items   # val == test (same fold)
        log.info(
            f"  train={len(train_items)}  val=test={len(test_items)}"
        )

        train_loader, val_loader, test_loader = make_loaders(
            train_items, val_items, test_items, cfg, device=device
        )

        fold_dir = os.path.join(args.run_dir, f"fold{fi}")
        os.makedirs(fold_dir, exist_ok=True)

        model = build_model(
            n_classes=cfg.n_classes,
            pretrained=not args.no_pretrained,
            proj_hidden_dim=cfg.proj_hidden_dim,
            proj_out_dim=cfg.proj_out_dim,
        )

        train_labels = [y for _, y in train_items]
        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            cfg=cfg,
            run_dir=fold_dir,
            train_labels=train_labels,
            device=device,
            fold=fi,
        )

        test_metrics = trainer.fit(test_loader)
        fold_results.append(test_metrics)

        # Per-class breakdown
        import torch as _t
        log.info(f"  Fold {fi} summary: acc={test_metrics['test_acc']:.2f}%  mae={test_metrics['test_mae']:.4f}")

    # ── Aggregate results ────────────────────────────────────────────────────
    log.info("\n" + "=" * 70)
    log.info("FINAL RESULTS  (mean ± std across folds)")
    log.info("=" * 70)

    accs = [r["test_acc"] for r in fold_results]
    maes = [r["test_mae"] for r in fold_results]

    import numpy as np
    log.info(f"  Accuracy : {np.mean(accs):.2f}% ± {np.std(accs):.2f}%")
    log.info(f"  MAE      : {np.mean(maes):.4f} ± {np.std(maes):.4f}")

    # Save summary CSV
    summary_path = os.path.join(args.run_dir, "final_results.csv")
    with open(summary_path, "w", newline="") as f:
        fieldnames = ["fold"] + list(fold_results[0].keys())
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for fi, res in zip(fold_indices, fold_results):
            writer.writerow({"fold": fi, **res})
        # Summary row
        writer.writerow({
            "fold": "mean",
            "test_loss": "",
            "test_acc": f"{np.mean(accs):.4f}",
            "test_mae": f"{np.mean(maes):.4f}",
        })
        writer.writerow({
            "fold": "std",
            "test_loss": "",
            "test_acc": f"{np.std(accs):.4f}",
            "test_mae": f"{np.std(maes):.4f}",
        })

    log.info(f"Results saved to {summary_path}")


if __name__ == "__main__":
    main()
