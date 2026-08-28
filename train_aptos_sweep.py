"""
WandB sweep agent for OPTIC-C hyperparameter search on APTOS 2019.

Runs fold 0 only as a fast proxy (~1-2 h per trial on a single GPU).
The best hyperparameter set found here transfers directly to full 10-fold
Kaggle DR training — APTOS and DR share the same 5-grade fundus grading task.

Usage:
  # 1. Create sweep once (produces a sweep_id):
  #    wandb sweep sweep_config_aptos_optic.yaml --project espacol-new
  #
  # 2. Launch agents (one per SLURM job):
  #    python train_aptos_sweep.py --sweep_id <entity/project/sweep_id> --count 5
  #    or via:  sbatch submit_aptos_sweep.sh <sweep_id>
"""

from __future__ import annotations

import argparse
import logging
import os
import random
import sys
import time

import numpy as np
import pandas as pd
import torch
import wandb
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from configs.config import DRConfig
from Datasets.dataloaders import (
    ImageLabelDataset,
    StratifiedBatchSampler,
    build_tile_transform,
)
from models.framework import build_model
from training.cross_val import DRCrossValidator
from training.trainer import Trainer
from utils.checkpoint import save_checkpoint, load_checkpoint


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_aptos_items(aptos_root: str) -> list:
    """Read train.csv → list of (image_path, label) tuples."""
    csv_path = os.path.join(aptos_root, "train.csv")
    img_dir  = os.path.join(aptos_root, "train_images")
    df = pd.read_csv(csv_path)
    items = []
    for _, row in df.iterrows():
        path  = os.path.join(img_dir, str(row["id_code"]) + ".png")
        label = int(row["diagnosis"])
        items.append((path, label))
    return items


# ── Sweep-aware trainer ────────────────────────────────────────────────────────

class SweepTrainer(Trainer):
    """Extends Trainer with per-epoch wandb logging and early abandonment."""

    def fit(self, test_loader) -> dict:
        best_val_acc   = -float("inf")
        best_val_mae   = float("inf")
        best_score     = -float("inf")
        best_ckpt_path = os.path.join(self.run_dir, "fold0_best.pth")

        for epoch in range(1, self.cfg.epochs + 1):
            t0 = time.time()
            train_metrics = self._train_epoch(epoch)
            val_metrics   = self._eval_epoch(self.val_loader, prefix="val")
            elapsed       = time.time() - t0
            lr_now        = self.optimizer.param_groups[0]["lr"]

            self._log_epoch(epoch, elapsed, lr_now, train_metrics, val_metrics)

            val_acc  = val_metrics["val_acc"]
            val_mae  = val_metrics["val_mae"]
            val_loss = val_metrics["val_loss"]

            # Joint score: +1% accuracy ≈ 0.1 MAE improvement
            score   = val_acc - 10.0 * val_mae
            is_best = score > best_score
            if is_best:
                best_score   = score
                best_val_acc = val_acc
                best_val_mae = val_mae

            wandb.log(
                {
                    "epoch":        epoch,
                    "lr":           lr_now,
                    "best_val_acc": best_val_acc,
                    "best_val_mae": best_val_mae,
                    "best_score":   best_score,
                    **train_metrics,
                    **val_metrics,
                },
                step=epoch,
            )

            # Early abandonment for clearly underperforming runs
            if epoch == 10 and best_val_acc < 58.0:
                logging.getLogger(__name__).info(
                    f"Abandoning at epoch {epoch}: best_val_acc={best_val_acc:.1f}%"
                )
                break
            if epoch >= 20 and best_val_acc < 72.0:
                logging.getLogger(__name__).info(
                    f"Abandoning at epoch {epoch}: best_val_acc={best_val_acc:.1f}%"
                )
                break

            # Save current checkpoint; keep only current + best
            cur_ckpt = os.path.join(self.run_dir, f"fold0_epoch{epoch}.pth")
            save_checkpoint(
                path=cur_ckpt,
                model=self.model,
                optimizer=self.optimizer,
                epoch=epoch,
                metrics={**train_metrics, **val_metrics},
                is_best=is_best,
                text_encoder=self.text_encoder,
            )
            if is_best:
                save_checkpoint(
                    path=best_ckpt_path,
                    model=self.model,
                    optimizer=self.optimizer,
                    epoch=epoch,
                    metrics={**train_metrics, **val_metrics},
                    is_best=False,
                    text_encoder=self.text_encoder,
                )

            # Delete previous epoch (not the best)
            if epoch > 1:
                prev_ckpt = os.path.join(self.run_dir, f"fold0_epoch{epoch - 1}.pth")
                if os.path.exists(prev_ckpt) and prev_ckpt != best_ckpt_path:
                    os.remove(prev_ckpt)

            self.scheduler.step(val_loss)
            if self.early_stopping.step(val_acc):
                logging.getLogger(__name__).info(f"Early stopping at epoch {epoch}")
                break

        # Evaluate on test set using best checkpoint
        if os.path.exists(best_ckpt_path):
            load_checkpoint(best_ckpt_path, self.model, None,
                            self.text_encoder, self.device)

        test_metrics = self._eval_epoch(test_loader, prefix="test")

        wandb.log({
            "test_acc": test_metrics["test_acc"],
            "test_mae": test_metrics["test_mae"],
        })
        wandb.summary["test_acc"]    = test_metrics["test_acc"]
        wandb.summary["test_mae"]    = test_metrics["test_mae"]
        wandb.summary["best_val_acc"] = best_val_acc
        wandb.summary["best_val_mae"] = best_val_mae
        wandb.summary["best_score"]   = best_score

        return test_metrics


# ── Config builder ─────────────────────────────────────────────────────────────

def build_cfg(wc) -> DRConfig:
    """Build DRConfig from sweep wandb.config, fixing the OPTIC-C v9 baseline."""
    cfg = DRConfig()

    # ── Fixed OPTIC-C v9 baseline (same as submit_optic_concept_cv_v9.sh) ──
    cfg.dataset              = "DR"       # use DR text encoder descriptions for APTOS
    cfg.n_classes            = 5
    cfg.n_folds              = 5          # APTOS: 5-fold
    cfg.val_fraction         = 0.1
    cfg.epochs               = 100
    cfg.batch_size           = 24
    cfg.lr                   = 2e-4
    cfg.stratified           = True
    cfg.pretrained           = True
    cfg.amp                  = True
    cfg.use_multi_tile       = True
    cfg.tile_grid            = 3
    cfg.use_tile_transformer = True
    cfg.use_grade_prototypes = True
    cfg.use_ordinal_head     = True
    cfg.use_concept_prototype= True
    cfg.alpha                = 0.0        # no old SCOLw losses
    cfg.beta                 = 0.0
    cfg.gamma                = 0.0
    cfg.lambda_osd           = 0.0
    cfg.lambda_tcl           = 0.0
    cfg.early_stop_patience  = 30

    # ── Sweep parameters ──
    cfg.lambda_proto_ce       = float(wc.get("lambda_proto_ce",       1.0))
    cfg.lambda_tile_concept   = float(wc.get("lambda_tile_concept",   0.5))
    cfg.lambda_gpa            = float(wc.get("lambda_gpa",            0.1))
    cfg.proto_temperature     = float(wc.get("proto_temperature",     0.15))
    cfg.proto_label_smoothing = float(wc.get("proto_label_smoothing", 0.07))
    cfg.new_component_lr_mult = float(wc.get("new_component_lr_mult", 2.5))
    cfg.backbone_freeze_epochs= int(wc.get("backbone_freeze_epochs",  25))
    cfg.lr_patience           = int(wc.get("lr_patience",             12))
    cfg.weight_decay          = float(wc.get("weight_decay",          1e-5))
    cfg.use_cosine_lr         = bool(wc.get("use_cosine_lr",          False))

    return cfg


# ── Main sweep run ─────────────────────────────────────────────────────────────

def run_sweep(args) -> dict:
    run = wandb.init()
    wc  = wandb.config

    cfg         = build_cfg(wc)
    run_dir     = os.path.join(args.run_dir, run.id)
    cfg.run_dir = run_dir
    os.makedirs(run_dir, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
        force=True,
    )
    log = logging.getLogger("aptos_sweep")
    log.info(f"Run {run.id} — sweep config: {dict(wc)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Device: {device}")

    set_seed(cfg.seed)

    # ── Data ──
    all_items = load_aptos_items(args.aptos_root)
    log.info(f"APTOS: {len(all_items)} images")

    cv = DRCrossValidator(
        all_items, n_folds=cfg.n_folds,
        val_fraction=cfg.val_fraction, seed=cfg.seed,
    )
    train_items_raw, val_items_held, test_items = cv.get_fold(0)
    # Use held-out val fold as part of training (same protocol as train_dr.py)
    train_items = train_items_raw + val_items_held
    val_items   = test_items

    log.info(f"Fold 0: train={len(train_items)}  val=test={len(test_items)}")

    tile_tfm_train = build_tile_transform(
        tile_size=cfg.img_size, tile_grid=cfg.tile_grid, augment=True)
    tile_tfm_eval  = build_tile_transform(
        tile_size=cfg.img_size, tile_grid=cfg.tile_grid, augment=False)

    train_ds  = ImageLabelDataset(train_items, transform=tile_tfm_train)
    val_ds    = ImageLabelDataset(val_items,   transform=tile_tfm_eval)
    test_ds   = ImageLabelDataset(test_items,  transform=tile_tfm_eval)

    train_labels = [y for _, y in train_items]
    sampler = StratifiedBatchSampler(
        train_labels, batch_size=cfg.batch_size, drop_last=True, seed=cfg.seed)

    loader_kwargs = dict(num_workers=4, pin_memory=True,
                         prefetch_factor=4, persistent_workers=True)
    train_loader = DataLoader(train_ds, batch_sampler=sampler, **loader_kwargs)
    val_loader   = DataLoader(val_ds,   batch_size=cfg.batch_size,
                              shuffle=False, **loader_kwargs)
    test_loader  = DataLoader(test_ds,  batch_size=cfg.batch_size,
                              shuffle=False, **loader_kwargs)

    # ── Model ──
    model = build_model(
        n_classes=cfg.n_classes,
        pretrained=True,
        use_multi_tile=True,
        tile_grid=cfg.tile_grid,
        use_tile_transformer=True,
        use_grade_prototypes=True,
        use_ordinal_head=True,
        use_concept_prototype=True,
        proto_temperature=cfg.proto_temperature,
    )

    # ── Trainer ──
    trainer = SweepTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        cfg=cfg,
        run_dir=run_dir,
        train_labels=train_labels,
        device=device,
        fold=0,
    )

    test_metrics = trainer.fit(test_loader)
    log.info(
        f"Final: test_acc={test_metrics['test_acc']:.2f}%  "
        f"test_mae={test_metrics['test_mae']:.4f}"
    )

    run.finish()
    return test_metrics


def main():
    parser = argparse.ArgumentParser(
        description="WandB sweep agent — OPTIC-C on APTOS 2019 (fold 0 proxy)")
    parser.add_argument(
        "--aptos_root", default="Datasets/aptos2019-blindness-detection",
        help="Path to APTOS root directory (contains train.csv + train_images/)")
    parser.add_argument(
        "--run_dir", default="runs/aptos_optic_sweep",
        help="Directory for checkpoints")
    parser.add_argument(
        "--sweep_id", type=str, default=None,
        help="W&B sweep ID (entity/project/sweep_id). If given, runs as agent.")
    parser.add_argument(
        "--count", type=int, default=None,
        help="Number of trials for this agent (None = run until sweep is done)")
    args = parser.parse_args()

    if args.sweep_id:
        wandb.agent(
            args.sweep_id,
            function=lambda: run_sweep(args),
            count=args.count,
        )
    else:
        # Single run without sweep controller (for debugging)
        import wandb
        wandb.init(project="espacol-new", config={
            "lambda_proto_ce": 1.0, "lambda_tile_concept": 0.5,
            "lambda_gpa": 0.1, "proto_temperature": 0.15,
            "proto_label_smoothing": 0.07, "new_component_lr_mult": 2.5,
            "backbone_freeze_epochs": 25, "lr_patience": 12,
            "weight_decay": 1e-5, "use_cosine_lr": False,
        })
        run_sweep(args)


if __name__ == "__main__":
    main()
