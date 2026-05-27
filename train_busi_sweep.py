"""
Wandb sweep agent for BUSI hyperparameter tuning.

Runs fold 0 only (fastest proxy for full CV performance).
Optimizes: alpha, beta, temperature, weight_decay, lr.

Usage:
  # 1. Create the sweep (once):
  wandb sweep sweep_config.yaml

  # 2. Start agents (one per GPU/machine, or run sequentially):
  wandb agent <entity>/<project>/<sweep_id>

  # Or run a single sweep trial with default wandb.config:
  python train_busi_sweep.py --busi_root Datasets/BUSI
"""

import argparse
import logging
import os
import random
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))

import wandb
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


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_loaders(train_items, val_items, test_items, cfg: BUSIConfig, device=None):
    train_tfm = build_train_transform(cfg.img_size)
    eval_tfm = build_transform(cfg.img_size)

    use_mps = device is not None and device.type == "mps"
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
        val_ds, batch_size=cfg.batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_ds, batch_size=cfg.batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory,
    )
    return train_loader, val_loader, test_loader


class SweepTrainer(Trainer):
    """Trainer subclass that logs per-epoch metrics to wandb."""

    def fit(self, test_loader) -> dict:
        best_val_acc = -float("inf")
        best_ckpt_path = os.path.join(self.run_dir, f"fold{self.fold}_best.pth")

        for epoch in range(1, self.cfg.epochs + 1):
            import time
            t0 = time.time()

            train_metrics = self._train_epoch(epoch)
            val_metrics = self._eval_epoch(self.val_loader, prefix="val")

            elapsed = time.time() - t0
            lr_now = self.optimizer.param_groups[0]["lr"]

            self._log_epoch(epoch, elapsed, lr_now, train_metrics, val_metrics)

            val_acc = val_metrics["val_acc"]
            val_loss = val_metrics["val_loss"]

            # Log to wandb every epoch
            wandb.log({
                "epoch": epoch,
                "lr": lr_now,
                **train_metrics,
                **val_metrics,
            }, step=epoch)

            is_best = val_acc > best_val_acc
            if is_best:
                best_val_acc = val_acc

            ckpt_path = os.path.join(self.run_dir, f"fold{self.fold}_epoch{epoch}.pth")
            from utils.checkpoint import save_checkpoint
            save_checkpoint(
                ckpt_path, self.model, self.optimizer, epoch,
                {**train_metrics, **val_metrics}, is_best=is_best,
            )
            if is_best:
                save_checkpoint(
                    best_ckpt_path, self.model, self.optimizer, epoch,
                    {**train_metrics, **val_metrics}, is_best=False,
                )
            if epoch > 1:
                prev_ckpt = os.path.join(self.run_dir, f"fold{self.fold}_epoch{epoch - 1}.pth")
                if os.path.exists(prev_ckpt) and prev_ckpt != best_ckpt_path:
                    os.remove(prev_ckpt)

            self.scheduler.step(val_loss)
            if self.early_stopping.step(val_loss):
                logging.getLogger(__name__).info(
                    f"Early stopping at epoch {epoch}"
                )
                break

        if os.path.exists(best_ckpt_path):
            from utils.checkpoint import load_checkpoint
            load_checkpoint(best_ckpt_path, self.model, device=self.device)

        test_metrics = self._eval_epoch(test_loader, prefix="test")
        # Log final test metrics as summary values
        wandb.log({
            "fold0_test_acc": test_metrics["test_acc"],
            "fold0_test_mae": test_metrics["test_mae"],
        })
        wandb.summary["fold0_test_acc"] = test_metrics["test_acc"]
        wandb.summary["fold0_test_mae"] = test_metrics["test_mae"]
        wandb.summary["best_val_acc"] = best_val_acc
        return test_metrics


def run_sweep(args):
    run = wandb.init()
    wc = wandb.config

    # Build config from sweep params (override defaults with wandb.config values)
    cfg = BUSIConfig()
    cfg.alpha = float(wc.get("alpha", cfg.alpha))
    cfg.beta = float(wc.get("beta", cfg.beta))
    cfg.temperature = float(wc.get("temperature", cfg.temperature))
    cfg.weight_decay = float(wc.get("weight_decay", cfg.weight_decay))
    cfg.lr = float(wc.get("lr", cfg.lr))

    run_dir = os.path.join(args.run_dir, run.id)
    cfg.run_dir = run_dir

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log = logging.getLogger("sweep")
    log.info(f"Run {run.id}: alpha={cfg.alpha}, beta={cfg.beta}, "
             f"tau={cfg.temperature}, wd={cfg.weight_decay}, lr={cfg.lr}")

    set_seed(cfg.seed)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else
        "mps"  if torch.backends.mps.is_available() else
        "cpu"
    )
    log.info(f"Device: {device}")

    ds = BUSIDataset(root_dir=args.busi_root, split="all")
    all_items = ds.items
    log.info(f"Total BUSI images: {len(all_items)}")

    cv = BUSICrossValidator(
        all_items, n_folds=cfg.n_folds, val_fraction=cfg.val_fraction, seed=cfg.seed
    )

    # Fold 0 only
    set_seed(cfg.seed + 0)
    train_items_raw, val_items_held_out, test_items = cv.get_fold(0)
    train_items = train_items_raw + val_items_held_out
    val_items = test_items

    log.info(f"train={len(train_items)}  val=test={len(test_items)}")

    train_loader, val_loader, test_loader = make_loaders(
        train_items, val_items, test_items, cfg, device=device
    )

    fold_dir = os.path.join(run_dir, "fold0")
    os.makedirs(fold_dir, exist_ok=True)

    model = build_model(
        n_classes=cfg.n_classes,
        pretrained=True,
        proj_hidden_dim=cfg.proj_hidden_dim,
        proj_out_dim=cfg.proj_out_dim,
    )

    train_labels = [y for _, y in train_items]
    trainer = SweepTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        cfg=cfg,
        run_dir=fold_dir,
        train_labels=train_labels,
        device=device,
        fold=0,
    )

    test_metrics = trainer.fit(test_loader)
    log.info(f"Result: acc={test_metrics['test_acc']:.2f}%  mae={test_metrics['test_mae']:.4f}")
    run.finish()
    return test_metrics


def main():
    parser = argparse.ArgumentParser(description="Wandb sweep agent for BUSI")
    parser.add_argument("--busi_root", type=str, default="Datasets/BUSI")
    parser.add_argument("--run_dir", type=str, default="runs/busi_sweep")
    # Sweep parameters — wandb agent passes these as CLI args
    parser.add_argument("--alpha", type=float, default=None)
    parser.add_argument("--beta", type=float, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--weight_decay", type=float, default=None)
    parser.add_argument("--lr", type=float, default=None)
    args = parser.parse_args()

    run_sweep(args)


if __name__ == "__main__":
    main()
