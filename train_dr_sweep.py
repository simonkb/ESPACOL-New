"""
WandB sweep agent for DR hyperparameter tuning.

Runs fold 0 only (fastest proxy for full 10-fold CV performance).
Optimizes: alpha, beta, temperature, weight_decay, lr.

Usage:
  # 1. Create the sweep (once):
  wandb sweep sweep_config_dr.yaml

  # 2. Start agent:
  wandb agent <entity>/<project>/<sweep_id>

  # Or run a single trial manually:
  python train_dr_sweep.py --dr_root Datasets/DR
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
from configs.config import DRConfig
from Datasets.dataloaders import (
    DRDataset,
    ImageLabelDataset,
    StratifiedBatchSampler,
    build_transform,
    build_train_transform,
    preload_dr_images,
)
from models.framework import build_model
from training.cross_val import DRCrossValidator
from training.trainer import Trainer
from torch.utils.data import DataLoader


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_loaders(train_items, val_items, test_items, cfg: DRConfig, device=None, img_cache=None):
    train_tfm = build_train_transform(cfg.img_size)
    eval_tfm = build_transform(cfg.img_size)

    use_mps = device is not None and device.type == "mps"
    num_workers = 0 if use_mps else cfg.num_workers
    pin_memory = False if use_mps else cfg.pin_memory
    prefetch = 4 if num_workers > 0 else None
    persistent = num_workers > 0

    train_ds = ImageLabelDataset(train_items, transform=train_tfm, img_cache=img_cache)
    val_ds   = ImageLabelDataset(val_items,   transform=eval_tfm,  img_cache=img_cache)
    test_ds  = ImageLabelDataset(test_items,  transform=eval_tfm,  img_cache=img_cache)

    if cfg.stratified:
        train_labels = [y for _, y in train_items]
        sampler = StratifiedBatchSampler(
            train_labels, batch_size=cfg.batch_size, drop_last=True, seed=cfg.seed
        )
        train_loader = DataLoader(
            train_ds, batch_sampler=sampler,
            num_workers=num_workers, pin_memory=pin_memory,
            persistent_workers=persistent, prefetch_factor=prefetch,
        )
    else:
        train_loader = DataLoader(
            train_ds, batch_size=cfg.batch_size, shuffle=True, drop_last=True,
            num_workers=num_workers, pin_memory=pin_memory,
            persistent_workers=persistent, prefetch_factor=prefetch,
        )

    val_loader = DataLoader(
        val_ds, batch_size=cfg.batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory, prefetch_factor=prefetch,
    )
    test_loader = DataLoader(
        test_ds, batch_size=cfg.batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory, prefetch_factor=prefetch,
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

            is_best = val_acc > best_val_acc
            if is_best:
                best_val_acc = val_acc

            # running_best_val_acc logged every epoch — required for WandB
            # hyperband early termination to compare trials at the same step.
            wandb.log({
                "epoch": epoch,
                "lr": lr_now,
                "running_best_val_acc": best_val_acc,
                **train_metrics,
                **val_metrics,
            }, step=epoch)

            # Two-stage early abandon.
            # Epoch 10: use CURRENT val_acc (not best) to catch oscillating/
            #           unstable runs where one lucky epoch inflated best_val_acc.
            #           A run heading to 83%+ should be consistently above 50% by ep 10.
            # Epoch 20: use best_val_acc — must have broken through the naive
            #           ~74% baseline at least once to be a candidate for 83%+.
            if (epoch == 10 and val_acc < 50.0) or \
               (epoch >= 20 and best_val_acc < 72.0):
                logging.getLogger(__name__).info(
                    f"Abandoning at epoch {epoch}: "
                    f"val_acc={val_acc:.1f}%  best={best_val_acc:.1f}%"
                )
                break

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
            if self.early_stopping.step(val_acc):
                logging.getLogger(__name__).info(f"Early stopping at epoch {epoch}")
                break

        if os.path.exists(best_ckpt_path):
            from utils.checkpoint import load_checkpoint
            load_checkpoint(best_ckpt_path, self.model, device=self.device)

        test_metrics = self._eval_epoch(test_loader, prefix="test")
        wandb.log({
            "fold0_test_acc": test_metrics["test_acc"],
            "fold0_test_mae": test_metrics["test_mae"],
        })
        wandb.summary["fold0_test_acc"] = test_metrics["test_acc"]
        wandb.summary["fold0_test_mae"] = test_metrics["test_mae"]
        wandb.summary["best_val_acc"] = best_val_acc
        return test_metrics


def run_sweep(args, img_cache=None):
    run = wandb.init()
    wc = wandb.config

    cfg = DRConfig()
    cfg.alpha        = float(wc.get("alpha",        cfg.alpha))
    cfg.beta         = float(wc.get("beta",         cfg.beta))
    cfg.temperature  = float(wc.get("temperature",  cfg.temperature))
    cfg.weight_decay = float(wc.get("weight_decay", cfg.weight_decay))
    cfg.lr           = float(wc.get("lr",           cfg.lr))
    cfg.lr_patience  = int(wc.get("lr_patience",   cfg.lr_patience))
    cfg.epochs       = int(wc.get("epochs",         cfg.epochs))
    cfg.batch_size   = int(wc.get("batch_size",     cfg.batch_size))
    cfg.early_stop_patience = int(wc.get("early_stop_patience", cfg.early_stop_patience))

    run_dir = os.path.join(args.run_dir, run.id)
    cfg.run_dir = run_dir

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log = logging.getLogger("dr_sweep")
    log.info(
        f"Run {run.id}: alpha={cfg.alpha:.5f}, beta={cfg.beta:.5f}, "
        f"tau={cfg.temperature}, wd={cfg.weight_decay}, lr={cfg.lr}"
    )

    set_seed(cfg.seed)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else
        "mps"  if torch.backends.mps.is_available() else
        "cpu"
    )
    log.info(f"Device: {device}")

    ds = DRDataset(root_dir=args.dr_root, split="train", csv_path=args.train_csv)
    all_items = ds.items
    log.info(f"Total DR images: {len(all_items)}")

    # Load image cache if not pre-loaded (16 threads — I/O bound, safe to oversubscribe)
    if img_cache is None and not args.no_cache:
        log.info("Pre-loading DR images into RAM (16 threads) ...")
        img_cache = preload_dr_images(all_items, img_size=cfg.img_size, n_threads=16)
        log.info(f"Cache ready: {len(img_cache)} images")

    cv = DRCrossValidator(
        all_items, n_folds=cfg.n_folds, val_fraction=cfg.val_fraction, seed=cfg.seed
    )

    set_seed(cfg.seed + 0)
    train_items_raw, val_items_held_out, test_items = cv.get_fold(0)
    train_items = train_items_raw + val_items_held_out
    val_items = test_items
    log.info(f"train={len(train_items)}  val=test={len(test_items)}")

    train_loader, val_loader, test_loader = make_loaders(
        train_items, val_items, test_items, cfg, device=device, img_cache=img_cache
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
    parser = argparse.ArgumentParser(description="WandB sweep agent for DR")
    parser.add_argument("--dr_root",   type=str, default="Datasets/DR")
    parser.add_argument("--train_csv", type=str, default=None)
    parser.add_argument("--run_dir",   type=str, default="runs/dr_sweep")
    parser.add_argument("--no_cache",  action="store_true",
                        help="Disable RAM image cache (use if RAM < 12 GB free)")
    parser.add_argument(
        "--cache_dir", type=str, default="Datasets/DR/train_cache",
        help="Directory for pre-decoded .pt files — built once, reused on restarts"
    )
    parser.add_argument(
        "--sweep_id", type=str, default=None,
        help="WandB sweep ID (entity/project/sweep_id). When provided, runs all "
             "trials in-process so the image cache is loaded once and shared across "
             "all trials. Without this flag the agent spawns a new subprocess per "
             "trial and reloads the cache each time."
    )
    parser.add_argument("--count", type=int, default=None,
                        help="Max number of trials to run (default: unlimited)")
    # WandB agent also passes sweep params as CLI args when running via subprocess —
    # declare them so argparse doesn't error. Values are read from wandb.config.
    parser.add_argument("--alpha",        type=float, default=None)
    parser.add_argument("--beta",         type=float, default=None)
    parser.add_argument("--temperature",  type=float, default=None)
    parser.add_argument("--weight_decay", type=float, default=None)
    parser.add_argument("--lr",           type=float, default=None)
    parser.add_argument("--lr_patience",  type=int,   default=None)
    parser.add_argument("--epochs",       type=int,   default=None)
    parser.add_argument("--batch_size",   type=int,   default=None)
    parser.add_argument("--early_stop_patience", type=int, default=None)
    args = parser.parse_args()

    if args.train_csv is None:
        args.train_csv = os.path.join(args.dr_root, "trainLabels.csv")

    if args.sweep_id:
        # ── In-process agent: cache once, reuse across all trials ────────────
        img_cache = None
        if not args.no_cache:
            from Datasets.dataloaders import DRDataset
            ds = DRDataset(root_dir=args.dr_root, split="train", csv_path=args.train_csv)
            cache_dir = args.cache_dir if args.cache_dir else None
            print(f"Pre-loading DR images into RAM (16 threads, disk cache: {cache_dir}) ...")
            img_cache = preload_dr_images(ds.items, img_size=300, n_threads=16, cache_dir=cache_dir)
            print(f"Cache ready: {len(img_cache)} images — shared across all trials")

        wandb.agent(
            args.sweep_id,
            function=lambda: run_sweep(args, img_cache=img_cache),
            count=args.count,
        )
    else:
        # ── Subprocess mode (wandb agent CLI spawns this script per trial) ───
        run_sweep(args)


if __name__ == "__main__":
    main()
