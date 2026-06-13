"""
WandB Bayesian hyperparameter sweep for DR — BiomedCLIP backbone.

Sweeps: alpha, beta, gamma, temperature, lr (heads).
Fixed:  freeze_backbone_epochs=10, image_encoder_lr=2e-5, fold=0.

Usage
-----
1. Create sweep on WandB (run once locally):
       python train_dr_hparam_sweep.py --create_sweep \
           --wandb_project <project> --wandb_entity <entity>
   Copy the printed sweep ID.

2. Launch agent on remote (can run multiple agents in parallel):
       HF_HUB_OFFLINE=1 python train_dr_hparam_sweep.py \
           --dr_root Datasets/DR --run_dir runs/dr_hparam_sweep \
           --sweep_id <entity>/<project>/<sweep_id> --count 30 --cache_dir ""
"""

from __future__ import annotations

import argparse
import logging
import os
import random
import sys
import time

import numpy as np
import torch
import wandb
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(__file__))

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
from utils.checkpoint import save_checkpoint, load_checkpoint


# ---------------------------------------------------------------------------
# WandB sweep config (random, 40 trials)
#
# Method is random (not bayes) so that zero values are sampled uniformly —
# a Bayesian optimizer would deprioritize 0s after the first few trials and
# we'd never get the ablation coverage we want.
#
# alpha/beta/gamma each include 0.0 (ablation — no that loss) and 1.0
# (maximum weight — stress-test that loss).  This means ~1/7 of trials will
# have alpha=0, ~1/7 with beta=0, etc., giving a natural ablation study
# embedded in the sweep at no extra cost.
# ---------------------------------------------------------------------------

SWEEP_CONFIG = {
    "method": "random",
    "metric": {"name": "best_val_acc", "goal": "maximize"},
    "parameters": {
        # 0.0 = no PCOL loss,  1.0 = maximum PCOL weight
        "alpha": {
            "distribution": "categorical",
            "values": [0.0, 0.001, 0.005, 0.01, 0.02, 0.05, 1.0],
        },
        # 0.0 = no SCOLw loss, 1.0 = maximum SCOLw weight
        "beta": {
            "distribution": "categorical",
            "values": [0.0, 0.01, 0.05, 0.1, 0.2, 0.3, 1.0],
        },
        # 0.0 = no IT loss (text encoder disabled), 1.0 = maximum IT weight
        "gamma": {
            "distribution": "categorical",
            "values": [0.0, 0.01, 0.05, 0.1, 0.2, 1.0],
        },
        "temperature": {
            "distribution": "categorical",
            "values": [0.3, 0.5, 0.7, 1.0],
        },
        "lr": {
            "distribution": "categorical",
            "values": [5e-5, 1e-4, 2e-4, 5e-4],
        },
    },
}


# ---------------------------------------------------------------------------
# Fixed sweep settings (from v5 — known-good training scheme)
# ---------------------------------------------------------------------------

FIXED = dict(
    freeze_backbone_epochs=10,
    image_encoder_lr=2e-5,
    epochs=70,
    lr_patience=12,
    lr_min=1e-7,
    early_stop_patience=20,
    lr_factor=0.2,
    batch_size=24,
    weight_decay=1e-6,
    lambda_ord_it=2.0,
    finetune_text_encoder=True,
    text_finetune_layers=2,
    text_encoder_lr=1e-6,
    text_finetune_start_epoch=20,
    use_image_text=True,
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------

def make_loaders(train_items, val_items, test_items, cfg: DRConfig, device=None, img_cache=None):
    clip_norm = getattr(cfg, "use_clip_normalization", True)
    train_tfm = build_train_transform(cfg.img_size, use_clip_norm=clip_norm)
    eval_tfm  = build_transform(cfg.img_size, use_clip_norm=clip_norm)

    use_mps   = device is not None and device.type == "mps"
    nw        = 0 if use_mps else cfg.num_workers
    pm        = False if use_mps else cfg.pin_memory
    pf        = {"prefetch_factor": 4} if nw > 0 else {}

    train_ds = ImageLabelDataset(train_items, transform=train_tfm, img_cache=img_cache)
    val_ds   = ImageLabelDataset(val_items,   transform=eval_tfm,  img_cache=img_cache)
    test_ds  = ImageLabelDataset(test_items,  transform=eval_tfm,  img_cache=img_cache)

    train_labels = [y for _, y in train_items]
    sampler = StratifiedBatchSampler(train_labels, batch_size=cfg.batch_size, drop_last=True, seed=cfg.seed)

    train_loader = DataLoader(train_ds, batch_sampler=sampler,
                              num_workers=nw, pin_memory=pm,
                              persistent_workers=(nw > 0), **pf)
    val_loader   = DataLoader(val_ds,   batch_size=cfg.batch_size, shuffle=False,
                              num_workers=nw, pin_memory=pm, **pf)
    test_loader  = DataLoader(test_ds,  batch_size=cfg.batch_size, shuffle=False,
                              num_workers=nw, pin_memory=pm, **pf)

    return train_loader, val_loader, test_loader, train_labels


# ---------------------------------------------------------------------------
# Sweep trainer (adds WandB logging)
# ---------------------------------------------------------------------------

class HparamSweepTrainer(Trainer):
    """Trainer subclass that logs per-epoch metrics to W&B and supports early abandon."""

    def fit(self, test_loader) -> dict:
        best_val_acc  = -float("inf")
        best_val_mae  = float("inf")
        best_ckpt_path = os.path.join(self.run_dir, "fold0_best.pth")

        log = logging.getLogger(__name__)

        for epoch in range(1, self.cfg.epochs + 1):
            t0 = time.time()
            self._maybe_unfreeze_backbone(epoch)
            self._maybe_enable_text_finetune(epoch)

            train_metrics = self._train_epoch(epoch)
            val_metrics   = self._eval_epoch(self.val_loader, prefix="val")

            elapsed = time.time() - t0
            lr_now  = self.optimizer.param_groups[0]["lr"]

            self._log_epoch(epoch, elapsed, lr_now, train_metrics, val_metrics)

            val_acc  = val_metrics["val_acc"]
            val_mae  = val_metrics["val_mae"]
            val_loss = val_metrics["val_loss"]

            is_best = val_acc > best_val_acc
            if is_best:
                best_val_acc = val_acc
                best_val_mae = val_mae
                save_checkpoint(
                    path=best_ckpt_path,
                    model=self.model,
                    optimizer=None,
                    epoch=epoch,
                    metrics={**train_metrics, **val_metrics},
                    is_best=False,
                    text_encoder=None,
                )

            wandb.log(
                {
                    "epoch": epoch,
                    "lr": lr_now,
                    "best_val_acc": best_val_acc,
                    "best_val_mae": best_val_mae,
                    **train_metrics,
                    **val_metrics,
                },
                step=epoch,
            )

            # Early abandon: clearly bad runs.
            # Thresholds are lenient to allow ablation runs (single loss,
            # gamma=0, etc.) enough time to converge before being abandoned.
            if epoch == 25 and best_val_acc < 35.0:
                log.info(f"Abandoning at epoch {epoch}: best_val_acc={best_val_acc:.2f}% < 35%")
                break
            if epoch == 45 and best_val_acc < 55.0:
                log.info(f"Abandoning at epoch {epoch}: best_val_acc={best_val_acc:.2f}% < 55%")
                break

            self.scheduler.step(val_loss)

            if self.early_stopping.step(val_acc):
                log.info(f"Early stopping at epoch {epoch}")
                break

        if os.path.exists(best_ckpt_path):
            load_checkpoint(
                path=best_ckpt_path,
                model=self.model,
                optimizer=None,
                text_encoder=self.text_encoder,
                device=self.device,
            )

        test_metrics = self._eval_epoch(test_loader, prefix="test")

        wandb.log({
            "fold0_test_acc": test_metrics["test_acc"],
            "fold0_test_mae": test_metrics["test_mae"],
            "best_val_acc":   best_val_acc,
            "best_val_mae":   best_val_mae,
        })
        wandb.summary["fold0_test_acc"] = test_metrics["test_acc"]
        wandb.summary["fold0_test_mae"] = test_metrics["test_mae"]
        wandb.summary["best_val_acc"]   = best_val_acc
        wandb.summary["best_val_mae"]   = best_val_mae

        return test_metrics


# ---------------------------------------------------------------------------
# Single sweep run
# ---------------------------------------------------------------------------

def run_sweep(args, img_cache=None):
    run = wandb.init()
    wc  = wandb.config

    cfg = DRConfig()

    # Apply fixed settings
    for k, v in FIXED.items():
        setattr(cfg, k, v)

    # Apply swept parameters
    cfg.alpha       = float(wc.get("alpha",       0.00662))
    cfg.beta        = float(wc.get("beta",        0.0552))
    cfg.gamma       = float(wc.get("gamma",       0.05))
    cfg.temperature = float(wc.get("temperature", 0.7))
    cfg.lr          = float(wc.get("lr",          2e-4))

    # gamma=0 means no image-text loss — disable text encoder entirely to
    # save memory and skip the text encoder forward pass each epoch.
    if cfg.gamma == 0.0:
        cfg.use_image_text = False
        cfg.finetune_text_encoder = False

    # Derive a human-readable ablation label for logging.
    ablation_parts = []
    if cfg.alpha == 0.0:
        ablation_parts.append("no_pcol")
    if cfg.beta == 0.0:
        ablation_parts.append("no_scolw")
    if cfg.gamma == 0.0:
        ablation_parts.append("no_it")
    ablation_label = "+".join(ablation_parts) if ablation_parts else "full"

    run_dir = os.path.join(args.run_dir, run.id)
    os.makedirs(run_dir, exist_ok=True)
    cfg.run_dir = run_dir

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
        force=True,
    )

    log = logging.getLogger("hparam_sweep")
    log.info(
        f"Run {run.id} [{ablation_label}]: "
        f"alpha={cfg.alpha} beta={cfg.beta} gamma={cfg.gamma} "
        f"tau={cfg.temperature} lr={cfg.lr:.2e} use_it={cfg.use_image_text}"
    )

    wandb.config.update(
        {
            "effective_alpha":        cfg.alpha,
            "effective_beta":         cfg.beta,
            "effective_gamma":        cfg.gamma,
            "effective_temperature":  cfg.temperature,
            "effective_lr":           cfg.lr,
            "ablation_label":         ablation_label,
            "use_image_text":         cfg.use_image_text,
            "image_encoder_lr":       cfg.image_encoder_lr,
            "freeze_backbone_epochs": cfg.freeze_backbone_epochs,
            "epochs":                 cfg.epochs,
        },
        allow_val_change=True,
    )

    set_seed(cfg.seed)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else
        "mps"  if torch.backends.mps.is_available() else "cpu"
    )

    ds        = DRDataset(root_dir=args.dr_root, split="train", csv_path=args.train_csv)
    all_items = ds.items

    if img_cache is None and not args.no_cache:
        cache_dir = args.cache_dir if args.cache_dir else None
        log.info(f"Pre-loading DR images (disk cache={cache_dir}) ...")
        img_cache = preload_dr_images(all_items, img_size=cfg.img_size, n_threads=16, cache_dir=cache_dir)
        log.info(f"Cache ready: {len(img_cache)} images")

    cv = DRCrossValidator(all_items, n_folds=cfg.n_folds, val_fraction=cfg.val_fraction, seed=cfg.seed)
    set_seed(cfg.seed)

    train_items_raw, val_items_held_out, test_items = cv.get_fold(0)
    train_items = train_items_raw + val_items_held_out
    val_items   = test_items  # same protocol as main training

    log.info(f"train={len(train_items)} val=test={len(test_items)}")

    train_loader, val_loader, test_loader, train_labels = make_loaders(
        train_items, val_items, test_items, cfg, device=device, img_cache=img_cache,
    )

    fold_dir = os.path.join(run_dir, "fold0")
    os.makedirs(fold_dir, exist_ok=True)

    model = build_model(
        n_classes=cfg.n_classes,
        pretrained=True,
        proj_hidden_dim=cfg.proj_hidden_dim,
        proj_out_dim=cfg.proj_out_dim,
        use_image_text=cfg.use_image_text,
        image_encoder_name=cfg.image_encoder_name,
    )

    trainer = HparamSweepTrainer(
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

    log.info(
        f"Result: acc={test_metrics['test_acc']:.2f}% mae={test_metrics['test_mae']:.4f} | "
        f"alpha={cfg.alpha:.4f} beta={cfg.beta:.4f} gamma={cfg.gamma:.4f} "
        f"tau={cfg.temperature} lr={cfg.lr:.2e}"
    )

    run.finish()
    return test_metrics


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="WandB Bayesian hparam sweep for DR")

    parser.add_argument("--dr_root",   type=str, default="Datasets/DR")
    parser.add_argument("--train_csv", type=str, default=None)
    parser.add_argument("--run_dir",   type=str, default="runs/dr_hparam_sweep")

    parser.add_argument("--no_cache",  action="store_true")
    parser.add_argument("--cache_dir", type=str, default="",
                        help="Disk cache dir for pre-decoded images. Empty string = RAM only.")

    parser.add_argument("--sweep_id",  type=str, default=None,
                        help="WandB sweep ID (entity/project/sweep_id). Runs the agent.")
    parser.add_argument("--count",     type=int, default=40,
                        help="Number of trials for this agent.")

    parser.add_argument("--create_sweep", action="store_true",
                        help="Create the WandB sweep and print the ID, then exit.")
    parser.add_argument("--wandb_project", type=str, default="espacol-dr-hparam")
    parser.add_argument("--wandb_entity",  type=str, default=None)

    args = parser.parse_args()

    if args.train_csv is None:
        args.train_csv = os.path.join(args.dr_root, "trainLabels.csv")

    # --- Create sweep and exit ---
    if args.create_sweep:
        sweep_id = wandb.sweep(
            SWEEP_CONFIG,
            project=args.wandb_project,
            entity=args.wandb_entity,
        )
        print(f"\nSweep created: {sweep_id}")
        print(f"Run agent with: --sweep_id {args.wandb_entity}/{args.wandb_project}/{sweep_id}")
        return

    # --- Run agent ---
    if args.sweep_id:
        img_cache = None

        if not args.no_cache:
            ds        = DRDataset(root_dir=args.dr_root, split="train", csv_path=args.train_csv)
            cache_dir = args.cache_dir if args.cache_dir else None
            print(f"Pre-loading DR images once for this agent (disk cache={cache_dir}) ...")
            img_cache = preload_dr_images(ds.items, img_size=224, n_threads=16, cache_dir=cache_dir)
            print(f"Cache ready: {len(img_cache)} images")

        wandb.agent(
            args.sweep_id,
            function=lambda: run_sweep(args, img_cache=img_cache),
            count=args.count,
        )
    else:
        # Single trial with current DRConfig defaults (for local testing)
        run_sweep(args)


if __name__ == "__main__":
    main()
