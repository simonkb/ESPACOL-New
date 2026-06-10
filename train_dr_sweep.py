from __future__ import annotations

"""
WandB sweep agent for DR image-text extension and ablation study.

This sweep keeps the already-tuned DR baseline hyperparameters fixed and varies only:

  - ablation_mode
  - gamma
  - lambda_ord_it

Ablation modes:
  full_it   : PCOL + SCOLw + ImageText + RMSE
  no_pcol   : SCOLw + ImageText + RMSE
  no_scolw  : PCOL + ImageText + RMSE
  only_it   : ImageText + RMSE

Runs fold 0 only as a fast proxy before full 10-fold evaluation.
"""

import argparse
import csv
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


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_loaders(
    train_items,
    val_items,
    test_items,
    cfg: DRConfig,
    device=None,
    img_cache=None,
):
    train_tfm = build_train_transform(cfg.img_size)
    eval_tfm = build_transform(cfg.img_size)

    use_mps = device is not None and device.type == "mps"
    num_workers = 0 if use_mps else cfg.num_workers
    pin_memory = False if use_mps else cfg.pin_memory
    pf_kwargs = {"prefetch_factor": 4} if num_workers > 0 else {}

    train_ds = ImageLabelDataset(train_items, transform=train_tfm, img_cache=img_cache)
    val_ds = ImageLabelDataset(val_items, transform=eval_tfm, img_cache=img_cache)
    test_ds = ImageLabelDataset(test_items, transform=eval_tfm, img_cache=img_cache)

    if cfg.stratified:
        train_labels = [y for _, y in train_items]
        sampler = StratifiedBatchSampler(
            train_labels,
            batch_size=cfg.batch_size,
            drop_last=True,
            seed=cfg.seed,
        )

        train_loader = DataLoader(
            train_ds,
            batch_sampler=sampler,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=(num_workers > 0),
            **pf_kwargs,
        )
    else:
        train_loader = DataLoader(
            train_ds,
            batch_size=cfg.batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=(num_workers > 0),
            **pf_kwargs,
        )

    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        **pf_kwargs,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        **pf_kwargs,
    )

    return train_loader, val_loader, test_loader


class SweepTrainer(Trainer):
    """Trainer subclass that logs per-epoch metrics to W&B."""

    def fit(self, test_loader) -> dict:
        best_score = -float("inf")
        best_val_acc = -float("inf")
        best_val_mae = float("inf")

        best_ckpt_path = os.path.join(self.run_dir, f"fold{self.fold}_best.pth")

        for epoch in range(1, self.cfg.epochs + 1):
            t0 = time.time()

            train_metrics = self._train_epoch(epoch)
            val_metrics = self._eval_epoch(self.val_loader, prefix="val")

            elapsed = time.time() - t0
            lr_now = self.optimizer.param_groups[0]["lr"]

            self._log_epoch(epoch, elapsed, lr_now, train_metrics, val_metrics)

            val_acc = val_metrics["val_acc"]
            val_mae = val_metrics["val_mae"]
            val_loss = val_metrics["val_loss"]

            # Joint score: prioritize accuracy, but penalize worse MAE.
            # Example: +1% acc is worth roughly 0.1 MAE.
            score = val_acc - 10.0 * val_mae

            is_best = score > best_score
            if is_best:
                best_score = score
                best_val_acc = val_acc
                best_val_mae = val_mae

            wandb.log(
                {
                    "epoch": epoch,
                    "lr": lr_now,
                    "running_best_score": best_score,
                    "running_best_val_acc": best_val_acc,
                    "running_best_val_mae": best_val_mae,
                    **train_metrics,
                    **val_metrics,
                },
                step=epoch,
            )

            # Early abandon for clearly bad runs.
            if (epoch == 10 and val_acc < 50.0) or (
                epoch >= 20 and best_val_acc < 72.0
            ):
                logging.getLogger(__name__).info(
                    f"Abandoning at epoch {epoch}: "
                    f"val_acc={val_acc:.2f}% best_acc={best_val_acc:.2f}%"
                )
                break

            ckpt_path = os.path.join(self.run_dir, f"fold{self.fold}_epoch{epoch}.pth")

            save_checkpoint(
                path=ckpt_path,
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

            if epoch > 1:
                prev_ckpt = os.path.join(
                    self.run_dir, f"fold{self.fold}_epoch{epoch - 1}.pth"
                )
                if os.path.exists(prev_ckpt) and prev_ckpt != best_ckpt_path:
                    os.remove(prev_ckpt)

            self.scheduler.step(val_loss)

            if self.early_stopping.step(val_acc):
                logging.getLogger(__name__).info(f"Early stopping at epoch {epoch}")
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

        wandb.log(
            {
                "fold0_test_acc": test_metrics["test_acc"],
                "fold0_test_mae": test_metrics["test_mae"],
                "best_val_acc": best_val_acc,
                "best_val_mae": best_val_mae,
                "best_score": best_score,
            }
        )

        wandb.summary["fold0_test_acc"] = test_metrics["test_acc"]
        wandb.summary["fold0_test_mae"] = test_metrics["test_mae"]
        wandb.summary["best_val_acc"] = best_val_acc
        wandb.summary["best_val_mae"] = best_val_mae
        wandb.summary["best_score"] = best_score

        return test_metrics


def apply_sweep_config(cfg: DRConfig, wc) -> str:
    """
    Keep baseline hyperparameters fixed and only apply image-text sweep variables.
    """

    # Fixed best DR baseline values.
    cfg.alpha = 0.00662474091401746
    cfg.beta = 0.05516050165777829
    cfg.temperature = 0.7
    cfg.lr = 2e-4
    cfg.weight_decay = 1e-6
    cfg.lr_patience = 8
    cfg.early_stop_patience = 20
    cfg.batch_size = 24
    cfg.epochs = 75

    ablation_mode = str(wc.get("ablation_mode", "full_it"))
    cfg.gamma = float(wc.get("gamma", 0.01))
    cfg.lambda_ord_it = float(wc.get("lambda_ord_it", 1.0))

    if ablation_mode == "full_it":
        cfg.use_image_text = True

    elif ablation_mode == "no_pcol":
        cfg.use_image_text = True
        cfg.alpha = 0.0

    elif ablation_mode == "no_scolw":
        cfg.use_image_text = True
        cfg.beta = 0.0

    elif ablation_mode == "only_it":
        cfg.use_image_text = True
        cfg.alpha = 0.0
        cfg.beta = 0.0

    else:
        raise ValueError(f"Unknown ablation_mode: {ablation_mode}")

    return ablation_mode


def run_sweep(args, img_cache=None):
    run = wandb.init()
    wc = wandb.config

    cfg = DRConfig()
    cfg.run_dir = args.run_dir

    ablation_mode = apply_sweep_config(cfg, wc)

    run_dir = os.path.join(
        args.run_dir,
        f"{run.id}_{ablation_mode}_g{cfg.gamma}_l{cfg.lambda_ord_it}",
    )
    cfg.run_dir = run_dir

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
        force=True,
    )

    log = logging.getLogger("dr_it_sweep")

    log.info("=" * 70)
    log.info("DR Fold-0 Image-Text Ablation Sweep")
    log.info("=" * 70)
    log.info(
        f"Run {run.id}: mode={ablation_mode}, "
        f"alpha={cfg.alpha}, beta={cfg.beta}, gamma={cfg.gamma}, "
        f"lambda_ord_it={cfg.lambda_ord_it}, tau={cfg.temperature}, lr={cfg.lr}"
    )

    wandb.config.update(
        {
            "effective_alpha": cfg.alpha,
            "effective_beta": cfg.beta,
            "effective_gamma": cfg.gamma,
            "effective_lambda_ord_it": cfg.lambda_ord_it,
            "effective_temperature": cfg.temperature,
            "effective_lr": cfg.lr,
            "effective_weight_decay": cfg.weight_decay,
            "effective_batch_size": cfg.batch_size,
            "effective_epochs": cfg.epochs,
        },
        allow_val_change=True,
    )

    set_seed(cfg.seed)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else
        "mps" if torch.backends.mps.is_available() else "cpu"
    )
    log.info(f"Device: {device}")

    ds = DRDataset(root_dir=args.dr_root, split="train", csv_path=args.train_csv)
    all_items = ds.items
    log.info(f"Total DR images: {len(all_items)}")

    if img_cache is None and not args.no_cache:
        cache_dir = args.cache_dir if args.cache_dir else None
        log.info(f"Pre-loading DR images into RAM (disk cache={cache_dir}) ...")
        img_cache = preload_dr_images(
            all_items,
            img_size=cfg.img_size,
            n_threads=16,
            cache_dir=cache_dir,
        )
        log.info(f"Cache ready: {len(img_cache)} images")

    cv = DRCrossValidator(
        all_items,
        n_folds=cfg.n_folds,
        val_fraction=cfg.val_fraction,
        seed=cfg.seed,
    )

    set_seed(cfg.seed)

    train_items_raw, val_items_held_out, test_items = cv.get_fold(0)

    # Same replication protocol: held-out CV fold is validation/test fold.
    train_items = train_items_raw + val_items_held_out
    val_items = test_items

    log.info(f"train={len(train_items)} val=test={len(test_items)}")

    train_loader, val_loader, test_loader = make_loaders(
        train_items,
        val_items,
        test_items,
        cfg,
        device=device,
        img_cache=img_cache,
    )

    fold_dir = os.path.join(run_dir, "fold0")
    os.makedirs(fold_dir, exist_ok=True)

    model = build_model(
        n_classes=cfg.n_classes,
        pretrained=True,
        proj_hidden_dim=cfg.proj_hidden_dim,
        proj_out_dim=cfg.proj_out_dim,
        use_image_text=cfg.use_image_text,
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

    log.info(
        f"Result: mode={ablation_mode} gamma={cfg.gamma} "
        f"lambda={cfg.lambda_ord_it} acc={test_metrics['test_acc']:.2f}% "
        f"mae={test_metrics['test_mae']:.4f}"
    )

    run.finish()
    return test_metrics


def main():
    parser = argparse.ArgumentParser(description="WandB sweep agent for DR image-text ablation")

    parser.add_argument("--dr_root", type=str, default="Datasets/DR")
    parser.add_argument("--train_csv", type=str, default=None)
    parser.add_argument("--run_dir", type=str, default="runs/dr_it_sweep")

    parser.add_argument(
        "--no_cache",
        action="store_true",
        help="Disable RAM image cache",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default="Datasets/DR/train_cache",
        help="Directory for pre-decoded .pt image cache",
    )

    parser.add_argument(
        "--sweep_id",
        type=str,
        default=None,
        help="W&B sweep ID: entity/project/sweep_id. If provided, runs in-process.",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=None,
        help="Number of trials for this agent.",
    )

    # W&B sweep params, accepted so argparse does not fail in subprocess mode.
    parser.add_argument("--ablation_mode", type=str, default=None)
    parser.add_argument("--gamma", type=float, default=None)
    parser.add_argument("--lambda_ord_it", type=float, default=None)

    args = parser.parse_args()

    if args.train_csv is None:
        args.train_csv = os.path.join(args.dr_root, "trainLabels.csv")

    if args.sweep_id:
        img_cache = None

        if not args.no_cache:
            ds = DRDataset(root_dir=args.dr_root, split="train", csv_path=args.train_csv)
            cache_dir = args.cache_dir if args.cache_dir else None

            print(f"Pre-loading DR images once for this agent (disk cache={cache_dir}) ...")
            img_cache = preload_dr_images(
                ds.items,
                img_size=300,
                n_threads=16,
                cache_dir=cache_dir,
            )
            print(f"Cache ready: {len(img_cache)} images")

        wandb.agent(
            args.sweep_id,
            function=lambda: run_sweep(args, img_cache=img_cache),
            count=args.count,
        )
    else:
        run_sweep(args)


if __name__ == "__main__":
    main()