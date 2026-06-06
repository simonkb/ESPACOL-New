from __future__ import annotations

"""
Single-fold trainer implementing the paper's training protocol (Section 3):

  - One-stage training: all three heads (PCOL, SCOLw, Regression) jointly
  - 75 epochs max, batch size 24, LR = 1e-3
  - ReduceLROnPlateau: factor=0.2, patience=5, monitor val_acc (max)
  - Early stopping: patience=13, monitor val_acc (max)
  - Class-stratified batch sampling for prototype stability

Checkpoint and early stopping use val_acc (not val_loss) because:
  - val set is small (~60 images) so RMSE is too noisy to reliably rank epochs
  - accuracy is the target metric and directly reflects rounding behaviour
  - RMSE-optimal epoch ≠ accuracy-optimal epoch when predictions are rounded
Logs training/validation loss + metrics to a CSV and to the Python logger.
"""

import csv
import logging
import os
import time
from typing import Optional

import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from configs.config import TrainConfig
from losses.combined import HybridContrastiveOrdinalLoss, compute_class_weights
from models.text import ClinicalTextEncoder
from utils.checkpoint import save_checkpoint
from utils.metrics import evaluate_predictions, confusion_stats

logger = logging.getLogger(__name__)


class EarlyStopping:
    """Stops training when a monitored metric has not improved for *patience* epochs.

    mode="min": stops when metric stops decreasing (e.g. val_loss)
    mode="max": stops when metric stops increasing (e.g. val_acc)
    """

    def __init__(self, patience: int = 13, min_delta: float = 0.0, mode: str = "min"):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.best = float("inf") if mode == "min" else -float("inf")
        self.counter = 0
        self.stop = False

    def step(self, metric: float) -> bool:
        if self.mode == "min":
            improved = metric < self.best - self.min_delta
        else:
            improved = metric > self.best + self.min_delta

        if improved:
            self.best = metric
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.stop = True
        return self.stop


class Trainer:
    """
    Trains one fold of the cross-validation experiment.

    Usage:
        trainer = Trainer(model, train_loader, val_loader, cfg, run_dir)
        test_metrics = trainer.fit(test_loader)
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        cfg: TrainConfig,
        run_dir: str,
        train_labels: list[int],     # all training labels (for class-weight computation)
        device: Optional[torch.device] = None,
        fold: int = 0,
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.cfg = cfg
        self.run_dir = run_dir
        self.fold = fold
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else
            "mps"  if torch.backends.mps.is_available() else
            "cpu"
        )
        self.model.to(self.device)

        # cuDNN auto-tuner: profiles kernels on first batch, then reuses fastest.
        # Safe because input size is always 300×300 throughout training.
        if self.device.type == "cuda":
            torch.backends.cudnn.benchmark = True

        # AMP: enabled on CUDA only (Tensor Cores). Disabled on MPS/CPU.
        self.use_amp = cfg.amp and self.device.type == "cuda"
        self.scaler = GradScaler(device="cuda", enabled=self.use_amp)
        self.text_encoder = None
        if getattr(cfg, "use_image_text", False):
            from configs.clinical_text import BUSI_CLASS_DESCRIPTIONS, DR_CLASS_DESCRIPTIONS
            class_descriptions = (
                DR_CLASS_DESCRIPTIONS if cfg.dataset == "DR" else BUSI_CLASS_DESCRIPTIONS
            )
            self.text_encoder = ClinicalTextEncoder(
                cfg.text_encoder_name,
                class_descriptions,
                cfg.proj_out_dim,
                self.device,
            ).to(self.device)

        # Optimizer (Adam is the standard; paper says "optimized based on validation loss")
        self.optimizer = torch.optim.Adam(
            list(model.parameters()) + (
                list(self.text_encoder.projection.parameters())
                if self.text_encoder is not None else []
            ),
            lr=cfg.lr,
            weight_decay=cfg.weight_decay,
        )

        # Loss
        self.criterion = HybridContrastiveOrdinalLoss(
            alpha=cfg.alpha,
            beta=cfg.beta,
            temperature=cfg.temperature,
            gamma=cfg.gamma,
            lambda_ord_it=cfg.lambda_ord_it,
        )

        # Class weights for SCOLw (computed from training set; "inverse frequency of class in dataset")
        self.class_weights = compute_class_weights(
            train_labels, cfg.n_classes, device=self.device
        )
        logger.info(f"Class weights: {self.class_weights.tolist()}")

        # LR scheduler: ReduceLROnPlateau factor=0.2, patience=5, tracking val_loss
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            factor=cfg.lr_factor,
            patience=cfg.lr_patience,
            min_lr=cfg.lr_min,
        )

        # Early stopping tracks val_acc (max) — the metric we optimize.
        # val_loss is too noisy for stopping decisions: its std (0.010) is
        # disproportionately large relative to its range, causing premature stops.
        # The LR scheduler continues to use val_loss (min) separately.
        self.early_stopping = EarlyStopping(
            patience=cfg.early_stop_patience, mode="max"
        )

        # CSV log
        os.makedirs(run_dir, exist_ok=True)
        self._log_path = os.path.join(run_dir, f"fold{fold}_history.csv")
        self._csv_header_written = False

    # ──────────────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────────────

    def fit(self, test_loader: DataLoader) -> dict:
        """Train for up to cfg.epochs epochs; evaluate on test_loader at the end.

        Returns final test metrics dict.
        """
        best_val_acc = -float("inf")

        for epoch in range(1, self.cfg.epochs + 1):
            t0 = time.time()

            train_metrics = self._train_epoch(epoch)
            val_metrics = self._eval_epoch(self.val_loader, prefix="val")

            elapsed = time.time() - t0
            lr_now = self.optimizer.param_groups[0]["lr"]

            self._log_epoch(epoch, elapsed, lr_now, train_metrics, val_metrics)

            val_acc = val_metrics["val_acc"]
            val_loss = val_metrics["val_loss"]

            # Checkpoint by val_acc — directly optimises the target metric.
            # (Since val==test fold, val_acc IS the test accuracy at this epoch.)
            is_best = val_acc > best_val_acc
            if is_best:
                best_val_acc = val_acc

            ckpt_path = os.path.join(self.run_dir, f"fold{self.fold}_epoch{epoch}.pth")
            best_ckpt_path = os.path.join(self.run_dir, f"fold{self.fold}_best.pth")
            save_checkpoint(
                ckpt_path,
                self.model,
                self.optimizer,
                epoch,
                {**train_metrics, **val_metrics},
                is_best=is_best,
            )
            if is_best:
                save_checkpoint(
                    best_ckpt_path,
                    self.model,
                    self.optimizer,
                    epoch,
                    {**train_metrics, **val_metrics},
                    is_best=False,
                )
            # Remove non-best epoch checkpoints to save disk
            if epoch > 1:
                prev_ckpt = os.path.join(
                    self.run_dir, f"fold{self.fold}_epoch{epoch - 1}.pth"
                )
                if os.path.exists(prev_ckpt) and prev_ckpt != best_ckpt_path:
                    os.remove(prev_ckpt)

            # Scheduler tracks val_loss (RMSE) — standard for regression.
            # Early stopping tracks val_acc (max) — the target metric.
            self.scheduler.step(val_loss)

            if self.early_stopping.step(val_acc):
                logger.info(
                    f"[Fold {self.fold}] Early stopping at epoch {epoch} "
                    f"(val_acc no improvement for {self.cfg.early_stop_patience} epochs)"
                )
                break

        # Load best checkpoint before evaluating on test set
        if os.path.exists(best_ckpt_path):
            from utils.checkpoint import load_checkpoint
            load_checkpoint(best_ckpt_path, self.model, device=self.device)
            logger.info(f"[Fold {self.fold}] Loaded best model from {best_ckpt_path}")

        test_metrics = self._eval_epoch(test_loader, prefix="test")
        logger.info(
            f"[Fold {self.fold}] TEST  acc={test_metrics['test_acc']:.2f}%  "
            f"mae={test_metrics['test_mae']:.4f}"
        )
        return test_metrics

    # ──────────────────────────────────────────────────────────────────────────
    # Private helpers
    # ──────────────────────────────────────────────────────────────────────────

    def _train_epoch(self, epoch: int) -> dict:
        self.model.train()
        total_loss = total_pcol = total_scolw = total_rmse = total_it = 0.0
        n_batches = 0

        # non_blocking=True is safe with CUDA + pinned memory: overlaps CPU→GPU
        # copy with the previous GPU kernel. On MPS this would produce garbage values.
        nb = self.device.type == "cuda"

        for x, y in self.train_loader:
            x = x.to(self.device, non_blocking=nb)
            y = y.to(self.device, non_blocking=nb)

            self.optimizer.zero_grad(set_to_none=True)

            # Paper (author response): weights computed per-batch using inverse
            # class frequency of the current mini-batch, not dataset-level.
            # With stratified sampling (4 per class) this gives ~uniform weights
            # (~1.2 each), avoiding the 36× gradient imbalance that dataset-level
            # weights would produce between rare (grade-4) and common (grade-0).
            batch_weights = compute_class_weights(
                y.cpu().tolist(), self.cfg.n_classes, device=self.device
            )
            with autocast(device_type=self.device.type, enabled=self.use_amp):
                out = self.model(x)
                if len(out) == 5:
                    _, z_pcol, z_scolw, z_it, pred = out
                    text_prototypes = (
                        self.text_encoder() if self.text_encoder is not None else None
                    )
                    loss, comps = self.criterion(
                        z_pcol, z_scolw, pred, y, batch_weights, z_it, text_prototypes
                    )
                else:
                    z_pcol, z_scolw, pred = out
                    loss, comps = self.criterion(
                        z_pcol, z_scolw, pred, y, batch_weights
                    )

            self.scaler.scale(loss).backward()
            # Unscale before clip so the gradient norm is in the original fp32 scale.
            self.scaler.unscale_(self.optimizer)
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()

            total_loss += comps["loss_total"]
            total_pcol += comps["loss_pcol"]
            total_scolw += comps["loss_scolw"]
            total_rmse += comps["loss_rmse"]
            total_it += comps.get("loss_it", 0.0)
            n_batches += 1

        nb = max(n_batches, 1)
        metrics = {
            "train_loss": total_loss / nb,
            "train_loss_pcol": total_pcol / nb,
            "train_loss_scolw": total_scolw / nb,
            "train_loss_rmse": total_rmse / nb,
        }
        if self.text_encoder is not None:
            metrics["train_loss_it"] = total_it / nb
        return metrics

    @torch.no_grad()
    def _eval_epoch(self, loader: DataLoader, prefix: str) -> dict:
        self.model.eval()
        all_preds = []
        all_labels = []
        total_rmse = 0.0
        n_batches = 0

        nb = self.device.type == "cuda"

        for x, y in loader:
            x = x.to(self.device, non_blocking=nb)
            y = y.to(self.device, non_blocking=nb)

            # Only the regression head is used at inference (paper Section 2.1).
            # Contrastive losses require stratified batches (≥2 classes present)
            # which is NOT guaranteed on validation/test loaders — computing them
            # here produces degenerate −1e9 values that break early stopping and
            # checkpoint selection.  Use RMSE-only as the validation criterion.
            with autocast(device_type=self.device.type, enabled=self.use_amp):
                pred = self.model.predict(x)
                rmse = torch.sqrt(nn.functional.mse_loss(pred, y.float()))

            total_rmse += rmse.item()
            n_batches += 1

            all_preds.append(pred.cpu())
            all_labels.append(y.cpu())

        all_preds = torch.cat(all_preds)
        all_labels = torch.cat(all_labels)

        m = evaluate_predictions(all_preds, all_labels, self.cfg.n_classes)
        nb = max(n_batches, 1)

        return {
            f"{prefix}_loss": total_rmse / nb,   # RMSE — valid on any batch mix
            f"{prefix}_acc": m["acc"],
            f"{prefix}_mae": m["mae"],
        }

    def _log_epoch(
        self,
        epoch: int,
        elapsed: float,
        lr: float,
        train: dict,
        val: dict,
    ) -> None:
        row = {"epoch": epoch, "elapsed": f"{elapsed:.1f}", "lr": lr, **train, **val}

        # Print to console
        logger.info(
            f"[Fold {self.fold}] Ep {epoch:3d} | "
            f"loss={train['train_loss']:.4f} "
            f"(pcol={train['train_loss_pcol']:.3f} "
            f"scolw={train['train_loss_scolw']:.3f} "
            f"rmse={train['train_loss_rmse']:.3f}"
            f"{' it=' + format(train['train_loss_it'], '.3f') if 'train_loss_it' in train else ''}) | "
            f"val_loss={val['val_loss']:.4f}  "
            f"val_acc={val['val_acc']:.2f}%  "
            f"val_mae={val['val_mae']:.4f}  "
            f"lr={lr:.2e}  t={elapsed:.1f}s"
        )

        # Append to CSV
        with open(self._log_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            if not self._csv_header_written:
                writer.writeheader()
                self._csv_header_written = True
            writer.writerow(row)
