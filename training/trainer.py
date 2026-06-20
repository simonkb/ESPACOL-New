from __future__ import annotations

"""
Single-fold trainer implementing the paper's training protocol (Section 3),
extended with the gamma image-text ordinal loss:

  - One-stage training: PCOL, SCOLw, (Image-Text), Regression heads jointly
  - 75 epochs max, batch size 24
  - ReduceLROnPlateau: factor=0.2, patience=5, monitor val_loss (min)
  - Early stopping: monitor val_acc (max)
  - Class-stratified batch sampling for prototype stability

Gamma extension:
  - When cfg.use_image_text, a BioMedCLIP text encoder produces class text
    prototypes; the model's image_text_head produces z_it; the loss adds
    gamma * L_IT.  The image backbone stays EfficientNet-V2S.
  - The text encoder's projection (and, for DR, the top text transformer layers
    from text_finetune_start_epoch onward) are trained alongside the model.

Checkpoint and early stopping use val_acc (not val_loss) because:
  - val set is small so RMSE is too noisy to reliably rank epochs
  - accuracy is the target metric and directly reflects rounding behaviour
The text encoder is NOT needed at inference (eval uses only the regression head),
so checkpoints store the model only.
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

        # ── Gamma image-text extension: BioMedCLIP text encoder ────────────────
        # Builds class text prototypes (frozen by default; for DR the top text
        # layers are unfrozen from text_finetune_start_epoch).  Only the text
        # side is added — the image backbone remains EfficientNet-V2S.
        self.text_encoder = None
        self._text_finetune_enabled = False
        if getattr(cfg, "use_image_text", False):
            from configs.clinical_text import (
                BUSI_CLASS_DESCRIPTIONS,
                DR_CLASS_DESCRIPTIONS,
            )

            class_descriptions = (
                DR_CLASS_DESCRIPTIONS if cfg.dataset == "DR" else BUSI_CLASS_DESCRIPTIONS
            )
            self.text_encoder = ClinicalTextEncoder(
                model_name=cfg.text_encoder_name,
                class_descriptions=class_descriptions,
                proj_out_dim=cfg.proj_out_dim,
                device=self.device,
                finetune_text_encoder=getattr(cfg, "finetune_text_encoder", False),
                finetune_layers=getattr(cfg, "text_finetune_layers", 0),
            ).to(self.device)

        # Optimizer (Adam). The whole EfficientNet model trains at cfg.lr (as on
        # this branch — no separate backbone LR). The text encoder's projection
        # head trains at cfg.lr too; for DR the top text layers are added as a
        # separate group (requires_grad toggled on at text_finetune_start_epoch).
        optim_params = [{"params": list(self.model.parameters()), "lr": cfg.lr}]
        if self.text_encoder is not None:
            optim_params.append(
                {"params": list(self.text_encoder.projection.parameters()), "lr": cfg.lr}
            )
            if getattr(cfg, "finetune_text_encoder", False):
                self.text_encoder.set_text_finetune(True)
                text_params = self.text_encoder.trainable_text_parameters()
                self.text_encoder.set_text_finetune(False)  # off until start_epoch
                if text_params:
                    optim_params.append(
                        {"params": text_params, "lr": getattr(cfg, "text_encoder_lr", 1e-6)}
                    )
                logger.info(
                    f"Text fine-tuning configured: layers={getattr(cfg, 'text_finetune_layers', 0)} "
                    f"start_epoch={getattr(cfg, 'text_finetune_start_epoch', 1)} "
                    f"lr={getattr(cfg, 'text_encoder_lr', 1e-6):.2e}"
                )

        self.optimizer = torch.optim.Adam(optim_params, weight_decay=cfg.weight_decay)

        # Loss (gamma image-text term active only when cfg.use_image_text and gamma>0)
        self.criterion = HybridContrastiveOrdinalLoss(
            alpha=cfg.alpha,
            beta=cfg.beta,
            gamma=getattr(cfg, "gamma", 0.0),
            temperature=cfg.temperature,
            use_image_text=getattr(cfg, "use_image_text", False),
            lambda_ord_it=getattr(cfg, "lambda_ord_it", 1.0),
        )

        # Class weights for SCOLw (computed from training set; inverse frequency)
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

            self._maybe_enable_text_finetune(epoch)

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

    def _maybe_enable_text_finetune(self, epoch: int) -> None:
        """Unfreeze the top text transformer layers at text_finetune_start_epoch.

        The layers are already registered as an optimizer param group (with
        requires_grad=False); flipping requires_grad lets Adam start updating them.
        """
        if self.text_encoder is None or self._text_finetune_enabled:
            return
        if not getattr(self.cfg, "finetune_text_encoder", False):
            return
        start_epoch = getattr(self.cfg, "text_finetune_start_epoch", 1)
        if epoch < start_epoch:
            return
        n_text_params = self.text_encoder.set_text_finetune(True)
        self._text_finetune_enabled = True
        logger.info(
            f"[Fold {self.fold}] Enabled text encoder fine-tuning at epoch {epoch} "
            f"({n_text_params} trainable text params, "
            f"lr={getattr(self.cfg, 'text_encoder_lr', 1e-6):.2e})"
        )

    def _train_epoch(self, epoch: int) -> dict:
        self.model.train()
        if self.text_encoder is not None:
            self.text_encoder.train()
            # Keep the frozen text transformer in eval mode until fine-tuning starts
            # (avoids dropout/LN-stat drift on the frozen prototypes).
            if (
                getattr(self.cfg, "finetune_text_encoder", False)
                and not self._text_finetune_enabled
            ):
                self.text_encoder.text_model.eval()

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
            batch_weights = compute_class_weights(
                y.cpu().tolist(), self.cfg.n_classes, device=self.device
            )
            with autocast(device_type=self.device.type, enabled=self.use_amp):
                out = self.model(x)
                z_pcol = out["z_pcol"]
                z_scolw = out["z_scolw"]
                z_it = out.get("z_it", None)
                pred = out["pred"]

                text_prototypes = None
                if self.text_encoder is not None and self.cfg.use_image_text:
                    text_prototypes = self.text_encoder()

                loss, comps = self.criterion(
                    z_pcol=z_pcol,
                    z_scolw=z_scolw,
                    pred=pred,
                    labels=y,
                    class_weights=batch_weights,
                    z_it=z_it,
                    text_prototypes=text_prototypes,
                )

            self.scaler.scale(loss).backward()
            # Unscale before clip so the gradient norm is in the original fp32 scale.
            self.scaler.unscale_(self.optimizer)
            clip_params = list(self.model.parameters())
            if self.text_encoder is not None:
                clip_params += list(self.text_encoder.projection.parameters())
                clip_params += self.text_encoder.trainable_text_parameters()
            nn.utils.clip_grad_norm_(clip_params, max_norm=1.0)
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
        if self.text_encoder is not None and self.cfg.use_image_text:
            metrics["train_loss_it"] = total_it / nb
        return metrics

    @torch.no_grad()
    def _eval_epoch(self, loader: DataLoader, prefix: str) -> dict:
        self.model.eval()
        if self.text_encoder is not None:
            self.text_encoder.eval()
        all_preds = []
        all_labels = []
        total_rmse = 0.0
        n_batches = 0

        nb = self.device.type == "cuda"

        for x, y in loader:
            x = x.to(self.device, non_blocking=nb)
            y = y.to(self.device, non_blocking=nb)

            # Only the regression head is used at inference (paper Section 2.1).
            # Contrastive / image-text losses require stratified batches and the
            # text encoder, neither needed for the target metric.
            with autocast(device_type=self.device.type, enabled=self.use_amp):
                pred = self.model.predict(x)
                rmse = torch.sqrt(nn.functional.mse_loss(pred, y.float()) + 1e-8)

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

        it_text = ""
        if "train_loss_it" in train:
            it_text = f" it={train['train_loss_it']:.3f}"

        # Print to console
        logger.info(
            f"[Fold {self.fold}] Ep {epoch:3d} | "
            f"loss={train['train_loss']:.4f} "
            f"(pcol={train['train_loss_pcol']:.3f} "
            f"scolw={train['train_loss_scolw']:.3f} "
            f"rmse={train['train_loss_rmse']:.3f}"
            f"{it_text}) | "
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
