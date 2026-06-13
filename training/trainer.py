from __future__ import annotations

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
from utils.checkpoint import save_checkpoint, load_checkpoint
from utils.metrics import evaluate_predictions

logger = logging.getLogger(__name__)


class EarlyStopping:
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
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        cfg: TrainConfig,
        run_dir: str,
        train_labels: list[int],
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
            "mps" if torch.backends.mps.is_available() else
            "cpu"
        )
        self.model.to(self.device)

        if self.device.type == "cuda":
            torch.backends.cudnn.benchmark = True

        self.use_amp = cfg.amp and self.device.type == "cuda"
        self.scaler = GradScaler(device="cuda", enabled=self.use_amp)

        self.text_encoder = None
        if getattr(cfg, "use_image_text", False):
            from configs.clinical_text import BUSI_CLASS_DESCRIPTIONS, DR_CLASS_DESCRIPTIONS

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

        image_encoder_lr = getattr(cfg, "image_encoder_lr", cfg.lr)
        self._image_encoder_lr = image_encoder_lr
        self._freeze_backbone_epochs = getattr(cfg, "freeze_backbone_epochs", 0)
        self._backbone_unfrozen = self._freeze_backbone_epochs == 0

        backbone_params = list(self.model.backbone.parameters())
        backbone_param_ids = {id(p) for p in backbone_params}
        head_params = [p for p in self.model.parameters()
                       if id(p) not in backbone_param_ids]

        if self._freeze_backbone_epochs > 0:
            # Freeze backbone for warmup: heads initialize from strong pretrained
            # features before any ViT weights are disturbed.
            for p in backbone_params:
                p.requires_grad = False
            optim_params = [
                {"params": head_params, "lr": cfg.lr},
            ]
            logger.info(
                f"Backbone frozen for first {self._freeze_backbone_epochs} epochs "
                f"— heads only: {sum(p.numel() for p in head_params):,} params "
                f"lr={cfg.lr:.2e}"
            )
        else:
            optim_params = [
                {"params": backbone_params, "lr": image_encoder_lr},
                {"params": head_params,     "lr": cfg.lr},
            ]
            logger.info(
                f"Optimizer param groups — backbone: {sum(p.numel() for p in backbone_params):,} params "
                f"lr={image_encoder_lr:.2e} | heads: {sum(p.numel() for p in head_params):,} params "
                f"lr={cfg.lr:.2e}"
            )

        if self.text_encoder is not None:
            optim_params.append(
                {"params": list(self.text_encoder.projection.parameters()), "lr": cfg.lr}
            )
            if getattr(cfg, "finetune_text_encoder", False):
                n_text_params = self.text_encoder.set_text_finetune(True)
                text_params = self.text_encoder.trainable_text_parameters()
                self.text_encoder.set_text_finetune(False)
                if text_params:
                    optim_params.append(
                        {
                            "params": text_params,
                            "lr": getattr(cfg, "text_encoder_lr", 1e-6),
                        }
                    )
                logger.info(
                    f"Text encoder fine-tuning: layers={getattr(cfg, 'text_finetune_layers', 0)} "
                    f"start_epoch={getattr(cfg, 'text_finetune_start_epoch', 1)} "
                    f"trainable_params={n_text_params}"
                )

        self.optimizer = torch.optim.Adam(
            optim_params,
            weight_decay=cfg.weight_decay,
        )

        self.criterion = HybridContrastiveOrdinalLoss(
            alpha=cfg.alpha,
            beta=cfg.beta,
            gamma=cfg.gamma,
            temperature=cfg.temperature,
            use_image_text=cfg.use_image_text,
            lambda_ord_it=cfg.lambda_ord_it,
        )

        self.class_weights = compute_class_weights(
            train_labels, cfg.n_classes, device=self.device
        )
        logger.info(f"Class weights: {self.class_weights.tolist()}")

        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            factor=cfg.lr_factor,
            patience=cfg.lr_patience,
            min_lr=cfg.lr_min,
        )

        self.early_stopping = EarlyStopping(
            patience=cfg.early_stop_patience,
            mode="max",
        )

        os.makedirs(run_dir, exist_ok=True)
        self._log_path = os.path.join(run_dir, f"fold{fold}_history.csv")
        self._csv_header_written = False
        self._text_finetune_enabled = False

    def fit(self, test_loader: DataLoader) -> dict:
        best_val_acc = -float("inf")
        best_ckpt_path = os.path.join(self.run_dir, f"fold{self.fold}_best.pth")

        for epoch in range(1, self.cfg.epochs + 1):
            t0 = time.time()
            self._maybe_unfreeze_backbone(epoch)
            self._maybe_enable_text_finetune(epoch)

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
                # Save weights-only best checkpoint (no optimizer state, no full
                # text encoder) to keep disk usage minimal (~350 MB vs ~2 GB).
                save_checkpoint(
                    path=best_ckpt_path,
                    model=self.model,
                    optimizer=None,
                    epoch=epoch,
                    metrics={**train_metrics, **val_metrics},
                    is_best=False,
                    text_encoder=None,
                )

            self.scheduler.step(val_loss)

            if self.early_stopping.step(val_acc):
                logger.info(
                    f"[Fold {self.fold}] Early stopping at epoch {epoch} "
                    f"(val_acc no improvement for {self.cfg.early_stop_patience} epochs)"
                )
                break

        if os.path.exists(best_ckpt_path):
            load_checkpoint(
                path=best_ckpt_path,
                model=self.model,
                optimizer=None,
                text_encoder=self.text_encoder,
                device=self.device,
            )
            logger.info(f"[Fold {self.fold}] Loaded best model from {best_ckpt_path}")

        test_metrics = self._eval_epoch(test_loader, prefix="test")
        logger.info(
            f"[Fold {self.fold}] TEST  acc={test_metrics['test_acc']:.2f}%  "
            f"mae={test_metrics['test_mae']:.4f}"
        )
        return test_metrics

    def _maybe_unfreeze_backbone(self, epoch: int) -> None:
        if self._backbone_unfrozen:
            return
        if epoch < self._freeze_backbone_epochs + 1:
            return
        for p in self.model.backbone.parameters():
            p.requires_grad = True
        self.optimizer.add_param_group({
            "params": list(self.model.backbone.parameters()),
            "lr": self._image_encoder_lr,
        })
        # ReduceLROnPlateau stores min_lrs indexed by param group. Adding a new
        # group via add_param_group() doesn't update that list, causing an
        # IndexError on the next scheduler.step(). Extend it manually.
        self.scheduler.min_lrs.append(self.cfg.lr_min)
        # Reset patience counter so the unfreeze-induced loss change doesn't
        # immediately trigger a LR reduction.
        self.scheduler.num_bad_epochs = 0
        self._backbone_unfrozen = True
        n = sum(p.numel() for p in self.model.backbone.parameters())
        logger.info(
            f"[Fold {self.fold}] Backbone unfrozen at epoch {epoch} "
            f"({n:,} params, lr={self._image_encoder_lr:.2e}) — full model training starts"
        )

    def _maybe_enable_text_finetune(self, epoch: int) -> None:
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
            f"({n_text_params} trainable text params, lr={getattr(self.cfg, 'text_encoder_lr', 1e-6):.2e})"
        )

    def _train_epoch(self, epoch: int) -> dict:
        self.model.train()
        if self.text_encoder is not None:
            self.text_encoder.train()
            if (
                getattr(self.cfg, "finetune_text_encoder", False)
                and not self._text_finetune_enabled
            ):
                self.text_encoder.text_model.eval()

        total_loss = 0.0
        total_pcol = 0.0
        total_scolw = 0.0
        total_rmse = 0.0
        total_it = 0.0
        n_batches = 0

        nb = self.device.type == "cuda"

        for x, y in self.train_loader:
            x = x.to(self.device, non_blocking=nb)
            y = y.to(self.device, non_blocking=nb)

            self.optimizer.zero_grad(set_to_none=True)

            batch_weights = compute_class_weights(
                y.cpu().tolist(),
                self.cfg.n_classes,
                device=self.device,
            )

            with autocast(device_type=self.device.type, enabled=self.use_amp):
                out = self.model(x)

                if isinstance(out, dict):
                    z_pcol = out["z_pcol"]
                    z_scolw = out["z_scolw"]
                    z_it = out.get("z_it", None)
                    pred = out["pred"]
                elif len(out) == 5:
                    _, z_pcol, z_scolw, z_it, pred = out
                else:
                    z_pcol, z_scolw, pred = out
                    z_it = None

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

        nbatches = max(n_batches, 1)

        metrics = {
            "train_loss": total_loss / nbatches,
            "train_loss_pcol": total_pcol / nbatches,
            "train_loss_scolw": total_scolw / nbatches,
            "train_loss_rmse": total_rmse / nbatches,
        }

        if self.text_encoder is not None and self.cfg.use_image_text:
            metrics["train_loss_it"] = total_it / nbatches

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
        nbatches = max(n_batches, 1)

        return {
            f"{prefix}_loss": total_rmse / nbatches,
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

        with open(self._log_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            if not self._csv_header_written:
                writer.writeheader()
                self._csv_header_written = True
            writer.writerow(row)