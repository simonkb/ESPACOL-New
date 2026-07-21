from __future__ import annotations

import csv
import logging
import os
import random
import time
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from configs.config import TrainConfig
from losses.combined import HybridContrastiveOrdinalLoss, compute_class_weights
from losses.prototype import GradePrototypeMemory
from losses.faithfulness import FaithfulnessLoss, soft_occlude
from models.text import ClinicalTextEncoder
from utils.checkpoint import save_checkpoint, load_checkpoint
from utils.metrics import evaluate_predictions


def _compute_batch_concept_cams(
    acts: dict,
    grads: dict,
    layer_indices: list,
    target_h: int,
    target_w: int,
    device: torch.device,
) -> torch.Tensor:
    """
    Batch Grad-CAM++ heatmaps from LayerCAM hook tensors.
    Returns (N, target_h, target_w) float tensor with values in [0, 1].
    Used by the faithfulness training loop — gradient-free (acts/grads are detached).
    """
    N = next(iter(acts.values())).shape[0]
    all_maps = []

    for idx in layer_indices:
        if idx not in acts or idx not in grads:
            continue
        A = acts[idx].float()    # (N, C, H_l, W_l)
        G = grads[idx].float()   # (N, C, H_l, W_l)

        G2 = G ** 2
        G3 = G ** 3
        sum_AG3 = (A * G3).sum(dim=(2, 3), keepdim=True)    # (N, C, 1, 1)
        denom = 2.0 * G2 + sum_AG3 + 1e-8
        alpha = G2 / denom                                    # (N, C, H_l, W_l)
        w_cam = (alpha * F.relu(G)).sum(dim=(2, 3))          # (N, C)

        cam = (w_cam.unsqueeze(-1).unsqueeze(-1) * A).sum(dim=1)  # (N, H_l, W_l)
        cam = F.relu(cam)

        flat = cam.flatten(1)
        lo = flat.min(dim=1)[0].view(N, 1, 1)
        hi = flat.max(dim=1)[0].view(N, 1, 1)
        cam = (cam - lo) / (hi - lo + 1e-8)

        cam_up = F.interpolate(
            cam.unsqueeze(1), size=(target_h, target_w),
            mode="bilinear", align_corners=False,
        ).squeeze(1)  # (N, H, W)
        all_maps.append(cam_up)

    if not all_maps:
        return torch.zeros(N, target_h, target_w, device=device)

    return torch.stack(all_maps).max(dim=0)[0]  # (N, H, W)

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

        # ── Concept spine setup ──────────────────────────────────────────────
        self.concept_text_emb = None
        self.concept_phrases = None
        self.layer_cam = None
        self.faith_loss_fn = None
        use_concept_spine = getattr(cfg, "use_concept_spine", False)
        concept_grade_mask = None

        if use_concept_spine:
            from configs.clinical_text import (
                BUSI_CONCEPTS, DR_CONCEPTS,
                BUSI_CONCEPT_GRADE_MASK, DR_CONCEPT_GRADE_MASK,
            )
            from explainability import LayerCAM

            if self.text_encoder is None:
                raise RuntimeError(
                    "use_concept_spine requires use_image_text=True "
                    "(text encoder must be enabled to encode concept phrases)."
                )

            is_dr = cfg.dataset == "DR"
            self.concept_phrases = DR_CONCEPTS if is_dr else BUSI_CONCEPTS
            concept_grade_mask = DR_CONCEPT_GRADE_MASK if is_dr else BUSI_CONCEPT_GRADE_MASK

            self.concept_text_emb = self.text_encoder.get_concept_embeddings(
                self.concept_phrases
            ).to(self.device)

            self.layer_cam = LayerCAM(self.model)
            self.faith_loss_fn = FaithfulnessLoss(
                margin=getattr(cfg, "faith_margin", 0.1),
                tau=getattr(cfg, "faith_tau", 0.05),
            )
            logger.info(
                f"[Fold {fold}] Concept spine: {len(self.concept_phrases)} concepts | "
                f"faithfulness from epoch {getattr(cfg, 'faith_start_epoch', 10)}, "
                f"every {getattr(cfg, 'faith_every_n', 4)} batches"
            )
        # ────────────────────────────────────────────────────────────────────

        # Differential learning rates: pretrained EfficientNet backbone uses base lr;
        # randomly-initialised modules (CTOT, heads) use transformer_lr_mult × base lr
        # so they can converge at the correct speed without over-updating the backbone.
        use_transformer = getattr(cfg, "use_tile_transformer", False)
        transformer_lr_mult = getattr(cfg, "transformer_lr_mult", 5.0)

        if use_transformer and hasattr(self.model.backbone, "base"):
            backbone_params = list(self.model.backbone.base.parameters())
            non_backbone_params = [
                p for name, p in self.model.named_parameters()
                if not name.startswith("backbone.base")
            ]
            new_module_lr = cfg.lr * transformer_lr_mult
            optim_params = [
                {"params": backbone_params, "lr": cfg.lr},
                {"params": non_backbone_params, "lr": new_module_lr},
            ]
            logger.info(
                f"Differential lr: backbone={cfg.lr:.2e}  "
                f"new_modules={new_module_lr:.2e}  (mult={transformer_lr_mult}×)"
            )
        else:
            optim_params = [{"params": list(self.model.parameters()), "lr": cfg.lr}]

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
        # Snapshot the intended LRs so we can restore them after the frozen phase.
        self._initial_lrs = [pg["lr"] for pg in self.optimizer.param_groups]

        # Backbone freeze warmup: keep EfficientNet frozen for the first N epochs so
        # randomly-initialised CTOT can stabilise before corrupting pretrained features.
        self.ctot_warmup_epochs = (
            getattr(cfg, "ctot_warmup_epochs", 0)
            if use_transformer and hasattr(self.model.backbone, "base")
            else 0
        )
        if self.ctot_warmup_epochs > 0:
            for p in self.model.backbone.base.parameters():
                p.requires_grad_(False)
            logger.info(
                f"[Fold {fold}] Backbone frozen for first {self.ctot_warmup_epochs} epochs (CTOT warmup)"
            )

        # Per-class counts for balanced CORAL pos_weight
        use_ordinal = getattr(cfg, "use_ordinal_head", False)
        ordinal_counts = None
        if use_ordinal:
            from collections import Counter
            cnt = Counter(train_labels)
            ordinal_counts = [cnt.get(k, 0) for k in range(cfg.n_classes)]
            logger.info(f"[Fold {fold}] CORAL class counts for pos_weight: {ordinal_counts}")

        self.criterion = HybridContrastiveOrdinalLoss(
            alpha=cfg.alpha,
            beta=cfg.beta,
            gamma=cfg.gamma,
            temperature=cfg.temperature,
            use_image_text=cfg.use_image_text,
            lambda_ord_it=cfg.lambda_ord_it,
            delta=getattr(cfg, "delta", 0.0),
            eta=getattr(cfg, "eta", 0.0),
            use_concept_spine=use_concept_spine,
            concept_grade_mask=concept_grade_mask,
            n_concepts=getattr(cfg, "n_concepts", 9),
            n_classes=cfg.n_classes,
            use_ordinal_head=use_ordinal,
            ordinal_class_counts=ordinal_counts,
            lambda_tile=getattr(cfg, "lambda_tile", 0.0),
        )

        # Grade Prototype Memory — EMA per-grade prototypes for stable contrastive
        # supervision of rare DR classes (grades 3-4 appear in ~40% of batches)
        self.lambda_proto = getattr(cfg, "lambda_proto", 0.0)
        if self.lambda_proto > 0.0:
            self.proto_memory = GradePrototypeMemory(
                feat_dim=cfg.proj_out_dim,   # 128-dim SCOLw embedding
                n_classes=cfg.n_classes,
                momentum=0.9,
                temperature=0.1,
            ).to(self.device)
            logger.info(
                f"[Fold {fold}] Grade prototype memory: feat_dim={cfg.proj_out_dim} "
                f"K={cfg.n_classes} λ_proto={self.lambda_proto}"
            )
        else:
            self.proto_memory = None

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
            self._maybe_enable_text_finetune(epoch)
            self._maybe_unfreeze_backbone(epoch)

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

        use_tta = getattr(self.cfg, "use_tta", False)
        test_metrics = self._eval_epoch(test_loader, prefix="test", use_tta=use_tta)
        tta_tag = " (TTA×4)" if use_tta else ""
        logger.info(
            f"[Fold {self.fold}] TEST{tta_tag}  acc={test_metrics['test_acc']:.2f}%  "
            f"mae={test_metrics['test_mae']:.4f}"
        )
        return test_metrics

    def _maybe_unfreeze_backbone(self, epoch: int) -> None:
        if self.ctot_warmup_epochs <= 0:
            return
        if epoch != self.ctot_warmup_epochs + 1:
            return
        for p in self.model.backbone.base.parameters():
            p.requires_grad_(True)
        # Restore LRs — the scheduler may have dropped them during the frozen phase
        # because val_loss was meaningless while the backbone was fixed. Reset everything
        # so end-to-end fine-tuning starts with the full intended learning rates.
        for pg, lr in zip(self.optimizer.param_groups, self._initial_lrs):
            pg["lr"] = lr
        self.scheduler.num_bad_epochs = 0
        self.scheduler.best = float("inf")  # mode='min' (val_loss)
        logger.info(
            f"[Fold {self.fold}] Backbone unfrozen at epoch {epoch} — "
            f"LR restored to {self._initial_lrs}, scheduler reset"
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

        use_cs = getattr(self.cfg, "use_concept_spine", False)

        # Refresh concept text embeddings each epoch if text encoder is being fine-tuned
        # (fine-tuning changes the text tower weights, so embeddings drift)
        if use_cs and self._text_finetune_enabled and self.concept_text_emb is not None:
            self.concept_text_emb = self.text_encoder.get_concept_embeddings(
                self.concept_phrases
            ).to(self.device)

        faith_start = getattr(self.cfg, "faith_start_epoch", 10)
        faith_every_n = getattr(self.cfg, "faith_every_n", 4)
        nu_peak = getattr(self.cfg, "nu", 0.0)
        nu_start = getattr(self.cfg, "faith_nu_start", nu_peak)  # default: no ramp
        nu_ramp_epochs = getattr(self.cfg, "faith_nu_ramp_epochs", 1)
        # Compute current epoch's nu via linear curriculum ramp
        if epoch >= faith_start and nu_peak > 0.0:
            ramp_progress = min((epoch - faith_start) / max(nu_ramp_epochs, 1), 1.0)
            nu = nu_start + (nu_peak - nu_start) * ramp_progress
        else:
            nu = 0.0
        do_faith_this_epoch = (
            use_cs
            and self.faith_loss_fn is not None
            and epoch >= faith_start
            and nu_peak > 0.0
        )

        total_loss = total_pcol = total_scolw = total_reg = total_it = 0.0
        total_pic = total_cons = total_faith = total_tile = total_proto = 0.0
        n_batches = 0
        use_ordinal = getattr(self.cfg, "use_ordinal_head", False)
        reg_key = "loss_ord" if use_ordinal else "loss_rmse"

        nb = self.device.type == "cuda"

        mixup_alpha = getattr(self.cfg, "mixup_alpha", 0.0)

        for batch_idx, (x, y) in enumerate(self.train_loader):
            x = x.to(self.device, non_blocking=nb)
            y = y.to(self.device, non_blocking=nb)

            # Mixup: interpolate pairs of inputs in pixel space.
            # Contrastive losses still use original hard labels (y); RMSE sees the
            # mixed input and trains the regression head on a smooth interpolant.
            if mixup_alpha > 0.0:
                lam = random.betavariate(mixup_alpha, mixup_alpha)
                perm = torch.randperm(x.size(0), device=self.device)
                x = lam * x + (1.0 - lam) * x[perm]

            self.optimizer.zero_grad(set_to_none=True)

            batch_weights = compute_class_weights(
                y.cpu().tolist(),
                self.cfg.n_classes,
                device=self.device,
            )

            with autocast(device_type=self.device.type, enabled=self.use_amp):
                out = self.model(
                    x,
                    concept_text_emb=self.concept_text_emb if use_cs else None,
                )

                z_pcol = out["z_pcol"]
                z_scolw = out["z_scolw"]
                z_it = out.get("z_it")
                pred = out["pred"]
                c = out.get("c")              # (N, M) or None
                E = out.get("E")              # (N,)  or None
                ordinal_probs = out.get("ordinal_probs")  # (N, K-1) or None

                text_prototypes = None
                if self.text_encoder is not None and self.cfg.use_image_text:
                    text_prototypes = self.text_encoder()

                main_loss, comps = self.criterion(
                    z_pcol=z_pcol,
                    z_scolw=z_scolw,
                    pred=pred,
                    labels=y,
                    class_weights=batch_weights,
                    z_it=z_it,
                    text_prototypes=text_prototypes,
                    c=c,
                    E=E,
                    ordinal_probs=ordinal_probs,
                    tile_preds=out.get("tile_preds"),
                )

                # Grade prototype contrastive loss (stable signal for rare grades)
                if self.proto_memory is not None and self.proto_memory.is_ready:
                    l_proto = self.proto_memory.prototype_loss(z_scolw, y)
                    main_loss = main_loss + self.lambda_proto * l_proto
                    comps["loss_proto"] = l_proto.item()

            # ── Faithfulness loop ────────────────────────────────────────────
            L_faith_val = 0.0
            do_faith = do_faith_this_epoch and (batch_idx % faith_every_n == 0)

            if do_faith and c is not None and E is not None:
                w_detach = out["w"].detach()

                # Top concept per image by absolute importance score |w_i * c_i|
                top_k_idx = (w_detach * c.detach()).abs().argmax(dim=1)  # (N,)

                # Separate forward pass for CAM — avoids retain_graph=True (halves graph memory)
                self.model.zero_grad()
                with torch.enable_grad():
                    out_cam = self.model(
                        x.detach(), concept_text_emb=self.concept_text_emb
                    )
                    cam_score = (
                        (out_cam["z_c"] * self.concept_text_emb[top_k_idx]).sum(dim=1)
                    ).float().sum()
                    cam_score.backward()  # frees this graph immediately after

                # Extract batch-level heatmaps from hooks (detached, gradient-free)
                # x is (N, T, C, H, W) for multi-tile or (N, C, H, W) for single-tile
                is_tiled = x.dim() == 5
                H, W = (x.shape[3], x.shape[4]) if is_tiled else (x.shape[2], x.shape[3])
                heatmaps = _compute_batch_concept_cams(
                    self.layer_cam._acts,
                    self.layer_cam._grads,
                    self.layer_cam.layer_indices,
                    H, W, self.device,
                )  # (N*T, H, W) for tiled, (N, H, W) for single

                # Clear CAM gradients — main graph (for main_loss) is still alive
                self.model.zero_grad()

                # Soft occlusion of attended region — NO torch.no_grad() so gradients
                # flow back through the soft mask to the concept heads and heatmap head
                if is_tiled:
                    N_b, T_b, C_b, H_b, W_b = x.shape
                    x_occ = soft_occlude(
                        x.view(N_b * T_b, C_b, H_b, W_b), heatmaps,
                        sigma=self.cfg.faith_sigma,
                        kernel_size=self.cfg.faith_blur_kernel,
                    ).view(N_b, T_b, C_b, H_b, W_b)
                else:
                    x_occ = soft_occlude(
                        x, heatmaps,
                        sigma=self.cfg.faith_sigma,
                        kernel_size=self.cfg.faith_blur_kernel,
                    )

                # Second forward for faithfulness loss (gradients flow through c_prime, E_prime)
                with autocast(device_type=self.device.type, enabled=self.use_amp):
                    out_occ = self.model(x_occ, concept_text_emb=self.concept_text_emb)
                    L_faith, faith_comps = self.faith_loss_fn(
                        c=c.detach(),
                        c_prime=out_occ["c"],
                        E=E.detach(),
                        E_prime=out_occ["E"],
                        w=w_detach,
                        top_k_idx=top_k_idx,
                    )

                L_faith_val = faith_comps["loss_faith"]
                # Split backward: backward main first, then faith.
                # Each graph is freed as soon as its backward completes,
                # so peak memory = max(main_graph, occ_graph) not their sum.
                scale = self.scaler.get_scale()
                (main_loss * scale).backward()
                (nu * L_faith * scale).backward()
            else:
                self.scaler.scale(main_loss).backward()
            # ────────────────────────────────────────────────────────────────

            self.scaler.unscale_(self.optimizer)

            clip_params = list(self.model.parameters())
            if self.text_encoder is not None:
                clip_params += list(self.text_encoder.projection.parameters())
                clip_params += self.text_encoder.trainable_text_parameters()

            nn.utils.clip_grad_norm_(clip_params, max_norm=1.0)

            self.scaler.step(self.optimizer)
            self.scaler.update()

            # EMA prototype update — outside autocast, float32 for stability
            if self.proto_memory is not None:
                with torch.no_grad():
                    self.proto_memory.update(z_scolw.float().detach(), y)

            total_loss += comps["loss_total"]
            total_pcol += comps["loss_pcol"]
            total_scolw += comps["loss_scolw"]
            total_reg += comps.get(reg_key, 0.0)
            total_it += comps.get("loss_it", 0.0)
            total_tile += comps.get("loss_tile", 0.0)
            total_proto += comps.get("loss_proto", 0.0)
            total_pic += comps.get("loss_pic", 0.0)
            total_cons += comps.get("loss_cons", 0.0)
            total_faith += L_faith_val
            n_batches += 1

        nb_ = max(n_batches, 1)

        train_reg_key = f"train_{reg_key}"
        metrics = {
            "train_loss": total_loss / nb_,
            "train_loss_pcol": total_pcol / nb_,
            "train_loss_scolw": total_scolw / nb_,
            train_reg_key: total_reg / nb_,
        }

        if self.text_encoder is not None and self.cfg.use_image_text:
            metrics["train_loss_it"] = total_it / nb_

        if getattr(self.cfg, "lambda_tile", 0.0) > 0.0:
            metrics["train_loss_tile"] = total_tile / nb_
        if self.proto_memory is not None:
            metrics["train_loss_proto"] = total_proto / nb_

        if use_cs:
            metrics["train_loss_pic"] = total_pic / nb_
            metrics["train_loss_cons"] = total_cons / nb_
            metrics["train_loss_faith"] = total_faith / nb_

        return metrics

    @torch.no_grad()
    def _tta_predict_batch(self, x: torch.Tensor) -> torch.Tensor:
        """4-fold TTA: average original + H-flip + V-flip + 180° rotation.

        Works for both single-tile (N,C,H,W) and multi-tile (N,T,C,H,W) inputs.
        AttentionPool is order-invariant so per-tile spatial flips are valid.
        """
        augs = [
            x,
            x.flip(-1),                       # horizontal flip
            x.flip(-2),                       # vertical flip
            x.rot90(2, dims=[-2, -1]),        # 180° rotation
        ]
        preds = [self.model.predict(aug) for aug in augs]
        return torch.stack(preds).mean(0)

    @torch.no_grad()
    def _eval_epoch(self, loader: DataLoader, prefix: str, use_tta: bool = False) -> dict:
        self.model.eval()
        if self.text_encoder is not None:
            self.text_encoder.eval()

        use_cs = getattr(self.cfg, "use_concept_spine", False)

        all_preds = []
        all_concept_grades = []
        all_labels = []
        total_rmse = 0.0
        n_batches = 0

        nb = self.device.type == "cuda"

        for x, y in loader:
            x = x.to(self.device, non_blocking=nb)
            y = y.to(self.device, non_blocking=nb)

            with torch.no_grad(), autocast(device_type=self.device.type, enabled=self.use_amp):
                if use_cs and self.concept_text_emb is not None:
                    # Single forward pass — no redundant backbone pass for concept_acc
                    out = self.model(x, concept_text_emb=self.concept_text_emb)
                    pred = out["pred"]
                    if "E" in out:
                        y_concept = self.model.concept_spine.predict_grade(out["E"])
                        all_concept_grades.append(y_concept.cpu())
                elif use_tta:
                    pred = self._tta_predict_batch(x)
                else:
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

        result = {
            f"{prefix}_loss": total_rmse / nbatches,
            f"{prefix}_acc": m["acc"],
            f"{prefix}_mae": m["mae"],
        }

        if all_concept_grades:
            all_cg = torch.cat(all_concept_grades)
            concept_acc = (all_cg == all_labels.long()).float().mean().item() * 100.0
            result[f"{prefix}_concept_acc"] = concept_acc

        return result

    def _log_epoch(
        self,
        epoch: int,
        elapsed: float,
        lr: float,
        train: dict,
        val: dict,
    ) -> None:
        row = {"epoch": epoch, "elapsed": f"{elapsed:.1f}", "lr": lr, **train, **val}

        extras = ""
        if "train_loss_tile" in train:
            extras += f" tile={train['train_loss_tile']:.3f}"
        if "train_loss_proto" in train:
            extras += f" proto={train['train_loss_proto']:.3f}"
        if "train_loss_it" in train:
            extras += f" it={train['train_loss_it']:.3f}"
        if "train_loss_pic" in train:
            extras += f" pic={train['train_loss_pic']:.3f}"
        if "train_loss_cons" in train:
            extras += f" cons={train['train_loss_cons']:.3f}"
        if "train_loss_faith" in train:
            extras += f" faith={train['train_loss_faith']:.3f}"

        concept_acc_text = ""
        if "val_concept_acc" in val:
            concept_acc_text = f"  val_concept_acc={val['val_concept_acc']:.2f}%"

        reg_label = "ord" if "train_loss_ord" in train else "rmse"
        reg_val = train.get("train_loss_ord", train.get("train_loss_rmse", 0.0))
        logger.info(
            f"[Fold {self.fold}] Ep {epoch:3d} | "
            f"loss={train['train_loss']:.4f} "
            f"(pcol={train['train_loss_pcol']:.3f} "
            f"scolw={train['train_loss_scolw']:.3f} "
            f"{reg_label}={reg_val:.3f}"
            f"{extras}) | "
            f"val_loss={val['val_loss']:.4f}  "
            f"val_acc={val['val_acc']:.2f}%  "
            f"val_mae={val['val_mae']:.4f}"
            f"{concept_acc_text}  "
            f"lr={lr:.2e}  t={elapsed:.1f}s"
        )

        with open(self._log_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            if not self._csv_header_written:
                writer.writeheader()
                self._csv_header_written = True
            writer.writerow(row)