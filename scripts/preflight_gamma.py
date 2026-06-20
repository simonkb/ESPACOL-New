#!/usr/bin/env python
"""
Pre-flight check for the gamma (image-text) port — runs on CPU.

When run on the login node (with internet) it ALSO populates TORCH_HOME / HF_HOME
with the EfficientNet ImageNet weights and the BioMedCLIP text encoder, so the
offline GPU compute nodes can load them from cache.

Validates, end to end:
  1. model(use_image_text=True) forward returns z_it; the image-text loss
     produces a finite, non-zero L_IT; backward populates image_text_head grads.
  2. A full Trainer.fit() run on a tiny BUSI subset (frozen text encoder).
  3. A full Trainer.fit() run with text fine-tuning enabled (DR-style path),
     which exercises locating + unfreezing the BioMedCLIP text layers.

Exit code 0 and "PREFLIGHT_OK" printed => the port is wired correctly.
"""
from __future__ import annotations

import os
import sys
import logging

import torch

# Make repo importable when run from anywhere
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("preflight")

from configs.config import BUSIConfig
from models.framework import build_model
from losses.combined import HybridContrastiveOrdinalLoss, compute_class_weights
from models.text import ClinicalTextEncoder
from configs.clinical_text import BUSI_CLASS_DESCRIPTIONS
from training.trainer import Trainer

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BUSI_ROOT = os.path.join(REPO, "Datasets", "BUSI")


def phase1_manual_step():
    log.info("\n[phase 1] manual forward/loss/backward with z_it + text prototypes")
    cfg = BUSIConfig()
    model = build_model(
        n_classes=cfg.n_classes, pretrained=True,
        proj_hidden_dim=cfg.proj_hidden_dim, proj_out_dim=cfg.proj_out_dim,
        use_image_text=True,
    ).to(DEVICE).train()

    text_encoder = ClinicalTextEncoder(
        model_name=cfg.text_encoder_name,
        class_descriptions=BUSI_CLASS_DESCRIPTIONS,
        proj_out_dim=cfg.proj_out_dim,
        device=DEVICE,
    ).to(DEVICE)

    criterion = HybridContrastiveOrdinalLoss(
        alpha=cfg.alpha, beta=cfg.beta, gamma=cfg.gamma,
        temperature=cfg.temperature, use_image_text=True,
        lambda_ord_it=cfg.lambda_ord_it,
    )

    x = torch.randn(6, 3, cfg.img_size, cfg.img_size, device=DEVICE)
    y = torch.tensor([0, 1, 2, 0, 1, 2], device=DEVICE)
    w = compute_class_weights(y.cpu().tolist(), cfg.n_classes, device=DEVICE)

    out = model(x)
    assert out["z_it"] is not None, "z_it head missing — use_image_text not wired"
    proto = text_encoder()
    assert proto.shape == (cfg.n_classes, cfg.proj_out_dim), f"bad proto shape {proto.shape}"

    loss, comps = criterion(
        z_pcol=out["z_pcol"], z_scolw=out["z_scolw"], pred=out["pred"],
        labels=y, class_weights=w, z_it=out["z_it"], text_prototypes=proto,
    )
    assert torch.isfinite(loss), f"loss not finite: {loss}"
    assert comps["loss_it"] > 0.0, f"loss_it should be > 0, got {comps['loss_it']}"
    loss.backward()
    grad = model.image_text_head.net[0].weight.grad
    assert grad is not None and torch.isfinite(grad).all(), "image_text_head got no grad"
    log.info(f"  OK: total={comps['loss_total']:.4f}  it={comps['loss_it']:.4f}  "
             f"pcol={comps['loss_pcol']:.4f}  scolw={comps['loss_scolw']:.4f}  rmse={comps['loss_rmse']:.4f}")


def _tiny_busi_loaders(cfg, per_class_train=8, per_class_eval=4):
    from train_busi import load_all_busi_items, make_loaders
    from collections import defaultdict
    items = load_all_busi_items(BUSI_ROOT)
    by_c = defaultdict(list)
    for p, yy in items:
        by_c[yy].append((p, yy))
    train_items, eval_items = [], []
    for yy, lst in by_c.items():
        train_items += lst[:per_class_train]
        eval_items += lst[per_class_train:per_class_train + per_class_eval]
    tl, vl, te = make_loaders(train_items, eval_items, eval_items, cfg, device=DEVICE)
    return tl, vl, te, [y for _, y in train_items]


def phase_fit(name, finetune):
    log.info(f"\n[{name}] Trainer.fit on tiny BUSI subset (finetune_text={finetune})")
    cfg = BUSIConfig(run_dir=f"/tmp/preflight_{name}")
    cfg.epochs = 2
    if finetune:
        cfg.finetune_text_encoder = True
        cfg.text_finetune_layers = 2
        cfg.text_finetune_start_epoch = 1
    tl, vl, te, train_labels = _tiny_busi_loaders(cfg)
    model = build_model(
        n_classes=cfg.n_classes, pretrained=True,
        proj_hidden_dim=cfg.proj_hidden_dim, proj_out_dim=cfg.proj_out_dim,
        use_image_text=cfg.use_image_text,
    )
    trainer = Trainer(model, tl, vl, cfg, run_dir=cfg.run_dir,
                      train_labels=train_labels, device=DEVICE, fold=0)
    metrics = trainer.fit(te)
    assert all(isinstance(v, (int, float)) for v in metrics.values()), metrics
    log.info(f"  OK: {metrics}")


if __name__ == "__main__":
    log.info(f"device={DEVICE}  torch={torch.__version__}")
    phase1_manual_step()
    phase_fit("frozen", finetune=False)
    phase_fit("finetune", finetune=True)
    log.info("\nPREFLIGHT_OK")
