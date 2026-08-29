"""
Optuna hyperparameter search for OPTIC-C on APTOS 2019.

Uses SQLite so multiple SLURM array tasks share one Optuna study — no wandb
server or internet access required.

Usage:
    # Submit 5 parallel workers (25 total trials, 5 per worker):
    sbatch --array=0-4 submit_aptos_optuna.sh

    # Run 1 trial locally for debugging:
    python train_aptos_optuna.py --n_trials 1

    # Print top results from the study DB:
    python train_aptos_optuna.py --report
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
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from configs.config import DRConfig
from Datasets.aptos_loader import load_all_aptos_items
from Datasets.dataloaders import (
    ImageLabelDataset,
    StratifiedBatchSampler,
    build_tile_transform,
)
from models.framework import build_model
from training.cross_val import APTOSCrossValidator
from training.trainer import Trainer
from utils.checkpoint import save_checkpoint, load_checkpoint

log = logging.getLogger("aptos_optuna")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ── Optuna-aware trainer ───────────────────────────────────────────────────────

class OptunaTrainer(Trainer):
    """Extends Trainer with per-epoch Optuna pruning reports and early abandonment."""

    def __init__(self, *args, trial, **kwargs):
        super().__init__(*args, **kwargs)
        self.trial = trial

    def fit(self, test_loader: DataLoader) -> tuple[float, dict]:
        import optuna

        best_val_acc = -float("inf")
        best_val_mae = float("inf")
        best_score = -float("inf")
        best_ckpt_path = os.path.join(self.run_dir, "best.pth")

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

            score = val_acc - 10.0 * val_mae
            is_best = score > best_score
            if is_best:
                best_score = score
                best_val_acc = val_acc
                best_val_mae = val_mae

            # Report to Optuna after each epoch — enables Hyperband pruning
            self.trial.report(best_val_acc, epoch)
            if self.trial.should_prune():
                log.info(f"Pruned at epoch {epoch} (best_val_acc={best_val_acc:.1f}%)")
                raise optuna.exceptions.TrialPruned()

            # Hard early abandonment — kills clearly hopeless runs even before
            # Hyperband fires (Hyperband min_resource=10, so first check at ep10)
            if epoch == 10 and best_val_acc < 58.0:
                log.info(f"Abandoned at epoch {epoch}: best_val_acc={best_val_acc:.1f}%")
                break
            if epoch >= 20 and best_val_acc < 72.0:
                log.info(f"Abandoned at epoch {epoch}: best_val_acc={best_val_acc:.1f}%")
                break

            cur_ckpt = os.path.join(self.run_dir, f"epoch{epoch}.pth")
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
            if epoch > 1:
                prev = os.path.join(self.run_dir, f"epoch{epoch - 1}.pth")
                if os.path.exists(prev) and prev != best_ckpt_path:
                    os.remove(prev)

            self.scheduler.step(val_loss)
            if self.early_stopping.step(val_acc):
                log.info(f"Early stopping at epoch {epoch}")
                break

        if os.path.exists(best_ckpt_path):
            load_checkpoint(best_ckpt_path, self.model, None, self.text_encoder, self.device)

        test_metrics = self._eval_epoch(test_loader, prefix="test")
        log.info(
            f"Trial {self.trial.number} done — "
            f"best_val_acc={best_val_acc:.2f}%  "
            f"test_acc={test_metrics['test_acc']:.2f}%  "
            f"test_mae={test_metrics['test_mae']:.4f}"
        )
        return best_val_acc, test_metrics


# ── Config builder ─────────────────────────────────────────────────────────────

def build_cfg(params: dict) -> DRConfig:
    cfg = DRConfig()
    # Fixed OPTIC-C v9 baseline
    cfg.dataset               = "DR"   # DR text descriptions — same task as APTOS
    cfg.n_classes             = 5
    cfg.n_folds               = 5
    cfg.val_fraction          = 0.1
    cfg.epochs                = 100
    cfg.batch_size            = 24
    cfg.lr                    = 2e-4
    cfg.stratified            = True
    cfg.pretrained            = True
    cfg.amp                   = True
    cfg.use_multi_tile        = True
    cfg.tile_grid             = 3
    cfg.use_tile_transformer  = True
    cfg.use_grade_prototypes  = True
    cfg.use_ordinal_head      = True
    cfg.use_concept_prototype = True
    cfg.alpha                 = 0.0
    cfg.beta                  = 0.0
    cfg.gamma                 = 0.0
    cfg.lambda_osd            = 0.0
    cfg.lambda_tcl            = 0.0
    cfg.early_stop_patience   = 30
    # Sweep parameters
    cfg.lambda_proto_ce       = params["lambda_proto_ce"]
    cfg.lambda_tile_concept   = params["lambda_tile_concept"]
    cfg.lambda_gpa            = params["lambda_gpa"]
    cfg.proto_temperature     = params["proto_temperature"]
    cfg.proto_label_smoothing = params["proto_label_smoothing"]
    cfg.new_component_lr_mult = params["new_component_lr_mult"]
    cfg.backbone_freeze_epochs= params["backbone_freeze_epochs"]
    cfg.lr_patience           = params["lr_patience"]
    cfg.weight_decay          = params["weight_decay"]
    cfg.use_cosine_lr         = params["use_cosine_lr"]
    return cfg


# ── Objective factory ──────────────────────────────────────────────────────────

def make_objective(args, all_items: list, device: torch.device):
    def objective(trial) -> float:
        params = {
            "lambda_proto_ce":        trial.suggest_float("lambda_proto_ce",       0.2,  3.0,  log=True),
            "lambda_tile_concept":    trial.suggest_float("lambda_tile_concept",   0.05, 1.2,  log=True),
            "lambda_gpa":             trial.suggest_float("lambda_gpa",            0.03, 0.5,  log=True),
            "proto_temperature":      trial.suggest_categorical("proto_temperature",      [0.08, 0.10, 0.12, 0.15, 0.20, 0.25]),
            "proto_label_smoothing":  trial.suggest_categorical("proto_label_smoothing",  [0.0, 0.03, 0.05, 0.07, 0.10, 0.15]),
            "new_component_lr_mult":  trial.suggest_categorical("new_component_lr_mult",  [1.5, 2.0, 2.5, 3.0, 4.0]),
            "backbone_freeze_epochs": trial.suggest_categorical("backbone_freeze_epochs", [15, 20, 25, 30]),
            "lr_patience":            trial.suggest_categorical("lr_patience",            [8, 10, 12, 15]),
            "weight_decay":           trial.suggest_categorical("weight_decay",           [1e-6, 5e-6, 1e-5, 3e-5]),
            "use_cosine_lr":          trial.suggest_categorical("use_cosine_lr",          [True, False]),
        }
        log.info(f"Trial {trial.number} — params: {params}")

        cfg = build_cfg(params)
        set_seed(cfg.seed)

        run_dir = os.path.join(args.study_dir, f"trial_{trial.number:03d}")
        os.makedirs(run_dir, exist_ok=True)
        cfg.run_dir = run_dir

        cv = APTOSCrossValidator(
            all_items, n_folds=cfg.n_folds,
            val_fraction=cfg.val_fraction, seed=cfg.seed,
        )
        # Fold 0 only — fast proxy (~1-2 h).
        # get_fold returns (train, val, test); merge train+val → training set, test → eval.
        train_raw, val_held, test_items = cv.get_fold(0)
        train_items = train_raw + val_held

        tile_train = build_tile_transform(tile_size=cfg.img_size, tile_grid=cfg.tile_grid, augment=True)
        tile_eval  = build_tile_transform(tile_size=cfg.img_size, tile_grid=cfg.tile_grid, augment=False)

        train_ds = ImageLabelDataset(train_items, transform=tile_train)
        val_ds   = ImageLabelDataset(test_items,  transform=tile_eval)
        test_ds  = ImageLabelDataset(test_items,  transform=tile_eval)

        train_labels = [y for _, y in train_items]
        sampler = StratifiedBatchSampler(
            train_labels, batch_size=cfg.batch_size, drop_last=True, seed=cfg.seed)

        kw = dict(num_workers=4, pin_memory=True, prefetch_factor=4, persistent_workers=True)
        train_loader = DataLoader(train_ds, batch_sampler=sampler, **kw)
        val_loader   = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, **kw)
        test_loader  = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False, **kw)

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

        trainer = OptunaTrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            cfg=cfg,
            run_dir=run_dir,
            train_labels=train_labels,
            device=device,
            fold=0,
            trial=trial,
        )

        best_val_acc, _ = trainer.fit(test_loader)
        return best_val_acc

    return objective


# ── Reporting ──────────────────────────────────────────────────────────────────

def print_report(study) -> None:
    import optuna

    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    pruned    = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
    failed    = [t for t in study.trials if t.state == optuna.trial.TrialState.FAIL]

    print(f"\nStudy: {study.study_name}")
    print(f"  Completed: {len(completed)}  Pruned: {len(pruned)}  Failed: {len(failed)}")

    if not completed:
        print("  No completed trials yet.")
        return

    completed.sort(key=lambda t: t.value, reverse=True)
    print(f"\nTop 10 trials (by best_val_acc):")
    for i, t in enumerate(completed[:10]):
        print(
            f"  #{i+1:2d}  trial={t.number:3d}  val_acc={t.value:.2f}%  "
            f"lambda_proto_ce={t.params.get('lambda_proto_ce', '?'):.3f}  "
            f"lambda_tile_concept={t.params.get('lambda_tile_concept', '?'):.3f}  "
            f"lambda_gpa={t.params.get('lambda_gpa', '?'):.3f}"
        )
    print(f"\nBest params:")
    for k, v in study.best_params.items():
        print(f"  {k}: {v}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Optuna hyperparameter search — OPTIC-C on APTOS 2019 (fold 0 proxy)")
    parser.add_argument("--aptos_root", default="Datasets/aptos2019-blindness-detection")
    parser.add_argument("--study_dir",  default="runs/aptos_optic_optuna",
                        help="Directory for SQLite DB and per-trial checkpoints")
    parser.add_argument("--n_trials",   type=int, default=5,
                        help="Trials this worker will run")
    parser.add_argument("--report",     action="store_true",
                        help="Print top results from the study DB and exit")
    args = parser.parse_args()

    os.makedirs(args.study_dir, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
        force=True,
    )

    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    db_path = os.path.join(os.path.abspath(args.study_dir), "study.db")
    storage = f"sqlite:///{db_path}"

    study = optuna.create_study(
        study_name="aptos_optic_v9",
        storage=storage,
        load_if_exists=True,
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.HyperbandPruner(min_resource=10, reduction_factor=2),
    )

    if args.report:
        print_report(study)
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Device: {device}  DB: {db_path}")

    all_items = load_all_aptos_items(args.aptos_root)
    log.info(f"APTOS: {len(all_items)} images")

    objective = make_objective(args, all_items, device)

    study.optimize(
        objective,
        n_trials=args.n_trials,
        callbacks=[
            lambda s, t: log.info(
                f"Trial {t.number} done — val_acc={t.value:.2f}%  "
                f"best so far={s.best_value:.2f}%"
            ) if t.value is not None else None
        ],
    )

    log.info(
        f"Worker finished {args.n_trials} trials.  "
        f"Study best: {study.best_value:.2f}%"
    )


if __name__ == "__main__":
    main()
