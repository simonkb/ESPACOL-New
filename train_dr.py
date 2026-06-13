"""
DR 10-fold subject-independent cross-validation training script.

Baseline:
    EfficientNet-V2S + PCOL + SCOLw + RMSE

ESPAOCL extension:
    EfficientNet-V2S + PCOL + SCOLw + Image-Text Ordinal Loss + RMSE
"""

import argparse
import csv
import logging
import os
import random
import sys

import numpy as np
import torch
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


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def setup_logging(run_dir: str) -> None:
    os.makedirs(run_dir, exist_ok=True)

    fmt = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    handlers = [
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(os.path.join(run_dir, "train.log")),
    ]

    logging.basicConfig(
        level=logging.INFO,
        format=fmt,
        handlers=handlers,
        force=True,
    )


def load_all_dr_items(dr_root: str, train_csv: str) -> list:
    ds = DRDataset(
        root_dir=dr_root,
        split="train",
        csv_path=train_csv,
    )
    return ds.items


def make_loaders(
    train_items,
    val_items,
    test_items,
    cfg: DRConfig,
    device=None,
    img_cache=None,
):
    clip_norm = getattr(cfg, "use_clip_normalization", True)
    train_tfm = build_train_transform(cfg.img_size, use_clip_norm=clip_norm)
    eval_tfm = build_transform(cfg.img_size, use_clip_norm=clip_norm)

    use_mps = device is not None and device.type == "mps"
    num_workers = 0 if use_mps else cfg.num_workers
    pin_memory = False if use_mps else cfg.pin_memory

    pf_kwargs = {"prefetch_factor": 4} if num_workers > 0 else {}

    train_ds = ImageLabelDataset(
        train_items,
        transform=train_tfm,
        img_cache=img_cache,
    )
    val_ds = ImageLabelDataset(
        val_items,
        transform=eval_tfm,
        img_cache=img_cache,
    )
    test_ds = ImageLabelDataset(
        test_items,
        transform=eval_tfm,
        img_cache=img_cache,
    )

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
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=True,
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


def main():
    parser = argparse.ArgumentParser(description="Train DR 10-fold CV")

    parser.add_argument(
        "--dr_root",
        type=str,
        default="Datasets/DR",
        help="Path to DR dataset root containing train/ and trainLabels.csv",
    )
    parser.add_argument(
        "--train_csv",
        type=str,
        default=None,
        help="Path to training label CSV. Default: <dr_root>/trainLabels.csv",
    )
    parser.add_argument(
        "--run_dir",
        type=str,
        default="runs/dr",
        help="Directory for checkpoints and logs",
    )
    parser.add_argument(
        "--folds",
        type=str,
        default="all",
        help="Comma-separated fold indices, e.g. '0,1,2', or 'all'",
    )

    parser.add_argument("--no_pretrained", action="store_true")

    parser.add_argument(
        "--no_cache",
        action="store_true",
        help="Disable image cache if RAM is limited",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default="Datasets/DR/train_cache_224",
        help="Directory for pre-decoded .pt image cache",
    )

    parser.add_argument(
        "--use_image_text",
        action="store_true",
        help="Enable ESPAOCL image-text ordinal alignment loss",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=None,
        help="Weight for image-text ordinal loss",
    )
    parser.add_argument(
        "--lambda_ord_it",
        type=float,
        default=None,
        help="Ordinal penalty strength inside image-text loss",
    )
    parser.add_argument(
        "--text_encoder_name",
        type=str,
        default=None,
        help="Optional override for BioMedCLIP/open_clip text encoder name",
    )
    parser.add_argument(
        "--use_tamo",
        action="store_true",
        help="Enable TAMO loss (Text-Anchored Metric Ordinality) — upgrades all "
             "projection heads to DeepProjectionHead and adds prototype geometry "
             "distillation from text space.  Implies use_image_text=True.",
    )
    parser.add_argument(
        "--gamma_tamo",
        type=float,
        default=None,
        help="Weight for TAMO loss in total loss (overrides DRConfig default)",
    )
    parser.add_argument(
        "--lambda_orc",
        type=float,
        default=None,
        help="Relative weight of ORC term within TAMO loss",
    )

    args = parser.parse_args()

    if args.train_csv is None:
        args.train_csv = os.path.join(args.dr_root, "trainLabels.csv")

    cfg = DRConfig(run_dir=args.run_dir)

    if args.use_image_text:
        cfg.use_image_text = True

    if args.gamma is not None:
        cfg.gamma = args.gamma
        cfg.use_image_text = cfg.gamma > 0.0

    if args.lambda_ord_it is not None:
        cfg.lambda_ord_it = args.lambda_ord_it

    if args.text_encoder_name is not None:
        cfg.text_encoder_name = args.text_encoder_name

    if args.use_tamo:
        cfg.use_tamo = True
        cfg.use_image_text = True   # TAMO requires a text encoder

    if args.gamma_tamo is not None:
        cfg.gamma_tamo = args.gamma_tamo

    if args.lambda_orc is not None:
        cfg.lambda_orc = args.lambda_orc

    setup_logging(args.run_dir)
    log = logging.getLogger("train_dr")

    set_seed(cfg.seed)

    log.info("=" * 70)
    if getattr(cfg, "use_tamo", False):
        log.info("DR 10-fold CV  (BiomedCLIP ViT-B/16 + PCOL + SCOLw + IT + TAMO [DeepHeads])")
    elif cfg.use_image_text:
        log.info("DR 10-fold CV  (BiomedCLIP ViT-B/16 + PCOL + SCOLw + ImageText)")
    else:
        log.info("DR 10-fold CV  (BiomedCLIP ViT-B/16 + PCOL + SCOLw)")
    log.info("=" * 70)
    log.info(f"Config: {cfg}")

    all_items = load_all_dr_items(args.dr_root, args.train_csv)
    log.info(f"Total DR training images: {len(all_items)}")

    from collections import Counter

    dist = Counter(y for _, y in all_items)
    log.info(f"Class distribution: {dict(sorted(dist.items()))}")

    cv = DRCrossValidator(
        all_items,
        n_folds=cfg.n_folds,
        val_fraction=cfg.val_fraction,
        seed=cfg.seed,
    )

    if args.folds == "all":
        fold_indices = list(range(cfg.n_folds))
    else:
        fold_indices = [int(f.strip()) for f in args.folds.split(",") if f.strip()]

    device = torch.device(
        "cuda" if torch.cuda.is_available() else
        "mps" if torch.backends.mps.is_available() else
        "cpu"
    )
    log.info(f"Device: {device}")

    img_cache = None
    if not args.no_cache:
        n_threads = 16
        cache_dir = args.cache_dir if args.cache_dir else None

        log.info(
            f"Pre-loading DR images ({n_threads} threads) "
            f"{'(disk cache: ' + cache_dir + ')' if cache_dir else '(no disk cache)'}"
        )

        img_cache = preload_dr_images(
            all_items,
            img_size=cfg.img_size,
            n_threads=n_threads,
            cache_dir=cache_dir,
        )
        log.info(f"Image cache ready: {len(img_cache)} images")

    fold_results = []

    for fi in fold_indices:
        log.info("")
        log.info("-" * 60)
        log.info(f"FOLD {fi + 1} / {cfg.n_folds}")
        log.info("-" * 60)

        set_seed(cfg.seed + fi)

        train_items_raw, val_items_held_out, test_items = cv.get_fold(fi)

        # Follow the existing replication protocol:
        # use the held-out CV fold as validation/test fold.
        train_items = train_items_raw + val_items_held_out
        val_items = test_items

        log.info(f"  train={len(train_items)}  val=test={len(test_items)}")

        dist_fold = Counter(y for _, y in train_items)
        log.info(f"  Train class dist: {dict(sorted(dist_fold.items()))}")

        train_loader, val_loader, test_loader = make_loaders(
            train_items,
            val_items,
            test_items,
            cfg,
            device=device,
            img_cache=img_cache,
        )

        fold_dir = os.path.join(args.run_dir, f"fold{fi}")
        os.makedirs(fold_dir, exist_ok=True)

        model = build_model(
            n_classes=cfg.n_classes,
            pretrained=not args.no_pretrained,
            proj_hidden_dim=cfg.proj_hidden_dim,
            proj_out_dim=cfg.proj_out_dim,
            use_image_text=cfg.use_image_text,
            use_tamo=getattr(cfg, "use_tamo", False),
            image_encoder_name=cfg.image_encoder_name,
        )

        train_labels = [y for _, y in train_items]

        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            cfg=cfg,
            run_dir=fold_dir,
            train_labels=train_labels,
            device=device,
            fold=fi,
        )

        test_metrics = trainer.fit(test_loader)
        fold_results.append(test_metrics)

        log.info(
            f"  Fold {fi} summary: acc={test_metrics['test_acc']:.2f}%  "
            f"mae={test_metrics['test_mae']:.4f}"
        )

    log.info("")
    log.info("=" * 70)
    log.info("FINAL RESULTS  (mean ± std across folds)")
    log.info("=" * 70)

    accs = [r["test_acc"] for r in fold_results]
    maes = [r["test_mae"] for r in fold_results]

    log.info(f"  Accuracy : {np.mean(accs):.2f}% ± {np.std(accs):.2f}%")
    log.info(f"  MAE      : {np.mean(maes):.4f} ± {np.std(maes):.4f}")

    summary_path = os.path.join(args.run_dir, "final_results.csv")

    with open(summary_path, "w", newline="") as f:
        fieldnames = ["fold"] + list(fold_results[0].keys())
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for fi, res in zip(fold_indices, fold_results):
            writer.writerow({"fold": fi, **res})

        writer.writerow(
            {
                "fold": "mean",
                "test_loss": "",
                "test_acc": f"{np.mean(accs):.4f}",
                "test_mae": f"{np.mean(maes):.4f}",
            }
        )
        writer.writerow(
            {
                "fold": "std",
                "test_loss": "",
                "test_acc": f"{np.std(accs):.4f}",
                "test_mae": f"{np.std(maes):.4f}",
            }
        )

    log.info(f"Results saved to {summary_path}")


if __name__ == "__main__":
    main()