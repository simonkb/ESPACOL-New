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
    build_tile_transform,
    preload_dr_images,
)
from Datasets.idrid_loader import load_all_idrid_items
from models.framework import build_model
from training.cross_val import DRCrossValidator, IDRiDCrossValidator
from training.trainer import Trainer


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def setup_logging(run_dir: str) -> None:
    # Retry loop guards against NFS race condition: on shared cluster filesystems
    # exist_ok=True can still raise FileExistsError when multiple fold processes
    # race to create the same parent directory simultaneously.
    for _attempt in range(10):
        try:
            os.makedirs(run_dir, exist_ok=True)
            break
        except FileExistsError:
            if os.path.isdir(run_dir):
                break
            import time; time.sleep(0.5 * (_attempt + 1))

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
    if getattr(cfg, "use_multi_tile", False):
        tile_grid = getattr(cfg, "tile_grid", 3)
        train_tfm = build_tile_transform(tile_size=cfg.img_size, tile_grid=tile_grid, augment=True)
        eval_tfm  = build_tile_transform(tile_size=cfg.img_size, tile_grid=tile_grid, augment=False)
    else:
        train_tfm = build_train_transform(cfg.img_size)
        eval_tfm  = build_transform(cfg.img_size)

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
    parser = argparse.ArgumentParser(description="Train DR/IDRiD CV")

    parser.add_argument(
        "--dataset",
        type=str,
        default="dr",
        choices=["dr", "idrid"],
        help="Dataset to train on: 'dr' (Kaggle DR, 10-fold) or 'idrid' (IDRiD, 5-fold)",
    )
    parser.add_argument(
        "--dr_root",
        type=str,
        default="Datasets/DR",
        help="Path to DR or IDRiD dataset root",
    )
    parser.add_argument(
        "--train_csv",
        type=str,
        default=None,
        help="(DR only) Path to training label CSV. Default: <dr_root>/trainLabels.csv",
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
        "--alpha",
        type=float,
        default=None,
        help="Weight for PCOL loss. Set 0 to disable PCOL entirely (no computation).",
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=None,
        help="Weight for SCOLw loss. Set 0 to disable SCOLw entirely (no computation).",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume training from existing best checkpoint (for SLURM preemption recovery). "
             "Without this flag the trainer always starts from epoch 1.",
    )
    parser.add_argument(
        "--lr_patience",
        type=int,
        default=None,
        help="Epochs of no val improvement before LR reduction (default from DRConfig=8). "
             "Use 15 when training new randomly-initialised components alongside a pretrained backbone.",
    )
    parser.add_argument(
        "--lr_min",
        type=float,
        default=None,
        help="Minimum LR floor for ReduceLROnPlateau (default from DRConfig=1e-6). "
             "Set to 8e-6 to cap at one fine-tuning drop (4e-5 → 8e-6) and prevent dead-LR cascades.",
    )
    parser.add_argument(
        "--new_component_lr_mult",
        type=float,
        default=None,
        help="LR multiplier for new OPTIC components (CTOT/GPA/ODH) vs pretrained backbone. "
             "E.g. 2.5 gives new components 2.5× higher LR.",
    )
    parser.add_argument(
        "--lambda_gpa",
        type=float,
        default=None,
        help="Weight for GradePrototypeCELoss — direct supervision on GPA tile evidence.",
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="Override cfg.batch_size",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override cfg.epochs",
    )
    parser.add_argument(
        "--grad_checkpoint",
        action="store_true",
        help="Enable gradient checkpointing in the tiled backbone to save GPU memory",
    )

    parser.add_argument(
        "--use_multi_tile",
        action="store_true",
        help="Enable multi-tile input with AttentionPool aggregation",
    )
    parser.add_argument(
        "--tile_grid",
        type=int,
        default=3,
        help="Tile grid size (default 3 → 3×3 = 9 local tiles + 1 global = 10 total)",
    )

    parser.add_argument(
        "--no_cache",
        action="store_true",
        help="Disable image cache if RAM is limited",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default="Datasets/DR/train_cache",
        help="Directory for pre-decoded .pt image cache",
    )

    parser.add_argument(
        "--gamma",
        type=float,
        default=None,
        help="(unused — kept for script compatibility)",
    )
    parser.add_argument(
        "--text_encoder_name",
        type=str,
        default=None,
        help="Optional override for BioMedCLIP/open_clip text encoder name",
    )

    # ── OPTIC architecture flags ──────────────────────────────────────────────
    parser.add_argument(
        "--use_tile_transformer",
        action="store_true",
        help="Replace AttentionPool with CrossTileOrdinalTransformer (CTOT)",
    )
    parser.add_argument(
        "--use_grade_prototypes",
        action="store_true",
        help="Enable GradePrototypeAttention for tile-level explainability (requires CTOT)",
    )
    parser.add_argument(
        "--use_ordinal_head",
        action="store_true",
        help="Replace RMSE regression with CORAL OrdinalDistributionHead",
    )
    parser.add_argument(
        "--use_osd_loss",
        action="store_true",
        help="Add OrdinalStochasticDominanceLoss (requires ordinal head)",
    )
    parser.add_argument(
        "--lambda_osd",
        type=float,
        default=None,
        help="Weight for OSD loss (default from DRConfig)",
    )
    parser.add_argument(
        "--use_tile_consistency",
        action="store_true",
        help="Add TileConsistencyLoss (requires grade prototypes)",
    )
    parser.add_argument(
        "--lambda_tcl",
        type=float,
        default=None,
        help="Weight for TileConsistencyLoss (default from DRConfig)",
    )
    parser.add_argument(
        "--osd_margin",
        type=float,
        default=None,
        help="Margin for OSD loss — 0 penalises any CDF violation; >0 enforces minimum CDF gap between grade CDFs",
    )
    parser.add_argument(
        "--tcl_margin",
        type=float,
        default=None,
        help="Margin for TileConsistencyLoss — 0 penalises any tile-image disagreement (recommended)",
    )
    parser.add_argument(
        "--backbone_freeze_epochs",
        type=int,
        default=None,
        help="Freeze EfficientNet backbone for this many epochs so CTOT/GPA can stabilise "
             "on fixed pretrained features before joint fine-tuning. 0 = no freeze.",
    )
    parser.add_argument(
        "--early_stop_patience",
        type=int,
        default=None,
        help="Epochs of no val_acc improvement before early stopping (default from DRConfig=20).",
    )
    parser.add_argument(
        "--text_finetune_start_epoch",
        type=int,
        default=None,
        help="Epoch at which text encoder fine-tuning begins (default from DRConfig=20).",
    )

    # ── OPTIC-C: Concept-Grounded Grade Prototype flags ──────────────────────
    parser.add_argument(
        "--use_concept_prototype",
        action="store_true",
        help="Enable ConceptGradePrototypeModule (OPTIC-C). Requires use_tile_transformer. "
             "Set alpha=0 beta=0 to remove PCOL/SCOLw and make novel losses dominant.",
    )
    parser.add_argument(
        "--lambda_proto_ce",
        type=float,
        default=None,
        help="Weight for L_proto_CE — cosine prototype CrossEntropy (dominant novel loss). "
             "Recommended: 1.5",
    )
    parser.add_argument(
        "--lambda_tile_concept",
        type=float,
        default=None,
        help="Weight for L_tile_concept — per-tile concept BCE vs clinical grade-concept targets. "
             "Recommended: 0.3",
    )
    parser.add_argument(
        "--proto_temperature",
        type=float,
        default=None,
        help="Cosine similarity temperature for grade prototype logits (default: 0.1). "
             "Lower = sharper prototype boundaries.",
    )
    parser.add_argument(
        "--proto_label_smoothing",
        type=float,
        default=None,
        help="Label smoothing for proto_CE cross-entropy (default: 0.0). "
             "Use 0.1 to regularise when proto_CE collapses to near-zero on training data.",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=None,
        help="AdamW weight decay (default: 1e-6). Increase to 1e-5 when backbone LR is higher.",
    )
    parser.add_argument(
        "--use_cosine_lr",
        action="store_true",
        default=False,
        help="Switch to CosineAnnealingLR at backbone unfreeze instead of continuing with "
             "ReduceLROnPlateau. Eliminates LR-drop timing luck across folds.",
    )

    args = parser.parse_args()

    if args.train_csv is None:
        args.train_csv = os.path.join(args.dr_root, "trainLabels.csv")

    cfg = DRConfig(run_dir=args.run_dir)

    if args.batch_size is not None:
        cfg.batch_size = args.batch_size
    if args.epochs is not None:
        cfg.epochs = args.epochs

    if args.use_multi_tile:
        cfg.use_multi_tile = True
        cfg.tile_grid = args.tile_grid

    if args.gamma is not None:
        cfg.gamma = args.gamma

    if args.text_encoder_name is not None:
        cfg.text_encoder_name = args.text_encoder_name

    # OPTIC flags
    if args.use_tile_transformer:
        cfg.use_tile_transformer = True
    if args.use_grade_prototypes:
        cfg.use_grade_prototypes = True
    if args.use_ordinal_head:
        cfg.use_ordinal_head = True
    if args.use_osd_loss:
        cfg.use_osd_loss = True
    if args.lambda_osd is not None:
        cfg.lambda_osd = args.lambda_osd
    if args.use_tile_consistency:
        cfg.use_tile_consistency = True
    if args.lambda_tcl is not None:
        cfg.lambda_tcl = args.lambda_tcl
    if args.osd_margin is not None:
        cfg.osd_margin = args.osd_margin
    if args.tcl_margin is not None:
        cfg.tcl_margin = args.tcl_margin
    if args.resume:
        cfg.resume = True
    if args.lr_patience is not None:
        cfg.lr_patience = args.lr_patience
    if args.lr_min is not None:
        cfg.lr_min = args.lr_min
    if args.new_component_lr_mult is not None:
        cfg.new_component_lr_mult = args.new_component_lr_mult
    if args.lambda_gpa is not None:
        cfg.lambda_gpa = args.lambda_gpa
    if args.backbone_freeze_epochs is not None:
        cfg.backbone_freeze_epochs = args.backbone_freeze_epochs
    if args.early_stop_patience is not None:
        cfg.early_stop_patience = args.early_stop_patience
    if args.text_finetune_start_epoch is not None:
        cfg.text_finetune_start_epoch = args.text_finetune_start_epoch

    # OPTIC-C concept prototype flags
    if args.alpha is not None:
        cfg.alpha = args.alpha
    if args.beta is not None:
        cfg.beta = args.beta

    if args.use_concept_prototype:
        cfg.use_concept_prototype = True
    if args.lambda_proto_ce is not None:
        cfg.lambda_proto_ce = args.lambda_proto_ce
    if args.lambda_tile_concept is not None:
        cfg.lambda_tile_concept = args.lambda_tile_concept
    if args.proto_temperature is not None:
        cfg.proto_temperature = args.proto_temperature
    if args.proto_label_smoothing is not None:
        cfg.proto_label_smoothing = args.proto_label_smoothing
    if args.weight_decay is not None:
        cfg.weight_decay = args.weight_decay
    if args.use_cosine_lr:
        cfg.use_cosine_lr = True

    setup_logging(args.run_dir)
    log = logging.getLogger("train_dr")

    set_seed(cfg.seed)

    dataset_name = args.dataset.upper()
    log.info("=" * 70)
    log.info(f"{dataset_name} CV  (EfficientNet-V2S + OPTIC-C)")
    log.info("=" * 70)
    log.info(f"Config: {cfg}")

    from collections import Counter

    if args.dataset == "idrid":
        all_items = load_all_idrid_items(args.dr_root)
        cfg.n_folds = 5
        log.info(f"Total IDRiD images: {len(all_items)}")
        dist = Counter(y for _, y in all_items)
        log.info(f"Class distribution: {dict(sorted(dist.items()))}")
        cv = IDRiDCrossValidator(
            all_items,
            n_folds=cfg.n_folds,
            val_fraction=cfg.val_fraction,
            seed=cfg.seed,
        )
    else:
        all_items = load_all_dr_items(args.dr_root, args.train_csv)
        log.info(f"Total DR training images: {len(all_items)}")
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

        if args.dataset == "idrid":
            # IDRiD is small (516 images) — RAM cache only, no disk persistence needed
            cache_dir = None
        elif cfg.use_multi_tile and args.cache_dir == "Datasets/DR/train_cache":
            canvas_size = cfg.tile_grid * cfg.img_size
            cache_dir = f"Datasets/DR/train_cache_tiles_{canvas_size}"
        else:
            cache_dir = args.cache_dir if args.cache_dir else None

        preload_img_size = cfg.tile_grid * cfg.img_size if cfg.use_multi_tile else cfg.img_size

        log.info(
            f"Pre-loading images ({n_threads} threads, size={preload_img_size}) "
            f"{'(disk cache: ' + cache_dir + ')' if cache_dir else '(no disk cache)'}"
        )

        img_cache = preload_dr_images(
            all_items,
            img_size=preload_img_size,
            n_threads=n_threads,
            cache_dir=cache_dir,
            crop_fundus=cfg.use_multi_tile,
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
            use_multi_tile=cfg.use_multi_tile,
            grad_checkpoint=args.grad_checkpoint,
            tile_grid=cfg.tile_grid,
            use_tile_transformer=cfg.use_tile_transformer,
            tile_transformer_dim=cfg.tile_transformer_dim,
            tile_transformer_nhead=cfg.tile_transformer_nhead,
            tile_transformer_layers=cfg.tile_transformer_layers,
            tile_transformer_dropout=cfg.tile_transformer_dropout,
            use_grade_prototypes=cfg.use_grade_prototypes,
            use_ordinal_head=cfg.use_ordinal_head,
            use_concept_prototype=cfg.use_concept_prototype,
            n_concepts=cfg.n_concepts,
            proto_temperature=cfg.proto_temperature,
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
    qwks = [r.get("test_qwk", 0.0) for r in fold_results]

    log.info(f"  Accuracy : {np.mean(accs):.2f}% ± {np.std(accs):.2f}%")
    log.info(f"  MAE      : {np.mean(maes):.4f} ± {np.std(maes):.4f}")
    log.info(f"  QWK      : {np.mean(qwks):.4f} ± {np.std(qwks):.4f}")

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