"""
Faithfulness STEP-3 (occlusion specificity) demo — the on- vs off-target control.

A bare concept-drop (step 1) is ambiguous: the score can fall because we hit the
concept's real evidence (faithful), because blurring *any* equal chunk perturbs the
embedding (generic), or because the heatmap covers most of the image (trivial). This
demo runs the decisive control: blur the top concept's heatmap region (ON-target) vs.
blur the SAME-SIZE mask moved elsewhere (OFF-target), and compares the drops.

    specificity = drop_on - drop_off    (> 0 ⇒ the heatmap LOCATION matters ⇒ faithful)

Two modes:
  * single image (default): one 4-panel figure
        Original | WHERE H_c | ON-target occluded | OFF-target occluded
  * aggregate (--per-class N): a per-class / per-grade table of mean drop_on,
    drop_off, specificity and mask% (no figure).

Run (from the repo root, ESPACOL-New/):
    source slurm/_env.sh
    # single-image figures
    python Hady/demo_faith_control.py --dataset busi --seed 0 --save ctrl_busi.png --no-show
    python Hady/demo_faith_control.py --dataset dr --grade 2 --save ctrl_dr.png --no-show
    # aggregate specificity table across all classes/grades
    python Hady/demo_faith_control.py --dataset both --per-class 3

Only meaningful with --encoder biomedclip + the recovered text projections.
"""

import argparse
import os
import random
import sys

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _REPO_ROOT)

from Datasets.dataloaders import BUSIDataset, DRDataset
from models.framework import build_model
from utils.checkpoint import load_checkpoint
from explainability import ExplainabilityPipeline, BUSI_CONCEPTS, DR_CONCEPTS

from demo_explain_busi import (
    _ExplainAdapter, load_biomedclip, to_display_image,
    _safe_mask as busi_safe_mask, resolve_busi_root, N_CLASSES as BUSI_N,
)
from demo_explain_dr import (
    _safe_mask as dr_safe_mask, resolve_dr_root, N_CLASSES as DR_N,
)
from faithfulness import (
    occlusion_specificity, build_specificity_figure, print_specificity_summary,
)
from text_projection import load_text_projection


# ─────────────────────────────────────────────────────────────────────────────
# Dataset/model setup (one pipeline, reused for all images of that dataset)
# ─────────────────────────────────────────────────────────────────────────────

def setup(dataset, args, device, text_model, tokenizer, encoder):
    """Return a dict bundling the pipeline + dataset + metadata for `dataset`."""
    if dataset == "busi":
        root = resolve_busi_root(args.busi_root)
        if root is None:
            return None
        ds = BUSIDataset(root_dir=root, split="all")
        n_classes, concepts, safe_mask = BUSI_N, BUSI_CONCEPTS["concepts"], busi_safe_mask
        ckpt = args.busi_ckpt
        proj_path = args.busi_text_projection
        labels = sorted(BUSIDataset.CLASS_TO_LABEL.values())
        namer = lambda l: ["normal", "benign", "malignant"][l]
    else:
        root = resolve_dr_root(args.dr_root)
        if root is None:
            return None
        csv = args.train_csv or os.path.join(root, "trainLabels.csv")
        print("Indexing DR dataset (this can take a few seconds)...")
        ds = DRDataset(root_dir=root, split="train", csv_path=csv)
        n_classes, concepts, safe_mask = DR_N, DR_CONCEPTS["concepts"], dr_safe_mask
        ckpt = args.dr_ckpt
        proj_path = args.dr_text_projection
        labels = list(range(DR_N))
        namer = lambda l: f"grade{l} ({DR_CONCEPTS['class_names'][l]})"

    base = build_model(n_classes=n_classes, pretrained=False, use_image_text=True)
    load_checkpoint(ckpt, base, optimizer=None, device=device)
    base.to(device).eval()
    model = _ExplainAdapter(base).to(device).eval()

    text_projection = None
    if encoder == "biomedclip" and os.path.isfile(proj_path):
        text_projection, _ = load_text_projection(proj_path, device)
        print(f"Loaded recovered text projection: {proj_path}")
    elif encoder == "biomedclip":
        print(f"NOTE: no recovered text projection at {proj_path}; scores are NOT gamma-aligned.")

    pipe = ExplainabilityPipeline(
        model, dataset=dataset, device=device, encoder=encoder,
        text_model=text_model, tokenizer=tokenizer, text_projection=text_projection,
        top_k=len(concepts),
    )
    print(f"{dataset.upper()} images found: {len(ds)}")
    return dict(pipeline=pipe, ds=ds, n_classes=n_classes, safe_mask=safe_mask,
                labels=labels, namer=namer)


def _pred_name(pipe, reg, n_classes):
    return pipe.class_names[int(round(min(max(reg, 0.0), n_classes - 1)))]


# ─────────────────────────────────────────────────────────────────────────────
# Modes
# ─────────────────────────────────────────────────────────────────────────────

def run_single(bundle, args, encoder):
    pipe, ds, n_classes, safe_mask = (bundle["pipeline"], bundle["ds"],
                                      bundle["n_classes"], bundle["safe_mask"])
    # pick one image
    if args.dataset == "dr" and args.grade is not None:
        pool = [i for i, (_, lab) in enumerate(ds.items) if lab == args.grade]
        if not pool:
            sys.exit(f"No images with grade {args.grade}.")
    else:
        pool = list(range(len(ds)))
    if args.seed is not None:
        random.seed(args.seed)
    idx = args.index if args.index is not None else random.choice(pool)
    x, y = ds[idx]
    x0 = x.unsqueeze(0).to(pipe.device)
    image_np = to_display_image(x)
    tissue = safe_mask(image_np)

    spec = occlusion_specificity(
        pipe, x0, tissue_mask=(None if args.no_tissue_mask else tissue),
        occlusion=args.occlusion, blur_sigma=args.blur_sigma,
        n_roll=args.controls, rng=random.Random(args.seed or 0),
    )
    pipe.remove_hooks()

    true_name = pipe.class_names[int(y)]
    pred_before = _pred_name(pipe, spec["reg_before"], n_classes)
    print(f"Image: {os.path.basename(ds.items[idx][0])}  (index {idx})")
    print_specificity_summary(spec, true_name, pred_before, encoder)

    show = not args.no_show
    import matplotlib
    if show:
        try:
            matplotlib.use("TkAgg", force=True)
        except Exception:
            show = False
    if not show:
        matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    on_np = to_display_image(spec["x_on"].squeeze(0))
    off_np = to_display_image(spec["x_ctrl0"].squeeze(0))
    fig = build_specificity_figure(
        plt, image_np, on_np, off_np, spec, true_name, pred_before,
        dataset_name=args.dataset.upper(), baseline_word=args.occlusion,
    )
    if args.save:
        fig.savefig(args.save, dpi=150, bbox_inches="tight")
        print(f"Saved figure to {args.save}")
    if show:
        try:
            plt.show()
        except Exception as exc:
            print(f"WARNING: could not display window ({exc}). Try --save <path> --no-show.")


def run_table(bundle, args):
    pipe, ds, n_classes, safe_mask = (bundle["pipeline"], bundle["ds"],
                                      bundle["n_classes"], bundle["safe_mask"])
    name = args._cur.upper()
    rng = random.Random(args.seed if args.seed is not None else 0)
    ctrl_rng = random.Random(123)
    print(f"\n{name}: {'group':<24}{'dropON':>8}{'dropOFF':>8}{'specific':>9}{'mask%':>7}  on>off")
    for label in bundle["labels"]:
        pool = [i for i, (_, lab) in enumerate(ds.items) if int(lab) == int(label)]
        rng.shuffle(pool)
        idxs = pool[:args.per_class]
        d_on, d_off, spec_v, area, won = [], [], [], [], 0
        for idx in idxs:
            x, _ = ds[idx]
            x0 = x.unsqueeze(0).to(pipe.device)
            tissue = safe_mask(to_display_image(x))
            s = occlusion_specificity(
                pipe, x0, tissue_mask=(None if args.no_tissue_mask else tissue),
                occlusion=args.occlusion, blur_sigma=args.blur_sigma,
                n_roll=args.controls, rng=ctrl_rng,
            )
            d_on.append(s["drop_on"]); d_off.append(s["drop_ctrl"])
            spec_v.append(s["specificity"]); area.append(s["mask_frac"])
            won += s["drop_on"] > s["drop_ctrl"]
        if idxs:
            print(f"{name}: {bundle['namer'](label):<24}{np.mean(d_on):>+8.3f}"
                  f"{np.mean(d_off):>+8.3f}{np.mean(spec_v):>+9.3f}"
                  f"{100*np.mean(area):>6.1f}%{won:>6}/{len(idxs)}")
    pipe.remove_hooks()


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Faithfulness step-3 occlusion specificity: on- vs off-target control."
    )
    p.add_argument("--dataset", choices=["busi", "dr", "both"], default="busi",
                   help="Dataset. 'both' is only valid with --per-class (table mode).")
    p.add_argument("--per-class", type=int, default=None,
                   help="If set, run the AGGREGATE specificity table with N images per "
                        "class/grade (no figure). Otherwise render a single-image figure.")
    p.add_argument("--index", type=int, default=None, help="Image index (single-image mode).")
    p.add_argument("--seed", type=int, default=None, help="Seed for random image pick / sampling.")
    p.add_argument("--grade", type=int, default=None, choices=range(DR_N),
                   help="DR only: restrict the single-image pick to this grade (0-4).")
    p.add_argument("--controls", type=int, default=3,
                   help="Number of random-shift controls (plus 2 reflections). Default 3 ⇒ 5 total.")
    p.add_argument("--occlusion", choices=["blur", "gray"], default="blur")
    p.add_argument("--blur-sigma", type=float, default=21.0)
    p.add_argument("--no-tissue-mask", action="store_true")
    p.add_argument("--encoder", choices=["biomedclip", "random"], default="biomedclip")
    p.add_argument("--save", default=None, help="Save the single-image figure as PNG.")
    p.add_argument("--no-show", action="store_true", help="Do not pop up a window.")
    # roots / checkpoints / projections (auto-resolved if omitted)
    p.add_argument("--busi_root", default=None)
    p.add_argument("--dr_root", default=None)
    p.add_argument("--train_csv", default=None)
    p.add_argument("--busi_ckpt", default=os.path.join(_HERE, "BUSI_best_model.pth"))
    p.add_argument("--dr_ckpt", default=os.path.join(_HERE, "DR_best_model.pth"))
    p.add_argument("--busi_text_projection", default=os.path.join(_HERE, "BUSI_text_projection.pth"))
    p.add_argument("--dr_text_projection", default=os.path.join(_HERE, "DR_text_projection.pth"))
    return p.parse_args()


def main():
    args = parse_args()
    if args.dataset == "both" and args.per_class is None:
        sys.exit("--dataset both is only valid in table mode (pass --per-class N).")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    encoder = args.encoder
    text_model = tokenizer = None
    if encoder == "biomedclip":
        text_model, tokenizer = load_biomedclip(device)
        if text_model is None:
            print("Falling back to --encoder random.")
            encoder = "random"
    if encoder == "random":
        print("NOTE: random encoder ⇒ specificity is meaningless (concept vectors are placeholders).")

    datasets = ["busi", "dr"] if args.dataset == "both" else [args.dataset]
    for d in datasets:
        bundle = setup(d, args, device, text_model, tokenizer, encoder)
        if bundle is None:
            print(f"WARNING: {d.upper()} dataset/checkpoint not found; skipping.")
            continue
        if args.per_class is not None:
            args._cur = d
            run_table(bundle, args)
        else:
            run_single(bundle, args, encoder)

    if args.per_class is not None:
        print("\nReading: specific>0 and on>off ⇒ occluding the HEATMAP region drops the "
              "concept\nmore than occluding an equal-size mask elsewhere (location matters = faithful).")


if __name__ == "__main__":
    main()
