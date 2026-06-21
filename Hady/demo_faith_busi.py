"""
Faithfulness STEP-1 (concept-drop) demo for the trained BUSI model.

Runs the first step of the ESPACOL Perturbation–Consistency Faithfulness Loop as a
DEMO (no training): get the actual prediction + concept scores, soft-occlude the TOP
concept's heatmap region, re-forward through the SAME model, and check whether that
concept's score actually DROPS (faithful) or not (unfaithful).

Pops up / saves a SINGLE window with four panels:

    Original | WHERE H_c (top concept) | Occluded input | before-vs-after concept scores

Run (from the repo root, ESPACOL-New/):
    python Hady/demo_faith_busi.py                       # random image, blur occlusion
    python Hady/demo_faith_busi.py --index 0             # deterministic single image
    python Hady/demo_faith_busi.py --occlusion gray      # mean/gray occlusion instead of blur
    python Hady/demo_faith_busi.py --seed 0 --save out.png --no-show   # headless PNG

Notes
-----
* Mirrors demo_explain_busi.py for model/encoder/dataset setup and reuses its helpers;
  the genuinely-new logic (occlude / re-score / drop-check / figure) lives in
  Hady/faithfulness.py.
* The concept-drop is only interpretable with --encoder biomedclip AND the recovered
  text projection (Hady/BUSI_text_projection.pth). With --encoder random the WHY scores
  are placeholders and the drop is meaningless (a warning is printed).
"""

import argparse
import os
import sys

import torch

# Make the repo root (parent of Hady/) importable regardless of CWD.
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _REPO_ROOT)

import random

from Datasets.dataloaders import BUSIDataset
from models.framework import build_model
from utils.checkpoint import load_checkpoint
from explainability import ExplainabilityPipeline, BUSI_CONCEPTS

# Reuse the explain demo's setup helpers (no third copy of the scaffolding).
from demo_explain_busi import (
    _ExplainAdapter,
    load_biomedclip,
    to_display_image,
    _safe_mask,
    resolve_busi_root,
    N_CLASSES,
    LABEL_TO_CLASS,
)
from faithfulness import (
    run_concept_drop,
    build_faithfulness_figure,
    print_faith_summary,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="BUSI faithfulness step-1 (concept-drop) demo: "
                    "Original | WHERE H_c | Occluded | before-vs-after scores."
    )
    parser.add_argument(
        "--checkpoint",
        default=os.path.join(_HERE, "BUSI_best_model.pth"),
        help="Path to the trained BUSI checkpoint (default: next to this script).",
    )
    parser.add_argument(
        "--busi_root", default=None,
        help="BUSI dataset root (benign/ malignant/ normal/). Auto-resolved if omitted.",
    )
    parser.add_argument("--index", type=int, default=None, help="Image index (default: random).")
    parser.add_argument("--seed", type=int, default=None, help="Seed for the random image choice.")
    parser.add_argument(
        "--encoder", choices=["biomedclip", "random"], default="biomedclip",
        help="Concept (WHY) encoder. 'biomedclip' (default) downloads ~350 MB on first run; "
             "'random' is offline but its WHY scores are placeholders (drop not meaningful).",
    )
    parser.add_argument(
        "--text_projection", default=None,
        help="Recovered z_it-space text projection .pth (default: Hady/BUSI_text_projection.pth).",
    )
    # ── occlusion controls ────────────────────────────────────────────────────
    parser.add_argument(
        "--occlusion", choices=["blur", "gray"], default="blur",
        help="How to obscure the top concept's region. 'blur' (default, RISE-style) "
             "replaces it with a blurred copy; 'gray' replaces it with the ImageNet mean.",
    )
    parser.add_argument(
        "--no-tissue-mask", action="store_true",
        help="Do not restrict occlusion to the ultrasound tissue region (occlude the raw H_c).",
    )
    parser.add_argument(
        "--threshold", type=float, default=0.0,
        help="Zero out H_c values below this before occluding (sharpen the mask). Default 0.",
    )
    parser.add_argument(
        "--blur-sigma", type=float, default=21.0,
        help="Gaussian sigma for --occlusion blur (default 21).",
    )
    parser.add_argument("--save", default=None, help="Optional path to save the figure as PNG.")
    parser.add_argument("--no-show", action="store_true", help="Do not pop up a window (save only).")
    return parser.parse_args()


def main():
    args = parse_args()

    if not os.path.isfile(args.checkpoint):
        sys.exit(f"Checkpoint not found: {args.checkpoint}")

    busi_root = resolve_busi_root(args.busi_root)
    if busi_root is None:
        sys.exit(
            "BUSI dataset not found. Pass its location with --busi_root "
            "(it should contain benign/ malignant/ normal/ folders)."
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Build repo model, load weights, wrap in the explainability adapter ────
    base = build_model(n_classes=N_CLASSES, pretrained=False, use_image_text=True)
    metrics = load_checkpoint(args.checkpoint, base, optimizer=None, device=device)
    base.to(device).eval()
    if metrics:
        va, mae = metrics.get("val_acc"), metrics.get("val_mae")
        bits = []
        if va is not None:
            bits.append(f"val_acc={va:.2f}%")
        if mae is not None:
            bits.append(f"val_mae={mae:.4f}")
        print(f"Loaded checkpoint ({', '.join(bits) if bits else 'no metrics'})")
    model = _ExplainAdapter(base).to(device).eval()

    # ── Concept (WHY) encoder ─────────────────────────────────────────────────
    encoder = args.encoder
    text_model = tokenizer = None
    if encoder == "biomedclip":
        text_model, tokenizer = load_biomedclip(device)
        if text_model is None:
            print("Falling back to --encoder random.")
            encoder = "random"
    if encoder == "random":
        print("NOTE: WHY concept scores are random placeholders; the concept-drop is NOT meaningful.")

    # ── Recovered text projection (maps concepts into the aligned z_it space) ──
    text_projection = None
    if encoder == "biomedclip":
        proj_path = args.text_projection or os.path.join(_HERE, "BUSI_text_projection.pth")
        if os.path.isfile(proj_path):
            from text_projection import load_text_projection
            text_projection, _bundle = load_text_projection(proj_path, device)
            print(f"Loaded recovered text projection: {proj_path}")
        else:
            print(f"NOTE: no recovered text projection at {proj_path}; WHY scores use "
                  f"the random-orthogonal bridge (NOT gamma-aligned). "
                  f"Run Hady/recover_text_projection.py first.")

    # ── Pick one image (BUSIDataset applies the eval transform internally) ────
    ds = BUSIDataset(root_dir=busi_root, split="all")
    print(f"BUSI images found: {len(ds)}")
    if args.seed is not None:
        random.seed(args.seed)
    if args.index is not None:
        if not 0 <= args.index < len(ds):
            sys.exit(f"--index must be in [0, {len(ds) - 1}]")
        idx = args.index
    else:
        idx = random.randrange(len(ds))
    x, y = ds[idx]
    img_path = ds.items[idx][0]
    true_name = LABEL_TO_CLASS[int(y)]

    # ── Run explainability (gives prediction + concept scores + concept heatmaps) ──
    n_concepts = len(BUSI_CONCEPTS["concepts"])
    pipeline = ExplainabilityPipeline(
        model, dataset="busi", device=device, encoder=encoder,
        text_model=text_model, tokenizer=tokenizer, text_projection=text_projection,
        top_k=n_concepts,
    )
    # ── STEP 1: explain → occlude the top concept's region → check the drop ───
    x0 = x.unsqueeze(0).to(device)
    image_np = to_display_image(x)
    mask = _safe_mask(image_np)
    tissue = None if args.no_tissue_mask else mask

    print("Computing explanation + occluding the top concept (re-forward) ...")
    result, x_occ, M, scores_before, scores_after, reg_after, check = run_concept_drop(
        pipeline, x0, tissue_mask=tissue, occlusion=args.occlusion,
        threshold=args.threshold, blur_sigma=args.blur_sigma,
    )
    top_concept = check["concept"]
    pipeline.remove_hooks()  # all forwards done; release LayerCAM hooks now

    # ── Predictions before/after (for display) ────────────────────────────────
    pred_before = result["predicted_class"]
    reg_before = result["regression_score"]
    after_label = int(round(min(max(reg_after, 0.0), N_CLASSES - 1)))
    pred_after = pipeline.class_names[after_label]

    print("-" * 60)
    print(f"Image      : {os.path.basename(img_path)}  (index {idx})")
    print(f"True label : {true_name} ({int(y)})")
    print_faith_summary(check, pred_before, pred_after, reg_before, reg_after, encoder)

    # ── Backend: interactive window unless --no-show ──────────────────────────
    show = not args.no_show
    import matplotlib
    if show:
        try:
            matplotlib.use("TkAgg", force=True)
        except Exception as exc:
            print(f"WARNING: interactive backend unavailable ({exc}); use --save for a PNG.")
            show = False
    if not show:
        matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    occ_np = to_display_image(x_occ.squeeze(0))
    fig = build_faithfulness_figure(
        plt, image_np, occ_np, M, top_concept, scores_before, scores_after, check,
        mask, true_name, pred_before, pred_after, dataset_name="BUSI",
        baseline_word=args.occlusion,
    )

    if args.save:
        fig.savefig(args.save, dpi=150, bbox_inches="tight")
        print(f"Saved figure to {args.save}")
    if show:
        try:
            plt.show()
        except Exception as exc:
            print(f"WARNING: could not display window ({exc}). Try --save <path> --no-show.")


if __name__ == "__main__":
    main()
