"""
Explainability demo for the trained DR (diabetic retinopathy) model.

Loads DR_best_model.pth (EfficientNet-V2S + regression head, the ESPACOL
Hybrid Contrastive Ordinal model), picks one fundus image, runs the
explainability pipeline (explainability.py), and pops up a SINGLE window with
five panels:

    Original | WHERE (LayerCAM heatmap) | WHY (concept scores)
    + WHY+WHERE (one concept-guided heatmap per clinical concept)

Run (from the repo root, ESPACOL-New/):
    python Hady/demo_explain_dr.py                 # random image (usually grade 0)
    python Hady/demo_explain_dr.py --grade 4       # random Proliferative-DR image
    python Hady/demo_explain_dr.py --index 0       # deterministic single image
    python Hady/demo_explain_dr.py --encoder random            # offline, no download
    python Hady/demo_explain_dr.py --grade 4 --save out.png --no-show   # headless

Notes
-----
* The repo model is built with use_image_text=True so it produces the gamma
  image-text embedding z_it; a tiny adapter (_ExplainAdapter) just exposes
  .features for the CAM hooks and forwards the model's dict output.
* --encoder biomedclip (default) downloads BioMedCLIP (~350 MB) on first run.
  --encoder random needs no download but its WHY scores are placeholders;
  the WHERE heatmap is fully meaningful either way.
* WHY concept scores are computed in the aligned z_it space using the RECOVERED
  text projection (Hady/DR_text_projection.pth, produced by
  recover_text_projection.py). If that file is missing, the demo falls back to a
  random-orthogonal text bridge and prints a warning (scores not gamma-aligned).
* ~74% of DR images are grade 0 (No DR), so a plain random pick usually shows a
  lesion-free fundus. Use --grade to see a specific severity.
"""

import argparse
import math
import os
import random
import sys

import numpy as np
import torch
import torch.nn as nn

# Make the repo root (parent of Hady/) importable regardless of CWD.
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _REPO_ROOT)

from Datasets.dataloaders import DRDataset, _IMAGENET_MEAN, _IMAGENET_STD
from models.framework import build_model
from utils.checkpoint import load_checkpoint
from utils.metrics import round_predictions

# Importing explainability locks matplotlib to the Agg backend (line ~60 there);
# we override the backend at runtime in main() when a pop-up window is wanted.
from explainability import (
    ExplainabilityPipeline,
    LayerCAM,
    DR_CONCEPTS,
    _fundus_mask,
)

CLASS_NAMES = DR_CONCEPTS["class_names"]          # ["No DR","Mild NPDR",...,"PDR"]
N_CLASSES = len(CLASS_NAMES)                       # 5
BIOMEDCLIP_ID = "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"


# ─────────────────────────────────────────────────────────────────────────────
# Model adapter: repo 3-output model -> 5-output shape explainability.py expects
# ─────────────────────────────────────────────────────────────────────────────

class _ExplainAdapter(nn.Module):
    """
    Thin wrapper around the repo's HybridContrastiveOrdinalModel that exposes
    `.features` (the EfficientNet stages) so LayerCAM can hook
    model.features[2,4,6,7], and forwards the model's dict output
    {features, z_pcol, z_scolw, z_it, pred} unchanged.
    """

    def __init__(self, base: nn.Module):
        super().__init__()
        self.base = base
        # Same nn.Sequential the CAM hooks need (models/backbone.py: base.features).
        self.features = base.backbone.features

    def forward(self, x: torch.Tensor):
        # Full forward runs backbone.features (CAM hooks fire) and returns the
        # dict explainability reads — including the aligned z_it embedding.
        return self.base(x)


# ─────────────────────────────────────────────────────────────────────────────
# BioMedCLIP loader (default WHY encoder)
# ─────────────────────────────────────────────────────────────────────────────

def load_biomedclip(device):
    """
    Load the BioMedCLIP text tower (same model the notebook uses). Returns
    (model, tokenizer), or (None, None) if open_clip / the download is unavailable
    so the caller can fall back to the random encoder.
    """
    try:
        import open_clip
    except ImportError:
        print("NOTE: open_clip not installed; cannot use --encoder biomedclip.")
        print("      Install with 'pip install open_clip_torch' or run --encoder random.")
        return None, None
    try:
        print(f"Loading BioMedCLIP ({BIOMEDCLIP_ID}) -- first run downloads ~350 MB ...")
        model, _, _ = open_clip.create_model_and_transforms(BIOMEDCLIP_ID)
        tokenizer = open_clip.get_tokenizer(BIOMEDCLIP_ID)
        model = model.to(device).eval()
        for p in model.parameters():
            p.requires_grad = False
        print("BioMedCLIP loaded.")
        return model, tokenizer
    except Exception as exc:  # download/network/HF errors
        print(f"NOTE: could not load BioMedCLIP ({exc}).")
        return None, None


# ─────────────────────────────────────────────────────────────────────────────
# Display helpers
# ─────────────────────────────────────────────────────────────────────────────

def to_display_image(x: torch.Tensor) -> np.ndarray:
    """De-normalize an ImageNet-normalized (3,H,W) tensor to (H,W,3) float [0,1]."""
    mean = torch.tensor(_IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(_IMAGENET_STD).view(3, 1, 1)
    img = (x.detach().cpu() * std + mean).clamp(0.0, 1.0)
    return img.permute(1, 2, 0).numpy()


def _safe_mask(image_np: np.ndarray) -> np.ndarray:
    """Soft fundus-disc mask; falls back to all-ones if scipy is missing."""
    try:
        return _fundus_mask(image_np)
    except Exception:
        return np.ones(image_np.shape[:2], dtype=np.float32)


def _apply_mask(heatmap: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Suppress out-of-fundus activations, then renormalize to [0,1] (as visualize() does)."""
    hm = heatmap * mask
    if hm.max() > hm.min():
        hm = (hm - hm.min()) / (hm.max() - hm.min() + 1e-8)
    return hm


# ─────────────────────────────────────────────────────────────────────────────
# Combined single-window figure
# ─────────────────────────────────────────────────────────────────────────────

def build_combined_figure(plt, pipeline, image_np, result, true_name, encoder):
    """One figure: top row (Original | WHERE | WHY bars), bottom (WHY+WHERE grid)."""
    import matplotlib.gridspec as gridspec

    mask = _safe_mask(image_np)
    class_names = pipeline.class_names

    # ── data ────────────────────────────────────────────────────────────────
    where_hm = _apply_mask(result["heatmap"], mask)
    names = [c for c, _ in result["concept_scores"]]
    vals = [s for _, s in result["concept_scores"]]
    cmaps = result.get("concept_heatmaps", {})
    all_sims = dict(result["concept_scores"])

    fig = plt.figure(figsize=(16, 9))
    outer = gridspec.GridSpec(2, 1, height_ratios=[1.0, 1.25], hspace=0.32)

    # ── top row: 3 panels ─────────────────────────────────────────────────────
    top = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=outer[0], wspace=0.25)

    ax_orig = fig.add_subplot(top[0])
    ax_orig.imshow((image_np * 255).astype(np.uint8))
    ax_orig.axis("off")
    ax_orig.set_title(f"Original\n(true: {true_name})")

    ax_where = fig.add_subplot(top[1])
    ax_where.imshow(LayerCAM.overlay(image_np, where_hm))
    ax_where.axis("off")
    ax_where.set_title(
        f"WHERE -> {class_names[result['target_class']]}\n"
        f"Pred: {result['predicted_class']} (score {result['regression_score']:.2f})"
    )

    ax_why = fig.add_subplot(top[2])
    colors = ["steelblue" if v >= 0 else "tomato" for v in vals]
    ax_why.barh(names[::-1], vals[::-1], color=colors[::-1])
    ax_why.axvline(0, color="black", linewidth=0.8)
    ax_why.set_xlim(-1, 1)
    ax_why.set_xlabel("Cosine similarity")
    ax_why.set_title(f"WHY: concept scores ({encoder})")

    # ── bottom block: per-concept WHY+WHERE grid ──────────────────────────────
    if cmaps:
        n = len(cmaps)
        ncols = min(5, n)
        nrows = math.ceil(n / ncols)
        bottom = gridspec.GridSpecFromSubplotSpec(
            nrows, ncols, subplot_spec=outer[1], wspace=0.1, hspace=0.4
        )
        for i, (concept, cmap) in enumerate(cmaps.items()):
            ax = fig.add_subplot(bottom[i])
            cmap_m = _apply_mask(cmap, mask)
            ax.imshow(LayerCAM.overlay(image_np, cmap_m, alpha=0.5))
            ax.axis("off")
            ax.set_title(f"{concept}\nsim={all_sims.get(concept, 0.0):.2f}", fontsize=8)

    fig.suptitle(
        f"ESPACOL explainability  |  {pipeline.dataset.upper()}  |  {result['predicted_class']}"
        f"   (Original | WHERE | WHY | WHY+WHERE)",
        fontsize=12, fontweight="bold",
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Dataset root resolution (mirrors demo_dr.py)
# ─────────────────────────────────────────────────────────────────────────────

def resolve_dr_root(explicit):
    """Return the first valid DR dataset root (contains train/ and trainLabels.csv), or None."""
    def _valid(p):
        return os.path.isdir(os.path.join(p, "train")) and os.path.isfile(
            os.path.join(p, "trainLabels.csv")
        )

    candidates = []
    if explicit:
        candidates.append(explicit)
    candidates += [
        os.path.join(_REPO_ROOT, "Datasets", "DR"),
        os.path.join(_REPO_ROOT, os.pardir, "Datasets", "DR"),
    ]
    for c in candidates:
        c = os.path.normpath(c)
        if _valid(c):
            return c
    return None


def parse_args():
    parser = argparse.ArgumentParser(
        description="DR explainability demo: pops up Original | WHERE | WHY | WHY+WHERE."
    )
    parser.add_argument(
        "--checkpoint",
        default=os.path.join(_HERE, "DR_best_model.pth"),
        help="Path to the trained DR checkpoint (default: next to this script).",
    )
    parser.add_argument(
        "--dr_root", default=None,
        help="DR dataset root (contains train/ and trainLabels.csv). Auto-resolved if omitted.",
    )
    parser.add_argument(
        "--train_csv", default=None,
        help="Labels CSV (default: <dr_root>/trainLabels.csv).",
    )
    parser.add_argument("--index", type=int, default=None, help="Image index (default: random).")
    parser.add_argument("--seed", type=int, default=None, help="Seed for the random image choice.")
    parser.add_argument(
        "--grade", type=int, default=None, choices=range(N_CLASSES),
        help="Restrict the random pick to images with this true grade (0-4). "
             "Useful because ~74%% of DR images are grade 0.",
    )
    parser.add_argument(
        "--encoder", choices=["biomedclip", "random"], default="biomedclip",
        help="Concept (WHY) encoder. 'biomedclip' (default) downloads ~350 MB on first run; "
             "'random' is offline but its WHY scores are placeholders.",
    )
    parser.add_argument(
        "--text_projection", default=None,
        help="Recovered z_it-space text projection .pth (default: Hady/DR_text_projection.pth). "
             "Produced by recover_text_projection.py; needed for gamma-aligned WHY scores.",
    )
    parser.add_argument("--save", default=None, help="Optional path to save the figure as PNG.")
    parser.add_argument("--no-show", action="store_true", help="Do not pop up a window (save only).")
    return parser.parse_args()


def main():
    args = parse_args()

    if not os.path.isfile(args.checkpoint):
        sys.exit(f"Checkpoint not found: {args.checkpoint}")

    dr_root = resolve_dr_root(args.dr_root)
    if dr_root is None:
        sys.exit(
            "DR dataset not found. Pass its location with --dr_root "
            "(it should contain a train/ folder and trainLabels.csv)."
        )
    train_csv = args.train_csv or os.path.join(dr_root, "trainLabels.csv")
    if not os.path.isfile(train_csv):
        sys.exit(f"Labels CSV not found: {train_csv}")

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
        print("NOTE: WHY concept scores are random placeholders; WHERE heatmap is real.")

    # ── Recovered text projection (maps concepts into the aligned z_it space) ──
    text_projection = None
    if encoder == "biomedclip":
        proj_path = args.text_projection or os.path.join(_HERE, "DR_text_projection.pth")
        if os.path.isfile(proj_path):
            from text_projection import load_text_projection
            text_projection, _bundle = load_text_projection(proj_path, device)
            print(f"Loaded recovered text projection: {proj_path}")
        else:
            print(f"NOTE: no recovered text projection at {proj_path}; WHY scores use "
                  f"the random-orthogonal bridge (NOT gamma-aligned). "
                  f"Run Hady/recover_text_projection.py first.")

    # ── Pick one image (DRDataset applies the eval transform internally) ──────
    print("Indexing DR dataset (this can take a few seconds)...")
    ds = DRDataset(root_dir=dr_root, split="train", csv_path=train_csv)
    print(f"DR images found: {len(ds)}")

    if args.grade is not None:
        candidates = [i for i, (_, lab) in enumerate(ds.items) if lab == args.grade]
        if not candidates:
            sys.exit(f"No images with grade {args.grade} found in {train_csv}.")
    else:
        candidates = list(range(len(ds)))

    if args.seed is not None:
        random.seed(args.seed)
    if args.index is not None:
        if not 0 <= args.index < len(ds):
            sys.exit(f"--index must be in [0, {len(ds) - 1}]")
        idx = args.index
    else:
        idx = random.choice(candidates)
    x, y = ds[idx]
    img_path = ds.items[idx][0]
    true_label = int(y)
    true_name = CLASS_NAMES[true_label]

    # ── Run explainability ────────────────────────────────────────────────────
    n_concepts = len(DR_CONCEPTS["concepts"])
    pipeline = ExplainabilityPipeline(
        model, dataset="dr", device=device, encoder=encoder,
        text_model=text_model, tokenizer=tokenizer, text_projection=text_projection,
        top_k=n_concepts,
    )
    print("Computing explanations (LayerCAM + per-concept heatmaps) ...")
    result = pipeline.explain(x.unsqueeze(0).to(device), concept_heatmaps=True)
    pipeline.remove_hooks()

    print("-" * 60)
    print(f"Image      : {os.path.basename(img_path)}  (index {idx})")
    print(f"True grade : {true_name} ({true_label})")
    print(f"Reg. score : {result['regression_score']:.4f}")
    print(f"Prediction : {result['predicted_class']} ({result['predicted_label']})")
    print(f"Result     : {'CORRECT' if result['predicted_label'] == true_label else 'WRONG'}")
    print("-" * 60)

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

    fig = build_combined_figure(plt, pipeline, to_display_image(x), result, true_name, encoder)

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
