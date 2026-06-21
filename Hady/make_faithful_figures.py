"""
Batch-render faithfulness STEP-1 (concept-drop) figures across all classes/grades.

Loads each trained model + BioMedCLIP encoder ONCE and iterates over a sampled set of
images, saving a 4-panel concept-drop figure per image (Original | WHERE H_c | Occluded |
before-vs-after concept scores) plus a summary.csv, into a single folder.

    BUSI : normal / benign / malignant      (N images each)
    DR   : grades 0..4 (No DR..PDR)         (N images each)

Run (from the repo root, ESPACOL-New/):
    source slurm/_env.sh
    python Hady/make_faithful_figures.py --per-class 5 --seed 0 \
        --busi_root "$BUSI_ROOT" --dr_root "$DR_ROOT" --out Hady/faithful_figures

    # quick end-to-end smoke check (~1 min):
    python Hady/make_faithful_figures.py --datasets busi --per-class 1 --out /tmp/faith_smoke

Notes
-----
* Reuses the single-image demos' setup helpers and the shared step-1 logic in
  Hady/faithfulness.py (run_concept_drop). One ExplainabilityPipeline per dataset is
  reused across all of that dataset's images; hooks are removed once at the end.
* The concept-drop is only interpretable with --encoder biomedclip AND the recovered
  text projections (Hady/BUSI_text_projection.pth / Hady/DR_text_projection.pth).
* DR caveat: for severe/proliferative DR all cosines go negative — the drop check is
  sign-agnostic (c_before - c_after), so a faithful H_c still yields a positive drop.
"""

import argparse
import csv
import os
import random
import sys
import traceback

import matplotlib
matplotlib.use("Agg")  # always headless
import matplotlib.pyplot as plt

import numpy as np
import torch
from PIL import Image

# Make the repo root (parent of Hady/) importable regardless of CWD.
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _REPO_ROOT)

from Datasets.dataloaders import BUSIDataset, DRDataset
from models.framework import build_model
from utils.checkpoint import load_checkpoint
from explainability import ExplainabilityPipeline, LayerCAM, BUSI_CONCEPTS, DR_CONCEPTS

# Shared setup helpers from the single-image demos (identical copies live in both;
# import the dataset-agnostic ones from one module, dataset-specific ones aliased).
from demo_explain_busi import (
    _ExplainAdapter,
    load_biomedclip,
    to_display_image,
    _safe_mask as busi_safe_mask,
    resolve_busi_root,
    LABEL_TO_CLASS as BUSI_LABEL_TO_CLASS,
    N_CLASSES as BUSI_N,
)
from demo_explain_dr import (
    _safe_mask as dr_safe_mask,
    resolve_dr_root,
    CLASS_NAMES as DR_CLASS_NAMES,
    N_CLASSES as DR_N,
)
from faithfulness import run_concept_drop, build_faithfulness_figure, _apply_mask

from text_projection import load_text_projection


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _sample_indices(items, label, n, rng):
    """Up to n image indices whose label == `label`, sampled with `rng` (reproducible)."""
    pool = [i for i, (_, lab) in enumerate(items) if int(lab) == int(label)]
    if not pool:
        return []
    rng.shuffle(pool)
    return pool[:n]


def _save_rgb(arr, path):
    """Save an (H,W,3) image (float [0,1] or uint8) as a clean standalone PNG."""
    if arr.dtype != np.uint8:
        arr = (np.clip(arr, 0.0, 1.0) * 255).astype(np.uint8)
    Image.fromarray(arr).save(path)


def _load_projection(path, device, what):
    if os.path.isfile(path):
        proj, _bundle = load_text_projection(path, device)
        print(f"Loaded recovered text projection: {path}")
        return proj
    print(f"NOTE: no recovered text projection at {path}; WHY scores are NOT gamma-aligned "
          f"(run Hady/recover_text_projection.py to produce it). [{what}]")
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Per-dataset batch run
# ─────────────────────────────────────────────────────────────────────────────

def run_dataset(
    dataset_name, ds, n_classes, class_names, concepts, safe_mask,
    checkpoint, text_projection_path, device, encoder, text_model, tokenizer,
    labels, label_namer, file_namer, args, out_dir, rows,
):
    """Build one model+pipeline for `dataset_name`, then render per-image figures."""
    print("=" * 64)
    print(f"{dataset_name}: building model and pipeline ...")
    base = build_model(n_classes=n_classes, pretrained=False, use_image_text=True)
    load_checkpoint(checkpoint, base, optimizer=None, device=device)
    base.to(device).eval()
    model = _ExplainAdapter(base).to(device).eval()

    text_projection = None
    if encoder == "biomedclip":
        text_projection = _load_projection(text_projection_path, device, dataset_name)

    pipeline = ExplainabilityPipeline(
        model, dataset=dataset_name.lower(), device=device, encoder=encoder,
        text_model=text_model, tokenizer=tokenizer, text_projection=text_projection,
        top_k=len(concepts),
    )

    panels_dir = None
    if args.dump_panels:
        panels_dir = os.path.join(out_dir, "panels")
        os.makedirs(panels_dir, exist_ok=True)

    rng = random.Random(args.seed)
    n_pass = n_fail = 0
    for label in labels:
        idxs = _sample_indices(ds.items, label, args.per_class, rng)
        lab_name = label_namer(label)
        if not idxs:
            print(f"  [{dataset_name} {lab_name}] no images found — skipping.")
            continue
        for idx in idxs:
            try:
                x, y = ds[idx]
                true_label = int(y)
                true_name = class_names[true_label]
                x0 = x.unsqueeze(0).to(device)
                image_np = to_display_image(x)
                mask = safe_mask(image_np)
                tissue = None if args.no_tissue_mask else mask

                result, x_occ, M, scores_before, scores_after, reg_after, check = run_concept_drop(
                    pipeline, x0, tissue_mask=tissue, occlusion=args.occlusion,
                    threshold=args.threshold, blur_sigma=args.blur_sigma,
                )

                pred_before = result["predicted_class"]
                reg_before = result["regression_score"]
                after_label = int(round(min(max(reg_after, 0.0), n_classes - 1)))
                pred_after = class_names[after_label]
                verdict = "PASS" if check["passed"] else "FAIL"
                n_pass += check["passed"]
                n_fail += not check["passed"]

                occ_np = to_display_image(x_occ.squeeze(0))
                fig = build_faithfulness_figure(
                    plt, image_np, occ_np, M, check["concept"], scores_before, scores_after,
                    check, mask, true_name, pred_before, pred_after,
                    dataset_name=dataset_name, baseline_word=args.occlusion,
                )
                fname = file_namer(label, idx, verdict)
                fpath = os.path.join(out_dir, fname)
                fig.savefig(fpath, dpi=150, bbox_inches="tight")
                plt.close(fig)

                # Standalone before/after (and the masked WHERE region) as clean PNGs.
                original_file = occluded_file = where_file = ""
                if panels_dir is not None:
                    stem = fname.rsplit("_", 1)[0]  # drop the "_<verdict>.png"
                    where_overlay = LayerCAM.overlay(image_np, _apply_mask(M, mask))
                    _save_rgb(image_np, os.path.join(panels_dir, stem + "_original.png"))
                    _save_rgb(occ_np, os.path.join(panels_dir, stem + "_occluded.png"))
                    _save_rgb(where_overlay, os.path.join(panels_dir, stem + "_where.png"))
                    original_file = os.path.join("panels", stem + "_original.png")
                    occluded_file = os.path.join("panels", stem + "_occluded.png")
                    where_file = os.path.join("panels", stem + "_where.png")

                img_base = os.path.basename(ds.items[idx][0])
                print(f"  [{dataset_name} {lab_name}] idx {idx:>6} {img_base:<22} "
                      f"top={check['concept']:<28} drop={check['drop']:+.3f} {verdict} -> {fname}")
                rows.append({
                    "dataset": dataset_name, "label_name": lab_name, "true_label": true_label,
                    "image": img_base, "top_concept": check["concept"],
                    "c_before": f"{check['c_before']:.4f}", "c_after": f"{check['c_after']:.4f}",
                    "drop": f"{check['drop']:.4f}", "l_drop": f"{check['l_drop']:.4f}",
                    "verdict": verdict, "pred_before": pred_before, "pred_after": pred_after,
                    "reg_before": f"{reg_before:.4f}", "reg_after": f"{reg_after:.4f}",
                    "occlusion": args.occlusion, "file": fname,
                    "original_file": original_file, "occluded_file": occluded_file,
                    "where_file": where_file,
                })
            except Exception as exc:  # never let one image abort the whole batch
                print(f"  [{dataset_name} {lab_name}] idx {idx}: ERROR {exc}")
                traceback.print_exc()

    pipeline.remove_hooks()
    print(f"{dataset_name}: {n_pass} PASS / {n_fail} FAIL")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Batch-render faithfulness (concept-drop) figures across BUSI classes "
                    "and DR grades into one folder."
    )
    p.add_argument("--per-class", type=int, default=5,
                   help="Images per BUSI class / DR grade (default 5 → 15 BUSI + 25 DR).")
    p.add_argument("--datasets", choices=["both", "busi", "dr"], default="both")
    p.add_argument("--occlusion", choices=["blur", "gray"], default="blur",
                   help="How to obscure the top concept's region (default blur).")
    p.add_argument("--seed", type=int, default=0, help="Seed for reproducible image sampling.")
    p.add_argument("--out", default=os.path.join(_HERE, "faithful_figures"),
                   help="Output folder (default: Hady/faithful_figures).")
    p.add_argument("--encoder", choices=["biomedclip", "random"], default="biomedclip")
    p.add_argument("--threshold", type=float, default=0.0,
                   help="Zero out H_c below this before occluding (sharpen the mask).")
    p.add_argument("--blur-sigma", type=float, default=21.0,
                   help="Gaussian sigma for --occlusion blur (default 21).")
    p.add_argument("--no-tissue-mask", action="store_true",
                   help="Do not restrict occlusion to the tissue/fundus region.")
    p.add_argument("--dump-panels", action=argparse.BooleanOptionalAction, default=True,
                   help="Also save standalone <stem>_original/_occluded/_where PNGs into "
                        "<out>/panels/ (default on; use --no-dump-panels to skip).")
    # roots / checkpoints / projections (auto-resolved if omitted, mirroring the demos)
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
    os.makedirs(args.out, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}  |  out: {args.out}  |  per-class: {args.per_class}  "
          f"|  occlusion: {args.occlusion}")

    do_busi = args.datasets in ("both", "busi")
    do_dr = args.datasets in ("both", "dr")

    # Load BioMedCLIP once, reuse for both datasets (text tower is dataset-agnostic).
    encoder = args.encoder
    text_model = tokenizer = None
    if encoder == "biomedclip":
        text_model, tokenizer = load_biomedclip(device)
        if text_model is None:
            print("Falling back to --encoder random (WHY scores are placeholders).")
            encoder = "random"
    if encoder == "random":
        print("NOTE: WHY concept scores are random placeholders; the concept-drop is NOT meaningful.")

    rows = []

    if do_busi:
        busi_root = resolve_busi_root(args.busi_root)
        if busi_root is None:
            print("WARNING: BUSI dataset not found; skipping BUSI. Pass --busi_root.")
        elif not os.path.isfile(args.busi_ckpt):
            print(f"WARNING: BUSI checkpoint not found ({args.busi_ckpt}); skipping BUSI.")
        else:
            ds = BUSIDataset(root_dir=busi_root, split="all")
            print(f"BUSI images found: {len(ds)}")
            run_dataset(
                dataset_name="BUSI", ds=ds, n_classes=BUSI_N,
                class_names=[BUSI_LABEL_TO_CLASS[i] for i in range(BUSI_N)],
                concepts=BUSI_CONCEPTS["concepts"], safe_mask=busi_safe_mask,
                checkpoint=args.busi_ckpt, text_projection_path=args.busi_text_projection,
                device=device, encoder=encoder, text_model=text_model, tokenizer=tokenizer,
                labels=sorted(BUSIDataset.CLASS_TO_LABEL.values()),
                label_namer=lambda lab: BUSI_LABEL_TO_CLASS[lab],
                file_namer=lambda lab, idx, v: f"busi_{BUSI_LABEL_TO_CLASS[lab]}_idx{idx}_{v}.png",
                args=args, out_dir=args.out, rows=rows,
            )

    if do_dr:
        dr_root = resolve_dr_root(args.dr_root)
        if dr_root is None:
            print("WARNING: DR dataset not found; skipping DR. Pass --dr_root.")
        elif not os.path.isfile(args.dr_ckpt):
            print(f"WARNING: DR checkpoint not found ({args.dr_ckpt}); skipping DR.")
        else:
            train_csv = args.train_csv or os.path.join(dr_root, "trainLabels.csv")
            print("Indexing DR dataset (this can take a few seconds)...")
            ds = DRDataset(root_dir=dr_root, split="train", csv_path=train_csv)
            print(f"DR images found: {len(ds)}")
            run_dataset(
                dataset_name="DR", ds=ds, n_classes=DR_N, class_names=DR_CLASS_NAMES,
                concepts=DR_CONCEPTS["concepts"], safe_mask=dr_safe_mask,
                checkpoint=args.dr_ckpt, text_projection_path=args.dr_text_projection,
                device=device, encoder=encoder, text_model=text_model, tokenizer=tokenizer,
                labels=list(range(DR_N)),
                label_namer=lambda lab: f"grade{lab} ({DR_CLASS_NAMES[lab]})",
                file_namer=lambda lab, idx, v: f"dr_grade{lab}_idx{idx}_{v}.png",
                args=args, out_dir=args.out, rows=rows,
            )

    # ── summary.csv ───────────────────────────────────────────────────────────
    if rows:
        csv_path = os.path.join(args.out, "summary.csv")
        fields = ["dataset", "label_name", "true_label", "image", "top_concept",
                  "c_before", "c_after", "drop", "l_drop", "verdict",
                  "pred_before", "pred_after", "reg_before", "reg_after", "occlusion", "file",
                  "original_file", "occluded_file", "where_file"]
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(rows)
        n_pass = sum(r["verdict"] == "PASS" for r in rows)
        print("=" * 64)
        print(f"Wrote {len(rows)} figures to {args.out}  ({n_pass} PASS / {len(rows) - n_pass} FAIL)")
        print(f"Summary: {csv_path}")
    else:
        print("No figures rendered (check dataset roots / checkpoints).")


if __name__ == "__main__":
    main()
