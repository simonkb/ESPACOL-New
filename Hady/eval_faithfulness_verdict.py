"""
Faithfulness VERDICT run — scaled, distribution-aware, statistically tested.

The 40-image concept-drop run (make_faithful_figures.py / OBSCURATION_RESULTS.md)
is a pilot: n=5 per group, point estimates only, no control at scale. It can show
that obscuration *can* work, not *whether* it works. This script produces a
verdict-grade result by fixing the three weaknesses called out in the analysis:

  1. SCALE      — N per group (default 100) instead of 5, so a single image can no
                  longer swing a group.
  2. CONTROL    — runs the decisive step-3 specificity control (on-target vs
                  off-target blur) on EVERY image, not just the bare concept-drop.
                  The verdict rests on `drop_on - drop_off`, not the raw drop.
  3. STATISTICS — reports per-group DISTRIBUTIONS (median/IQR), paired Wilcoxon
                  significance tests (is the drop real? is the specificity real?),
                  bootstrap CIs, Benjamini-Hochberg FDR across the 8 groups, and an
                  auto-derived verdict label per group. Also tracks the grade-collapse
                  coupling (reg before vs after on-target occlusion).

Outputs (into --out):
  * per_image.csv          one row per image (all raw metrics)
  * per_group_verdict.csv  one row per group (stats + tests + verdict)
  * verdict_report.md      human-readable summary table + method/caveats
  * verdict.json           machine-readable per-group verdict
  * verdict_specificity_box.png   per-group box plot of per-image specificity
  * verdict_dropon_vs_off.png     per-group median drop_on vs drop_off (IQR bars)

Run — ALWAYS via SLURM (never `python ...` on a login node; see Hady/CLAUDE.md):
    sbatch slurm/faithful_verdict.sbatch
or on an interactive compute node:
    source slurm/_env.sh
    python Hady/eval_faithfulness_verdict.py --datasets both --per-class 100 \
        --controls 4 --seed 0 --busi_root "$BUSI_ROOT" --dr_root "$DR_ROOT" \
        --out "$RUNS/faithful_verdict"

Only meaningful with --encoder biomedclip + the recovered text projections.
"""

import argparse
import csv
import json
import os
import random
import sys
import traceback

import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt

import numpy as np
import torch

# Make the repo root (parent of Hady/) importable regardless of CWD.
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _REPO_ROOT)

from demo_explain_busi import load_biomedclip, to_display_image
from demo_faith_control import setup
from faithfulness import occlusion_specificity, recompute_scores

# scipy is in requirements.txt; degrade gracefully if a test can't be computed.
try:
    from scipy.stats import wilcoxon, bootstrap
    _HAVE_SCIPY = True
except Exception:  # pragma: no cover
    _HAVE_SCIPY = False
    print("WARNING: scipy not available — significance tests/CIs will be NaN.")


# Verdict thresholds (documented in the report; the CSV carries the raw numbers so
# a human can override the auto-label).
INERT_EPS = 0.02     # |median drop_on| below this ⇒ nothing removable
SPEC_EPS = 0.02      # median specificity above this (and significant) ⇒ faithful
BIG_MASK = 0.35      # median mask fraction above this ⇒ flag the big-mask confound
ALPHA = 0.05         # significance level (after BH-FDR correction)


# ─────────────────────────────────────────────────────────────────────────────
# Held-out (no-data-leak) reconstruction
# ─────────────────────────────────────────────────────────────────────────────
#
# Both gamma checkpoints are k-fold CV models trained on the FULL dataset
# (DR: 35,126 imgs, 10-fold patient-stratified; BUSI: 780 imgs, 5-fold). For each
# checkpoint exactly one fold was HELD OUT (val=test, never in gradient updates):
# BUSI fold 4, DR fold 2 — the folds these checkpoints are named for. We rebuild
# the same split (training/cross_val.py, seed 42) on the SAME item list the eval
# loaded, so every image can be tagged train vs heldout. This lets us report the
# verdict on the data-leak-free held-out subset and quantify in-sample inflation.

# The reconstruction is validated by the MEMORISATION GAP, not a hardcoded
# accuracy (which would be confounded by class-balanced sampling vs the natural
# distribution): a correct held-out tag marks images the model never trained on,
# so the model fits TRAIN at least as well as HELD-OUT. See _heldout_self_check.


def heldout_paths(dataset, items, fold_idx):
    """Set of image PATHS in the checkpoint's held-out CV fold.

    Joins by path, not integer index (BUSI's os.listdir order — and thus idx — is
    not stable across processes). BUSI additionally prefers a recovered manifest:
    the storage migration scrambled BUSI's os.listdir order, so reconstructing from
    `items` would yield the WRONG fold. `recover_busi_fold.py` rebuilds the true fold
    from the pre-migration z_it cache into Hady/busi_fold4_heldout.json; we use it
    when present (matched by basename). DR keys on CSV row order and is unaffected.
    """
    from training.cross_val import BUSICrossValidator, DRCrossValidator
    if dataset == "busi":
        manifest = os.path.join(_HERE, "busi_fold4_heldout.json")
        if os.path.isfile(manifest):
            with open(manifest) as f:
                m = json.load(f)
            if int(m.get("fold", -1)) == int(fold_idx):
                held = set(m["heldout_basenames"])
                paths = {p for p, _ in items if os.path.basename(p) in held}
                print(f"BUSI held-out: {len(paths)} paths from recovered manifest "
                      f"{os.path.basename(manifest)} (fold {fold_idx}).")
                return paths
            print(f"NOTE: {os.path.basename(manifest)} is for fold {m.get('fold')} "
                  f"!= {fold_idx}; falling back to os.listdir reconstruction.")
        else:
            print("NOTE: no recovered BUSI manifest; reconstructing from os.listdir "
                  "(UNRELIABLE if /dpc storage was migrated — run recover_busi_fold.py).")
        cv = BUSICrossValidator(items, n_folds=5, val_fraction=0.1, seed=42)
    else:
        cv = DRCrossValidator(items, n_folds=10, val_fraction=0.1, seed=42)
    _, _, test_items = cv.get_fold(fold_idx)
    return {p for p, _ in test_items}


def _check_ckpt_fold(ckpt_path, fold_idx, dataset):
    """Warn if the loaded checkpoint's filename doesn't match the requested fold."""
    real = os.path.realpath(ckpt_path)  # resolve e.g. the BUSI_best_model.pth symlink
    if f"fold{fold_idx}" not in os.path.basename(real):
        print(f"WARNING: {dataset.upper()} checkpoint {os.path.basename(real)} does not "
              f"mention fold{fold_idx}; the held-out reconstruction (--{dataset}-fold "
              f"{fold_idx}) may not match the fold this checkpoint was trained against.")


def _heldout_self_check(dataset, fold_idx, dataset_rows):
    """Diagnose the reconstruction via the MEMORISATION GAP (sampling-robust).

    A correct held-out tag marks images the model never trained on, so the model
    fits TRAIN at least as well as HELD-OUT: held-out MAE >= train MAE. A held-out
    set that scores as well as (or better than) train is capturing training images
    — a broken reconstruction (e.g. changed os.listdir order). Comparing to a
    hardcoded logged accuracy would be confounded by class-balanced sampling, so we
    use the within-run train-vs-heldout gap instead. Reuses the per-image predicted
    grade (grade_before); no extra forward passes.
    """
    train = [r for r in dataset_rows if r["split"] == "train"]
    held = [r for r in dataset_rows if r["split"] == "heldout"]
    n = len(dataset_rows)
    print(f"[{dataset}] partition: n={n}  train={len(train)}  heldout={len(held)}  "
          f"(heldout {100 * len(held) / max(n, 1):.1f}% of evaluated images)")
    if len(train) + len(held) != n:
        print(f"  WARNING: train+heldout ({len(train)}+{len(held)}) != {n} — overlap?")
    if not held or not train:
        print("  NOTE: a partition is empty at this sample size — gap check skipped.")
        return

    def _mae(rows):
        return float(np.mean([abs(r["grade_before"] - r["true_label"]) for r in rows]))

    def _acc(rows):
        return 100.0 * np.mean([r["grade_before"] == r["true_label"] for r in rows])

    tr_mae, ho_mae = _mae(train), _mae(held)
    print(f"  train  : acc={_acc(train):5.2f}%  MAE={tr_mae:.4f}")
    print(f"  heldout: acc={_acc(held):5.2f}%  MAE={ho_mae:.4f}   "
          f"(memorisation gap heldout-train MAE = {ho_mae - tr_mae:+.4f})")
    if ho_mae < tr_mae - 0.005:
        print("  WARNING: held-out fits BETTER than train — the held-out tag is likely "
              "capturing TRAINING images, so the reconstruction is WRONG for this "
              "dataset. Treat its train/heldout split as unreliable (the OVERALL "
              "verdict is unaffected).")


# ─────────────────────────────────────────────────────────────────────────────
# Stats helpers
# ─────────────────────────────────────────────────────────────────────────────

def _wilcoxon_greater(x, y):
    """One-sided paired Wilcoxon: median(x - y) > 0. Returns p-value or NaN."""
    if not _HAVE_SCIPY:
        return float("nan")
    x, y = np.asarray(x, float), np.asarray(y, float)
    d = x - y
    if d.size < 6 or np.allclose(d, 0.0):  # too few / all-tied ⇒ undefined
        return float("nan")
    try:
        return float(wilcoxon(x, y, alternative="greater", zero_method="wilcox").pvalue)
    except Exception:
        return float("nan")


def _median_ci(values, seed=0):
    """Percentile bootstrap 95% CI for the median. Returns (lo, hi) or (nan, nan)."""
    v = np.asarray(values, float)
    if not _HAVE_SCIPY or v.size < 8 or np.allclose(v, v[0]):
        return (float("nan"), float("nan"))
    try:
        res = bootstrap((v,), np.median, n_resamples=2000, confidence_level=0.95,
                        method="percentile", random_state=np.random.default_rng(seed))
        return (float(res.confidence_interval.low), float(res.confidence_interval.high))
    except Exception:
        return (float("nan"), float("nan"))


def _bh_fdr(pvals):
    """Benjamini-Hochberg adjusted p-values; NaNs pass through as NaN."""
    p = np.asarray(pvals, float)
    keep = ~np.isnan(p)
    out = np.full(p.shape, np.nan)
    pk = p[keep]
    n = pk.size
    if n == 0:
        return out
    order = np.argsort(pk)
    sp = pk[order]
    adj = np.empty(n)
    cummin = 1.0
    for i in range(n - 1, -1, -1):
        cummin = min(cummin, sp[i] * n / (i + 1))
        adj[i] = cummin
    back = np.empty(n)
    back[order] = np.clip(adj, 0, 1)
    out[keep] = back
    return out


def _verdict(median_drop_on, median_spec, p_drop, p_spec_adj, median_mask):
    """Auto-derive a regime label from the group statistics."""
    drop_sig = (not np.isnan(p_drop)) and p_drop < ALPHA
    spec_sig = (not np.isnan(p_spec_adj)) and p_spec_adj < ALPHA

    if abs(median_drop_on) < INERT_EPS and not drop_sig:
        return "INERT"
    if median_drop_on < -INERT_EPS:
        return "ANTI-FAITHFUL"
    if spec_sig and median_spec > SPEC_EPS:
        return "FAITHFUL"
    if drop_sig and not spec_sig:
        return "GENERIC" + (" (big-mask)" if median_mask > BIG_MASK else "")
    return "WEAK/INCONCLUSIVE"


# ─────────────────────────────────────────────────────────────────────────────
# Per-group evaluation
# ─────────────────────────────────────────────────────────────────────────────

def eval_group(bundle, label, dataset, args, ctrl_rng, sample_rng):
    """Run the specificity control over up to args.per_class images of one group."""
    pipe, ds, n_classes, safe_mask = (bundle["pipeline"], bundle["ds"],
                                      bundle["n_classes"], bundle["safe_mask"])
    group = bundle["namer"](label)
    pool = [i for i, (_, lab) in enumerate(ds.items) if int(lab) == int(label)]
    sample_rng.shuffle(pool)
    idxs = pool if args.per_class in (0, -1) else pool[:args.per_class]

    rows = []
    for idx in idxs:
        try:
            x, y = ds[idx]
            x0 = x.unsqueeze(0).to(pipe.device)
            image_np = to_display_image(x)
            tissue = None if args.no_tissue_mask else safe_mask(image_np)

            spec = occlusion_specificity(
                pipe, x0, tissue_mask=tissue, occlusion=args.occlusion,
                blur_sigma=args.blur_sigma, n_roll=args.controls, rng=ctrl_rng,
            )
            # grade-collapse coupling: grade before vs after the ON-target occlusion
            _, reg_on = recompute_scores(pipe, spec["x_on"])
            g_before = int(round(min(max(spec["reg_before"], 0.0), n_classes - 1)))
            g_after = int(round(min(max(reg_on, 0.0), n_classes - 1)))

            rows.append({
                "dataset": dataset.upper(), "group": group, "true_label": int(y),
                "image": os.path.basename(ds.items[idx][0]), "index": idx,
                "split": "heldout" if ds.items[idx][0] in bundle["heldout"] else "train",
                "top_concept": spec["top"],
                "c_before": spec["c_before"], "c_on": spec["c_on"],
                "drop_on": spec["drop_on"], "drop_off": spec["drop_ctrl"],
                "specificity": spec["specificity"], "mask_frac": spec["mask_frac"],
                "reg_before": spec["reg_before"], "reg_on": float(reg_on),
                "grade_before": g_before, "grade_after": g_after,
                "grade_changed": int(g_before != g_after),
            })
        except Exception as exc:  # never let one image abort the group
            print(f"  [{dataset} {group}] idx {idx}: ERROR {exc}")
            traceback.print_exc()

    print(f"  [{dataset} {group}] n={len(rows)}")
    return rows


def summarize_group(rows, ci_seed):
    """Group-level distribution stats + paired significance tests."""
    g = lambda k: np.array([r[k] for r in rows], float)
    drop_on, drop_off, spec = g("drop_on"), g("drop_off"), g("specificity")
    c_before, c_on, mask = g("c_before"), g("c_on"), g("mask_frac")
    spec_lo, spec_hi = _median_ci(spec, seed=ci_seed)
    return {
        "n": len(rows),
        "drop_on_med": float(np.median(drop_on)), "drop_on_mean": float(np.mean(drop_on)),
        "drop_on_iqr": [float(np.percentile(drop_on, 25)), float(np.percentile(drop_on, 75))],
        "drop_off_med": float(np.median(drop_off)),
        "spec_med": float(np.median(spec)), "spec_mean": float(np.mean(spec)),
        "spec_iqr": [float(np.percentile(spec, 25)), float(np.percentile(spec, 75))],
        "spec_ci": [spec_lo, spec_hi],
        "mask_med": float(np.median(mask)),
        "pass_rate": float(np.mean(drop_on > 0)),          # bare concept-drop PASS
        "locspec_rate": float(np.mean(drop_on > drop_off)),  # per-image on>off
        "anti_rate": float(np.mean(drop_on < 0)),
        "grade_change_rate": float(np.mean(g("grade_changed"))),
        "p_drop": _wilcoxon_greater(c_before, c_on),         # is the drop real?
        "p_spec_raw": _wilcoxon_greater(drop_on, drop_off),  # is specificity real?
    }


# ─────────────────────────────────────────────────────────────────────────────
# Plots
# ─────────────────────────────────────────────────────────────────────────────

_VCOLOR = {"FAITHFUL": "#2ca02c", "GENERIC": "#ff7f0e", "INERT": "#7f7f7f",
           "ANTI-FAITHFUL": "#d62728", "WEAK/INCONCLUSIVE": "#9467bd"}


def _vcolor(v):
    return next((c for k, c in _VCOLOR.items() if v.startswith(k)), "#1f77b4")


def plot_specificity_box(per_group, per_image_by_group, out_path):
    groups = [g["group_key"] for g in per_group]
    data = [per_image_by_group[k] for k in groups]
    colors = [_vcolor(g["verdict"]) for g in per_group]
    fig, ax = plt.subplots(figsize=(max(8, 1.4 * len(groups)), 6))
    bp = ax.boxplot(data, patch_artist=True, showmeans=True, widths=0.6)
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c); patch.set_alpha(0.55)
    ax.axhline(0, color="black", lw=1.0, ls="--")
    ax.set_xticks(range(1, len(groups) + 1))
    ax.set_xticklabels([g.replace(" ", "\n") for g in groups], fontsize=8)
    ax.set_ylabel("per-image specificity  (drop_on − drop_off)")
    ax.set_title("Faithfulness verdict — per-group specificity distribution\n"
                 "(box above the dashed 0 line ⇒ heatmap location matters)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_dropon_vs_off(per_group, out_path):
    groups = [g["group_key"] for g in per_group]
    x = np.arange(len(groups))
    on = [g["drop_on_med"] for g in per_group]
    off = [g["drop_off_med"] for g in per_group]
    on_err = [[g["drop_on_med"] - g["drop_on_iqr"][0] for g in per_group],
              [g["drop_on_iqr"][1] - g["drop_on_med"] for g in per_group]]
    fig, ax = plt.subplots(figsize=(max(8, 1.4 * len(groups)), 6))
    ax.bar(x - 0.2, on, width=0.4, yerr=on_err, capsize=3, label="drop_on (median, IQR)",
           color="steelblue")
    ax.bar(x + 0.2, off, width=0.4, label="drop_off (median)", color="orange")
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([g.replace(" ", "\n") for g in groups], fontsize=8)
    ax.set_ylabel("concept-score drop")
    ax.set_title("Faithfulness verdict — on-target vs off-target drop per group")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Report
# ─────────────────────────────────────────────────────────────────────────────

_PART_TITLES = {"overall": "Overall (all images)",
                "train": "Train-only (in-sample)",
                "heldout": "Held-out only (no data leak)"}


def _verdict_table(A, groups):
    """Append one markdown verdict table for a list of per-group stat dicts."""
    A("| group | n | drop_on med [IQR] | drop_off | specificity med (95% CI) | "
      "p_drop | p_spec(FDR) | mask% | grade-Δ% | verdict |")
    A("|---|--:|---|--:|---|--:|--:|--:|--:|---|")
    for g in groups:
        A("| {grp} | {n} | {don:+.3f} [{q1:+.2f},{q3:+.2f}] | {doff:+.3f} | "
          "{spec:+.3f} ({lo:+.2f},{hi:+.2f}) | {pd} | {ps} | {mask:.0f}% | "
          "{gc:.0f}% | **{v}** |".format(
              grp=g["group_key"], n=g["n"], don=g["drop_on_med"],
              q1=g["drop_on_iqr"][0], q3=g["drop_on_iqr"][1], doff=g["drop_off_med"],
              spec=g["spec_med"], lo=g["spec_ci"][0], hi=g["spec_ci"][1],
              pd=_fmt_p(g["p_drop"]), ps=_fmt_p(g["p_spec_adj"]),
              mask=100 * g["mask_med"], gc=100 * g["grade_change_rate"], v=g["verdict"]))


def write_report(per_group, args, out_dir, n_images):
    lines = []
    A = lines.append
    per_class = "all" if args.per_class in (0, -1) else f"up to {args.per_class}"
    n_groups = len(per_group["overall"])
    A("# Faithfulness verdict — full dataset, train vs held-out\n")
    A(f"- **Images**: {n_images} total  |  {per_class} per group, "
      f"{n_groups} groups  |  seed {args.seed}")
    A(f"- **Occlusion**: {args.occlusion} (σ={args.blur_sigma}), "
      f"{args.controls + 2} off-target controls/image  |  encoder: {args.encoder}")
    A(f"- **Tests**: paired one-sided Wilcoxon; specificity p-values BH-FDR corrected "
      f"within each partition; α={ALPHA}.")
    A(f"- **Partitions**: each image is tagged by whether it was in the checkpoint's "
      f"k-fold TRAINING set or its HELD-OUT fold (BUSI fold {args.busi_fold}, "
      f"DR fold {args.dr_fold}). The held-out table is the data-leak-free verdict; "
      "train vs held-out measures in-sample (memorisation) inflation.\n")
    A("Verdict rule: **FAITHFUL** = specificity significant & median > "
      f"{SPEC_EPS}; **GENERIC** = drop real but specificity not (mask>{BIG_MASK} ⇒ "
      "big-mask); **INERT** = |median drop|<%.2f & drop n.s.; **ANTI-FAITHFUL** = "
      "median drop_on<0.\n" % INERT_EPS)

    faith = {}
    for part in ("overall", "train", "heldout"):
        groups = per_group.get(part, [])
        if not groups:
            continue
        A(f"\n## {_PART_TITLES[part]}\n")
        _verdict_table(A, groups)
        faith[part] = sum(g["verdict"].startswith("FAITHFUL") for g in groups)

    A(f"\n**Bottom line.** FAITHFUL groups — overall: {faith.get('overall', 0)}, "
      f"train-only: {faith.get('train', 0)}, held-out-only: {faith.get('heldout', 0)} "
      f"(of {n_groups}). **If held-out ≈ train, running the verdict on training images "
      "did NOT inflate faithfulness** (the data-leak concern is empirically dead); a drop "
      "from train to held-out quantifies in-sample/memorisation inflation. The decisive "
      "column is `specificity med (95% CI)` with a significant `p_spec(FDR)` — not the "
      "bare `drop_on`. `grade-Δ%` is the fraction of images whose predicted grade changed "
      "under the on-target occlusion (the WHY↔grade coupling).\n")
    A("**Caveats.** Numbers require `--encoder biomedclip` + recovered bridges. "
      "Diffuse heatmaps (high mask%) let the off-target control overlap the real region, "
      "*understating* specificity — sharpen with `--threshold` or the proposed seg loss. "
      "DR severe/PDR cosines are negative; the test is sign-agnostic (uses the change).")
    path = os.path.join(out_dir, "verdict_report.md")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    return path


def _fmt_p(p):
    if np.isnan(p):
        return "—"
    if p < 1e-3:
        return "<.001"
    return f"{p:.3f}"


# ─────────────────────────────────────────────────────────────────────────────
# CLI / main
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Scaled, controlled, statistically-tested faithfulness verdict run."
    )
    p.add_argument("--datasets", choices=["both", "busi", "dr"], default="both")
    p.add_argument("--per-class", type=int, default=100,
                   help="Max images per BUSI class / DR grade (default 100; 0 or -1 ⇒ all).")
    p.add_argument("--controls", type=int, default=4,
                   help="Random-shift off-target controls per image (plus 2 reflections).")
    p.add_argument("--occlusion", choices=["blur", "gray"], default="blur")
    p.add_argument("--blur-sigma", type=float, default=21.0)
    p.add_argument("--no-tissue-mask", action="store_true")
    p.add_argument("--encoder", choices=["biomedclip", "random"], default="biomedclip")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--busi-fold", type=int, default=4,
                   help="BUSI checkpoint's held-out CV fold (default 4 = BUSI_best_fold4).")
    p.add_argument("--dr-fold", type=int, default=2,
                   help="DR checkpoint's held-out CV fold (default 2 = DR_best_acc_fold2).")
    p.add_argument("--out", default=os.path.join(_HERE, "faithful_verdict"))
    # roots / checkpoints / projections (auto-resolved if omitted; names match setup())
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
          f"|  controls: {args.controls + 2}/image")

    encoder = args.encoder
    text_model = tokenizer = None
    if encoder == "biomedclip":
        text_model, tokenizer = load_biomedclip(device)
        if text_model is None:
            print("Falling back to --encoder random (verdict NOT meaningful).")
            encoder = "random"
    if encoder == "random":
        print("NOTE: random encoder ⇒ concept vectors are placeholders; verdict is meaningless.")

    datasets = ["busi", "dr"] if args.datasets == "both" else [args.datasets]
    # Stats are computed three ways: over all images, train-only, held-out-only.
    PARTITIONS = ("overall", "train", "heldout")
    all_rows = []
    per_group = {p: [] for p in PARTITIONS}
    per_image_by_group = {p: {} for p in PARTITIONS}

    for d in datasets:
        bundle = setup(d, args, device, text_model, tokenizer, encoder)
        if bundle is None:
            print(f"WARNING: {d.upper()} dataset/checkpoint not found; skipping.")
            continue
        # Rebuild the checkpoint's held-out CV fold so each image is tagged
        # train vs heldout (the in-sample / memorisation control).
        fold = args.busi_fold if d == "busi" else args.dr_fold
        ckpt = args.busi_ckpt if d == "busi" else args.dr_ckpt
        _check_ckpt_fold(ckpt, fold, d)
        bundle["heldout"] = heldout_paths(d, bundle["ds"].items, fold)
        # shared, fixed control rng per dataset (reproducible off-target placements)
        ctrl_rng = random.Random(123)
        sample_rng = random.Random(args.seed)
        dataset_rows = []
        for label in bundle["labels"]:
            rows = eval_group(bundle, label, d, args, ctrl_rng, sample_rng)
            if not rows:
                continue
            all_rows.extend(rows)
            dataset_rows.extend(rows)
            key = f"{d.upper()}:{bundle['namer'](label)}"
            for part in PARTITIONS:
                sub = rows if part == "overall" else [r for r in rows if r["split"] == part]
                if not sub:
                    continue
                stats = summarize_group(sub, ci_seed=args.seed + label)
                stats["group_key"] = key
                stats["partition"] = part
                per_group[part].append(stats)
                per_image_by_group[part][key] = [r["specificity"] for r in sub]
        _heldout_self_check(d, fold, dataset_rows)
        bundle["pipeline"].remove_hooks()

    if not any(per_group.values()):
        print("No groups evaluated (check dataset roots / checkpoints).")
        return

    # BH-FDR WITHIN each partition (each is an internally consistent verdict
    # table), derive verdicts, then flatten for CSV/JSON output.
    flat = []
    for part in PARTITIONS:
        groups = per_group[part]
        if not groups:
            continue
        p_spec_adj = _bh_fdr([g["p_spec_raw"] for g in groups])
        for g, padj in zip(groups, p_spec_adj):
            g["p_spec_adj"] = float(padj)
            g["verdict"] = _verdict(g["drop_on_med"], g["spec_med"], g["p_drop"],
                                    g["p_spec_adj"], g["mask_med"])
            flat.append(g)

    # ── write per_image.csv ──────────────────────────────────────────────────
    img_fields = ["dataset", "group", "true_label", "image", "index", "split",
                  "top_concept",
                  "c_before", "c_on", "drop_on", "drop_off", "specificity", "mask_frac",
                  "reg_before", "reg_on", "grade_before", "grade_after", "grade_changed"]
    with open(os.path.join(args.out, "per_image.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=img_fields)
        w.writeheader()
        for r in all_rows:
            w.writerow({k: (f"{r[k]:.4f}" if isinstance(r[k], float) else r[k])
                        for k in img_fields})

    # ── write per_group_verdict.csv (one row per group × partition) ───────────
    grp_fields = ["group_key", "partition", "n", "verdict", "drop_on_med", "drop_on_mean",
                  "drop_off_med", "spec_med", "spec_mean", "spec_ci", "spec_iqr",
                  "mask_med", "pass_rate", "locspec_rate", "anti_rate",
                  "grade_change_rate", "p_drop", "p_spec_raw", "p_spec_adj"]
    with open(os.path.join(args.out, "per_group_verdict.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=grp_fields)
        w.writeheader()
        for g in flat:
            w.writerow({k: g.get(k) for k in grp_fields})

    with open(os.path.join(args.out, "verdict.json"), "w") as f:
        json.dump(flat, f, indent=2)

    # ── plots (overall + held-out) + report ──────────────────────────────────
    plot_specificity_box(per_group["overall"], per_image_by_group["overall"],
                         os.path.join(args.out, "verdict_specificity_box.png"))
    plot_dropon_vs_off(per_group["overall"],
                       os.path.join(args.out, "verdict_dropon_vs_off.png"))
    if per_group["heldout"]:
        plot_specificity_box(per_group["heldout"], per_image_by_group["heldout"],
                             os.path.join(args.out, "verdict_specificity_box_heldout.png"))
        plot_dropon_vs_off(per_group["heldout"],
                           os.path.join(args.out, "verdict_dropon_vs_off_heldout.png"))
    write_report(per_group, args, args.out, n_images=len(all_rows))
    print(f"\nWrote verdict artifacts to {args.out}")


if __name__ == "__main__":
    main()
