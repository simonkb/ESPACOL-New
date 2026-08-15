"""
Causal faithfulness evaluation for OPTIC-C's GradePrototypeAttention (GPA).

Read-only: loads a trained checkpoint, evaluates one fold, writes CSV/JSON/PNG
into --out_dir. No training, no mutation of any existing module.

------------------------------------------------------------------------------
WHAT IS ACTUALLY INTERVENED ON, AND WHY
------------------------------------------------------------------------------
GPA is a *side branch*. In OPTICConceptModel.forward the prediction is:

    features, ctot_feats, raw_tiles = backbone.forward_tiles(x)
    grade_features, tile_evidence, tile_weights = gpa(ctot_feats)   # <- side branch
    pred = sigmoid(ordinal_head(features)).sum(1)                   # <- uses `features`

`features` is CTOT's [GRADE] token (position 0) projected back to 1280-d.
GPA reads ctot_feats[:, 1:, :] and its outputs (grade_features / tile_evidence /
tile_weights) feed nothing downstream of `pred`.

Consequence: zeroing a GPA tile weight and renormalising over the remaining
tiles CANNOT change `pred` — delta_pred would be identically 0 for every tile,
and every correlation below would be a measurement of floating-point noise.

So the causal intervention used here is to remove tile t from CTOT's
self-attention via `src_key_padding_mask`, then re-run CTOT and the heads:

    tile t masked -> [GRADE] token can no longer attend to it
                  -> `features` changes -> `pred` changes -> delta_pred[t]

Masking (rather than deleting the token) is deliberate: CTOT adds a learned
positional embedding per sequence slot, so dropping a token would shift every
later tile onto the wrong positional embedding and confound the ablation.

The faithfulness question this answers is therefore the meaningful one:
    "Do the tiles GPA claims drive grade k match the tiles that actually move
     the model's prediction?"

------------------------------------------------------------------------------
WHY THE BACKBONE IS NOT RE-RUN
------------------------------------------------------------------------------
TiledEfficientNetBackbone._encode_tiles applies the CNN to tiles independently
(flattened to N*T), so per-tile embeddings do not depend on which other tiles
are present. They are computed once per image and reused for every ablation.
Only CTOT (2 layers, d=512, sequence length 11) and the heads re-run — a
rounding error next to the CNN. Runtime is ~1 inference pass over the fold.

------------------------------------------------------------------------------
CORRECTNESS GUARD
------------------------------------------------------------------------------
This script re-implements CTOT's forward pass so it can inject an attention
mask (the module's own forward takes no mask argument, and editing it is out of
scope). On the first batch it asserts that the re-implementation, run with no
mask, reproduces the model's own `features` and `pred`. If OPTIC-C's CTOT ever
changes, this fails loudly instead of reporting quietly wrong numbers.

Usage:
    python eval_gpa_faithfulness.py \
        --ckpt runs/optic_concept_fold0_v4/fold0/fold0_best.pth \
        --dr_root Datasets/DR \
        --folds 0 \
        --out_dir runs/optic_concept_fold0_v4/gpa_faithfulness
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import sys

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from configs.config import DRConfig
from Datasets.dataloaders import DRDataset, ImageLabelDataset, build_tile_transform
from models.framework import build_model
from training.cross_val import DRCrossValidator
from utils.metrics import round_predictions


# ─────────────────────────────────────────────────────────────────────────────
# Architecture detection
# ─────────────────────────────────────────────────────────────────────────────

def detect_architecture(sd: dict) -> dict:
    """
    Infer build_model(...) kwargs from a checkpoint's state_dict.

    Everything except `nhead` is recoverable from tensor shapes; nhead is not
    (nn.MultiheadAttention packs in_proj as 3d x d regardless of head count),
    so it is taken from --tile_transformer_nhead, defaulting to the config value.
    """
    has_ctot = any(k.startswith("backbone.pool.transformer.") for k in sd)
    has_attnpool = "backbone.pool.query" in sd
    has_gpa = "gpa.prototypes" in sd
    has_ordinal = any(k.startswith("ordinal_head.") for k in sd)
    has_concept = any(k.startswith("concept_module.") for k in sd)

    if not has_ctot:
        if has_attnpool:
            raise SystemExit(
                "This checkpoint uses AttentionPool, not CTOT+GPA. There is no GPA "
                "tile attention to evaluate — point --ckpt at an OPTIC run built with "
                "--use_tile_transformer --use_grade_prototypes."
            )
        raise SystemExit(
            "Checkpoint has no tiled CTOT backbone (`backbone.pool.transformer.*` "
            "keys absent). This script only evaluates multi-tile OPTIC models."
        )
    if not has_gpa:
        raise SystemExit(
            "Checkpoint has no GPA module (`gpa.prototypes` absent). Nothing to "
            "evaluate — rerun training with --use_grade_prototypes."
        )

    # (1, T+1, d_model)
    pos = sd["backbone.pool.pos_embed"]
    seq_len, d_model = pos.shape[1], pos.shape[2]
    n_tiles = seq_len - 1                      # T (local + global)
    n_local = n_tiles - 1                      # last tile is the global view
    tile_grid = int(round(math.sqrt(n_local)))
    if tile_grid * tile_grid != n_local:
        raise SystemExit(
            f"Cannot infer tile_grid: {n_tiles} tiles implies {n_local} local tiles, "
            "which is not a perfect square."
        )

    n_layers = len({
        k.split(".")[3] for k in sd if k.startswith("backbone.pool.transformer.layers.")
    })

    n_classes = sd["gpa.prototypes"].shape[0]

    n_concepts = 9
    if has_concept and "concept_module.grade_concept_targets" in sd:
        n_concepts = sd["concept_module.grade_concept_targets"].shape[1]

    proj_out_dim = 128
    if "pcol_head.net.3.weight" in sd:
        proj_out_dim = sd["pcol_head.net.3.weight"].shape[0]
    proj_hidden_dim = 1280
    if "pcol_head.net.0.weight" in sd:
        proj_hidden_dim = sd["pcol_head.net.0.weight"].shape[0]

    return {
        "n_classes": n_classes,
        "proj_hidden_dim": proj_hidden_dim,
        "proj_out_dim": proj_out_dim,
        "use_image_text": any(k.startswith("image_text_head.") for k in sd),
        "use_multi_tile": True,
        "tile_grid": tile_grid,
        "use_tile_transformer": True,
        "tile_transformer_dim": d_model,
        "tile_transformer_layers": n_layers,
        "use_grade_prototypes": True,
        "use_ordinal_head": has_ordinal,
        "use_concept_prototype": has_concept,
        "n_concepts": n_concepts,
        "_n_tiles": n_tiles,
        "_n_local": n_local,
    }


# ─────────────────────────────────────────────────────────────────────────────
# CTOT forward with an attention mask  (mirrors CrossTileOrdinalTransformer)
# ─────────────────────────────────────────────────────────────────────────────

def ctot_forward(pool, tile_feats: torch.Tensor, tile_mask: torch.Tensor | None):
    """
    tile_feats : (N, T, input_dim)  raw per-tile backbone embeddings
    tile_mask  : (N, T) bool, True = tile is REMOVED (masked out), or None

    Returns (N, input_dim) aggregated [GRADE] features — same as
    CrossTileOrdinalTransformer.forward(tile_feats) when tile_mask is None.
    """
    N, T, _ = tile_feats.shape
    h = pool.input_proj(tile_feats)
    grade_tok = pool.grade_token.expand(N, 1, pool.d_model)
    h = torch.cat([grade_tok, h], dim=1)                 # (N, T+1, d)
    h = h + pool.pos_embed[:, : T + 1, :]

    key_padding = None
    if tile_mask is not None:
        # [GRADE] token at position 0 is never masked
        grade_col = torch.zeros(N, 1, dtype=torch.bool, device=tile_mask.device)
        key_padding = torch.cat([grade_col, tile_mask], dim=1)   # (N, T+1)

    h = pool.transformer(h, src_key_padding_mask=key_padding)
    h = pool.norm(h)
    return pool.output_proj(h[:, 0, :])


def head_pred(model, features: torch.Tensor) -> torch.Tensor:
    """Replicates the `pred` computation in OPTIC(Concept)Model.forward."""
    if getattr(model, "ordinal_head", None) is not None:
        return torch.sigmoid(model.ordinal_head(features)).sum(dim=1)
    return model.regression_head(features)


# ─────────────────────────────────────────────────────────────────────────────
# Statistics (no scipy dependency — HPC envs vary)
# ─────────────────────────────────────────────────────────────────────────────

def _rankdata(a: np.ndarray) -> np.ndarray:
    """Average ranks, matching scipy.stats.rankdata's 'average' tie policy."""
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty(len(a), dtype=np.float64)
    ranks[order] = np.arange(1, len(a) + 1, dtype=np.float64)
    # average ties
    sorted_a = a[order]
    i = 0
    while i < len(a):
        j = i
        while j + 1 < len(a) and sorted_a[j + 1] == sorted_a[i]:
            j += 1
        if j > i:
            ranks[order[i : j + 1]] = ranks[order[i : j + 1]].mean()
        i = j + 1
    return ranks


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    """Spearman rho. Returns nan when either input is constant (rho undefined)."""
    if len(x) < 2:
        return float("nan")
    rx, ry = _rankdata(x), _rankdata(y)
    rx = rx - rx.mean()
    ry = ry - ry.mean()
    denom = math.sqrt(float((rx ** 2).sum()) * float((ry ** 2).sum()))
    if denom < 1e-12:
        return float("nan")
    return float((rx * ry).sum() / denom)


def nanmean(values) -> float:
    vals = [v for v in values if v is not None and not math.isnan(v)]
    return float(np.mean(vals)) if vals else float("nan")


def auc_normalised(ys: list[float]) -> float:
    """Trapezoidal AUC over evenly spaced x, normalised to the x-range."""
    if len(ys) < 2:
        return float("nan")
    return float(np.trapz(np.asarray(ys, dtype=np.float64)) / (len(ys) - 1))


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(
        description="Causal faithfulness evaluation of OPTIC-C GPA tile attention"
    )
    p.add_argument("--ckpt", type=str, required=True,
                   help="Path to fold checkpoint, e.g. runs/<run>/fold0/fold0_best.pth")
    p.add_argument("--dr_root", type=str, default="Datasets/DR",
                   help="DR dataset root containing train/ and trainLabels.csv")
    p.add_argument("--train_csv", type=str, default=None,
                   help="Label CSV. Default: <dr_root>/trainLabels.csv")
    p.add_argument("--folds", type=str, default="0",
                   help="Fold index whose val split is evaluated (single fold, e.g. '0')")
    p.add_argument("--out_dir", type=str, default=None,
                   help="Output directory. Default: <ckpt dir>/gpa_faithfulness")
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--max_images", type=int, default=None,
                   help="Cap images for a smoke test (default: whole val split)")
    p.add_argument("--n_random_orders", type=int, default=3,
                   help="Random deletion orders averaged for the baseline curve")
    p.add_argument("--tile_transformer_nhead", type=int, default=8,
                   help="CTOT attention heads — NOT recoverable from the state dict; "
                        "must match training (config default 8)")
    p.add_argument("--seed", type=int, default=42,
                   help="Must match the training seed so the CV split is identical")
    p.add_argument("--device", type=str, default=None)
    args = p.parse_args()

    if args.train_csv is None:
        args.train_csv = os.path.join(args.dr_root, "trainLabels.csv")
    if args.out_dir is None:
        args.out_dir = os.path.join(os.path.dirname(args.ckpt), "gpa_faithfulness")
    os.makedirs(args.out_dir, exist_ok=True)

    fold_ids = [int(f.strip()) for f in args.folds.split(",") if f.strip()]
    if len(fold_ids) != 1:
        raise SystemExit("--folds takes exactly one fold index (the checkpoint is per-fold).")
    fold = fold_ids[0]

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device(
        args.device if args.device else
        ("cuda" if torch.cuda.is_available() else "cpu")
    )

    # ── Load checkpoint and rebuild the architecture ──────────────────────────
    print(f"Loading checkpoint: {args.ckpt}")
    try:
        state = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    except TypeError:                      # torch < 2.0 has no weights_only kwarg
        state = torch.load(args.ckpt, map_location="cpu")

    sd = state["model_state"] if "model_state" in state else state
    arch = detect_architecture(sd)
    n_tiles, n_local = arch.pop("_n_tiles"), arch.pop("_n_local")
    global_idx = n_tiles - 1

    print("Detected architecture:")
    for k, v in arch.items():
        print(f"    {k:<26} {v}")
    print(f"    {'tiles':<26} {n_tiles} ({n_local} local + 1 global, "
          f"global index {global_idx})")
    if "epoch" in state:
        print(f"    {'checkpoint epoch':<26} {state['epoch']}")
    if isinstance(state.get("metrics"), dict):
        print(f"    {'checkpoint metrics':<26} {state['metrics']}")

    model = build_model(
        pretrained=False,
        tile_transformer_nhead=args.tile_transformer_nhead,
        **arch,
    )
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing or unexpected:
        raise SystemExit(
            f"State dict mismatch — architecture detection is wrong.\n"
            f"  missing:    {sorted(missing)[:10]}\n"
            f"  unexpected: {sorted(unexpected)[:10]}"
        )
    model.to(device).eval()

    n_classes = arch["n_classes"]
    tile_grid = arch["tile_grid"]

    # ── Fold-0 val split — mirrors train_dr.py exactly (val == test fold) ─────
    cfg = DRConfig()
    items = DRDataset(root_dir=args.dr_root, split="train", csv_path=args.train_csv).items
    cv = DRCrossValidator(
        items, n_folds=cfg.n_folds, val_fraction=cfg.val_fraction, seed=args.seed
    )
    _train_raw, _val_held_out, test_items = cv.get_fold(fold)
    val_items = test_items          # train_dr.py: `val_items = test_items`
    if args.max_images:
        val_items = val_items[: args.max_images]
    print(f"Fold {fold} val split: {len(val_items)} images")

    tfm = build_tile_transform(tile_size=cfg.img_size, tile_grid=tile_grid, augment=False)
    loader = DataLoader(
        ImageLabelDataset(val_items, transform=tfm),
        batch_size=args.batch_size,
        shuffle=False,                       # keeps row order aligned with val_items
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    pool = model.backbone.pool
    gen = torch.Generator().manual_seed(args.seed)

    rows: list[dict] = []
    # deletion curves: accuracy at k = 0..n_local removed
    del_correct_attn = np.zeros(n_local + 1, dtype=np.float64)
    del_correct_rand = np.zeros(n_local + 1, dtype=np.float64)   # float: averaged over orders
    n_seen = 0
    checked = False

    print("Evaluating (backbone runs once per image; CTOT re-runs per ablation)...")
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            N = x.shape[0]

            # One backbone pass — tile embeddings cached for every ablation below
            feats_full, ctot_feats, raw_tiles = model.backbone.forward_tiles(x)
            _gf, _te, tile_weights = model.gpa(ctot_feats)     # (N, K, T)
            pred_full = head_pred(model, feats_full)            # (N,)

            # ── Guard: our CTOT re-implementation must match the module's own ──
            if not checked:
                feats_check = ctot_forward(pool, raw_tiles, None)
                if not torch.allclose(feats_check, feats_full, atol=1e-4, rtol=1e-3):
                    raise SystemExit(
                        "CTOT re-implementation does not reproduce the model's own "
                        "forward pass (max abs diff "
                        f"{(feats_check - feats_full).abs().max().item():.3e}). "
                        "CrossTileOrdinalTransformer.forward has changed — update "
                        "ctot_forward() in this script before trusting any results."
                    )
                print("  [guard] CTOT re-implementation matches model forward ✓")
                checked = True

            pred_grade = round_predictions(pred_full.cpu(), n_classes).numpy()
            labels_np = y.cpu().numpy()

            # a[t] for the predicted grade (matches inference/explain.py's heatmap)
            idx = torch.arange(N, device=device)
            a_full = tile_weights[idx, torch.as_tensor(pred_grade, device=device), :]  # (N, T)
            a_np = a_full.cpu().numpy()

            # ── Single-tile ablations ─────────────────────────────────────────
            deltas = np.zeros((N, n_tiles), dtype=np.float64)
            for t in range(n_tiles):
                mask = torch.zeros(N, n_tiles, dtype=torch.bool, device=device)
                mask[:, t] = True
                feats_t = ctot_forward(pool, raw_tiles, mask)
                deltas[:, t] = (head_pred(model, feats_t) - pred_full).cpu().numpy()

            # ── Deletion curves ───────────────────────────────────────────────
            a_local = a_full[:, :n_local]
            attn_order = torch.argsort(a_local, dim=1, descending=True)   # (N, n_local)

            correct0 = (pred_grade == labels_np)
            del_correct_attn[0] += int(correct0.sum())
            del_correct_rand[0] += int(correct0.sum())

            for k in range(1, n_local + 1):
                mask = torch.zeros(N, n_tiles, dtype=torch.bool, device=device)
                mask.scatter_(1, attn_order[:, :k], True)
                pk = head_pred(model, ctot_forward(pool, raw_tiles, mask))
                gk = round_predictions(pk.cpu(), n_classes).numpy()
                del_correct_attn[k] += int((gk == labels_np).sum())

                acc_r = 0
                for _ in range(args.n_random_orders):
                    perm = torch.argsort(torch.rand(N, n_local, generator=gen), dim=1)
                    rmask = torch.zeros(N, n_tiles, dtype=torch.bool)
                    rmask.scatter_(1, perm[:, :k], True)
                    pr = head_pred(model, ctot_forward(pool, raw_tiles, rmask.to(device)))
                    gr = round_predictions(pr.cpu(), n_classes).numpy()
                    acc_r += int((gr == labels_np).sum())
                del_correct_rand[k] += acc_r / args.n_random_orders

            # ── Per-image records ─────────────────────────────────────────────
            for i in range(N):
                al = a_np[i, :n_local]
                dl = np.abs(deltas[i, :n_local])
                rows.append({
                    "index": n_seen + i,
                    "path": val_items[n_seen + i][0],
                    "label": int(labels_np[i]),
                    "pred_continuous": float(pred_full[i].item()),
                    "pred_grade": int(pred_grade[i]),
                    "spearman_a_vs_absdelta": spearman(al, dl),
                    "top1_match": bool(int(np.argmax(al)) == int(np.argmax(dl))),
                    "a_argmax": int(np.argmax(al)),
                    "delta_argmax": int(np.argmax(dl)),
                    "a_global": float(a_np[i, global_idx]),
                    "delta_global": float(deltas[i, global_idx]),
                    "abs_delta_local_mean": float(dl.mean()),
                    **{f"a_{t}": float(a_np[i, t]) for t in range(n_local)},
                    **{f"delta_{t}": float(deltas[i, t]) for t in range(n_local)},
                })

            n_seen += N
            if n_seen % (args.batch_size * 10) < args.batch_size:
                print(f"  {n_seen}/{len(val_items)} images")

    if not rows:
        raise SystemExit("No images evaluated — check --dr_root and the fold split.")

    # ── Aggregate ─────────────────────────────────────────────────────────────
    total = len(rows)
    acc_attn = (del_correct_attn / total * 100.0).tolist()
    acc_rand = (del_correct_rand / total * 100.0).tolist()
    auc_attn, auc_rand = auc_normalised(acc_attn), auc_normalised(acc_rand)

    def summarise(subset: list[dict]) -> dict:
        if not subset:
            return {"n": 0}
        return {
            "n": len(subset),
            "spearman_mean": nanmean([r["spearman_a_vs_absdelta"] for r in subset]),
            "top1_match_frac": float(np.mean([r["top1_match"] for r in subset])),
            "abs_delta_local_mean": float(np.mean([r["abs_delta_local_mean"] for r in subset])),
            "abs_delta_global_mean": float(np.mean([abs(r["delta_global"]) for r in subset])),
            "a_global_mean": float(np.mean([r["a_global"] for r in subset])),
        }

    summary = {
        "checkpoint": os.path.abspath(args.ckpt),
        "fold": fold,
        "n_images": total,
        "n_local_tiles": n_local,
        "global_tile_index": global_idx,
        "intervention": "mask tile from CTOT self-attention (src_key_padding_mask), "
                        "re-run CTOT + heads from cached backbone tile embeddings",
        "attention_reference": "tile_weights[:, predicted_grade, :]",
        "overall": summarise(rows),
        "per_grade": {
            str(g): summarise([r for r in rows if r["label"] == g])
            for g in range(n_classes)
        },
        "deletion_curve": {
            "n_removed": list(range(n_local + 1)),
            "acc_attention_order": acc_attn,
            "acc_random_order": acc_rand,
            "auc_attention_order": auc_attn,
            "auc_random_order": auc_rand,
            "auc_gap_random_minus_attention": auc_rand - auc_attn,
            "n_random_orders": args.n_random_orders,
        },
    }

    # ── Write outputs ─────────────────────────────────────────────────────────
    per_image_csv = os.path.join(args.out_dir, f"per_image_fold{fold}.csv")
    with open(per_image_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    curve_csv = os.path.join(args.out_dir, f"deletion_curve_fold{fold}.csv")
    with open(curve_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["n_removed", "acc_attention_order", "acc_random_order"])
        for k in range(n_local + 1):
            w.writerow([k, f"{acc_attn[k]:.4f}", f"{acc_rand[k]:.4f}"])

    summary_json = os.path.join(args.out_dir, f"summary_fold{fold}.json")
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    png_path = os.path.join(args.out_dir, f"deletion_curve_fold{fold}.png")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        ks = list(range(n_local + 1))
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(ks, acc_attn, marker="o", label=f"attention order (AUC {auc_attn:.2f})")
        ax.plot(ks, acc_rand, marker="s", linestyle="--",
                label=f"random order (AUC {auc_rand:.2f})")
        ax.set_xlabel("tiles removed")
        ax.set_ylabel("accuracy (%)")
        ax.set_title(f"GPA deletion curve — fold {fold}\n"
                     f"AUC gap (random − attention) = {auc_rand - auc_attn:.2f}")
        ax.legend()
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(png_path, dpi=150)
        plt.close(fig)
    except Exception as e:                       # matplotlib optional on HPC
        png_path = f"(not written: {e})"

    # ── Report ────────────────────────────────────────────────────────────────
    ov = summary["overall"]
    print("\n" + "=" * 72)
    print(f"GPA CAUSAL FAITHFULNESS — fold {fold}, {total} images")
    print("=" * 72)
    print(f"  Spearman(a[t], |delta_pred[t]|)  : {ov['spearman_mean']:+.4f}  "
          f"(per-image mean over {n_local} local tiles)")
    print(f"  Top-attention == top-|delta|     : {ov['top1_match_frac'] * 100:.2f}%  "
          f"(chance ≈ {100.0 / n_local:.2f}%)")
    print(f"  Mean |delta_pred| local tile     : {ov['abs_delta_local_mean']:.5f}")
    print(f"  Mean |delta_pred| GLOBAL tile    : {ov['abs_delta_global_mean']:.5f}  "
          f"(a_global mean {ov['a_global_mean']:.4f})")
    print()
    print("  Per grade:")
    print(f"    {'grade':<7}{'n':>7}{'spearman':>12}{'top1 match':>13}{'mean|Δ|':>11}")
    for g in range(n_classes):
        s = summary["per_grade"][str(g)]
        if s["n"] == 0:
            print(f"    {g:<7}{0:>7}{'—':>12}{'—':>13}{'—':>11}")
            continue
        print(f"    {g:<7}{s['n']:>7}{s['spearman_mean']:>+12.4f}"
              f"{s['top1_match_frac'] * 100:>12.2f}%{s['abs_delta_local_mean']:>11.5f}")
    print()
    print("  Deletion curve (accuracy %, tiles removed 0..n):")
    print("    k      : " + " ".join(f"{k:>6}" for k in range(n_local + 1)))
    print("    attn   : " + " ".join(f"{v:>6.2f}" for v in acc_attn))
    print("    random : " + " ".join(f"{v:>6.2f}" for v in acc_rand))
    print(f"    AUC attention={auc_attn:.3f}  random={auc_rand:.3f}  "
          f"gap={auc_rand - auc_attn:+.3f}")
    print("    (positive gap ⇒ attention-ranked tiles degrade accuracy faster ⇒ faithful)")
    print()
    print(f"  per-image CSV : {per_image_csv}")
    print(f"  curve CSV     : {curve_csv}")
    print(f"  summary JSON  : {summary_json}")
    print(f"  plot          : {png_path}")


if __name__ == "__main__":
    main()
