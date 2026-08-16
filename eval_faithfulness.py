"""
Post-hoc faithfulness measurement for a trained concept-spine checkpoint.

Read-only: loads a checkpoint, evaluates one fold's val split, writes a CSV and
prints a summary. No training, no mutation of any existing module.

------------------------------------------------------------------------------
WHY THIS EXISTS
------------------------------------------------------------------------------
The trainer's faithfulness block is gated on `nu > 0` (trainer.py: nu_peak > 0
in do_faith_this_epoch). A run trained with --nu 0 therefore never enters it,
so its logged drop / dir / spec / occ / mask are structurally 0.0 — absence of
measurement, not absence of the effect. Comparing those logged numbers between
a faith_on and a faith_off run would be meaningless.

This script applies the *same* measurement to any checkpoint after the fact, so
both arms are measured identically:

    runs/faith_on/fold0/fold0_best.pth
    runs/faith_off/fold0/fold0_best.pth

Run it twice with identical flags; the only thing that differs is --ckpt.

------------------------------------------------------------------------------
WHAT IT COMPUTES  (extracted from the faithfulness block in training/trainer.py)
------------------------------------------------------------------------------
Per image, for a given concept index k:
  1. Forward -> concept activations c (N,M), evidence E (N,), weights w (M,)
  2. Grad-CAM++ heatmap for concept k, via the same LayerCAM hooks and the same
     cam_score = <z_c, concept_text_emb[k]> objective the trainer backprops
  3. soft_occlude with the HARD threshold mask (heatmap > faith_threshold)
  4. Re-forward the occluded image -> c', E'
  5. FaithfulnessLoss -> L_drop, L_dir, L_spec  (the module itself, not a copy)
  plus the two diagnostics: occ_effect = mean|c' - c|, mask_frac = mean(mask)

------------------------------------------------------------------------------
PERMUTATION CONTROL
------------------------------------------------------------------------------
The whole procedure runs TWICE per image:

  REAL    k* = argmax |w * c|            — the dominant concept
  CONTROL k~ = a randomly chosen concept != k*

The control is the null hypothesis for the headline comparison. If faith_on
shows a lower L_drop than faith_off, the obvious objection is that occluding any
salient region depresses any concept's activation — that the effect is about
occlusion, not about the concept being the one the model actually relied on.
The control answers it: k~ gets its own Grad-CAM, its own occlusion, its own
L_drop. The claim to defend is not "faith_on has lower real drop" but

    (control_drop - real_drop) is larger in faith_on than in faith_off

i.e. the specificity gap widens. That gap is reported as `drop_gap`, per grade
and overall, and is the number to quote.

The control concept is drawn deterministically from (seed, image index, k*):

    offset = rng(seed, image_index).integers(1, M)
    k~     = (k* + offset) % M

so it is reproducible, never equals k*, and — critically — is the SAME concept
for the same image across both checkpoints whenever k* agrees. Where the two
checkpoints disagree on k*, the control shifts with it; both columns are in the
CSV so any such image can be audited.

Cost: the control doubles the Grad-CAM backwards and occluded forwards, so
expect roughly 2x the runtime of the real-concept measurement alone.

------------------------------------------------------------------------------
DIFFERENCES FROM THE TRAINING-TIME COMPUTATION  (both deliberate)
------------------------------------------------------------------------------
  - model.eval() — BatchNorm uses running statistics, so a per-image value does
    not depend on which other images share its batch. Training-time numbers are
    computed in train() mode.
  - float32 throughout, no autocast. Removes AMP nondeterminism from a
    measurement whose entire purpose is a small between-checkpoint difference.

Per-image components come from calling FaithfulnessLoss on one-image slices
(it reduces with .mean() internally, so a batched call would return batch means
and the CSV could not be per-image). Forwards stay batched; only the loss call
is per-image, and it is pure tensor arithmetic.

Grades are grouped by TRUE label, not predicted grade — the grouping must be
identical across the two checkpoints for the comparison to be meaningful.

Usage:
    # preflight: verify a checkpoint will load, without touching the dataset
    python eval_faithfulness.py --ckpt runs/faith_off/fold0/fold0_best.pth --inspect

    python eval_faithfulness.py --ckpt runs/faith_on/fold0/fold0_best.pth \
        --dr_root Datasets/DR --folds 0
    python eval_faithfulness.py --ckpt runs/faith_off/fold0/fold0_best.pth \
        --dr_root Datasets/DR --folds 0
"""

from __future__ import annotations

import argparse
import csv
import os
import random
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from configs.config import DRConfig
from configs.clinical_text import DR_CONCEPTS
from Datasets.dataloaders import (
    DRDataset,
    ImageLabelDataset,
    build_transform,
    build_tile_transform,
)
from explainability import LayerCAM
from losses.faithfulness import FaithfulnessLoss, soft_occlude
from models.framework import build_model
from models.text import ClinicalTextEncoder
from training.cross_val import DRCrossValidator
from training.trainer import _compute_batch_concept_cams


# ─────────────────────────────────────────────────────────────────────────────
# Checkpoint inspection / architecture detection
# ─────────────────────────────────────────────────────────────────────────────

def inspect_checkpoint(state: dict, sd: dict) -> bool:
    """Print what the checkpoint contains. Returns True if this script can load it."""
    has_spine = any(k.startswith("concept_spine.") for k in sd)
    has_text = "text_encoder_state" in state
    has_it_head = any(k.startswith("image_text_head.") for k in sd)
    n_concepts = int(sd["concept_spine.raw_w"].shape[0]) if "concept_spine.raw_w" in sd else None

    def mark(ok: bool) -> str:
        return "OK  " if ok else "FAIL"

    print("  top-level keys      :", sorted(state.keys()))
    print(f"  [{mark(has_spine)}] concept_spine.*     "
          f"{sum(1 for k in sd if k.startswith('concept_spine.'))} keys")
    print(f"  [{mark(has_text)}] text_encoder_state  "
          f"{len(state['text_encoder_state']) if has_text else 0} keys")
    print(f"  [{mark(has_it_head)}] image_text_head.*   present={has_it_head}")
    print(f"  ---- multi_tile         : {any(k.startswith('backbone.pool.') for k in sd)}")
    print(f"  ---- n_concepts         : {n_concepts} (DR_CONCEPTS defines {len(DR_CONCEPTS)})")
    print(f"  ---- epoch              : {state.get('epoch')}")
    if isinstance(state.get("metrics"), dict):
        m = state["metrics"]
        keep = {k: v for k, v in m.items() if k.startswith(("val_", "test_"))}
        print(f"  ---- metrics            : {keep or m}")

    ok = has_spine and has_text and has_it_head
    if ok and n_concepts is not None and n_concepts != len(DR_CONCEPTS):
        print(f"  [FAIL] concept count {n_concepts} != len(DR_CONCEPTS) {len(DR_CONCEPTS)}")
        ok = False
    print(f"\n  => {'LOADABLE' if ok else 'NOT LOADABLE'} by eval_faithfulness.py")
    return ok


def detect_architecture(sd: dict) -> dict:
    use_concept_spine = any(k.startswith("concept_spine.") for k in sd)
    if not use_concept_spine:
        raise SystemExit(
            "Checkpoint has no concept spine (`concept_spine.*` keys absent), so it "
            "has no concept activations to occlude. Faithfulness is undefined here — "
            "point --ckpt at a run trained with --use_concept_spine."
        )

    arch = {
        "use_multi_tile": any(k.startswith("backbone.pool.") for k in sd),
        "use_concept_spine": True,
        "use_image_text": any(k.startswith("image_text_head.") for k in sd),
        "n_concepts": int(sd["concept_spine.raw_w"].shape[0]),
    }
    if "pcol_head.net.0.weight" in sd:
        arch["proj_hidden_dim"] = int(sd["pcol_head.net.0.weight"].shape[0])
    if "pcol_head.net.3.weight" in sd:
        arch["proj_out_dim"] = int(sd["pcol_head.net.3.weight"].shape[0])
    return arch


# ─────────────────────────────────────────────────────────────────────────────
# Permutation control
# ─────────────────────────────────────────────────────────────────────────────

def control_concept(top_k: int, n_concepts: int, image_index: int, seed: int) -> int:
    """
    Deterministic non-dominant concept for the null condition.

    Keyed on (seed, image_index) rather than on batch position, so the draw is
    independent of --batch_size and identical across checkpoints whenever they
    agree on top_k. The offset in [1, M-1] guarantees k~ != k*.
    """
    rng = np.random.default_rng(seed * 1_000_003 + image_index)
    offset = int(rng.integers(1, n_concepts))
    return (top_k + offset) % n_concepts


# ─────────────────────────────────────────────────────────────────────────────
# One measurement pass for a given per-image concept index
# ─────────────────────────────────────────────────────────────────────────────

def measure_for_concept(
    model, layer_cam, faith_loss_fn,
    x, concept_idx, concept_text_emb,
    c, E, w, cfg, faith_threshold, device,
) -> list[dict]:
    """
    Grad-CAM for concept_idx -> hard-mask occlusion -> re-forward -> components.

    concept_idx: (N,) long — which concept to attribute and occlude per image.
    Returns one dict of scalars per image in the batch.
    """
    N = x.shape[0]

    # Grad-CAM++ — the only step that needs gradients
    model.zero_grad(set_to_none=True)
    with torch.enable_grad():
        out_cam = model(x.detach(), concept_text_emb=concept_text_emb)
        cam_score = (
            (out_cam["z_c"] * concept_text_emb[concept_idx]).sum(dim=1)
        ).float().sum()
        cam_score.backward()

    is_tiled = x.dim() == 5
    H, W = (x.shape[3], x.shape[4]) if is_tiled else (x.shape[2], x.shape[3])
    heatmaps = _compute_batch_concept_cams(
        layer_cam._acts, layer_cam._grads, layer_cam.layer_indices, H, W, device,
    )                                            # (N*T, H, W) tiled, else (N, H, W)
    model.zero_grad(set_to_none=True)

    with torch.no_grad():
        if is_tiled:
            N_b, T_b, C_b, H_b, W_b = x.shape
            x_occ = soft_occlude(
                x.view(N_b * T_b, C_b, H_b, W_b), heatmaps,
                sigma=cfg.faith_sigma,
                kernel_size=cfg.faith_blur_kernel,
                threshold=faith_threshold,
            ).view(N_b, T_b, C_b, H_b, W_b)
        else:
            x_occ = soft_occlude(
                x, heatmaps,
                sigma=cfg.faith_sigma,
                kernel_size=cfg.faith_blur_kernel,
                threshold=faith_threshold,
            )
        out_occ = model(x_occ, concept_text_emb=concept_text_emb)

    c_prime = out_occ["c"].detach()
    E_prime = out_occ["E"].detach()
    T = heatmaps.shape[0] // N if is_tiled else 1

    results = []
    for i in range(N):
        with torch.no_grad():
            _lf, comps = faith_loss_fn(
                c=c[i : i + 1],
                c_prime=c_prime[i : i + 1],
                E=E[i : i + 1],
                E_prime=E_prime[i : i + 1],
                w=w,
                top_k_idx=concept_idx[i : i + 1],
            )
            occ_effect = (c_prime[i] - c[i]).abs().mean().item()
            hm_i = heatmaps[i * T : (i + 1) * T] if is_tiled else heatmaps[i]
            mask_frac = (hm_i > faith_threshold).float().mean().item()

        results.append({
            "drop": comps["loss_drop"],
            "dir": comps["loss_dir"],
            "spec": comps["loss_spec"],
            "faith": comps["loss_faith"],
            "occ_effect": occ_effect,
            "mask_frac": mask_frac,
        })
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Summary helpers
# ─────────────────────────────────────────────────────────────────────────────

_METRICS = ["drop", "dir", "spec", "faith", "occ_effect", "mask_frac"]


def summarise(rows: list[dict]) -> dict:
    if not rows:
        return {"n": 0}
    out = {"n": len(rows)}
    for cond in ("real", "ctrl"):
        for m in _METRICS:
            out[f"{cond}_{m}"] = float(np.mean([r[f"{cond}_{m}"] for r in rows]))
    # Specificity gap: how much more the dominant concept is suppressed than a
    # random one. L_drop is a penalty, so control - real; positive = specific.
    out["drop_gap"] = out["ctrl_drop"] - out["real_drop"]
    out["occ_gap"] = out["real_occ_effect"] - out["ctrl_occ_effect"]
    return out


def main() -> None:
    p = argparse.ArgumentParser(
        description="Post-hoc faithfulness measurement of a concept-spine checkpoint"
    )
    p.add_argument("--ckpt", type=str, required=True,
                   help="Checkpoint, e.g. runs/faith_on/fold0/fold0_best.pth")
    p.add_argument("--dr_root", type=str, default="Datasets/DR",
                   help="DR dataset root containing train/ and trainLabels.csv")
    p.add_argument("--folds", type=str, default="0",
                   help="Single fold index whose val split is measured")
    p.add_argument("--max_images", type=int, default=None,
                   help="Cap images (smoke test). Default: whole val split")
    p.add_argument("--inspect", action="store_true",
                   help="Print the checkpoint's key structure and whether it can be "
                        "loaded, then exit. Does not touch the dataset.")
    p.add_argument("--train_csv", type=str, default=None,
                   help="Label CSV. Default: <dr_root>/trainLabels.csv")
    p.add_argument("--out_csv", type=str, default=None,
                   help="Per-image CSV. Default: <ckpt dir>/faithfulness_eval.csv")
    p.add_argument("--batch_size", type=int, default=8,
                   help="Forward batch size (per-image values are unaffected)")
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--faith_threshold", type=float, default=None,
                   help="Occlusion mask threshold. Default: config value (0.5). "
                        "MUST match across the two checkpoints being compared.")
    p.add_argument("--seed", type=int, default=42,
                   help="Must match the training seed so the CV split — and the "
                        "control-concept draw — are identical")
    p.add_argument("--device", type=str, default=None)
    args = p.parse_args()

    # ── Load checkpoint ──────────────────────────────────────────────────────
    print(f"Loading checkpoint: {args.ckpt}")
    try:
        state = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    except TypeError:                       # older torch has no weights_only kwarg
        state = torch.load(args.ckpt, map_location="cpu")

    sd = state["model_state"] if "model_state" in state else state

    if args.inspect:
        print()
        ok = inspect_checkpoint(state, sd)
        raise SystemExit(0 if ok else 1)

    if args.train_csv is None:
        args.train_csv = os.path.join(args.dr_root, "trainLabels.csv")
    if args.out_csv is None:
        args.out_csv = os.path.join(os.path.dirname(args.ckpt), "faithfulness_eval.csv")
    os.makedirs(os.path.dirname(os.path.abspath(args.out_csv)), exist_ok=True)

    fold_ids = [int(f.strip()) for f in args.folds.split(",") if f.strip()]
    if len(fold_ids) != 1:
        raise SystemExit("--folds takes exactly one fold index (checkpoints are per-fold).")
    fold = fold_ids[0]

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device(
        args.device if args.device else
        ("cuda" if torch.cuda.is_available() else "cpu")
    )

    cfg = DRConfig()
    faith_threshold = (
        args.faith_threshold if args.faith_threshold is not None
        else getattr(cfg, "faith_threshold", 0.5)
    )

    arch = detect_architecture(sd)
    print("Detected architecture:")
    for k, v in arch.items():
        print(f"    {k:<20} {v}")
    if "epoch" in state:
        print(f"    {'epoch':<20} {state['epoch']}")

    model = build_model(n_classes=cfg.n_classes, pretrained=False, **arch)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing or unexpected:
        raise SystemExit(
            "State dict mismatch — architecture detection is wrong.\n"
            f"  missing:    {sorted(missing)[:10]}\n"
            f"  unexpected: {sorted(unexpected)[:10]}"
        )
    model.to(device).eval()

    # ── Text encoder: required for the concept embeddings ────────────────────
    if not arch["use_image_text"]:
        raise SystemExit(
            "Checkpoint has no image-text head, so the concept spine could not have "
            "been trained with a text encoder. Cannot rebuild concept embeddings."
        )

    from configs.clinical_text import DR_CLASS_DESCRIPTIONS

    text_encoder = ClinicalTextEncoder(
        model_name=cfg.text_encoder_name,
        class_descriptions=DR_CLASS_DESCRIPTIONS,
        proj_out_dim=arch.get("proj_out_dim", cfg.proj_out_dim),
        device=device,
        finetune_text_encoder=False,        # measurement only — nothing is trained
        finetune_layers=0,
    ).to(device)

    if "text_encoder_state" not in state:
        raise SystemExit(
            "Checkpoint has no `text_encoder_state`. The text projection head is "
            "trained, so without it the concept embeddings would be random and every "
            "number below would be meaningless."
        )
    t_missing, t_unexpected = text_encoder.load_state_dict(
        state["text_encoder_state"], strict=False
    )
    if t_missing or t_unexpected:
        print(f"  [warn] text encoder partial load — missing={len(t_missing)} "
              f"unexpected={len(t_unexpected)}")
    text_encoder.eval()

    concept_phrases = DR_CONCEPTS
    if len(concept_phrases) != arch["n_concepts"]:
        raise SystemExit(
            f"Concept count mismatch: checkpoint has {arch['n_concepts']} concepts, "
            f"DR_CONCEPTS defines {len(concept_phrases)}."
        )
    concept_text_emb = text_encoder.get_concept_embeddings(concept_phrases).to(device)
    M = len(concept_phrases)

    layer_cam = LayerCAM(model)
    faith_loss_fn = FaithfulnessLoss(
        margin=getattr(cfg, "faith_margin", 0.1),
        tau=getattr(cfg, "faith_tau", 0.05),
    )

    print("\nMeasurement settings (must be identical across checkpoints):")
    print(f"    faith_margin      {getattr(cfg, 'faith_margin', 0.1)}")
    print(f"    faith_tau         {getattr(cfg, 'faith_tau', 0.05)}")
    print(f"    faith_sigma       {getattr(cfg, 'faith_sigma', 7.0)}")
    print(f"    faith_blur_kernel {getattr(cfg, 'faith_blur_kernel', 21)}")
    print(f"    faith_threshold   {faith_threshold}")
    print(f"    seed              {args.seed}  (also keys the control-concept draw)")
    print(f"    mode              eval(), float32, no autocast")

    # ── Fold val split — mirrors train_dr.py (val == test fold) ──────────────
    items = DRDataset(root_dir=args.dr_root, split="train", csv_path=args.train_csv).items
    cv = DRCrossValidator(
        items, n_folds=cfg.n_folds, val_fraction=cfg.val_fraction, seed=args.seed
    )
    _train_raw, _val_held_out, test_items = cv.get_fold(fold)
    val_items = test_items
    if args.max_images:
        val_items = val_items[: args.max_images]
    print(f"\nFold {fold} val split: {len(val_items)} images")

    if arch["use_multi_tile"]:
        tfm = build_tile_transform(
            tile_size=cfg.img_size, tile_grid=getattr(cfg, "tile_grid", 3), augment=False
        )
    else:
        tfm = build_transform(cfg.img_size)

    loader = DataLoader(
        ImageLabelDataset(val_items, transform=tfm),
        batch_size=args.batch_size,
        shuffle=False,                       # keeps row order aligned with val_items
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    rows: list[dict] = []
    n_seen = 0

    print("Measuring (real concept + permutation control — 2 passes per batch)...")
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        N = x.shape[0]

        # Clean forward — concept activations, evidence, weights
        with torch.no_grad():
            out = model(x, concept_text_emb=concept_text_emb)
        c = out["c"].detach()                       # (N, M)
        E = out["E"].detach()                       # (N,)
        w = out["w"].detach()                       # (M,)

        # REAL: dominant concept by |w * c|
        top_k_idx = (w * c).abs().argmax(dim=1)     # (N,)

        # CONTROL: deterministic non-dominant concept per image
        ctrl_list = [
            control_concept(int(top_k_idx[i].item()), M, n_seen + i, args.seed)
            for i in range(N)
        ]
        ctrl_idx = torch.tensor(ctrl_list, dtype=torch.long, device=device)

        real = measure_for_concept(
            model, layer_cam, faith_loss_fn, x, top_k_idx, concept_text_emb,
            c, E, w, cfg, faith_threshold, device,
        )
        ctrl = measure_for_concept(
            model, layer_cam, faith_loss_fn, x, ctrl_idx, concept_text_emb,
            c, E, w, cfg, faith_threshold, device,
        )

        for i in range(N):
            k_real, k_ctrl = int(top_k_idx[i].item()), ctrl_list[i]
            rows.append({
                "index": n_seen + i,
                "path": val_items[n_seen + i][0],
                "label": int(y[i].item()),
                "real_concept_idx": k_real,
                "real_concept": concept_phrases[k_real],
                "ctrl_concept_idx": k_ctrl,
                "ctrl_concept": concept_phrases[k_ctrl],
                **{f"real_{m}": real[i][m] for m in _METRICS},
                **{f"ctrl_{m}": ctrl[i][m] for m in _METRICS},
                "drop_gap": ctrl[i]["drop"] - real[i]["drop"],
                "occ_gap": real[i]["occ_effect"] - ctrl[i]["occ_effect"],
            })

        n_seen += N
        if n_seen % (args.batch_size * 20) < args.batch_size:
            print(f"  {n_seen}/{len(val_items)} images")

    layer_cam.remove_hooks()

    if not rows:
        raise SystemExit("No images measured — check --dr_root and the fold split.")

    with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
        wtr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wtr.writeheader()
        wtr.writerows(rows)

    # ── Summary ──────────────────────────────────────────────────────────────
    overall = summarise(rows)
    per_grade = {
        g: summarise([r for r in rows if r["label"] == g]) for g in range(cfg.n_classes)
    }
    grades = [g for g in range(cfg.n_classes) if per_grade[g]["n"] > 0]

    print("\n" + "=" * 94)
    print(f"FAITHFULNESS — {os.path.abspath(args.ckpt)}")
    print(f"fold {fold}, {overall['n']} images, grouped by TRUE grade")
    print("=" * 94)

    h1 = (f"  {'grade':<7}{'n':>7}{'drop_real':>12}{'drop_ctrl':>12}{'drop_gap':>12}"
          f"{'occ_real':>12}{'occ_ctrl':>12}{'mask_real':>12}")
    print(h1)
    print("  " + "-" * (len(h1) - 2))
    for g in grades:
        s = per_grade[g]
        print(f"  {g:<7}{s['n']:>7}{s['real_drop']:>12.5f}{s['ctrl_drop']:>12.5f}"
              f"{s['drop_gap']:>12.5f}{s['real_occ_effect']:>12.5f}"
              f"{s['ctrl_occ_effect']:>12.5f}{s['real_mask_frac']:>12.5f}")
    print("  " + "-" * (len(h1) - 2))
    print(f"  {'ALL':<7}{overall['n']:>7}{overall['real_drop']:>12.5f}"
          f"{overall['ctrl_drop']:>12.5f}{overall['drop_gap']:>12.5f}"
          f"{overall['real_occ_effect']:>12.5f}{overall['ctrl_occ_effect']:>12.5f}"
          f"{overall['real_mask_frac']:>12.5f}")

    h2 = (f"  {'grade':<7}{'n':>7}{'dir_real':>12}{'dir_ctrl':>12}"
          f"{'spec_real':>12}{'spec_ctrl':>12}{'faith_real':>12}{'faith_ctrl':>12}")
    print("\n" + h2)
    print("  " + "-" * (len(h2) - 2))
    for g in grades:
        s = per_grade[g]
        print(f"  {g:<7}{s['n']:>7}{s['real_dir']:>12.5f}{s['ctrl_dir']:>12.5f}"
              f"{s['real_spec']:>12.5f}{s['ctrl_spec']:>12.5f}"
              f"{s['real_faith']:>12.5f}{s['ctrl_faith']:>12.5f}")
    print("  " + "-" * (len(h2) - 2))
    print(f"  {'ALL':<7}{overall['n']:>7}{overall['real_dir']:>12.5f}"
          f"{overall['ctrl_dir']:>12.5f}{overall['real_spec']:>12.5f}"
          f"{overall['ctrl_spec']:>12.5f}{overall['real_faith']:>12.5f}"
          f"{overall['ctrl_faith']:>12.5f}")

    print()
    print(f"  drop_gap = drop_ctrl - drop_real  ->  {overall['drop_gap']:+.5f}")
    print("    positive = occluding the DOMINANT concept suppresses it more than")
    print("    occluding a random concept suppresses that one. This is the number to")
    print("    compare between faith_on and faith_off; a lower real_drop alone does")
    print("    not rule out 'occluding anything salient lowers everything'.")
    print(f"  occ_gap  = occ_real - occ_ctrl    ->  {overall['occ_gap']:+.5f}")
    print(f"\n  drop floor at margin={getattr(cfg, 'faith_margin', 0.1)} means occlusion "
          "did not reduce the target concept")
    print(f"  mask ≈ 0 means the CAMs are too diffuse for threshold={faith_threshold}")
    print(f"\n  per-image CSV: {args.out_csv}")


if __name__ == "__main__":
    main()
