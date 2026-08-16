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
--mode eval (default) vs --mode train
------------------------------------------------------------------------------
eval (default) departs from the training-time computation in two deliberate ways:
  - model.eval() — BatchNorm uses running statistics, so a per-image value does
    not depend on which other images share its batch.
  - float32 throughout, no autocast. Removes AMP nondeterminism from a
    measurement whose entire purpose is a small between-checkpoint difference.
This is the mode to use for the faith_on/faith_off comparison.

train reproduces the trainer's conditions instead — model.train() plus autocast,
placed exactly where trainer.py places it (main forward and the occluded forward
+ loss inside; the Grad-CAM pass outside). Use it to test whether those
conditions account for a gap between the trainer's logged drop and the eval-mode
value on the same checkpoint. Three things change together, so a match confirms
the combination rather than isolating a cause:
  - BatchNorm normalises with batch statistics -> per-image values now depend on
    batch composition, so --batch_size must match training (24) to compare
  - EfficientNet's stochastic depth becomes active -> outputs are stochastic;
    the run is seeded, but two different seeds will not agree exactly
  - fp16 autocast arithmetic replaces float32
Nothing is optimised in either mode. BN running buffers drift in train mode but
do not feed the outputs, and the checkpoint on disk is never rewritten.

Per-image components come from calling FaithfulnessLoss on one-image slices
(it reduces with .mean() internally, so a batched call would return batch means
and the CSV could not be per-image). Forwards stay batched; only the loss call
is per-image, and it is pure tensor arithmetic.

Grades are grouped by TRUE label, not predicted grade — the grouping must be
identical across the two checkpoints for the comparison to be meaningful.

Subsetting is stratified, never head-of-list: --per_grade N takes N images from
every grade (preferred here, since grades 3 and 4 are thin), --max_images N
takes N total allocated proportionally. Both are seeded, so the two checkpoints
measure exactly the same images.

--split selects which images. `val` (default) is the held-out fold that
train_dr.py uses as val==test. `train` is train_items_raw + val_items_held_out,
i.e. exactly the images the trainer fit and measured its logged drop over — use
it to test whether a train/val gap rather than eval-vs-train mode explains that
number. One residual difference remains in `train`: this script always uses eval
transforms, while the trainer saw the same images augmented.

After the tables the script prints the trainer's own logged row from
fold{fold}_history.csv at the checkpoint's stored epoch, so the comparison is
like-for-like rather than against a remembered log line from another epoch.

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
from contextlib import nullcontext
from torch.amp import autocast
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

def trainer_reference(ckpt_path: str, fold: int, epoch: int | None) -> dict | None:
    """
    Pull the trainer's own logged faithfulness row for a given epoch.

    The trainer writes <run_dir>/fold{fold}_history.csv per epoch (trainer.py:252),
    and run_dir is the checkpoint's own directory, so the history sits beside the
    .pth. Returns the row whose `epoch` matches, or None if unavailable.
    """
    if epoch is None:
        return None
    path = os.path.join(os.path.dirname(os.path.abspath(ckpt_path)),
                        f"fold{fold}_history.csv")
    if not os.path.exists(path):
        return None
    try:
        with open(path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if row.get("epoch", "").strip() == str(epoch):
                    return row
    except (OSError, csv.Error):
        return None
    return None


def select_subset(
    pairs: list, n_classes: int, seed: int,
    max_images: int | None, per_grade: int | None,
) -> list[int]:
    """
    Choose which val-split images to measure, returning ORIGINAL indices, sorted.

    Taking the first N would follow the CV split's ordering and leave grades 3
    and 4 (778 / 640 images across all of train) barely represented, so both
    modes stratify by true grade:

      --per_grade N   N images from every grade — equal representation, so the
                      rare-grade rows in the summary carry the same weight as
                      grade 0. Use this for the faith_on/faith_off comparison.
      --max_images N  N images total, allocated proportionally to each grade's
                      share of the split (largest-remainder), preserving the
                      natural class balance.

    Sampling is seeded, so both checkpoints measure exactly the same images.
    Indices are original val-split positions, which is what keys the
    control-concept draw — a given image gets the same control concept whether
    it was reached via --per_grade, --max_images, or a full run.
    """
    by_grade: dict[int, list[int]] = {g: [] for g in range(n_classes)}
    for i, (_path, label) in enumerate(pairs):
        by_grade[int(label)].append(i)

    rng = np.random.default_rng(seed)

    if per_grade is not None:
        quotas = {g: min(per_grade, len(by_grade[g])) for g in range(n_classes)}
    else:
        total = len(pairs)
        exact = {g: max_images * len(by_grade[g]) / total for g in range(n_classes)}
        quotas = {g: min(int(np.floor(v)), len(by_grade[g])) for g, v in exact.items()}
        # largest-remainder pass so the quotas sum to max_images where pools allow
        shortfall = max_images - sum(quotas.values())
        order = sorted(range(n_classes), key=lambda g: exact[g] - np.floor(exact[g]),
                       reverse=True)
        while shortfall > 0:
            progressed = False
            for g in order:
                if shortfall == 0:
                    break
                if quotas[g] < len(by_grade[g]):
                    quotas[g] += 1
                    shortfall -= 1
                    progressed = True
            if not progressed:      # every pool exhausted
                break

    picked: list[int] = []
    for g in range(n_classes):
        pool, k = by_grade[g], quotas[g]
        if k <= 0:
            continue
        if k >= len(pool):
            picked.extend(pool)
        else:
            chosen = rng.choice(len(pool), size=k, replace=False)
            picked.extend(pool[j] for j in chosen)

    return sorted(picked)


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
    c, E, w, cfg, faith_threshold, device, amp_enabled: bool = False,
) -> list[dict]:
    """
    Grad-CAM for concept_idx -> hard-mask occlusion -> re-forward -> components.

    concept_idx: (N,) long — which concept to attribute and occlude per image.
    Returns one dict of scalars per image in the batch.

    autocast placement mirrors the trainer exactly: the CAM forward/backward runs
    OUTSIDE autocast (trainer.py wraps it in enable_grad only), while the occluded
    forward and the FaithfulnessLoss call run INSIDE it (trainer.py:500-513).
    """
    N = x.shape[0]

    def amp():
        return (autocast(device_type=device.type, enabled=True)
                if amp_enabled else nullcontext())

    # Grad-CAM++ — the only step that needs gradients. No autocast: the trainer
    # does not wrap this pass either.
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
        with amp():
            out_occ = model(x_occ, concept_text_emb=concept_text_emb)

    c_prime = out_occ["c"].detach()
    E_prime = out_occ["E"].detach()
    T = heatmaps.shape[0] // N if is_tiled else 1

    results = []
    for i in range(N):
        with torch.no_grad(), amp():
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
                   help="Single fold index to measure")
    p.add_argument("--split", choices=["train", "val"], default="val",
                   help="val (default): the held-out fold, i.e. what train_dr.py uses "
                        "as val==test. train: the images the trainer actually fit and "
                        "measured its logged drop on (train_items_raw + val_held_out). "
                        "Use train to test whether a train/val gap, rather than "
                        "eval-vs-train mode, explains the trainer's number.")
    sampling = p.add_mutually_exclusive_group()
    sampling.add_argument("--max_images", type=int, default=None,
                          help="Measure N images total, sampled proportionally across "
                               "true grades. Default: whole val split")
    sampling.add_argument("--per_grade", type=int, default=None,
                          help="Measure N images from EACH true grade, so grades 3 and 4 "
                               "carry the same weight as grade 0. Preferred for the "
                               "faith_on/faith_off comparison")
    p.add_argument("--inspect", action="store_true",
                   help="Print the checkpoint's key structure and whether it can be "
                        "loaded, then exit. Does not touch the dataset.")
    p.add_argument("--train_csv", type=str, default=None,
                   help="Label CSV. Default: <dr_root>/trainLabels.csv")
    p.add_argument("--out_csv", type=str, default=None,
                   help="Per-image CSV. Default: <ckpt dir>/faithfulness_eval.csv")
    p.add_argument("--mode", choices=["eval", "train"], default="eval",
                   help="eval (default): model.eval(), float32, no autocast — per-image "
                        "values are batch-independent and reproducible. "
                        "train: model.train() + autocast, reproducing the trainer's "
                        "exact conditions (BatchNorm batch statistics, AMP, active "
                        "stochastic depth) to test whether they explain a gap between "
                        "the trainer's logged drop and this script's eval-mode value.")
    p.add_argument("--batch_size", type=int, default=8,
                   help="Forward batch size. In --mode eval this does not affect "
                        "per-image values; in --mode train it does (BatchNorm uses "
                        "batch statistics), so match the training batch size — 24.")
    p.add_argument("--num_workers", type=int, default=1,
                   help="DataLoader workers. Kept at 1: the compute here is 5 model "
                        "passes per batch, not image decoding, and the HPC node warns "
                        "when more workers are requested than it has available")
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
    model.to(device)

    # --mode train reproduces the trainer's conditions: BatchNorm normalises with
    # batch statistics rather than running estimates, EfficientNet's stochastic
    # depth becomes active, and AMP is enabled on CUDA exactly as cfg.amp does.
    # Nothing is optimised in either mode; BN running buffers drift in train mode
    # but that does not feed the outputs, which use batch statistics.
    amp_enabled = (args.mode == "train") and cfg.amp and device.type == "cuda"
    if args.mode == "train":
        model.train()
    else:
        model.eval()

    def amp_ctx():
        return (autocast(device_type=device.type, enabled=True)
                if amp_enabled else nullcontext())

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
    print(f"    split             {args.split}")
    print(f"    seed              {args.seed}  (also keys the control-concept draw)")
    if args.mode == "train":
        print(f"    mode              train()  [trainer conditions: BN batch stats, "
              f"stochastic depth active]")
        print(f"    autocast          {amp_enabled}  (cfg.amp={cfg.amp}, device={device.type})")
        print(f"    batch_size        {args.batch_size}  <- affects BN stats; training used "
              f"{cfg.batch_size}")
    else:
        print(f"    mode              eval()   [float32, no autocast, batch-independent]")

    # ── Fold val split — mirrors train_dr.py (val == test fold) ──────────────
    items = DRDataset(root_dir=args.dr_root, split="train", csv_path=args.train_csv).items
    cv = DRCrossValidator(
        items, n_folds=cfg.n_folds, val_fraction=cfg.val_fraction, seed=args.seed
    )
    _train_raw, _val_held_out, test_items = cv.get_fold(fold)

    # train_dr.py: train_items = train_items_raw + val_items_held_out; val = test fold.
    # The trainer's logged drop was measured over the train split, so reproduce it here.
    if args.split == "train":
        pool = _train_raw + _val_held_out
    else:
        pool = test_items

    if args.max_images is not None or args.per_grade is not None:
        orig_index = select_subset(
            pool, cfg.n_classes, args.seed, args.max_images, args.per_grade
        )
    else:
        orig_index = list(range(len(pool)))
    val_items = [pool[i] for i in orig_index]

    split_dist = {g: 0 for g in range(cfg.n_classes)}
    for _p, lab in val_items:
        split_dist[int(lab)] += 1
    print(f"\nFold {fold} {args.split} split: {len(pool)} images, "
          f"measuring {len(val_items)}")
    print(f"  grade distribution of the measured set: {split_dist}")
    if args.split == "train":
        print("  NOTE: eval transforms (no augmentation). The trainer measured these "
              "images with random flips/rotation/jitter applied.")

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

        # Clean forward — concept activations, evidence, weights.
        # autocast here mirrors the trainer's main forward (trainer.py:408).
        with torch.no_grad(), amp_ctx():
            out = model(x, concept_text_emb=concept_text_emb)
        c = out["c"].detach()                       # (N, M)
        E = out["E"].detach()                       # (N,)
        w = out["w"].detach()                       # (M,)

        # REAL: dominant concept by |w * c|
        top_k_idx = (w * c).abs().argmax(dim=1)     # (N,)

        # CONTROL: deterministic non-dominant concept per image. Keyed on the
        # ORIGINAL val-split index, so an image draws the same control concept
        # whether it arrived via --per_grade, --max_images, or a full run.
        ctrl_list = [
            control_concept(
                int(top_k_idx[i].item()), M, orig_index[n_seen + i], args.seed
            )
            for i in range(N)
        ]
        ctrl_idx = torch.tensor(ctrl_list, dtype=torch.long, device=device)

        real = measure_for_concept(
            model, layer_cam, faith_loss_fn, x, top_k_idx, concept_text_emb,
            c, E, w, cfg, faith_threshold, device, amp_enabled,
        )
        ctrl = measure_for_concept(
            model, layer_cam, faith_loss_fn, x, ctrl_idx, concept_text_emb,
            c, E, w, cfg, faith_threshold, device, amp_enabled,
        )

        for i in range(N):
            k_real, k_ctrl = int(top_k_idx[i].item()), ctrl_list[i]
            rows.append({
                "index": orig_index[n_seen + i],
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

    # ── Trainer's own logged row, for a like-for-like comparison ─────────────
    ckpt_epoch = state.get("epoch")
    ref = trainer_reference(args.ckpt, fold, ckpt_epoch)
    print()
    print(f"  TRAINER REFERENCE — fold{fold}_history.csv @ epoch {ckpt_epoch} "
          f"(the checkpoint's stored epoch)")
    if ref is None:
        print("    unavailable: no fold history CSV beside the checkpoint, or no row "
              f"for epoch {ckpt_epoch}")
    else:
        cols = [
            ("train_loss_drop", "drop"), ("train_loss_dir", "dir"),
            ("train_loss_spec", "spec"), ("train_loss_faith", "faith"),
            ("train_occ_effect", "occ"), ("train_mask_frac", "mask"),
        ]
        shown = [(lbl, ref[key]) for key, lbl in cols if key in ref and ref[key] != ""]
        if shown:
            print("    " + "  ".join(f"{lbl}={val}" for lbl, val in shown))
        else:
            print("    row found but no faithfulness columns "
                  "(run trained with nu=0, or an older trainer)")
        print("    measured by the trainer in train() mode, with AMP, over the TRAIN "
              "split, with augmentation, at batch_size="
              f"{cfg.batch_size}, averaged over faithfulness batches only.")
        print(f"    this run: mode={args.mode} split={args.split} "
              f"batch_size={args.batch_size} augment=False")

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
