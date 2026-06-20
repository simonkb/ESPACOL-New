"""
Recover the lost text projection ("Interpreter B") for gamma-aligned explainability.
====================================================================================

Why this exists
---------------
The gamma checkpoints save the model only (backbone + heads, including the
``image_text_head`` that produces ``z_it``). They do NOT save the BioMedCLIP text
projection head that, during training, mapped class text descriptions into the
same 128-d space as ``z_it``. Without it, the explainability demo cannot embed
clinical concept phrases into the aligned space, so concept ("WHY") scores don't
reflect the gamma training.

We recover a functionally-equivalent text projection post-hoc. The key fact:
the image-text loss pulls each image's ``z_it`` toward its class's text
prototype, so after training a class's text prototype sits near the centroid (on
the unit sphere) of that class's ``z_it`` embeddings. Since ``z_it`` is frozen and
saved, we can fit a fresh projection ``B'`` so that the class-description text
prototypes land on those centroids — reproducing the alignment geometry.

For BUSI the text tower was frozen the whole run, so this is essentially exact.
For DR the last 2 text-tower layers were fine-tuned and are lost, so we fit
against the base BioMedCLIP tower — a faithful approximation (documented).

What it does
------------
1. Load a gamma checkpoint, freeze the model, run the (training) images through
   it and cache ``z_it`` + labels.
2. Compute per-class ``z_it`` centroids = exact class anchors + validation oracle.
3. Fit a small, regularized LINEAR projection ``B'`` (768 -> 128) with:
     - the original ImageTextOrdinalLoss over the cached z_it, at the dataset's
       real temperature + lambda_ord_it (NOT the loss-class defaults);
     - an orthogonality penalty to keep B' near-isometric;
     - a concept-phrase isometry penalty so B' generalizes to arbitrary concept
       phrases (not just the 3-5 class descriptions).
4. Save B' + centroids + class prototypes, and print sanity checks.

Run (HPC, env sourced via slurm/_env.sh):
    python Hady/recover_text_projection.py --dataset busi \
        --ckpt Hady/BUSI_best_fold4.pth --busi_root $BUSI_ROOT \
        --out Hady/BUSI_text_projection.pth
    python Hady/recover_text_projection.py --dataset dr \
        --ckpt Hady/DR_best_acc_fold2.pth --dr_root $DR_ROOT \
        --out Hady/DR_text_projection.pth
"""

from __future__ import annotations

import argparse
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

# Make the repo root (parent of Hady/) importable regardless of CWD.
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _REPO_ROOT)

from Datasets.dataloaders import BUSIDataset, DRDataset, build_transform
from configs.config import BUSIConfig, DRConfig
from configs.clinical_text import (
    BUSI_CLASS_DESCRIPTIONS,
    DR_CLASS_DESCRIPTIONS,
    BUSI_CONCEPTS,
    DR_CONCEPTS,
)
from losses.image_text import ImageTextOrdinalLoss
from models.framework import build_model
from models.text import ClinicalTextEncoder
from utils.checkpoint import load_checkpoint

sys.path.insert(0, _HERE)  # so `text_projection` (same dir) imports cleanly
from text_projection import RecoveredTextProjection, save_text_projection

# Concept-phrase templates — MUST match explainability.py::_encode_biomedclip so
# the geometry B' is regularized on is the geometry B' will see at inference.
_TEMPLATES = {
    "busi": "a breast ultrasound image showing {}",
    "dr": "a fundus photograph showing {}",
}


# ─────────────────────────────────────────────────────────────────────────────
# Step 1-2: collect z_it and per-class centroids
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def collect_zit(model, loader, device):
    """Run the frozen model over the dataset and stack (z_it, labels)."""
    zs, ys = [], []
    n_done = 0
    for x, y in loader:
        out = model(x.to(device, non_blocking=True))
        z = out["z_it"]
        if z is None:
            raise RuntimeError(
                "Model produced z_it=None — the checkpoint has no image_text_head. "
                "Was it trained with gamma (use_image_text=True)?"
            )
        zs.append(F.normalize(z, dim=-1).cpu())
        ys.append(y.clone())
        n_done += x.shape[0]
        if n_done % 5000 < loader.batch_size:
            print(f"  z_it: {n_done} images ...", flush=True)
    return torch.cat(zs, 0), torch.cat(ys, 0)


def class_centroids(z_it: torch.Tensor, labels: torch.Tensor, n_classes: int) -> torch.Tensor:
    """Per-class mean direction of z_it (unit-normalized) — the exact anchors."""
    cents = []
    for c in range(n_classes):
        mask = labels == c
        if mask.sum() == 0:
            raise RuntimeError(f"No samples for class {c}; cannot form a centroid.")
        cents.append(F.normalize(z_it[mask].mean(0), dim=0))
    return torch.stack(cents, 0)  # (n_classes, dim)


# ─────────────────────────────────────────────────────────────────────────────
# Step 3: fit the regularized linear projection B'
# ─────────────────────────────────────────────────────────────────────────────

def fit_projection(
    raw_text: torch.Tensor,      # (n_classes, 768)  class-description tower embeddings
    concept_raw: torch.Tensor,   # (n_concepts, 768) concept-phrase tower embeddings
    z_it: torch.Tensor,          # (N, dim)          cached image embeddings
    labels: torch.Tensor,        # (N,)
    centroids: torch.Tensor,     # (n_classes, dim)  validation oracle
    *,
    out_dim: int,
    temperature: float,
    lambda_ord_it: float,
    lambda_orth: float,
    lambda_iso: float,
    epochs: int,
    batch_size: int,
    lr: float,
    device,
):
    proj = RecoveredTextProjection(raw_text.shape[-1], out_dim).to(device)
    opt = torch.optim.Adam(proj.parameters(), lr=lr)
    it_loss = ImageTextOrdinalLoss(temperature=temperature, lambda_ord=lambda_ord_it)

    raw_text = raw_text.to(device)
    concept_raw = concept_raw.to(device)
    z_it = z_it.to(device)
    labels = labels.to(device)
    centroids = centroids.to(device)
    eye = torch.eye(out_dim, device=device)

    # Pairwise BioMedCLIP cosines among concept phrases — the geometry to preserve.
    concept_cos = concept_raw @ concept_raw.t()  # (n_concepts, n_concepts)

    n = z_it.shape[0]
    best = {"score": -1.0, "state": None, "epoch": -1}
    for epoch in range(epochs):
        proj.train()
        perm = torch.randperm(n, device=device)
        for i in range(0, n, batch_size):
            idx = perm[i : i + batch_size]
            zb, yb = z_it[idx], labels[idx]

            prototypes = proj(raw_text)                       # (n_classes, dim)
            l_it = it_loss(zb, yb, prototypes)

            w = proj.proj.weight                              # (out_dim, in_dim)
            l_orth = ((w @ w.t() - eye) ** 2).mean()          # rows ~ orthonormal

            cproj = proj(concept_raw)                         # (n_concepts, dim)
            l_iso = ((cproj @ cproj.t() - concept_cos) ** 2).mean()

            loss = l_it + lambda_orth * l_orth + lambda_iso * l_iso
            opt.zero_grad()
            loss.backward()
            opt.step()

        # ── validate against the centroid oracle (diagonal dominance) ──
        proj.eval()
        with torch.no_grad():
            protos = proj(raw_text)                           # (n_classes, dim)
            S = protos @ centroids.t()                        # (n_classes, n_classes)
            diag_hit = (S.argmax(1) == torch.arange(S.shape[0], device=device)).float().mean().item()
            diag_mean = S.diag().mean().item()
        score = diag_hit + 0.01 * diag_mean  # prefer perfect self-retrieval, tie-break on margin
        if score > best["score"]:
            best = {"score": score, "state": {k: v.clone() for k, v in proj.state_dict().items()},
                    "epoch": epoch}
        if epoch % 25 == 0 or epoch == epochs - 1:
            print(f"  [fit] ep {epoch:4d} | L_it={l_it.item():.4f} "
                  f"L_orth={l_orth.item():.4f} L_iso={l_iso.item():.4f} "
                  f"| diag_hit={diag_hit:.3f} diag_mean={diag_mean:.3f}", flush=True)

    proj.load_state_dict(best["state"])
    proj.eval()
    print(f"  [fit] best epoch={best['epoch']} (selected by centroid self-retrieval)")
    return proj


# ─────────────────────────────────────────────────────────────────────────────
# Sanity checks
# ─────────────────────────────────────────────────────────────────────────────

def _fmt_matrix(M: torch.Tensor, labels) -> str:
    head = "          " + "".join(f"{l:>9}" for l in labels)
    rows = [head]
    for i, l in enumerate(labels):
        rows.append(f"{l:>9} " + "".join(f"{M[i, j].item():>9.3f}" for j in range(M.shape[1])))
    return "\n".join(rows)


@torch.no_grad()
def sanity_checks(proj, raw_text, centroids, z_it, labels, n_classes, device):
    proj.eval()
    raw_text = raw_text.to(device)
    centroids = centroids.to(device)
    z_it = z_it.to(device)
    labels = labels.to(device)
    protos = proj(raw_text)  # (n_classes, dim)
    short = [str(c) for c in range(n_classes)]

    print("\n" + "=" * 70)
    print("SANITY CHECKS")
    print("=" * 70)

    # 1. prototype <-> centroid cosine (rows=desc prototype, cols=z_it centroid)
    S = protos @ centroids.t()
    print("\n[1] prototype(row) <-> z_it-centroid(col) cosine  (want diagonal-dominant):")
    print(_fmt_matrix(S, short))
    diag_hit = (S.argmax(1) == torch.arange(n_classes, device=device)).float().mean().item()
    print(f"    diagonal self-retrieval: {diag_hit*100:.1f}%   (diag mean cos={S.diag().mean():.3f})")

    # 2. nearest-prototype accuracy + ordinal MAE on the z_it cloud
    sims = z_it @ protos.t()                 # (N, n_classes)
    pred = sims.argmax(1)
    acc = (pred == labels).float().mean().item()
    mae = (pred - labels).abs().float().mean().item()
    print(f"\n[2] z_it nearest-prototype accuracy: {acc*100:.2f}%   ordinal MAE: {mae:.3f}")

    # 3. ordinal-distance correlation: cos(P_i,P_j) should fall as |i-j| grows
    P = protos @ protos.t()
    print("\n[3] prototype-prototype cosine (want it to decrease with |i-j|):")
    print(_fmt_matrix(P, short))

    print("=" * 70 + "\n", flush=True)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def build_dataset(args, n_classes):
    tfm = build_transform(300)
    if args.dataset == "busi":
        root = args.busi_root or os.path.join(_REPO_ROOT, "Datasets", "BUSI")
        ds = BUSIDataset(root_dir=root, split="all", transform=tfm)
    else:
        root = args.dr_root or os.path.join(_REPO_ROOT, "Datasets", "DR")
        csv = args.train_csv or os.path.join(root, "trainLabels.csv")
        ds = DRDataset(root_dir=root, split="train", csv_path=csv, transform=tfm)
    return ds


def main():
    ap = argparse.ArgumentParser(description="Recover the gamma text projection (Interpreter B).")
    ap.add_argument("--dataset", choices=["busi", "dr"], required=True)
    ap.add_argument("--ckpt", required=True, help="Gamma checkpoint (.pth) for this dataset/fold.")
    ap.add_argument("--busi_root", default=None)
    ap.add_argument("--dr_root", default=None)
    ap.add_argument("--train_csv", default=None)
    ap.add_argument("--out", default=None, help="Output projection .pth (default Hady/<DS>_text_projection.pth).")
    ap.add_argument("--zit_cache", default=None, help="Where to cache z_it (default Hady/z_it_<DS>.pt).")
    ap.add_argument("--recompute", action="store_true", help="Recompute z_it even if cache exists.")
    # fit hyperparameters
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-2)
    ap.add_argument("--lambda_orth", type=float, default=1.0)
    ap.add_argument("--lambda_iso", type=float, default=1.0)
    ap.add_argument("--infer_batch", type=int, default=64, help="Batch size for z_it forward pass.")
    args = ap.parse_args()

    cfg = BUSIConfig() if args.dataset == "busi" else DRConfig()
    n_classes = cfg.n_classes
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Dataset={args.dataset}  n_classes={n_classes}  device={device}")
    print(f"temperature={cfg.temperature}  lambda_ord_it={cfg.lambda_ord_it}  "
          f"proj_out_dim={cfg.proj_out_dim}")

    out_path = args.out or os.path.join(_HERE, f"{args.dataset.upper()}_text_projection.pth")
    zit_path = args.zit_cache or os.path.join(_HERE, f"z_it_{args.dataset}.pt")

    # ── 1. cache z_it (or load cache) ────────────────────────────────────────
    if os.path.isfile(zit_path) and not args.recompute:
        print(f"Loading cached z_it from {zit_path}")
        cache = torch.load(zit_path, map_location="cpu")
        z_it, labels = cache["z_it"], cache["labels"]
    else:
        print(f"Building model (use_image_text=True) and loading {args.ckpt}")
        model = build_model(
            n_classes=n_classes, pretrained=False,
            proj_hidden_dim=cfg.proj_hidden_dim, proj_out_dim=cfg.proj_out_dim,
            use_image_text=True,
        )
        metrics = load_checkpoint(args.ckpt, model, optimizer=None, device=device)
        if metrics:
            print(f"  checkpoint metrics: { {k: metrics[k] for k in metrics if k in ('val_acc','val_mae')} }")
        model = model.to(device).eval()
        for p in model.parameters():
            p.requires_grad = False

        ds = build_dataset(args, n_classes)
        print(f"  images: {len(ds)}")
        loader = DataLoader(ds, batch_size=args.infer_batch, shuffle=False,
                            num_workers=4, pin_memory=True)
        z_it, labels = collect_zit(model, loader, device)
        torch.save({"z_it": z_it, "labels": labels}, zit_path)
        print(f"  cached z_it -> {zit_path}  (z_it={tuple(z_it.shape)})")

    counts = torch.bincount(labels, minlength=n_classes).tolist()
    print(f"z_it cloud: {z_it.shape[0]} samples, per-class counts={counts}")

    # ── 2. exact centroids ───────────────────────────────────────────────────
    centroids = class_centroids(z_it, labels, n_classes)

    # ── 3. base-tower text embeddings for descriptions + concept phrases ─────
    descriptions = BUSI_CLASS_DESCRIPTIONS if args.dataset == "busi" else DR_CLASS_DESCRIPTIONS
    concepts = BUSI_CONCEPTS if args.dataset == "busi" else DR_CONCEPTS
    template = _TEMPLATES[args.dataset]

    print(f"Loading BioMedCLIP text tower: {cfg.text_encoder_name}")
    enc = ClinicalTextEncoder(
        model_name=cfg.text_encoder_name,
        class_descriptions=descriptions,
        proj_out_dim=cfg.proj_out_dim,
        device=device,
    )
    enc.eval()
    with torch.no_grad():
        raw_text = enc.raw_text_embeddings.detach().float()   # (n_classes, 768), L2-normed
        concept_tokens = enc.tokenizer([template.format(c) for c in concepts]).to(device)
        concept_raw = F.normalize(enc.text_model.encode_text(concept_tokens).float(), dim=-1)
    print(f"  raw_text={tuple(raw_text.shape)}  concept_raw={tuple(concept_raw.shape)}")

    # ── 4. fit B' ─────────────────────────────────────────────────────────────
    print("Fitting regularized linear text projection B' ...")
    proj = fit_projection(
        raw_text=raw_text, concept_raw=concept_raw,
        z_it=z_it, labels=labels, centroids=centroids,
        out_dim=cfg.proj_out_dim,
        temperature=cfg.temperature, lambda_ord_it=cfg.lambda_ord_it,
        lambda_orth=args.lambda_orth, lambda_iso=args.lambda_iso,
        epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
        device=device,
    )

    # ── 5. save + sanity ──────────────────────────────────────────────────────
    with torch.no_grad():
        class_prototypes = proj(raw_text.to(device))
    save_text_projection(
        out_path, proj, centroids=centroids, class_prototypes=class_prototypes,
        meta={
            "dataset": args.dataset, "ckpt": os.path.abspath(args.ckpt),
            "temperature": cfg.temperature, "lambda_ord_it": cfg.lambda_ord_it,
            "text_encoder_name": cfg.text_encoder_name, "n_classes": n_classes,
            "n_zit_samples": int(z_it.shape[0]), "per_class_counts": counts,
            "lambda_orth": args.lambda_orth, "lambda_iso": args.lambda_iso,
        },
    )
    print(f"Saved recovered projection -> {out_path}")

    sanity_checks(proj, raw_text, centroids, z_it, labels, n_classes, device)


if __name__ == "__main__":
    main()
