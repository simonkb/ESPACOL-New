"""
TSCGP Sanity Check — zero retraining required.

Tests two things:
  1. Does the text-derived R matrix recover known clinical structure (vs existing V)?
  2. Using V as the grade-concept prior, does the profile-likelihood decoder
     produce meaningful standalone grade predictions from already-trained concept scores?

If (2) gives >= 55-60% standalone accuracy, the gated fusion is worth training.
If R agrees with V, computing R from text is scientifically defensible.

Usage (on cluster, after activating env G):
    python sanity_check_tscgp.py \
        --checkpoint /dpc/kuin0170/ESPACOL-New/runs/<exp>/fold0_best.pth \
        --dataset dr \
        --fold 0

If no checkpoint is given, only the R vs V comparison runs (no GPU needed).
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ── Clinical prior V (already in the codebase) ────────────────────────────────
# Concepts: MA, HE, EX, CWS, VB, IRMA, NV, PreRet, VitHem
_V = torch.tensor([
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # Grade 0
    [0.9, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # Grade 1
    [0.7, 0.7, 0.6, 0.3, 0.0, 0.0, 0.0, 0.0, 0.0],  # Grade 2
    [0.5, 0.9, 0.4, 0.5, 0.8, 0.7, 0.0, 0.0, 0.0],  # Grade 3
    [0.3, 0.8, 0.2, 0.2, 0.5, 0.5, 0.9, 0.8, 0.7],  # Grade 4
], dtype=torch.float32)

_CONCEPT_NAMES = ["MA", "HE", "EX", "CWS", "VB", "IRMA", "NV", "PreRet", "VitHem"]
_GRADE_NAMES   = ["G0", "G1", "G2", "G3", "G4"]


# ── Part 1: R vs V ────────────────────────────────────────────────────────────

def compute_R_from_text(text_encoder) -> torch.Tensor:
    """
    R_{k,c} = cos(g_k^raw, e_c^raw)  in [-1, 1]
    Uses raw frozen BiomedCLIP embeddings (before the trainable projection).
    """
    G = text_encoder.raw_text_embeddings    # (K, 1280) — L2-normalised
    E = text_encoder.raw_concept_embeddings # (C, 1280) — L2-normalised
    R = torch.matmul(G, E.T)               # (K, C) — cosine similarities
    return R.cpu()


def print_matrix(name: str, M: torch.Tensor, row_names, col_names) -> None:
    print(f"\n{'─'*60}")
    print(f" {name}  ({M.shape[0]}×{M.shape[1]})")
    print(f"{'─'*60}")
    header = f"{'':>6}" + "".join(f"{c:>8}" for c in col_names)
    print(header)
    for i, row_name in enumerate(row_names):
        row = f"{row_name:>6}" + "".join(f"{M[i,j]:8.3f}" for j in range(M.shape[1]))
        print(row)


def compare_R_V(R: torch.Tensor) -> None:
    V = _V
    # Normalise R to [0,1] using sigmoid for comparison (raw cosine ∈ [-1,1])
    R_prob = torch.sigmoid(R * 5.0)   # temperature 5 → moderate sharpening

    print_matrix("V — clinical prior (ground truth)", V, _GRADE_NAMES, _CONCEPT_NAMES)
    print_matrix("R — BiomedCLIP cosine (raw)",      R, _GRADE_NAMES, _CONCEPT_NAMES)
    print_matrix("R → sigmoid(5R) [0,1 scaled]",    R_prob, _GRADE_NAMES, _CONCEPT_NAMES)

    # Pearson correlation between V and R_prob (flattened)
    v_flat = V.flatten()
    r_flat = R_prob.flatten()
    corr = torch.corrcoef(torch.stack([v_flat, r_flat]))[0, 1].item()
    print(f"\nPearson(V, sigmoid(5R)) = {corr:.4f}   (1.0 = perfect recovery)")

    # Per-grade winner concept (highest value)
    print("\nHighest-scoring concept per grade:")
    for k in range(5):
        v_top = _CONCEPT_NAMES[V[k].argmax()]
        r_top = _CONCEPT_NAMES[R_prob[k].argmax()]
        match = "✓" if v_top == r_top else "✗"
        print(f"  Grade {k}: V→{v_top:6s}  R→{r_top:6s}  {match}")


# ── Part 2: Profile-likelihood standalone accuracy ────────────────────────────

def profile_likelihood(mean_concept_logits: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    """
    G_{n,k} = Σ_c [ V_{k,c} log(p_{n,c}) + (1−V_{k,c}) log(1−p_{n,c}) ]

    mean_concept_logits: (N, C) — raw concept logits (before sigmoid)
    V: (K, C)
    Returns: (N, K) grade compatibility scores
    """
    p = torch.sigmoid(mean_concept_logits)         # (N, C)
    p = p.clamp(1e-6, 1 - 1e-6)

    log_p     = torch.log(p)                       # (N, C)
    log_1mp   = torch.log(1 - p)                   # (N, C)

    # G_{n,k} = V_k · log_p_n + (1-V_k) · log_1mp_n
    G = log_p.unsqueeze(1) * V.unsqueeze(0) + \
        log_1mp.unsqueeze(1) * (1 - V.unsqueeze(0))   # (N, K, C)
    return G.sum(dim=-1)                           # (N, K)


def eval_concept_decoder(checkpoint_path: str, args) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Build model ──
    from configs.config import DRConfig
    from models.framework import build_model
    from utils.checkpoint import load_checkpoint
    from models.text import ClinicalTextEncoder
    from configs.clinical_text import DR_CLASS_DESCRIPTIONS, DR_CONCEPTS

    cfg = DRConfig()
    cfg.dataset = args.dataset.upper()

    model = build_model(
        n_classes=cfg.n_classes,
        pretrained=False,
        use_multi_tile=True,
        tile_grid=cfg.tile_grid,
        use_tile_transformer=True,
        use_grade_prototypes=True,
        use_ordinal_head=True,
        use_concept_prototype=True,
        proto_temperature=cfg.proto_temperature,
    ).to(device)

    text_encoder = ClinicalTextEncoder(
        model_name=cfg.text_encoder_name,
        class_descriptions=DR_CLASS_DESCRIPTIONS,
        proj_out_dim=cfg.proj_out_dim,
        device=device,
        concept_descriptions=DR_CONCEPTS,
    ).to(device)

    load_checkpoint(checkpoint_path, model, None, text_encoder, device)
    model.eval()
    text_encoder.eval()

    # ── R vs V ──
    R = compute_R_from_text(text_encoder)
    compare_R_V(R)

    # ── Data ──
    from Datasets.dataloaders import ImageLabelDataset, build_tile_transform
    from Datasets.aptos_loader import load_all_aptos_items
    from training.cross_val import APTOSCrossValidator, DRCrossValidator

    if args.dataset.lower() in ("aptos",):
        all_items = load_all_aptos_items(args.data_root)
        cv = APTOSCrossValidator(all_items, n_folds=5, seed=42)
    else:
        from Datasets.dr_loader import load_all_dr_items
        all_items = load_all_dr_items(args.data_root)
        cv = DRCrossValidator(all_items, n_folds=10, seed=42)

    _, _, test_items = cv.get_fold(args.fold)
    tile_tfm = build_tile_transform(tile_size=cfg.img_size, tile_grid=cfg.tile_grid, augment=False)
    from torch.utils.data import DataLoader
    loader = DataLoader(
        ImageLabelDataset(test_items, transform=tile_tfm),
        batch_size=16, shuffle=False,
        num_workers=4, pin_memory=True,
    )

    # ── Collect predictions ──
    all_labels, all_visual_preds, all_concept_preds = [], [], []

    with torch.no_grad():
        concept_embeds = text_encoder.get_concept_embeds().to(device)
        V = _V.to(device)

        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            out = model(imgs)

            # Visual ODH prediction
            ordinal_logits = out.get("ordinal_logits")      # (N, K-1)
            if ordinal_logits is not None:
                probs = torch.sigmoid(ordinal_logits)       # (N, K-1)
                visual_pred = probs.sum(dim=-1).round().long().clamp(0, cfg.n_classes - 1)
            else:
                visual_pred = out["logits"].argmax(dim=-1)

            # Concept-based prediction via profile likelihood
            raw_tile = out.get("raw_tile_features")         # (N, T, 1280)
            if raw_tile is not None:
                N, T, D = raw_tile.shape
                from models.concept_prototype import ConceptGradePrototypeModule
                # Extract mean concept logits from the model's CGPM
                # Access through the model's cgpm module directly
                cgpm = model.cgpm if hasattr(model, "cgpm") else model.head.cgpm
                tiles_flat = raw_tile.view(N * T, D)
                proj_tiles = F.normalize(cgpm.concept_projection(tiles_flat), dim=-1).view(N, T, -1)
                concept_norm = F.normalize(concept_embeds, dim=-1)
                scores = torch.einsum("ntd,cd->ntc", proj_tiles, concept_norm)  # (N, T, C)
                mean_logits = scores.mean(dim=1) * 10.0    # (N, C) — match training scaling

                G = profile_likelihood(mean_logits, V)     # (N, K)
                concept_pred = G.argmax(dim=-1)            # (N,)
            else:
                concept_pred = torch.zeros_like(labels)

            all_labels.append(labels.cpu())
            all_visual_preds.append(visual_pred.cpu())
            all_concept_preds.append(concept_pred.cpu())

    labels_np  = torch.cat(all_labels).numpy()
    visual_np  = torch.cat(all_visual_preds).numpy()
    concept_np = torch.cat(all_concept_preds).numpy()

    # ── Report ──
    visual_acc  = (visual_np  == labels_np).mean() * 100
    concept_acc = (concept_np == labels_np).mean() * 100

    print(f"\n{'═'*60}")
    print(f" Fold {args.fold} — {len(labels_np)} images")
    print(f"{'═'*60}")
    print(f" Visual ODH standalone accuracy  : {visual_acc:.2f}%")
    print(f" Concept decoder standalone acc  : {concept_acc:.2f}%")
    print(f"{'═'*60}")

    # Confusion matrix for concept decoder
    K = cfg.n_classes
    conf = np.zeros((K, K), dtype=int)
    for t, p in zip(labels_np, concept_np):
        conf[t, p] += 1
    print("\nConcept decoder confusion matrix (rows=true, cols=pred):")
    header = f"{'':>5}" + "".join(f"{i:>5}" for i in range(K))
    print(header)
    for i in range(K):
        row = f"G{i:>3} " + "".join(f"{conf[i,j]:>5}" for j in range(K))
        print(row)

    # Per-grade F1
    print("\nPer-grade precision / recall:")
    for k in range(K):
        tp = conf[k, k]
        prec = tp / conf[:, k].sum() if conf[:, k].sum() > 0 else 0
        rec  = tp / conf[k, :].sum() if conf[k, :].sum() > 0 else 0
        print(f"  Grade {k}: prec={prec:.3f}  rec={rec:.3f}  N={conf[k,:].sum()}")

    print("\nInterpretation:")
    print(f"  concept_acc >= 60% → V-based decoder is meaningful → fusion worth training")
    print(f"  concept_acc <  40% → concept scores are grade-uninformative → redesign needed")
    print(f"  Pearson(R,V) >= 0.7 → text embeddings recover clinical structure")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="TSCGP sanity check — no retraining")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to a trained checkpoint (.pth). "
                             "If omitted, only the R-vs-V comparison runs (no GPU needed).")
    parser.add_argument("--dataset",   type=str, default="dr",
                        choices=["dr", "aptos"],
                        help="Dataset used to train the checkpoint.")
    parser.add_argument("--data_root", type=str,
                        default="Datasets/aptos2019-blindness-detection",
                        help="Dataset root directory.")
    parser.add_argument("--fold",      type=int, default=0,
                        help="Which fold's test set to evaluate on.")
    args = parser.parse_args()

    if args.checkpoint is None:
        print("No checkpoint given — running R vs V comparison only (CPU, no data needed).")
        from configs.config import DRConfig
        from models.text import ClinicalTextEncoder
        from configs.clinical_text import DR_CLASS_DESCRIPTIONS, DR_CONCEPTS
        cfg = DRConfig()
        device = torch.device("cpu")
        text_encoder = ClinicalTextEncoder(
            model_name=cfg.text_encoder_name,
            class_descriptions=DR_CLASS_DESCRIPTIONS,
            proj_out_dim=cfg.proj_out_dim,
            device=device,
            concept_descriptions=DR_CONCEPTS,
        )
        R = compute_R_from_text(text_encoder)
        compare_R_V(R)
        return

    eval_concept_decoder(args.checkpoint, args)


if __name__ == "__main__":
    main()
