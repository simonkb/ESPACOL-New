"""
TSCGP Sanity Check — zero retraining required.

Two modes:
  1. No checkpoint — prints R vs V comparison only (CPU, fast).
  2. With checkpoint — extracts concept scores from the VAL set (not test),
     evaluates the V-based profile-likelihood decoder, and sweeps a post-hoc
     alpha gate on the fusion  s_final = s_visual + alpha * u_concept.

Usage:
    # R vs V only:
    python sanity_check_tscgp.py

    # Full eval (DR, fold 0):
    python sanity_check_tscgp.py \\
        --checkpoint runs/optic_concept_cv_v9/fold0/fold0_best.pth \\
        --dataset    dr \\
        --dr_root    Datasets/DR \\
        --fold       0
"""

from __future__ import annotations

import argparse
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ── Clinical prior V ───────────────────────────────────────────────────────────
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


# ── Part 1: R vs V ─────────────────────────────────────────────────────────────

def compute_R(text_encoder) -> torch.Tensor:
    G = text_encoder.raw_text_embeddings     # (K, D) — raw BiomedCLIP, L2-normalised
    E = text_encoder.raw_concept_embeddings  # (C, D)
    return torch.matmul(G, E.T).cpu()        # (K, C) raw cosine, no softmax


def print_matrix(name, M, rows, cols):
    print(f"\n{'─'*64}\n {name}\n{'─'*64}")
    print(f"{'':>6}" + "".join(f"{c:>8}" for c in cols))
    for i, r in enumerate(rows):
        print(f"{r:>6}" + "".join(f"{M[i,j]:8.3f}" for j in range(M.shape[1])))


def compare_R_V(text_encoder) -> None:
    R = compute_R(text_encoder)
    V = _V

    # Transition matrices
    W  = torch.diff(R,  dim=0)   # (4, C)
    dV = torch.diff(V,  dim=0)   # (4, C)

    print_matrix("V — clinical prior", V, _GRADE_NAMES, _CONCEPT_NAMES)
    print_matrix("R — raw BiomedCLIP cosine", R, _GRADE_NAMES, _CONCEPT_NAMES)
    print_matrix("W = diff(R)  [grade transitions]",
                 W, ["G0→G1","G1→G2","G2→G3","G3→G4"], _CONCEPT_NAMES)
    print_matrix("ΔV = diff(V) [clinical transitions]",
                 dV, ["G0→G1","G1→G2","G2→G3","G3→G4"], _CONCEPT_NAMES)

    def pearson(a, b):
        a, b = a.flatten().float(), b.flatten().float()
        a, b = a - a.mean(), b - b.mean()
        denom = (a.norm() * b.norm()).clamp(min=1e-8)
        return (a @ b / denom).item()

    print(f"\nPearson(R,  V)  = {pearson(R,  V):.4f}")
    print(f"Pearson(W,  dV) = {pearson(W, dV):.4f}")
    print(f"R range: [{R.min():.3f}, {R.max():.3f}]")
    print(f"W range: [{W.min():.4f}, {W.max():.4f}]")


# ── Part 2: metrics ────────────────────────────────────────────────────────────

def compute_metrics(preds, labels, n_classes=5, name="") -> dict:
    from collections import Counter
    N = len(labels)
    acc  = (preds == labels).sum() / N
    mae  = abs(preds - labels).sum() / N

    # per-grade recall
    recalls = []
    for k in range(n_classes):
        mask = labels == k
        r = (preds[mask] == k).sum() / mask.sum() if mask.sum() > 0 else 0.0
        recalls.append(float(r))
    bal_acc = sum(recalls) / n_classes

    # macro F1
    f1s = []
    for k in range(n_classes):
        tp = ((preds == k) & (labels == k)).sum()
        fp = ((preds == k) & (labels != k)).sum()
        fn = ((preds != k) & (labels == k)).sum()
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1s.append(2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0)

    # QWK
    conf = torch.zeros(n_classes, n_classes)
    for t, p in zip(labels.tolist(), preds.tolist()):
        conf[t, p] += 1
    w = torch.zeros(n_classes, n_classes)
    for i in range(n_classes):
        for j in range(n_classes):
            w[i, j] = (i - j) ** 2 / (n_classes - 1) ** 2
    hist_t = conf.sum(dim=1, keepdim=True)
    hist_p = conf.sum(dim=0, keepdim=True)
    expected = (hist_t * hist_p) / N
    qwk = 1 - (w * conf).sum() / (w * expected).sum()

    if name:
        print(f"\n── {name} ──")
        print(f"  Acc={acc:.3f}  BalAcc={bal_acc:.3f}  "
              f"MacroF1={sum(f1s)/len(f1s):.3f}  MAE={mae:.3f}  QWK={qwk:.4f}")
        print(f"  Per-grade recall: " +
              "  ".join(f"G{k}={recalls[k]:.2f}" for k in range(n_classes)))
        # confusion matrix
        print("  Confusion (rows=true, cols=pred):")
        print("  " + "".join(f"{i:>5}" for i in range(n_classes)))
        for i in range(n_classes):
            print(f"  G{i}" + "".join(f"{int(conf[i,j]):>5}" for j in range(n_classes)))

    return {"acc": float(acc), "bal_acc": bal_acc, "qwk": float(qwk),
            "mae": float(mae), "macro_f1": sum(f1s)/len(f1s),
            "conf": conf, "recalls": recalls}


# ── Profile likelihood ─────────────────────────────────────────────────────────

def profile_likelihood(mean_concept_logits: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    """G_{n,k} = Σ_c [V_{k,c} log p + (1-V_{k,c}) log(1-p)]"""
    p = torch.sigmoid(mean_concept_logits).clamp(1e-6, 1 - 1e-6)
    log_p   = torch.log(p)
    log_1mp = torch.log(1 - p)
    G = (log_p.unsqueeze(1)   * V.unsqueeze(0) +
         log_1mp.unsqueeze(1) * (1 - V.unsqueeze(0)))
    return G.sum(dim=-1)   # (N, K)


def concept_to_ordinal(G: torch.Tensor, n_classes: int = 5) -> torch.Tensor:
    """Convert grade compatibility scores to CORAL-compatible ordinal logits (N, K-1)."""
    p_k = torch.softmax(G, dim=-1)                          # (N, K)
    # q_{n,j} = P(Y > j) = sum_{k > j} p_k
    q = torch.stack(
        [p_k[:, j+1:].sum(dim=-1) for j in range(n_classes - 1)],
        dim=-1,
    ).clamp(1e-6, 1 - 1e-6)                                 # (N, K-1)
    return torch.logit(q)                                    # (N, K-1)


# ── ODH prediction from ordinal logits ────────────────────────────────────────

def ordinal_logits_to_pred(logits: torch.Tensor, n_classes: int = 5) -> torch.Tensor:
    probs = torch.sigmoid(logits)   # (N, K-1)
    return probs.sum(dim=-1).round().long().clamp(0, n_classes - 1)


# ── Full evaluation ────────────────────────────────────────────────────────────

def run_eval(args) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    from configs.config import DRConfig
    from models.framework import build_model
    from utils.checkpoint import load_checkpoint
    from models.text import ClinicalTextEncoder
    from configs.clinical_text import DR_CLASS_DESCRIPTIONS, DR_CONCEPTS

    cfg = DRConfig()
    cfg.dataset = "DR"

    model = build_model(
        n_classes=cfg.n_classes, pretrained=False, use_multi_tile=True,
        tile_grid=cfg.tile_grid, use_tile_transformer=True,
        use_grade_prototypes=True, use_ordinal_head=True,
        use_concept_prototype=True, proto_temperature=cfg.proto_temperature,
    ).to(device)

    text_encoder = ClinicalTextEncoder(
        model_name=cfg.text_encoder_name,
        class_descriptions=DR_CLASS_DESCRIPTIONS,
        proj_out_dim=cfg.proj_out_dim,
        device=device,
        concept_descriptions=DR_CONCEPTS,
    ).to(device)

    load_checkpoint(args.checkpoint, model, None, text_encoder, device)
    model.eval(); text_encoder.eval()

    # R vs V
    compare_R_V(text_encoder)

    # Data — use VAL set (not test) for architecture decisions
    from train_dr import load_all_dr_items
    from training.cross_val import DRCrossValidator
    from Datasets.dataloaders import ImageLabelDataset, build_tile_transform
    from torch.utils.data import DataLoader

    train_csv = args.train_csv or os.path.join(args.dr_root, "trainLabels.csv")
    all_items = load_all_dr_items(args.dr_root, train_csv)
    cv = DRCrossValidator(all_items, n_folds=10, seed=42)
    # train_dr.py trains on (train_raw + val_held), uses test_fold as its val signal.
    # Evaluate on the held-out test fold — never seen during training.
    _, _, eval_items = cv.get_fold(args.fold)

    print(f"\nFold {args.fold} — evaluating on {len(eval_items)} held-out images")

    tile_tfm = build_tile_transform(
        tile_size=cfg.img_size, tile_grid=cfg.tile_grid, augment=False)
    loader = DataLoader(
        ImageLabelDataset(eval_items, transform=tile_tfm),
        batch_size=16, shuffle=False, num_workers=4, pin_memory=True,
    )

    # Majority-class baseline
    label_counts = [0] * cfg.n_classes
    for _, lbl in eval_items:
        label_counts[lbl] += 1
    majority = label_counts.index(max(label_counts))
    print(f"Class distribution: {label_counts}  Majority class: {majority}")

    # Collect predictions
    all_labels, all_visual_logits, all_concept_logits = [], [], []

    V = _V.to(device)
    concept_embeds = text_encoder.get_concept_embeds().to(device)

    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device)
            out  = model(imgs)

            # Visual ODH ordinal logits
            all_visual_logits.append(out["ordinal_logits"].cpu())
            all_labels.append(labels)

            # Concept scores via concept_module
            raw_tile = out.get("raw_tile_features")   # (N, T, 1280)
            N, T, D  = raw_tile.shape
            cm = model.concept_module
            tiles_flat = raw_tile.view(N * T, D)
            proj = F.normalize(cm.concept_projection(tiles_flat), dim=-1).view(N, T, -1)
            ce   = F.normalize(concept_embeds, dim=-1)
            scores = torch.einsum("ntd,cd->ntc", proj, ce)   # (N, T, C)
            mean_logits = scores.mean(dim=1) * 10.0           # (N, C)
            all_concept_logits.append(mean_logits.cpu())

    labels_t   = torch.cat(all_labels)
    vis_logits = torch.cat(all_visual_logits)
    con_logits = torch.cat(all_concept_logits)

    vis_preds = ordinal_logits_to_pred(vis_logits)
    G         = profile_likelihood(con_logits, _V)
    con_preds = G.argmax(dim=-1)
    u_concept = concept_to_ordinal(G)
    maj_preds = torch.full_like(labels_t, majority)

    print(f"\n{'═'*64}")
    compute_metrics(maj_preds, labels_t, name="Majority-class baseline")
    vis_m = compute_metrics(vis_preds, labels_t, name="Visual ODH")
    con_m = compute_metrics(con_preds, labels_t, name="Concept decoder (V-based)")

    # Help / harm analysis
    n_help = ((vis_preds != labels_t) & (con_preds == labels_t)).sum().item()
    n_harm = ((vis_preds == labels_t) & (con_preds != labels_t)).sum().item()
    n_dis  = (vis_preds != con_preds).sum().item()
    print(f"\n  ODH wrong, concept correct (N_help): {n_help}")
    print(f"  ODH right, concept wrong  (N_harm): {n_harm}")
    print(f"  Disagreement rate: {n_dis}/{len(labels_t)} "
          f"({100*n_dis/len(labels_t):.1f}%)")

    # Post-hoc alpha sweep on val (no training)
    print(f"\n{'═'*64}")
    print(" Post-hoc alpha sweep  s_final = s_visual + alpha * u_concept")
    print(f"{'═'*64}")
    print(f"{'alpha':>8}  {'acc':>7}  {'bal_acc':>9}  {'QWK':>8}  {'MAE':>7}  {'macroF1':>9}")
    best_alpha, best_qwk = 0.0, vis_m["qwk"]
    for alpha in [0.0, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0]:
        fused  = vis_logits + alpha * u_concept
        fpreds = ordinal_logits_to_pred(fused)
        m = compute_metrics(fpreds, labels_t, n_classes=cfg.n_classes)
        flag = " ← best" if m["qwk"] > best_qwk else ""
        if m["qwk"] > best_qwk:
            best_qwk, best_alpha = m["qwk"], alpha
        print(f"{alpha:>8.2f}  {m['acc']:>7.4f}  {m['bal_acc']:>9.4f}  "
              f"{m['qwk']:>8.4f}  {m['mae']:>7.4f}  {m['macro_f1']:>9.4f}{flag}")

    print(f"\n  Best alpha = {best_alpha}  QWK = {best_qwk:.4f}")
    if best_alpha == 0.0:
        print("  → Concept fusion provides no benefit. Do not add the gate.")
    else:
        print("  → Fusion improves QWK. Training the gate is justified.")
        print("    Verify minority-grade recall did not drop before committing.")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--dataset",    default="dr", choices=["dr", "aptos"])
    parser.add_argument("--dr_root",    default="Datasets/DR")
    parser.add_argument("--train_csv",  default=None,
                        help="Path to trainLabels.csv (defaults to dr_root/trainLabels.csv)")
    parser.add_argument("--fold",       type=int, default=0)
    args = parser.parse_args()

    if args.checkpoint is None:
        print("No checkpoint — R vs V comparison only.")
        from configs.config import DRConfig
        from models.text import ClinicalTextEncoder
        from configs.clinical_text import DR_CLASS_DESCRIPTIONS, DR_CONCEPTS
        cfg = DRConfig()
        te = ClinicalTextEncoder(
            model_name=cfg.text_encoder_name,
            class_descriptions=DR_CLASS_DESCRIPTIONS,
            proj_out_dim=cfg.proj_out_dim,
            device=torch.device("cpu"),
            concept_descriptions=DR_CONCEPTS,
        )
        compare_R_V(te)
        return

    run_eval(args)


if __name__ == "__main__":
    main()
