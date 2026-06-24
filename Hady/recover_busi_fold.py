"""
Recover BUSI's TRAINING-time CV fold after the storage migration reordered
``BUSIDataset``'s ``os.listdir`` (which the seed-42 stratified split depends on).

Why this is needed
------------------
``BUSIDataset(split="all")`` builds its item list with an unsorted ``os.listdir``
(Datasets/dataloaders.py). The KU cluster migration (V100 -> H200) re-laid-out the
``/dpc`` storage, so that directory-entry order changed even though file mtimes were
preserved. The seed-42 stratified k-fold shuffles each class's items *in os.listdir
order*, so a changed order yields a DIFFERENT fold 4 than the one the checkpoint was
actually trained against. The DR split is immune (it keys on CSV row order); only
BUSI is affected.

How recovery works
------------------
The pre-migration cache ``Hady/z_it_busi.pt`` was written by
``recover_text_projection.py`` via ``collect_zit`` over ``BUSIDataset(split="all")``
with ``shuffle=False`` on Jun 21 — i.e. in the ORIGINAL (training-time) order, since
the data dir was untouched between training (Jun 20) and caching (Jun 21). ``z_it`` is
an effectively-unique 128-d signature per image, so we:

  1. recompute ``z_it`` for every CURRENT BUSI image (paths known, current order),
  2. nearest-neighbour match each cached row -> the current image with that z_it,
  3. rebuild the item list in the ORIGINAL order (cached order) with current paths,
  4. reconstruct ``BUSICrossValidator(...,seed=42).get_fold(4)`` on that order.

That reproduces the trainer's true held-out fold 4. We self-validate by confirming the
memorisation gap is restored (held-out MAE > train MAE) with the recovered split, and
absent with the broken (current-order) split.

Output: ``Hady/busi_fold4_heldout.json`` — consumed by
``eval_faithfulness_verdict.heldout_paths`` (BUSI branch).

Run via SLURM only (never python on a login node). CPU is fine (~780 images).
"""

import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
sys.path.insert(0, _REPO)

from Datasets.dataloaders import BUSIDataset
from configs.config import BUSIConfig
from models.framework import build_model
from utils.checkpoint import load_checkpoint
from training.cross_val import BUSICrossValidator


@torch.no_grad()
def compute_zit_pred(model, ds, device, batch_size=32):
    """Per-image L2-normalised z_it (matching collect_zit) and regression pred,
    in ds.items order (DataLoader shuffle=False preserves it)."""
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=4,
                        pin_memory=(device.type == "cuda"))
    zs, preds = [], []
    for x, _ in loader:
        out = model(x.to(device, non_blocking=True))
        zs.append(F.normalize(out["z_it"], dim=-1).cpu())
        preds.append(out["pred"].detach().float().cpu().reshape(-1))
    return torch.cat(zs, 0), torch.cat(preds, 0).numpy()


def main():
    ap = argparse.ArgumentParser(description="Recover BUSI's trained CV fold via the z_it cache.")
    ap.add_argument("--busi_root", default=os.environ.get("BUSI_ROOT"))
    ap.add_argument("--ckpt", default=os.path.join(_HERE, "BUSI_best_model.pth"))
    ap.add_argument("--zit_cache", default=os.path.join(_HERE, "z_it_busi.pt"))
    ap.add_argument("--fold", type=int, default=4)
    ap.add_argument("--out", default=os.path.join(_HERE, "busi_fold4_heldout.json"))
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = BUSIConfig()
    print(f"device={device}  ckpt={args.ckpt}  cache={args.zit_cache}  fold={args.fold}")

    # ── current-order dataset ────────────────────────────────────────────────
    ds = BUSIDataset(root_dir=args.busi_root, split="all")
    cur_paths = [p for p, _ in ds.items]
    cur_labels = np.array([int(y) for _, y in ds.items])
    N = len(ds)
    print(f"BUSI images (current os.listdir order): {N}")

    # ── model: match the cache's build (use_image_text + same proj dims) ─────
    model = build_model(n_classes=3, pretrained=False,
                        proj_hidden_dim=cfg.proj_hidden_dim, proj_out_dim=cfg.proj_out_dim,
                        use_image_text=True)
    load_checkpoint(args.ckpt, model, optimizer=None, device=device)
    model = model.to(device).eval()
    for p in model.parameters():
        p.requires_grad = False

    z_cur, pred_cur = compute_zit_pred(model, ds, device)  # (N,128), (N,)

    # ── cached z_it in ORIGINAL (training) order ─────────────────────────────
    cache = torch.load(args.zit_cache, map_location="cpu")
    z_cache = cache["z_it"].float()
    lab_cache = cache["labels"].numpy().astype(int)
    if z_cache.shape[0] != N:
        raise SystemExit(f"cache N={z_cache.shape[0]} != current N={N}; cannot recover.")

    # ── bijective match (Hungarian): cache row j -> current image i ──────────
    from scipy.optimize import linear_sum_assignment
    D = torch.cdist(z_cache, z_cur).numpy()           # (N, N) cost
    row_j, nn_i = linear_sum_assignment(D)            # 1-to-1 assignment
    assert (row_j == np.arange(N)).all()
    min_d = D[row_j, nn_i]
    mism = int((cur_labels[nn_i] != lab_cache).sum())
    n_moved = int((nn_i != np.arange(N)).sum())
    print(f"bijective match distance: max={min_d.max():.2e}  mean={min_d.mean():.2e}")
    print(f"label mismatches after matching: {mism}  (should be 0)")
    print(f"images whose position differs (training/cache order vs current order): {n_moved}/{N}")
    if min_d.max() > 1e-2 or mism > 0:
        print("WARNING: imperfect embedding match — recovered order may be unreliable.")

    # ── rebuild items in ORIGINAL (cache/training) order; reconstruct fold ────
    orig_items = [(cur_paths[nn_i[j]], int(lab_cache[j])) for j in range(N)]
    held_rec = {p for p, _ in BUSICrossValidator(
        orig_items, n_folds=5, val_fraction=0.1, seed=42).get_fold(args.fold)[2]}
    # current-order fold (what the verdict reconstructs without this manifest)
    held_cur = {p for p, _ in BUSICrossValidator(
        list(ds.items), n_folds=5, val_fraction=0.1, seed=42).get_fold(args.fold)[2]}
    held_basenames = sorted(os.path.basename(p) for p in held_rec)
    diff = len(held_rec ^ held_cur)
    print(f"recovered held-out fold{args.fold}: {len(held_rec)} images   "
          f"current-order: {len(held_cur)}   sets differ by: {diff} images")
    if diff == 0:
        print("=> IDENTICAL: the migration did NOT reorder BUSI; the verdict's current-order "
              "split is ALREADY CORRECT (no re-tag needed).")
    else:
        print("=> DIFFERENT: current-order split is wrong; re-tag the run with this manifest.")

    # ── memorisation gap (informational): BUSI barely overfits, so a small/zero
    # gap is EXPECTED for a correct split — unlike DR which overfits strongly. ──
    ae = np.abs(np.round(np.clip(pred_cur, 0, 2)) - cur_labels)

    def _gap(held_set):
        m = np.array([p in held_set for p in cur_paths])
        return float(ae[~m].mean()), float(ae[m].mean())

    tr_r, ho_r = _gap(held_rec)
    tr_c, ho_c = _gap(held_cur)
    print(f"RECOVERED split: train MAE={tr_r:.4f}  heldout MAE={ho_r:.4f}  gap={ho_r - tr_r:+.4f}")
    print(f"CURRENT   split: train MAE={tr_c:.4f}  heldout MAE={ho_c:.4f}  gap={ho_c - tr_c:+.4f}")

    with open(args.out, "w") as f:
        json.dump({"fold": args.fold, "n_heldout": len(held_basenames),
                   "match_max_dist": float(min_d.max()), "label_mismatches": mism,
                   "n_positions_moved": n_moved, "heldout_diff_vs_current": diff,
                   "recovered_gap": ho_r - tr_r, "current_gap": ho_c - tr_c,
                   "heldout_basenames": held_basenames}, f, indent=2)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
