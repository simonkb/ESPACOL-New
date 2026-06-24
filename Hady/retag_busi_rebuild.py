"""
Post-process a completed verdict run: re-tag BUSI rows with the RECOVERED held-out
fold (Hady/busi_fold4_heldout.json) and rebuild the partitioned artifacts from the
existing per_image.csv — WITHOUT re-running the model. DR rows are already correctly
tagged (CSV-deterministic) and are left untouched; the BUSI/DR `overall` partition is
unchanged by re-tagging (only the train<->heldout assignment moves).

Reuses summarize_group / _bh_fdr / _verdict / write_report / plot_* from
eval_faithfulness_verdict, so the rebuilt artifacts match the run's format exactly,
including per-partition BH-FDR across all 8 groups.

Run via SLURM (prod/CPU). Example:
    python Hady/retag_busi_rebuild.py --run /dpc/.../runs/faithful_verdict_full
"""

import argparse
import csv
import json
import os
import sys
from types import SimpleNamespace

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
sys.path.insert(0, _HERE)   # import the sibling eval module
sys.path.insert(0, _REPO)

import eval_faithfulness_verdict as V  # provides stats/report/plot fns (+ heavy imports)

_FLOAT = {"c_before", "c_on", "drop_on", "drop_off", "specificity", "mask_frac",
          "reg_before", "reg_on"}
_INT = {"true_label", "index", "grade_before", "grade_after", "grade_changed"}


def _read_rows(csv_path):
    rows = []
    with open(csv_path) as f:
        for r in csv.DictReader(f):
            d = dict(r)
            for k in _FLOAT:
                if d.get(k, "") != "":
                    d[k] = float(d[k])
            for k in _INT:
                if d.get(k, "") != "":
                    d[k] = int(float(d[k]))
            rows.append(d)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="run dir containing per_image.csv")
    ap.add_argument("--manifest", default=os.path.join(_HERE, "busi_fold4_heldout.json"))
    ap.add_argument("--out", default=None, help="output dir (default: in place = --run)")
    # report metadata — must match the original run for the header to read correctly
    ap.add_argument("--per-class", type=int, default=0)
    ap.add_argument("--controls", type=int, default=4)
    ap.add_argument("--occlusion", default="blur")
    ap.add_argument("--blur-sigma", type=float, default=21.0)
    ap.add_argument("--encoder", default="biomedclip")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--busi-fold", type=int, default=4)
    ap.add_argument("--dr-fold", type=int, default=2)
    args = ap.parse_args()
    out = args.out or args.run
    os.makedirs(out, exist_ok=True)

    rows = _read_rows(os.path.join(args.run, "per_image.csv"))
    with open(args.manifest) as f:
        held = set(json.load(f)["heldout_basenames"])

    n_re = 0
    for r in rows:
        if str(r["dataset"]).upper() == "BUSI":
            new = "heldout" if r["image"] in held else "train"
            n_re += int(new != r["split"])
            r["split"] = new
    print(f"re-tagged BUSI: {n_re} rows changed using {len(held)} recovered held-out basenames")

    # ── rebuild per_group exactly like eval main() ───────────────────────────
    PARTS = ("overall", "train", "heldout")
    per_group = {p: [] for p in PARTS}
    per_image_by_group = {p: {} for p in PARTS}
    keys, by_key = [], {}
    for r in rows:
        k = f'{r["dataset"]}:{r["group"]}'
        if k not in by_key:
            by_key[k] = []
            keys.append(k)
        by_key[k].append(r)

    for k in keys:
        grp_rows = by_key[k]
        ci_seed = args.seed + int(grp_rows[0]["true_label"])  # matches main(): seed+label
        for part in PARTS:
            sub = grp_rows if part == "overall" else [r for r in grp_rows if r["split"] == part]
            if not sub:
                continue
            stats = V.summarize_group(sub, ci_seed=ci_seed)
            stats["group_key"] = k
            stats["partition"] = part
            per_group[part].append(stats)
            per_image_by_group[part][k] = [r["specificity"] for r in sub]

    flat = []
    for part in PARTS:
        groups = per_group[part]
        if not groups:
            continue
        padj = V._bh_fdr([g["p_spec_raw"] for g in groups])
        for g, pa in zip(groups, padj):
            g["p_spec_adj"] = float(pa)
            g["verdict"] = V._verdict(g["drop_on_med"], g["spec_med"], g["p_drop"],
                                      g["p_spec_adj"], g["mask_med"])
            flat.append(g)

    # ── write corrected artifacts ────────────────────────────────────────────
    img_fields = ["dataset", "group", "true_label", "image", "index", "split",
                  "top_concept", "c_before", "c_on", "drop_on", "drop_off",
                  "specificity", "mask_frac", "reg_before", "reg_on",
                  "grade_before", "grade_after", "grade_changed"]
    with open(os.path.join(out, "per_image.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=img_fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{r[k]:.4f}" if isinstance(r[k], float) else r[k])
                        for k in img_fields})

    grp_fields = ["group_key", "partition", "n", "verdict", "drop_on_med", "drop_on_mean",
                  "drop_off_med", "spec_med", "spec_mean", "spec_ci", "spec_iqr",
                  "mask_med", "pass_rate", "locspec_rate", "anti_rate",
                  "grade_change_rate", "p_drop", "p_spec_raw", "p_spec_adj"]
    with open(os.path.join(out, "per_group_verdict.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=grp_fields)
        w.writeheader()
        for g in flat:
            w.writerow({k: g.get(k) for k in grp_fields})
    with open(os.path.join(out, "verdict.json"), "w") as f:
        json.dump(flat, f, indent=2)

    rep_args = SimpleNamespace(
        per_class=args.per_class, occlusion=args.occlusion, blur_sigma=args.blur_sigma,
        controls=args.controls, encoder=args.encoder, seed=args.seed,
        busi_fold=args.busi_fold, dr_fold=args.dr_fold)
    V.plot_specificity_box(per_group["overall"], per_image_by_group["overall"],
                           os.path.join(out, "verdict_specificity_box.png"))
    V.plot_dropon_vs_off(per_group["overall"], os.path.join(out, "verdict_dropon_vs_off.png"))
    if per_group["heldout"]:
        V.plot_specificity_box(per_group["heldout"], per_image_by_group["heldout"],
                               os.path.join(out, "verdict_specificity_box_heldout.png"))
        V.plot_dropon_vs_off(per_group["heldout"],
                             os.path.join(out, "verdict_dropon_vs_off_heldout.png"))
    V.write_report(per_group, rep_args, out, n_images=len(rows))
    print(f"rebuilt corrected artifacts in {out}")


if __name__ == "__main__":
    main()
