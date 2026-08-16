"""
Collapse several eval_faithfulness.py per-image CSVs into one markdown table.

Read-only. Each --entry is LABEL=PATH, where PATH is an eval_faithfulness.py
--out_csv. Means are recomputed from the per-image rows rather than scraped from
stdout, so the table cannot drift from the data.

Optionally appends the trainer's own logged row for the checkpoint's epoch, so
the four measured configurations sit next to the number they are explaining.

Usage:
    python make_bisect_table.py \
        --entry "eval / val=runs/faith_on/fold0/bisect_eval_val.csv" \
        --entry "train / train=runs/faith_on/fold0/bisect_train_train.csv" \
        --entry "eval / train=runs/faith_on/fold0/bisect_eval_train.csv" \
        --entry "train / val=runs/faith_on/fold0/bisect_train_val.csv" \
        --history runs/faith_on/fold0/fold0_history.csv --epoch 42 \
        --out runs/faith_on/fold0/bisect.md
"""

from __future__ import annotations

import argparse
import csv
import os
import statistics

COLUMNS = [
    ("real_drop", "drop"),
    ("ctrl_drop", "drop_ctrl"),
    ("drop_gap", "drop_gap"),
    ("real_dir", "dir"),
    ("real_spec", "spec"),
    ("real_occ_effect", "occ"),
    ("real_mask_frac", "mask"),
]


def load(path: str) -> dict:
    if not os.path.exists(path):
        return {"error": "missing file"}
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return {"error": "empty"}

    out = {"n": len(rows)}
    for key, _label in COLUMNS:
        vals = []
        for r in rows:
            try:
                vals.append(float(r[key]))
            except (KeyError, TypeError, ValueError):
                pass
        out[key] = statistics.fmean(vals) if vals else None
    return out


def trainer_row(history: str, epoch: int) -> dict | None:
    if not history or not os.path.exists(history):
        return None
    with open(history, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r.get("epoch", "").strip() == str(epoch):
                return r
    return None


def fmt(v) -> str:
    return "—" if v is None else f"{v:.5f}"


def main() -> None:
    p = argparse.ArgumentParser(description="Combine bisect CSVs into a markdown table")
    p.add_argument("--entry", action="append", required=True,
                   help="LABEL=PATH, repeatable")
    p.add_argument("--history", type=str, default=None,
                   help="fold{N}_history.csv, to append the trainer's own row")
    p.add_argument("--epoch", type=int, default=None,
                   help="Epoch to pull from --history (the checkpoint's stored epoch)")
    p.add_argument("--out", type=str, default=None, help="Write markdown here")
    p.add_argument("--title", type=str, default="Faithfulness bisect")
    args = p.parse_args()

    entries = []
    for e in args.entry:
        if "=" not in e:
            raise SystemExit(f"--entry must be LABEL=PATH, got: {e}")
        label, path = e.split("=", 1)
        entries.append((label.strip(), path.strip()))

    header = "| configuration | n | " + " | ".join(l for _k, l in COLUMNS) + " |"
    sep = "|---" * (len(COLUMNS) + 2) + "|"
    lines = [f"## {args.title}", "", header, sep]

    for label, path in entries:
        d = load(path)
        if "error" in d:
            lines.append(f"| {label} | — | " + " | ".join("—" for _ in COLUMNS)
                         + f" |  <!-- {d['error']}: {path} -->")
            continue
        lines.append(f"| {label} | {d['n']} | "
                     + " | ".join(fmt(d[k]) for k, _l in COLUMNS) + " |")

    ref = trainer_row(args.history, args.epoch) if args.epoch is not None else None
    if ref is not None:
        mapping = {
            "real_drop": "train_loss_drop", "ctrl_drop": None, "drop_gap": None,
            "real_dir": "train_loss_dir", "real_spec": "train_loss_spec",
            "real_occ_effect": "train_occ_effect", "real_mask_frac": "train_mask_frac",
        }
        cells = []
        for key, _label in COLUMNS:
            src = mapping.get(key)
            raw = ref.get(src, "") if src else ""
            try:
                cells.append(f"{float(raw):.5f}")
            except (TypeError, ValueError):
                cells.append("—")
        lines.append(f"| **trainer @ epoch {args.epoch}** | — | " + " | ".join(cells) + " |")

    lines += [
        "",
        "`drop_gap = drop_ctrl - drop_real`; positive means occluding the dominant "
        "concept suppresses it more than occluding a random one.",
        "",
        "Trainer row measured in train() mode, with AMP, over the TRAIN split, "
        "**with augmentation**, at the training batch size, averaged over "
        "faithfulness batches only. The measured rows use eval transforms in every "
        "configuration, so augmentation is the one factor this bisect does not vary.",
    ]

    md = "\n".join(lines)
    print(md)
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(md + "\n")
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
