"""
Plot the faithfulness-loop collapse from a fold history CSV.

Read-only. Produces the figure for the degenerate-minimum argument: the mask
coverage and occlusion effect falling toward zero while L_drop also falls — the
loss decreasing because the perturbation vanished, not because the concepts
became causal.

Panels (shared x = epoch):
  top     mask_frac and occ_effect on a log axis — the perturbation collapsing
  bottom  L_drop and L_dir on a linear axis — the loss following it down
Both panels carry the nu curriculum ramp on a right-hand axis, plus a marker at
faith_start_epoch and a shaded ramp window, so the timing lines up visually.

Columns are taken from the trainer's history CSV (trainer.py writes
<run_dir>/fold{N}_history.csv): train_mask_frac, train_occ_effect,
train_loss_drop, train_loss_dir. Any that are absent are skipped with a warning
rather than failing — older runs predate the component logging.

Usage:
    python plot_faith_collapse.py --history runs/faith_on/fold0/fold0_history.csv
    python plot_faith_collapse.py --history runs/faith_on/fold0/fold0_history.csv \
        --start_epoch 25 --out runs/faith_on/faith_collapse.png
"""

from __future__ import annotations

import argparse
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from configs.config import DRConfig


# Series to plot: csv column -> (panel, label, colour, marker)
SERIES = {
    "train_mask_frac":  (0, "mask_frac (occluded pixel fraction)", "#1f77b4", "o"),
    "train_occ_effect": (0, "occ_effect  mean|c' - c|",            "#d62728", "s"),
    "train_loss_drop":  (1, "L_drop",                              "#2ca02c", "o"),
    "train_loss_dir":   (1, "L_dir",                               "#9467bd", "s"),
}


def read_history(path: str, start_epoch: int) -> tuple[list[int], dict[str, list]]:
    """Return (epochs, {column: values}) for rows with epoch >= start_epoch."""
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise SystemExit(f"{path} has no data rows.")

    epochs, cols = [], {k: [] for k in SERIES}
    for r in rows:
        try:
            ep = int(float(r["epoch"]))
        except (KeyError, ValueError):
            continue
        if ep < start_epoch:
            continue
        epochs.append(ep)
        for k in SERIES:
            raw = r.get(k, "")
            try:
                cols[k].append(float(raw))
            except (TypeError, ValueError):
                cols[k].append(float("nan"))
    return epochs, cols


def nu_at(epoch: int, nu_peak: float, nu_start: float,
          faith_start: int, ramp_epochs: int) -> float:
    """The trainer's curriculum ramp (trainer.py:375-379), recomputed for the overlay."""
    if epoch < faith_start or nu_peak <= 0.0:
        return 0.0
    progress = min((epoch - faith_start) / max(ramp_epochs, 1), 1.0)
    return nu_start + (nu_peak - nu_start) * progress


def main() -> None:
    cfg = DRConfig()
    p = argparse.ArgumentParser(description="Plot faithfulness collapse from fold history")
    p.add_argument("--history", type=str, required=True,
                   help="Path to fold{N}_history.csv")
    p.add_argument("--out", type=str, default=None,
                   help="Output PNG. Default: <history dir>/faith_collapse.png")
    p.add_argument("--start_epoch", type=int, default=25,
                   help="First epoch to plot (default 25 = DR faith_start_epoch)")
    p.add_argument("--title", type=str, default=None)
    p.add_argument("--nu", type=float, default=cfg.nu)
    p.add_argument("--nu_start", type=float, default=cfg.faith_nu_start)
    p.add_argument("--faith_start_epoch", type=int, default=cfg.faith_start_epoch)
    p.add_argument("--nu_ramp_epochs", type=int, default=cfg.faith_nu_ramp_epochs)
    p.add_argument("--dpi", type=int, default=160)
    args = p.parse_args()

    if not os.path.exists(args.history):
        raise SystemExit(f"History CSV not found: {args.history}")
    if args.out is None:
        args.out = os.path.join(os.path.dirname(os.path.abspath(args.history)),
                                "faith_collapse.png")

    epochs, cols = read_history(args.history, args.start_epoch)
    if not epochs:
        raise SystemExit(
            f"No rows with epoch >= {args.start_epoch} in {args.history}."
        )

    present = [k for k in SERIES if any(v == v for v in cols[k])]   # any non-NaN
    missing = [k for k in SERIES if k not in present]
    if missing:
        print(f"[warn] columns absent or empty, skipping: {', '.join(missing)}")
    if not present:
        raise SystemExit(
            "None of the faithfulness columns are present. This history predates the "
            "drop/dir/spec/occ/mask logging (commit 82ecbee)."
        )

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        raise SystemExit("matplotlib is required: pip install matplotlib")

    fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True,
                             gridspec_kw={"height_ratios": [1, 1]})
    nus = [nu_at(e, args.nu, args.nu_start, args.faith_start_epoch,
                 args.nu_ramp_epochs) for e in epochs]
    ramp_end = args.faith_start_epoch + args.nu_ramp_epochs

    for panel, ax in enumerate(axes):
        for key in present:
            pnl, label, colour, marker = SERIES[key]
            if pnl != panel:
                continue
            ax.plot(epochs, cols[key], marker=marker, markersize=3.5,
                    linewidth=1.6, color=colour, label=label)

        ax.axvspan(args.faith_start_epoch, ramp_end, color="grey", alpha=0.10,
                   label="nu ramp window" if panel == 0 else None)
        ax.axvline(args.faith_start_epoch, color="grey", linestyle=":", linewidth=1.2)
        ax.grid(alpha=0.3)

        axn = ax.twinx()
        axn.plot(epochs, nus, linestyle="--", linewidth=1.4, color="#888888",
                 label="nu (curriculum ramp)")
        axn.set_ylabel("nu")
        axn.set_ylim(0, max(args.nu * 1.25, 1e-6))

        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = axn.get_legend_handles_labels()
        # Panel 0's curves fall left-to-right and the nu ramp climbs into the
        # upper right, so keep that legend low-left; panel 1 has room up top.
        ax.legend(h1 + h2, l1 + l2, fontsize=8, framealpha=0.9,
                  loc="lower left" if panel == 0 else "upper right")

    axes[0].set_yscale("log")
    axes[0].set_ylabel("perturbation magnitude (log)")
    axes[0].set_title("Occluder collapse: the perturbation vanishes")
    axes[1].set_ylabel("loss component")
    axes[1].set_title("L_faith components follow it down")
    axes[1].set_xlabel("epoch")

    run_name = os.path.basename(os.path.dirname(os.path.abspath(args.history)))
    parent = os.path.basename(
        os.path.dirname(os.path.dirname(os.path.abspath(args.history)))
    )
    fig.suptitle(args.title or f"Faithfulness collapse — {parent}/{run_name}",
                 fontsize=12, y=0.98)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(args.out, dpi=args.dpi)
    print(f"wrote {args.out}")

    # Console summary — the numbers the figure is making the case with
    print(f"\nepochs {epochs[0]}..{epochs[-1]}  ({len(epochs)} rows)")
    for key in present:
        vals = [v for v in cols[key] if v == v]
        if vals:
            print(f"  {key:<18} first={vals[0]:.6f}  last={vals[-1]:.6f}  "
                  f"min={min(vals):.6f}  max={max(vals):.6f}")


if __name__ == "__main__":
    main()
