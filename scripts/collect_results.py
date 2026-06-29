"""Collect final_results.csv from all ablation and sweep run directories.

Usage (after all SLURM jobs finish):
    python scripts/collect_results.py

Outputs runs/ablation_summary.csv and prints a ranked summary table.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
CONFIG_FILE = REPO_ROOT / "configs" / "ablation_configs.json"

# Maps run_dir (relative) → config dict for joining config params to results
def load_config_index() -> dict[str, dict]:
    if not CONFIG_FILE.exists():
        return {}
    configs = json.loads(CONFIG_FILE.read_text())
    return {c["run_dir"]: c for c in configs}


def read_fold_result(csv_path: Path) -> dict | None:
    try:
        rows = list(csv.DictReader(csv_path.open()))
    except Exception:
        return None
    # Find fold 0 row (skip mean/std summary rows)
    for row in rows:
        if row.get("fold", "").strip() == "0":
            try:
                return {
                    "test_acc": float(row["test_acc"]),
                    "test_mae": float(row["test_mae"]),
                    "test_loss": float(row["test_loss"]),
                }
            except (KeyError, ValueError):
                pass
    return None


def main() -> None:
    config_index = load_config_index()

    patterns = [
        "runs/ablation/*/final_results.csv",
        "runs/sweep/*/final_results.csv",
    ]

    rows: list[dict] = []
    for pattern in patterns:
        for csv_path in sorted(REPO_ROOT.glob(pattern)):
            result = read_fold_result(csv_path)
            if result is None:
                print(f"[skip] {csv_path} — missing or unreadable")
                continue

            # Derive run_dir relative to repo root (csv lives directly inside run_dir)
            run_dir = str(csv_path.parent.relative_to(REPO_ROOT))
            cfg = config_index.get(run_dir, {})
            name = cfg.get("name", run_dir)

            rows.append(
                {
                    "name": name,
                    "run_dir": run_dir,
                    "test_acc": result["test_acc"],
                    "test_mae": result["test_mae"],
                    "test_loss": result["test_loss"],
                    "alpha": cfg.get("alpha", ""),
                    "beta": cfg.get("beta", ""),
                    "gamma": cfg.get("gamma", ""),
                    "delta": cfg.get("delta", ""),
                    "eta": cfg.get("eta", ""),
                    "nu": cfg.get("nu", ""),
                    "use_image_text": cfg.get("use_image_text", ""),
                    "use_concept_spine": cfg.get("use_concept_spine", ""),
                }
            )

    if not rows:
        print("No results found yet. Run jobs first.")
        return

    rows.sort(key=lambda r: r["test_acc"], reverse=True)

    # Write summary CSV
    out_path = REPO_ROOT / "runs" / "ablation_summary.csv"
    fieldnames = [
        "name", "test_acc", "test_mae", "test_loss",
        "alpha", "beta", "gamma", "delta", "eta", "nu",
        "use_image_text", "use_concept_spine", "run_dir",
    ]
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Summary written to {out_path}  ({len(rows)} configs)\n")

    # Print ablation progression (prefix 01_ → 07_)
    ablation = [r for r in rows if r["name"].startswith("0") and "_" in r["name"]]
    ablation.sort(key=lambda r: r["name"])
    if ablation:
        print("=== Ablation Progression (ranked by name) ===")
        header = f"{'Name':<22} {'test_acc':>9} {'test_mae':>9}"
        print(header)
        print("-" * len(header))
        for r in ablation:
            print(f"{r['name']:<22} {r['test_acc']:>8.2f}% {r['test_mae']:>9.4f}")
        print()

    # Print top-10 sweep configs
    sweep = [r for r in rows if r["name"].startswith("sweep_")]
    if sweep:
        print("=== Top-10 Sweep Configs (by test_acc) ===")
        print(
            f"{'Name':<14} {'acc':>7} {'mae':>7} "
            f"{'α':>7} {'β':>7} {'γ':>7} {'δ':>7} {'η':>7} {'ν':>7}"
        )
        print("-" * 72)
        for r in sweep[:10]:
            print(
                f"{r['name']:<14} {r['test_acc']:>6.2f}% {r['test_mae']:>7.4f} "
                f"{r['alpha']:>7} {r['beta']:>7} {r['gamma']:>7} "
                f"{r['delta']:>7} {r['eta']:>7} {r['nu']:>7}"
            )


if __name__ == "__main__":
    main()
