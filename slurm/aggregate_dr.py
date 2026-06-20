#!/usr/bin/env python
"""Aggregate the DR job-array per-fold results into final_results.csv (mean±std).

Reads <run_dir>/fold*/test_result.json (written race-safely by each array task)
and writes <run_dir>/final_results.csv. Stdlib only — runs anywhere.

Usage: python aggregate_dr.py [run_dir]
"""
import csv
import glob
import json
import os
import statistics as st
import sys

run_dir = sys.argv[1] if len(sys.argv) > 1 else "/dpc/kuin0170/u100066980/runs/dr_gamma"

files = sorted(glob.glob(os.path.join(run_dir, "fold*", "test_result.json")))
results = [json.load(open(f)) for f in files]
results.sort(key=lambda r: r["fold"])

if not results:
    print(f"No per-fold results found under {run_dir}/fold*/test_result.json")
    sys.exit(1)

accs = [r["test_acc"] for r in results]
maes = [r["test_mae"] for r in results]

out = os.path.join(run_dir, "final_results.csv")
with open(out, "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["fold", "test_acc", "test_mae"])
    for r in results:
        w.writerow([r["fold"], f"{r['test_acc']:.4f}", f"{r['test_mae']:.4f}"])
    w.writerow(["mean", f"{st.mean(accs):.4f}", f"{st.mean(maes):.4f}"])
    w.writerow(["std",
                f"{st.pstdev(accs):.4f}" if len(accs) > 1 else "0.0000",
                f"{st.pstdev(maes):.4f}" if len(maes) > 1 else "0.0000"])

print(f"Aggregated {len(results)}/10 folds -> {out}")
for r in results:
    print(f"  fold {r['fold']}: acc={r['test_acc']:.2f}%  mae={r['test_mae']:.4f}")
print(f"  MEAN acc={st.mean(accs):.2f}%  mae={st.mean(maes):.4f}"
      + (f"  (over {len(results)} folds)" if len(results) < 10 else ""))
