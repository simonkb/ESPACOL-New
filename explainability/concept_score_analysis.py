"""
Concept score analysis: verify clinical ordering of tile_concept_scores.

For each DR grade, compute the mean concept activation score across all images
in that grade from the Kaggle DR inference (or IDRiD inference).

Expected clinical ordering (from grade_concept_targets in concept_prototype.py):
  Grade 0: all concepts ~0
  Grade 1: Microaneurysms high, rest ~0
  Grade 2: Microaneurysms, Haemorrhages, Hard Exudates
  Grade 3: Haemorrhages dominant, + Cotton Wool Spots, Venous Beading, IRMA
  Grade 4: Neovascularization, Preretinal/Vitreous Haemorrhage high

Usage:
  python explainability/concept_score_analysis.py \
      --inference_dir explainability/idrid_outputs \
      --output_csv    explainability/concept_scores_per_grade.csv

Or for Kaggle DR (if you run a separate inference script):
  python explainability/concept_score_analysis.py \
      --inference_dir explainability/dr_outputs \
      --output_csv    explainability/concept_scores_per_grade.csv
"""

from __future__ import annotations

import argparse
import csv
import os
import sys

import numpy as np

CONCEPT_NAMES = [
    "Microaneurysms",
    "Dot/Blot Haemorrhages",
    "Hard Exudates",
    "Cotton Wool Spots",
    "Venous Beading",
    "IRMA",
    "Neovascularization",
    "Preretinal Haemorrhage",
    "Vitreous Haemorrhage",
]

GRADE_NAMES = ["Grade 0 (No DR)", "Grade 1 (Mild)", "Grade 2 (Moderate)",
               "Grade 3 (Severe)", "Grade 4 (PDR)"]


def evaluate(args):
    npz_files = sorted(f for f in os.listdir(args.inference_dir) if f.endswith("_outputs.npz"))
    if not npz_files:
        raise FileNotFoundError(f"No outputs in {args.inference_dir}")

    # Aggregate: grade -> list of mean concept scores per image
    grade_scores = {g: [] for g in range(5)}

    for fname in npz_files:
        data = np.load(os.path.join(args.inference_dir, fname), allow_pickle=True)
        labels = data["labels"]                         # (N,)
        tcs = data["tile_concept_scores"]               # (N, T, C)

        # Mean over tiles for each image -> (N, C)
        mean_scores = tcs.mean(axis=1)

        for i in range(len(labels)):
            g = int(labels[i])
            if 0 <= g <= 4:
                grade_scores[g].append(mean_scores[i])

    print("\n=== Concept Score per Grade (mean ± std) ===")
    print(f"{'Concept':<30}", end="")
    for g in range(5):
        print(f"  Gr{g} (N={len(grade_scores[g]):3d})", end="")
    print()
    print("-" * (30 + 5 * 18))

    rows = []
    for c_idx, c_name in enumerate(CONCEPT_NAMES):
        row = {"concept": c_name}
        print(f"{c_name:<30}", end="")
        for g in range(5):
            if grade_scores[g]:
                scores = np.array(grade_scores[g])[:, c_idx]
                m, s = scores.mean(), scores.std()
                print(f"  {m:.3f}±{s:.3f}", end="")
                row[f"grade{g}_mean"] = f"{m:.4f}"
                row[f"grade{g}_std"] = f"{s:.4f}"
            else:
                print(f"  {'N/A':>9}", end="")
                row[f"grade{g}_mean"] = "N/A"
                row[f"grade{g}_std"] = "N/A"
        print()
        rows.append(row)

    if args.output_csv:
        os.makedirs(os.path.dirname(args.output_csv) or ".", exist_ok=True)
        fieldnames = ["concept"] + [f"grade{g}_{s}" for g in range(5) for s in ("mean", "std")]
        with open(args.output_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nSaved to {args.output_csv}")

    # Check clinical ordering: grade 1 should rank Microaneurysms highest
    for g in range(1, 5):
        if not grade_scores[g]:
            continue
        scores = np.array(grade_scores[g]).mean(axis=0)
        top_concept = CONCEPT_NAMES[int(np.argmax(scores))]
        print(f"Grade {g} top concept: {top_concept}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--inference_dir", default="explainability/idrid_outputs")
    parser.add_argument("--output_csv", default="explainability/concept_scores_per_grade.csv")
    args = parser.parse_args()
    evaluate(args)
