"""
Create a stratified ~4K subset of the DR training dataset for fast sweeps.

Sampling is subject-independent: images named '10_left.jpeg' and '10_right.jpeg'
share subject ID '10' and are always kept together (never split).

Subject grade = max grade across both eyes (prevents data leakage across splits).

Output: Datasets/DR/trainLabels_4k.csv  (same columns: image, level)

Usage:
    python create_dr_subset.py [--csv Datasets/DR/trainLabels.csv] [--out Datasets/DR/trainLabels_4k.csv] [--target 4000]
"""

import argparse
import random
from collections import defaultdict
from pathlib import Path

import pandas as pd


TARGET_COUNTS = {0: 2940, 1: 278, 2: 603, 3: 99, 4: 81}  # ~4001 total


def extract_subject_id(image_name: str) -> str:
    """'10_left' or '10_left.jpeg' -> '10'"""
    stem = Path(image_name).stem          # strip extension if present
    return stem.rsplit("_", 1)[0]         # drop '_left' / '_right'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="Datasets/DR/trainLabels.csv")
    parser.add_argument("--out", default="Datasets/DR/trainLabels_4k.csv")
    parser.add_argument("--target", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    print(f"Loaded {len(df)} rows from {args.csv}")
    print(f"Original class distribution:\n{df['level'].value_counts().sort_index().to_dict()}\n")

    # Group images by subject
    subject_images: dict[str, list[tuple[str, int]]] = defaultdict(list)
    for _, row in df.iterrows():
        sid = extract_subject_id(str(row["image"]))
        subject_images[sid].append((str(row["image"]), int(row["level"])))

    # Assign each subject its max grade
    subject_grade: dict[str, int] = {
        sid: max(lbl for _, lbl in imgs)
        for sid, imgs in subject_images.items()
    }

    # Group subjects by their assigned grade
    grade_subjects: dict[int, list[str]] = defaultdict(list)
    for sid, grade in subject_grade.items():
        grade_subjects[grade].append(sid)

    print("Subjects per grade (before sampling):")
    for g in sorted(grade_subjects):
        n_imgs = sum(len(subject_images[s]) for s in grade_subjects[g])
        print(f"  Grade {g}: {len(grade_subjects[g])} subjects, {n_imgs} images")
    print()

    # Compute per-grade image targets proportional to original distribution
    total_original = len(df)
    class_fractions = df["level"].value_counts(normalize=True).sort_index().to_dict()
    target_total = args.target
    image_targets = {
        g: max(1, round(class_fractions.get(g, 0) * target_total))
        for g in sorted(grade_subjects)
    }
    print(f"Image targets per grade (sum={sum(image_targets.values())}):")
    for g, t in sorted(image_targets.items()):
        print(f"  Grade {g}: {t}")
    print()

    rng = random.Random(args.seed)

    selected_images: list[tuple[str, int]] = []

    for grade in sorted(grade_subjects):
        subjects = grade_subjects[grade].copy()
        rng.shuffle(subjects)

        target_imgs = image_targets[grade]
        chosen: list[str] = []
        accumulated = 0

        for sid in subjects:
            n = len(subject_images[sid])
            chosen.append(sid)
            accumulated += n
            if accumulated >= target_imgs:
                break

        for sid in chosen:
            selected_images.extend(subject_images[sid])

    # Build output DataFrame
    out_df = pd.DataFrame(selected_images, columns=["image", "level"])
    out_df = out_df.sort_values("image").reset_index(drop=True)

    print(f"Subset size: {len(out_df)} images")
    print(f"Subset class distribution:\n{out_df['level'].value_counts().sort_index().to_dict()}\n")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
