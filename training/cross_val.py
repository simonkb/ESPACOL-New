from __future__ import annotations

"""
Subject-independent cross-validation utilities.

BUSI: 5-fold stratified CV on image level (no explicit patient IDs).
DR:   10-fold subject-independent CV; left and right eyes of the same patient
      are kept in the same fold (patient ID extracted from filename stem).

Both validators return fold splits as lists of (path, label) tuples,
ready for use with ImageLabelDataset.
"""

import random
from collections import defaultdict
from typing import List, Tuple

Item = Tuple[str, int]   # (image_path, integer_label)


# ─────────────────────────────────────────────────────────────────────────────
# Generic stratified k-fold helpers
# ─────────────────────────────────────────────────────────────────────────────

def _stratified_kfold(
    items: List[Item],
    n_folds: int,
    seed: int = 42,
) -> List[List[Item]]:
    """
    Stratified k-fold split on individual items (BUSI).

    Each class is split into k chunks independently to maintain class balance
    across folds.  Returns list of k fold-item-lists.
    """
    rng = random.Random(seed)

    by_class: dict[int, List[Item]] = defaultdict(list)
    for item in items:
        by_class[item[1]].append(item)

    folds: List[List[Item]] = [[] for _ in range(n_folds)]
    for cls_items in by_class.values():
        cls_shuffled = cls_items.copy()
        rng.shuffle(cls_shuffled)
        # Round-robin assignment to folds
        for i, it in enumerate(cls_shuffled):
            folds[i % n_folds].append(it)

    return folds


def _patient_stratified_kfold(
    items: List[Item],
    n_folds: int,
    get_patient_id,      # callable: item -> str patient ID
    seed: int = 42,
) -> List[List[Item]]:
    """
    Patient-level stratified k-fold (DR).

    Groups items by patient, then assigns patients to folds using stratified
    sampling on the majority class of each patient (so grade distribution is
    roughly balanced across folds).
    """
    rng = random.Random(seed)

    # Group by patient ID
    patient_items: dict[str, List[Item]] = defaultdict(list)
    for item in items:
        pid = get_patient_id(item)
        patient_items[pid].append(item)

    # For each patient, determine representative label (most common label)
    patients = list(patient_items.keys())
    patient_labels: dict[str, int] = {}
    for pid in patients:
        labels = [it[1] for it in patient_items[pid]]
        patient_labels[pid] = max(set(labels), key=labels.count)

    # Stratified assignment of patients to folds
    by_label: dict[int, List[str]] = defaultdict(list)
    for pid in patients:
        by_label[patient_labels[pid]].append(pid)

    folds_patients: List[List[str]] = [[] for _ in range(n_folds)]
    for label_patients in by_label.values():
        shuffled = label_patients.copy()
        rng.shuffle(shuffled)
        for i, pid in enumerate(shuffled):
            folds_patients[i % n_folds].append(pid)

    # Convert patient folds -> item folds
    folds: List[List[Item]] = []
    for fold_pids in folds_patients:
        fold_items = []
        for pid in fold_pids:
            fold_items.extend(patient_items[pid])
        folds.append(fold_items)

    return folds


def _split_train_val(
    train_items: List[Item],
    val_fraction: float,
    seed: int = 42,
) -> Tuple[List[Item], List[Item]]:
    """Split training items into train/val with stratification."""
    rng = random.Random(seed)

    by_class: dict[int, List[Item]] = defaultdict(list)
    for item in train_items:
        by_class[item[1]].append(item)

    train_split, val_split = [], []
    for cls_items in by_class.values():
        shuffled = cls_items.copy()
        rng.shuffle(shuffled)
        n_val = max(1, int(len(shuffled) * val_fraction))
        val_split.extend(shuffled[:n_val])
        train_split.extend(shuffled[n_val:])

    return train_split, val_split


# ─────────────────────────────────────────────────────────────────────────────
# BUSI cross-validator
# ─────────────────────────────────────────────────────────────────────────────

class BUSICrossValidator:
    """
    5-fold stratified cross-validation for BUSI.

    Paper: "Following [12,9], we perform 5-fold subject-independent cross-validation."
    """

    def __init__(
        self,
        all_items: List[Item],
        n_folds: int = 5,
        val_fraction: float = 0.1,
        seed: int = 42,
    ):
        self.all_items = all_items
        self.n_folds = n_folds
        self.val_fraction = val_fraction
        self.seed = seed
        self._folds = _stratified_kfold(all_items, n_folds, seed)

    def __len__(self) -> int:
        return self.n_folds

    def get_fold(self, fold_idx: int) -> Tuple[List[Item], List[Item], List[Item]]:
        """Return (train_items, val_items, test_items) for fold *fold_idx*."""
        test_items = self._folds[fold_idx]
        train_items_raw = []
        for i, fold in enumerate(self._folds):
            if i != fold_idx:
                train_items_raw.extend(fold)

        train_items, val_items = _split_train_val(
            train_items_raw, self.val_fraction, seed=self.seed + fold_idx
        )
        return train_items, val_items, test_items


# ─────────────────────────────────────────────────────────────────────────────
# DR cross-validator
# ─────────────────────────────────────────────────────────────────────────────

def _dr_patient_id(item: Item) -> str:
    """Extract patient ID from DR image path.

    Example: '.../train/10_left.jpeg' -> '10'
    """
    import os
    stem = os.path.splitext(os.path.basename(item[0]))[0]  # '10_left'
    return stem.rsplit("_", 1)[0]                            # '10'


class DRCrossValidator:
    """
    10-fold subject-independent cross-validation for DR.

    Paper: "Following [18,3,12], we use 10-fold subject-independent
    cross-validation for evaluation."

    Left and right eye images of the same patient are always in the same fold.
    """

    def __init__(
        self,
        all_items: List[Item],
        n_folds: int = 10,
        val_fraction: float = 0.1,
        seed: int = 42,
    ):
        self.all_items = all_items
        self.n_folds = n_folds
        self.val_fraction = val_fraction
        self.seed = seed
        self._folds = _patient_stratified_kfold(
            all_items, n_folds, _dr_patient_id, seed
        )

    def __len__(self) -> int:
        return self.n_folds

    def get_fold(self, fold_idx: int) -> Tuple[List[Item], List[Item], List[Item]]:
        """Return (train_items, val_items, test_items) for fold *fold_idx*."""
        test_items = self._folds[fold_idx]
        train_items_raw = []
        for i, fold in enumerate(self._folds):
            if i != fold_idx:
                train_items_raw.extend(fold)

        train_items, val_items = _split_train_val(
            train_items_raw, self.val_fraction, seed=self.seed + fold_idx
        )
        return train_items, val_items, test_items
