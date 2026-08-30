"""Full-canvas data path for MOSAIC.

Unlike OPTIC-C's tiled loader, this module preserves one continuous retinal
canvas.  After isolating the dominant fundus field, every acquisition is
canonicalized to the same square coordinates and receives the same centred
elliptical proof-valid mask.  Consequently camera aspect ratio and border
width cannot leak through the number of valid proof cells.  The mask is
preprocessing metadata, not a learned or manually annotated lesion mask.
"""

from __future__ import annotations

import os
import random
from collections import defaultdict
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF

from .dataloaders import StratifiedBatchSampler, _IMAGENET_MEAN, _IMAGENET_STD
from utils.spatial_mask import centered_ellipse_mask


Item = Tuple[str, int]
_IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")
MOSAIC_PREPROCESSING_VERSION = "canonical-square-fixed-ellipse-v1"


def _resolve_image(directory: str | Path, image_id: str) -> str:
    """Resolve an image identifier with or without a file extension."""
    directory = Path(directory)
    exact = directory / image_id
    if exact.is_file():
        return str(exact)
    stem = Path(image_id).stem
    for suffix in _IMAGE_EXTENSIONS:
        candidate = directory / f"{stem}{suffix}"
        if candidate.is_file():
            return str(candidate)
    raise FileNotFoundError(f"No image found for {image_id!r} in {directory}")


def load_aptos_items(root: str, csv_path: Optional[str] = None) -> List[Item]:
    """Load APTOS 2019 training images and ordinal grades."""
    csv_path = csv_path or os.path.join(root, "train.csv")
    frame = pd.read_csv(csv_path)
    required = {"id_code", "diagnosis"}
    if not required.issubset(frame.columns):
        raise ValueError(f"APTOS CSV must contain {sorted(required)}, got {list(frame.columns)}")
    image_dir = os.path.join(root, "train_images")
    return [
        (_resolve_image(image_dir, str(row.id_code)), int(row.diagnosis))
        for row in frame.itertuples(index=False)
    ]


def load_eyepacs_items(root: str, csv_path: Optional[str] = None) -> List[Item]:
    """Load the Kaggle/EyePACS DR image-level training set."""
    csv_path = csv_path or os.path.join(root, "trainLabels.csv")
    frame = pd.read_csv(csv_path)
    required = {"image", "level"}
    if not required.issubset(frame.columns):
        raise ValueError(f"DR CSV must contain {sorted(required)}, got {list(frame.columns)}")
    image_dir = os.path.join(root, "train")
    return [
        (_resolve_image(image_dir, str(row.image)), int(row.level))
        for row in frame.itertuples(index=False)
    ]


def _longest_true_run(values: np.ndarray) -> tuple[int, int] | None:
    """Return the longest half-open run, preferring the image-centred tie."""

    values = np.asarray(values, dtype=bool).reshape(-1)
    padded = np.pad(values.astype(np.int8), (1, 1))
    changes = np.diff(padded)
    starts = np.flatnonzero(changes == 1)
    stops = np.flatnonzero(changes == -1)
    if starts.size == 0:
        return None
    lengths = stops - starts
    max_length = int(lengths.max())
    candidates = np.flatnonzero(lengths == max_length)
    axis_centre = values.size / 2.0
    winner = min(
        candidates.tolist(),
        key=lambda index: abs((starts[index] + stops[index]) / 2.0 - axis_centre),
    )
    return int(starts[winner]), int(stops[winner])


def _dominant_field_bounds(
    image: Image.Image,
    *,
    threshold: int = 10,
    min_axis_fraction: float = 0.01,
) -> tuple[int, int, int, int] | None:
    """Locate the dominant retinal field while rejecting small bright islands.

    Fundus photographs sometimes contain disconnected camera labels or border
    reflections.  A bounding box over *all* non-black pixels would turn those
    artifacts into valid proof cells.  The retinal field, unlike such islands,
    forms the longest densely-supported run along both image axes.  Selecting
    those runs is deterministic, dependency-free, and linear in the pixel count.
    """

    if not 0.0 <= min_axis_fraction <= 1.0:
        raise ValueError("min_axis_fraction must lie in [0, 1]")
    array = np.asarray(image.convert("L"))
    foreground = array > threshold
    if foreground.sum() < max(64, int(0.01 * foreground.size)):
        return None
    row_run = _longest_true_run(
        foreground.mean(axis=1) >= min_axis_fraction
    )
    col_run = _longest_true_run(
        foreground.mean(axis=0) >= min_axis_fraction
    )
    if row_run is None or col_run is None:
        return None
    top, bottom = row_run
    left, right = col_run
    if bottom <= top or right <= left:
        return None
    return left, top, right, bottom


def _tight_field_crop(image: Image.Image, threshold: int = 10) -> Image.Image:
    """Remove black borders and disconnected acquisition artifacts."""

    bounds = _dominant_field_bounds(image, threshold=threshold)
    return image if bounds is None else image.crop(bounds)


def _canonical_square(
    image: Image.Image,
    size: int,
    interpolation=Image.Resampling.BILINEAR,
) -> Image.Image:
    """Map a tightly cropped fundus field to one canonical square canvas.

    This deliberate direct resize removes the acquisition-dependent aspect
    ratio and padding pattern.  The small anisotropic rescaling is preferable
    here to exposing camera geometry to a proof system whose valid-cell count
    must carry disease evidence only.
    """

    size = int(size)
    if size <= 0:
        raise ValueError(f"size must be positive, got {size}")
    return image.resize((size, size), interpolation)


def _fixed_field_mask(size: int | tuple[int, int]) -> Image.Image:
    """Return an image-independent centred elliptical proof-valid mask.

    In production ``size`` is square, so this is a centred circle inscribed in
    the canonical canvas.  It is filled deliberately: dark vessels, the fovea,
    and lesions remain valid retinal tissue.  Its geometry depends only on the
    configured output size, never on pixels, labels, camera type, or source
    aspect ratio.
    """

    if isinstance(size, int):
        width = height = int(size)
    else:
        width, height = (int(value) for value in size)
    if width <= 0 or height <= 0:
        raise ValueError(f"mask dimensions must be positive, got {(width, height)}")
    mask = centered_ellipse_mask(height, width)[0, 0].numpy().astype(np.uint8) * 255
    return Image.fromarray(mask, mode="L")


def _field_mask(image: Image.Image, threshold: int = 10) -> Image.Image:
    """Backward-compatible wrapper for the fixed, pixel-independent mask.

    ``threshold`` is retained only so older callers do not break; it is
    intentionally ignored.  Source pixels must never determine proof-valid
    support.
    """

    del threshold
    return _fixed_field_mask(image.size)


class MosaicFundusTransform:
    """Return an ImageNet-normalized canvas and its pixel-level valid mask."""

    def __init__(self, size: int = 896, augment: bool = False):
        self.size = int(size)
        if self.size <= 0:
            raise ValueError(f"size must be positive, got {self.size}")
        self.augment = bool(augment)
        self.preprocessing_version = MOSAIC_PREPROCESSING_VERSION
        self._canonical_mask = _fixed_field_mask(self.size)
        self.photometric = transforms.ColorJitter(
            brightness=0.20,
            contrast=0.20,
            saturation=0.10,
            hue=0.02,
        )
        self.normalize = transforms.Normalize(_IMAGENET_MEAN, _IMAGENET_STD)

    def canonical_valid_mask(self) -> torch.Tensor:
        """Return a fresh copy of the exact evaluation-time proof mask."""

        return TF.pil_to_tensor(self._canonical_mask.copy()).to(dtype=torch.bool)

    def __call__(self, image: Image.Image) -> tuple[torch.Tensor, torch.Tensor]:
        image = _tight_field_crop(image.convert("RGB"))
        image = _canonical_square(image, self.size)
        mask = self._canonical_mask.copy()

        if self.augment:
            if random.random() < 0.5:
                image = TF.hflip(image)
                mask = TF.hflip(mask)
            if random.random() < 0.5:
                image = TF.vflip(image)
                mask = TF.vflip(mask)
            angle = random.uniform(-30.0, 30.0)
            image = TF.rotate(
                image,
                angle,
                interpolation=InterpolationMode.BILINEAR,
                fill=[0, 0, 0],
            )
            mask = TF.rotate(
                mask,
                angle,
                interpolation=InterpolationMode.NEAREST,
                fill=0,
            )
            image = self.photometric(image)

        tensor = self.normalize(TF.to_tensor(image))
        valid = TF.pil_to_tensor(mask).to(dtype=torch.bool)
        return tensor, valid


class MosaicImageDataset(Dataset):
    """Image-level severity data with stable sample indices for certificates."""

    def __init__(self, items: Sequence[Item], transform: MosaicFundusTransform):
        self.items = list(items)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int):
        path, label = self.items[index]
        with Image.open(path) as source:
            image = source.convert("RGB")
        tensor, valid_mask = self.transform(image)
        return tensor, valid_mask, torch.tensor(label, dtype=torch.long), index


def stratified_folds(items: Sequence[Item], n_folds: int, seed: int) -> List[List[Item]]:
    """Deterministic image-level stratified folds (APTOS has no patient IDs)."""
    groups: dict[int, list[Item]] = defaultdict(list)
    for item in items:
        groups[int(item[1])].append(item)
    rng = random.Random(seed)
    folds: List[List[Item]] = [[] for _ in range(n_folds)]
    for group in groups.values():
        group = group.copy()
        rng.shuffle(group)
        for index, item in enumerate(group):
            folds[index % n_folds].append(item)
    return folds


def aptos_fold(
    items: Sequence[Item],
    fold: int,
    n_folds: int = 5,
    val_fraction: float = 0.1,
    seed: int = 42,
) -> tuple[List[Item], List[Item], List[Item]]:
    """Return train/validation/test lists without using the test fold for tuning."""
    folds = stratified_folds(items, n_folds=n_folds, seed=seed)
    if fold < 0 or fold >= n_folds:
        raise ValueError(f"fold must be in [0, {n_folds - 1}], got {fold}")
    test = folds[fold]
    pool = [item for index, split in enumerate(folds) if index != fold for item in split]

    groups: dict[int, list[Item]] = defaultdict(list)
    for item in pool:
        groups[int(item[1])].append(item)
    rng = random.Random(seed + fold)
    train: List[Item] = []
    val: List[Item] = []
    for group in groups.values():
        rng.shuffle(group)
        count = max(1, int(round(len(group) * val_fraction)))
        val.extend(group[:count])
        train.extend(group[count:])
    return train, val, test


def _eyepacs_patient_id(item: Item) -> str:
    """EyePACS names paired eyes as ``<patient>_left/right``."""
    return Path(item[0]).stem.rsplit("_", 1)[0]


def eyepacs_fold(
    items: Sequence[Item],
    fold: int,
    n_folds: int = 10,
    val_fraction: float = 0.1,
    seed: int = 42,
) -> tuple[List[Item], List[Item], List[Item]]:
    """Patient-disjoint EyePACS train/validation/test partition.

    Both eyes remain together not only across the outer test split but also
    across the inner validation split used for checkpoint selection.
    """
    patients: dict[str, list[Item]] = defaultdict(list)
    for item in items:
        patients[_eyepacs_patient_id(item)].append(item)

    def representative(patient_items: Sequence[Item]) -> int:
        labels = [int(label) for _, label in patient_items]
        # Stable tie break toward the more severe grade.
        return max(set(labels), key=lambda label: (labels.count(label), label))

    by_grade: dict[int, list[str]] = defaultdict(list)
    for patient, patient_items in patients.items():
        by_grade[representative(patient_items)].append(patient)
    rng = random.Random(seed)
    fold_patients: list[list[str]] = [[] for _ in range(n_folds)]
    for group in by_grade.values():
        rng.shuffle(group)
        for index, patient in enumerate(group):
            fold_patients[index % n_folds].append(patient)
    if fold < 0 or fold >= n_folds:
        raise ValueError(f"fold must be in [0, {n_folds - 1}], got {fold}")

    test_patients = set(fold_patients[fold])
    pool_patients = [patient for patient in patients if patient not in test_patients]
    pool_by_grade: dict[int, list[str]] = defaultdict(list)
    for patient in pool_patients:
        pool_by_grade[representative(patients[patient])].append(patient)
    validation_patients: set[str] = set()
    inner_rng = random.Random(seed + fold)
    for group in pool_by_grade.values():
        inner_rng.shuffle(group)
        count = max(1, int(round(len(group) * val_fraction)))
        validation_patients.update(group[:count])
    train_patients = set(pool_patients) - validation_patients

    def expand(patient_ids: set[str]) -> List[Item]:
        return [item for patient in sorted(patient_ids) for item in patients[patient]]

    return expand(train_patients), expand(validation_patients), expand(test_patients)


def make_mosaic_loaders(
    train_items: Sequence[Item],
    val_items: Sequence[Item],
    test_items: Sequence[Item],
    *,
    image_size: int,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
    seed: int,
    stratified: bool = False,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Build full-canvas loaders with geometry-consistent masks."""
    train_dataset = MosaicImageDataset(
        train_items, MosaicFundusTransform(image_size, augment=True)
    )
    eval_transform = MosaicFundusTransform(image_size, augment=False)
    val_dataset = MosaicImageDataset(val_items, eval_transform)
    test_dataset = MosaicImageDataset(test_items, eval_transform)
    common = {
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        # Recreate workers at each epoch so restoring the checkpointed torch
        # RNG also recreates the worker seeds and augmentation stream.  A
        # persistent worker keeps private Python RNG state that PyTorch cannot
        # serialize, making a resumed run silently non-reproducible.
        "persistent_workers": False,
    }
    if num_workers > 0:
        common["prefetch_factor"] = 2

    if stratified:
        class_count = len({int(label) for _, label in train_items})
        if batch_size < class_count:
            raise ValueError(
                "stratified batches require batch_size >= number of training "
                f"classes ({batch_size} < {class_count}); otherwise the sampler "
                "would silently emit batches larger than requested"
            )
        sampler = StratifiedBatchSampler(
            [label for _, label in train_items],
            batch_size=batch_size,
            drop_last=True,
            seed=seed,
        )
        train_loader = DataLoader(train_dataset, batch_sampler=sampler, **common)
    else:
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            drop_last=True,
            **common,
        )
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, **common)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, **common)
    return train_loader, val_loader, test_loader


def class_histogram(items: Iterable[Item], n_classes: int = 5) -> list[int]:
    counts = [0] * n_classes
    for _, label in items:
        counts[int(label)] += 1
    return counts


__all__ = [
    "Item",
    "MOSAIC_PREPROCESSING_VERSION",
    "MosaicFundusTransform",
    "MosaicImageDataset",
    "load_aptos_items",
    "load_eyepacs_items",
    "aptos_fold",
    "eyepacs_fold",
    "make_mosaic_loaders",
    "class_histogram",
]
