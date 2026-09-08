"""Dataset adapters for ORIGIN's full-canvas evidence field.

APTOS and EyePACS deliberately delegate to :mod:`Datasets.mosaic_data` for
both preprocessing and fold construction.  Those functions define the
already-audited split identities, including EyePACS patient grouping; ORIGIN
must not fork that logic merely to rename it.
"""

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Iterable, Mapping, Optional, Sequence

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF

from .dataloaders import StratifiedBatchSampler, _IMAGENET_MEAN, _IMAGENET_STD
from .mosaic_data import (
    Item,
    MOSAIC_PREPROCESSING_VERSION,
    MosaicFundusTransform,
    MosaicImageDataset,
    aptos_fold as _audited_aptos_fold,
    class_histogram,
    eyepacs_fold as _audited_eyepacs_fold,
    load_aptos_items as _load_audited_aptos_items,
    load_eyepacs_items as _load_audited_eyepacs_items,
    make_mosaic_loaders,
)


ORIGIN_PREPROCESSING_VERSION = MOSAIC_PREPROCESSING_VERSION
_IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")
BUSI_CLASS_TO_LABEL = {"normal": 0, "benign": 1, "malignant": 2}


class OriginFundusTransform(MosaicFundusTransform):
    """Audited canonical fundus transform with ORIGIN's 640-pixel default."""

    def __init__(self, size: int = 640, augment: bool = False):
        super().__init__(size=size, augment=augment)
        self.preprocessing_version = ORIGIN_PREPROCESSING_VERSION


class OriginGenericTransform:
    """Canonical full-canvas transform for non-fundus severity images.

    Unlike fundus images, a BUSI acquisition has no fixed circular retinal
    support.  The complete resized frame is therefore valid evidence.  When a
    geometric augmentation introduces padding, the same operation is applied
    to the mask so padded pixels cannot become evidence sites.
    """

    def __init__(self, size: int = 640, augment: bool = False):
        self.size = int(size)
        if self.size <= 0:
            raise ValueError(f"size must be positive, got {self.size}")
        self.augment = bool(augment)
        self.preprocessing_version = "origin-generic-square-full-mask-v1"
        self.photometric = transforms.ColorJitter(
            brightness=0.15,
            contrast=0.15,
            saturation=0.05,
            hue=0.01,
        )
        self.normalize = transforms.Normalize(_IMAGENET_MEAN, _IMAGENET_STD)

    def canonical_valid_mask(self) -> torch.Tensor:
        return torch.ones(1, self.size, self.size, dtype=torch.bool)

    def __call__(self, image: Image.Image) -> tuple[torch.Tensor, torch.Tensor]:
        image = image.convert("RGB").resize(
            (self.size, self.size), Image.Resampling.BILINEAR
        )
        mask = Image.new("L", (self.size, self.size), color=255)
        if self.augment:
            if random.random() < 0.5:
                image = TF.hflip(image)
                mask = TF.hflip(mask)
            if random.random() < 0.5:
                image = TF.vflip(image)
                mask = TF.vflip(mask)
            angle = random.uniform(-20.0, 20.0)
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
        return (
            self.normalize(TF.to_tensor(image)),
            TF.pil_to_tensor(mask).to(dtype=torch.bool),
        )


# Stable semantic name while retaining the audited sample contract:
# (image, pixel_valid_mask, label, stable_index).
OriginImageDataset = MosaicImageDataset


def load_aptos_items(root: str, csv_path: Optional[str] = None) -> list[Item]:
    """Use the exact APTOS loader already used by the MOSAIC baselines."""

    return _load_audited_aptos_items(root, csv_path)


def load_eyepacs_items(root: str, csv_path: Optional[str] = None) -> list[Item]:
    """Use the exact EyePACS loader already used by the MOSAIC baselines."""

    return _load_audited_eyepacs_items(root, csv_path)


def aptos_fold(
    items: Sequence[Item],
    fold: int,
    n_folds: int = 5,
    val_fraction: float = 0.1,
    seed: int = 42,
) -> tuple[list[Item], list[Item], list[Item]]:
    return _audited_aptos_fold(items, fold, n_folds, val_fraction, seed)


def eyepacs_fold(
    items: Sequence[Item],
    fold: int,
    n_folds: int = 10,
    val_fraction: float = 0.1,
    seed: int = 42,
) -> tuple[list[Item], list[Item], list[Item]]:
    return _audited_eyepacs_fold(items, fold, n_folds, val_fraction, seed)


def load_folder_items(
    root: str | Path,
    class_to_label: Mapping[str, int],
    *,
    exclude_name_substrings: Sequence[str] = (),
) -> list[Item]:
    """Load one ordinal class per folder without interpreting mask files."""

    root = Path(root)
    excluded = tuple(value.lower() for value in exclude_name_substrings)
    items: list[Item] = []
    for class_name, label in sorted(class_to_label.items(), key=lambda pair: pair[1]):
        class_dir = root / class_name
        if not class_dir.is_dir():
            continue
        for path in sorted(class_dir.iterdir()):
            lower_name = path.name.lower()
            if (
                path.is_file()
                and path.suffix.lower() in _IMAGE_EXTENSIONS
                and not any(fragment in lower_name for fragment in excluded)
            ):
                items.append((str(path), int(label)))
    if not items:
        raise RuntimeError(
            f"no class images found under {root}; expected folders "
            f"{sorted(class_to_label)}"
        )
    validate_ordinal_labels(items)
    return items


def load_busi_items(root: str | Path) -> list[Item]:
    """Load BUSI images; segmentation masks are intentionally excluded."""

    return load_folder_items(
        root,
        BUSI_CLASS_TO_LABEL,
        exclude_name_substrings=("mask",),
    )


def _resolve_csv_image(image_root: Path, identifier: str) -> str:
    path = Path(identifier)
    candidates = [path] if path.is_absolute() else [image_root / path]
    if not path.suffix:
        candidates.extend((image_root / path).with_suffix(suffix) for suffix in _IMAGE_EXTENSIONS)
    for candidate in candidates:
        if candidate.is_file():
            return str(candidate)
    raise FileNotFoundError(f"could not resolve image {identifier!r} under {image_root}")


def load_csv_items(
    root: str | Path,
    csv_path: str | Path,
    *,
    image_column: str,
    label_column: str,
    image_dir: Optional[str | Path] = None,
) -> list[Item]:
    """Generic ordinal dataset adapter with explicit, auditable columns."""

    root = Path(root)
    csv_path = Path(csv_path)
    if not csv_path.is_absolute() and not csv_path.is_file():
        csv_path = root / csv_path
    frame = pd.read_csv(csv_path)
    required = {image_column, label_column}
    if not required.issubset(frame.columns):
        raise ValueError(
            f"CSV must contain {sorted(required)}, got {list(frame.columns)}"
        )
    image_root = Path(image_dir) if image_dir is not None else root
    if not image_root.is_absolute():
        image_root = root / image_root
    items = [
        (
            _resolve_csv_image(image_root, str(row[image_column])),
            int(row[label_column]),
        )
        for _, row in frame.iterrows()
    ]
    validate_ordinal_labels(items)
    return items


def validate_ordinal_labels(items: Sequence[Item], n_classes: Optional[int] = None) -> int:
    """Require contiguous integer severity labels and return their class count."""

    if not items:
        raise ValueError("the dataset is empty")
    labels = sorted({int(label) for _, label in items})
    inferred = labels[-1] + 1
    expected = list(range(inferred))
    if labels != expected:
        raise ValueError(
            f"ordinal labels must be contiguous from zero, got {labels}"
        )
    if n_classes is not None and inferred != int(n_classes):
        raise ValueError(
            f"dataset contains {inferred} classes but configuration requests {n_classes}"
        )
    return inferred


def image_stratified_fold(
    items: Sequence[Item],
    fold: int,
    n_folds: int = 5,
    val_fraction: float = 0.1,
    seed: int = 42,
) -> tuple[list[Item], list[Item], list[Item]]:
    """Image-level split for datasets that expose no patient identifier."""

    return _audited_aptos_fold(items, fold, n_folds, val_fraction, seed)


def split_origin_items(
    dataset: str,
    items: Sequence[Item],
    fold: int,
    *,
    n_folds: int,
    val_fraction: float,
    seed: int,
) -> tuple[list[Item], list[Item], list[Item]]:
    dataset = dataset.lower()
    if dataset == "dr":
        return eyepacs_fold(items, fold, n_folds, val_fraction, seed)
    if dataset in {"aptos", "busi", "generic"}:
        return image_stratified_fold(items, fold, n_folds, val_fraction, seed)
    raise ValueError(f"unsupported ORIGIN dataset: {dataset!r}")


def make_origin_loaders(
    train_items: Sequence[Item],
    val_items: Sequence[Item],
    test_items: Sequence[Item],
    *,
    image_size: int,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
    seed: int,
    fundus: bool,
    stratified: bool = False,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Build ORIGIN loaders while preserving the shared four-field batch API."""

    if fundus:
        return make_mosaic_loaders(
            train_items,
            val_items,
            test_items,
            image_size=image_size,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            seed=seed,
            stratified=stratified,
        )

    train_dataset = OriginImageDataset(
        train_items, OriginGenericTransform(image_size, augment=True)
    )
    eval_transform = OriginGenericTransform(image_size, augment=False)
    val_dataset = OriginImageDataset(val_items, eval_transform)
    test_dataset = OriginImageDataset(test_items, eval_transform)
    common: dict[str, object] = {
        "num_workers": int(num_workers),
        "pin_memory": bool(pin_memory),
        "persistent_workers": False,
    }
    if num_workers > 0:
        common["prefetch_factor"] = 2
    if stratified:
        class_count = len({int(label) for _, label in train_items})
        if batch_size < class_count:
            raise ValueError(
                "stratified batches require batch_size >= number of classes"
            )
        batch_sampler = StratifiedBatchSampler(
            [int(label) for _, label in train_items],
            batch_size=batch_size,
            drop_last=True,
            seed=seed,
        )
        train_loader = DataLoader(train_dataset, batch_sampler=batch_sampler, **common)
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


def load_origin_items(
    dataset: str,
    root: str,
    *,
    labels_csv: Optional[str] = None,
    image_column: Optional[str] = None,
    label_column: Optional[str] = None,
    image_dir: Optional[str] = None,
) -> list[Item]:
    dataset = dataset.lower()
    if dataset == "aptos":
        return load_aptos_items(root, labels_csv)
    if dataset == "dr":
        return load_eyepacs_items(root, labels_csv)
    if dataset == "busi":
        if labels_csv is not None:
            raise ValueError("BUSI uses class folders; --labels_csv is not applicable")
        return load_busi_items(root)
    if dataset == "generic":
        if labels_csv is None or image_column is None or label_column is None:
            raise ValueError(
                "generic data requires --labels_csv, --image_column, and --label_column"
            )
        return load_csv_items(
            root,
            labels_csv,
            image_column=image_column,
            label_column=label_column,
            image_dir=image_dir,
        )
    raise ValueError(f"unsupported ORIGIN dataset: {dataset!r}")


def split_paths_are_disjoint(*splits: Iterable[Item]) -> bool:
    """Small integrity helper used by preflight checks and tests."""

    seen: set[str] = set()
    for split in splits:
        current = {str(Path(path).resolve()) for path, _ in split}
        if seen.intersection(current):
            return False
        seen.update(current)
    return True


__all__ = [
    "BUSI_CLASS_TO_LABEL",
    "Item",
    "ORIGIN_PREPROCESSING_VERSION",
    "OriginFundusTransform",
    "OriginGenericTransform",
    "OriginImageDataset",
    "aptos_fold",
    "class_histogram",
    "eyepacs_fold",
    "image_stratified_fold",
    "load_aptos_items",
    "load_busi_items",
    "load_csv_items",
    "load_eyepacs_items",
    "load_folder_items",
    "load_origin_items",
    "make_origin_loaders",
    "split_origin_items",
    "split_paths_are_disjoint",
    "validate_ordinal_labels",
]
