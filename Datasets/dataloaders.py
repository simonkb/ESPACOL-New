# PyTorch dataloaders consistent with the paper's described preprocessing:
# - Resize to 300x300
# - ToTensor + ImageNet normalization (mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
# - DR: labels from CSV (image, level), images found in train/ or test/
# - BUSI: labels from class folders (benign/malignant/normal), ignore mask files
# - Class-stratified batch sampling for stability of prototypes

from __future__ import annotations

import os
import random
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader, Sampler
from torchvision import transforms
from torchvision.transforms.functional import to_pil_image


# ----------------------------
# Common transforms
# ----------------------------
_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD  = [0.229, 0.224, 0.225]

# CLIP normalization used by BiomedCLIP / open_clip ViT models
_CLIP_MEAN = [0.48145466, 0.4578275,  0.40821073]
_CLIP_STD  = [0.26862954, 0.26130258, 0.27577711]


def build_transform(img_size: int = 224, use_clip_norm: bool = True) -> Callable:
    """Evaluation transform: resize + ToTensor + normalization.

    use_clip_norm=True  → CLIP normalization (for BiomedCLIP backbone).
    use_clip_norm=False → ImageNet normalization (for EfficientNet backbone).
    """
    mean = _CLIP_MEAN if use_clip_norm else _IMAGENET_MEAN
    std  = _CLIP_STD  if use_clip_norm else _IMAGENET_STD
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])


def build_train_transform(img_size: int = 224, use_clip_norm: bool = True) -> Callable:
    """Training transform: augmentation + resize + ToTensor + normalization.

    Augmentations are mild and clinically safe for ultrasound/fundus images:
    horizontal/vertical flips preserve diagnostic content, small rotations
    and slight brightness/contrast shifts simulate acquisition variability.

    use_clip_norm=True  → CLIP normalization (for BiomedCLIP backbone).
    use_clip_norm=False → ImageNet normalization (for EfficientNet backbone).
    """
    mean = _CLIP_MEAN if use_clip_norm else _IMAGENET_MEAN
    std  = _CLIP_STD  if use_clip_norm else _IMAGENET_STD
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])


# ----------------------------
# Helpers
# ----------------------------
_IMG_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp")


def _find_existing_image(path_no_ext: str) -> str:
    """
    Given a path that may or may not include an extension, return an existing filepath.
    Tries:
      - exact path
      - path + common extensions
    Raises FileNotFoundError if none exist.
    """
    if os.path.isfile(path_no_ext):
        return path_no_ext

    root, ext = os.path.splitext(path_no_ext)
    if ext:  # had extension but file doesn't exist
        # try without extension, then with other extensions
        path_no_ext = root

    for e in _IMG_EXTS:
        p = path_no_ext + e
        if os.path.isfile(p):
            return p

    raise FileNotFoundError(f"Could not find image for base path: {path_no_ext}")


def _pil_loader(path: str, rgb: bool = True) -> Image.Image:
    with Image.open(path) as im:
        return im.convert("RGB") if rgb else im.copy()


# ----------------------------
# DR Dataset (CSV: image,level)
# ----------------------------
class DRDataset(Dataset):
    """
    Expected structure:
      Datasets/DR/
        train/
          *.png
        test/
          *.png
        trainLabels.csv   # columns: image, level
        testLabels.csv    # if you have it (optional)

    """
    def __init__(
        self,
        root_dir: str,
        split: str,
        csv_path: str,
        transform: Optional[Callable] = None,
        rgb: bool = True,
        cache: bool = False,
    ):
        """
        root_dir: path to DR dataset folder (e.g., 'Datasets/DR')
        split: 'train' or 'test'
        csv_path: path to CSV file (e.g., 'Datasets/DR/trainLabels.csv')
        cache: if True, preload all images into RAM as uint8 tensors (~9.5GB for 35K images)
        """
        if split not in ("train", "test"):
            raise ValueError("split must be 'train' or 'test'")

        self.root_dir = root_dir
        self.split = split
        self.split_dir = os.path.join(root_dir, split)
        self.csv_path = csv_path
        self.transform = transform or build_transform(300)
        self.rgb = rgb

        df = pd.read_csv(csv_path)
        if "image" not in df.columns or "level" not in df.columns:
            raise ValueError(f"CSV must contain columns ['image','level'], got {list(df.columns)}")

        self.items: List[Tuple[str, int]] = []
        for _, row in df.iterrows():
            image_id = str(row["image"])
            label = int(row["level"])
            # image_id may have extension or not
            candidate = os.path.join(self.split_dir, image_id)
            img_path = _find_existing_image(candidate)
            self.items.append((img_path, label))

        if len(self.items) == 0:
            raise RuntimeError(f"No items found for DR split={split} using {csv_path}")

        # Preload all images into RAM as resized uint8 tensors
        self._cached = cache
        self._cache: List[torch.Tensor] = []
        if cache:
            from concurrent.futures import ThreadPoolExecutor
            import multiprocessing

            resize = transforms.Resize((300, 300))
            n = len(self.items)

            def _load_one(idx: int) -> Tuple[int, torch.Tensor]:
                img_path = self.items[idx][0]
                img = _pil_loader(img_path, rgb=self.rgb)
                img = resize(img)
                t = torch.from_numpy(np.array(img)).permute(2, 0, 1)
                return idx, t

            n_workers = min(multiprocessing.cpu_count(), 16)
            self._cache = [None] * n  # type: ignore[list-item]
            done = 0
            with ThreadPoolExecutor(max_workers=n_workers) as pool:
                for idx, t in pool.map(_load_one, range(n)):
                    self._cache[idx] = t
                    done += 1
                    if done % 5000 == 0 or done == n:
                        print(f"  Caching images: {done}/{n} ({n_workers} threads)")

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        label = self.items[idx][1]
        if self._cached:
            # Convert uint8 -> float32 [0,1] (no disk I/O)
            x = self._cache[idx].float() / 255.0
        else:
            img_path = self.items[idx][0]
            img = _pil_loader(img_path, rgb=self.rgb)
            x = self.transform(img) if self.transform else img
        y = torch.tensor(label, dtype=torch.long)
        return x, y


# ----------------------------
# BUSI Dataset (folders: benign/malignant/normal)
# ----------------------------
class BUSIDataset(Dataset):
    """
    Expected structure:
      Datasets/BUSI/
        benign/
          benign (1).png
          benign (1)_mask.png
          ...
        malignant/
          ...
        normal/
          ...

    We IGNORE mask files: anything containing 'mask' in the filename (case-insensitive).
    Labels are inferred from folder name.
    """
    CLASS_TO_LABEL = {"normal": 0, "benign": 1, "malignant": 2}

    def __init__(
        self,
        root_dir: str,
        split: str = "all",
        transform: Optional[Callable] = None,
        rgb: bool = True,
        seed: int = 42,
        train_ratio: float = 0.8,
        cache: bool = False,
    ):
        """
        If split is 'all': returns the whole dataset.
        If split is 'train' or 'test': we create a deterministic split per class (stratified).
        cache: if True, preload all images into RAM as uint8 tensors.
        """
        if split not in ("all", "train", "test"):
            raise ValueError("split must be 'all', 'train', or 'test'")

        self.root_dir = root_dir
        self.split = split
        self.transform = transform or build_transform(300)
        self.rgb = rgb

        # gather (path,label)
        all_items: List[Tuple[str, int]] = []
        for cls_name, cls_label in self.CLASS_TO_LABEL.items():
            cls_dir = os.path.join(root_dir, cls_name)
            if not os.path.isdir(cls_dir):
                # allow missing folders, but warn via exception if none found overall
                continue

            for fn in os.listdir(cls_dir):
                if "mask" in fn.lower():
                    continue
                if not fn.lower().endswith(_IMG_EXTS):
                    continue
                all_items.append((os.path.join(cls_dir, fn), cls_label))

        if len(all_items) == 0:
            raise RuntimeError(f"No BUSI images found under {root_dir}. Expected folders: {list(self.CLASS_TO_LABEL)}")

        if split == "all":
            self.items = all_items
        else:
            # deterministic stratified split per class
            rng = random.Random(seed)
            by_class: Dict[int, List[str]] = {0: [], 1: [], 2: []}
            for p, y in all_items:
                by_class[y].append(p)

            split_items: List[Tuple[str, int]] = []
            for y, paths in by_class.items():
                rng.shuffle(paths)
                n_train = int(len(paths) * train_ratio)
                if split == "train":
                    chosen = paths[:n_train]
                else:
                    chosen = paths[n_train:]
                split_items.extend([(p, y) for p in chosen])

            self.items = split_items

        if len(self.items) == 0:
            raise RuntimeError(f"BUSI split '{split}' ended up empty. Check train_ratio and data.")

        # Preload all images into RAM as resized uint8 tensors
        self._cached = cache
        self._cache: List[torch.Tensor] = []
        if cache:
            from concurrent.futures import ThreadPoolExecutor
            import multiprocessing

            resize = transforms.Resize((300, 300))
            n = len(self.items)

            def _load_one(idx: int) -> Tuple[int, torch.Tensor]:
                img_path = self.items[idx][0]
                img = _pil_loader(img_path, rgb=self.rgb)
                img = resize(img)
                t = torch.from_numpy(np.array(img)).permute(2, 0, 1)
                return idx, t

            n_workers = min(multiprocessing.cpu_count(), 8)
            self._cache = [None] * n  # type: ignore[list-item]
            done = 0
            with ThreadPoolExecutor(max_workers=n_workers) as pool:
                for idx, t in pool.map(_load_one, range(n)):
                    self._cache[idx] = t
                    done += 1
            print(f"  BUSI cached: {n} images ({n_workers} threads)")

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        label = self.items[idx][1]
        if self._cached:
            x = self._cache[idx].float() / 255.0
        else:
            img_path = self.items[idx][0]
            img = _pil_loader(img_path, rgb=self.rgb)
            x = self.transform(img) if self.transform else img
        y = torch.tensor(label, dtype=torch.long)
        return x, y


# ----------------------------
# Augmented subset wrapper
# ----------------------------
class AugmentedSubset(Dataset):
    """Wraps a Subset and applies a transform to the input tensor (x) on the fly."""

    def __init__(self, subset, transform: Callable):
        self.subset = subset
        self.transform = transform

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        x, y = self.subset[idx]
        return self.transform(x), y


# ----------------------------
# Class-stratified batch sampler
# ----------------------------
class StratifiedBatchSampler(Sampler[List[int]]):
    """
    Produces batches with a (roughly) equal number of samples per class.

    Example:
      batch_size=24, n_classes=3 => 8 samples/class per batch.

    If a class runs out, it reshuffles and continues (oversampling minority classes),
    which is typically what we want when stabilizing prototype estimates.
    """
    def __init__(
        self,
        labels: Sequence[int],
        batch_size: int,
        drop_last: bool = True,
        seed: int = 42,
    ):
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        self.labels = list(map(int, labels))
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.seed = seed

        self.class_to_indices: Dict[int, List[int]] = {}
        for i, y in enumerate(self.labels):
            self.class_to_indices.setdefault(y, []).append(i)

        self.classes = sorted(self.class_to_indices.keys())
        self.n_classes = len(self.classes)
        if self.n_classes == 0:
            raise ValueError("No classes found in labels")

        # must be divisible for exact balance; if not, we do best-effort
        self.per_class = max(1, self.batch_size // self.n_classes)

    def __iter__(self):
        # Vary seed per epoch so batch composition changes across epochs.
        # A fixed seed would produce the identical 1317-batch sequence every epoch
        # (only augmentation would differ), limiting gradient diversity.
        self._epoch = getattr(self, "_epoch", -1) + 1
        rng = random.Random(self.seed + self._epoch)

        pools: Dict[int, List[int]] = {}
        for c in self.classes:
            idxs = self.class_to_indices[c].copy()
            rng.shuffle(idxs)
            pools[c] = idxs

        # approximate number of batches
        total = len(self.labels)
        n_batches = total // self.batch_size if self.drop_last else (total + self.batch_size - 1) // self.batch_size

        for _ in range(n_batches):
            batch: List[int] = []
            for c in self.classes:
                need = self.per_class
                while need > 0:
                    if len(pools[c]) == 0:
                        # refill (oversample)
                        refill = self.class_to_indices[c].copy()
                        rng.shuffle(refill)
                        pools[c] = refill
                    take = min(need, len(pools[c]))
                    batch.extend(pools[c][:take])
                    pools[c] = pools[c][take:]
                    need -= take

            # if batch_size not divisible by n_classes, top up with random picks
            while len(batch) < self.batch_size:
                c = rng.choice(self.classes)
                if len(pools[c]) == 0:
                    refill = self.class_to_indices[c].copy()
                    rng.shuffle(refill)
                    pools[c] = refill
                batch.append(pools[c].pop())

            rng.shuffle(batch)
            yield batch

    def __len__(self) -> int:
        total = len(self.labels)
        return total // self.batch_size if self.drop_last else (total + self.batch_size - 1) // self.batch_size


# ----------------------------
# Convenience builders
# ----------------------------
@dataclass
class LoaderConfig:
    batch_size: int = 24
    num_workers: int = 4
    pin_memory: bool = True
    img_size: int = 300
    seed: int = 42
    stratified: bool = True


def make_dr_loaders(
    dr_root: str,
    train_csv: str,
    test_csv: Optional[str],
    cfg: LoaderConfig,
) -> Tuple[DataLoader, Optional[DataLoader]]:
    tfm = build_transform(cfg.img_size)

    train_ds = DRDataset(root_dir=dr_root, split="train", csv_path=train_csv, transform=tfm)
    if cfg.stratified:
        labels = [y for _, y in train_ds.items]
        sampler = StratifiedBatchSampler(labels, batch_size=cfg.batch_size, drop_last=True, seed=cfg.seed)
        train_loader = DataLoader(
            train_ds,
            batch_sampler=sampler,
            num_workers=cfg.num_workers,
            pin_memory=cfg.pin_memory,
        )
    else:
        train_loader = DataLoader(
            train_ds,
            batch_size=cfg.batch_size,
            shuffle=True,
            num_workers=cfg.num_workers,
            pin_memory=cfg.pin_memory,
            drop_last=True,
        )

    test_loader = None
    if test_csv is not None:
        test_ds = DRDataset(root_dir=dr_root, split="test", csv_path=test_csv, transform=tfm)
        test_loader = DataLoader(
            test_ds,
            batch_size=cfg.batch_size,
            shuffle=False,
            num_workers=cfg.num_workers,
            pin_memory=cfg.pin_memory,
        )
    return train_loader, test_loader


def make_busi_loaders(
    busi_root: str,
    cfg: LoaderConfig,
    train_ratio: float = 0.8,
) -> Tuple[DataLoader, DataLoader]:
    tfm = build_transform(cfg.img_size)

    train_ds = BUSIDataset(busi_root, split="train", transform=tfm, seed=cfg.seed, train_ratio=train_ratio)
    test_ds = BUSIDataset(busi_root, split="test", transform=tfm, seed=cfg.seed, train_ratio=train_ratio)

    if cfg.stratified:
        train_labels = [y for _, y in train_ds.items]
        sampler = StratifiedBatchSampler(train_labels, batch_size=cfg.batch_size, drop_last=True, seed=cfg.seed)
        train_loader = DataLoader(
            train_ds,
            batch_sampler=sampler,
            num_workers=cfg.num_workers,
            pin_memory=cfg.pin_memory,
        )
    else:
        train_loader = DataLoader(
            train_ds,
            batch_size=cfg.batch_size,
            shuffle=True,
            num_workers=cfg.num_workers,
            pin_memory=cfg.pin_memory,
            drop_last=True,
        )

    test_loader = DataLoader(
        test_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
    )

    return train_loader, test_loader


# ----------------------------
# DR image pre-loader (cache all images in RAM before fold loop)
# ----------------------------
def preload_dr_images(
    items: List[Tuple[str, int]],
    img_size: int = 300,
    n_threads: int = 16,
    cache_dir: Optional[str] = None,
) -> Dict[str, torch.Tensor]:
    """
    Load all DR images into RAM as resized uint8 tensors.

    If cache_dir is given:
      - First call: decode JPEGs, resize, save as .pt files in cache_dir (~30 min).
      - Later calls: load .pt files directly (~30 sec). Survives process restarts.
    Without cache_dir: always decodes from source (no disk persistence).

    Returns {path: uint8 tensor (C, H, W)}.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)

    resize = transforms.Resize((img_size, img_size))
    paths = list(dict.fromkeys(path for path, _ in items))
    n = len(paths)
    cache: Dict[str, torch.Tensor] = {}

    def _pt_path(src_path: str) -> str:
        stem = os.path.splitext(os.path.basename(src_path))[0]
        return os.path.join(cache_dir, f"{stem}.pt")  # type: ignore[arg-type]

    def _load(path: str):
        if cache_dir:
            pt = _pt_path(path)
            if os.path.exists(pt):
                return path, torch.load(pt, weights_only=True)
        img = _pil_loader(path, rgb=True)
        img = resize(img)
        t = torch.from_numpy(np.array(img)).permute(2, 0, 1).contiguous()
        if cache_dir:
            torch.save(t, _pt_path(path))
        return path, t

    with ThreadPoolExecutor(max_workers=n_threads) as pool:
        futures = {pool.submit(_load, p): p for p in paths}
        done = 0
        for future in as_completed(futures):
            path, tensor = future.result()
            cache[path] = tensor
            done += 1
            if done % 5000 == 0 or done == n:
                approx_gb = done * img_size * img_size * 3 / 1e9
                print(f"  Caching DR images: {done}/{n}  (~{approx_gb:.1f} GB in RAM)")

    return cache


# ----------------------------
# Generic dataset for CV splits
# ----------------------------
class ImageLabelDataset(Dataset):
    """
    Minimal dataset that accepts a pre-built list of (path, label) tuples.
    Used by the cross-validation splits (training/cross_val.py) instead of
    the class-folder / CSV datasets above, which assume fixed train/test splits.

    Pass img_cache (from preload_dr_images) to skip per-epoch JPEG decode.
    Workers read the shared cache read-only (CoW on Linux fork) — no RAM copies.
    """

    def __init__(
        self,
        items: List[Tuple[str, int]],
        transform: Optional[Callable] = None,
        rgb: bool = True,
        img_cache: Optional[Dict[str, torch.Tensor]] = None,
    ):
        self.items = items
        self.transform = transform or build_transform(300)
        self.rgb = rgb
        self._img_cache = img_cache

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        img_path, label = self.items[idx]
        if self._img_cache is not None and img_path in self._img_cache:
            # Cached path: uint8 tensor -> PIL -> transform (augmentation included).
            # to_pil_image is zero-copy on uint8; the Resize in transform is a no-op
            # since the image is already at img_size × img_size.
            img = to_pil_image(self._img_cache[img_path])
        else:
            img = _pil_loader(img_path, rgb=self.rgb)
        x = self.transform(img)
        y = torch.tensor(label, dtype=torch.long)
        return x, y