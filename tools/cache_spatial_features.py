#!/usr/bin/env python3
"""Cache leakage-safe pretrained local maps for rapid MOSAIC head screens.

The cache stores the raw EfficientNet tap map, before MOSAIC's trainable
pointwise projection or ordinal head.  It is therefore reusable across proof
hyperparameters while retaining a trainable local representation head.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from Datasets.mosaic_data import (
    MOSAIC_PREPROCESSING_VERSION,
    MosaicFundusTransform,
    MosaicImageDataset,
    load_aptos_items,
    load_eyepacs_items,
)
from models.local_efficientnet import (
    LocalEfficientNetV2S,
    downsample_retinal_field_mask,
)


CACHE_SCHEMA = "mosaic-spatial-cache-v2"


def state_hash(module: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(module.state_dict().items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(tuple(tensor.shape)).encode())
        digest.update(str(tensor.dtype).encode())
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def source_hash(items) -> str:
    digest = hashlib.sha256()
    for path, label in sorted(items):
        stat = os.stat(path)
        record = f"{Path(path).name}\0{label}\0{stat.st_size}\0{stat.st_mtime_ns}\n"
        digest.update(record.encode())
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Cache raw MOSAIC spatial tap maps",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dataset", choices=("aptos", "dr"), required=True)
    parser.add_argument("--data_root", default=None)
    parser.add_argument("--labels_csv", default=None)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--image_size", type=int, default=896)
    parser.add_argument("--local_stage", choices=("rf_small", "rf_medium", "rf_large"), default="rf_medium")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--no_pretrained", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--limit", type=int, default=0, help="debug only; 0 caches all images")
    args = parser.parse_args()

    root = args.data_root or (
        "Datasets/aptos2019-blindness-detection" if args.dataset == "aptos" else "Datasets/DR"
    )
    items = (
        load_aptos_items(root, args.labels_csv)
        if args.dataset == "aptos"
        else load_eyepacs_items(root, args.labels_csv)
    )
    if args.limit > 0:
        items = items[: args.limit]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = LocalEfficientNetV2S(
        tap=args.local_stage,
        local_dim=8,  # unused: raw trunk maps are cached
        pretrained=not args.no_pretrained,
    ).to(device).eval()
    preprocessing = {
        "version": MOSAIC_PREPROCESSING_VERSION,
        "image_size": args.image_size,
        "crop": "dominant_nonblack_field",
        "resize": "direct_fundus_field_to_square",
        "normalization": "imagenet",
        "augmentation": False,
        "mask": "fixed_centered_ellipse",
    }
    identity = {
        "schema": CACHE_SCHEMA,
        "dataset": args.dataset,
        "local_stage": args.local_stage,
        "tap_channels": encoder.tap_channels,
        "output_stride": encoder.output_stride,
        "receptive_field": encoder.receptive_field,
        "pretrained": not args.no_pretrained,
        "preprocessing": preprocessing,
        "source_hash": source_hash(items),
        "encoder_trunk_hash": state_hash(encoder.trunk),
    }
    identity_hash = hashlib.sha256(
        json.dumps(identity, sort_keys=True).encode()
    ).hexdigest()

    output_dir = Path(args.output_dir)
    feature_dir = output_dir / "features"
    manifest_path = output_dir / "manifest.json"
    if manifest_path.exists() and not args.overwrite:
        existing = json.loads(manifest_path.read_text())
        if existing.get("identity_hash") == identity_hash:
            print(f"matching cache already exists: {manifest_path}")
            return
        raise RuntimeError(
            "cache directory contains a different preprocessing/encoder identity; "
            "choose another output directory or pass --overwrite"
        )
    feature_dir.mkdir(parents=True, exist_ok=True)

    dataset = MosaicImageDataset(
        items,
        MosaicFundusTransform(args.image_size, augment=False),
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    records = [None] * len(items)
    with torch.no_grad():
        for batch_number, (images, masks, labels, indices) in enumerate(loader, start=1):
            images = images.to(device, non_blocking=True)
            raw = encoder.forward_trunk_map(images)
            valid = downsample_retinal_field_mask(
                masks.to(device, non_blocking=True),
                raw.shape[-2:],
                min_valid_fraction=encoder.mask_valid_fraction,
            ).flatten(1)
            for position, dataset_index_tensor in enumerate(indices):
                dataset_index = int(dataset_index_tensor)
                path, label = items[dataset_index]
                key = hashlib.sha256(
                    f"{dataset_index}\0{Path(path).name}".encode()
                ).hexdigest()[:20]
                relative_file = f"features/{key}.pt"
                torch.save(
                    {
                        "schema": CACHE_SCHEMA,
                        "identity_hash": identity_hash,
                        "features": raw[position].detach().cpu().half(),
                        "valid_mask": valid[position].detach().cpu().bool(),
                        "label": int(label),
                        "source_path": path,
                    },
                    output_dir / relative_file,
                )
                records[dataset_index] = {
                    "dataset_index": dataset_index,
                    "source_path": path,
                    "label": int(label),
                    "file": relative_file,
                }
            if batch_number % 25 == 0 or batch_number == len(loader):
                print(f"cached {min(batch_number * args.batch_size, len(items))}/{len(items)}")

    manifest = {
        **identity,
        "identity_hash": identity_hash,
        "num_items": len(items),
        "feature_shape": [encoder.tap_channels, *raw.shape[-2:]],
        "dtype": "float16",
        "records": records,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"cache complete: {manifest_path}")


if __name__ == "__main__":
    main()
