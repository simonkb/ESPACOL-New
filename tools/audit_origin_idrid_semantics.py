#!/usr/bin/env python3
"""External, zero-shot semantic audit of ORIGIN ledgers on IDRiD.

This program deliberately evaluates a narrower claim than pixel causality.  It
asks whether spatial ledger entries are enriched in externally annotated
lesion bins and whether deleting those *stored entries* has a larger ordinal
effect than deleting count-matched non-lesion entries.  Images are never
masked or re-encoded during the intervention.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass, fields
import hashlib
import json
import math
from pathlib import Path
import random
import sys
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision.transforms import functional as TF

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from configs.origin_config import OriginConfig
from Datasets.dataloaders import _IMAGENET_MEAN, _IMAGENET_STD
from Datasets.mosaic_data import _dominant_field_bounds
from models.origin import OriginOutput, build_origin_model, replay_without
from utils.spatial_mask import centered_ellipse_mask


LESIONS: Mapping[str, tuple[str, str]] = {
    "microaneurysm": ("1. Microaneurysms", "MA"),
    "haemorrhage": ("2. Haemorrhages", "HE"),
    "hard_exudate": ("3. Hard Exudates", "EX"),
    "soft_exudate": ("4. Soft Exudates", "SE"),
}
SCALE_GROUPS: Mapping[str, tuple[str, ...]] = {
    "fine": ("s4", "s8"),
    "coarse": ("s16", "s32"),
    "all": ("s4", "s8", "s16", "s32"),
}


@dataclass(frozen=True)
class IDRiDSample:
    image_id: str
    image_path: Path
    masks: Mapping[str, Path | None]
    grade: int
    segmentation_split: str


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def binary_average_precision(scores: torch.Tensor, labels: torch.Tensor) -> float:
    """Average precision with stable tie handling and no sklearn dependency."""

    scores = scores.detach().double().flatten().cpu()
    labels = labels.detach().bool().flatten().cpu()
    positives = int(labels.sum())
    if positives == 0 or positives == labels.numel():
        return float("nan")
    order = torch.argsort(scores, descending=True, stable=True)
    sorted_labels = labels[order].double()
    precision = sorted_labels.cumsum(0) / torch.arange(
        1, labels.numel() + 1, dtype=torch.double
    )
    return float((precision * sorted_labels).sum() / positives)


def lesion_cell_mask(pixel_mask: torch.Tensor, output_size: Sequence[int]) -> torch.Tensor:
    """Mark lattice bins containing at least one annotated lesion pixel."""

    if pixel_mask.ndim == 2:
        pixel_mask = pixel_mask[None, None]
    elif pixel_mask.ndim == 3:
        pixel_mask = pixel_mask[:, None]
    if pixel_mask.ndim != 4 or pixel_mask.shape[1] != 1:
        raise ValueError("pixel_mask must have shape HW, NHW, or N1HW")
    pooled = F.adaptive_max_pool2d(
        pixel_mask.float(), (int(output_size[0]), int(output_size[1]))
    )
    return pooled[:, 0] > 0


def count_matched_nonlesion_mask(
    lesion_mask: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    generator: torch.Generator,
) -> torch.Tensor:
    """Sample the same number of valid non-lesion cells for each image."""

    lesion_mask = lesion_mask.bool() & valid_mask.bool()
    result = torch.zeros_like(lesion_mask)
    for row in range(lesion_mask.shape[0]):
        count = int(lesion_mask[row].sum())
        candidates = torch.nonzero(
            valid_mask[row].bool() & ~lesion_mask[row], as_tuple=False
        )
        if count == 0:
            continue
        count = min(count, int(candidates.shape[0]))
        order = torch.randperm(candidates.shape[0], generator=generator)[:count]
        chosen = candidates[order]
        result[row, chosen[:, 0], chosen[:, 1]] = True
    return result


def _read_grades(idrid_root: Path) -> dict[int, int]:
    csv_path = (
        idrid_root
        / "B. Disease Grading"
        / "2. Groundtruths"
        / "a. IDRiD_Disease Grading_Training Labels.csv"
    )
    grades: dict[int, int] = {}
    with csv_path.open(newline="") as stream:
        for row in csv.DictReader(stream):
            identifier = int(str(row["Image name"]).split("_")[-1])
            grades[identifier] = int(row["Retinopathy grade"])
    return grades


def discover_idrid_segmentation(idrid_root: Path) -> list[IDRiDSample]:
    """Return the official 81-image segmentation subset in stable ID order."""

    segmentation = idrid_root / "A. Segmentation"
    image_root = segmentation / "1. Original Images"
    mask_root = segmentation / "2. All Segmentation Groundtruths"
    grades = _read_grades(idrid_root)
    samples: list[IDRiDSample] = []
    for split_name, split_dir, mask_split in (
        ("training", "a. Training Set", "a. Training Set"),
        ("testing", "b. Testing Set", "b. Testing Set"),
    ):
        for image_path in sorted((image_root / split_dir).glob("IDRiD_*.jpg")):
            numeric_id = int(image_path.stem.split("_")[-1])
            image_id = f"IDRiD_{numeric_id:02d}"
            if numeric_id not in grades:
                raise ValueError(f"no official grading label for {image_id}")
            masks: dict[str, Path | None] = {}
            for lesion, (folder, suffix) in LESIONS.items():
                path = mask_root / mask_split / folder / f"{image_id}_{suffix}.tif"
                masks[lesion] = path if path.is_file() else None
            samples.append(
                IDRiDSample(
                    image_id=image_id,
                    image_path=image_path,
                    masks=masks,
                    grade=grades[numeric_id],
                    segmentation_split=split_name,
                )
            )
    if len(samples) != 81:
        raise ValueError(f"expected 81 IDRiD segmentation images, found {len(samples)}")
    return samples


def _canonicalize_sample(
    sample: IDRiDSample, size: int
) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    with Image.open(sample.image_path) as opened:
        image = opened.convert("RGB")
    original_size = image.size
    bounds = _dominant_field_bounds(image)
    cropped = image if bounds is None else image.crop(bounds)
    image_tensor = TF.to_tensor(
        cropped.resize((size, size), Image.Resampling.BILINEAR)
    )
    image_tensor = TF.normalize(image_tensor, _IMAGENET_MEAN, _IMAGENET_STD)
    valid = centered_ellipse_mask(size, size)[0].bool()
    lesion_masks: dict[str, torch.Tensor] = {}
    for lesion, path in sample.masks.items():
        if path is None:
            lesion_masks[lesion] = torch.zeros(size, size, dtype=torch.bool)
            continue
        with Image.open(path) as opened:
            mask = opened.convert("L")
        if mask.size != original_size:
            raise ValueError(
                f"mask/image size mismatch for {sample.image_id}: "
                f"{mask.size} vs {original_size}"
            )
        mask = mask if bounds is None else mask.crop(bounds)
        mask = mask.resize((size, size), Image.Resampling.NEAREST)
        lesion_masks[lesion] = TF.pil_to_tensor(mask)[0] > 0
    return image_tensor, valid, lesion_masks


def _build_from_checkpoint(
    checkpoint: Path, device: torch.device
) -> tuple[torch.nn.Module, OriginConfig, dict[str, Any]]:
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if state.get("schema") != "origin-checkpoint-v3":
        raise ValueError(f"{checkpoint} is not an ORIGIN-v3 checkpoint")
    allowed = {field.name for field in fields(OriginConfig)}
    cfg = OriginConfig(
        **{key: value for key, value in state["config"].items() if key in allowed}
    )
    model = build_origin_model(
        num_classes=cfg.n_classes,
        encoder_name=cfg.encoder,
        pretrained=False,
        evidence_scales=cfg.evidence_scales,
        projection_dim=cfg.projection_dim,
        reference_count=cfg.reference_count,
        atom_rate_init=cfg.atom_rate_init,
        prior_rate_init=cfg.prior_rate_init,
        boundary_scale_init=cfg.boundary_scale_init,
        total_rate_cap=cfg.total_rate_cap,
        prior_rate_cap=cfg.prior_rate_cap,
        boundary_scale_cap=cfg.boundary_scale_cap,
        rate_roundoff_margin=cfg.rate_roundoff_margin,
        atom_mode=cfg.atom_mode,
        hybrid_cumulative_init=cfg.hybrid_cumulative_init,
        evidence_dropout=cfg.evidence_dropout,
        mask_valid_fraction=cfg.mask_valid_fraction,
        grad_checkpoint=False,
    )
    model.load_state_dict(state["model_state"], strict=True)
    model.to(device).eval()
    metadata = {
        "fold": int(state["fold"]),
        "epoch": int(state["epoch"]),
        "checkpoint_sha256": _sha256(checkpoint),
        "architecture_signature": state.get("architecture_signature"),
        "config_signature": state.get("config_signature"),
    }
    return model, cfg, metadata


def _mean_or_nan(values: Iterable[float]) -> float:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    return float(np.mean(finite)) if finite else float("nan")


def _replay_delta(
    output: OriginOutput, removal_masks: Mapping[str, torch.Tensor]
) -> tuple[torch.Tensor, torch.Tensor]:
    replayed = replay_without(output, removal_masks).output
    expected = output.expected_grade - replayed.expected_grade
    tails = output.cumulative_probs - replayed.cumulative_probs
    return expected, tails


def audit_checkpoint(
    checkpoint: Path,
    samples: Sequence[IDRiDSample],
    *,
    device: torch.device,
    batch_size: int,
    random_repeats: int,
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    model, cfg, metadata = _build_from_checkpoint(checkpoint, device)
    records: list[dict[str, Any]] = []
    for start in range(0, len(samples), batch_size):
        batch = samples[start : start + batch_size]
        materialized = [_canonicalize_sample(sample, cfg.img_size) for sample in batch]
        images = torch.stack([item[0] for item in materialized]).to(device)
        valid_pixels = torch.stack([item[1] for item in materialized]).to(device)
        lesion_pixels = {
            lesion: torch.stack([item[2][lesion] for item in materialized]).to(device)
            for lesion in LESIONS
        }
        union_pixels = torch.stack(
            [torch.stack(list(item[2].values())).any(0) for item in materialized]
        ).to(device)
        with torch.inference_mode(), torch.autocast(
            device_type=device.type, enabled=device.type == "cuda"
        ):
            output = model(
                images,
                pixel_valid_mask=valid_pixels,
                force_decoder_fp64=True,
            )
        per_scale_union: dict[str, torch.Tensor] = {}
        per_scale_lesion: dict[str, dict[str, torch.Tensor]] = {}
        for scale, valid in output.valid_masks.items():
            shape = valid.shape[-2:]
            per_scale_union[scale] = lesion_cell_mask(union_pixels, shape) & valid
            per_scale_lesion[scale] = {
                lesion: lesion_cell_mask(pixel_mask, shape) & valid
                for lesion, pixel_mask in lesion_pixels.items()
            }

        # External-mask alignment is reported per scale and boundary.  AP is
        # accompanied by prevalence because tiny lesions make raw AP hard to
        # interpret in isolation.
        for row, sample in enumerate(batch):
            target_boundary = min(max(sample.grade - 1, 0), cfg.n_classes - 2)
            for scale, evidence in output.scale_evidence.items():
                valid = evidence.valid_mask[row]
                for lesion in ("union", *LESIONS.keys()):
                    lesion_mask = (
                        per_scale_union[scale][row]
                        if lesion == "union"
                        else per_scale_lesion[scale][lesion][row]
                    )
                    for boundary in range(cfg.n_classes - 1):
                        scores = evidence.local_rate_map[row, boundary][valid]
                        labels = lesion_mask[valid]
                        prevalence = float(labels.float().mean())
                        ap = binary_average_precision(scores, labels)
                        lesion_mean = (
                            float(scores[labels].mean()) if bool(labels.any()) else float("nan")
                        )
                        background = ~labels
                        background_mean = (
                            float(scores[background].mean())
                            if bool(background.any())
                            else float("nan")
                        )
                        records.append(
                            {
                                "kind": "alignment",
                                "fold": metadata["fold"],
                                "image_id": sample.image_id,
                                "grade": sample.grade,
                                "split": sample.segmentation_split,
                                "scale": scale,
                                "scale_group": (
                                    "fine" if scale in SCALE_GROUPS["fine"] else "coarse"
                                ),
                                "boundary": boundary,
                                "target_boundary": target_boundary,
                                "lesion": lesion,
                                "prevalence": prevalence,
                                "average_precision": ap,
                                "ap_over_prevalence": (
                                    ap / prevalence
                                    if prevalence > 0.0 and math.isfinite(ap)
                                    else float("nan")
                                ),
                                "lesion_mean_rate": lesion_mean,
                                "nonlesion_mean_rate": background_mean,
                                "mean_rate_ratio": (
                                    lesion_mean / background_mean
                                    if math.isfinite(lesion_mean)
                                    and math.isfinite(background_mean)
                                    and background_mean > 0.0
                                    else float("nan")
                                ),
                            }
                        )

            # Delete externally annotated bins from the immutable ledger and
            # compare with deterministic count-matched non-lesion deletions.
            for group, group_scales in SCALE_GROUPS.items():
                selected = {
                    scale: (
                        per_scale_union[scale][row : row + 1]
                        if scale in group_scales
                        else torch.zeros_like(output.valid_masks[scale][row : row + 1])
                    )
                    for scale in output.scale_evidence
                }
                single_output = OriginOutput(
                    scale_evidence={
                        name: type(value)(
                            **{
                                field.name: (
                                    getattr(value, field.name)[row : row + 1]
                                    if torch.is_tensor(getattr(value, field.name))
                                    and getattr(value, field.name).ndim > 0
                                    and getattr(value, field.name).shape[0] == len(batch)
                                    else getattr(value, field.name)
                                )
                                for field in fields(type(value))
                            }
                        )
                        for name, value in output.scale_evidence.items()
                    },
                    prior_rates=output.prior_rates[row : row + 1],
                    total_rates=output.total_rates[row : row + 1],
                    generator=output.generator[row : row + 1],
                    transition_matrix=output.transition_matrix[row : row + 1],
                    class_probs=output.class_probs[row : row + 1],
                    log_class_probs=output.log_class_probs[row : row + 1],
                    cumulative_probs=output.cumulative_probs[row : row + 1],
                    expected_grade=output.expected_grade[row : row + 1],
                    posterior_median=output.posterior_median[row : row + 1],
                    class_map=output.class_map[row : row + 1],
                    atom_mode=output.atom_mode,
                    scale_simplex=output.scale_simplex,
                    boundary_scales=output.boundary_scales,
                    total_rate_cap=output.total_rate_cap,
                    prior_rate_cap=output.prior_rate_cap,
                    boundary_scale_cap=output.boundary_scale_cap,
                    atom_mass_cap=output.atom_mass_cap,
                    rate_roundoff_margin=output.rate_roundoff_margin,
                    encoder_output=None,
                )
                lesion_delta, lesion_tails = _replay_delta(single_output, selected)
                generator = torch.Generator(device="cpu")
                generator.manual_seed(seed + 1009 * metadata["fold"] + int(sample.image_id[-2:]))
                controls_expected: list[float] = []
                controls_tail: list[float] = []
                for _ in range(random_repeats):
                    control: dict[str, torch.Tensor] = {}
                    for scale in output.scale_evidence:
                        if scale not in group_scales:
                            control[scale] = torch.zeros_like(selected[scale])
                        else:
                            control[scale] = count_matched_nonlesion_mask(
                                per_scale_union[scale][row : row + 1].cpu(),
                                output.valid_masks[scale][row : row + 1].cpu(),
                                generator=generator,
                            ).to(device)
                    expected_delta, tail_delta = _replay_delta(single_output, control)
                    controls_expected.append(float(expected_delta[0]))
                    controls_tail.append(float(tail_delta[0, target_boundary]))
                records.append(
                    {
                        "kind": "external_mask_deletion",
                        "fold": metadata["fold"],
                        "image_id": sample.image_id,
                        "grade": sample.grade,
                        "split": sample.segmentation_split,
                        "scale_group": group,
                        "target_boundary": target_boundary,
                        "deleted_cells": int(sum(mask.sum() for mask in selected.values())),
                        "expected_grade_delta": float(lesion_delta[0]),
                        "target_tail_delta": float(lesion_tails[0, target_boundary]),
                        "random_expected_grade_delta_mean": _mean_or_nan(controls_expected),
                        "random_target_tail_delta_mean": _mean_or_nan(controls_tail),
                        "expected_grade_delta_lift": float(lesion_delta[0])
                        - _mean_or_nan(controls_expected),
                        "target_tail_delta_lift": float(lesion_tails[0, target_boundary])
                        - _mean_or_nan(controls_tail),
                        "random_repeats": random_repeats,
                    }
                )
        del output, images, valid_pixels
    return records, metadata


def summarize(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    summaries: dict[str, Any] = {}
    alignment = [row for row in records if row["kind"] == "alignment"]
    for group in ("fine", "coarse"):
        target = [
            row
            for row in alignment
            if row["scale_group"] == group
            and row["lesion"] == "union"
            and row["boundary"] == row["target_boundary"]
        ]
        summaries[f"alignment_{group}"] = {
            "n": len(target),
            "mean_average_precision": _mean_or_nan(
                row["average_precision"] for row in target
            ),
            "mean_prevalence": _mean_or_nan(row["prevalence"] for row in target),
            "mean_ap_over_prevalence": _mean_or_nan(
                row["ap_over_prevalence"] for row in target
            ),
            "mean_rate_ratio": _mean_or_nan(row["mean_rate_ratio"] for row in target),
        }
    deletion = [row for row in records if row["kind"] == "external_mask_deletion"]
    for group in SCALE_GROUPS:
        target = [row for row in deletion if row["scale_group"] == group]
        summaries[f"deletion_{group}"] = {
            "n": len(target),
            "mean_expected_grade_delta": _mean_or_nan(
                row["expected_grade_delta"] for row in target
            ),
            "mean_random_expected_grade_delta": _mean_or_nan(
                row["random_expected_grade_delta_mean"] for row in target
            ),
            "mean_expected_grade_lift": _mean_or_nan(
                row["expected_grade_delta_lift"] for row in target
            ),
            "mean_target_tail_delta": _mean_or_nan(
                row["target_tail_delta"] for row in target
            ),
            "mean_random_target_tail_delta": _mean_or_nan(
                row["random_target_tail_delta_mean"] for row in target
            ),
        }
    return summaries


def _json_safe(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--idrid-root", required=True, type=Path)
    parser.add_argument("--checkpoints", nargs="+", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--random-repeats", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20261001)
    args = parser.parse_args()
    if args.batch_size < 1 or args.random_repeats < 1:
        raise ValueError("batch-size and random-repeats must be positive")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    samples = discover_idrid_segmentation(args.idrid_root)
    all_records: list[dict[str, Any]] = []
    checkpoint_metadata: list[dict[str, Any]] = []
    for checkpoint in args.checkpoints:
        records, metadata = audit_checkpoint(
            checkpoint.resolve(),
            samples,
            device=device,
            batch_size=args.batch_size,
            random_repeats=args.random_repeats,
            seed=args.seed,
        )
        all_records.extend(records)
        checkpoint_metadata.append(metadata)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    records_path = args.output_dir / "idrid_semantic_records.jsonl"
    with records_path.open("w") as stream:
        for record in all_records:
            stream.write(json.dumps(_json_safe(record), sort_keys=True) + "\n")
    payload = {
        "schema": "origin-idrid-semantic-audit-v1",
        "scope": "external_zero_shot_cell_bin_alignment_and_internal_ledger_deletion",
        "non_claims": [
            "not a pixel intervention",
            "not proof of lesion identity",
            "not proof of biological causality",
            "coarse cell indices do not guarantee local visual dependence",
        ],
        "idrid_images": len(samples),
        "grade_source": (
            "official IDRiD disease-grading training labels for IDs 001--081; "
            "grades select the audited ordinal boundary only"
        ),
        "checkpoints": checkpoint_metadata,
        "random_repeats": args.random_repeats,
        "summaries": summarize(all_records),
        "records": records_path.name,
        "records_sha256": _sha256(records_path),
    }
    summary_path = args.output_dir / "idrid_semantic_summary.json"
    summary_path.write_text(json.dumps(_json_safe(payload), indent=2, sort_keys=True) + "\n")
    print(json.dumps({"summary": str(summary_path), **payload["summaries"]}, indent=2))


if __name__ == "__main__":
    main()
