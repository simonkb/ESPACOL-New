#!/usr/bin/env python3
"""IDRiD semantic audit with bidirectional, scale-matched ledger controls.

This is a new, non-overwriting v2 audit.  It preserves the v1 within-image
average-precision estimand and record structure while making AP invariant to
the spatial order of tied scores.  It replaces the v1 deletion control with
paired sampling on both sides of the lesion mask.  For every image,
checkpoint, scale, and repeat, exactly

    m = min(number of lesion cells, number of non-lesion cells)

unique lesion cells and ``m`` unique non-lesion cells are deleted.  Matching
within each scale makes cell count, scale composition, conserved geometry
exposure, and the pre-atom target-boundary multiplier identical.  Actual
stored rate mass is intentionally *not* matched: its difference is the signal
being audited.  As in v1, these are interventions on an immutable stored
ledger, not pixel interventions and not claims of biological causality.
"""

from __future__ import annotations

import argparse
from dataclasses import fields
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models.origin import OriginOutput
from tools.audit_origin_idrid_semantics import (
    IDRiDSample,
    LESIONS,
    SCALE_GROUPS,
    _build_from_checkpoint,
    _canonicalize_sample,
    _json_safe,
    _mean_or_nan,
    _replay_delta,
    _sha256,
    discover_idrid_segmentation,
    lesion_cell_mask,
)


AUDIT_SCHEMA = "origin-idrid-semantic-audit-v2"
AUDIT_SCOPE = (
    "external_zero_shot_cell_bin_alignment_and_bidirectionally_matched_"
    "internal_ledger_deletion"
)
DELETION_KIND = "external_mask_deletion_v2"
MATCHING_CONTRACT = "within_image_within_scale_unique_bidirectional_min_count_v1"
AP_CONTRACT = "noninterpolated_threshold_ap_with_exact_score_ties_grouped_v1"
SCALES = ("s4", "s8", "s16", "s32")


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _dataset_inventory(
    idrid_root: Path, samples: Sequence[IDRiDSample]
) -> dict[str, Any]:
    """Content-address every image, mask, and grade file used by the audit."""

    grade_csv = (
        idrid_root
        / "B. Disease Grading"
        / "2. Groundtruths"
        / "a. IDRiD_Disease Grading_Training Labels.csv"
    )
    inputs = {grade_csv.resolve()}
    for sample in samples:
        inputs.add(sample.image_path.resolve())
        inputs.update(path.resolve() for path in sample.masks.values() if path is not None)
    root = idrid_root.resolve()
    entries: list[dict[str, Any]] = []
    for path in sorted(inputs):
        if not path.is_file():
            raise FileNotFoundError(path)
        try:
            relative = path.relative_to(root)
        except ValueError:
            raise AssertionError(f"IDRiD input lies outside the declared root: {path}") from None
        entries.append(
            {
                "path": relative.as_posix(),
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
        )
    manifest: dict[str, Any] = {
        "root_label": "IDRiD",
        "files": entries,
        "file_count": len(entries),
    }
    manifest["content_checksum_sha256"] = _canonical_sha256(manifest)
    return manifest


def tie_grouped_binary_average_precision(
    scores: torch.Tensor, labels: torch.Tensor
) -> float:
    """Average precision that is invariant to ordering within score ties.

    A stable element-wise sort is not sufficient tie handling: if every score
    is equal, the result can otherwise depend on the spatial flattening order
    and exceed the lesion prevalence.  This implementation evaluates
    precision only at the end of each distinct-score block, matching the
    threshold-based, non-interpolated AP definition used by standard metric
    libraries.
    """

    scores = scores.detach().double().flatten().cpu()
    labels = labels.detach().bool().flatten().cpu()
    if scores.numel() != labels.numel():
        raise ValueError("scores and labels must contain the same number of values")
    if not bool(torch.isfinite(scores).all()):
        raise ValueError("average-precision scores must be finite")
    positives = int(labels.sum())
    if positives == 0 or positives == labels.numel():
        return float("nan")
    order = torch.argsort(scores, descending=True, stable=True)
    ordered_scores = scores[order]
    ordered_labels = labels[order].double()
    cumulative_positives = ordered_labels.cumsum(0)
    group_ends = torch.ones_like(ordered_scores, dtype=torch.bool)
    group_ends[:-1] = ordered_scores[:-1] != ordered_scores[1:]
    end_indices = torch.nonzero(group_ends, as_tuple=False).flatten()
    positives_at_end = cumulative_positives[end_indices]
    positives_before = torch.cat(
        (torch.zeros(1, dtype=torch.double), positives_at_end[:-1])
    )
    positives_in_group = positives_at_end - positives_before
    precision_at_end = positives_at_end / (end_indices.double() + 1.0)
    return float((precision_at_end * positives_in_group).sum() / positives)


def _stable_seed(seed: int, *parts: Any) -> int:
    message = ":".join((str(seed), *(str(part) for part in parts)))
    return int.from_bytes(hashlib.sha256(message.encode()).digest()[:8], "little")


def balanced_paired_cell_masks(
    lesion_mask: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, int]]:
    """Draw equally sized, unique lesion and non-lesion cell sets.

    Inputs are one two-dimensional lattice.  The operation is deliberately
    symmetric: the more abundant side is randomly subsampled, while the less
    abundant side is used in full (up to a random permutation).  The returned
    masks are disjoint subsets of ``valid_mask``.
    """

    if lesion_mask.ndim != 2 or valid_mask.ndim != 2:
        raise ValueError("lesion_mask and valid_mask must be two-dimensional")
    if lesion_mask.shape != valid_mask.shape:
        raise ValueError("lesion_mask and valid_mask must have identical shape")
    lesion = lesion_mask.bool() & valid_mask.bool()
    nonlesion = valid_mask.bool() & ~lesion
    lesion_indices = torch.nonzero(lesion, as_tuple=False)
    nonlesion_indices = torch.nonzero(nonlesion, as_tuple=False)
    lesion_count = int(lesion_indices.shape[0])
    nonlesion_count = int(nonlesion_indices.shape[0])
    matched = min(lesion_count, nonlesion_count)
    guided = torch.zeros_like(lesion)
    control = torch.zeros_like(lesion)
    if matched:
        lesion_order = torch.randperm(lesion_count, generator=generator)[:matched]
        nonlesion_order = torch.randperm(nonlesion_count, generator=generator)[:matched]
        guided_cells = lesion_indices[lesion_order]
        control_cells = nonlesion_indices[nonlesion_order]
        guided[guided_cells[:, 0], guided_cells[:, 1]] = True
        control[control_cells[:, 0], control_cells[:, 1]] = True
    counts = {
        "valid_cells": int(valid_mask.bool().sum()),
        "lesion_candidates": lesion_count,
        "nonlesion_candidates": nonlesion_count,
        "matched_cells": matched,
        "guided_cells": int(guided.sum()),
        "control_cells": int(control.sum()),
    }
    if counts["valid_cells"] != lesion_count + nonlesion_count:
        raise AssertionError("valid lattice did not partition into lesion/non-lesion cells")
    if counts["guided_cells"] != matched or counts["control_cells"] != matched:
        raise AssertionError("bidirectional matcher violated exact count equality")
    if bool((guided & control).any()) or bool((guided & ~valid_mask.bool()).any()):
        raise AssertionError("paired masks are not disjoint valid-cell subsets")
    return guided, control, counts


def _slice_output(output: OriginOutput, row: int, batch_size: int) -> OriginOutput:
    return OriginOutput(
        scale_evidence={
            name: type(value)(
                **{
                    field.name: (
                        getattr(value, field.name)[row : row + 1]
                        if torch.is_tensor(getattr(value, field.name))
                        and getattr(value, field.name).ndim > 0
                        and getattr(value, field.name).shape[0] == batch_size
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


def _alignment_records(
    output: OriginOutput,
    *,
    row: int,
    sample: IDRiDSample,
    fold: int,
    target_boundary: int,
    per_scale_union: Mapping[str, torch.Tensor],
    per_scale_lesion: Mapping[str, Mapping[str, torch.Tensor]],
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for scale, evidence in output.scale_evidence.items():
        valid = evidence.valid_mask[row]
        for lesion in ("union", *LESIONS.keys()):
            lesion_mask = (
                per_scale_union[scale][row]
                if lesion == "union"
                else per_scale_lesion[scale][lesion][row]
            )
            for boundary in range(output.class_probs.shape[1] - 1):
                scores = evidence.local_rate_map[row, boundary][valid]
                labels = lesion_mask[valid]
                prevalence = float(labels.float().mean())
                ap = tie_grouped_binary_average_precision(scores, labels)
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
                        "fold": fold,
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
                        "score_definition": (
                            "stored_local_rate_map_at_the_named_scale_and_boundary"
                        ),
                        "average_precision_definition": AP_CONTRACT,
                        "positive_bin_rule": "any_annotated_lesion_pixel_in_valid_bin",
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
    return records


def _paired_deletion_record(
    output: OriginOutput,
    *,
    row: int,
    sample: IDRiDSample,
    fold: int,
    group: str,
    group_scales: Sequence[str],
    target_boundary: int,
    per_scale_union: Mapping[str, torch.Tensor],
    repeat: int,
    seed: int,
) -> dict[str, Any]:
    single_output = _slice_output(output, row, len(output.class_probs))
    guided: dict[str, torch.Tensor] = {}
    control: dict[str, torch.Tensor] = {}
    scale_counts: dict[str, dict[str, Any]] = {}
    for scale, evidence in output.scale_evidence.items():
        valid_cpu = evidence.valid_mask[row].detach().cpu()
        lesion_cpu = per_scale_union[scale][row].detach().cpu()
        scale_seed = _stable_seed(seed, fold, sample.image_id, group, repeat, scale)
        if scale in group_scales:
            generator = torch.Generator(device="cpu").manual_seed(scale_seed)
            guided_cpu, control_cpu, counts = balanced_paired_cell_masks(
                lesion_cpu, valid_cpu, generator=generator
            )
        else:
            guided_cpu = torch.zeros_like(valid_cpu)
            control_cpu = torch.zeros_like(valid_cpu)
            lesion_candidates = int((lesion_cpu & valid_cpu).sum())
            valid_cells = int(valid_cpu.sum())
            counts = {
                "valid_cells": valid_cells,
                "lesion_candidates": lesion_candidates,
                "nonlesion_candidates": valid_cells - lesion_candidates,
                "matched_cells": 0,
                "guided_cells": 0,
                "control_cells": 0,
            }
        guided[scale] = guided_cpu[None].to(output.class_probs.device)
        control[scale] = control_cpu[None].to(output.class_probs.device)
        valid_cells = counts["valid_cells"]
        matched = counts["matched_cells"]
        geometry_weight = float(evidence.geometry_weight[row])
        target_multiplier = float(
            evidence.scale_weights[target_boundary]
            * output.boundary_scales[target_boundary]
        )
        reference_exposure = matched * geometry_weight
        weighted_exposure = reference_exposure * target_multiplier
        scale_counts[scale] = {
            "included_in_group": scale in group_scales,
            "matching_seed": int(scale_seed),
            **counts,
            "nominal_valid_fraction": (
                0.0 if valid_cells == 0 else matched / valid_cells
            ),
            "geometry_weight": geometry_weight,
            "target_boundary_pre_atom_multiplier": target_multiplier,
            "guided_nominal_reference_exposure": reference_exposure,
            "control_nominal_reference_exposure": reference_exposure,
            "guided_nominal_weighted_exposure": weighted_exposure,
            "control_nominal_weighted_exposure": weighted_exposure,
            "guided_target_rate_removed": float(
                evidence.local_rate_map[row, target_boundary][guided[scale][0]].sum()
            ),
            "control_target_rate_removed": float(
                evidence.local_rate_map[row, target_boundary][control[scale][0]].sum()
            ),
        }

    guided_delta, guided_tails = _replay_delta(single_output, guided)
    control_delta, control_tails = _replay_delta(single_output, control)
    guided_cells = sum(item["guided_cells"] for item in scale_counts.values())
    control_cells = sum(item["control_cells"] for item in scale_counts.values())
    guided_exposure = sum(
        item["guided_nominal_weighted_exposure"] for item in scale_counts.values()
    )
    control_exposure = sum(
        item["control_nominal_weighted_exposure"] for item in scale_counts.values()
    )
    if guided_cells != control_cells or not math.isclose(
        guided_exposure, control_exposure, rel_tol=0.0, abs_tol=1e-10
    ):
        raise AssertionError("paired deletion lost count or nominal-budget equality")
    return {
        "kind": DELETION_KIND,
        "matching_contract": MATCHING_CONTRACT,
        "fold": fold,
        "image_id": sample.image_id,
        "grade": sample.grade,
        "split": sample.segmentation_split,
        "scale_group": group,
        "target_boundary": target_boundary,
        "repeat": repeat,
        "paired_repeats": 1,
        "scale_counts": scale_counts,
        "guided_deleted_cells": guided_cells,
        "control_deleted_cells": control_cells,
        "guided_nominal_weighted_exposure": guided_exposure,
        "control_nominal_weighted_exposure": control_exposure,
        "guided_expected_grade_delta": float(guided_delta[0]),
        "control_expected_grade_delta": float(control_delta[0]),
        "expected_grade_delta_lift": float(guided_delta[0] - control_delta[0]),
        "guided_target_tail_delta": float(guided_tails[0, target_boundary]),
        "control_target_tail_delta": float(control_tails[0, target_boundary]),
        "target_tail_delta_lift": float(
            guided_tails[0, target_boundary] - control_tails[0, target_boundary]
        ),
        "guided_target_rate_removed": sum(
            item["guided_target_rate_removed"] for item in scale_counts.values()
        ),
        "control_target_rate_removed": sum(
            item["control_target_rate_removed"] for item in scale_counts.values()
        ),
    }


def audit_checkpoint(
    checkpoint: Path,
    samples: Sequence[IDRiDSample],
    *,
    device: torch.device,
    batch_size: int,
    paired_repeats: int,
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    model, cfg, metadata = _build_from_checkpoint(checkpoint, device)
    if tuple(cfg.evidence_scales) != SCALES:
        raise ValueError(f"v2 audit requires scales {SCALES}, got {cfg.evidence_scales}")
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
            output = model(images, pixel_valid_mask=valid_pixels, force_decoder_fp64=True)
        per_scale_union: dict[str, torch.Tensor] = {}
        per_scale_lesion: dict[str, dict[str, torch.Tensor]] = {}
        for scale, valid in output.valid_masks.items():
            shape = valid.shape[-2:]
            per_scale_union[scale] = lesion_cell_mask(union_pixels, shape) & valid
            per_scale_lesion[scale] = {
                lesion: lesion_cell_mask(pixel_mask, shape) & valid
                for lesion, pixel_mask in lesion_pixels.items()
            }
        for row, sample in enumerate(batch):
            target_boundary = min(max(sample.grade - 1, 0), cfg.n_classes - 2)
            records.extend(
                _alignment_records(
                    output,
                    row=row,
                    sample=sample,
                    fold=metadata["fold"],
                    target_boundary=target_boundary,
                    per_scale_union=per_scale_union,
                    per_scale_lesion=per_scale_lesion,
                )
            )
            for group, group_scales in SCALE_GROUPS.items():
                for repeat in range(paired_repeats):
                    records.append(
                        _paired_deletion_record(
                            output,
                            row=row,
                            sample=sample,
                            fold=metadata["fold"],
                            group=group,
                            group_scales=group_scales,
                            target_boundary=target_boundary,
                            per_scale_union=per_scale_union,
                            repeat=repeat,
                            seed=seed,
                        )
                    )
        del output, images, valid_pixels
    metadata = {
        **metadata,
        "image_size": cfg.img_size,
        "evidence_scales": list(cfg.evidence_scales),
        "reference_count": cfg.reference_count,
    }
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
        }
    deletion = [row for row in records if row["kind"] == DELETION_KIND]
    lattice_census: dict[tuple[str, str], tuple[int, int, int]] = {}
    for row in deletion:
        for scale, item in row["scale_counts"].items():
            key = (str(row["image_id"]), str(scale))
            value = (
                int(item["valid_cells"]),
                int(item["lesion_candidates"]),
                int(item["nonlesion_candidates"]),
            )
            if key in lattice_census and lattice_census[key] != value:
                raise AssertionError("raw image/scale candidate census changed across rows")
            lattice_census[key] = value
    dense_cases = sorted(
        f"{image}:{scale}"
        for (image, scale), (_, lesion, nonlesion) in lattice_census.items()
        if lesion > nonlesion
    )
    summaries["matching_census"] = {
        "unique_image_scale_lattices": len(lattice_census),
        "dense_image_scale_cases": len(dense_cases),
        "dense_image_scale_case_ids": dense_cases,
        "v1_one_sided_control_would_truncate_cases": len(dense_cases),
        "v2_resolution": (
            "subsample_both_sides_to_m_equals_min_lesion_and_nonlesion_per_scale"
        ),
    }
    for group in SCALE_GROUPS:
        target = [row for row in deletion if row["scale_group"] == group]
        summaries[f"deletion_{group}"] = {
            "n_paired_repeat_rows": len(target),
            "mean_matched_cells": _mean_or_nan(
                row["guided_deleted_cells"] for row in target
            ),
            "mean_guided_expected_grade_delta": _mean_or_nan(
                row["guided_expected_grade_delta"] for row in target
            ),
            "mean_control_expected_grade_delta": _mean_or_nan(
                row["control_expected_grade_delta"] for row in target
            ),
            "mean_expected_grade_lift": _mean_or_nan(
                row["expected_grade_delta_lift"] for row in target
            ),
            "mean_target_tail_lift": _mean_or_nan(
                row["target_tail_delta_lift"] for row in target
            ),
        }
    return summaries


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--idrid-root", required=True, type=Path)
    parser.add_argument("--checkpoints", nargs="+", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--paired-repeats", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20261011)
    parser.add_argument(
        "--source-commit",
        required=True,
        help="immutable Git commit whose tracked audit implementation is running",
    )
    args = parser.parse_args()
    if args.batch_size < 1 or args.paired_repeats < 1:
        raise ValueError("batch-size and paired-repeats must be positive")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError("refusing to overwrite a non-empty v2 audit directory")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    samples = discover_idrid_segmentation(args.idrid_root)
    source_commit = str(args.source_commit).strip().lower()
    if len(source_commit) != 40 or any(character not in "0123456789abcdef" for character in source_commit):
        raise ValueError("source-commit must be a full 40-character hexadecimal Git commit")
    dataset_inventory = _dataset_inventory(args.idrid_root, samples)
    all_records: list[dict[str, Any]] = []
    checkpoint_metadata: list[dict[str, Any]] = []
    for checkpoint in args.checkpoints:
        records, metadata = audit_checkpoint(
            checkpoint.resolve(),
            samples,
            device=device,
            batch_size=args.batch_size,
            paired_repeats=args.paired_repeats,
            seed=args.seed,
        )
        all_records.extend(records)
        checkpoint_metadata.append(metadata)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    records_path = args.output_dir / "idrid_semantic_records_v2.jsonl"
    with records_path.open("x", encoding="utf-8") as stream:
        for record in all_records:
            stream.write(json.dumps(_json_safe(record), sort_keys=True) + "\n")
    payload = {
        "schema": AUDIT_SCHEMA,
        "scope": AUDIT_SCOPE,
        "matching_contract": MATCHING_CONTRACT,
        "average_precision_definition": AP_CONTRACT,
        "non_claims": [
            "not a pixel intervention",
            "not proof of lesion identity",
            "not proof of biological causality",
            "coarse cell indices do not guarantee local visual dependence",
            "actual removed rate mass is an outcome and is not matched",
            "controls are not matched by retinal anatomy or radial position",
        ],
        "idrid_images": len(samples),
        "grade_source": (
            "official IDRiD disease-grading training labels for IDs 001--081; "
            "grades select the audited ordinal boundary only"
        ),
        "image_and_mask_alignment": {
            "crop": "identical dominant-field bounds applied to image and masks",
            "resize": "image bilinear; binary masks nearest-neighbour",
            "canonical_size": checkpoint_metadata[0]["image_size"],
            "lattice_projection": "adaptive max pooling; any lesion pixel is positive",
            "valid_region": "same centered ellipse propagated through model lattice",
        },
        "scalar_score_definition": (
            "one stored local-rate scalar per image, scale, cell, and ordinal boundary"
        ),
        "primary_boundary_rule": "min(max(IDRiD_grade-1,0),3)",
        "checkpoints": checkpoint_metadata,
        "paired_repeats": args.paired_repeats,
        "seed": args.seed,
        "generation": {
            "source_commit": source_commit,
            "audit_implementation_sha256": _sha256(Path(__file__).resolve()),
            "v1_audit_dependency_sha256": _sha256(
                Path(__file__).resolve().with_name("audit_origin_idrid_semantics.py")
            ),
            "checkpoint_order": [int(item["fold"]) for item in checkpoint_metadata],
        },
        "input_dataset_inventory": dataset_inventory,
        "summaries": summarize(all_records),
        "records": records_path.name,
        "records_sha256": _sha256(records_path),
    }
    safe_payload = _json_safe(payload)
    safe_payload["content_checksum_sha256"] = _canonical_sha256(safe_payload)
    summary_path = args.output_dir / "idrid_semantic_summary_v2.json"
    summary_path.write_text(
        json.dumps(safe_payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {"summary": str(summary_path), **safe_payload["summaries"]}, indent=2
        )
    )


if __name__ == "__main__":
    main()
