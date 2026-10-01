#!/usr/bin/env python3
"""Image-cluster inference for the v2 bidirectionally matched IDRiD audit."""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sys
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.origin_v3_cv_common import (
    canonical_sha256,
    file_sha256,
    read_json,
    verify_checksummed_payload,
    write_json_atomic,
)
from tools.analyze_origin_idrid_semantics import (
    LESIONS,
    SCALES,
    StatisticsConfig,
    _alignment_specs,
    _finite,
    _metric_inference,
    _paired_tests,
    _write_jsonl,
    aggregate_alignment_image_units,
)
from tools.audit_origin_idrid_semantics_v2 import (
    AP_CONTRACT,
    AUDIT_SCHEMA,
    AUDIT_SCOPE,
    DELETION_KIND,
    MATCHING_CONTRACT,
    _stable_seed,
)


PROTOCOL_SCHEMA = "origin-idrid-semantic-statistics-protocol-v2"
ALIGNMENT_SCHEMA = "origin-idrid-semantic-alignment-statistics-v2"
DELETION_SCHEMA = "origin-idrid-semantic-deletion-statistics-v2"
UNIT_SCHEMA = "origin-idrid-semantic-image-unit-v2"
MANIFEST_SCHEMA = "origin-idrid-semantic-statistics-manifest-v2"
SCALE_GROUPS = {
    "overall": SCALES,
    "fine": ("s4", "s8"),
    "coarse": ("s16", "s32"),
}
RAW_GROUPS = {"overall": "all", "fine": "fine", "coarse": "coarse"}


def _mean(values: Iterable[Any]) -> float | None:
    finite = [value for item in values if (value := _finite(item)) is not None]
    return None if not finite else float(np.mean(finite))


def load_protocol(path: str | Path) -> tuple[dict[str, Any], StatisticsConfig]:
    payload = read_json(path)
    verify_checksummed_payload(payload, schema=PROTOCOL_SCHEMA)
    expected = {
        "input_schema": AUDIT_SCHEMA,
        "matching_contract": MATCHING_CONTRACT,
        "alignment_average_precision_definition": AP_CONTRACT,
        "inference_unit": "idrid_image_after_repeat_then_checkpoint_aggregation",
        "interpretation_scope": "stored_internal_ledger_not_causal_pixel_intervention",
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise ValueError(f"IDRiD v2 protocol field {key!r} changed")
    config = payload.get("config")
    if not isinstance(config, Mapping):
        raise TypeError("IDRiD v2 statistics config is missing")
    return payload, StatisticsConfig(
        bootstrap_samples=config["bootstrap_samples"],
        bootstrap_seed=config["bootstrap_seed"],
        confidence_level=config["confidence_level"],
        permutation_samples=config["permutation_samples"],
        permutation_seed=config["permutation_seed"],
        exact_sign_flip_max_n=config["exact_sign_flip_max_n"],
        zero_tolerance=config["zero_tolerance"],
    )


def _identity(record: Mapping[str, Any]) -> tuple[int, str]:
    return int(record["fold"]), str(record["image_id"])


def validate_input_records_v2(
    summary: Mapping[str, Any],
    records: Sequence[Mapping[str, Any]],
    *,
    expected_folds: Sequence[int],
    expected_images: int,
    expected_paired_repeats: int,
    expected_matching_seed: int,
    expected_dense_image_scale_cases: int,
    expected_checkpoint_sha256_by_fold: Mapping[str, str],
    expected_architecture_signature: str,
    expected_config_signature: str,
    expected_source_commit: str,
) -> dict[str, Any]:
    unsigned_summary = dict(summary)
    summary_checksum = unsigned_summary.pop("content_checksum_sha256", None)
    if summary_checksum != canonical_sha256(unsigned_summary):
        raise AssertionError("IDRiD v2 audit summary checksum mismatch")
    if summary.get("schema") != AUDIT_SCHEMA or summary.get("scope") != AUDIT_SCOPE:
        raise ValueError("unexpected IDRiD v2 semantic input contract")
    if summary.get("matching_contract") != MATCHING_CONTRACT:
        raise ValueError("unexpected IDRiD v2 matching contract")
    if summary.get("average_precision_definition") != AP_CONTRACT:
        raise ValueError("unexpected IDRiD v2 average-precision contract")
    if int(summary.get("idrid_images", -1)) != expected_images:
        raise AssertionError("IDRiD image count changed")
    if int(summary.get("paired_repeats", -1)) != expected_paired_repeats:
        raise AssertionError("paired-repeat count changed")
    if int(summary.get("seed", -1)) != expected_matching_seed:
        raise AssertionError("matching seed changed")
    checkpoints = summary.get("checkpoints")
    if not isinstance(checkpoints, list):
        raise TypeError("checkpoint metadata are missing")
    checkpoint_folds = [int(item["fold"]) for item in checkpoints]
    if sorted(checkpoint_folds) != sorted(int(value) for value in expected_folds):
        raise AssertionError("checkpoint fold set changed")
    if len({str(item["checkpoint_sha256"]) for item in checkpoints}) != len(checkpoints):
        raise AssertionError("checkpoint hashes are not unique")
    expected_checkpoint_hashes = dict(expected_checkpoint_sha256_by_fold)
    if set(expected_checkpoint_hashes) != {str(int(value)) for value in expected_folds}:
        raise AssertionError("sealed checkpoint-hash map does not cover the expected folds")
    if any(
        len(value) != 64 or any(character not in "0123456789abcdef" for character in value)
        for value in expected_checkpoint_hashes.values()
    ):
        raise AssertionError("sealed checkpoint hash is not canonical SHA-256")
    observed_checkpoint_hashes = {
        str(int(item["fold"])): str(item["checkpoint_sha256"])
        for item in checkpoints
    }
    if observed_checkpoint_hashes != expected_checkpoint_hashes:
        raise AssertionError("checkpoint hashes do not match the sealed v3 identities")
    if {
        str(item.get("architecture_signature")) for item in checkpoints
    } != {expected_architecture_signature}:
        raise AssertionError("checkpoint architecture signature changed")
    if {str(item.get("config_signature")) for item in checkpoints} != {
        expected_config_signature
    }:
        raise AssertionError("checkpoint config signature changed")
    reference_counts = {float(item["reference_count"]) for item in checkpoints}
    if len(reference_counts) != 1:
        raise AssertionError("checkpoints disagree on reference_count")
    reference_count = reference_counts.pop()

    generation = summary.get("generation")
    if not isinstance(generation, Mapping):
        raise TypeError("IDRiD v2 generation provenance is missing")
    if str(generation.get("source_commit")) != expected_source_commit:
        raise AssertionError("IDRiD v2 audit was generated by a different source commit")
    if str(generation.get("audit_implementation_sha256")) != file_sha256(
        REPO_ROOT / "tools" / "audit_origin_idrid_semantics_v2.py"
    ):
        raise AssertionError("IDRiD v2 audit implementation hash changed after generation")
    if str(generation.get("v1_audit_dependency_sha256")) != file_sha256(
        REPO_ROOT / "tools" / "audit_origin_idrid_semantics.py"
    ):
        raise AssertionError("IDRiD v1 audit dependency hash changed after generation")
    if generation.get("checkpoint_order") != checkpoint_folds:
        raise AssertionError("generation checkpoint order does not replay")
    inventory = summary.get("input_dataset_inventory")
    if not isinstance(inventory, Mapping):
        raise TypeError("IDRiD input dataset inventory is missing")
    unsigned_inventory = dict(inventory)
    inventory_checksum = unsigned_inventory.pop("content_checksum_sha256", None)
    if inventory_checksum != canonical_sha256(unsigned_inventory):
        raise AssertionError("IDRiD input dataset inventory checksum mismatch")
    files = inventory.get("files")
    if (
        not isinstance(files, list)
        or int(inventory.get("file_count", -1)) != len(files)
        or len({str(item.get("path")) for item in files}) != len(files)
    ):
        raise AssertionError("IDRiD input dataset inventory is malformed")
    for item in files:
        if not isinstance(item, Mapping):
            raise AssertionError("IDRiD input inventory row is not a mapping")
        digest = str(item.get("sha256", ""))
        relative = Path(str(item.get("path", "")))
        if (
            not relative.parts
            or relative.is_absolute()
            or ".." in relative.parts
            or int(item.get("bytes", -1)) < 0
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise AssertionError("IDRiD input inventory row is malformed")

    alignment_keys: set[tuple[Any, ...]] = set()
    deletion_keys: set[tuple[Any, ...]] = set()
    image_metadata: dict[str, tuple[int, str, int]] = {}
    prevalence: dict[tuple[str, str, str], float] = {}
    lattice_census: dict[tuple[str, str], tuple[int, int, int]] = {}
    dense_cases: set[tuple[str, str]] = set()
    unequal_raw_sides: set[tuple[str, str]] = set()
    for record in records:
        kind = record.get("kind")
        fold, image = _identity(record)
        if fold not in expected_folds:
            raise AssertionError("record references an unexpected checkpoint fold")
        grade, split = int(record["grade"]), str(record["split"])
        target_boundary = int(record.get("target_boundary", min(max(grade - 1, 0), 3)))
        if target_boundary != min(max(grade - 1, 0), 3):
            raise AssertionError("target boundary no longer follows the declared grade rule")
        metadata = (grade, split, target_boundary)
        if image in image_metadata and image_metadata[image] != metadata:
            raise AssertionError("image grade/split metadata differ across checkpoints")
        image_metadata[image] = metadata
        if kind == "alignment":
            scale, lesion, boundary = (
                str(record["scale"]), str(record["lesion"]), int(record["boundary"])
            )
            if scale not in SCALES or lesion not in LESIONS or boundary not in range(4):
                raise AssertionError("invalid alignment scale, lesion, or boundary")
            expected_group = "fine" if scale in SCALE_GROUPS["fine"] else "coarse"
            if record.get("scale_group") != expected_group:
                raise AssertionError("alignment scale-group label changed")
            if record.get("average_precision_definition") != AP_CONTRACT:
                raise AssertionError("alignment AP definition changed")
            key = (fold, image, scale, lesion, boundary)
            if key in alignment_keys:
                raise AssertionError("duplicate raw alignment record")
            alignment_keys.add(key)
            value = _finite(record.get("prevalence"))
            if value is None or not 0.0 <= value <= 1.0:
                raise AssertionError("invalid lesion prevalence")
            ap = _finite(record.get("average_precision"))
            enrichment = _finite(record.get("ap_over_prevalence"))
            if ap is not None:
                if not 0.0 < value < 1.0 or not 0.0 <= ap <= 1.0:
                    raise AssertionError("defined AP requires both label classes")
                if enrichment is None or not math.isclose(
                    enrichment, ap / value, rel_tol=1e-6, abs_tol=1e-8
                ):
                    raise AssertionError("AP/prevalence enrichment does not replay")
            elif enrichment is not None:
                raise AssertionError("undefined AP cannot have defined enrichment")
            prevalence_key = (image, scale, lesion)
            if prevalence_key in prevalence and not math.isclose(
                prevalence[prevalence_key], value, rel_tol=0.0, abs_tol=1e-12
            ):
                raise AssertionError("mask prevalence changed across folds or boundaries")
            prevalence[prevalence_key] = value
            continue
        if kind != DELETION_KIND:
            raise AssertionError(f"unknown v2 semantic record kind: {kind!r}")
        if record.get("matching_contract") != MATCHING_CONTRACT:
            raise AssertionError("deletion row has wrong matching contract")
        group = str(record["scale_group"])
        if group not in ("all", "fine", "coarse"):
            raise AssertionError("invalid deletion scale group")
        repeat = int(record["repeat"])
        if repeat not in range(expected_paired_repeats):
            raise AssertionError("invalid paired-repeat index")
        key = (fold, image, group, repeat)
        if key in deletion_keys:
            raise AssertionError("duplicate raw v2 deletion record")
        deletion_keys.add(key)
        counts = record.get("scale_counts")
        if not isinstance(counts, Mapping) or set(counts) != set(SCALES):
            raise AssertionError("deletion row lacks the exact four-scale census")
        included_scales = set(SCALES if group == "all" else SCALE_GROUPS[group])
        guided_total = control_total = 0
        guided_budget = control_budget = 0.0
        for scale in SCALES:
            item = counts[scale]
            if not isinstance(item, Mapping):
                raise AssertionError("scale matching census must be a mapping")
            valid = int(item["valid_cells"])
            lesion = int(item["lesion_candidates"])
            nonlesion = int(item["nonlesion_candidates"])
            matched = int(item["matched_cells"])
            guided = int(item["guided_cells"])
            control = int(item["control_cells"])
            included = bool(item["included_in_group"])
            expected_seed = _stable_seed(
                expected_matching_seed, fold, image, group, repeat, scale
            )
            if int(item["matching_seed"]) != expected_seed:
                raise AssertionError("per-scale matching seed does not replay")
            if valid != lesion + nonlesion:
                raise AssertionError("valid cells do not partition into both candidate sets")
            expected_matched = min(lesion, nonlesion) if scale in included_scales else 0
            if included != (scale in included_scales):
                raise AssertionError("scale inclusion label disagrees with group")
            if matched != expected_matched or guided != matched or control != matched:
                raise AssertionError("per-scale bidirectional matched counts are unequal")
            expected_fraction = 0.0 if valid == 0 else matched / valid
            if not math.isclose(
                float(item["nominal_valid_fraction"]), expected_fraction,
                rel_tol=0.0, abs_tol=1e-12,
            ):
                raise AssertionError("nominal scale fraction does not replay")
            geometry = float(item["geometry_weight"])
            expected_geometry = 0.0 if valid == 0 else reference_count / valid
            if not math.isclose(geometry, expected_geometry, rel_tol=2e-6, abs_tol=2e-6):
                raise AssertionError("stored geometry normalization does not replay")
            multiplier = float(item["target_boundary_pre_atom_multiplier"])
            expected_reference = matched * geometry
            expected_weighted = expected_reference * multiplier
            for prefix in ("guided", "control"):
                if not math.isclose(
                    float(item[f"{prefix}_nominal_reference_exposure"]),
                    expected_reference, rel_tol=1e-7, abs_tol=1e-8,
                ):
                    raise AssertionError("nominal reference exposure does not replay")
                if not math.isclose(
                    float(item[f"{prefix}_nominal_weighted_exposure"]),
                    expected_weighted, rel_tol=1e-7, abs_tol=1e-8,
                ):
                    raise AssertionError("nominal weighted exposure does not replay")
            lattice_key = (image, scale)
            census_value = (valid, lesion, nonlesion)
            if lattice_key in lattice_census and lattice_census[lattice_key] != census_value:
                raise AssertionError("image/scale candidate census changed across rows")
            lattice_census[lattice_key] = census_value
            if lesion > nonlesion:
                dense_cases.add(lattice_key)
            if lesion != nonlesion:
                unequal_raw_sides.add(lattice_key)
            guided_total += guided
            control_total += control
            guided_budget += float(item["guided_nominal_weighted_exposure"])
            control_budget += float(item["control_nominal_weighted_exposure"])
        if guided_total != int(record["guided_deleted_cells"]):
            raise AssertionError("guided total does not equal per-scale census")
        if control_total != int(record["control_deleted_cells"]):
            raise AssertionError("control total does not equal per-scale census")
        if guided_total != control_total:
            raise AssertionError("paired deletion cell totals differ")
        if not math.isclose(guided_budget, control_budget, rel_tol=0.0, abs_tol=1e-10):
            raise AssertionError("paired nominal weighted budgets differ")
        if not math.isclose(
            guided_budget, float(record["guided_nominal_weighted_exposure"]),
            rel_tol=1e-7, abs_tol=1e-8,
        ) or not math.isclose(
            control_budget, float(record["control_nominal_weighted_exposure"]),
            rel_tol=1e-7, abs_tol=1e-8,
        ):
            raise AssertionError("record-level nominal budget does not replay")
        metric_names = (
            "guided_expected_grade_delta", "control_expected_grade_delta",
            "expected_grade_delta_lift", "guided_target_tail_delta",
            "control_target_tail_delta", "target_tail_delta_lift",
            "guided_target_rate_removed", "control_target_rate_removed",
        )
        values = {name: _finite(record.get(name)) for name in metric_names}
        if any(value is None for value in values.values()):
            raise AssertionError("deletion effects must be finite")
        if not math.isclose(
            values["expected_grade_delta_lift"],
            values["guided_expected_grade_delta"] - values["control_expected_grade_delta"],
            abs_tol=2e-6,
        ):
            raise AssertionError("expected-grade lift does not replay")
        if not math.isclose(
            values["target_tail_delta_lift"],
            values["guided_target_tail_delta"] - values["control_target_tail_delta"],
            abs_tol=2e-6,
        ):
            raise AssertionError("target-tail lift does not replay")

    images = sorted(image_metadata)
    if len(images) != expected_images:
        raise AssertionError("semantic records do not cover expected unique images")
    expected_alignment = len(expected_folds) * expected_images * len(SCALES) * len(LESIONS) * 4
    expected_deletion = (
        len(expected_folds) * expected_images * 3 * expected_paired_repeats
    )
    if len(alignment_keys) != expected_alignment or len(deletion_keys) != expected_deletion:
        raise AssertionError(
            "semantic row census is incomplete: "
            f"alignment={len(alignment_keys)}/{expected_alignment}, "
            f"deletion={len(deletion_keys)}/{expected_deletion}"
        )
    if len(dense_cases) != expected_dense_image_scale_cases:
        raise AssertionError(
            "dense image/scale census changed: "
            f"observed={len(dense_cases)}, expected={expected_dense_image_scale_cases}"
        )
    raw_matching_census = summary.get("summaries", {}).get("matching_census", {})
    expected_dense_ids = [f"{image}:{scale}" for image, scale in sorted(dense_cases)]
    if (
        int(raw_matching_census.get("dense_image_scale_cases", -1)) != len(dense_cases)
        or raw_matching_census.get("dense_image_scale_case_ids") != expected_dense_ids
        or int(raw_matching_census.get("v1_one_sided_control_would_truncate_cases", -1))
        != len(dense_cases)
    ):
        raise AssertionError("raw v2 summary does not reproduce the dense-case census")
    split_counts: dict[str, int] = defaultdict(int)
    for _, split, _ in image_metadata.values():
        split_counts[split] += 1
    return {
        "images": images,
        "alignment_rows": len(alignment_keys),
        "deletion_paired_repeat_rows": len(deletion_keys),
        "split_counts": dict(sorted(split_counts.items())),
        "unique_image_scale_lattices": len(lattice_census),
        "dense_image_scale_cases": len(dense_cases),
        "dense_image_scale_case_ids": expected_dense_ids,
        "unequal_candidate_side_image_scale_cases": len(unequal_raw_sides),
        "matching_rule": "m=min(n_lesion,n_nonlesion) independently at every scale",
        "reference_count": reference_count,
    }


def aggregate_deletion_image_units_v2(
    records: Sequence[Mapping[str, Any]],
    *,
    scale_group: str,
    grade: int | None = None,
    target_boundary: int | None = None,
    split: str | None = None,
    expected_folds: Sequence[int] = tuple(range(10)),
    expected_repeats: int = 20,
) -> list[dict[str, Any]]:
    if scale_group not in RAW_GROUPS:
        raise ValueError("unknown deletion scale group")
    raw_group = RAW_GROUPS[scale_group]
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for record in records:
        if record.get("kind") != DELETION_KIND or record.get("scale_group") != raw_group:
            continue
        if grade is not None and int(record["grade"]) != grade:
            continue
        if target_boundary is not None and int(record["target_boundary"]) != target_boundary:
            continue
        if split is not None and str(record["split"]) != split:
            continue
        grouped[str(record["image_id"])].append(record)
    metric_names = (
        "guided_expected_grade_delta", "control_expected_grade_delta",
        "expected_grade_delta_lift", "guided_target_tail_delta",
        "control_target_tail_delta", "target_tail_delta_lift",
        "guided_target_rate_removed", "control_target_rate_removed",
    )
    units: list[dict[str, Any]] = []
    for image, rows in sorted(grouped.items()):
        per_fold: list[dict[str, float]] = []
        matched_counts: set[int] = set()
        nominal_budgets: list[float] = []
        repeat_lifts: list[float] = []
        for fold in expected_folds:
            fold_rows = [row for row in rows if int(row["fold"]) == fold]
            if len(fold_rows) != expected_repeats or {
                int(row["repeat"]) for row in fold_rows
            } != set(range(expected_repeats)):
                raise AssertionError("image/fold lacks exactly one row per paired repeat")
            matched_counts.update(int(row["guided_deleted_cells"]) for row in fold_rows)
            for row in fold_rows:
                if int(row["guided_deleted_cells"]) != int(row["control_deleted_cells"]):
                    raise AssertionError("guided/control totals differ during aggregation")
                nominal_budgets.append(float(row["guided_nominal_weighted_exposure"]))
                repeat_lifts.append(float(row["expected_grade_delta_lift"]))
            fold_unit: dict[str, float] = {}
            for metric in metric_names:
                value = _mean(row[metric] for row in fold_rows)
                if value is None:
                    raise AssertionError("repeat-aggregated fold metric is non-finite")
                fold_unit[metric] = value
            per_fold.append(fold_unit)
        if len(matched_counts) != 1:
            raise AssertionError("matched cell count changed across folds or repeats")
        metrics: dict[str, float] = {}
        for metric in metric_names:
            value = _mean(fold[metric] for fold in per_fold)
            if value is None:
                raise AssertionError("checkpoint-aggregated image metric is non-finite")
            metrics[metric] = value
        if not math.isclose(
            metrics["expected_grade_delta_lift"],
            metrics["guided_expected_grade_delta"] - metrics["control_expected_grade_delta"],
            abs_tol=2e-6,
        ) or not math.isclose(
            metrics["target_tail_delta_lift"],
            metrics["guided_target_tail_delta"] - metrics["control_target_tail_delta"],
            abs_tol=2e-6,
        ):
            raise AssertionError("image-level paired lift does not replay")
        first = rows[0]
        units.append(
            {
                "schema": UNIT_SCHEMA,
                "row_type": "deletion_image_unit_v2",
                "image_id": image,
                "grade": int(first["grade"]),
                "split": str(first["split"]),
                "target_boundary": int(first["target_boundary"]),
                "scale_group": scale_group,
                "paired_repeats_aggregated_per_checkpoint": expected_repeats,
                "checkpoints_aggregated": len(expected_folds),
                "matched_cells_per_side_per_repeat": matched_counts.pop(),
                "mean_nominal_weighted_exposure_per_side": float(np.mean(nominal_budgets)),
                "repeat_level_expected_lift_sd_descriptive": float(np.std(repeat_lifts)),
                **metrics,
            }
        )
    return units


def _deletion_specs(records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grades = sorted({int(row["grade"]) for row in records if row.get("kind") == DELETION_KIND})
    boundaries = sorted(
        {int(row["target_boundary"]) for row in records if row.get("kind") == DELETION_KIND}
    )
    splits = sorted({str(row["split"]) for row in records if row.get("kind") == DELETION_KIND})
    specs: dict[str, dict[str, Any]] = {}

    def add(**spec: Any) -> None:
        key = "|".join(f"{name}={spec[name]}" for name in sorted(spec) if name != "family")
        specs.setdefault(key, {"spec_id": key, **spec})

    for group in SCALE_GROUPS:
        add(family="primary_scale", scale_group=group)
        for grade in grades:
            add(family="grade_stratum", scale_group=group, grade=grade)
        for boundary in boundaries:
            add(
                family="target_boundary_stratum", scale_group=group,
                target_boundary=boundary,
            )
    for split in splits:
        add(family="segmentation_split_stratum", scale_group="overall", split=split)
    return list(specs.values())


def analyze_idrid_semantics_v2(
    *,
    summary_path: Path,
    protocol_path: Path,
    output_dir: Path,
    expected_source_commit: str,
) -> dict[str, Any]:
    manifest_path = output_dir / "idrid_statistics_manifest_v2.json"
    if manifest_path.exists():
        raise FileExistsError("refusing to overwrite completed IDRiD v2 statistics")
    protocol, config = load_protocol(protocol_path)
    summary = read_json(summary_path)
    records_path = summary_path.parent / str(summary.get("records", ""))
    if not records_path.is_file():
        raise FileNotFoundError(records_path)
    try:
        records_path.resolve().relative_to(summary_path.parent.resolve())
    except ValueError:
        raise AssertionError("IDRiD v2 summary points outside its audit directory") from None
    if file_sha256(records_path) != summary.get("records_sha256"):
        raise AssertionError("IDRiD v2 semantic record checksum mismatch")
    records: list[dict[str, Any]] = []
    with records_path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid JSONL at line {line_number}") from error
            if not isinstance(row, dict):
                raise TypeError("IDRiD v2 JSONL rows must be objects")
            records.append(row)
    expected_folds = tuple(int(value) for value in protocol["expected_checkpoint_folds"])
    repeats = int(protocol["expected_paired_repeats"])
    census = validate_input_records_v2(
        summary,
        records,
        expected_folds=expected_folds,
        expected_images=int(protocol["expected_images"]),
        expected_paired_repeats=repeats,
        expected_matching_seed=int(protocol["expected_matching_seed"]),
        expected_dense_image_scale_cases=int(protocol["expected_dense_image_scale_cases"]),
        expected_checkpoint_sha256_by_fold=protocol[
            "expected_checkpoint_sha256_by_fold"
        ],
        expected_architecture_signature=str(protocol["expected_architecture_signature"]),
        expected_config_signature=str(protocol["expected_config_signature"]),
        expected_source_commit=expected_source_commit,
    )
    expected_split_counts = {
        str(key): int(value)
        for key, value in protocol["expected_segmentation_split_counts"].items()
    }
    if census["split_counts"] != expected_split_counts:
        raise AssertionError("IDRiD v2 segmentation split census changed")
    output_dir.mkdir(parents=True, exist_ok=True)

    alignment_statistics: list[dict[str, Any]] = []
    deletion_statistics: list[dict[str, Any]] = []
    unit_rows: list[dict[str, Any]] = []
    for spec in _alignment_specs(records):
        units = aggregate_alignment_image_units(
            records,
            scale_group=spec["scale_group"],
            lesion=spec["lesion"],
            boundary=spec["boundary"],
            grade=spec.get("grade"),
            split=spec.get("split"),
            expected_folds=expected_folds,
        )
        for unit in units:
            unit_rows.append({"spec_id": spec["spec_id"], **unit, "schema": UNIT_SCHEMA})
        eligible = [
            unit for unit in units
            if _finite(unit.get("average_precision")) is not None
            and _finite(unit.get("prevalence")) is not None
            and _finite(unit.get("ap_minus_prevalence")) is not None
        ]
        metrics = {
            metric: _metric_inference(
                eligible, metric, config=config, seed_label=f"alignment-v2:{spec['spec_id']}"
            )
            for metric in (
                "average_precision", "prevalence", "mean_ap_over_prevalence",
                "ratio_of_mean_ap_to_mean_prevalence", "ap_minus_prevalence",
                "mean_rate_ratio",
            )
        }
        differences = [float(unit["ap_minus_prevalence"]) for unit in eligible]
        alignment_statistics.append(
            {
                **spec,
                "n_images_in_stratum": len(units),
                "n_images_with_defined_ap": len(eligible),
                "metrics": metrics,
                "enrichment_tests_against_ap_equals_prevalence": (
                    None if not differences else _paired_tests(
                        differences, config=config,
                        seed_label=f"alignment-v2:{spec['spec_id']}:enrichment",
                    )
                ),
            }
        )

    for spec in _deletion_specs(records):
        units = aggregate_deletion_image_units_v2(
            records,
            scale_group=spec["scale_group"],
            grade=spec.get("grade"),
            target_boundary=spec.get("target_boundary"),
            split=spec.get("split"),
            expected_folds=expected_folds,
            expected_repeats=repeats,
        )
        for unit in units:
            unit_rows.append({"spec_id": spec["spec_id"], **unit})
        eligible = [
            unit for unit in units
            if int(unit["matched_cells_per_side_per_repeat"]) > 0
        ]
        metrics = {
            metric: _metric_inference(
                eligible, metric, config=config, seed_label=f"deletion-v2:{spec['spec_id']}"
            )
            for metric in (
                "guided_expected_grade_delta", "control_expected_grade_delta",
                "expected_grade_delta_lift", "guided_target_tail_delta",
                "control_target_tail_delta", "target_tail_delta_lift",
                "guided_target_rate_removed", "control_target_rate_removed",
            )
        }
        expected_lifts = [float(unit["expected_grade_delta_lift"]) for unit in eligible]
        tail_lifts = [float(unit["target_tail_delta_lift"]) for unit in eligible]
        deletion_statistics.append(
            {
                **spec,
                "n_images_in_stratum": len(units),
                "n_images_with_nonzero_matched_cells": len(eligible),
                "n_zero_match_images_excluded": len(units) - len(eligible),
                "metrics": metrics,
                "expected_grade_lift_tests": (
                    None if not expected_lifts else _paired_tests(
                        expected_lifts, config=config,
                        seed_label=f"deletion-v2:{spec['spec_id']}:expected",
                    )
                ),
                "target_tail_lift_tests": (
                    None if not tail_lifts else _paired_tests(
                        tail_lifts, config=config,
                        seed_label=f"deletion-v2:{spec['spec_id']}:tail",
                    )
                ),
            }
        )

    common = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol_checksum_sha256": protocol["content_checksum_sha256"],
        "input_summary_path": str(summary_path.resolve()),
        "input_summary_sha256": file_sha256(summary_path),
        "input_records_path": str(records_path.resolve()),
        "input_records_sha256": file_sha256(records_path),
        "input_census": census,
        "audit_generation": summary["generation"],
        "input_dataset_inventory_checksum_sha256": summary[
            "input_dataset_inventory"
        ]["content_checksum_sha256"],
        "inference_unit": (
            "IDRiD image after paired-repeat averaging within checkpoint, then "
            "averaging all 10 checkpoint evaluations"
        ),
        "cluster_limit": "IDRiD provides no patient linkage; the image is the available cluster",
        "interpretation_scope": "stored internal ledger intervention, not a causal pixel intervention",
        "matching_scope": (
            "unique lesion and non-lesion cells matched separately at every image and scale; "
            "cell count, scale composition, geometry exposure, and pre-atom multiplier equal"
        ),
    }
    alignment_payload: dict[str, Any] = {
        "schema": ALIGNMENT_SCHEMA,
        **common,
        "estimand": "image-weighted mean after within-image checkpoint and scale aggregation",
        "records": alignment_statistics,
        "multiplicity": "no adjustment; stratum intervals/tests are descriptive",
    }
    deletion_payload: dict[str, Any] = {
        "schema": DELETION_SCHEMA,
        **common,
        "estimand": (
            "image-weighted paired guided-minus-control stored-ledger deletion effect "
            "after nested repeat and checkpoint aggregation"
        ),
        "records": deletion_statistics,
        "multiplicity": "no adjustment; stratum intervals/tests are descriptive",
    }
    for payload in (alignment_payload, deletion_payload):
        payload["content_checksum_sha256"] = canonical_sha256(payload)
    alignment_path = output_dir / "IDRID_ALIGNMENT_CLUSTER_STATISTICS_V2.json"
    deletion_path = output_dir / "IDRID_DELETION_CLUSTER_STATISTICS_V2.json"
    unit_path = output_dir / "IDRID_IMAGE_LEVEL_UNITS_V2.jsonl"
    write_json_atomic(alignment_path, alignment_payload)
    write_json_atomic(deletion_path, deletion_payload)
    unit_count = _write_jsonl(unit_path, unit_rows)
    artifacts = {
        "alignment_statistics": {
            "path": str(alignment_path.resolve()),
            "sha256": file_sha256(alignment_path),
            "content_checksum_sha256": alignment_payload["content_checksum_sha256"],
            "records": len(alignment_statistics),
        },
        "deletion_statistics": {
            "path": str(deletion_path.resolve()),
            "sha256": file_sha256(deletion_path),
            "content_checksum_sha256": deletion_payload["content_checksum_sha256"],
            "records": len(deletion_statistics),
        },
        "image_level_units": {
            "path": str(unit_path.resolve()),
            "sha256": file_sha256(unit_path),
            "row_schema": UNIT_SCHEMA,
            "rows": unit_count,
        },
    }
    manifest: dict[str, Any] = {
        "schema": MANIFEST_SCHEMA,
        **common,
        "analysis_implementation_sha256": file_sha256(Path(__file__).resolve()),
        "audit_implementation_sha256": file_sha256(
            REPO_ROOT / "tools" / "audit_origin_idrid_semantics_v2.py"
        ),
        "v1_analysis_dependency_sha256": file_sha256(
            REPO_ROOT / "tools" / "analyze_origin_idrid_semantics.py"
        ),
        "artifacts": artifacts,
    }
    manifest["content_checksum_sha256"] = canonical_sha256(manifest)
    write_json_atomic(manifest_path, manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument(
        "--protocol",
        type=Path,
        default=REPO_ROOT / "scripts" / "protocols" / "origin_idrid_semantic_statistics_protocol_v2.json",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--expected-source-commit",
        required=True,
        help="full immutable Git commit recorded by the generating audit",
    )
    args = parser.parse_args()
    manifest = analyze_idrid_semantics_v2(
        summary_path=args.summary,
        protocol_path=args.protocol,
        output_dir=args.output_dir,
        expected_source_commit=args.expected_source_commit,
    )
    print(
        "ORIGIN IDRiD v2 semantic statistics complete: "
        f"manifest={args.output_dir / 'idrid_statistics_manifest_v2.json'} "
        f"checksum={manifest['content_checksum_sha256']}"
    )


if __name__ == "__main__":
    main()
