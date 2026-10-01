#!/usr/bin/env python3
"""Aggregate the preregistered three-seed APTOS shortcut pilot."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import sys
from pathlib import Path
from typing import Any, Mapping

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.ordinal_shortcut import SHORTCUT_FAMILIES
from benchmarks.shortcut_metrics import (
    GateBThresholds,
    assess_diffuse_nonfocal_gate,
    assess_gate_b,
    boundary_response_selectivity,
    internal_pixel_effect_audit,
    json_ready,
)
from models.origin_acceptance_baselines import ACCEPTANCE_BASELINE_VARIANTS


_FILE_HASH_CACHE: dict[str, str] = {}


def _file_sha256(path: Path) -> str:
    key = str(path.resolve())
    if key in _FILE_HASH_CACHE:
        return _FILE_HASH_CACHE[key]
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    result = digest.hexdigest()
    _FILE_HASH_CACHE[key] = result
    return result


def _canonical_sha256(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            payload, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
    ).hexdigest()


def _read_audit(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        payload = json.load(stream)
    if payload.get("schema") != "origin-ordinal-shortcut-audit-v1":
        raise ValueError(f"unexpected audit schema in {path}")
    recorded = payload.get("content_checksum_sha256")
    unsigned = dict(payload)
    unsigned.pop("content_checksum_sha256", None)
    observed = _canonical_sha256(unsigned)
    if recorded != observed:
        raise ValueError(f"audit checksum mismatch in {path}")
    prediction_artifact = Path(payload.get("prediction_artifact", ""))
    if not prediction_artifact.is_file():
        raise FileNotFoundError(
            f"audit prediction artifact is missing for {path}: {prediction_artifact}"
        )
    if _file_sha256(prediction_artifact) != payload.get("prediction_artifact_sha256"):
        raise ValueError(f"prediction artifact checksum mismatch for {path}")
    checkpoint = Path(payload.get("checkpoint", ""))
    if not checkpoint.is_file():
        raise FileNotFoundError(f"audit checkpoint is missing for {path}: {checkpoint}")
    if _file_sha256(checkpoint) != payload.get("checkpoint_sha256"):
        raise ValueError(f"checkpoint checksum mismatch for {path}")
    # Protocol-v1 summary files used an invalid flattened image×boundary
    # bootstrap.  The immutable JSONL contains every image/boundary effect, so
    # recompute valid image-cluster intervals without mutating the sealed v1
    # artifact.  This is an inference correction, not a new model run.
    effect_rows: dict[tuple[str, int], dict[str, Any]] = {}
    with prediction_artifact.open(encoding="utf-8") as stream:
        for line in stream:
            record = json.loads(line)
            if record.get("record_type") != "internal_pixel_effect":
                continue
            key = (str(record["sample_id"]), int(record["boundary"]))
            if key in effect_rows:
                raise ValueError(f"duplicate internal effect row {key} in {prediction_artifact}")
            effect_rows[key] = record
    if effect_rows:
        sample_ids = sorted({sample_id for sample_id, _ in effect_rows})
        boundaries = sorted({boundary for _, boundary in effect_rows})
        if boundaries != [0, 1, 2, 3]:
            raise ValueError("shortcut effects must contain boundaries 0..3")
        internal = np.empty((len(sample_ids), 4), dtype=np.float64)
        random = np.empty_like(internal)
        pixel = np.empty_like(internal)
        for row, sample_id in enumerate(sample_ids):
            for boundary in boundaries:
                record = effect_rows.get((sample_id, boundary))
                if record is None:
                    raise ValueError(
                        f"incomplete image cluster {sample_id} in {prediction_artifact}"
                    )
                internal[row, boundary] = record["normalized_internal_effect"]
                random[row, boundary] = record["normalized_random_effect"]
                pixel[row, boundary] = record["factorial_pixel_marginal_effect"]
        effects = internal_pixel_effect_audit(
            internal, random, pixel, bootstrap_replicates=2000, seed=7194
        )
        effects["per_boundary"] = [
            internal_pixel_effect_audit(
                internal[:, boundary],
                random[:, boundary],
                pixel[:, boundary],
                bootstrap_replicates=2000,
                seed=7204 + boundary,
            )
            for boundary in boundaries
        ]
        effects["recomputed_from_sealed_prediction_artifact"] = True
        payload["internal_pixel_effects"] = effects
        payload["effect_inference_revision"] = (
            "image-cluster bootstrap preserving all four boundary observations"
        )
    # Recompute from the sealed factorial response matrix so protocol-v1
    # reports that serialized an infinite perfect-selectivity ratio as null
    # remain adjudicable without changing any prediction.
    response_matrix = payload.get("boundary_response_matrix")
    if response_matrix is not None:
        payload["boundary_response_selectivity"] = boundary_response_selectivity(
            response_matrix
        )
    payload["_path"] = str(path.resolve())
    return payload


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as stream:
            json.dump(json_ready(payload), stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _seed_index(reports: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    result: dict[int, dict[str, Any]] = {}
    for report in reports:
        seed = int(report["training_seed"])
        if seed in result:
            raise ValueError(
                f"duplicate {report['shortcut_arm']}/{report['shortcut_family']} seed {seed}"
            )
        result[seed] = report
    return result


def _descriptive(reports: list[dict[str, Any]]) -> dict[str, Any]:
    if not reports:
        return {"n_seeds": 0}
    summaries: dict[str, list[float]] = {
        "macro_auprc": [],
        "macro_pointing_accuracy": [],
        "effective_support_fraction": [],
        "diagonal_minus_off_diagonal": [],
        "internal_pixel_spearman": [],
    }
    def append_if_finite(name: str, value: Any) -> None:
        if value is None:
            return
        number = float(value)
        if np.isfinite(number):
            summaries[name].append(number)

    for report in reports:
        localization = report["localization"]
        # Effective support remains meaningful for the diffuse negative-
        # locality family even though lesion-style localization is explicitly
        # inapplicable.  Do not silently discard it with the focal metrics.
        append_if_finite(
            "effective_support_fraction",
            localization.get("macro_effective_support_fraction"),
        )
        if localization.get(
            "localization_applicable", localization.get("applicable", True)
        ):
            append_if_finite("macro_auprc", localization["macro_auprc"])
            append_if_finite(
                "macro_pointing_accuracy", localization["macro_pointing_accuracy"]
            )
        append_if_finite(
            "diagonal_minus_off_diagonal",
            report["boundary_response_selectivity"]["diagonal_minus_off_diagonal"],
        )
        effects = report["internal_pixel_effects"]
        if effects.get("applicable", True):
            append_if_finite(
                "internal_pixel_spearman", effects["internal_pixel_spearman"]
            )
    return {
        "n_seeds": len(reports),
        "training_seeds": sorted(int(report["training_seed"]) for report in reports),
        "metrics": {
            name: {
                "values": values,
                "median": (statistics.median(values) if values else None),
                "minimum": (min(values) if values else None),
                "maximum": (max(values) if values else None),
            }
            for name, values in summaries.items()
        },
    }


def aggregate(
    root: Path,
    expected_seeds: tuple[int, ...],
    *,
    reference_root: Path | None = None,
) -> dict[str, Any]:
    paths = sorted(root.rglob("shortcut_audit*.json"))
    if reference_root is not None:
        paths.extend(sorted(reference_root.rglob("shortcut_audit*.json")))
    if not paths:
        raise FileNotFoundError(f"no shortcut_audit*.json files below {root}")
    reports = [_read_audit(path) for path in paths]
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for report in reports:
        if not report.get("complete_outer_fold"):
            raise ValueError(f"smoke-test audit cannot enter pilot aggregate: {report['_path']}")
        variant = str(report.get("model_variant", "origin_ctmc"))
        key = (
            variant,
            str(report["shortcut_arm"]),
            str(report["shortcut_family"]),
        )
        grouped.setdefault(key, []).append(report)

    expected = set(expected_seeds)
    observed_variants = {
        str(report.get("model_variant", "origin_ctmc")) for report in reports
    }
    if reference_root is not None and observed_variants != set(ACCEPTANCE_BASELINE_VARIANTS):
        raise ValueError(
            "expanded comparator aggregate requires exactly the registered model "
            f"variants; observed {sorted(observed_variants)}"
        )
    variants = tuple(
        variant
        for variant in ACCEPTANCE_BASELINE_VARIANTS
        if variant in observed_variants
    )
    for variant in variants:
        for family in SHORTCUT_FAMILIES:
            for arm in ("shortcut", "cue_only", "clean"):
                observed = set(
                    _seed_index(grouped.get((variant, arm, family), [])).keys()
                )
                if observed != expected:
                    raise ValueError(
                        f"{variant}/{arm}/{family} seeds {sorted(observed)} "
                        f"!= expected {sorted(expected)}"
                    )

    thresholds = GateBThresholds()
    model_reports: dict[str, Any] = {}
    gate_passes: list[bool] = []
    for variant in variants:
        family_reports: dict[str, Any] = {}
        for family in SHORTCUT_FAMILIES:
            shortcut = sorted(
                grouped.get((variant, "shortcut", family), []),
                key=lambda item: item["training_seed"],
            )
            clean = sorted(
                grouped.get((variant, "clean", family), []),
                key=lambda item: item["training_seed"],
            )
            cue_only = sorted(
                grouped.get((variant, "cue_only", family), []),
                key=lambda item: item["training_seed"],
            )
            record: dict[str, Any] = {
                "shortcut": _descriptive(shortcut),
                "cue_only": _descriptive(cue_only),
                "clean": _descriptive(clean),
                "behavioral": {
                    "aligned_minus_neutral_accuracy": [
                        float(item["condition_performance"]["aligned_minus_neutral_accuracy"])
                        for item in shortcut
                    ],
                    "aligned_minus_inverted_accuracy": [
                        float(item["condition_performance"]["aligned_minus_inverted_accuracy"])
                        for item in shortcut
                    ],
                    "cue_only_accuracy": [
                        float(item["condition_performance"]["cue_only"]["accuracy"])
                        for item in cue_only
                    ],
                },
            }
            local_applicable = all(
                bool(item.get("local_ledger_applicable", True)) for item in shortcut
            )
            if family != "diffuse" and local_applicable:
                record["gate_b"] = assess_gate_b(
                    shortcut, clean, cue_only, thresholds=thresholds
                )
                if variant == "origin_ctmc":
                    gate_passes.append(bool(record["gate_b"]["passed"]))
            elif family == "diffuse" and local_applicable:
                record["gate_b"] = assess_diffuse_nonfocal_gate(
                    shortcut, cue_only, thresholds=thresholds
                )
                if variant == "origin_ctmc":
                    gate_passes.append(bool(record["gate_b"]["passed"]))
            else:
                record["gate_b"] = {
                    "localization_gate_applicable": False,
                    "reason": "pooled comparator exposes no spatial prediction ledger",
                    "passed": False,
                    "must_not_be_reported_as_focal_lesion": True,
                }
            family_reports[family] = record
        model_reports[variant] = {
            "implementation_origin": reports[
                next(
                    index
                    for index, item in enumerate(reports)
                    if str(item.get("model_variant", "origin_ctmc")) == variant
                )
            ].get("comparator_implementation_origin", "in_repo_proposed_method"),
            "official_author_implementation": False,
            "families": family_reports,
        }

    return {
        "schema": "origin-ordinal-shortcut-pilot-aggregate-v2",
        "source_audit_schema": "origin-ordinal-shortcut-audit-v1",
        "inference_revision": (
            "image-cluster bootstrap retains all four boundary observations; "
            "cue-only accuracy is a necessary non-sufficient positive control; "
            "boundary selectivity and diffuse non-focality are fail-closed gates"
        ),
        "audit_root": str(root.resolve()),
        "expected_training_seeds": list(expected_seeds),
        "gate_b_thresholds": thresholds.as_dict(),
        "model_variants": model_reports,
        "origin_family_gates_passed": gate_passes,
        "passed": bool(gate_passes) and all(gate_passes),
        "audit_files": [report["_path"] for report in reports],
    }


def _parse_seeds(value: str) -> tuple[int, ...]:
    seeds = tuple(int(token.strip()) for token in value.split(",") if token.strip())
    if len(seeds) != 3 or len(set(seeds)) != 3:
        raise argparse.ArgumentTypeError("pilot requires exactly three distinct seeds")
    return seeds


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Aggregate complete APTOS controlled-shortcut audits",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--audit-root", required=True, type=Path)
    parser.add_argument(
        "--reference-root",
        type=Path,
        default=None,
        help="sealed protocol-v1 ORIGIN root imported without mutation",
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seeds", type=_parse_seeds, default=_parse_seeds("1701,2603,3907"))
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite aggregate: {args.output}")
    payload = aggregate(
        args.audit_root, args.seeds, reference_root=args.reference_root
    )
    unsigned = json_ready(payload)
    payload["content_checksum_sha256"] = _canonical_sha256(unsigned)
    _write_json_atomic(args.output, payload)
    print(json.dumps({"output": str(args.output), "passed": payload["passed"]}))


if __name__ == "__main__":
    main()
