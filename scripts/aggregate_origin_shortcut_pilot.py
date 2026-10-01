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

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.ordinal_shortcut import SHORTCUT_FAMILIES
from benchmarks.shortcut_metrics import GateBThresholds, assess_gate_b, json_ready


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
    for report in reports:
        summaries["macro_auprc"].append(float(report["localization"]["macro_auprc"]))
        summaries["macro_pointing_accuracy"].append(
            float(report["localization"]["macro_pointing_accuracy"])
        )
        summaries["effective_support_fraction"].append(
            float(report["localization"]["macro_effective_support_fraction"])
        )
        summaries["diagonal_minus_off_diagonal"].append(
            float(report["boundary_response_selectivity"]["diagonal_minus_off_diagonal"])
        )
        summaries["internal_pixel_spearman"].append(
            float(report["internal_pixel_effects"]["internal_pixel_spearman"])
        )
    return {
        "n_seeds": len(reports),
        "training_seeds": sorted(int(report["training_seed"]) for report in reports),
        "metrics": {
            name: {
                "values": values,
                "median": statistics.median(values),
                "minimum": min(values),
                "maximum": max(values),
            }
            for name, values in summaries.items()
        },
    }


def aggregate(root: Path, expected_seeds: tuple[int, ...]) -> dict[str, Any]:
    paths = sorted(root.rglob("shortcut_audit*.json"))
    if not paths:
        raise FileNotFoundError(f"no shortcut_audit*.json files below {root}")
    reports = [_read_audit(path) for path in paths]
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for report in reports:
        if not report.get("complete_outer_fold"):
            raise ValueError(f"smoke-test audit cannot enter pilot aggregate: {report['_path']}")
        key = (str(report["shortcut_arm"]), str(report["shortcut_family"]))
        grouped.setdefault(key, []).append(report)

    expected = set(expected_seeds)
    for family in SHORTCUT_FAMILIES:
        for arm in ("shortcut", "cue_only", "clean"):
            observed = set(_seed_index(grouped.get((arm, family), [])).keys())
            if observed != expected:
                raise ValueError(
                    f"{arm}/{family} seeds {sorted(observed)} != expected {sorted(expected)}"
                )

    thresholds = GateBThresholds()
    family_reports: dict[str, Any] = {}
    gate_passes: list[bool] = []
    for family in SHORTCUT_FAMILIES:
        shortcut = sorted(
            grouped.get(("shortcut", family), []), key=lambda item: item["training_seed"]
        )
        clean = sorted(
            grouped.get(("clean", family), []), key=lambda item: item["training_seed"]
        )
        cue_only = sorted(
            grouped.get(("cue_only", family), []), key=lambda item: item["training_seed"]
        )
        record: dict[str, Any] = {
            "shortcut": _descriptive(shortcut),
            "cue_only": _descriptive(cue_only),
            "clean": _descriptive(clean),
        }
        if family != "diffuse":
            record["gate_b"] = assess_gate_b(
                shortcut, clean, thresholds=thresholds
            )
            gate_passes.append(bool(record["gate_b"]["passed"]))
        else:
            support = [
                float(report["localization"]["macro_effective_support_fraction"])
                for report in shortcut
            ]
            record["gate_b"] = {
                "localization_gate_applicable": False,
                "reason": "diffuse illumination has global procedural support",
                "broad_support_values": support,
                "broad_support_median": statistics.median(support),
                "must_not_be_reported_as_focal_lesion": True,
            }
        family_reports[family] = record

    return {
        "schema": "origin-ordinal-shortcut-pilot-aggregate-v1",
        "audit_root": str(root.resolve()),
        "expected_training_seeds": list(expected_seeds),
        "gate_b_thresholds": thresholds.as_dict(),
        "families": family_reports,
        "localized_family_gates_passed": gate_passes,
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
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seeds", type=_parse_seeds, default=_parse_seeds("1701,2603,3907"))
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite aggregate: {args.output}")
    payload = aggregate(args.audit_root, args.seeds)
    unsigned = json_ready(payload)
    payload["content_checksum_sha256"] = _canonical_sha256(unsigned)
    _write_json_atomic(args.output, payload)
    print(json.dumps({"output": str(args.output), "passed": payload["passed"]}))


if __name__ == "__main__":
    main()
