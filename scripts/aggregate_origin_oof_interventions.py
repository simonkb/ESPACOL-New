#!/usr/bin/env python3
"""Verify and aggregate per-fold ORIGIN OOF intervention artifacts."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import math
from pathlib import Path
import sys
from typing import Any, Iterable, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.audit_origin_oof_interventions import AtomicJsonlGzipWriter, iter_jsonl_gzip
from scripts.origin_oof_intervention_common import (
    AGGREGATE_SCHEMA,
    FOLD_SCHEMA,
    METHODS,
    ROW_SCHEMA,
    load_audit_protocol,
)
from scripts.origin_v3_cv_common import (
    ARCHITECTURE_SIGNATURE,
    BASE_COMMIT,
    DATASET_SPECS,
    IMPLEMENTATION_SIGNATURE,
    canonical_sha256,
    file_sha256,
    read_json,
    validate_pinned_config,
    verify_checksummed_payload,
    write_json_atomic,
)


@dataclass
class OnlineStats:
    count: int = 0
    total: float = 0.0
    total_square: float = 0.0

    def add(self, value: Any) -> None:
        if value is None or isinstance(value, bool):
            return
        try:
            number = float(value)
        except (TypeError, ValueError):
            return
        if not math.isfinite(number):
            return
        self.count += 1
        self.total += number
        self.total_square += number * number

    def record(self) -> dict[str, Any]:
        if self.count == 0:
            return {"n": 0, "mean": None, "sample_sd": None}
        mean = self.total / self.count
        variance = (
            0.0 if self.count < 2 else
            max(0.0, (self.total_square - self.count * mean * mean) / (self.count - 1))
        )
        return {
            "n": self.count,
            "mean": mean,
            "sample_sd": math.sqrt(variance) if self.count > 1 else None,
        }


def _ensure_artifact(manifest_path: Path, artifact: Mapping[str, Any]) -> Path:
    path = Path(str(artifact.get("path", "")))
    if not path.is_file():
        raise FileNotFoundError(f"missing fold audit artifact: {path}")
    try:
        path.resolve().relative_to(manifest_path.parent.resolve())
    except ValueError:
        raise AssertionError("fold manifest points outside its audit directory") from None
    if file_sha256(path) != artifact.get("sha256"):
        raise AssertionError(f"fold audit artifact checksum mismatch: {path}")
    if artifact.get("row_schema") != ROW_SCHEMA:
        raise AssertionError("fold audit row schema changed")
    return path


def _strata(row: Mapping[str, Any]) -> Iterable[tuple[str, str]]:
    yield "overall", "all"
    yield "true_grade", str(int(row["true_grade"]))
    yield "predicted_grade", str(int(row["predicted_grade"]))
    yield "correctness", "correct" if bool(row["correct"]) else "error"
    yield (
        "true_pred_correct",
        f"true={int(row['true_grade'])}|pred={int(row['predicted_grade'])}|"
        f"correct={bool(row['correct'])}",
    )


def aggregate_manifests(
    *,
    dataset: str,
    manifest_paths: Sequence[Path],
    output_dir: Path,
    expected_n_images: int,
    expected_folds: Sequence[int],
    audit_protocol_checksum: str,
    cv_protocol_checksum: str,
) -> dict[str, Any]:
    """Aggregate verified fold manifests without re-running a model."""

    if (output_dir / "audit_manifest.json").exists():
        raise FileExistsError("refusing to overwrite a completed OOF aggregate")
    if not manifest_paths:
        raise ValueError("at least one fold manifest is required")
    expected_folds = tuple(int(value) for value in expected_folds)
    manifests: list[tuple[Path, dict[str, Any]]] = []
    seen_folds: set[int] = set()
    config_hash: str | None = None
    audit_implementation_hash: str | None = None
    audit_common_implementation_hash: str | None = None
    for manifest_path in manifest_paths:
        manifest = read_json(manifest_path)
        verify_checksummed_payload(manifest, schema=FOLD_SCHEMA)
        fold = int(manifest.get("fold", -1))
        if fold in seen_folds:
            raise AssertionError(f"duplicate fold audit {fold}")
        seen_folds.add(fold)
        for key, expected in {
            "dataset": dataset,
            "scope": "every_locked_outer_fold_image_exact_stored_ledger_interventions",
            "audit_protocol_checksum_sha256": audit_protocol_checksum,
            "cv_protocol_checksum_sha256": cv_protocol_checksum,
            "implementation_signature": IMPLEMENTATION_SIGNATURE,
            "architecture_signature": ARCHITECTURE_SIGNATURE,
        }.items():
            if manifest.get(key) != expected:
                raise AssertionError(f"fold {fold} manifest {key} mismatch")
        if config_hash is None:
            config_hash = str(manifest["audit_config_sha256"])
        elif manifest.get("audit_config_sha256") != config_hash:
            raise AssertionError("fold intervention audit configs differ")
        if audit_implementation_hash is None:
            audit_implementation_hash = str(manifest.get("audit_implementation_sha256"))
            audit_common_implementation_hash = str(
                manifest.get("audit_common_implementation_sha256")
            )
        elif (
            manifest.get("audit_implementation_sha256") != audit_implementation_hash
            or manifest.get("audit_common_implementation_sha256")
            != audit_common_implementation_hash
        ):
            raise AssertionError("fold intervention audit implementations differ")
        manifests.append((manifest_path, manifest))
    if seen_folds != set(expected_folds):
        raise AssertionError(
            f"fold set mismatch: expected={sorted(expected_folds)}, observed={sorted(seen_folds)}"
        )
    manifests.sort(key=lambda item: int(item[1]["fold"]))
    output_dir.mkdir(parents=True, exist_ok=True)
    output_paths = {
        "image_rows": output_dir / "OOF_IMAGE_BOUNDARY_CENSUS.jsonl.gz",
        "curve_rows": output_dir / "OOF_IMAGE_BOUNDARY_CURVES.jsonl.gz",
        "summary_rows": output_dir / "OOF_BOOTSTRAP_IMAGE_BOUNDARY_SUMMARIES.jsonl.gz",
    }
    image_keys: set[str] = set()
    physical_paths: set[str] = set()
    boundary_keys: set[tuple[str, int]] = set()
    summary_stats: dict[tuple[int, str, str, str, str], OnlineStats] = {}
    total_images_from_manifests = 0
    fold_records: list[dict[str, Any]] = []
    with (
        AtomicJsonlGzipWriter(output_paths["image_rows"]) as image_writer,
        AtomicJsonlGzipWriter(output_paths["curve_rows"]) as curve_writer,
        AtomicJsonlGzipWriter(output_paths["summary_rows"]) as summary_writer,
    ):
        for manifest_path, manifest in manifests:
            fold = int(manifest["fold"])
            total_images_from_manifests += int(manifest["n_images"])
            artifacts = manifest.get("artifacts")
            if not isinstance(artifacts, Mapping) or set(artifacts) != set(output_paths):
                raise AssertionError(f"fold {fold} artifact set is incomplete")
            fold_paths = {
                name: _ensure_artifact(manifest_path, artifacts[name])
                for name in output_paths
            }
            boundaries = int(manifest["n_boundaries"])
            budgets = manifest["audit_config"]["cell_budget_fractions"]
            expected_rows = {
                "image_rows": int(manifest["n_images"]) * boundaries,
                "curve_rows": int(manifest["n_images"]) * boundaries * len(METHODS) * len(budgets),
                "summary_rows": int(manifest["n_images"]) * boundaries * len(METHODS),
            }
            for name, expected in expected_rows.items():
                if int(artifacts[name].get("rows", -1)) != expected:
                    raise AssertionError(f"fold {fold} {name} manifest row count is invalid")
            observed = {name: 0 for name in output_paths}
            fold_images: set[str] = set()
            fold_image_paths: dict[str, str] = {}
            for row in iter_jsonl_gzip(fold_paths["image_rows"]):
                observed["image_rows"] += 1
                if row.get("schema") != ROW_SCHEMA or row.get("row_type") != "image_boundary_census":
                    raise AssertionError("unexpected image-census row schema")
                if row.get("dataset") != dataset or int(row.get("fold", -1)) != fold:
                    raise AssertionError("image-census row fold identity changed")
                image_key = str(row["image_key"])
                boundary = int(row["boundary"])
                if boundary not in range(boundaries):
                    raise AssertionError("invalid boundary index")
                key = (image_key, boundary)
                if key in boundary_keys:
                    raise AssertionError("duplicate image-boundary census row")
                boundary_keys.add(key)
                fold_images.add(image_key)
                recorded_path = str(row["image_path_relative"])
                if image_key in fold_image_paths and fold_image_paths[image_key] != recorded_path:
                    raise AssertionError("image census rows disagree on image path")
                fold_image_paths[image_key] = recorded_path
                image_writer.write(row)
            if len(fold_images) != int(manifest["n_images"]):
                raise AssertionError(f"fold {fold} image census coverage is incomplete")
            for image_key in fold_images:
                if image_key in image_keys:
                    raise AssertionError("duplicate image key across outer folds")
                image_keys.add(image_key)
            for path in fold_image_paths.values():
                if path in physical_paths:
                    raise AssertionError("an image appears in more than one outer fold")
                physical_paths.add(path)
            curve_keys: set[tuple[str, int, str, int]] = set()
            for row in iter_jsonl_gzip(fold_paths["curve_rows"]):
                observed["curve_rows"] += 1
                if row.get("schema") != ROW_SCHEMA or row.get("row_type") != "image_boundary_curve_point":
                    raise AssertionError("unexpected curve row schema")
                if row.get("method") not in METHODS:
                    raise AssertionError("unknown intervention curve method")
                if row.get("dataset") != dataset or int(row.get("fold", -1)) != fold:
                    raise AssertionError("curve row fold identity changed")
                curve_key = (
                    str(row["image_key"]), int(row["boundary"]),
                    str(row["method"]), int(row["budget_index"]),
                )
                if curve_key in curve_keys:
                    raise AssertionError("duplicate image-boundary curve point")
                curve_keys.add(curve_key)
                curve_writer.write(row)
            summary_keys: set[tuple[str, int, str]] = set()
            for row in iter_jsonl_gzip(fold_paths["summary_rows"]):
                observed["summary_rows"] += 1
                if row.get("schema") != ROW_SCHEMA or row.get("row_type") != "image_boundary_curve_summary":
                    raise AssertionError("unexpected curve-summary row schema")
                if row.get("dataset") != dataset or int(row.get("fold", -1)) != fold:
                    raise AssertionError("curve-summary row fold identity changed")
                summary_key = (
                    str(row["image_key"]), int(row["boundary"]), str(row["method"])
                )
                if summary_key in summary_keys:
                    raise AssertionError("duplicate image-boundary curve summary")
                summary_keys.add(summary_key)
                summary_writer.write(row)
                boundary = int(row["boundary"])
                method = str(row["method"])
                for stratum_type, stratum_value in _strata(row):
                    for field, value in row.items():
                        if not (field.startswith("auc_") or field.startswith("ranked_advantage_auc_")):
                            continue
                        key = (boundary, method, stratum_type, stratum_value, field)
                        summary_stats.setdefault(key, OnlineStats()).add(value)
            for name, expected in expected_rows.items():
                if observed[name] != expected:
                    raise AssertionError(
                        f"fold {fold} {name} rows changed: expected={expected}, observed={observed[name]}"
                    )
            fold_records.append(
                {
                    "fold": fold,
                    "n_images": int(manifest["n_images"]),
                    "split_signature": manifest["split_signature"],
                    "manifest_path": str(manifest_path.resolve()),
                    "manifest_sha256": file_sha256(manifest_path),
                    "content_checksum_sha256": manifest["content_checksum_sha256"],
                }
            )
    if total_images_from_manifests != expected_n_images or len(image_keys) != expected_n_images:
        raise AssertionError(
            f"OOF image coverage mismatch: expected={expected_n_images}, "
            f"manifest_total={total_images_from_manifests}, unique={len(image_keys)}"
        )
    strata_records = [
        {
            "boundary": key[0],
            "method": key[1],
            "stratum_type": key[2],
            "stratum": key[3],
            "metric": key[4],
            **statistics.record(),
        }
        for key, statistics in sorted(summary_stats.items())
    ]
    strata_path = output_dir / "OOF_STRATIFIED_CURVE_SUMMARY.json"
    strata_payload: dict[str, Any] = {
        "schema": "origin-oof-intervention-strata-v1",
        "dataset": dataset,
        "n_images": expected_n_images,
        "bootstrap_guidance": (
            "Use OOF_BOOTSTRAP_IMAGE_BOUNDARY_SUMMARIES.jsonl.gz; resample cluster_id "
            "for EyePACS and image_key for APTOS, preserving paired methods within an image."
        ),
        "records": strata_records,
    }
    strata_payload["content_checksum_sha256"] = canonical_sha256(strata_payload)
    write_json_atomic(strata_path, strata_payload)
    artifacts = {
        name: {
            "path": str(path.resolve()), "sha256": file_sha256(path),
            "rows": {
                "image_rows": len(boundary_keys),
                "curve_rows": sum(
                    int(manifest["artifacts"]["curve_rows"]["rows"])
                    for _, manifest in manifests
                ),
                "summary_rows": sum(
                    int(manifest["artifacts"]["summary_rows"]["rows"])
                    for _, manifest in manifests
                ),
            }[name],
            "row_schema": ROW_SCHEMA,
        }
        for name, path in output_paths.items()
    }
    artifacts["strata_summary"] = {
        "path": str(strata_path.resolve()), "sha256": file_sha256(strata_path),
        "records": len(strata_records),
    }
    payload: dict[str, Any] = {
        "schema": AGGREGATE_SCHEMA,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": dataset,
        "scope": "complete_out_of_fold_exact_grouped_intervention_census",
        "n_images": expected_n_images,
        "folds": list(expected_folds),
        "audit_protocol_checksum_sha256": audit_protocol_checksum,
        "cv_protocol_checksum_sha256": cv_protocol_checksum,
        "audit_config_sha256": config_hash,
        "fold_audit_implementation_sha256": audit_implementation_hash,
        "fold_audit_common_implementation_sha256": audit_common_implementation_hash,
        "aggregate_implementation_sha256": file_sha256(Path(__file__).resolve()),
        "implementation_signature": IMPLEMENTATION_SIGNATURE,
        "architecture_signature": ARCHITECTURE_SIGNATURE,
        "fold_manifests": fold_records,
        "artifacts": artifacts,
        "bootstrap_unit": "cluster_id_for_EyePACS_image_key_for_APTOS",
        "interpretation_scope": "exact_frozen_stored_ledger_intervention_not_causal_pixel_intervention",
    }
    payload["content_checksum_sha256"] = canonical_sha256(payload)
    write_json_atomic(output_dir / "audit_manifest.json", payload)
    return payload


def aggregate_production(
    *,
    dataset: str,
    cv_root: Path,
    cv_protocol_path: Path,
    audit_protocol_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    spec = DATASET_SPECS[dataset]
    cv_protocol = read_json(cv_protocol_path)
    verify_checksummed_payload(cv_protocol, schema="origin-v3-full-cv-protocol-v1")
    for key, expected in {
        "dataset": dataset,
        "n_folds": spec["n_folds"],
        "n_images": spec["n_images"],
        "base_commit": BASE_COMMIT,
        "implementation_signature": IMPLEMENTATION_SIGNATURE,
        "architecture_signature": ARCHITECTURE_SIGNATURE,
        "config_signature": spec["config_signature"],
        "evaluation_scope": "outer_test_after_selection",
    }.items():
        if cv_protocol.get(key) != expected:
            raise AssertionError(f"CV protocol {key} mismatch")
    validate_pinned_config(cv_protocol.get("frozen_config", {}), dataset)
    if Path(cv_protocol["cv_root"]).resolve() != cv_root.resolve():
        raise AssertionError("CV root differs from frozen protocol")
    audit_protocol, _ = load_audit_protocol(audit_protocol_path)
    folds = list(range(int(spec["n_folds"])))
    manifests = [
        cv_root / "workers" / f"fold{fold}" / f"fold{fold}" /
        "oof_intervention_audit" / "audit_manifest.json"
        for fold in folds
    ]
    return aggregate_manifests(
        dataset=dataset,
        manifest_paths=manifests,
        output_dir=output_dir,
        expected_n_images=int(spec["n_images"]),
        expected_folds=folds,
        audit_protocol_checksum=audit_protocol["content_checksum_sha256"],
        cv_protocol_checksum=cv_protocol["content_checksum_sha256"],
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=tuple(DATASET_SPECS), required=True)
    parser.add_argument("--cv-root", type=Path, required=True)
    parser.add_argument("--cv-protocol", type=Path, required=True)
    parser.add_argument(
        "--audit-protocol", type=Path,
        default=REPO_ROOT / "scripts" / "protocols" / "origin_oof_intervention_protocol.json",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    payload = aggregate_production(
        dataset=args.dataset, cv_root=args.cv_root,
        cv_protocol_path=args.cv_protocol, audit_protocol_path=args.audit_protocol,
        output_dir=args.output_dir,
    )
    print(
        "ORIGIN OOF intervention aggregate complete: "
        f"dataset={payload['dataset']} n={payload['n_images']} "
        f"manifest={args.output_dir / 'audit_manifest.json'}"
    )


if __name__ == "__main__":
    main()
