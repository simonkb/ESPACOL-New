#!/usr/bin/env python3
"""Audit one fold-9 ablation training run without touching its outer test set."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

try:  # Package import under pytest and ``python -m``.
    from scripts.origin_fold9_ablation_common import (
        DATASET,
        EXPECTED_SPLIT_COUNTS,
        FOLD,
        TRAINING_STREAM_SEED,
        TRAIN_MARKER_SCHEMA,
        canonical_sha256,
        config_signature_for_variant,
        file_sha256,
        read_json,
        validate_split_manifest,
        validate_variant_config,
        values_close,
        verify_checksummed_payload,
        verify_protocol,
        write_json_atomic,
    )
    from scripts.origin_v3_cv_common import metrics_from_confusion
except ModuleNotFoundError:  # Direct ``python scripts/<name>.py`` execution.
    from origin_fold9_ablation_common import (
        DATASET,
        EXPECTED_SPLIT_COUNTS,
        FOLD,
        TRAINING_STREAM_SEED,
        TRAIN_MARKER_SCHEMA,
        canonical_sha256,
        config_signature_for_variant,
        file_sha256,
        read_json,
        validate_split_manifest,
        validate_variant_config,
        values_close,
        verify_checksummed_payload,
        verify_protocol,
        write_json_atomic,
    )
    from origin_v3_cv_common import metrics_from_confusion


PRIMARY_METRICS = ("acc", "mae", "qwk", "balanced_acc", "macro_f1")


def _require_equal(observed: Any, expected: Any, label: str) -> None:
    if observed != expected:
        raise AssertionError(f"{label}: expected {expected!r}, observed {observed!r}")


def audit_training_run(
    *,
    variant: str,
    task_run_dir: Path,
    protocol_path: Path,
) -> dict[str, Any]:
    """Validate artifacts and emit the marker that unlocks outer release."""

    import torch

    protocol = read_json(protocol_path)
    selected = verify_protocol(protocol)
    if variant not in selected:
        raise ValueError(f"variant {variant!r} is not selected by this protocol")

    expected_task_dir = Path(protocol["experiment_root"]) / "workers" / variant
    if task_run_dir.resolve() != expected_task_dir.resolve():
        raise AssertionError(
            f"worker directory is outside the locked protocol: {task_run_dir}"
        )
    fold_dir = task_run_dir / f"fold{FOLD}"
    required = {
        "checkpoint": fold_dir / "best.pth",
        "result": fold_dir / "result.json",
        "split_manifest": fold_dir / "split_manifest.json",
        "history": fold_dir / "history.csv",
        "validation_certificate": fold_dir / "validation_certificates.json",
    }
    missing = [str(path) for path in required.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"training run is incomplete; missing {missing}")

    marker_path = fold_dir / "ABLATION_TRAIN_COMPLETE.json"
    if marker_path.exists():
        raise FileExistsError(f"refusing to overwrite training marker: {marker_path}")

    split = read_json(required["split_manifest"])
    validate_split_manifest(split)

    result = read_json(required["result"])
    _require_equal(result.get("fold"), FOLD, "result fold")
    _require_equal(
        result.get("training_stream_seed"),
        TRAINING_STREAM_SEED,
        "variant-independent training stream seed",
    )
    _require_equal(result.get("test_evaluated"), False, "outer-test seal")
    if result.get("test") not in (None, {}):
        raise AssertionError("validation-only result unexpectedly contains outer-test metrics")
    efficiency_scope = result.get("training_efficiency_scope")
    if efficiency_scope not in {
        "complete_training_process",
        "resumed_segment_only_not_cross_variant_comparable",
    }:
        raise AssertionError(
            f"invalid training_efficiency_scope: {efficiency_scope!r}"
        )
    reported_run_dir = result.get("run_dir")
    if reported_run_dir is not None and Path(str(reported_run_dir)).resolve() != fold_dir.resolve():
        raise AssertionError("result run_dir differs from the locked worker directory")
    validation = result.get("best_validation")
    if not isinstance(validation, Mapping):
        validation = result.get("best_validation_metrics")
    if not isinstance(validation, Mapping):
        raise TypeError("result has no selected inner-validation metrics")
    recomputed = metrics_from_confusion(validation.get("confusion", []))
    _require_equal(
        recomputed["n"], EXPECTED_SPLIT_COUNTS["validation"], "validation size"
    )
    _require_equal(
        recomputed["per_grade_support"],
        split["histograms"]["validation"],
        "validation class support",
    )
    for key in PRIMARY_METRICS:
        if not values_close(recomputed[key], validation.get(key)):
            raise AssertionError(
                f"validation {key} does not reproduce its confusion matrix"
            )

    checkpoint = torch.load(required["checkpoint"], map_location="cpu", weights_only=False)
    _require_equal(checkpoint.get("schema"), "origin-checkpoint-v3", "checkpoint schema")
    _require_equal(checkpoint.get("fold"), FOLD, "checkpoint fold")
    _require_equal(
        checkpoint.get("split_signature"),
        split["signature"],
        "checkpoint split signature",
    )
    _require_equal(checkpoint.get("epoch"), result.get("best_epoch"), "selected epoch")
    config = checkpoint.get("config")
    if not isinstance(config, Mapping):
        raise TypeError("checkpoint configuration is missing")
    validate_variant_config(config, variant)
    if Path(str(config.get("run_dir"))).resolve() != task_run_dir.resolve():
        raise AssertionError("checkpoint run_dir differs from the locked worker directory")
    checkpoint_metrics = checkpoint.get("metrics")
    if not isinstance(checkpoint_metrics, Mapping):
        raise TypeError("checkpoint selected metrics are missing")
    for key in PRIMARY_METRICS:
        if not values_close(checkpoint_metrics.get(key), validation.get(key)):
            raise AssertionError(f"checkpoint/result validation {key} mismatch")

    architecture = checkpoint.get("architecture")
    if not isinstance(architecture, Mapping):
        raise TypeError("checkpoint architecture declaration is missing")
    declared = architecture.get("declared")
    if not isinstance(declared, Mapping):
        raise TypeError("checkpoint architecture.declared is missing")
    _require_equal(
        declared.get("ablation_variant"), variant, "declared ablation variant"
    )
    architecture_signature = checkpoint.get("architecture_signature")
    implementation_signature = checkpoint.get("implementation_signature")
    checkpoint_config_signature = checkpoint.get("config_signature")
    for value, label in (
        (architecture_signature, "architecture signature"),
        (implementation_signature, "implementation signature"),
        (checkpoint_config_signature, "checkpoint config signature"),
    ):
        if not isinstance(value, str) or len(value) != 64:
            raise ValueError(f"checkpoint {label} is not a SHA-256 identity")

    certificate = read_json(required["validation_certificate"])
    verify_checksummed_payload(certificate)
    for key, expected in {
        "scope": "inner_validation_only",
        "fold": FOLD,
        "split_signature": split["signature"],
        "checkpoint_epoch": result["best_epoch"],
        "checkpoint_sha256": file_sha256(required["checkpoint"]),
        "implementation_signature": implementation_signature,
        "architecture_signature": architecture_signature,
    }.items():
        _require_equal(certificate.get(key), expected, f"validation certificate {key}")
    if certificate.get("ablation_variant") not in (None, variant):
        raise AssertionError("validation certificate variant mismatch")

    artifact_paths = {name: str(path.resolve()) for name, path in required.items()}
    artifact_hashes = {name: file_sha256(path) for name, path in required.items()}
    marker: dict[str, Any] = {
        "schema": TRAIN_MARKER_SCHEMA,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": DATASET,
        "fold": FOLD,
        "variant": variant,
        "evaluation_scope": "inner_validation_only_outer_locked",
        "outer_test_evaluated": False,
        "protocol_checksum_sha256": protocol["content_checksum_sha256"],
        "launch_commit": protocol["launch_commit"],
        "split_signature": split["signature"],
        "protocol_variant_config_signature": config_signature_for_variant(variant),
        "checkpoint_config_signature": checkpoint_config_signature,
        "implementation_signature": implementation_signature,
        "architecture_signature": architecture_signature,
        "architecture_declared": dict(declared),
        "best_epoch": int(result["best_epoch"]),
        "training_efficiency_scope": efficiency_scope,
        "validation": {
            **{key: recomputed[key] for key in ("n", *PRIMARY_METRICS)},
            "ece": float(validation["ece"]),
            "confusion": recomputed["confusion"],
            "per_grade_support": recomputed["per_grade_support"],
            "per_grade_recall": recomputed["per_grade_recall"],
        },
        "artifact_paths": artifact_paths,
        "artifact_sha256": artifact_hashes,
    }
    marker["content_checksum_sha256"] = canonical_sha256(marker)
    write_json_atomic(marker_path, marker)
    return {"marker": str(marker_path), "validation": marker["validation"]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", required=True)
    parser.add_argument("--task-run-dir", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    args = parser.parse_args()
    outcome = audit_training_run(
        variant=args.variant,
        task_run_dir=args.task_run_dir,
        protocol_path=args.protocol,
    )
    print(f"ORIGIN fold-9 ablation training audit passed: {outcome}")


if __name__ == "__main__":
    main()
