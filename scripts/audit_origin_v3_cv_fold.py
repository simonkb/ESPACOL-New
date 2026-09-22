#!/usr/bin/env python3
"""Fail-closed audit of one frozen ORIGIN-v3 outer-CV fold."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from origin_v3_cv_common import (
    ARCHITECTURE_SIGNATURE,
    BASE_COMMIT,
    DATASET_SPECS,
    IMPLEMENTATION_SIGNATURE,
    canonical_sha256,
    file_sha256,
    metrics_from_confusion,
    read_json,
    validate_pinned_config,
    values_close,
    verify_checksummed_payload,
    write_json_atomic,
)


PRIMARY_METRICS = ("acc", "mae", "qwk", "balanced_acc", "macro_f1")


def require_equal(observed: Any, expected: Any, label: str) -> None:
    if observed != expected:
        raise AssertionError(f"{label}: expected {expected!r}, observed {observed!r}")


def audit_fold(
    *,
    dataset: str,
    fold: int,
    task_run_dir: Path,
    protocol_path: Path,
) -> dict[str, Any]:
    import torch

    spec = DATASET_SPECS[dataset]
    if fold not in range(int(spec["n_folds"])):
        raise ValueError(f"invalid {dataset} fold {fold}")

    protocol = read_json(protocol_path)
    verify_checksummed_payload(protocol, schema="origin-v3-full-cv-protocol-v1")
    require_equal(protocol.get("dataset"), dataset, "protocol dataset")
    require_equal(protocol.get("base_commit"), BASE_COMMIT, "protocol base commit")
    require_equal(protocol.get("implementation_signature"), IMPLEMENTATION_SIGNATURE, "protocol implementation")
    require_equal(protocol.get("architecture_signature"), ARCHITECTURE_SIGNATURE, "protocol architecture")
    require_equal(protocol.get("n_folds"), spec["n_folds"], "protocol fold count")
    require_equal(protocol.get("evaluation_scope"), "outer_test_after_selection", "protocol evaluation scope")
    frozen_config = protocol.get("frozen_config")
    if not isinstance(frozen_config, Mapping):
        raise TypeError("protocol frozen_config is missing")
    validate_pinned_config(frozen_config, dataset)
    expected_task_run_dir = Path(protocol["cv_root"]) / "workers" / f"fold{fold}"
    if task_run_dir.resolve() != expected_task_run_dir.resolve():
        raise AssertionError(
            f"task run directory is outside the locked protocol: {task_run_dir}"
        )

    fold_dir = task_run_dir / f"fold{fold}"
    required = {
        "result": fold_dir / "result.json",
        "split_manifest": fold_dir / "split_manifest.json",
        "checkpoint": fold_dir / "best.pth",
        "validation_certificate": fold_dir / "validation_certificates.json",
        "outer_predictions_csv": fold_dir / "outer_predictions.csv",
        "outer_predictions_manifest": fold_dir / "outer_predictions_manifest.json",
    }
    missing = [str(path) for path in required.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"fold is incomplete; missing {missing}")

    result = read_json(required["result"])
    split = read_json(required["split_manifest"])
    require_equal(result.get("fold"), fold, "result fold")
    require_equal(result.get("test_evaluated"), True, "outer-test authorization")
    require_equal(split.get("schema"), "origin-split-v2", "split schema")
    require_equal(split.get("dataset"), dataset, "split dataset")
    require_equal(split.get("fold"), fold, "split fold")
    require_equal(split.get("evaluation_scope"), "outer_test_after_selection", "split scope")

    checkpoint = torch.load(required["checkpoint"], map_location="cpu", weights_only=False)
    require_equal(checkpoint.get("schema"), "origin-checkpoint-v3", "checkpoint schema")
    require_equal(checkpoint.get("fold"), fold, "checkpoint fold")
    require_equal(checkpoint.get("split_signature"), split.get("signature"), "checkpoint split signature")
    require_equal(checkpoint.get("implementation_signature"), IMPLEMENTATION_SIGNATURE, "checkpoint implementation")
    require_equal(checkpoint.get("architecture_signature"), ARCHITECTURE_SIGNATURE, "checkpoint architecture")
    require_equal(checkpoint.get("config_signature"), spec["config_signature"], "checkpoint config signature")
    require_equal(checkpoint.get("epoch"), result.get("best_epoch"), "selected epoch")
    config = checkpoint.get("config")
    if not isinstance(config, Mapping):
        raise TypeError("checkpoint config is missing")
    validate_pinned_config(config, dataset)
    require_equal(
        Path(str(config.get("run_dir"))).resolve(),
        task_run_dir.resolve(),
        "checkpoint run directory",
    )

    checkpoint_metrics = checkpoint.get("metrics")
    validation_metrics = result.get("best_validation")
    if not isinstance(checkpoint_metrics, Mapping) or not isinstance(validation_metrics, Mapping):
        raise TypeError("selected validation metrics are missing")
    for key in PRIMARY_METRICS:
        if not values_close(checkpoint_metrics.get(key), validation_metrics.get(key)):
            raise AssertionError(f"selected validation metric mismatch for {key}")

    test = result.get("test")
    if not isinstance(test, Mapping):
        raise TypeError("result has no outer-test metrics")
    recomputed = metrics_from_confusion(test.get("confusion", []))
    expected_n = int(split["counts"]["locked_test"])
    require_equal(recomputed["n"], expected_n, "test size vs split")
    require_equal(int(test.get("n", -1)), expected_n, "reported test size")
    if "n" in result:
        require_equal(int(result["n"]), expected_n, "top-level reported test size")
    require_equal(recomputed["per_grade_support"], split["histograms"]["locked_test"], "test class support")
    require_equal(recomputed["per_grade_support"], test.get("per_grade_support"), "reported test support")
    if not values_close(recomputed["per_grade_recall"], test.get("per_grade_recall")):
        raise AssertionError("reported per-grade recall does not reproduce confusion")
    for key in PRIMARY_METRICS:
        if not values_close(recomputed[key], test.get(key)):
            raise AssertionError(
                f"outer-test {key} mismatch: recomputed={recomputed[key]!r}, reported={test.get(key)!r}"
            )
        if key not in result:
            raise AssertionError(f"top-level outer-test mirror {key} is missing")
        if not values_close(result[key], test.get(key)):
            raise AssertionError(f"top-level outer-test {key} does not reproduce nested test")

    certificate = read_json(required["validation_certificate"])
    verify_checksummed_payload(
        certificate, schema="origin-exact-validation-certificates-v3"
    )
    require_equal(
        certificate.get("content_checksum_sha256"),
        result.get("validation_certificate_checksum"),
        "validation certificate checksum",
    )
    require_equal(certificate.get("scope"), "inner_validation_only", "certificate scope")
    require_equal(certificate.get("fold"), fold, "certificate fold")
    require_equal(
        certificate.get("checkpoint_epoch"), result.get("best_epoch"), "certificate epoch"
    )
    require_equal(certificate.get("split_signature"), split.get("signature"), "certificate split")
    require_equal(certificate.get("implementation_signature"), IMPLEMENTATION_SIGNATURE, "certificate implementation")
    require_equal(
        certificate.get("architecture_signature"),
        ARCHITECTURE_SIGNATURE,
        "certificate architecture",
    )
    require_equal(certificate.get("decision_rule"), "class_map", "certificate decision rule")
    require_equal(
        certificate.get("checkpoint_sha256"),
        file_sha256(required["checkpoint"]),
        "certificate checkpoint hash",
    )

    prediction_export = read_json(required["outer_predictions_manifest"])
    verify_checksummed_payload(
        prediction_export, schema="origin-v3-outer-predictions-v1"
    )
    for key, expected in {
        "dataset": dataset,
        "fold": fold,
        "scope": "locked_outer_test_after_inner_validation_selection",
        "decision_rule": "class_map",
        "n": expected_n,
        "class_histogram": recomputed["per_grade_support"],
        "split_signature": split.get("signature"),
        "protocol_checksum_sha256": protocol["content_checksum_sha256"],
        "launch_commit": protocol["launch_commit"],
        "implementation_signature": IMPLEMENTATION_SIGNATURE,
        "architecture_signature": ARCHITECTURE_SIGNATURE,
        "config_signature": spec["config_signature"],
        "checkpoint_epoch": result.get("best_epoch"),
        "checkpoint_sha256": file_sha256(required["checkpoint"]),
        "result_sha256": file_sha256(required["result"]),
        "csv_path": str(required["outer_predictions_csv"].resolve()),
        "csv_sha256": file_sha256(required["outer_predictions_csv"]),
    }.items():
        require_equal(prediction_export.get(key), expected, f"prediction export {key}")
    exported_metrics = prediction_export.get("metrics")
    if not isinstance(exported_metrics, Mapping):
        raise TypeError("outer-prediction export metrics are missing")
    for key in (*PRIMARY_METRICS, "ece", "confusion", "per_grade_recall", "per_grade_support"):
        if not values_close(exported_metrics.get(key), test.get(key)):
            raise AssertionError(f"outer-prediction export does not reproduce {key}")

    marker: dict[str, Any] = {
        "schema": "origin-v3-cv-fold-complete-v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": dataset,
        "fold": fold,
        "evaluation_scope": "outer_test_after_selection",
        "protocol_checksum_sha256": protocol["content_checksum_sha256"],
        "launch_commit": protocol["launch_commit"],
        "implementation_signature": IMPLEMENTATION_SIGNATURE,
        "architecture_signature": ARCHITECTURE_SIGNATURE,
        "config_signature": spec["config_signature"],
        "split_signature": split["signature"],
        "best_epoch": int(result["best_epoch"]),
        "outer_test": {
            **{
                key: recomputed[key]
                for key in ("n", *PRIMARY_METRICS, "confusion", "per_grade_support")
            },
            "ece": float(exported_metrics["ece"]),
        },
        "artifact_sha256": {name: file_sha256(path) for name, path in required.items()},
        "artifact_paths": {name: str(path.resolve()) for name, path in required.items()},
    }
    marker["content_checksum_sha256"] = canonical_sha256(marker)
    marker_path = fold_dir / "CV_FOLD_COMPLETE.json"
    write_json_atomic(marker_path, marker)
    return {"marker": str(marker_path), "outer_test": marker["outer_test"]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=tuple(DATASET_SPECS), required=True)
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--task-run-dir", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    args = parser.parse_args()
    outcome = audit_fold(
        dataset=args.dataset,
        fold=args.fold,
        task_run_dir=args.task_run_dir,
        protocol_path=args.protocol,
    )
    print(f"ORIGIN-v3 fold audit passed: {outcome}")


if __name__ == "__main__":
    main()
