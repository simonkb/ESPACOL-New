#!/usr/bin/env python3
"""Validate completed audit artifacts from console-only comparator failures.

Comparator-v2 pooled tasks can reach the final console print after atomically
writing both their audit JSON and prediction JSONL.  The historical print
assumed localization metrics that are intentionally absent for a pooled
comparator, so Slurm recorded failure despite complete scientific artifacts.
This utility performs a read-only, fail-closed validation and writes a separate
operational recovery record.  It never edits the frozen protocol, checkpoint,
audit, or prediction artifact.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
from collections import Counter
from typing import Any, Mapping, Sequence

from scripts.aggregate_origin_shortcut_pilot import _read_audit
from scripts.origin_shortcut_comparator_common import (
    PROTOCOL_CORE_SHA256,
    canonical_sha256,
    task_at,
)


RECOVERABLE_TASKS = tuple(range(21, 42))
RECOVERY_SCHEMA = "origin-shortcut-comparator-console-recovery-v1"


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_protocol(path: Path, suite_root: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        protocol = json.load(stream)
    recorded = protocol.get("content_checksum_sha256")
    unsigned = dict(protocol)
    unsigned.pop("content_checksum_sha256", None)
    if recorded != canonical_sha256(unsigned):
        raise ValueError("comparator protocol checksum mismatch")
    if protocol.get("schema") != "origin-ordinal-shortcut-comparator-protocol-v2":
        raise ValueError("unexpected comparator protocol schema")
    if protocol.get("protocol_core_sha256") != PROTOCOL_CORE_SHA256:
        raise ValueError("comparator protocol core changed")
    if Path(str(protocol.get("suite_root"))).resolve() != suite_root.resolve():
        raise ValueError("comparator suite root differs from protocol")
    worktree = Path(str(protocol.get("immutable_worktree"))).resolve()
    commit = str(protocol.get("launch_commit", ""))
    if not worktree.is_dir() or len(commit) != 40:
        raise ValueError("protocol immutable worktree/commit is invalid")
    observed_commit = subprocess.run(
        ["git", "-C", str(worktree), "rev-parse", "HEAD"],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    ).stdout.strip()
    if observed_commit != commit:
        raise ValueError("immutable worktree no longer matches launch commit")
    tracked = subprocess.run(
        [
            "git",
            "-C",
            str(worktree),
            "status",
            "--porcelain",
            "--untracked-files=no",
        ],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    ).stdout.strip()
    if tracked:
        raise ValueError("immutable comparator worktree contains tracked changes")
    return protocol


def validate_console_failure_artifacts(
    *,
    suite_root: Path,
    protocol_path: Path,
    task_ids: Sequence[int],
    training_disposition: str = "read_only_existing_artifacts",
    audits_created: int = 0,
    original_array_job: str | None = None,
    recovery_array_job: str | None = None,
    recovery_implementation_commit: str | None = None,
) -> dict[str, Any]:
    requested = tuple(int(value) for value in task_ids)
    if (
        not requested
        or len(requested) != len(set(requested))
        or tuple(sorted(requested)) != requested
        or not set(requested).issubset(RECOVERABLE_TASKS)
    ):
        raise ValueError(
            f"recovery tasks must be a sorted, unique subset of {RECOVERABLE_TASKS}; "
            f"observed {requested}"
        )
    protocol = _read_protocol(protocol_path.resolve(), suite_root.resolve())
    allowed_dispositions = {
        "read_only_existing_artifacts",
        "reused_complete_original_training",
        "fresh_training_after_original_task_cancellation",
    }
    if training_disposition not in allowed_dispositions:
        raise ValueError(f"invalid training disposition: {training_disposition}")
    if training_disposition != "read_only_existing_artifacts":
        expected_dispositions = {
            (
                "reused_complete_original_training"
                if task_id <= 26
                else "fresh_training_after_original_task_cancellation"
            )
            for task_id in requested
        }
        if expected_dispositions != {training_disposition}:
            raise ValueError(
                "training disposition conflicts with frozen original-task state: "
                f"tasks={requested}, disposition={training_disposition}"
            )
    if int(audits_created) < 0:
        raise ValueError("audits_created must be non-negative")
    for label, job_id in (
        ("original array", original_array_job),
        ("recovery array", recovery_array_job),
    ):
        if job_id is not None and (not str(job_id).isdigit()):
            raise ValueError(f"{label} job id must be numeric")
    if recovery_implementation_commit is not None:
        if len(recovery_implementation_commit) != 40:
            raise ValueError("recovery implementation commit must be a full SHA")
        recovery_root = Path(__file__).resolve().parents[1]
        observed_recovery_commit = subprocess.run(
            ["git", "-C", str(recovery_root), "rev-parse", "HEAD"],
            check=True,
            text=True,
            stdout=subprocess.PIPE,
        ).stdout.strip()
        if observed_recovery_commit != recovery_implementation_commit:
            raise ValueError("active recovery implementation commit mismatch")
    if (suite_root / "COMPARATOR_SHORTCUT_RESULTS_V2.json").exists():
        raise FileExistsError("expanded comparator aggregate already exists")

    records: list[dict[str, Any]] = []
    for task_id in requested:
        task = task_at(task_id)
        if task.model_variant != "pooled_conditional":
            raise AssertionError("recovery task is not a pooled-conditional cell")
        fold_dir = (
            suite_root
            / "workers"
            / task.model_variant
            / task.arm
            / task.family
            / f"seed{task.training_seed}"
            / "fold0"
        )
        expected_checkpoint = (fold_dir / "best_learned.pth").resolve()
        families = ("localized", "border", "diffuse") if task.arm == "clean" else (task.family,)
        audit_paths = (
            tuple(fold_dir / f"shortcut_audit_{family}.json" for family in families)
            if task.arm == "clean"
            else (fold_dir / "shortcut_audit.json",)
        )
        task_records: list[dict[str, Any]] = []
        for audit_family, audit_path in zip(families, audit_paths):
            if not audit_path.is_file():
                raise FileNotFoundError(
                    f"task {task_id} did not finish its atomic audit write: {audit_path}"
                )
            audit_file_hash_before = _file_sha256(audit_path)
            report = _read_audit(audit_path)
            audit_file_hash_after = _file_sha256(audit_path)
            if audit_file_hash_before != audit_file_hash_after:
                raise RuntimeError("read-only recovery changed an audit artifact")
            expected_identity = {
                "dataset": "aptos",
                "model_variant": task.model_variant,
                "shortcut_arm": task.arm,
                "shortcut_family": audit_family,
                "training_seed": task.training_seed,
                "fold": 0,
                "split_seed": 42,
                "complete_outer_fold": True,
                "official_author_implementation": False,
            }
            mismatches = {
                key: {"expected": expected, "observed": report.get(key)}
                for key, expected in expected_identity.items()
                if report.get(key) != expected
            }
            if mismatches:
                raise ValueError(f"task {task_id} audit identity mismatch: {mismatches}")
            if report.get("local_ledger_applicable") is not False:
                raise ValueError("pooled comparator fabricated a local ledger")
            localization = report.get("localization")
            effects = report.get("internal_pixel_effects")
            if not isinstance(localization, Mapping) or localization.get("applicable") is not False:
                raise ValueError("pooled localization non-applicability is missing")
            if not isinstance(effects, Mapping) or effects.get("applicable") is not False:
                raise ValueError("pooled intervention non-applicability is missing")
            if "macro_auprc" in localization or "internal_pixel_spearman" in effects:
                raise ValueError("pooled report contains fabricated spatial metrics")
            if int(report.get("sample_count", -1)) != 733:
                raise ValueError("pooled recovery requires the complete 733-image outer fold")
            checkpoint_path = Path(str(report["checkpoint"])).resolve()
            if checkpoint_path != expected_checkpoint:
                raise ValueError(
                    f"task {task_id} checkpoint path mismatch: "
                    f"expected {expected_checkpoint}, observed {checkpoint_path}"
                )
            expected_prediction_path = audit_path.with_name(
                f"{audit_path.stem}_predictions.jsonl"
            ).resolve()
            prediction_path = Path(str(report["prediction_artifact"])).resolve()
            if prediction_path != expected_prediction_path:
                raise ValueError(
                    f"task {task_id} prediction path mismatch: "
                    f"expected {expected_prediction_path}, observed {prediction_path}"
                )
            prediction_records: list[dict[str, Any]] = []
            with prediction_path.open(encoding="utf-8") as stream:
                for line_number, line in enumerate(stream, start=1):
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError as exc:
                        raise ValueError(
                            f"invalid prediction JSONL at line {line_number}"
                        ) from exc
                    if not isinstance(record, dict):
                        raise ValueError("prediction JSONL rows must be objects")
                    prediction_records.append(record)
            if not prediction_records:
                raise ValueError("prediction artifact is empty")
            first = prediction_records[0]
            manifest_identity = {
                "record_type": "manifest",
                "schema": "origin-ordinal-shortcut-predictions-v1",
                "dataset": "aptos",
                "model_variant": task.model_variant,
                "shortcut_arm": task.arm,
                "shortcut_family": audit_family,
                "training_seed": task.training_seed,
                "fold": 0,
                "split_seed": 42,
                "checkpoint": str(expected_checkpoint),
                "checkpoint_sha256": report["checkpoint_sha256"],
            }
            manifest_mismatches = {
                key: {"expected": expected, "observed": first.get(key)}
                for key, expected in manifest_identity.items()
                if first.get(key) != expected
            }
            if manifest_mismatches:
                raise ValueError(
                    f"task {task_id} prediction manifest mismatch: {manifest_mismatches}"
                )
            expected_count_fields = (
                "condition_prediction_records",
                "factorial_prediction_records",
                "internal_effect_records",
            )
            record_type_to_field = {
                "condition_prediction": "condition_prediction_records",
                "factorial_prediction": "factorial_prediction_records",
                "internal_pixel_effect": "internal_effect_records",
            }
            observed_types = Counter(
                str(record.get("record_type"))
                for record in prediction_records[1:]
            )
            unexpected_types = set(observed_types) - set(record_type_to_field)
            if unexpected_types:
                raise ValueError(
                    f"unexpected prediction record types: {sorted(unexpected_types)}"
                )
            for record_type, count_field in record_type_to_field.items():
                observed_count = observed_types[record_type]
                manifest_count = first.get(count_field)
                report_count = report.get(count_field)
                if manifest_count != observed_count or report_count != observed_count:
                    raise ValueError(
                        f"prediction count mismatch for {count_field}: "
                        f"JSONL={observed_count}, manifest={manifest_count}, "
                        f"report={report_count}"
                    )
            if len(prediction_records) != 1 + sum(
                int(first[field]) for field in expected_count_fields
            ):
                raise ValueError("prediction artifact row total is inconsistent")
            if int(first["internal_effect_records"]) != 0:
                raise ValueError("pooled comparator must not claim internal ledger effects")
            task_records.append(
                {
                    "family": audit_family,
                    "audit": str(audit_path.resolve()),
                    "audit_sha256": audit_file_hash_before,
                    "audit_content_checksum_sha256": report[
                        "content_checksum_sha256"
                    ],
                    "prediction_artifact": str(prediction_path),
                    "prediction_artifact_sha256": report[
                        "prediction_artifact_sha256"
                    ],
                    "checkpoint": str(checkpoint_path),
                    "checkpoint_sha256": report["checkpoint_sha256"],
                    "sample_count": int(report["sample_count"]),
                }
            )
        records.append(
            {"task_id": task_id, "task_key": task.key, "audits": task_records}
        )

    expected_audit_count = sum(
        3 if task_at(task_id).arm == "clean" else 1 for task_id in requested
    )
    if int(audits_created) > expected_audit_count:
        raise ValueError(
            f"audits_created exceeds task audit census: {audits_created} > "
            f"{expected_audit_count}"
        )

    payload: dict[str, Any] = {
        "schema": RECOVERY_SCHEMA,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "read_only_validation_of_atomic_artifacts_after_console_only_failure",
        "failure_cause": (
            "terminal summary indexed intentionally absent pooled localization metrics "
            "after audit and prediction artifacts were atomically written"
        ),
        "scientific_protocol_modified": False,
        "validator_modified_scientific_artifacts": False,
        "recovery_execution": {
            "training_disposition": training_disposition,
            "audits_created": int(audits_created),
            "scientific_artifacts_created": bool(
                training_disposition == "fresh_training_after_original_task_cancellation"
                or int(audits_created) > 0
            ),
            "original_array_job": original_array_job,
            "recovery_array_job": recovery_array_job,
            "recovery_implementation_commit": recovery_implementation_commit,
        },
        "suite_root": str(suite_root.resolve()),
        "protocol": str(protocol_path.resolve()),
        "protocol_file_sha256": _file_sha256(protocol_path),
        "protocol_content_checksum_sha256": protocol[
            "content_checksum_sha256"
        ],
        "launch_commit": protocol["launch_commit"],
        "validated_tasks": records,
    }
    payload["content_checksum_sha256"] = canonical_sha256(payload)
    return payload


def _write_json_exclusive(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("x", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite-root", required=True, type=Path)
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--task-ids", default="21,22")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--training-disposition",
        required=True,
        choices=(
            "reused_complete_original_training",
            "fresh_training_after_original_task_cancellation",
        ),
    )
    parser.add_argument("--audits-created", required=True, type=int)
    parser.add_argument("--original-array-job", required=True)
    parser.add_argument("--recovery-array-job", required=True)
    parser.add_argument("--recovery-implementation-commit", required=True)
    args = parser.parse_args()
    task_ids = tuple(int(token) for token in args.task_ids.split(",") if token)
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite recovery record: {args.output}")
    payload = validate_console_failure_artifacts(
        suite_root=args.suite_root,
        protocol_path=args.protocol,
        task_ids=task_ids,
        training_disposition=args.training_disposition,
        audits_created=args.audits_created,
        original_array_job=args.original_array_job,
        recovery_array_job=args.recovery_array_job,
        recovery_implementation_commit=args.recovery_implementation_commit,
    )
    _write_json_exclusive(args.output, payload)
    print(json.dumps({"output": str(args.output), "validated_tasks": list(task_ids)}))


if __name__ == "__main__":
    main()
