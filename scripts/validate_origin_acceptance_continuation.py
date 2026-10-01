#!/usr/bin/env python3
"""Fail-closed validation and provenance for an acceptance-suite continuation.

This helper deliberately does not submit scheduler jobs.  It proves that a
sealed experiment contains exactly the completed five-worker canary, that its
parent submission matches the caller's explicit digest/job expectations, and
that no downstream stage has started.  The shell launcher consumes the
resulting preflight record and writes a checksummed continuation submission
only after all replacement jobs have been accepted by Slurm.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
from typing import Any, Mapping

from scripts.audit_origin_acceptance_baselines import audit_scope
from scripts.origin_acceptance_baseline_common import (
    PROTOCOL_ID,
    PROTOCOL_SHA256,
    canonical_sha256,
    canary_tasks,
    file_sha256,
)


def _load_object(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise TypeError(f"expected a JSON object at {path}")
    return value


def _verify_content_checksum(payload: Mapping[str, Any], *, label: str) -> None:
    expected = payload.get("content_checksum_sha256")
    unsigned = dict(payload)
    unsigned.pop("content_checksum_sha256", None)
    if expected != canonical_sha256(unsigned):
        raise ValueError(f"{label} content checksum mismatch")


def _resolved(value: str | Path) -> Path:
    return Path(value).expanduser().resolve()


def _git(path: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(path), *args],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def validate_parent_submission(
    *,
    experiment_root: Path,
    expected_submission_sha256: str,
    expected_canary_job: str,
    expected_failed_canary_audit_job: str,
    source_repository: Path,
    aptos_root: Path,
    dr_root: Path,
) -> dict[str, Any]:
    """Validate the immutable parent launch without trusting ambient state."""

    root = _resolved(experiment_root)
    submission_path = root / "SUBMISSION.json"
    if len(expected_submission_sha256) != 64:
        raise ValueError("expected parent submission SHA-256 must have 64 hex digits")
    try:
        int(expected_submission_sha256, 16)
    except ValueError as error:
        raise ValueError("expected parent submission SHA-256 is not hexadecimal") from error
    actual_digest = file_sha256(submission_path)
    if actual_digest != expected_submission_sha256.lower():
        raise ValueError("parent submission file digest mismatch")

    submission = _load_object(submission_path)
    _verify_content_checksum(submission, label="parent submission")
    if submission.get("schema") != "origin-acceptance-submission-v1":
        raise ValueError("parent submission schema mismatch")
    if submission.get("protocol_id") != PROTOCOL_ID:
        raise ValueError("parent protocol identifier mismatch")
    if submission.get("protocol_sha256") != PROTOCOL_SHA256:
        raise ValueError("parent protocol digest mismatch")

    jobs = submission.get("jobs")
    if not isinstance(jobs, dict):
        raise ValueError("parent submission has no scheduler job map")
    if str(jobs.get("aptos_fold0_canary_array")) != str(expected_canary_job):
        raise ValueError("parent canary job does not match explicit expectation")
    if str(jobs.get("canary_gate_audit")) != str(expected_failed_canary_audit_job):
        raise ValueError("parent failed canary-audit job does not match expectation")

    if _resolved(submission.get("source_repository", "")) != _resolved(source_repository):
        raise ValueError("parent source repository does not match this repository")
    data_roots = submission.get("data_roots")
    if not isinstance(data_roots, dict):
        raise ValueError("parent submission has no data-root binding")
    if _resolved(data_roots.get("aptos", "")) != _resolved(aptos_root):
        raise ValueError("APTOS data root differs from the sealed parent launch")
    if _resolved(data_roots.get("dr", "")) != _resolved(dr_root):
        raise ValueError("EyePACS data root differs from the sealed parent launch")

    old_snapshot = _resolved(submission.get("immutable_worktree", ""))
    if not old_snapshot.is_dir():
        raise FileNotFoundError(f"parent immutable worktree is missing: {old_snapshot}")
    old_commit = str(submission.get("launch_commit", ""))
    if len(old_commit) != 40 or _git(old_snapshot, "rev-parse", "HEAD") != old_commit:
        raise ValueError("parent immutable worktree commit mismatch")
    if _git(old_snapshot, "branch", "--show-current"):
        raise ValueError("parent immutable worktree is no longer detached")
    if _git(old_snapshot, "status", "--porcelain", "--untracked-files=no"):
        raise ValueError("parent immutable worktree has tracked modifications")

    return {
        "submission_path": str(submission_path),
        "submission_file_sha256": actual_digest,
        "submission_content_checksum_sha256": submission[
            "content_checksum_sha256"
        ],
        "parent_launch_commit": old_commit,
        "parent_immutable_worktree": str(old_snapshot),
        "parent_jobs": {key: str(value) for key, value in jobs.items()},
    }


def validate_canary_reuse(experiment_root: Path) -> dict[str, Any]:
    """Verify exact canary census and audit every immutable worker artifact."""

    root = _resolved(experiment_root)
    if (root / "RESTART_SUBMISSION.json").exists():
        raise ValueError("a continuation submission already exists")
    full_root = root / "full"
    if full_root.exists() and any(full_root.iterdir()):
        raise ValueError("full-suite artifacts already exist; refusing mixed continuation")

    training_root = root / "canary" / "training"
    if not training_root.is_dir():
        raise FileNotFoundError(training_root)
    expected = {task.key for task in canary_tasks()}
    actual = {entry.name for entry in training_root.iterdir() if entry.is_dir()}
    non_directories = [entry.name for entry in training_root.iterdir() if not entry.is_dir()]
    if actual != expected or non_directories:
        raise ValueError(
            "canary census mismatch: "
            f"missing={sorted(expected - actual)}, extra={sorted(actual - expected)}, "
            f"non_directories={sorted(non_directories)}"
        )

    audit = audit_scope(root, "canary")
    if audit.get("status") != "passed" or int(audit.get("worker_count", -1)) != len(expected):
        raise ValueError("canary artifact audit did not cover the exact task census")
    gate_path = root / "canary" / "CANARY_PASSED.json"
    gate_present = gate_path.is_file()
    if gate_present:
        gate = _load_object(gate_path)
        _verify_content_checksum(gate, label="existing canary gate")
        if gate != audit:
            raise ValueError("existing canary gate does not replay from sealed artifacts")
    return {
        "task_count": len(expected),
        "task_keys": sorted(expected),
        "training_audit_content_checksum_sha256": audit[
            "content_checksum_sha256"
        ],
        "valid_gate_already_present": gate_present,
        "worker_manifest_sha256": {
            worker["task"]["baseline_variant"]: worker["manifest_sha256"]
            for worker in audit["workers"]
        },
        "best_learned_sha256": {
            worker["task"]["baseline_variant"]: worker["best_learned_sha256"]
            for worker in audit["workers"]
        },
    }


def validate_continuation(
    *,
    experiment_root: Path,
    expected_submission_sha256: str,
    expected_canary_job: str,
    expected_failed_canary_audit_job: str,
    source_repository: Path,
    aptos_root: Path,
    dr_root: Path,
) -> dict[str, Any]:
    parent = validate_parent_submission(
        experiment_root=experiment_root,
        expected_submission_sha256=expected_submission_sha256,
        expected_canary_job=expected_canary_job,
        expected_failed_canary_audit_job=expected_failed_canary_audit_job,
        source_repository=source_repository,
        aptos_root=aptos_root,
        dr_root=dr_root,
    )
    canary = validate_canary_reuse(experiment_root)
    payload: dict[str, Any] = {
        "schema": "origin-acceptance-continuation-preflight-v1",
        "protocol_id": PROTOCOL_ID,
        "protocol_sha256": PROTOCOL_SHA256,
        "experiment_root": str(_resolved(experiment_root)),
        "parent": parent,
        "reused_canary": canary,
        "downstream_state": "absent",
    }
    payload["content_checksum_sha256"] = canonical_sha256(payload)
    return payload


def write_restart_submission(
    *,
    output: Path,
    preflight: Mapping[str, Any],
    launch_commit: str,
    immutable_worktree: Path,
    conda_environment: str,
    jobs: Mapping[str, str],
    scheduler_parent_states: Mapping[str, str],
    reaudit_dependency_mode: str,
) -> dict[str, Any]:
    _verify_content_checksum(preflight, label="continuation preflight")
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    if output.parent.resolve() != Path(str(preflight["experiment_root"])).resolve():
        raise ValueError("restart manifest must be written at the sealed experiment root")
    required_jobs = {
        "canary_reaudit",
        "full_training_array",
        "full_freeze_audit",
        "coordinated_outer_release_array",
        "release_aggregate",
    }
    if set(jobs) != required_jobs or any(not str(value).isdigit() for value in jobs.values()):
        raise ValueError("continuation scheduler job map is incomplete or malformed")
    if len(launch_commit) != 40:
        raise ValueError("continuation launch commit must be a full Git object id")
    if reaudit_dependency_mode not in {
        "afterok_live_reaudit",
        "verified_completed_no_dependency",
    }:
        raise ValueError("invalid canary re-audit dependency mode")

    payload: dict[str, Any] = {
        "schema": "origin-acceptance-restart-submission-v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol_id": PROTOCOL_ID,
        "protocol_sha256": PROTOCOL_SHA256,
        "experiment_root": str(Path(str(preflight["experiment_root"])).resolve()),
        "continuation_reason": (
            "parent canary audit failed before scientific auditing because direct "
            "script execution omitted the repository root from Python's import path"
        ),
        "scientific_protocol_changed": False,
        "canary_training_reused": True,
        "preflight": dict(preflight),
        "launch_commit": launch_commit,
        "immutable_worktree": str(Path(immutable_worktree).resolve()),
        "conda_environment": str(conda_environment),
        "scheduler_parent_states": dict(scheduler_parent_states),
        "canary_reaudit_dependency_mode": reaudit_dependency_mode,
        "jobs": {key: str(value) for key, value in jobs.items()},
    }
    payload["content_checksum_sha256"] = canonical_sha256(payload)

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("x", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    return payload


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate = subparsers.add_parser("validate")
    validate.add_argument("--experiment-root", type=Path, required=True)
    validate.add_argument("--expected-submission-sha256", required=True)
    validate.add_argument("--expected-canary-job", required=True)
    validate.add_argument("--expected-failed-canary-audit-job", required=True)
    validate.add_argument("--source-repository", type=Path, required=True)
    validate.add_argument("--aptos-root", type=Path, required=True)
    validate.add_argument("--dr-root", type=Path, required=True)
    validate.add_argument("--output", type=Path, required=True)

    write = subparsers.add_parser("write-manifest")
    write.add_argument("--output", type=Path, required=True)
    write.add_argument("--preflight", type=Path, required=True)
    write.add_argument("--launch-commit", required=True)
    write.add_argument("--immutable-worktree", type=Path, required=True)
    write.add_argument("--conda-environment", required=True)
    for name in (
        "canary-reaudit",
        "full-training-array",
        "full-freeze-audit",
        "coordinated-outer-release-array",
        "release-aggregate",
    ):
        write.add_argument(f"--job-{name}", required=True)
    write.add_argument("--parent-canary-state", required=True)
    write.add_argument("--parent-canary-audit-state", required=True)
    write.add_argument(
        "--reaudit-dependency-mode",
        choices=("afterok_live_reaudit", "verified_completed_no_dependency"),
        required=True,
    )

    args = parser.parse_args()
    if args.command == "validate":
        payload = validate_continuation(
            experiment_root=args.experiment_root,
            expected_submission_sha256=args.expected_submission_sha256,
            expected_canary_job=args.expected_canary_job,
            expected_failed_canary_audit_job=args.expected_failed_canary_audit_job,
            source_repository=args.source_repository,
            aptos_root=args.aptos_root,
            dr_root=args.dr_root,
        )
        _write_json(args.output, payload)
        print(json.dumps({"status": "passed", "output": str(args.output)}))
        return

    preflight = _load_object(args.preflight)
    jobs = {
        "canary_reaudit": args.job_canary_reaudit,
        "full_training_array": args.job_full_training_array,
        "full_freeze_audit": args.job_full_freeze_audit,
        "coordinated_outer_release_array": args.job_coordinated_outer_release_array,
        "release_aggregate": args.job_release_aggregate,
    }
    payload = write_restart_submission(
        output=args.output,
        preflight=preflight,
        launch_commit=args.launch_commit,
        immutable_worktree=args.immutable_worktree,
        conda_environment=args.conda_environment,
        jobs=jobs,
        scheduler_parent_states={
            "canary_array": args.parent_canary_state,
            "failed_canary_audit": args.parent_canary_audit_state,
        },
        reaudit_dependency_mode=args.reaudit_dependency_mode,
    )
    print(json.dumps({"status": "written", "checksum": payload["content_checksum_sha256"]}))


if __name__ == "__main__":
    main()
