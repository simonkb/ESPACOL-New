#!/usr/bin/env python3
"""Assemble a fail-closed, privacy-safe ORIGIN acceptance manifest.

The assembler never copies source images, raw identifiers, filesystem
locations, or unreferenced checkpoint tensors.  It does copy the registered
anonymous split memberships and privacy-safe per-image outer posteriors, plus
sanitized audits and protocols, into a deterministic inspectable bundle.
Checkpoint hashes are accompanied by dataset/fold aliases and sizes; an
independent-reproduction claim additionally requires a stable external archive
URI and archive SHA-256.

Scientific failure is not artifact corruption: a complete package may record a
failed Gate A or Gate B, but its claims are then explicitly withheld.  Pass
``--require-gates-pass`` for a submission-readiness job that refuses to create
a completion marker unless both gates pass.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
from pathlib import Path
import math
import re
import shutil
import subprocess
import sys
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_SCHEMA = "origin-acceptance-reproducibility-package-v2"
COMPLETION_SCHEMA = "origin-acceptance-reproducibility-complete-v2"
MANIFEST_FILENAME = "ORIGIN_ACCEPTANCE_PACKAGE_MANIFEST.json"
COMPLETION_FILENAME = "ORIGIN_ACCEPTANCE_PACKAGE_COMPLETE.json"
BUNDLE_DIRECTORY = "bundle"
BUNDLE_INDEX_FILENAME = "BUNDLE_INDEX.json"

EXPECTED_SHORTCUT_VARIANTS = (
    "ledger_sequential_hazard",
    "pooled_conditional",
    "ordinal_additive_mil",
    "sparse_bagnet",
    "origin_ctmc",
)
EXPECTED_DATASET_IMAGES = {"aptos": 3662, "dr": 35126}
HEX64 = re.compile(r"^[0-9a-f]{64}$")
HEX40 = re.compile(r"^[0-9a-f]{40}$")

BASELINE_SCALAR_METRICS = (
    "acc", "qwk", "mae", "expected_grade_mae", "balanced_acc", "macro_f1",
    "ece", "nll", "rps", "multiclass_brier", "threshold_ece",
    "threshold_binary_brier", "classwise_ece",
)
BASELINE_THRESHOLD_ECE_METRICS = tuple(
    f"threshold_ece_boundary_{boundary}" for boundary in range(4)
)
BASELINE_THRESHOLD_BRIER_METRICS = tuple(
    f"threshold_binary_brier_boundary_{boundary}" for boundary in range(4)
)
BASELINE_CLASSWISE_ECE_METRICS = tuple(
    f"classwise_ece_class_{grade}" for grade in range(5)
)
BASELINE_RECALL_METRICS = tuple(f"per_grade_recall_{grade}" for grade in range(5))
BASELINE_SUMMARY_METRICS = (
    BASELINE_SCALAR_METRICS
    + BASELINE_THRESHOLD_ECE_METRICS
    + BASELINE_THRESHOLD_BRIER_METRICS
    + BASELINE_CLASSWISE_ECE_METRICS
    + BASELINE_RECALL_METRICS
)
BASELINE_BOOTSTRAP_METRICS = (
    "acc", "qwk", "mae", "expected_grade_mae", "balanced_acc", "macro_f1",
    "nll", "rps", "multiclass_brier", "threshold_ece",
    "threshold_binary_brier", "classwise_ece",
) + (
    BASELINE_THRESHOLD_ECE_METRICS
    + BASELINE_THRESHOLD_BRIER_METRICS
    + BASELINE_CLASSWISE_ECE_METRICS
    + BASELINE_RECALL_METRICS
)

BASELINE_DATASET_FOLDS = {"aptos": tuple(range(5)), "dr": tuple(range(10))}
BASELINE_TRAINING_SEEDS = (42, 31415, 27182)
BASELINE_POSTERIOR_CONTRACT = {
    "multiclass_brier": "mean_sum_k_(p_k-onehot_k)^2",
    "threshold_binary_brier": (
        "per_boundary_mean_(P(Y>k)-1[Y>k])^2_then_unweighted_boundary_mean"
    ),
    "threshold_binary_brier_identity": (
        "unweighted_boundary_mean_equals_reported_ranked_probability_score"
    ),
    "threshold_ece": (
        "15_equal_width_bins_per_boundary_then_unweighted_boundary_mean"
    ),
    "classwise_ece": (
        "15_equal_width_bins_one_vs_rest_per_class_then_unweighted_class_mean"
    ),
    "reliability_scope": "pooled_out_of_fold_predictions_per_training_seed",
}

BASELINE_PARENT_JOB_KEYS = (
    "aptos_fold0_canary_array",
    "canary_gate_audit",
    "full_training_array",
    "full_freeze_audit",
    "coordinated_outer_release_array",
    "release_aggregate",
)
BASELINE_CONTINUATION_JOB_KEYS = (
    "canary_reaudit",
    "full_training_array",
    "full_freeze_audit",
    "coordinated_outer_release_array",
    "release_aggregate",
)
BASELINE_PARENT_FAILURE_STATES = {
    "FAILED", "CANCELLED", "TIMEOUT", "OUT_OF_MEMORY", "NODE_FAIL",
    "PREEMPTED", "BOOT_FAIL", "DEADLINE",
}

# A baseline continuation may repair orchestration and packaging, but it may
# not change the executable scientific protocol.  These paths are deliberately
# enumerated instead of treating a commit message or a manifest boolean as
# evidence that a continuation was infrastructure-only.
BASELINE_CONTINUATION_INFRASTRUCTURE_PREFIXES = ("docs/", "paper/", "tests/")
BASELINE_CONTINUATION_INFRASTRUCTURE_FILES = {
    "tools/assemble_origin_acceptance_package.py",
    "scripts/continue_origin_acceptance_baselines.sh",
    "scripts/validate_origin_acceptance_continuation.py",
    "scripts/launch_origin_acceptance_package.sh",
    "scripts/submit_origin_acceptance_package.sh",
    "scripts/submit_origin_shortcut_comparator_preflight_v2.sh",
    "scripts/submit_origin_shortcut_comparator_aggregate_v2.sh",
}
BASELINE_CONTINUATION_WRAPPERS = {
    "scripts/submit_origin_acceptance_aptos_f0_canary.sh",
    "scripts/submit_origin_acceptance_canary_audit.sh",
    "scripts/submit_origin_acceptance_full_array.sh",
    "scripts/submit_origin_acceptance_full_audit.sh",
    "scripts/submit_origin_acceptance_outer_release.sh",
    "scripts/submit_origin_acceptance_release_aggregate.sh",
}
BASELINE_EXECUTABLE_PROTOCOL = "scripts/origin_acceptance_baseline_common.py"


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _finite_number(value: Any, *, label: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} is not numeric") from error
    if not math.isfinite(number):
        raise ValueError(f"{label} is not finite")
    return number


def _require_numeric_vector(
    value: Any, *, length: int, label: str
) -> list[float]:
    if not isinstance(value, list) or len(value) != length:
        raise ValueError(f"{label} must contain exactly {length} values")
    return [
        _finite_number(item, label=f"{label}[{index}]")
        for index, item in enumerate(value)
    ]


def _expected_baseline_task_keys() -> set[str]:
    return {
        f"{dataset}__fold{fold}__{variant}__seed{seed}"
        for dataset, folds in BASELINE_DATASET_FOLDS.items()
        for fold in folds
        for variant in EXPECTED_SHORTCUT_VARIANTS
        for seed in BASELINE_TRAINING_SEEDS
    }


def read_json(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(encoding="utf-8") as stream:
        payload = json.load(stream)
    if not isinstance(payload, dict):
        raise TypeError(f"expected a JSON object: {path}")
    return payload


def verify_checksummed_payload(
    path: str | Path, *, schema: str | Sequence[str]
) -> dict[str, Any]:
    path = Path(path)
    payload = read_json(path)
    schemas = {schema} if isinstance(schema, str) else set(schema)
    if payload.get("schema") not in schemas:
        raise ValueError(
            f"unexpected schema {payload.get('schema')!r} in {path.name}; "
            f"expected {sorted(schemas)}"
        )
    recorded = payload.get("content_checksum_sha256")
    if not isinstance(recorded, str) or not HEX64.fullmatch(recorded):
        raise ValueError(f"missing canonical content checksum: {path}")
    unsigned = dict(payload)
    unsigned.pop("content_checksum_sha256", None)
    if canonical_sha256(unsigned) != recorded:
        raise ValueError(f"canonical content checksum mismatch: {path}")
    return payload


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    temporary.replace(path)


def _git(repo_root: Path, *arguments: str) -> str:
    return subprocess.run(
        ("git", "-C", str(repo_root), *arguments), check=True,
        capture_output=True, text=True,
    ).stdout.strip()


def _verify_commit(repo_root: Path, commit: Any, *, field: str) -> str:
    value = str(commit)
    if not HEX40.fullmatch(value):
        raise ValueError(f"{field} is not a full Git commit")
    try:
        subprocess.run(
            ("git", "-C", str(repo_root), "cat-file", "-e", f"{value}^{{commit}}"),
            check=True, capture_output=True,
        )
    except subprocess.CalledProcessError as error:
        raise ValueError(f"{field} is not present in the repository: {value}") from error
    return value


def _verify_checksummed_mapping(
    payload: Any, *, schema: str, label: str,
) -> Mapping[str, Any]:
    if not isinstance(payload, Mapping):
        raise ValueError(f"{label} is not a JSON object")
    if payload.get("schema") != schema:
        raise ValueError(f"{label} schema changed")
    recorded = payload.get("content_checksum_sha256")
    if not isinstance(recorded, str) or not HEX64.fullmatch(recorded):
        raise ValueError(f"{label} lacks a canonical content checksum")
    unsigned = dict(payload)
    unsigned.pop("content_checksum_sha256", None)
    if canonical_sha256(unsigned) != recorded:
        raise ValueError(f"{label} canonical content checksum mismatch")
    return payload


def _require_scheduler_job_map(
    value: Any, *, expected_keys: Sequence[str], label: str,
) -> dict[str, str]:
    if not isinstance(value, Mapping) or set(value) != set(expected_keys):
        raise ValueError(f"{label} scheduler job map is incomplete")
    jobs = {key: str(value[key]) for key in expected_keys}
    if any(not identifier.isdigit() for identifier in jobs.values()):
        raise ValueError(f"{label} scheduler job identifier is malformed")
    if len(set(jobs.values())) != len(jobs):
        raise ValueError(f"{label} scheduler job identifiers are not unique")
    return jobs


def _verify_recorded_worktree(
    repo_root: Path, declaration: Any, *, commit: str, label: str,
) -> dict[str, Any]:
    worktree = Path(str(declaration)).expanduser().resolve()
    if not worktree.is_dir():
        raise ValueError(f"{label} recorded worktree is unavailable")
    try:
        head = _git(worktree, "rev-parse", "HEAD")
        top_level = Path(_git(worktree, "rev-parse", "--show-toplevel")).resolve()
        branch = _git(worktree, "branch", "--show-current")
        dirty = _git(worktree, "status", "--porcelain", "--untracked-files=no")
    except (OSError, subprocess.CalledProcessError) as error:
        raise ValueError(f"{label} recorded worktree is not a valid Git worktree") from error
    if top_level != worktree:
        raise ValueError(f"{label} declaration does not name the worktree root")
    if head != commit:
        raise ValueError(f"{label} recorded worktree commit mismatch")
    if branch:
        raise ValueError(f"{label} recorded worktree is not detached")
    if dirty:
        raise ValueError(f"{label} recorded worktree has tracked modifications")
    # Confirm the object is also present in the assembly repository.  A
    # worktree path alone is not accepted as a second source of truth.
    _verify_commit(repo_root, commit, field=f"{label} commit")
    return {
        "commit": commit,
        "recorded_worktree_verified": True,
        "recorded_worktree_location_sha256": hashlib.sha256(
            str(worktree).encode("utf-8")
        ).hexdigest(),
        "detached_head": True,
        "tracked_worktree_clean": True,
    }


def _git_blob_text(repo_root: Path, commit: str, relative_path: str) -> str:
    try:
        completed = subprocess.run(
            ("git", "-C", str(repo_root), "show", f"{commit}:{relative_path}"),
            check=True, capture_output=True,
        )
    except subprocess.CalledProcessError as error:
        raise ValueError(
            f"continued baseline provenance cannot read {relative_path} at {commit}"
        ) from error
    try:
        return completed.stdout.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError(
            f"continued baseline infrastructure file is not UTF-8: {relative_path}"
        ) from error


def _normalized_baseline_wrapper(source: str) -> str:
    """Erase only the audited import-path repair from a Slurm wrapper."""

    normalized: list[str] = []
    for line in source.splitlines():
        if line == 'REPO_ROOT="${ORIGIN_REPO_ROOT:?ORIGIN_REPO_ROOT is required}"':
            continue
        if line == 'export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"':
            continue
        if line == 'cd "${REPO_ROOT}"':
            line = 'cd "${ORIGIN_REPO_ROOT:?ORIGIN_REPO_ROOT is required}"'
        if line == "python -m scripts.audit_origin_acceptance_baselines \\":
            line = "python scripts/audit_origin_acceptance_baselines.py " + '\\'
        if line == "python -m scripts.aggregate_origin_acceptance_release \\":
            line = "python scripts/aggregate_origin_acceptance_release.py " + '\\'
        normalized.append(line)
    return "\n".join(normalized).rstrip() + "\n"


def _verify_infrastructure_only_continuation(
    repo_root: Path, *, parent_commit: str, continuation_commit: str,
) -> dict[str, Any]:
    if parent_commit == continuation_commit:
        raise ValueError("continued baseline provenance reuses the parent commit")
    ancestry = subprocess.run(
        (
            "git", "-C", str(repo_root), "merge-base", "--is-ancestor",
            parent_commit, continuation_commit,
        ),
        capture_output=True,
    )
    if ancestry.returncode != 0:
        raise ValueError("continued baseline commit is not descended from its parent")
    changed_text = _git(
        repo_root, "diff", "--no-renames", "--name-only",
        parent_commit, continuation_commit,
    )
    changed = tuple(line for line in changed_text.splitlines() if line)
    if not changed:
        raise ValueError("continued baseline commit contains no auditable repair")
    disallowed: list[str] = []
    for relative in changed:
        if relative in BASELINE_CONTINUATION_WRAPPERS:
            before = _normalized_baseline_wrapper(
                _git_blob_text(repo_root, parent_commit, relative)
            )
            after = _normalized_baseline_wrapper(
                _git_blob_text(repo_root, continuation_commit, relative)
            )
            if before != after:
                disallowed.append(relative)
            continue
        if relative in BASELINE_CONTINUATION_INFRASTRUCTURE_FILES:
            continue
        if relative.startswith(BASELINE_CONTINUATION_INFRASTRUCTURE_PREFIXES):
            continue
        disallowed.append(relative)
    if disallowed:
        raise ValueError(
            "continued baseline commit changes non-infrastructure files: "
            + ", ".join(sorted(disallowed))
        )

    parent_protocol = _git(
        repo_root, "rev-parse", f"{parent_commit}:{BASELINE_EXECUTABLE_PROTOCOL}"
    )
    continuation_protocol = _git(
        repo_root, "rev-parse", f"{continuation_commit}:{BASELINE_EXECUTABLE_PROTOCOL}"
    )
    if parent_protocol != continuation_protocol:
        raise ValueError("continued baseline executable scientific protocol changed")
    protocol_sha256 = hashlib.sha256(
        _git_blob_text(repo_root, parent_commit, BASELINE_EXECUTABLE_PROTOCOL).encode(
            "utf-8"
        )
    ).hexdigest()
    patch = subprocess.run(
        (
            "git", "-C", str(repo_root), "diff", "--binary", "--no-renames",
            parent_commit, continuation_commit,
        ),
        check=True, capture_output=True,
    ).stdout
    return {
        "verified": True,
        "parent_is_ancestor": True,
        "changed_files": list(changed),
        "changed_file_count": len(changed),
        "git_patch_sha256": hashlib.sha256(patch).hexdigest(),
        "executable_protocol_git_blob_oid": parent_protocol,
        "executable_protocol_sha256": protocol_sha256,
        "wrapper_changes_limited_to_import_path_repair": all(
            path not in BASELINE_CONTINUATION_WRAPPERS
            or _normalized_baseline_wrapper(
                _git_blob_text(repo_root, parent_commit, path)
            ) == _normalized_baseline_wrapper(
                _git_blob_text(repo_root, continuation_commit, path)
            )
            for path in changed
        ),
    }


def _canary_task_key(task: Mapping[str, Any]) -> str:
    try:
        return (
            f"{task['dataset']}__fold{int(task['fold'])}__"
            f"{task['baseline_variant']}__seed{int(task['training_seed'])}"
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("continued baseline canary task identity is malformed") from error


def _validate_reused_canary(
    experiment_root: Path, *, protocol_id: str, protocol_sha256: str,
    preflight_record: Mapping[str, Any],
) -> dict[str, Any]:
    gate = verify_checksummed_payload(
        experiment_root / "canary" / "CANARY_PASSED.json",
        schema="origin-acceptance-training-audit-v1",
    )
    if (
        gate.get("scope") != "canary"
        or gate.get("status") != "passed"
        or gate.get("outer_test_released") is not False
        or int(gate.get("worker_count", 0)) != len(EXPECTED_SHORTCUT_VARIANTS)
    ):
        raise ValueError("continued baseline canary gate is not an exact passed canary")
    if gate.get("protocol_id") != protocol_id:
        raise ValueError("continued baseline canary protocol identifier mismatch")
    if gate.get("protocol_sha256") != protocol_sha256:
        raise ValueError("continued baseline canary protocol mismatch")
    workers = gate.get("workers")
    if not isinstance(workers, list) or len(workers) != len(EXPECTED_SHORTCUT_VARIANTS):
        raise ValueError("continued baseline canary worker census changed")
    expected_keys = {
        f"aptos__fold0__{variant}__seed42"
        for variant in EXPECTED_SHORTCUT_VARIANTS
    }
    observed_keys: set[str] = set()
    manifest_hashes: dict[str, str] = {}
    checkpoint_hashes: dict[str, str] = {}
    for worker in workers:
        if not isinstance(worker, Mapping) or not isinstance(worker.get("task"), Mapping):
            raise ValueError("continued baseline canary worker record is malformed")
        task = worker["task"]
        key = _canary_task_key(task)
        variant = str(task.get("baseline_variant"))
        if key not in expected_keys or key in observed_keys:
            raise ValueError("continued baseline canary task census changed")
        manifest_digest = str(worker.get("manifest_sha256"))
        checkpoint_digest = str(worker.get("best_learned_sha256"))
        if not HEX64.fullmatch(manifest_digest) or not HEX64.fullmatch(
            checkpoint_digest
        ):
            raise ValueError("continued baseline canary artifact digest is malformed")
        observed_keys.add(key)
        manifest_hashes[variant] = manifest_digest
        checkpoint_hashes[variant] = checkpoint_digest
    if observed_keys != expected_keys:
        raise ValueError("continued baseline canary task coverage is incomplete")

    expected_task_keys = preflight_record.get("task_keys")
    if expected_task_keys != sorted(expected_keys):
        raise ValueError("continued baseline preflight canary task binding changed")
    if int(preflight_record.get("task_count", 0)) != len(expected_keys):
        raise ValueError("continued baseline preflight canary count changed")
    if preflight_record.get("training_audit_content_checksum_sha256") != gate.get(
        "content_checksum_sha256"
    ):
        raise ValueError("continued baseline canary re-audit checksum mismatch")
    if preflight_record.get("worker_manifest_sha256") != manifest_hashes:
        raise ValueError("continued baseline canary manifest hashes changed")
    if preflight_record.get("best_learned_sha256") != checkpoint_hashes:
        raise ValueError("continued baseline canary checkpoint hashes changed")
    if not isinstance(preflight_record.get("valid_gate_already_present"), bool):
        raise ValueError("continued baseline preflight gate-state record is malformed")
    return {
        "verified": True,
        "worker_count": len(expected_keys),
        "task_keys": sorted(expected_keys),
        "training_audit_content_checksum_sha256": gate[
            "content_checksum_sha256"
        ],
    }


def _validate_baseline_execution_provenance(
    *, experiment_root: Path, submission: Mapping[str, Any],
    aggregate_protocol_sha256: str, repo_root: Path,
) -> dict[str, Any]:
    parent_commit = _verify_commit(
        repo_root, submission.get("launch_commit"),
        field="matched-baseline parent launch commit",
    )
    parent_protocol_sha256 = hashlib.sha256(
        _git_blob_text(repo_root, parent_commit, BASELINE_EXECUTABLE_PROTOCOL).encode(
            "utf-8"
        )
    ).hexdigest()
    if file_sha256(repo_root / BASELINE_EXECUTABLE_PROTOCOL) != parent_protocol_sha256:
        raise ValueError(
            "assembly repository executable protocol differs from the baseline run"
        )
    restart_path = experiment_root / "RESTART_SUBMISSION.json"
    if not restart_path.exists():
        return {
            "mode": "single_submission",
            "source_commit": parent_commit,
            "parent_canary_source_commit": parent_commit,
            "continuation_source_commit": None,
            "restart_submission_present": False,
            "executable_protocol_sha256": parent_protocol_sha256,
        }

    restart = verify_checksummed_payload(
        restart_path, schema="origin-acceptance-restart-submission-v1"
    )
    preflight = _verify_checksummed_mapping(
        restart.get("preflight"),
        schema="origin-acceptance-continuation-preflight-v1",
        label="matched-baseline continuation preflight",
    )
    parent_record = preflight.get("parent")
    canary_record = preflight.get("reused_canary")
    if not isinstance(parent_record, Mapping) or not isinstance(canary_record, Mapping):
        raise ValueError("matched-baseline continuation preflight is incomplete")

    protocol_id = submission.get("protocol_id")
    if not isinstance(protocol_id, str) or not protocol_id:
        raise ValueError("matched-baseline parent protocol identifier is missing")
    for label, record in (("restart", restart), ("preflight", preflight)):
        if record.get("protocol_id") != protocol_id:
            raise ValueError(f"matched-baseline {label} protocol identifier mismatch")
        if record.get("protocol_sha256") != aggregate_protocol_sha256:
            raise ValueError(f"matched-baseline {label} protocol digest mismatch")
    if submission.get("protocol_sha256") != aggregate_protocol_sha256:
        raise ValueError("matched-baseline parent protocol digest mismatch")
    if restart.get("scientific_protocol_changed") is not False:
        raise ValueError("matched-baseline continuation claims a scientific change")
    if restart.get("canary_training_reused") is not True:
        raise ValueError("matched-baseline continuation does not bind canary reuse")
    if not isinstance(restart.get("continuation_reason"), str) or not str(
        restart.get("continuation_reason")
    ).strip():
        raise ValueError("matched-baseline continuation reason is missing")
    if not isinstance(restart.get("conda_environment"), str) or not str(
        restart.get("conda_environment")
    ).strip():
        raise ValueError("matched-baseline continuation environment is missing")
    if Path(str(restart.get("experiment_root"))).resolve() != experiment_root.resolve():
        raise ValueError("matched-baseline restart experiment-root binding changed")
    if Path(str(preflight.get("experiment_root"))).resolve() != experiment_root.resolve():
        raise ValueError("matched-baseline preflight experiment-root binding changed")
    if preflight.get("downstream_state") != "absent":
        raise ValueError("matched-baseline continuation began after downstream work")

    submission_path = experiment_root / "SUBMISSION.json"
    if Path(str(parent_record.get("submission_path"))).resolve() != submission_path.resolve():
        raise ValueError("matched-baseline parent submission path binding changed")
    if parent_record.get("submission_file_sha256") != file_sha256(submission_path):
        raise ValueError("matched-baseline parent submission file digest mismatch")
    if parent_record.get("submission_content_checksum_sha256") != submission.get(
        "content_checksum_sha256"
    ):
        raise ValueError("matched-baseline parent submission checksum binding changed")
    if parent_record.get("parent_launch_commit") != parent_commit:
        raise ValueError("matched-baseline preflight parent commit mismatch")
    if parent_record.get("parent_immutable_worktree") != submission.get(
        "immutable_worktree"
    ):
        raise ValueError("matched-baseline parent worktree binding changed")

    parent_jobs = _require_scheduler_job_map(
        submission.get("jobs"), expected_keys=BASELINE_PARENT_JOB_KEYS,
        label="matched-baseline parent",
    )
    recorded_parent_jobs = _require_scheduler_job_map(
        parent_record.get("parent_jobs"), expected_keys=BASELINE_PARENT_JOB_KEYS,
        label="matched-baseline preflight parent",
    )
    if recorded_parent_jobs != parent_jobs:
        raise ValueError("matched-baseline preflight parent job chain mismatch")
    continuation_jobs = _require_scheduler_job_map(
        restart.get("jobs"), expected_keys=BASELINE_CONTINUATION_JOB_KEYS,
        label="matched-baseline continuation",
    )
    if set(continuation_jobs.values()) & set(parent_jobs.values()):
        raise ValueError("matched-baseline continuation reuses a parent scheduler job")

    states = restart.get("scheduler_parent_states")
    if not isinstance(states, Mapping) or set(states) != {
        "canary_array", "failed_canary_audit"
    }:
        raise ValueError("matched-baseline continuation scheduler-state record changed")
    if states.get("canary_array") != "COMPLETED":
        raise ValueError("matched-baseline parent canary did not complete")
    if states.get("failed_canary_audit") not in BASELINE_PARENT_FAILURE_STATES:
        raise ValueError("matched-baseline parent audit lacks a terminal failure state")
    dependency_mode = restart.get("canary_reaudit_dependency_mode")
    if dependency_mode not in {
        "afterok_live_reaudit", "verified_completed_no_dependency"
    }:
        raise ValueError("matched-baseline continuation dependency mode changed")

    continuation_commit = _verify_commit(
        repo_root, restart.get("launch_commit"),
        field="matched-baseline continuation launch commit",
    )
    parent_worktree = _verify_recorded_worktree(
        repo_root, submission.get("immutable_worktree"), commit=parent_commit,
        label="matched-baseline parent",
    )
    continuation_worktree = _verify_recorded_worktree(
        repo_root, restart.get("immutable_worktree"), commit=continuation_commit,
        label="matched-baseline continuation",
    )
    canary = _validate_reused_canary(
        experiment_root, protocol_id=protocol_id,
        protocol_sha256=aggregate_protocol_sha256,
        preflight_record=canary_record,
    )
    infrastructure = _verify_infrastructure_only_continuation(
        repo_root, parent_commit=parent_commit,
        continuation_commit=continuation_commit,
    )
    if infrastructure["executable_protocol_sha256"] != parent_protocol_sha256:
        raise ValueError(
            "continued baseline executable protocol digest changed unexpectedly"
        )
    return {
        "mode": "infrastructure_only_continuation",
        "source_commit": continuation_commit,
        "parent_canary_source_commit": parent_commit,
        "continuation_source_commit": continuation_commit,
        "restart_submission_present": True,
        "parent_submission_file_sha256": file_sha256(submission_path),
        "restart_submission_file_sha256": file_sha256(restart_path),
        "restart_submission_content_checksum_sha256": restart[
            "content_checksum_sha256"
        ],
        "protocol_unchanged": True,
        "parent_canary_worktree": parent_worktree,
        "continuation_worktree": continuation_worktree,
        "parent_job_chain": parent_jobs,
        "continuation_job_chain": continuation_jobs,
        "scheduler_parent_states": dict(states),
        "canary_reaudit_dependency_mode": dependency_mode,
        "reused_canary": canary,
        "infrastructure_only_commit_delta": infrastructure,
    }


def runtime_provenance(repo_root: Path, expected_commit: str | None) -> dict[str, Any]:
    commit = _git(repo_root, "rev-parse", "HEAD")
    if expected_commit is not None and commit != expected_commit:
        raise RuntimeError(
            f"assembly commit {commit} differs from expected {expected_commit}"
        )
    dirty = _git(repo_root, "status", "--porcelain", "--untracked-files=no")
    if dirty:
        raise RuntimeError("acceptance assembly requires a clean tracked worktree")
    _verify_commit(repo_root, commit, field="assembly commit")
    return {
        "git_commit": commit,
        "tracked_worktree_clean": True,
        "assembler_sha256": file_sha256(Path(__file__).resolve()),
    }


def _resolve_declared_file(
    declaration: Any, *, relative_to: Path | None = None,
    allowed_roots: Iterable[Path] | None = None,
) -> Path:
    path = Path(str(declaration))
    if not path.is_absolute():
        if relative_to is None:
            raise ValueError(f"relative artifact has no declared base: {path}")
        path = relative_to / path
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    if allowed_roots is not None:
        roots = tuple(root.resolve() for root in allowed_roots)
        if not any(path.is_relative_to(root) for root in roots):
            raise ValueError(f"artifact escapes its declared output root: {path.name}")
    return path


def _verify_declared_hash(path: Path, expected: Any, *, label: str) -> str:
    expected = str(expected)
    if not HEX64.fullmatch(expected):
        raise ValueError(f"{label} has no valid SHA-256 digest")
    observed = file_sha256(path)
    if observed != expected:
        raise ValueError(f"{label} SHA-256 mismatch")
    return observed


def _verify_json_artifact(
    manifest_path: Path, record: Mapping[str, Any], *, allowed_root: Path,
) -> dict[str, Any]:
    artifact_path = _resolve_declared_file(
        record.get("path"), relative_to=manifest_path.parent,
        allowed_roots=(allowed_root,),
    )
    _verify_declared_hash(artifact_path, record.get("sha256"), label=artifact_path.name)
    payload = read_json(artifact_path)
    if "content_checksum_sha256" in record:
        recorded = str(record["content_checksum_sha256"])
        if payload.get("content_checksum_sha256") != recorded:
            raise ValueError(f"nested content checksum mismatch: {artifact_path.name}")
        unsigned = dict(payload)
        unsigned.pop("content_checksum_sha256", None)
        if canonical_sha256(unsigned) != recorded:
            raise ValueError(f"nested canonical checksum mismatch: {artifact_path.name}")
    return payload


def validate_numerical(path: Path) -> dict[str, Any]:
    payload = verify_checksummed_payload(
        path, schema="origin-decoder-independent-numerical-audit-v1"
    )
    summary = payload.get("summary")
    if not isinstance(summary, Mapping) or summary.get("passed") is not True:
        raise ValueError("independent decoder numerical audit did not pass")
    if payload.get("independent_reference") != "mpmath.expm":
        raise ValueError("decoder audit is not independent of the deployed exponential")
    if payload.get("rate_domain") != [0.0, 64.0]:
        raise ValueError("decoder audit did not cover the registered rate domain [0,64]")
    if int(payload.get("n_forward_cases", 0)) < 128:
        raise ValueError("decoder numerical audit has fewer than 128 random cases")
    if int(payload.get("n_gradient_cases", 0)) < 12:
        raise ValueError("decoder numerical audit has fewer than 12 gradient cases")
    return {
        "schema": payload["schema"],
        "file_sha256": file_sha256(path),
        "content_checksum_sha256": payload["content_checksum_sha256"],
        "validation_implementation_sha256": file_sha256(
            REPO_ROOT / "tools" / "audit_origin_decoder_numeric.py"
        ),
        "passed": True,
        "n_forward_cases": int(payload["n_forward_cases"]),
        "n_gradient_cases": int(payload["n_gradient_cases"]),
        "rate_cap": 64.0,
    }


def validate_decoder_contract(path: Path) -> dict[str, Any]:
    payload = read_json(path)
    if payload.get("schema") != "origin-decoder-contract-audit-v1":
        raise ValueError("unexpected decoder-contract schema")
    if payload.get("passed") is not True or int(payload.get("trials", 0)) < 64:
        raise ValueError("decoder-contract audit did not pass its registered trials")
    if float(payload.get("maximum_ctmc_semigroup_error", 1.0)) > 1e-12:
        raise ValueError("CTMC semigroup contract exceeds tolerance")
    if float(payload.get("minimum_sequential_semigroup_failure", 0.0)) < 1e-6:
        raise ValueError("sequential-hazard negative control did not separate")
    return {
        "schema": payload["schema"],
        "file_sha256": file_sha256(path),
        "validation_implementation_sha256": file_sha256(
            REPO_ROOT / "tools" / "audit_origin_decoder_contracts.py"
        ),
        "passed": True,
        "trials": int(payload["trials"]),
        "contract": "ledger_replay_shared_ctmc_semigroup_decoder_specific",
    }


def _verify_anonymous_membership(path: Path, expected_rows: int) -> None:
    with gzip.open(path, "rt", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        expected_fields = [
            "image_id_sha256", "patient_cluster_sha256", "label", "split"
        ]
        if reader.fieldnames != expected_fields:
            raise ValueError(f"anonymous membership columns changed: {path.name}")
        rows = 0
        for row in reader:
            rows += 1
            if not HEX64.fullmatch(str(row["image_id_sha256"])):
                raise ValueError("membership contains a raw or invalid image identifier")
            if not HEX64.fullmatch(str(row["patient_cluster_sha256"])):
                raise ValueError("membership contains a raw or invalid cluster identifier")
            if row["split"] not in {"train", "validation", "outer_test"}:
                raise ValueError("membership contains an unknown split")
            if int(row["label"]) not in range(5):
                raise ValueError("membership contains an invalid label")
    if rows != expected_rows:
        raise ValueError(f"membership row census changed: {path.name}")


def validate_artifact_manifest(path: Path, repo_root: Path) -> dict[str, Any]:
    payload = verify_checksummed_payload(
        path, schema="origin-acceptance-artifact-manifest-v1"
    )
    if payload.get("raw_patient_data_redistributed") is not False:
        raise ValueError("artifact manifest permits raw patient-data redistribution")
    source = payload.get("source_provenance")
    if not isinstance(source, Mapping) or source.get("tracked_tree_dirty") is not False:
        raise ValueError("artifact audit was not run from a clean source tree")
    audit_commit = _verify_commit(
        repo_root, source.get("audit_source_commit"), field="artifact audit source commit"
    )
    submission = payload.get("full_cv_submission")
    if not isinstance(submission, Mapping):
        raise ValueError("full-CV launch provenance is missing")
    training_commit = _verify_commit(
        repo_root, submission.get("training_launch_commit"),
        field="full-CV training launch commit",
    )
    cv_root = Path(str(payload.get("cv_root", ""))).resolve()
    submission_path = _resolve_declared_file(
        submission.get("relative_path"), relative_to=cv_root, allowed_roots=(cv_root,)
    )
    _verify_declared_hash(
        submission_path, submission.get("sha256"), label="full-CV submission"
    )

    checkpoints = payload.get("checkpoints")
    memberships = payload.get("split_memberships")
    if not isinstance(checkpoints, list) or len(checkpoints) != 15:
        raise ValueError("artifact manifest must contain all 15 selected checkpoints")
    if not isinstance(memberships, list) or len(memberships) != 15:
        raise ValueError("artifact manifest must contain all 15 split memberships")
    expected_folds = {"aptos": set(range(5)), "dr": set(range(10))}
    seen_checkpoints = {name: set() for name in expected_folds}
    seen_memberships = {name: set() for name in expected_folds}
    for record in checkpoints:
        dataset, fold = str(record.get("dataset")), int(record.get("fold", -1))
        if dataset not in expected_folds or fold not in expected_folds[dataset]:
            raise ValueError("checkpoint identifies an unexpected dataset/fold")
        checkpoint = _resolve_declared_file(
            record.get("relative_path"), relative_to=cv_root, allowed_roots=(cv_root,)
        )
        _verify_declared_hash(checkpoint, record.get("sha256"), label="selected checkpoint")
        if checkpoint.stat().st_size != int(record.get("bytes", -1)):
            raise ValueError("selected checkpoint size mismatch")
        if record.get("checkpoint_schema") != "origin-checkpoint-v3":
            raise ValueError("selected checkpoint schema mismatch")
        for signature in (
            "split_signature", "implementation_signature", "architecture_signature",
            "config_signature",
        ):
            if not HEX64.fullmatch(str(record.get(signature, ""))):
                raise ValueError(f"selected checkpoint {signature} is invalid")
        seen_checkpoints[dataset].add(fold)
    for record in memberships:
        dataset, fold = str(record.get("dataset")), int(record.get("fold", -1))
        if dataset not in expected_folds or fold not in expected_folds[dataset]:
            raise ValueError("membership identifies an unexpected dataset/fold")
        membership = _resolve_declared_file(
            record.get("relative_path"), relative_to=path.parent,
            allowed_roots=(path.parent,),
        )
        _verify_declared_hash(membership, record.get("sha256"), label="split membership")
        _verify_anonymous_membership(membership, int(record.get("rows", -1)))
        seen_memberships[dataset].add(fold)
    if seen_checkpoints != expected_folds or seen_memberships != expected_folds:
        raise ValueError("artifact manifest fold coverage is incomplete")
    return {
        "schema": payload["schema"],
        "file_sha256": file_sha256(path),
        "content_checksum_sha256": payload["content_checksum_sha256"],
        "audit_source_commit": audit_commit,
        "training_launch_commit": training_commit,
        "checkpoints": 15,
        "anonymous_split_memberships": 15,
        "raw_patient_data_redistributed": False,
    }


def validate_idrid_manifest(path: Path, repo_root: Path) -> dict[str, Any]:
    payload = verify_checksummed_payload(
        path, schema="origin-idrid-semantic-statistics-manifest-v1"
    )
    implementation = repo_root / "tools" / "analyze_origin_idrid_semantics.py"
    if file_sha256(implementation) != payload.get("analysis_implementation_sha256"):
        raise ValueError("IDRiD analysis implementation differs from the sealed run")
    protocol_path = repo_root / "scripts" / "protocols" / "origin_idrid_semantic_statistics_protocol.json"
    protocol = verify_checksummed_payload(
        protocol_path, schema="origin-idrid-semantic-statistics-protocol-v1"
    )
    if protocol["content_checksum_sha256"] != payload.get("protocol_checksum_sha256"):
        raise ValueError("IDRiD statistics protocol checksum mismatch")
    census = payload.get("input_census")
    if not isinstance(census, Mapping) or len(census.get("images", [])) != 81:
        raise ValueError("IDRiD audit does not cover exactly 81 image clusters")

    # The raw summary/records are validated in place but never copied.
    semantic_root = path.parent.parent.resolve()
    for key, hash_key in (
        ("input_summary_path", "input_summary_sha256"),
        ("input_records_path", "input_records_sha256"),
    ):
        source = _resolve_declared_file(
            payload.get(key), allowed_roots=(semantic_root,)
        )
        _verify_declared_hash(source, payload.get(hash_key), label=key)
    artifacts = payload.get("artifacts")
    required = {"alignment_statistics", "deletion_statistics", "image_level_units"}
    if not isinstance(artifacts, Mapping) or set(artifacts) != required:
        raise ValueError("IDRiD statistics artifact set is incomplete")
    for name, record in artifacts.items():
        artifact = _resolve_declared_file(
            record.get("path"), relative_to=path.parent, allowed_roots=(path.parent,)
        )
        _verify_declared_hash(artifact, record.get("sha256"), label=name)
        if name != "image_level_units":
            nested = read_json(artifact)
            recorded = record.get("content_checksum_sha256")
            if nested.get("content_checksum_sha256") != recorded:
                raise ValueError(f"IDRiD nested checksum mismatch: {name}")
            unsigned = dict(nested)
            unsigned.pop("content_checksum_sha256", None)
            if canonical_sha256(unsigned) != recorded:
                raise ValueError(f"IDRiD nested canonical checksum mismatch: {name}")
        else:
            with artifact.open(encoding="utf-8") as stream:
                rows = sum(1 for line in stream if line.strip())
            if rows != int(record.get("rows", -1)):
                raise ValueError("IDRiD image-unit row census mismatch")
    return {
        "schema": payload["schema"],
        "file_sha256": file_sha256(path),
        "content_checksum_sha256": payload["content_checksum_sha256"],
        "protocol_checksum_sha256": protocol["content_checksum_sha256"],
        "image_clusters": 81,
        "checkpoint_evaluations_per_image": 10,
        "interpretation_scope": "stored_ledger_not_causal_pixel_intervention",
    }


def validate_oof_manifest(path: Path, repo_root: Path) -> tuple[dict[str, Any], str]:
    payload = verify_checksummed_payload(path, schema="origin-oof-statistics-manifest-v2")
    runtime = payload.get("statistics_runtime_source")
    if not isinstance(runtime, Mapping):
        raise ValueError("OOF statistics runtime provenance is missing")
    if runtime.get("git_commit_available") is not True or runtime.get(
        "tracked_worktree_clean"
    ) is not True:
        raise ValueError("OOF statistics were not produced from a clean Git commit")
    source_commit = _verify_commit(
        repo_root, runtime.get("git_commit"), field="OOF statistics source commit"
    )
    implementation = repo_root / "scripts" / "analyze_origin_oof_statistics.py"
    implementation_hash = file_sha256(implementation)
    if implementation_hash != payload.get("analysis_implementation_sha256"):
        raise ValueError("OOF analysis implementation differs from the sealed run")
    if implementation_hash != runtime.get("analysis_implementation_sha256"):
        raise ValueError("OOF runtime implementation hash is internally inconsistent")
    protocol_path = repo_root / "scripts" / "protocols" / "origin_oof_statistics_protocol.json"
    protocol = verify_checksummed_payload(
        protocol_path, schema="origin-oof-statistics-protocol-v2"
    )
    if protocol["content_checksum_sha256"] != payload.get(
        "statistics_protocol_checksum_sha256"
    ):
        raise ValueError("OOF Gate-A protocol checksum mismatch")

    inputs = payload.get("inputs")
    if not isinstance(inputs, list) or {row.get("dataset") for row in inputs} != {
        "aptos", "dr"
    }:
        raise ValueError("OOF statistics require complete APTOS and EyePACS inputs")
    for record in inputs:
        dataset = str(record["dataset"])
        source = _resolve_declared_file(record.get("path"))
        _verify_declared_hash(source, record.get("sha256"), label=f"{dataset} OOF aggregate")
        aggregate = verify_checksummed_payload(
            source, schema="origin-oof-intervention-aggregate-v1"
        )
        if aggregate["content_checksum_sha256"] != record.get(
            "content_checksum_sha256"
        ):
            raise ValueError(f"{dataset} OOF aggregate content checksum mismatch")
        if int(record.get("n_images", -1)) != EXPECTED_DATASET_IMAGES[dataset]:
            raise ValueError(f"{dataset} OOF image census changed")
        _verify_commit(
            repo_root, record.get("model_frozen_base_commit"),
            field=f"{dataset} frozen model commit",
        )

    artifacts = payload.get("artifacts")
    required = {
        "proper_score_calibration", "grading_bootstrap", "intervention_bootstrap",
        "rf_footprint_diagnostics", "gate_a",
    }
    if not isinstance(artifacts, Mapping) or set(artifacts) != required:
        raise ValueError("OOF statistics artifact set is incomplete")
    gate_payload: dict[str, Any] | None = None
    for name, record in artifacts.items():
        nested = _verify_json_artifact(path, record, allowed_root=path.parent)
        if name == "gate_a":
            gate_payload = nested
    if gate_payload is None or gate_payload.get("schema") != "origin-oof-gate-a-adjudication-v1":
        raise ValueError("OOF Gate-A adjudication is missing")
    gate_status = str(gate_payload.get("status"))
    if gate_status not in {"pass", "fail", "insufficient_data"}:
        raise ValueError("OOF Gate-A has an invalid status")
    authorized = gate_payload.get("claim_authorized") is True
    if authorized != (gate_status == "pass"):
        raise ValueError("OOF Gate-A authorization/status mismatch")
    if artifacts["gate_a"].get("gate_status") != gate_status:
        raise ValueError("OOF Gate-A manifest/artifact status mismatch")
    clauses = gate_payload.get("clauses")
    if not isinstance(clauses, Mapping) or set(clauses) != {
        "auc_advantage", "positive_grade_strata", "concentration", "retention_preservation"
    }:
        raise ValueError("OOF Gate-A clause census changed")
    if gate_status == "pass" and (
        gate_payload.get("input_completeness_issues")
        or any(row.get("status") != "pass" for row in clauses.values())
    ):
        raise ValueError("OOF Gate-A pass is not supported by every registered clause")
    return ({
        "schema": payload["schema"],
        "file_sha256": file_sha256(path),
        "content_checksum_sha256": payload["content_checksum_sha256"],
        "source_commit": source_commit,
        "protocol_checksum_sha256": protocol["content_checksum_sha256"],
        "datasets": ["aptos", "dr"],
        "gate_status": gate_status,
        "claim_authorized": authorized,
    }, gate_status)


def _verify_shortcut_audit(path: Path) -> tuple[str, str, str, int]:
    payload = verify_checksummed_payload(
        path, schema="origin-ordinal-shortcut-audit-v1"
    )
    if payload.get("complete_outer_fold") is not True:
        raise ValueError("shortcut aggregate contains a partial-fold audit")
    prediction = _resolve_declared_file(payload.get("prediction_artifact"))
    checkpoint = _resolve_declared_file(payload.get("checkpoint"))
    _verify_declared_hash(
        prediction, payload.get("prediction_artifact_sha256"),
        label="shortcut prediction artifact",
    )
    _verify_declared_hash(
        checkpoint, payload.get("checkpoint_sha256"), label="shortcut checkpoint"
    )
    return (
        str(payload.get("model_variant", "origin_ctmc")),
        str(payload.get("shortcut_arm")),
        str(payload.get("shortcut_family")),
        int(payload.get("training_seed", -1)),
    )


def validate_shortcut_aggregate(
    path: Path, repo_root: Path
) -> tuple[dict[str, Any], bool]:
    payload = verify_checksummed_payload(
        path, schema="origin-ordinal-shortcut-pilot-aggregate-v2"
    )
    if payload.get("expected_training_seeds") != [1701, 2603, 3907]:
        raise ValueError("shortcut training-seed set changed")
    model_reports = payload.get("model_variants")
    if not isinstance(model_reports, Mapping) or set(model_reports) != set(
        EXPECTED_SHORTCUT_VARIANTS
    ):
        raise ValueError("shortcut aggregate lacks the five matched model variants")
    for variant, record in model_reports.items():
        if record.get("official_author_implementation") is not False:
            raise ValueError(f"shortcut comparator {variant} is misattributed")
        families = record.get("families")
        if not isinstance(families, Mapping) or set(families) != {
            "localized", "border", "diffuse"
        }:
            raise ValueError(f"shortcut family coverage is incomplete for {variant}")

    audit_files = payload.get("audit_files")
    if not isinstance(audit_files, list) or len(audit_files) != 135:
        raise ValueError("shortcut aggregate must bind all 135 matched audit files")
    cells = [_verify_shortcut_audit(Path(item)) for item in audit_files]
    if len(set(cells)) != 135:
        raise ValueError("shortcut aggregate contains duplicate model/arm/family/seed audits")

    # Comparator-v2 aggregate, protocol, and immutable launch are siblings.
    submission_path = path.parent / "SUBMISSION.json"
    protocol_path = path.parent / "PROTOCOL_V2.json"
    submission = verify_checksummed_payload(
        submission_path, schema="origin-ordinal-shortcut-comparator-submission-v2"
    )
    protocol = verify_checksummed_payload(
        protocol_path, schema="origin-ordinal-shortcut-comparator-protocol-v2"
    )
    launch_commit = _verify_commit(
        repo_root, submission.get("launch_commit"), field="shortcut launch commit"
    )
    if protocol.get("launch_commit") != launch_commit:
        raise ValueError("shortcut protocol/submission launch commits differ")
    if protocol.get("additional_training_workers") != 84:
        raise ValueError("shortcut comparator protocol worker census changed")
    origin_families = model_reports["origin_ctmc"]["families"]
    origin_gate_results: list[bool] = []
    focal_checks = {
        "cue_only_positive_control", "cue_learned_all_seeds",
        "localization_seed_requirement", "deletion_all_seeds",
        "internal_pixel_correlation_all_seeds", "boundary_response_selectivity",
        "clean_negative_control",
    }
    diffuse_checks = {
        "cue_only_positive_control", "cue_learned_all_seeds",
        "boundary_response_selectivity", "diffuse_nonfocal_support",
    }
    for family in ("localized", "border", "diffuse"):
        gate = origin_families[family].get("gate_b")
        if not isinstance(gate, Mapping):
            raise ValueError(f"shortcut Gate-B result is missing for {family}")
        checks = gate.get("checks")
        expected_checks = diffuse_checks if family == "diffuse" else focal_checks
        if not isinstance(checks, Mapping) or set(checks) != expected_checks:
            raise ValueError(f"shortcut Gate-B clauses changed for {family}")
        gate_value = gate.get("passed") is True
        if gate_value != all(value is True for value in checks.values()):
            raise ValueError(f"shortcut Gate-B clause/status mismatch for {family}")
        if family == "diffuse":
            if gate.get("gate_kind") != "diffuse_nonfocal_negative_locality":
                raise ValueError("diffuse Gate-B is not the registered non-focal gate")
            if gate.get("localization_gate_applicable") is not False:
                raise ValueError("diffuse Gate-B is incorrectly treated as focal")
            if gate.get("must_not_be_reported_as_focal_lesion") is not True:
                raise ValueError("diffuse Gate-B lacks its non-focal claim restriction")
        origin_gate_results.append(gate_value)

    gate_passed = payload.get("passed") is True
    gates = payload.get("origin_family_gates_passed")
    if gates != origin_gate_results or len(origin_gate_results) != 3 or gate_passed != all(
        origin_gate_results
    ):
        raise ValueError("shortcut Gate-B aggregate is internally inconsistent")
    return ({
        "schema": payload["schema"],
        "file_sha256": file_sha256(path),
        "content_checksum_sha256": payload["content_checksum_sha256"],
        "source_commit": launch_commit,
        "training_seeds": [1701, 2603, 3907],
        "model_variants": list(EXPECTED_SHORTCUT_VARIANTS),
        "audit_files_verified": 135,
        "registered_gate_families": ["localized", "border", "diffuse"],
        "boundary_response_selectivity_verified": True,
        "diffuse_nonfocal_clause_verified": True,
        "gate_status": "pass" if gate_passed else "fail",
        "claim_authorized": gate_passed,
    }, gate_passed)


def _validate_reliability_bins(
    rows: Any, *, expected_n: int, label: str
) -> None:
    if not isinstance(rows, list) or len(rows) != 15:
        raise ValueError(f"{label} must contain 15 reliability bins")
    total = 0
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping) or int(row.get("bin", -1)) != index:
            raise ValueError(f"{label} has an invalid bin index")
        count = int(row.get("count", -1))
        if count < 0:
            raise ValueError(f"{label} contains a negative bin count")
        total += count
        _finite_number(row.get("lower"), label=f"{label}.lower")
        _finite_number(row.get("upper"), label=f"{label}.upper")
        for key in ("mean_predicted", "empirical_frequency", "absolute_gap"):
            if count == 0:
                if row.get(key) is not None:
                    raise ValueError(f"{label} empty bin has a non-null {key}")
            else:
                _finite_number(row.get(key), label=f"{label}.{key}")
    if total != expected_n:
        raise ValueError(f"{label} bin census {total} differs from n={expected_n}")


def _validate_threshold_reliability(
    value: Any, *, label: str, expected_n: int | None = None
) -> None:
    if not isinstance(value, Mapping) or int(value.get("bin_count", 0)) != 15:
        raise ValueError(f"{label} lacks the registered 15-bin contract")
    n = int(value.get("n", expected_n if expected_n is not None else -1))
    if n < 1:
        raise ValueError(f"{label} has no observations")
    _finite_number(value.get("threshold_ece"), label=f"{label}.threshold_ece")
    _finite_number(
        value.get("threshold_binary_brier"),
        label=f"{label}.threshold_binary_brier",
    )
    _require_numeric_vector(
        value.get("threshold_ece_by_boundary"), length=4,
        label=f"{label}.threshold_ece_by_boundary",
    )
    _require_numeric_vector(
        value.get("threshold_binary_brier_by_boundary"), length=4,
        label=f"{label}.threshold_binary_brier_by_boundary",
    )
    boundaries = value.get("boundaries")
    if not isinstance(boundaries, list) or len(boundaries) != 4:
        raise ValueError(f"{label} must contain four ordinal boundaries")
    for boundary, row in enumerate(boundaries):
        if not isinstance(row, Mapping) or int(row.get("boundary", -1)) != boundary:
            raise ValueError(f"{label} boundary ordering changed")
        _finite_number(row.get("ece"), label=f"{label}.boundary[{boundary}].ece")
        _finite_number(
            row.get("binary_brier"),
            label=f"{label}.boundary[{boundary}].binary_brier",
        )
        _validate_reliability_bins(
            row.get("bins"), expected_n=n,
            label=f"{label}.boundary[{boundary}].bins",
        )


def _validate_classwise_reliability(
    value: Any, *, label: str, expected_n: int | None = None
) -> None:
    if not isinstance(value, Mapping) or int(value.get("bin_count", 0)) != 15:
        raise ValueError(f"{label} lacks the registered 15-bin contract")
    n = int(value.get("n", expected_n if expected_n is not None else -1))
    if n < 1:
        raise ValueError(f"{label} has no observations")
    _finite_number(value.get("classwise_ece"), label=f"{label}.classwise_ece")
    _require_numeric_vector(
        value.get("classwise_ece_by_class"), length=5,
        label=f"{label}.classwise_ece_by_class",
    )
    classes = value.get("classes")
    if not isinstance(classes, list) or len(classes) != 5:
        raise ValueError(f"{label} must contain five classwise reliability tables")
    for grade, row in enumerate(classes):
        if not isinstance(row, Mapping) or int(row.get("class", -1)) != grade:
            raise ValueError(f"{label} class ordering changed")
        _finite_number(row.get("ece"), label=f"{label}.class[{grade}].ece")
        _finite_number(
            row.get("prevalence"), label=f"{label}.class[{grade}].prevalence"
        )
        _validate_reliability_bins(
            row.get("bins"), expected_n=n,
            label=f"{label}.class[{grade}].bins",
        )


def _validate_outer_prediction_archive(path: Path, *, expected_n: int) -> None:
    required = {
        "sample_index", "image_id", "cluster_id", "label", "prediction",
        "expected_grade", "class_probs", "cumulative_probs",
    }
    forbidden = {"image_path", "patient_id", "raw_cluster_id", "sample_id"}
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != required:
            missing = required - set(archive.files)
            unexpected = set(archive.files) - required
            raise ValueError(
                f"outer prediction archive columns changed; missing={sorted(missing)}, "
                f"unexpected={sorted(unexpected)}"
            )
        if forbidden & set(archive.files):
            raise ValueError("outer prediction archive exposes raw identifiers")
        lengths = {name: len(np.asarray(archive[name])) for name in required}
        if set(lengths.values()) != {expected_n}:
            raise ValueError("outer prediction archive row census changed")
        image_ids = np.asarray(archive["image_id"]).astype(str)
        cluster_ids = np.asarray(archive["cluster_id"]).astype(str)
        for name, values in (("image_id", image_ids), ("cluster_id", cluster_ids)):
            if values.ndim != 1 or any(not HEX64.fullmatch(value) for value in values):
                raise ValueError(f"{name} must contain dataset-scoped SHA-256 identifiers")
        if len(set(image_ids.tolist())) != expected_n:
            raise ValueError("outer prediction archive contains duplicate image identifiers")
        if np.asarray(archive["class_probs"]).shape != (expected_n, 5):
            raise ValueError("outer class probability matrix must have shape (N,5)")
        if np.asarray(archive["cumulative_probs"]).shape != (expected_n, 4):
            raise ValueError("outer cumulative probability matrix must have shape (N,4)")


def _validate_worker_posterior_quality(
    payload: Mapping[str, Any], *, label: str
) -> None:
    metrics = payload.get("metrics")
    if not isinstance(metrics, Mapping):
        raise ValueError(f"{label} lacks metrics")
    for name in BASELINE_SCALAR_METRICS:
        _finite_number(metrics.get(name), label=f"{label}.metrics.{name}")
    _require_numeric_vector(
        metrics.get("threshold_ece_by_boundary"), length=4,
        label=f"{label}.metrics.threshold_ece_by_boundary",
    )
    _require_numeric_vector(
        metrics.get("threshold_binary_brier_by_boundary"), length=4,
        label=f"{label}.metrics.threshold_binary_brier_by_boundary",
    )
    _require_numeric_vector(
        metrics.get("classwise_ece_by_class"), length=5,
        label=f"{label}.metrics.classwise_ece_by_class",
    )
    _require_numeric_vector(
        metrics.get("per_grade_recall"), length=5,
        label=f"{label}.metrics.per_grade_recall",
    )
    expected_n = int(metrics.get("n", -1))
    if expected_n < 1:
        raise ValueError(f"{label} has no posterior-quality samples")
    quality = payload.get("posterior_quality")
    if not isinstance(quality, Mapping):
        raise ValueError(f"{label} lacks posterior_quality")
    _validate_threshold_reliability(
        quality.get("threshold_reliability"),
        label=f"{label}.posterior_quality.threshold_reliability",
        expected_n=expected_n,
    )
    _validate_classwise_reliability(
        quality.get("classwise_reliability"),
        label=f"{label}.posterior_quality.classwise_reliability",
        expected_n=expected_n,
    )
    if quality.get("exact_per_sample_probabilities") != (
        "outer_predictions.npz:class_probs"
    ):
        raise ValueError(f"{label} no longer binds exact per-sample probabilities")


def _validate_aggregate_posterior_quality(payload: Mapping[str, Any]) -> None:
    if payload.get("posterior_quality_contract") != BASELINE_POSTERIOR_CONTRACT:
        raise ValueError("matched-baseline posterior-quality contract changed")
    summary = payload.get("summary")
    if not isinstance(summary, Mapping) or set(summary) != {"aptos", "dr"}:
        raise ValueError("matched-baseline dataset coverage is incomplete")
    for dataset in ("aptos", "dr"):
        if set(summary[dataset]) != set(EXPECTED_SHORTCUT_VARIANTS):
            raise ValueError(f"matched-baseline model coverage is incomplete for {dataset}")
        for model in EXPECTED_SHORTCUT_VARIANTS:
            record = summary[dataset][model]
            seed_means = record.get("seed_cv_means", {})
            expected_seeds = {str(seed) for seed in BASELINE_TRAINING_SEEDS}
            if set(seed_means) != expected_seeds:
                raise ValueError("matched-baseline replication seed coverage is incomplete")
            for seed, metrics in seed_means.items():
                if not isinstance(metrics, Mapping) or set(metrics) != set(
                    BASELINE_SUMMARY_METRICS
                ):
                    raise ValueError(f"seed summary metrics changed for {dataset}/{model}/{seed}")
                for name, value in metrics.items():
                    _finite_number(value, label=f"{dataset}/{model}/{seed}/{name}")
            replication = record.get("replication_summary")
            if not isinstance(replication, Mapping) or set(replication) != set(
                BASELINE_SUMMARY_METRICS
            ):
                raise ValueError(f"replication summary metrics changed for {dataset}/{model}")
            for name, metric in replication.items():
                if not isinstance(metric, Mapping):
                    raise ValueError(f"replication summary {name} is malformed")
                _finite_number(
                    metric.get("mean_of_seed_cv_means"), label=f"{name}.mean"
                )
                _finite_number(
                    metric.get("sd_across_seed_cv_means"), label=f"{name}.sd"
                )
                _require_numeric_vector(
                    metric.get("values"), length=3, label=f"{name}.values"
                )
            threshold = record.get("threshold_reliability_by_seed")
            classwise = record.get("classwise_reliability_by_seed")
            if not isinstance(threshold, Mapping) or set(threshold) != expected_seeds:
                raise ValueError("pooled threshold reliability seed coverage is incomplete")
            if not isinstance(classwise, Mapping) or set(classwise) != expected_seeds:
                raise ValueError("pooled classwise reliability seed coverage is incomplete")
            for seed in expected_seeds:
                _validate_threshold_reliability(
                    threshold[seed], label=f"{dataset}/{model}/seed{seed}/threshold"
                )
                _validate_classwise_reliability(
                    classwise[seed], label=f"{dataset}/{model}/seed{seed}/classwise"
                )
                if int(threshold[seed].get("n", -1)) != EXPECTED_DATASET_IMAGES[dataset]:
                    raise ValueError(f"{dataset} pooled threshold reliability census changed")
                if int(classwise[seed].get("n", -1)) != EXPECTED_DATASET_IMAGES[dataset]:
                    raise ValueError(f"{dataset} pooled classwise reliability census changed")

    bootstrap = payload.get("paired_cluster_bootstrap")
    if not isinstance(bootstrap, Mapping) or set(bootstrap) != {"aptos", "dr"}:
        raise ValueError("paired cluster-bootstrap dataset coverage is incomplete")
    comparators = set(EXPECTED_SHORTCUT_VARIANTS) - {"origin_ctmc"}
    for dataset in ("aptos", "dr"):
        if not isinstance(bootstrap[dataset], Mapping) or set(bootstrap[dataset]) != comparators:
            raise ValueError(f"paired comparator coverage is incomplete for {dataset}")
        for comparator, result in bootstrap[dataset].items():
            if not isinstance(result, Mapping) or result.get("reference_variant") != "origin_ctmc":
                raise ValueError("paired bootstrap reference changed")
            if result.get("comparator") != comparator or result.get("dataset") != dataset:
                raise ValueError("paired bootstrap identity changed")
            comparisons = result.get("comparisons")
            if not isinstance(comparisons, Mapping) or set(comparisons) != set(
                BASELINE_BOOTSTRAP_METRICS
            ):
                raise ValueError("paired bootstrap posterior-quality metrics are incomplete")
            for name, comparison in comparisons.items():
                if not isinstance(comparison, Mapping):
                    raise ValueError(f"paired bootstrap {name} is malformed")
                _finite_number(
                    comparison.get("origin_minus_comparator"), label=f"{name}.delta"
                )
                _finite_number(
                    comparison.get("bootstrap_mean_delta"), label=f"{name}.bootstrap_mean"
                )
                _require_numeric_vector(
                    comparison.get("ci95_percentile"), length=2,
                    label=f"{name}.ci95_percentile",
                )
                if not isinstance(comparison.get("higher_is_better"), bool):
                    raise ValueError(f"paired bootstrap {name} direction is missing")


def validate_baseline_aggregate(path: Path, repo_root: Path) -> dict[str, Any]:
    payload = verify_checksummed_payload(
        path, schema="origin-acceptance-outer-aggregate-v2"
    )
    if payload.get("status") != "complete" or int(payload.get("release_worker_count", 0)) != 225:
        raise ValueError("matched outer-release aggregate is incomplete")
    if payload.get("comparator_implementation_scope", {}).get("kind") != (
        "matched_in_repo_analogues"
    ):
        raise ValueError("matched-baseline implementation scope changed")
    _validate_aggregate_posterior_quality(payload)

    release_root = path.parent
    # OUTER_RELEASE_COMPLETE is the atomically written terminal commit marker.
    # Aggregate/CSV files left behind without this hash-binding marker are a
    # partial publication and must never be accepted or packaged.
    marker_path = release_root / "OUTER_RELEASE_COMPLETE.json"
    marker = read_json(marker_path)
    if marker.get("schema") != "origin-acceptance-outer-release-complete-v2":
        raise ValueError("outer-release completion marker schema changed")
    if marker.get("aggregate_sha256") != file_sha256(path):
        raise ValueError("outer-release marker does not bind the aggregate")
    metrics_path = release_root / "fold_metrics.csv"
    _verify_declared_hash(
        metrics_path, marker.get("fold_metrics_sha256"), label="outer fold metrics"
    )
    if int(marker.get("worker_count", 0)) != 225:
        raise ValueError("outer-release marker worker census changed")

    experiment_root = path.parents[2]
    submission = verify_checksummed_payload(
        experiment_root / "SUBMISSION.json", schema="origin-acceptance-submission-v1"
    )
    if submission.get("protocol_sha256") != payload.get("protocol_sha256"):
        raise ValueError("matched-baseline submission/aggregate protocol mismatch")
    execution_provenance = _validate_baseline_execution_provenance(
        experiment_root=experiment_root,
        submission=submission,
        aggregate_protocol_sha256=str(payload.get("protocol_sha256")),
        repo_root=repo_root,
    )
    frozen = verify_checksummed_payload(
        experiment_root / "full" / "TRAINING_FROZEN.json",
        schema="origin-acceptance-training-audit-v1",
    )
    if frozen.get("status") != "passed" or int(frozen.get("worker_count", 0)) != 225:
        raise ValueError("matched-baseline training freeze audit is incomplete")
    if frozen.get("protocol_sha256") != payload.get("protocol_sha256"):
        raise ValueError("matched-baseline freeze/aggregate protocol mismatch")

    workers = frozen.get("workers")
    if not isinstance(workers, list) or len(workers) != 225:
        raise ValueError("matched-baseline checkpoint inventory is incomplete")
    expected_keys = _expected_baseline_task_keys()
    frozen_by_key: dict[str, Mapping[str, Any]] = {}
    for worker in workers:
        task = worker.get("task") if isinstance(worker, Mapping) else None
        if not isinstance(task, Mapping):
            raise ValueError("matched-baseline frozen worker lacks its task identity")
        key = (
            f"{task.get('dataset')}__fold{int(task.get('fold', -1))}__"
            f"{task.get('baseline_variant')}__seed{int(task.get('training_seed', -1))}"
        )
        if key not in expected_keys or key in frozen_by_key:
            raise ValueError(f"matched-baseline frozen task identity is invalid: {key}")
        digest = str(worker.get("best_learned_sha256"))
        if not HEX64.fullmatch(digest):
            raise ValueError(f"matched-baseline checkpoint digest is invalid: {key}")
        fold_dir = Path(str(worker.get("fold_dir", ""))).resolve()
        if not fold_dir.is_relative_to(experiment_root.resolve()):
            raise ValueError(f"matched-baseline checkpoint escapes experiment root: {key}")
        checkpoint = fold_dir / "best_learned.pth"
        _verify_declared_hash(checkpoint, digest, label=f"{key} checkpoint")
        if not HEX64.fullmatch(str(worker.get("split_signature", ""))):
            raise ValueError(f"matched-baseline split signature is invalid: {key}")
        frozen_by_key[key] = worker
    if set(frozen_by_key) != expected_keys:
        raise ValueError("matched-baseline frozen task coverage is incomplete")

    for key in sorted(expected_keys):
        directory = release_root / key
        metrics_file = directory / "outer_metrics.json"
        predictions_file = directory / "outer_predictions.npz"
        complete_file = directory / "OUTER_COMPLETE.json"
        metrics_payload = verify_checksummed_payload(
            metrics_file, schema="origin-acceptance-outer-result-v2"
        )
        if metrics_payload.get("task") != frozen_by_key[key].get("task"):
            raise ValueError(f"matched-baseline worker task identity mismatch: {key}")
        if metrics_payload.get("best_learned_checkpoint_sha256") != (
            frozen_by_key[key].get("best_learned_sha256")
        ):
            raise ValueError(f"matched-baseline worker checkpoint provenance mismatch: {key}")
        if metrics_payload.get("protocol_sha256") != payload.get("protocol_sha256"):
            raise ValueError(f"matched-baseline worker protocol mismatch: {key}")
        if metrics_payload.get("test_evaluated") is not True or metrics_payload.get(
            "selection_reopened"
        ) is not False:
            raise ValueError(f"matched-baseline outer-test semantics changed: {key}")
        privacy = metrics_payload.get("identifier_privacy")
        if (
            not isinstance(privacy, Mapping)
            or privacy.get("dataset_scope") != frozen_by_key[key]["task"]["dataset"]
            or privacy.get("algorithm") != "sha256"
            or privacy.get("raw_image_paths_exported") is not False
            or privacy.get("raw_patient_or_cluster_identifiers_exported") is not False
        ):
            raise ValueError(f"matched-baseline identifier privacy contract changed: {key}")
        _validate_worker_posterior_quality(metrics_payload, label=key)
        n = int(metrics_payload.get("metrics", {}).get("n", -1))
        if n < 1:
            raise ValueError(f"matched-baseline worker has no outer-test records: {key}")
        _verify_declared_hash(
            predictions_file, metrics_payload.get("predictions_sha256"),
            label=f"{key} predictions",
        )
        _validate_outer_prediction_archive(predictions_file, expected_n=n)
        complete = read_json(complete_file)
        if complete.get("schema") != "origin-acceptance-outer-complete-v2":
            raise ValueError(f"matched-baseline worker completion schema changed: {key}")
        if complete.get("outer_metrics_sha256") != file_sha256(metrics_file):
            raise ValueError(f"matched-baseline worker completion hash mismatch: {key}")
        if complete.get("predictions_sha256") != file_sha256(predictions_file):
            raise ValueError(f"matched-baseline prediction completion hash mismatch: {key}")
        if complete.get("protocol_sha256") != payload.get("protocol_sha256"):
            raise ValueError(f"matched-baseline worker completion protocol mismatch: {key}")
        if complete.get("task") != frozen_by_key[key].get("task"):
            raise ValueError(f"matched-baseline worker completion task mismatch: {key}")
    return {
        "schema": payload["schema"],
        "file_sha256": file_sha256(path),
        "content_checksum_sha256": payload["content_checksum_sha256"],
        "source_commit": execution_provenance["source_commit"],
        "parent_canary_source_commit": execution_provenance[
            "parent_canary_source_commit"
        ],
        "continuation_source_commit": execution_provenance[
            "continuation_source_commit"
        ],
        "execution_provenance": execution_provenance,
        "protocol_sha256": payload["protocol_sha256"],
        "release_workers": 225,
        "datasets": ["aptos", "dr"],
        "training_seeds": [42, 31415, 27182],
        "model_variants": list(EXPECTED_SHORTCUT_VARIANTS),
        "identifier_contract": "dataset_scoped_sha256_no_raw_paths_or_patient_ids",
        "posterior_quality_contract_verified": True,
        "per_worker_prediction_archives_verified": 225,
    }


def _assert_privacy_safe_release(payload: Any, *, trail: tuple[str, ...] = ()) -> None:
    """Refuse absolute locations and raw individual identifiers in exported JSON."""

    if isinstance(payload, Mapping):
        for key, value in payload.items():
            lower = str(key).lower()
            allowed_location_keys = {
                "bundle_relative_path", "relative_path", "manifest_filename",
                "archive_uri", "filename",
            }
            if (
                lower not in allowed_location_keys
                and (lower == "path" or lower.endswith("_path") or lower.endswith("_paths"))
            ):
                raise ValueError(f"release payload exposes a filesystem field: {'.'.join(trail + (str(key),))}")
            if lower in {
                "image_id", "patient_id", "cluster_id", "sample_id", "subject_id",
                "case_id", "image_key", "patient_key", "raw_cluster_id",
            }:
                raise ValueError(f"release payload exposes an individual identifier: {lower}")
            _assert_privacy_safe_release(value, trail=trail + (str(key),))
    elif isinstance(payload, (list, tuple)):
        for index, value in enumerate(payload):
            _assert_privacy_safe_release(value, trail=trail + (str(index),))
    elif isinstance(payload, str):
        if (
            payload.startswith("file:")
            or payload.startswith("/")
            or bool(re.match(r"^[A-Za-z]:[\\/]", payload))
            or "\\" in payload
        ):
            raise ValueError(f"release payload exposes a filesystem location at {'.'.join(trail)}")


_DROP_FROM_PUBLIC_JSON = {
    "path", "paths", "audit_root", "audit_files", "cv_root", "fold_dir",
    "input_summary_path", "input_records_path", "prediction_artifact", "checkpoint",
    "image_id", "patient_id", "cluster_id", "sample_id", "raw_cluster_id",
    "subject_id", "case_id", "image_key", "patient_key", "image_path",
    "filename", "images",
}


def _sanitize_public_json(value: Any) -> Any:
    """Remove local locations and row identifiers while retaining negative results."""

    if isinstance(value, Mapping):
        sanitized: dict[str, Any] = {}
        for key, item in value.items():
            lower = str(key).lower()
            if (
                lower in _DROP_FROM_PUBLIC_JSON
                or lower.endswith("_path")
                or lower.endswith("_paths")
            ):
                continue
            sanitized[str(key)] = _sanitize_public_json(item)
        return sanitized
    if isinstance(value, list):
        return [_sanitize_public_json(item) for item in value]
    if isinstance(value, tuple):
        return [_sanitize_public_json(item) for item in value]
    if isinstance(value, str) and (
        value.startswith("/")
        or value.startswith("file:")
        or bool(re.match(r"^[A-Za-z]:[\\/]", value))
        or "\\" in value
    ):
        return "[local-location-withheld]"
    return value


def _write_sanitized_json_copy(
    source: Path, destination: Path, *, role: str
) -> None:
    source_payload = read_json(source)
    envelope: dict[str, Any] = {
        "schema": "origin-acceptance-sanitized-artifact-v1",
        "role": role,
        "source_schema": source_payload.get("schema"),
        "source_file_sha256": file_sha256(source),
        "source_content_checksum_sha256": source_payload.get(
            "content_checksum_sha256"
        ),
        "sanitized_payload": _sanitize_public_json(source_payload),
        "privacy_transform": (
            "absolute filesystem fields and individual identifiers removed; "
            "scientific pass/fail values retained"
        ),
    }
    envelope["content_checksum_sha256"] = canonical_sha256(envelope)
    _assert_privacy_safe_release(envelope)
    destination.parent.mkdir(parents=True, exist_ok=True)
    _atomic_json(destination, envelope)


class _BundleBuilder:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.entries: list[dict[str, Any]] = []

    def _destination(self, relative: str) -> Path:
        destination = (self.root / relative).resolve()
        if not destination.is_relative_to(self.root.resolve()):
            raise ValueError("bundle destination escapes bundle root")
        if destination.exists():
            raise FileExistsError(f"duplicate bundle destination: {relative}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        return destination

    def copy(self, source: Path, relative: str, *, role: str) -> None:
        destination = self._destination(relative)
        shutil.copyfile(source, destination)
        self.entries.append({
            "bundle_relative_path": relative,
            "role": role,
            "sha256": file_sha256(destination),
            "bytes": destination.stat().st_size,
        })

    def sanitized_json(self, source: Path, relative: str, *, role: str) -> None:
        destination = self._destination(relative)
        _write_sanitized_json_copy(source, destination, role=role)
        self.entries.append({
            "bundle_relative_path": relative,
            "role": role,
            "sha256": file_sha256(destination),
            "bytes": destination.stat().st_size,
        })

    def json(self, payload: Mapping[str, Any], relative: str, *, role: str) -> None:
        destination = self._destination(relative)
        value = dict(payload)
        value["content_checksum_sha256"] = canonical_sha256(value)
        _assert_privacy_safe_release(value)
        _atomic_json(destination, value)
        self.entries.append({
            "bundle_relative_path": relative,
            "role": role,
            "sha256": file_sha256(destination),
            "bytes": destination.stat().st_size,
        })


def _stable_archive_reference(
    uri: str | None, sha256: str | None, size_bytes: int | None, *, kind: str
) -> dict[str, Any]:
    if uri is None and sha256 is None and size_bytes is None:
        return {
            "status": "not_supplied",
            "archive_uri": None,
            "archive_sha256": None,
            "archive_bytes": None,
        }
    if uri is None or sha256 is None:
        raise ValueError(f"{kind} archive URI and SHA-256 must be supplied together")
    if not (
        uri.startswith("https://") or uri.startswith("doi:") or uri.startswith("zenodo:")
    ):
        raise ValueError(
            f"{kind} archive URI must be a stable https/doi/zenodo reference"
        )
    if not HEX64.fullmatch(sha256):
        raise ValueError(f"{kind} archive SHA-256 is invalid")
    if size_bytes is not None and size_bytes < 1:
        raise ValueError(f"{kind} archive byte count must be positive")
    return {
        "status": "externally_archived",
        "archive_uri": uri,
        "archive_sha256": sha256,
        "archive_bytes": size_bytes,
    }


def _artifact_release_sources(
    artifact_manifest: Path,
) -> tuple[list[dict[str, Any]], list[tuple[Path, str]], list[dict[str, Any]]]:
    payload = read_json(artifact_manifest)
    cv_root = Path(str(payload["cv_root"])).resolve()
    checkpoint_inventory: list[dict[str, Any]] = []
    membership_sources: list[tuple[Path, str]] = []
    membership_inventory: list[dict[str, Any]] = []
    for record in sorted(payload["checkpoints"], key=lambda row: (row["dataset"], row["fold"])):
        dataset, fold = str(record["dataset"]), int(record["fold"])
        checkpoint = _resolve_declared_file(
            record["relative_path"], relative_to=cv_root, allowed_roots=(cv_root,)
        )
        checkpoint_inventory.append({
            "alias": f"v3/{dataset}/fold{fold}",
            "dataset": dataset,
            "fold": fold,
            "model_variant": "origin_ctmc",
            "training_seed": None,
            "sha256": file_sha256(checkpoint),
            "bytes": checkpoint.stat().st_size,
            "checkpoint_schema": record.get("checkpoint_schema"),
            "split_signature": record.get("split_signature"),
            "implementation_signature": record.get("implementation_signature"),
            "architecture_signature": record.get("architecture_signature"),
            "config_signature": record.get("config_signature"),
            "tensor_location": "checkpoint_archive",
        })
    for record in sorted(payload["split_memberships"], key=lambda row: (row["dataset"], row["fold"])):
        dataset, fold = str(record["dataset"]), int(record["fold"])
        source = _resolve_declared_file(
            record["relative_path"], relative_to=artifact_manifest.parent,
            allowed_roots=(artifact_manifest.parent,),
        )
        relative = f"memberships/{dataset}_fold{fold}.csv.gz"
        membership_sources.append((source, relative))
        membership_inventory.append({
            "alias": f"v3/{dataset}/fold{fold}",
            "dataset": dataset,
            "fold": fold,
            "rows": int(record["rows"]),
            "sha256": file_sha256(source),
            "bundle_relative_path": relative,
        })
    return checkpoint_inventory, membership_sources, membership_inventory


def _baseline_release_sources(
    aggregate_path: Path,
) -> tuple[list[dict[str, Any]], list[tuple[Path, str, str]]]:
    release_root = aggregate_path.parent
    experiment_root = aggregate_path.parents[2]
    frozen = read_json(experiment_root / "full" / "TRAINING_FROZEN.json")
    checkpoint_inventory: list[dict[str, Any]] = []
    for worker in frozen["workers"]:
        task = worker["task"]
        key = (
            f"{task['dataset']}__fold{int(task['fold'])}__"
            f"{task['baseline_variant']}__seed{int(task['training_seed'])}"
        )
        checkpoint = Path(str(worker["fold_dir"])).resolve() / "best_learned.pth"
        checkpoint_inventory.append({
            "alias": f"baseline/{key}",
            "dataset": task["dataset"],
            "fold": int(task["fold"]),
            "model_variant": task["baseline_variant"],
            "training_seed": int(task["training_seed"]),
            "sha256": file_sha256(checkpoint),
            "bytes": checkpoint.stat().st_size,
            "split_signature": worker.get("split_signature"),
            "tensor_location": "checkpoint_archive",
        })
    sources: list[tuple[Path, str, str]] = []
    for key in sorted(_expected_baseline_task_keys()):
        directory = release_root / key
        sources.extend((
            (
                directory / "outer_metrics.json",
                f"matched_baselines/workers/{key}/outer_metrics.json",
                "matched_baseline_outer_metrics",
            ),
            (
                directory / "outer_predictions.npz",
                f"matched_baselines/workers/{key}/outer_predictions.npz",
                "matched_baseline_privacy_safe_per_image_probabilities",
            ),
        ))
    return checkpoint_inventory, sources


def _build_release_bundle(
    *, repo_root: Path, bundle_root: Path, numerical_audit: Path,
    decoder_contract_audit: Path,
    artifact_manifest: Path, idrid_manifest: Path, oof_manifest: Path,
    shortcut_aggregate: Path, baseline_aggregate: Path,
    checkpoint_archive: Mapping[str, Any],
) -> dict[str, Any]:
    builder = _BundleBuilder(bundle_root)
    builder.copy(numerical_audit, "audits/decoder_numeric.json", role="decoder_numeric_audit")
    builder.copy(
        decoder_contract_audit, "audits/decoder_contract.json",
        role="decoder_contract_audit",
    )

    checkpoint_inventory, memberships, membership_inventory = (
        _artifact_release_sources(artifact_manifest)
    )
    for source, relative in memberships:
        builder.copy(source, relative, role="anonymous_split_membership")

    idrid = read_json(idrid_manifest)
    builder.sanitized_json(
        idrid_manifest, "idrid/manifest_sanitized.json", role="idrid_manifest"
    )
    for name in ("alignment_statistics", "deletion_statistics"):
        record = idrid["artifacts"][name]
        source = _resolve_declared_file(
            record["path"], relative_to=idrid_manifest.parent,
            allowed_roots=(idrid_manifest.parent,),
        )
        builder.sanitized_json(
            source, f"idrid/{name}.json", role=f"idrid_{name}"
        )

    oof = read_json(oof_manifest)
    builder.sanitized_json(oof_manifest, "oof/manifest_sanitized.json", role="oof_manifest")
    for name, record in sorted(oof["artifacts"].items()):
        source = _resolve_declared_file(
            record["path"], relative_to=oof_manifest.parent,
            allowed_roots=(oof_manifest.parent,),
        )
        builder.sanitized_json(source, f"oof/{name}.json", role=f"oof_{name}")
    builder.copy(
        repo_root / "scripts" / "protocols" / "origin_oof_statistics_protocol.json",
        "protocols/origin_oof_statistics_protocol.json", role="oof_protocol",
    )
    builder.copy(
        repo_root / "scripts" / "protocols" / "origin_idrid_semantic_statistics_protocol.json",
        "protocols/origin_idrid_semantic_statistics_protocol.json", role="idrid_protocol",
    )

    builder.sanitized_json(
        shortcut_aggregate, "shortcut/gate_b_aggregate_sanitized.json",
        role="shortcut_gate_b_aggregate",
    )
    for sibling, relative, role in (
        (shortcut_aggregate.parent / "PROTOCOL_V2.json", "shortcut/PROTOCOL_V2.json", "shortcut_protocol"),
        (shortcut_aggregate.parent / "SUBMISSION.json", "shortcut/SUBMISSION.json", "shortcut_submission"),
    ):
        builder.sanitized_json(sibling, relative, role=role)

    builder.copy(
        baseline_aggregate, "matched_baselines/aggregate.json",
        role="matched_baseline_aggregate",
    )
    builder.copy(
        baseline_aggregate.parent / "fold_metrics.csv",
        "matched_baselines/fold_metrics.csv", role="matched_baseline_fold_metrics",
    )
    baseline_experiment = baseline_aggregate.parents[2]
    builder.sanitized_json(
        baseline_experiment / "SUBMISSION.json",
        "matched_baselines/SUBMISSION_sanitized.json",
        role="matched_baseline_submission_protocol_binding",
    )
    restart_submission = baseline_experiment / "RESTART_SUBMISSION.json"
    if restart_submission.is_file():
        builder.sanitized_json(
            restart_submission,
            "matched_baselines/RESTART_SUBMISSION_sanitized.json",
            role="matched_baseline_continuation_protocol_binding",
        )
    builder.sanitized_json(
        baseline_experiment / "full" / "TRAINING_FROZEN.json",
        "matched_baselines/TRAINING_FROZEN_sanitized.json",
        role="matched_baseline_training_freeze_inventory",
    )
    builder.copy(
        repo_root / "scripts" / "origin_acceptance_baseline_common.py",
        "protocols/origin_acceptance_baseline_common.py",
        role="matched_baseline_executable_protocol",
    )
    baseline_checkpoints, baseline_sources = _baseline_release_sources(
        baseline_aggregate
    )
    checkpoint_inventory.extend(baseline_checkpoints)
    tensor_location = (
        "external_checkpoint_archive"
        if checkpoint_archive.get("status") == "externally_archived"
        else "not_available_in_public_package"
    )
    for record in checkpoint_inventory:
        record["tensor_location"] = tensor_location
    for source, relative, role in baseline_sources:
        builder.copy(source, relative, role=role)

    checkpoint_inventory_payload = {
        "schema": "origin-acceptance-checkpoint-inventory-v1",
        "count": len(checkpoint_inventory),
        "archive_reference": dict(checkpoint_archive),
        "checkpoints": checkpoint_inventory,
    }
    builder.json(
        checkpoint_inventory_payload, "inventories/checkpoints.json",
        role="checkpoint_inventory",
    )
    builder.json(
        {
            "schema": "origin-acceptance-membership-inventory-v1",
            "count": len(membership_inventory),
            "memberships": membership_inventory,
        },
        "inventories/split_memberships.json", role="split_membership_inventory",
    )

    index_payload: dict[str, Any] = {
        "schema": "origin-acceptance-inspectable-bundle-v1",
        "entry_count": len(builder.entries),
        "entries": sorted(builder.entries, key=lambda row: row["bundle_relative_path"]),
        "licensed_pixels_copied": False,
        "raw_image_or_patient_identifiers_copied": False,
        "baseline_per_image_archives": 225,
        "oof_and_shortcut_per_image_archives": 0,
        "oof_and_shortcut_per_image_availability": (
            "external_privacy_safe_evidence_archive_required"
        ),
        "anonymous_split_memberships": 15,
        "checkpoint_tensors_embedded": False,
    }
    index_payload["content_checksum_sha256"] = canonical_sha256(index_payload)
    _assert_privacy_safe_release(index_payload)
    index_path = bundle_root / BUNDLE_INDEX_FILENAME
    _atomic_json(index_path, index_payload)
    return {
        "schema": index_payload["schema"],
        "bundle_relative_path": f"{BUNDLE_DIRECTORY}/{BUNDLE_INDEX_FILENAME}",
        "file_sha256": file_sha256(index_path),
        "content_checksum_sha256": index_payload["content_checksum_sha256"],
        "entry_count": index_payload["entry_count"],
        "bytes": sum(int(row["bytes"]) for row in builder.entries),
    }


def _assert_bundle_privacy(bundle_root: Path) -> None:
    """Fail closed on location/identifier leakage in every exported artifact."""

    for path in sorted(bundle_root.rglob("*")):
        if not path.is_file():
            continue
        if path.suffix == ".json":
            _assert_privacy_safe_release(read_json(path))
        elif path.suffix == ".npz":
            with np.load(path, allow_pickle=False) as archive:
                if "label" not in archive.files:
                    raise ValueError(f"bundled prediction archive has no labels: {path.name}")
                expected_n = len(np.asarray(archive["label"]))
            _validate_outer_prediction_archive(path, expected_n=expected_n)
        elif path.name.endswith(".csv.gz"):
            with gzip.open(path, "rt", encoding="utf-8", newline="") as stream:
                reader = csv.DictReader(stream)
                if reader.fieldnames != [
                    "image_id_sha256", "patient_cluster_sha256", "label", "split"
                ]:
                    raise ValueError("bundled membership is not anonymized")
                for row in reader:
                    if not HEX64.fullmatch(str(row["image_id_sha256"])):
                        raise ValueError("bundled membership exposes an image identifier")
                    if not HEX64.fullmatch(str(row["patient_cluster_sha256"])):
                        raise ValueError("bundled membership exposes a patient identifier")
        elif path.suffix == ".csv":
            text = path.read_text(encoding="utf-8")
            if re.search(r"(?:^|[,\n])(?:/|[A-Za-z]:[\\/])", text):
                raise ValueError(f"bundled CSV exposes an absolute filesystem location: {path.name}")


def assemble_package(
    *, repo_root: Path, output_dir: Path, numerical_audit: Path,
    decoder_contract_audit: Path, artifact_manifest: Path,
    idrid_manifest: Path, oof_manifest: Path, shortcut_aggregate: Path,
    baseline_aggregate: Path, expected_commit: str | None = None,
    require_gates_pass: bool = False,
    checkpoint_archive_uri: str | None = None,
    checkpoint_archive_sha256: str | None = None,
    checkpoint_archive_bytes: int | None = None,
    evidence_archive_uri: str | None = None,
    evidence_archive_sha256: str | None = None,
    evidence_archive_bytes: int | None = None,
) -> dict[str, Any]:
    repo_root = repo_root.resolve()
    output_dir = output_dir.resolve()
    manifest_path = output_dir / MANIFEST_FILENAME
    completion_path = output_dir / COMPLETION_FILENAME
    if output_dir.exists():
        raise FileExistsError("refusing to overwrite an existing acceptance package")
    provenance = runtime_provenance(repo_root, expected_commit)
    components: dict[str, Any] = {
        "independent_numerical_decoder_audit": validate_numerical(numerical_audit),
        "decoder_contract_audit": validate_decoder_contract(decoder_contract_audit),
        "checkpoint_split_artifact_audit": validate_artifact_manifest(
            artifact_manifest, repo_root
        ),
        "idrid_semantic_cluster_audit": validate_idrid_manifest(
            idrid_manifest, repo_root
        ),
    }
    oof_record, gate_a = validate_oof_manifest(oof_manifest, repo_root)
    shortcut_record, gate_b_passed = validate_shortcut_aggregate(
        shortcut_aggregate, repo_root
    )
    components["oof_gate_a"] = oof_record
    components["shortcut_gate_b"] = shortcut_record
    components["matched_baseline_outer_release"] = validate_baseline_aggregate(
        baseline_aggregate, repo_root
    )
    gates_pass = gate_a == "pass" and gate_b_passed
    if require_gates_pass and not gates_pass:
        raise RuntimeError(
            f"submission-readiness withheld: Gate A={gate_a}, "
            f"Gate B={'pass' if gate_b_passed else 'fail'}"
        )
    checkpoint_archive = _stable_archive_reference(
        checkpoint_archive_uri, checkpoint_archive_sha256,
        checkpoint_archive_bytes, kind="checkpoint",
    )
    evidence_archive = _stable_archive_reference(
        evidence_archive_uri, evidence_archive_sha256,
        evidence_archive_bytes, kind="per-image evidence",
    )
    reproduction_authorized = (
        checkpoint_archive["status"] == "externally_archived"
        and evidence_archive["status"] == "externally_archived"
    )
    temporary_dir = output_dir.with_name(f".{output_dir.name}.building")
    if temporary_dir.exists():
        raise FileExistsError(f"stale package staging directory exists: {temporary_dir.name}")
    temporary_dir.mkdir(parents=True)
    try:
        bundle = _build_release_bundle(
            repo_root=repo_root,
            bundle_root=temporary_dir / BUNDLE_DIRECTORY,
            numerical_audit=numerical_audit,
            decoder_contract_audit=decoder_contract_audit,
            artifact_manifest=artifact_manifest,
            idrid_manifest=idrid_manifest,
            oof_manifest=oof_manifest,
            shortcut_aggregate=shortcut_aggregate,
            baseline_aggregate=baseline_aggregate,
            checkpoint_archive=checkpoint_archive,
        )
        _assert_bundle_privacy(temporary_dir / BUNDLE_DIRECTORY)
    except Exception:
        shutil.rmtree(temporary_dir, ignore_errors=True)
        raise
    payload: dict[str, Any] = {
        "schema": PACKAGE_SCHEMA,
        "status": "verified_complete",
        "assembly_source": provenance,
        "required_component_count": 7,
        "components": components,
        "scientific_gates": {
            "gate_a": gate_a,
            "gate_b": "pass" if gate_b_passed else "fail",
            "all_passed": gates_pass,
            "claims_authorized": gates_pass,
            "claim_policy": (
                "auditability claims are withheld unless both preregistered gates pass"
            ),
        },
        "claim_authorized": gates_pass,
        "inspectable_bundle": bundle,
        "artifact_availability": {
            "status": (
                "complete_with_external_archives"
                if reproduction_authorized
                else "incomplete_external_archives"
            ),
            "checkpoint_tensors": checkpoint_archive,
            "remaining_per_image_evidence": evidence_archive,
            "checkpoint_inventory_count": 240,
            "privacy_safe_baseline_prediction_archives_bundled": 225,
            "anonymous_split_memberships_bundled": 15,
            "licensed_input_pixels_required_but_not_redistributed": True,
            "independent_reproduction_claim_authorized": reproduction_authorized,
            "withheld_for_privacy_or_licensing": [
                "licensed retinal image pixels",
                "raw IDRiD image-level identifiers",
                "raw OOF intervention rows and shortcut worker records pending a "
                "privacy-safe evidence archive",
            ],
        },
        "release_contract": {
            "input_artifacts_embedded": True,
            "licensed_pixels_copied": False,
            "checkpoints_copied": False,
            "raw_filesystem_locations_exported": False,
            "raw_image_or_patient_identifiers_exported": False,
            "exported_content": (
                "sanitized audits, protocols, anonymous split memberships, matched "
                "baseline per-image probabilities, reliability tables, and inventories"
            ),
            "public_reproducibility_policy": (
                "independent reproduction is authorized only when checkpoint tensors "
                "and remaining privacy-safe per-image evidence are each backed by a "
                "stable archive URI and SHA-256"
            ),
        },
    }
    payload["content_checksum_sha256"] = canonical_sha256(payload)
    _assert_privacy_safe_release(payload)

    staged_manifest = temporary_dir / MANIFEST_FILENAME
    staged_completion = temporary_dir / COMPLETION_FILENAME
    _atomic_json(staged_manifest, payload)
    completion: dict[str, Any] = {
        "schema": COMPLETION_SCHEMA,
        "status": "verified_complete",
        "manifest_filename": MANIFEST_FILENAME,
        "manifest_sha256": file_sha256(staged_manifest),
        "manifest_content_checksum_sha256": payload["content_checksum_sha256"],
        "all_required_components_verified": True,
        "scientific_gates_passed": gates_pass,
        "scientific_claims_authorized": gates_pass,
        "independent_reproduction_claim_authorized": reproduction_authorized,
        "artifact_availability_status": payload["artifact_availability"]["status"],
        "bundle_index_sha256": bundle["file_sha256"],
    }
    completion["content_checksum_sha256"] = canonical_sha256(completion)
    _assert_privacy_safe_release(completion)
    _atomic_json(staged_completion, completion)
    _assert_privacy_safe_release(read_json(staged_manifest))
    _assert_privacy_safe_release(read_json(staged_completion))
    temporary_dir.replace(output_dir)
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--numerical-audit", type=Path, required=True)
    parser.add_argument("--decoder-contract-audit", type=Path, required=True)
    parser.add_argument("--artifact-manifest", type=Path, required=True)
    parser.add_argument("--idrid-manifest", type=Path, required=True)
    parser.add_argument("--oof-manifest", type=Path, required=True)
    parser.add_argument("--shortcut-aggregate", type=Path, required=True)
    parser.add_argument("--baseline-aggregate", type=Path, required=True)
    parser.add_argument("--expected-commit")
    parser.add_argument("--require-gates-pass", action="store_true")
    parser.add_argument("--checkpoint-archive-uri")
    parser.add_argument("--checkpoint-archive-sha256")
    parser.add_argument("--checkpoint-archive-bytes", type=int)
    parser.add_argument("--evidence-archive-uri")
    parser.add_argument("--evidence-archive-sha256")
    parser.add_argument("--evidence-archive-bytes", type=int)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    payload = assemble_package(
        repo_root=args.repo_root,
        output_dir=args.output_dir,
        numerical_audit=args.numerical_audit,
        decoder_contract_audit=args.decoder_contract_audit,
        artifact_manifest=args.artifact_manifest,
        idrid_manifest=args.idrid_manifest,
        oof_manifest=args.oof_manifest,
        shortcut_aggregate=args.shortcut_aggregate,
        baseline_aggregate=args.baseline_aggregate,
        expected_commit=args.expected_commit,
        require_gates_pass=args.require_gates_pass,
        checkpoint_archive_uri=args.checkpoint_archive_uri,
        checkpoint_archive_sha256=args.checkpoint_archive_sha256,
        checkpoint_archive_bytes=args.checkpoint_archive_bytes,
        evidence_archive_uri=args.evidence_archive_uri,
        evidence_archive_sha256=args.evidence_archive_sha256,
        evidence_archive_bytes=args.evidence_archive_bytes,
    )
    print(json.dumps({
        "status": payload["status"],
        "manifest": MANIFEST_FILENAME,
        "completion_marker": COMPLETION_FILENAME,
        "content_checksum_sha256": payload["content_checksum_sha256"],
        "scientific_gates": payload["scientific_gates"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
