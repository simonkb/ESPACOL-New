#!/bin/bash
# Authorize exactly one fresh, validation-only restart of the failed fold-9
# coarse-scale ablation while preserving the signed scientific protocol.
#
# This is an incident-specific administrative recovery.  It archives the two
# failed attempts, then invokes task 7 of the ORIGINAL immutable training array.
# It never edits LOCKED_PROTOCOL.json and never changes a training argument.
#
# Usage:
#   bash scripts/restart_origin_fold9_ablation_variant_fresh.sh [--dry-run] \
#     /absolute/path/to/experiment \
#     origin_coarse_only \
#     repeated_forward_nan_after_exact_resume

set -Eeuo pipefail

EXPECTED_VARIANT="origin_coarse_only"
EXPECTED_INDEX=7
EXPECTED_LAST_EPOCH=34
EXPECTED_REASON="repeated_forward_nan_after_exact_resume"
DRY_RUN=false

if [[ "${1:-}" == "--dry-run" ]]; then
  DRY_RUN=true
  shift
fi
if [[ $# -ne 3 ]]; then
  echo "Usage: $0 [--dry-run] EXPERIMENT_ROOT origin_coarse_only ${EXPECTED_REASON}" >&2
  exit 64
fi

EXPERIMENT_ROOT="$1"
VARIANT="$2"
REASON="$3"
if [[ "${VARIANT}" != "${EXPECTED_VARIANT}" ]]; then
  echo "This recovery is restricted to ${EXPECTED_VARIANT}; received ${VARIANT}." >&2
  exit 65
fi
if [[ "${REASON}" != "${EXPECTED_REASON}" ]]; then
  echo "The explicit one-time reason must be ${EXPECTED_REASON}." >&2
  exit 66
fi
if (( BASH_VERSINFO[0] < 4 )); then
  echo "This cluster launcher requires Bash 4 or newer." >&2
  exit 67
fi

set +u
source /etc/profile.d/lmod.sh || exit 1
module load miniconda/3 || exit 1
source activate "${ORIGIN_CONDA_ENV:-G}" || exit 1
set -u

for command_name in flock git python sbatch scancel scontrol sha256sum squeue; do
  command -v "${command_name}" >/dev/null || {
    echo "Required command is unavailable: ${command_name}." >&2
    exit 67
  }
done

EXPERIMENT_ROOT="$(python -c 'import pathlib,sys; print(pathlib.Path(sys.argv[1]).resolve())' "${EXPERIMENT_ROOT}")"
LAUNCHER_PATH="$(python -c 'import pathlib,sys; print(pathlib.Path(sys.argv[1]).resolve())' "${BASH_SOURCE[0]}")"
LAUNCHER_SHA256="$(sha256sum "${LAUNCHER_PATH}" | awk '{print $1}')"
PROTOCOL="${EXPERIMENT_ROOT}/LOCKED_PROTOCOL.json"
[[ -f "${PROTOCOL}" ]] || { echo "Missing locked protocol: ${PROTOCOL}." >&2; exit 68; }
[[ -d "${EXPERIMENT_ROOT}/locks" ]] || {
  echo "Missing experiment lock directory: ${EXPERIMENT_ROOT}/locks." >&2
  exit 69
}

# Serialize against the supported resume launcher and retain the variant lock
# until all archive records and Slurm job identifiers have been written.
exec 8>"${EXPERIMENT_ROOT}/locks/retry_submission.lock"
flock -n 8 || {
  echo "Another retry/restart launcher owns ${EXPERIMENT_ROOT}." >&2
  exit 70
}
exec 9>"${EXPERIMENT_ROOT}/locks/${VARIANT}.lock"
flock -n 9 || {
  echo "Variant ${VARIANT} is still owned by another worker." >&2
  exit 71
}
exec 7>"${EXPERIMENT_ROOT}/locks/outer_release.lock"
flock -n 7 || {
  echo "Another process owns the outer-release lock." >&2
  exit 71
}

# Read only bootstrap identities with the standard library.  All semantic and
# checksum verification below imports helpers from this exact signed snapshot.
PROTOCOL_IDENTITY_RAW="$(python - "${PROTOCOL}" <<'PY'
import json
import pathlib
import sys

with pathlib.Path(sys.argv[1]).open(encoding="utf-8") as stream:
    payload = json.load(stream)
for key in ("immutable_worktree", "launch_commit", "data_root"):
    value = payload.get(key)
    if not isinstance(value, str) or not value:
        raise SystemExit(f"locked protocol has no valid {key}")
    print(value)
PY
)"
mapfile -t PROTOCOL_IDENTITY <<< "${PROTOCOL_IDENTITY_RAW}"
[[ ${#PROTOCOL_IDENTITY[@]} -eq 3 ]] || {
  echo "Could not read the three protocol identities." >&2
  exit 72
}
SNAPSHOT_ROOT="$(python -c 'import pathlib,sys; print(pathlib.Path(sys.argv[1]).resolve())' "${PROTOCOL_IDENTITY[0]}")"
LAUNCH_COMMIT="${PROTOCOL_IDENTITY[1]}"
DATA_ROOT="$(python -c 'import pathlib,sys; print(pathlib.Path(sys.argv[1]).resolve())' "${PROTOCOL_IDENTITY[2]}")"
PROJECT_ROOT="$(python -c 'import pathlib,sys; print(pathlib.Path(sys.argv[1]).resolve().parents[1])' "${DATA_ROOT}")"
LOG_ROOT="${PROJECT_ROOT}/origin_fold9_ablation_logs"

[[ -d "${SNAPSHOT_ROOT}/.git" || -f "${SNAPSHOT_ROOT}/.git" ]] || {
  echo "Immutable worktree is missing: ${SNAPSHOT_ROOT}." >&2
  exit 73
}
[[ "$(git -C "${SNAPSHOT_ROOT}" rev-parse HEAD)" == "${LAUNCH_COMMIT}" ]] || {
  echo "Immutable worktree commit differs from LOCKED_PROTOCOL.json." >&2
  exit 74
}
[[ -z "$(git -C "${SNAPSHOT_ROOT}" status --porcelain --untracked-files=no)" ]] || {
  echo "Immutable worktree contains tracked changes." >&2
  exit 75
}
for required in \
  scripts/origin_fold9_ablation_common.py \
  scripts/submit_origin_fold9_ablation_train_array.sh \
  scripts/submit_origin_fold9_ablation_outer_release.sh \
  scripts/submit_origin_fold9_ablation_aggregate.sh; do
  [[ -f "${SNAPSHOT_ROOT}/${required}" ]] || {
    echo "Launch snapshot lacks recovery component ${required}." >&2
    exit 76
  }
done
[[ -d "${LOG_ROOT}" ]] || { echo "Missing ablation log directory: ${LOG_ROOT}." >&2; exit 77; }

cd "${SNAPSHOT_ROOT}"

# Validate the complete pre-restart state.  This intentionally requires the
# incident-specific epoch and error signature, a finite saved state, exactly
# one incomplete variant, and no evidence that the outer release has begun.
ORIGIN_FRESH_PROTOCOL="${PROTOCOL}" \
ORIGIN_FRESH_COMMIT="${LAUNCH_COMMIT}" \
ORIGIN_FRESH_ROOT="${EXPERIMENT_ROOT}" \
ORIGIN_FRESH_DATA="${DATA_ROOT}" \
ORIGIN_FRESH_WORKTREE="${SNAPSHOT_ROOT}" \
ORIGIN_FRESH_LOG_ROOT="${LOG_ROOT}" \
ORIGIN_FRESH_VARIANT="${VARIANT}" \
ORIGIN_FRESH_EXPECTED_INDEX="${EXPECTED_INDEX}" \
ORIGIN_FRESH_EXPECTED_EPOCH="${EXPECTED_LAST_EPOCH}" \
PYTHONPATH="${SNAPSHOT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}" \
python - <<'PY'
import os
from pathlib import Path
from typing import Any, Mapping

import torch

from scripts.origin_fold9_ablation_common import (
    EXPECTED_SPLIT_SIGNATURE,
    FOLD,
    TRAIN_MARKER_SCHEMA,
    file_sha256,
    read_json,
    validate_split_manifest,
    validate_variant_config,
    verify_checksummed_payload,
    verify_protocol,
)
from training.origin_ablation_trainer import origin_ablation_implementation_signature


def require_absent_outer_artifacts(root: Path) -> None:
    forbidden = (
        root / "release",
        root / "OUTER_RELEASE_COMPLETE.json",
        root / "FOLD9_ABLATION_RESULTS.json",
        root / "FOLD9_ABLATION_RESULTS.csv",
        root / "FOLD9_ABLATION_RESULTS.md",
    )
    present = [
        str(path) for path in forbidden if path.exists() or path.is_symlink()
    ]
    present.extend(str(path) for path in sorted(root.glob(".release.staging.*")))
    if present:
        raise RuntimeError(f"outer release/aggregation evidence exists: {present}")


def bad_tensors(value: Any, prefix: str = "") -> list[str]:
    bad: list[str] = []
    if torch.is_tensor(value):
        if (value.is_floating_point() or value.is_complex()) and not bool(
            torch.isfinite(value).all()
        ):
            bad.append(prefix or "<root>")
    elif isinstance(value, Mapping):
        for key, item in value.items():
            child = f"{prefix}.{key}" if prefix else str(key)
            bad.extend(bad_tensors(item, child))
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            bad.extend(bad_tensors(item, f"{prefix}[{index}]"))
    return bad


def load_record(path: Path, schema: str) -> Mapping[str, Any]:
    record = read_json(path)
    if not isinstance(record, Mapping):
        raise TypeError(f"operational record is not an object: {path}")
    verify_checksummed_payload(record, schema=schema)
    return record


def collect_log_paths(root: Path, log_root: Path, variant: str, index: int) -> list[Path]:
    protocol_path = (root / "LOCKED_PROTOCOL.json").resolve()
    protocol = read_json(protocol_path)
    submission_path = root / "SUBMISSION.json"
    submission = load_record(
        submission_path, "origin-fold9-ablation-submission-v1"
    )
    jobs = submission.get("jobs", {})
    if Path(str(submission.get("experiment_root"))).resolve() != root:
        raise RuntimeError("SUBMISSION.json experiment root changed")
    if Path(str(submission.get("protocol"))).resolve() != protocol_path:
        raise RuntimeError("SUBMISSION.json protocol path changed")
    if submission.get("protocol_sha256") != file_sha256(protocol_path):
        raise RuntimeError("SUBMISSION.json protocol hash changed")
    if submission.get("launch_commit") != protocol.get("launch_commit"):
        raise RuntimeError("SUBMISSION.json launch commit changed")
    submitted_variants = list(submission.get("selected_variants", ()))
    if index >= len(submitted_variants) or submitted_variants[index] != variant:
        raise RuntimeError("SUBMISSION.json task index no longer identifies the variant")
    original_train = str(jobs.get("validation_training_array", ""))
    original_release = str(jobs.get("single_outer_release", ""))
    original_aggregate = str(jobs.get("afterany_aggregate", ""))
    if not all((original_train, original_release, original_aggregate)):
        raise RuntimeError("SUBMISSION.json omits an original job identity")
    paths = [
        log_root / f"train_{original_train}_{index}.out",
        log_root / f"train_{original_train}_{index}.err",
        log_root / f"release_{original_release}.out",
        log_root / f"release_{original_release}.err",
        log_root / f"aggregate_{original_aggregate}.out",
        log_root / f"aggregate_{original_aggregate}.err",
    ]
    original_error = paths[1]
    release_errors = [paths[3]]
    aggregate_errors = [paths[5]]

    retry_paths = sorted(root.glob("RETRY_SUBMISSION_*.json"))
    if len(retry_paths) != 1:
        raise RuntimeError(
            "incident-specific fresh restart requires exactly one prior retry "
            f"record; observed {len(retry_paths)}"
        )
    retry_count = 0
    retry_errors: list[Path] = []
    retry_outputs: list[Path] = []
    for retry_path in retry_paths:
        retry = load_record(
            retry_path, "origin-fold9-ablation-retry-submission-v1"
        )
        if Path(str(retry.get("protocol_path"))).resolve() != protocol_path:
            raise RuntimeError(f"retry protocol path changed: {retry_path}")
        if retry.get("protocol_file_sha256") != file_sha256(protocol_path):
            raise RuntimeError(f"retry protocol hash changed: {retry_path}")
        if retry.get("launch_commit") != protocol.get("launch_commit"):
            raise RuntimeError(f"retry launch commit changed: {retry_path}")
        variants = list(retry.get("resumed_variants", ()))
        if variant not in variants:
            continue
        retry_index = variants.index(variant)
        retry_jobs = retry.get("jobs", {})
        resume_job = str(retry_jobs.get("resume_array", ""))
        release_job = str(retry_jobs.get("single_outer_release", ""))
        aggregate_job = str(retry_jobs.get("afterany_aggregate", ""))
        if not all((resume_job, release_job, aggregate_job)):
            raise RuntimeError(f"retry record omits a job identity: {retry_path}")
        retry_output = log_root / f"retry_{resume_job}_{retry_index}.out"
        retry_error = log_root / f"retry_{resume_job}_{retry_index}.err"
        paths.extend(
            (
                retry_output,
                retry_error,
                log_root / f"release_{release_job}.out",
                log_root / f"release_{release_job}.err",
                log_root / f"aggregate_{aggregate_job}.out",
                log_root / f"aggregate_{aggregate_job}.err",
            )
        )
        retry_outputs.append(retry_output)
        retry_errors.append(retry_error)
        release_errors.append(log_root / f"release_{release_job}.err")
        aggregate_errors.append(log_root / f"aggregate_{aggregate_job}.err")
        retry_count += 1
    if retry_count != 1:
        raise RuntimeError(
            "the sole prior retry record is not the authenticated exact-resume "
            f"attempt for {variant}"
        )

    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"required failure logs are missing: {missing}")
    signature = "severity atom logits contains non-finite values"
    if signature not in original_error.read_text(encoding="utf-8", errors="replace"):
        raise RuntimeError("original failure log lacks the expected forward-NaN signature")
    for retry_error in retry_errors:
        if signature not in retry_error.read_text(encoding="utf-8", errors="replace"):
            raise RuntimeError(f"retry failure changed: {retry_error}")
        if retry_error.read_bytes() != original_error.read_bytes():
            raise RuntimeError(
                f"retry and original forward-NaN traces are not identical: {retry_error}"
            )
    for retry_output in retry_outputs:
        expected = "resumed ORIGIN fold=9 from epoch=34"
        if expected not in retry_output.read_text(encoding="utf-8", errors="replace"):
            raise RuntimeError(f"retry did not authenticate epoch 34: {retry_output}")
    for release_error in release_errors:
        text = release_error.read_text(encoding="utf-8", errors="replace")
        if "origin_coarse_only/fold9/ABLATION_TRAIN_COMPLETE.json" not in text:
            raise RuntimeError(
                f"release did not fail closed at the training-marker gate: {release_error}"
            )
    for aggregate_error in aggregate_errors:
        text = aggregate_error.read_text(encoding="utf-8", errors="replace")
        if "outer release did not complete; aggregation is intentionally blocked" not in text:
            raise RuntimeError(
                f"aggregate did not fail closed at the outer marker: {aggregate_error}"
            )
    return sorted(set(paths))


protocol_path = Path(os.environ["ORIGIN_FRESH_PROTOCOL"]).resolve()
root = Path(os.environ["ORIGIN_FRESH_ROOT"]).resolve()
data_root = Path(os.environ["ORIGIN_FRESH_DATA"]).resolve()
worktree = Path(os.environ["ORIGIN_FRESH_WORKTREE"]).resolve()
log_root = Path(os.environ["ORIGIN_FRESH_LOG_ROOT"]).resolve()
variant = os.environ["ORIGIN_FRESH_VARIANT"]
expected_index = int(os.environ["ORIGIN_FRESH_EXPECTED_INDEX"])
expected_epoch = int(os.environ["ORIGIN_FRESH_EXPECTED_EPOCH"])

protocol = read_json(protocol_path)
selected = verify_protocol(
    protocol,
    expected_launch_commit=os.environ["ORIGIN_FRESH_COMMIT"],
    expected_root=root,
    expected_data_root=data_root,
)
if protocol_path != root / "LOCKED_PROTOCOL.json":
    raise RuntimeError("fresh restart must use the canonical locked protocol")
if Path(protocol["immutable_worktree"]).resolve() != worktree:
    raise RuntimeError("immutable worktree differs from the signed protocol")
if variant not in selected or selected.index(variant) != expected_index:
    raise RuntimeError(
        f"{variant} is not immutable task index {expected_index}: {selected}"
    )

require_absent_outer_artifacts(root)
one_time_records = [
    *root.glob("FRESH_RESTART_ONCE.json"),
    *root.glob("FRESH_RESTART_AUTHORIZATION_*.json"),
    *root.glob("FRESH_RESTART_SUBMISSION_*.json"),
]
archive_parent = root / "failed_attempts" / variant
if one_time_records or (archive_parent.exists() and any(archive_parent.iterdir())):
    raise RuntimeError(
        "a fresh restart was already authorized or archived; one-time recovery refuses"
    )

missing_variants: list[str] = []
for candidate in selected:
    candidate_fold = root / "workers" / candidate / f"fold{FOLD}"
    marker_path = candidate_fold / "ABLATION_TRAIN_COMPLETE.json"
    if not marker_path.is_file():
        missing_variants.append(candidate)
        continue
    marker = read_json(marker_path)
    verify_checksummed_payload(marker, schema=TRAIN_MARKER_SCHEMA)
    if marker.get("variant") != candidate:
        raise RuntimeError(f"completion marker identity changed for {candidate}")
    for key, expected in {
        "dataset": protocol["dataset"],
        "fold": FOLD,
        "evaluation_scope": "inner_validation_only_outer_locked",
        "outer_test_evaluated": False,
        "protocol_checksum_sha256": protocol["content_checksum_sha256"],
        "launch_commit": protocol["launch_commit"],
        "split_signature": EXPECTED_SPLIT_SIGNATURE,
    }.items():
        if marker.get(key) != expected:
            raise RuntimeError(f"completion marker {key} changed for {candidate}")
    expected_paths = {
        "checkpoint": candidate_fold / "best.pth",
        "result": candidate_fold / "result.json",
        "split_manifest": candidate_fold / "split_manifest.json",
        "history": candidate_fold / "history.csv",
        "validation_certificate": candidate_fold / "validation_certificates.json",
    }
    if marker.get("artifact_paths") != {
        name: str(path.resolve()) for name, path in expected_paths.items()
    }:
        raise RuntimeError(f"completion marker artifact paths changed for {candidate}")
    hashes = marker.get("artifact_sha256", {})
    if set(hashes) != set(expected_paths):
        raise RuntimeError(f"completion marker artifact set changed for {candidate}")
    for name, path in expected_paths.items():
        if not path.is_file() or file_sha256(path) != hashes[name]:
            raise RuntimeError(f"sealed artifact changed for {candidate}: {name}")
if missing_variants != [variant]:
    raise RuntimeError(
        f"fresh restart requires {variant} to be the sole incomplete variant; "
        f"observed {missing_variants}"
    )

fold_dir = root / "workers" / variant / f"fold{FOLD}"
for unexpected in (
    fold_dir / "result.json",
    fold_dir / "validation_certificates.json",
    fold_dir / "ABLATION_TRAIN_COMPLETE.json",
):
    if unexpected.exists():
        raise RuntimeError(f"failed worker contains a completion artifact: {unexpected}")
split_path = fold_dir / "split_manifest.json"
history_path = fold_dir / "history.csv"
if not split_path.is_file() or not history_path.is_file():
    raise FileNotFoundError("failed worker lacks its split manifest or history")
split = read_json(split_path)
validate_split_manifest(split)
if split.get("ablation_variant") != variant:
    raise RuntimeError("failed worker split variant mismatch")

implementation = origin_ablation_implementation_signature()
for checkpoint_name in ("last.pth", "best.pth"):
    checkpoint_path = fold_dir / checkpoint_name
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"failed worker lacks {checkpoint_path}")
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if state.get("schema") != "origin-checkpoint-v3":
        raise RuntimeError(f"unexpected checkpoint schema: {checkpoint_path}")
    if state.get("fold") != FOLD or state.get("split_signature") != EXPECTED_SPLIT_SIGNATURE:
        raise RuntimeError(f"checkpoint fold/split mismatch: {checkpoint_path}")
    if state.get("implementation_signature") != implementation:
        raise RuntimeError(f"checkpoint implementation mismatch: {checkpoint_path}")
    if int(state.get("epoch", -1)) != expected_epoch:
        raise RuntimeError(
            f"{checkpoint_name} is not the incident epoch {expected_epoch}: "
            f"{state.get('epoch')}"
        )
    if int(state.get("best_epoch", -1)) != expected_epoch:
        raise RuntimeError(f"{checkpoint_name} does not select epoch {expected_epoch}")
    config = state.get("config")
    if not isinstance(config, Mapping):
        raise RuntimeError(f"checkpoint configuration is missing: {checkpoint_path}")
    validate_variant_config(config, variant)
    if Path(str(config.get("run_dir"))).resolve() != (root / "workers" / variant).resolve():
        raise RuntimeError(f"checkpoint run directory changed: {checkpoint_path}")
    for section in ("model_state", "criterion_state", "optimizer_state"):
        bad = bad_tensors(state.get(section, {}), section)
        if bad:
            raise FloatingPointError(
                f"saved {checkpoint_name} contains non-finite {section} tensors: {bad[:8]}"
            )
    del state

logs = collect_log_paths(root, log_root, variant, expected_index)
print(
    "fresh_restart_gate_verified",
    variant,
    "task_index",
    expected_index,
    "epoch",
    expected_epoch,
    "logs",
    len(logs),
)
PY

# Refuse every recorded job and every same-name ablation job still visible in
# one coherent Slurm snapshot.  Querying purged historical IDs directly can
# return "invalid job id" even when the queue is healthy, so intersect the
# recorded IDs with the current user's live array-base IDs instead.
PRIOR_JOBS_RAW="$(
  ORIGIN_FRESH_ROOT="${EXPERIMENT_ROOT}" \
  PYTHONPATH="${SNAPSHOT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}" \
  python - <<'PY'
import os
from pathlib import Path
from typing import Mapping

from scripts.origin_fold9_ablation_common import read_json, verify_checksummed_payload

root = Path(os.environ["ORIGIN_FRESH_ROOT"])
jobs: set[str] = set()
submission_path = root / "SUBMISSION.json"
submission = read_json(submission_path)
verify_checksummed_payload(submission, schema="origin-fold9-ablation-submission-v1")
if isinstance(submission.get("jobs"), Mapping):
    jobs.update(str(value) for value in submission["jobs"].values() if value is not None)
for path in sorted(root.glob("RETRY_SUBMISSION_*.json")):
    record = read_json(path)
    verify_checksummed_payload(
        record, schema="origin-fold9-ablation-retry-submission-v1"
    )
    if isinstance(record.get("jobs"), Mapping):
        jobs.update(str(value) for value in record["jobs"].values() if value is not None)
for job in sorted(jobs):
    print(job)
PY
)"
mapfile -t PRIOR_JOBS <<< "${PRIOR_JOBS_RAW}"
CURRENT_USER="${USER:-$(id -un)}"
active_queue="$(squeue -h -u "${CURRENT_USER}" -o '%F|%j')" || {
  echo "Could not query the current Slurm queue for ${CURRENT_USER}." >&2
  exit 80
}
while IFS='|' read -r active_base active_name; do
  active_base="${active_base//[[:space:]]/}"
  active_name="${active_name//[[:space:]]/}"
  [[ -z "${active_base}${active_name}" ]] && continue
  for prior_job in "${PRIOR_JOBS[@]}"; do
    if [[ "${active_base}" == "${prior_job}" ]]; then
      echo "Recorded fold-9 ablation job ${prior_job} is still active (${active_name})." >&2
      exit 79
    fi
  done
  case "${active_name}" in
    of9_ab_tr|of9_ab_re|of9_ab_out|of9_ab_sum)
      echo "Unrecorded fold-9 ablation job ${active_base} (${active_name}) is active." >&2
      exit 81
      ;;
  esac
done <<< "${active_queue}"

if [[ "${DRY_RUN}" == true ]]; then
  echo "Fresh-restart dry run passed; no worker files were moved and no jobs were submitted."
  echo "  variant:          ${VARIANT}"
  echo "  task index:       ${EXPECTED_INDEX}"
  echo "  immutable commit: ${LAUNCH_COMMIT}"
  echo "  reason:           ${REASON}"
  exit 0
fi

RESTART_TAG="$(date -u +%Y%m%dT%H%M%SZ)"
ARCHIVE_ROOT="${EXPERIMENT_ROOT}/failed_attempts/${VARIANT}/${RESTART_TAG}"
AUTHORIZATION_RECORD="${EXPERIMENT_ROOT}/FRESH_RESTART_AUTHORIZATION_${RESTART_TAG}.json"
ONE_TIME_SENTINEL="${EXPERIMENT_ROOT}/FRESH_RESTART_ONCE.json"
SUBMISSION_RECORD="${EXPERIMENT_ROOT}/FRESH_RESTART_SUBMISSION_${RESTART_TAG}.json"

# Copy every authenticated operational record and relevant Slurm log, then
# atomically move the failed worker into the archive.  The archive is sealed by
# a checksummed per-file manifest before its final directory name appears.
ORIGIN_FRESH_PROTOCOL="${PROTOCOL}" \
ORIGIN_FRESH_COMMIT="${LAUNCH_COMMIT}" \
ORIGIN_FRESH_ROOT="${EXPERIMENT_ROOT}" \
ORIGIN_FRESH_DATA="${DATA_ROOT}" \
ORIGIN_FRESH_WORKTREE="${SNAPSHOT_ROOT}" \
ORIGIN_FRESH_LOG_ROOT="${LOG_ROOT}" \
ORIGIN_FRESH_VARIANT="${VARIANT}" \
ORIGIN_FRESH_INDEX="${EXPECTED_INDEX}" \
ORIGIN_FRESH_EPOCH="${EXPECTED_LAST_EPOCH}" \
ORIGIN_FRESH_REASON="${REASON}" \
ORIGIN_FRESH_TAG="${RESTART_TAG}" \
ORIGIN_FRESH_ARCHIVE="${ARCHIVE_ROOT}" \
ORIGIN_FRESH_AUTHORIZATION="${AUTHORIZATION_RECORD}" \
ORIGIN_FRESH_SENTINEL="${ONE_TIME_SENTINEL}" \
ORIGIN_FRESH_SUBMISSION="${SUBMISSION_RECORD}" \
ORIGIN_FRESH_LAUNCHER="${LAUNCHER_PATH}" \
ORIGIN_FRESH_LAUNCHER_SHA256="${LAUNCHER_SHA256}" \
PYTHONPATH="${SNAPSHOT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}" \
python - <<'PY'
import atexit
import os
import signal
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import torch

from scripts.origin_fold9_ablation_common import (
    canonical_sha256,
    file_sha256,
    read_json,
    verify_checksummed_payload,
    verify_protocol,
    write_json_atomic,
)


def load_record(path: Path, schema: str) -> Mapping[str, Any]:
    value = read_json(path)
    if not isinstance(value, Mapping):
        raise TypeError(f"operational record is not an object: {path}")
    verify_checksummed_payload(value, schema=schema)
    return value


def collect_evidence(
    root: Path, log_root: Path, variant: str, index: int
) -> tuple[list[Path], list[Path], list[dict[str, Any]]]:
    record_paths = [root / "LOCKED_PROTOCOL.json", root / "SUBMISSION.json"]
    submission = load_record(
        root / "SUBMISSION.json", "origin-fold9-ablation-submission-v1"
    )
    jobs = submission["jobs"]
    attempts: list[dict[str, Any]] = [
        {
            "kind": "original_fresh_training",
            "job": str(jobs["validation_training_array"]),
            "array_index": index,
        }
    ]
    logs = [
        log_root / f"train_{jobs['validation_training_array']}_{index}.out",
        log_root / f"train_{jobs['validation_training_array']}_{index}.err",
        log_root / f"release_{jobs['single_outer_release']}.out",
        log_root / f"release_{jobs['single_outer_release']}.err",
        log_root / f"aggregate_{jobs['afterany_aggregate']}.out",
        log_root / f"aggregate_{jobs['afterany_aggregate']}.err",
    ]
    for retry_path in sorted(root.glob("RETRY_SUBMISSION_*.json")):
        retry = load_record(
            retry_path, "origin-fold9-ablation-retry-submission-v1"
        )
        record_paths.append(retry_path)
        variants = list(retry.get("resumed_variants", ()))
        if variant not in variants:
            continue
        retry_index = variants.index(variant)
        retry_jobs = retry["jobs"]
        attempts.append(
            {
                "kind": "exact_checkpoint_resume",
                "job": str(retry_jobs["resume_array"]),
                "array_index": retry_index,
            }
        )
        logs.extend(
            (
                log_root / f"retry_{retry_jobs['resume_array']}_{retry_index}.out",
                log_root / f"retry_{retry_jobs['resume_array']}_{retry_index}.err",
                log_root / f"release_{retry_jobs['single_outer_release']}.out",
                log_root / f"release_{retry_jobs['single_outer_release']}.err",
                log_root / f"aggregate_{retry_jobs['afterany_aggregate']}.out",
                log_root / f"aggregate_{retry_jobs['afterany_aggregate']}.err",
            )
        )
    return sorted(set(logs)), sorted(set(record_paths)), attempts


root = Path(os.environ["ORIGIN_FRESH_ROOT"]).resolve()
protocol_path = Path(os.environ["ORIGIN_FRESH_PROTOCOL"]).resolve()
data_root = Path(os.environ["ORIGIN_FRESH_DATA"]).resolve()
worktree = Path(os.environ["ORIGIN_FRESH_WORKTREE"]).resolve()
log_root = Path(os.environ["ORIGIN_FRESH_LOG_ROOT"]).resolve()
variant = os.environ["ORIGIN_FRESH_VARIANT"]
index = int(os.environ["ORIGIN_FRESH_INDEX"])
expected_epoch = int(os.environ["ORIGIN_FRESH_EPOCH"])
reason = os.environ["ORIGIN_FRESH_REASON"]
tag = os.environ["ORIGIN_FRESH_TAG"]
archive = Path(os.environ["ORIGIN_FRESH_ARCHIVE"]).resolve()
authorization_path = Path(os.environ["ORIGIN_FRESH_AUTHORIZATION"]).resolve()
sentinel_path = Path(os.environ["ORIGIN_FRESH_SENTINEL"]).resolve()
submission_path = Path(os.environ["ORIGIN_FRESH_SUBMISSION"]).resolve()
worker = root / "workers" / variant
try:
    archive.relative_to(root)
except ValueError as exc:
    raise RuntimeError("fresh restart archive resolves outside the experiment root") from exc
if (
    authorization_path.parent != root
    or sentinel_path.parent != root
    or submission_path.parent != root
):
    raise RuntimeError("fresh restart records resolve outside the experiment root")

protocol = read_json(protocol_path)
selected = verify_protocol(
    protocol,
    expected_launch_commit=os.environ["ORIGIN_FRESH_COMMIT"],
    expected_root=root,
    expected_data_root=data_root,
)
if selected.index(variant) != index:
    raise RuntimeError("variant index changed between validation and archive")
for forbidden in (
    root / "release",
    root / "OUTER_RELEASE_COMPLETE.json",
    root / "FOLD9_ABLATION_RESULTS.json",
    root / "FOLD9_ABLATION_RESULTS.csv",
    root / "FOLD9_ABLATION_RESULTS.md",
):
    if forbidden.exists() or forbidden.is_symlink():
        raise RuntimeError(f"outer artifact appeared during restart gate: {forbidden}")
if list(root.glob(".release.staging.*")):
    raise RuntimeError("outer-release staging appeared during restart gate")
if (
    sentinel_path.exists()
    or sentinel_path.is_symlink()
    or list(root.glob("FRESH_RESTART_AUTHORIZATION_*.json"))
    or list(root.glob("FRESH_RESTART_SUBMISSION_*.json"))
):
    raise RuntimeError("fresh restart was already recorded")
if archive.exists():
    raise FileExistsError(f"fresh restart archive already exists: {archive}")

archive_parent = archive.parent
archive_parent.mkdir(parents=True, exist_ok=True)
if any(archive_parent.iterdir()):
    raise RuntimeError("fresh restart archive parent is not empty")
staging = archive_parent / f".{tag}.{os.getpid()}.staging"
staging.mkdir(parents=False, exist_ok=False)

# Keep the canonical worker and the top-level authorization records
# all-or-nothing for every ordinary exception or interrupt.  SIGKILL and a
# machine/power loss are inherently outside process-level recovery, but the
# staging/final archive remains self-describing for a manual audit in that
# case.
cleanup_state = {"committed": False}


def rollback_uncommitted_archive() -> None:
    if cleanup_state["committed"]:
        return
    for record_path in (authorization_path, submission_path, sentinel_path):
        if record_path.is_file() or record_path.is_symlink():
            record_path.unlink()
    for container in (archive, staging):
        if not container.exists():
            continue
        contained_worker = container / "worker"
        if contained_worker.exists():
            if worker.exists():
                raise RuntimeError(
                    "cannot roll back archive because both canonical and archived "
                    "workers exist"
                )
            os.replace(contained_worker, worker)
        shutil.rmtree(container)


def interrupt_archive(signum: int, _frame: object) -> None:
    raise KeyboardInterrupt(f"fresh-restart archive interrupted by signal {signum}")


atexit.register(rollback_uncommitted_archive)
signal.signal(signal.SIGINT, interrupt_archive)
signal.signal(signal.SIGTERM, interrupt_archive)
logs_dir = staging / "logs"
records_dir = staging / "records"
logs_dir.mkdir()
records_dir.mkdir()

logs, records, attempts = collect_evidence(root, log_root, variant, index)
if [attempt["kind"] for attempt in attempts] != [
    "original_fresh_training",
    "exact_checkpoint_resume",
]:
    raise RuntimeError(f"unexpected failed-attempt history: {attempts}")
launcher = Path(os.environ["ORIGIN_FRESH_LAUNCHER"]).resolve()
if file_sha256(launcher) != os.environ["ORIGIN_FRESH_LAUNCHER_SHA256"]:
    raise RuntimeError("administrative launcher changed during authorization")
records.append(launcher)
records = sorted(set(records))
for source in logs:
    if not source.is_file():
        raise FileNotFoundError(f"failure log disappeared during archive: {source}")
    shutil.copy2(source, logs_dir / source.name)
for source in records:
    if not source.is_file():
        raise FileNotFoundError(f"operational record disappeared during archive: {source}")
    shutil.copy2(source, records_dir / source.name)

archived_worker = staging / "worker"
if worker.is_symlink() or not worker.is_dir():
    raise FileNotFoundError(f"failed worker disappeared before archive: {worker}")

last = torch.load(
    worker / "fold9" / "last.pth", map_location="cpu", weights_only=False
)
best = torch.load(
    worker / "fold9" / "best.pth", map_location="cpu", weights_only=False
)
checkpoint_summary = {
    "last_epoch": int(last["epoch"]),
    "best_checkpoint_epoch": int(best["epoch"]),
    "selected_best_epoch": int(last["best_epoch"]),
    "last_checkpoint_sha256": file_sha256(worker / "fold9" / "last.pth"),
    "best_checkpoint_sha256": file_sha256(worker / "fold9" / "best.pth"),
    "best_validation": dict(best.get("metrics", {})),
}
if any(value is None for value in checkpoint_summary.values()):
    raise RuntimeError("archived checkpoint summary is incomplete")
if checkpoint_summary["last_epoch"] != expected_epoch:
    raise RuntimeError("archived last checkpoint epoch changed")
del last, best

files: dict[str, dict[str, Any]] = {}
for path in sorted(item for item in worker.rglob("*") if item.is_file()):
    relative = (Path("worker") / path.relative_to(worker)).as_posix()
    files[relative] = {
        "bytes": path.stat().st_size,
        "sha256": file_sha256(path),
    }
for base in (logs_dir, records_dir):
    for path in sorted(item for item in base.rglob("*") if item.is_file()):
        relative = path.relative_to(staging).as_posix()
        files[relative] = {
            "bytes": path.stat().st_size,
            "sha256": file_sha256(path),
        }
manifest: dict[str, Any] = {
    "schema": "origin-fold9-ablation-fresh-restart-archive-v1",
    "created_at_utc": datetime.now(timezone.utc).isoformat(),
    "variant": variant,
    "original_worker_path": str(worker),
    "final_archive_path": str(archive),
    "protocol_content_checksum_sha256": protocol["content_checksum_sha256"],
    "files": files,
    "metadata_excluded_from_files": [
        "ARCHIVE_MANIFEST.json",
        "FRESH_RESTART_AUTHORIZATION.json",
    ],
}
manifest["content_checksum_sha256"] = canonical_sha256(manifest)
manifest_path = staging / "ARCHIVE_MANIFEST.json"
write_json_atomic(manifest_path, manifest)

authorization: dict[str, Any] = {
    "schema": "origin-fold9-ablation-fresh-restart-authorization-v1",
    "created_at_utc": datetime.now(timezone.utc).isoformat(),
    "one_time_fresh_restart": True,
    "fresh_restart_attempt_number": 1,
    "fresh_restart_attempt_limit": 1,
    "authorization_scope": "post_hoc_operational_recovery_before_outer_release",
    "reason": reason,
    "variant": variant,
    "immutable_array_task_index": index,
    "failed_attempts": attempts,
    "failed_attempt_count": len(attempts),
    "checkpoint_summary": checkpoint_summary,
    "protocol_path": str(protocol_path),
    "protocol_file_sha256": file_sha256(protocol_path),
    "protocol_content_checksum_sha256": protocol["content_checksum_sha256"],
    "locked_protocol_modified": False,
    "scientific_configuration_changed": False,
    "outer_test_opened": False,
    "outer_release_artifacts_absent_when_authorized": True,
    "immutable_worktree": str(worktree),
    "launch_commit": os.environ["ORIGIN_FRESH_COMMIT"],
    "fresh_worker_script": str(
        worktree / "scripts" / "submit_origin_fold9_ablation_train_array.sh"
    ),
    "fresh_worker_script_sha256": file_sha256(
        worktree / "scripts" / "submit_origin_fold9_ablation_train_array.sh"
    ),
    "administrative_launcher": os.environ["ORIGIN_FRESH_LAUNCHER"],
    "administrative_launcher_sha256": os.environ[
        "ORIGIN_FRESH_LAUNCHER_SHA256"
    ],
    "archive_path": str(archive),
    "one_time_sentinel_path": str(sentinel_path),
    "archive_manifest_path": str(archive / "ARCHIVE_MANIFEST.json"),
    "archive_manifest_file_sha256": file_sha256(manifest_path),
    "archive_manifest_content_checksum_sha256": manifest[
        "content_checksum_sha256"
    ],
    "fresh_restart_submission_record_path": str(submission_path),
    "result_reporting_requirement": (
        "Distribute and cite this authorization plus the fresh-restart submission "
        "record alongside every generated FOLD9_ABLATION_RESULTS artifact."
    ),
}
authorization["content_checksum_sha256"] = canonical_sha256(authorization)
embedded_authorization_path = staging / "FRESH_RESTART_AUTHORIZATION.json"
write_json_atomic(embedded_authorization_path, authorization)

submission_created_at = datetime.now(timezone.utc).isoformat()
submission: dict[str, Any] = {
    "schema": "origin-fold9-ablation-fresh-restart-submission-v1",
    "created_at_utc": submission_created_at,
    "updated_at_utc": submission_created_at,
    "status": "authorized",
    "variant": variant,
    "immutable_array_task_index": index,
    "reason": reason,
    "launch_commit": os.environ["ORIGIN_FRESH_COMMIT"],
    "protocol_path": str(protocol_path),
    "protocol_file_sha256": file_sha256(protocol_path),
    "authorization_path": str(authorization_path),
    "authorization_file_sha256": file_sha256(embedded_authorization_path),
    "archive_path": str(archive),
    "administrative_launcher": os.environ["ORIGIN_FRESH_LAUNCHER"],
    "administrative_launcher_sha256": os.environ[
        "ORIGIN_FRESH_LAUNCHER_SHA256"
    ],
    "events": [
        {
            "at_utc": submission_created_at,
            "event": "failed attempt archived; submission authorized",
        }
    ],
    "jobs": {},
}
submission["content_checksum_sha256"] = canonical_sha256(submission)

# The atexit rollback covers failures before, between, or after these atomic
# same-filesystem renames, including a failure while writing either top-level
# authorization record.
os.replace(worker, archived_worker)
os.replace(staging, archive)
write_json_atomic(authorization_path, authorization)
write_json_atomic(submission_path, submission)
write_json_atomic(sentinel_path, authorization)
cleanup_state["committed"] = True
atexit.unregister(rollback_uncommitted_archive)
print("fresh_restart_archive_sealed", archive)
print("fresh_restart_one_time_sentinel_written", sentinel_path)
print("fresh_restart_authorization_written", authorization_path)
PY

update_submission_record() {
  local status="$1"
  local event="$2"
  ORIGIN_FRESH_SUBMISSION="${SUBMISSION_RECORD}" \
  ORIGIN_FRESH_AUTHORIZATION="${AUTHORIZATION_RECORD}" \
  ORIGIN_FRESH_PROTOCOL="${PROTOCOL}" \
  ORIGIN_FRESH_ARCHIVE="${ARCHIVE_ROOT}" \
  ORIGIN_FRESH_VARIANT="${VARIANT}" \
  ORIGIN_FRESH_INDEX="${EXPECTED_INDEX}" \
  ORIGIN_FRESH_REASON="${REASON}" \
  ORIGIN_FRESH_COMMIT="${LAUNCH_COMMIT}" \
  ORIGIN_FRESH_LAUNCHER="${LAUNCHER_PATH}" \
  ORIGIN_FRESH_LAUNCHER_SHA256="${LAUNCHER_SHA256}" \
  ORIGIN_FRESH_STATUS="${status}" \
  ORIGIN_FRESH_EVENT="${event}" \
  ORIGIN_FRESH_TRAIN_JOB="${FRESH_JOB:-}" \
  ORIGIN_FRESH_RELEASE_JOB="${RELEASE_JOB:-}" \
  ORIGIN_FRESH_AGGREGATE_JOB="${AGGREGATE_JOB:-}" \
  PYTHONPATH="${SNAPSHOT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}" \
  python - <<'PY'
import os
from datetime import datetime, timezone
from pathlib import Path

from scripts.origin_fold9_ablation_common import (
    canonical_sha256,
    file_sha256,
    read_json,
    verify_checksummed_payload,
    write_json_atomic,
)

path = Path(os.environ["ORIGIN_FRESH_SUBMISSION"])
now = datetime.now(timezone.utc).isoformat()
if path.exists():
    payload = read_json(path)
    verify_checksummed_payload(
        payload, schema="origin-fold9-ablation-fresh-restart-submission-v1"
    )
    payload = dict(payload)
    payload.pop("content_checksum_sha256", None)
else:
    payload = {
        "schema": "origin-fold9-ablation-fresh-restart-submission-v1",
        "created_at_utc": now,
        "variant": os.environ["ORIGIN_FRESH_VARIANT"],
        "immutable_array_task_index": int(os.environ["ORIGIN_FRESH_INDEX"]),
        "reason": os.environ["ORIGIN_FRESH_REASON"],
        "launch_commit": os.environ["ORIGIN_FRESH_COMMIT"],
        "protocol_path": os.environ["ORIGIN_FRESH_PROTOCOL"],
        "protocol_file_sha256": file_sha256(os.environ["ORIGIN_FRESH_PROTOCOL"]),
        "authorization_path": os.environ["ORIGIN_FRESH_AUTHORIZATION"],
        "authorization_file_sha256": file_sha256(
            os.environ["ORIGIN_FRESH_AUTHORIZATION"]
        ),
        "archive_path": os.environ["ORIGIN_FRESH_ARCHIVE"],
        "administrative_launcher": os.environ["ORIGIN_FRESH_LAUNCHER"],
        "administrative_launcher_sha256": os.environ[
            "ORIGIN_FRESH_LAUNCHER_SHA256"
        ],
        "events": [],
        "jobs": {},
    }
payload["status"] = os.environ["ORIGIN_FRESH_STATUS"]
payload["updated_at_utc"] = now
payload.setdefault("events", []).append(
    {"at_utc": now, "event": os.environ["ORIGIN_FRESH_EVENT"]}
)
jobs = payload.setdefault("jobs", {})
for key, environment in (
    ("fresh_validation_training_array", "ORIGIN_FRESH_TRAIN_JOB"),
    ("single_outer_release", "ORIGIN_FRESH_RELEASE_JOB"),
    ("afterany_aggregate", "ORIGIN_FRESH_AGGREGATE_JOB"),
):
    value = os.environ.get(environment)
    if value:
        jobs[key] = value
payload["content_checksum_sha256"] = canonical_sha256(payload)
write_json_atomic(path, payload)
PY
}

FRESH_JOB=""
RELEASE_JOB=""
AGGREGATE_JOB=""
SUBMISSION_TRANSACTION_ARMED=false

abort_partial_submission() {
  local exit_code=$?
  trap - EXIT INT TERM
  if [[ "${SUBMISSION_TRANSACTION_ARMED}" == true ]]; then
    (( exit_code == 0 )) && exit_code=1
    set +e
    for submitted_job in "${AGGREGATE_JOB}" "${RELEASE_JOB}" "${FRESH_JOB}"; do
      if [[ "${submitted_job}" =~ ^[0-9]+$ ]]; then
        scancel "${submitted_job}"
      fi
    done
    update_submission_record \
      "submission_aborted_cancellation_requested" \
      "launcher exited before durable chain release; cancellation requested for every recorded job"
    echo "Fresh-restart submission aborted; cancellation was requested for every recorded job." >&2
  fi
  exit "${exit_code}"
}

trap abort_partial_submission EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

EXPORTS="ALL,ORIGIN_REPO_ROOT=${SNAPSHOT_ROOT},ORIGIN_LAUNCH_COMMIT=${LAUNCH_COMMIT},ORIGIN_FOLD9_ABLATION_ROOT=${EXPERIMENT_ROOT},ORIGIN_DATA_ROOT=${DATA_ROOT},ORIGIN_FOLD9_ABLATION_PROTOCOL=${PROTOCOL}"

SUBMISSION_TRANSACTION_ARMED=true
update_submission_record "fresh_worker_submission_pending" "submitting immutable fresh worker"
fresh_job_raw="$(sbatch --parsable \
  --hold \
  --array="${EXPECTED_INDEX}-${EXPECTED_INDEX}%1" \
  --export="${EXPORTS}" \
  "${SNAPSHOT_ROOT}/scripts/submit_origin_fold9_ablation_train_array.sh")"
FRESH_JOB="${fresh_job_raw%%;*}"
[[ "${FRESH_JOB}" =~ ^[0-9]+$ ]] || {
  echo "sbatch returned an invalid fresh-worker job id: ${fresh_job_raw}." >&2
  exit 82
}
update_submission_record "fresh_worker_submitted_held" "immutable fresh worker submitted on hold"

update_submission_record "outer_release_submission_pending" "submitting fail-closed outer release"
release_job_raw="$(sbatch --parsable --dependency="afterany:${FRESH_JOB}" \
  --export="${EXPORTS}" \
  "${SNAPSHOT_ROOT}/scripts/submit_origin_fold9_ablation_outer_release.sh")"
RELEASE_JOB="${release_job_raw%%;*}"
[[ "${RELEASE_JOB}" =~ ^[0-9]+$ ]] || {
  echo "sbatch returned an invalid outer-release job id: ${release_job_raw}." >&2
  exit 83
}
update_submission_record "outer_release_submitted" "fail-closed outer release submitted"

update_submission_record "aggregate_submission_pending" "submitting after-any aggregate audit"
aggregate_job_raw="$(sbatch --parsable --dependency="afterany:${RELEASE_JOB}" \
  --export="${EXPORTS}" \
  "${SNAPSHOT_ROOT}/scripts/submit_origin_fold9_ablation_aggregate.sh")"
AGGREGATE_JOB="${aggregate_job_raw%%;*}"
[[ "${AGGREGATE_JOB}" =~ ^[0-9]+$ ]] || {
  echo "sbatch returned an invalid aggregate job id: ${aggregate_job_raw}." >&2
  exit 84
}
update_submission_record "chain_submitted_held" "complete job chain recorded; fresh worker remains held"
# The worker and release job take these same locks.  Release both only after
# all three identifiers are durably recorded and immediately before making the
# held task runnable.
update_submission_record "chain_release_pending" "recorded held chain is ready for lock release"
flock -u 9
exec 9>&-
flock -u 7
exec 7>&-
scontrol release "${FRESH_JOB}"
update_submission_record "chain_released" "recorded job chain released for execution"
SUBMISSION_TRANSACTION_ARMED=false
trap - EXIT INT TERM

echo "One-time fresh restart submitted without changing the signed scientific protocol."
echo "  variant:             ${VARIANT}"
echo "  immutable task:      ${EXPECTED_INDEX}"
echo "  reason:              ${REASON}"
echo "  archived failures:   ${ARCHIVE_ROOT}"
echo "  authorization:       ${AUTHORIZATION_RECORD}"
echo "  one-time sentinel:    ${ONE_TIME_SENTINEL}"
echo "  fresh worker:        ${FRESH_JOB}"
echo "  outer release:       ${RELEASE_JOB}"
echo "  aggregate/audit:     ${AGGREGATE_JOB}"
echo "  submission record:   ${SUBMISSION_RECORD}"
echo "Monitor: squeue -u \"${CURRENT_USER}\""
