#!/bin/bash
# Resume one or more failed workers from an already locked fold-9 ablation.
#
# Usage:
#   bash scripts/retry_origin_fold9_ablation_variants.sh \
#     /absolute/path/to/experiment variant [variant ...]
#
# This is deliberately a resume-only operation.  It neither edits nor
# replaces LOCKED_PROTOCOL.json, and it refuses to run once any outer-release
# artifact exists.  All variants lacking completion markers must be named so
# that the suite-level release cannot silently proceed with an incomplete set.

set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 EXPERIMENT_ROOT VARIANT [VARIANT ...]" >&2
  exit 64
fi

EXPERIMENT_ROOT="$1"
shift
REQUESTED_VARIANTS=("$@")

set +u
source /etc/profile.d/lmod.sh || exit 1
module load miniconda/3 || exit 1
source activate "${ORIGIN_CONDA_ENV:-G}" || exit 1
set -u
command -v sbatch >/dev/null || { echo "sbatch is unavailable." >&2; exit 65; }
command -v squeue >/dev/null || { echo "squeue is unavailable." >&2; exit 66; }

EXPERIMENT_ROOT="$(python -c 'import pathlib,sys; print(pathlib.Path(sys.argv[1]).resolve())' "${EXPERIMENT_ROOT}")"
PROTOCOL="${EXPERIMENT_ROOT}/LOCKED_PROTOCOL.json"
[[ -f "${PROTOCOL}" ]] || { echo "Missing locked protocol: ${PROTOCOL}." >&2; exit 67; }
mkdir -p "${EXPERIMENT_ROOT}/locks"
exec 8>"${EXPERIMENT_ROOT}/locks/retry_submission.lock"
flock -n 8 || {
  echo "Another retry launcher owns ${EXPERIMENT_ROOT}." >&2
  exit 68
}

# Bootstrap only the two immutable identities with the standard library.  The
# complete checksum and semantic verification below is imported from that
# exact launch snapshot, not from the caller's possibly newer checkout.
mapfile -t PROTOCOL_IDENTITY < <(python - "${PROTOCOL}" <<'PY'
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
)
[[ ${#PROTOCOL_IDENTITY[@]} -eq 3 ]] || { echo "Could not read protocol identities." >&2; exit 69; }
SNAPSHOT_ROOT="$(python -c 'import pathlib,sys; print(pathlib.Path(sys.argv[1]).resolve())' "${PROTOCOL_IDENTITY[0]}")"
LAUNCH_COMMIT="${PROTOCOL_IDENTITY[1]}"
DATA_ROOT="$(python -c 'import pathlib,sys; print(pathlib.Path(sys.argv[1]).resolve())' "${PROTOCOL_IDENTITY[2]}")"

[[ -d "${SNAPSHOT_ROOT}/.git" || -f "${SNAPSHOT_ROOT}/.git" ]] || {
  echo "Immutable worktree is missing: ${SNAPSHOT_ROOT}." >&2; exit 70;
}
[[ "$(git -C "${SNAPSHOT_ROOT}" rev-parse HEAD)" == "${LAUNCH_COMMIT}" ]] || {
  echo "Immutable worktree commit differs from LOCKED_PROTOCOL.json." >&2; exit 71;
}
[[ -z "$(git -C "${SNAPSHOT_ROOT}" status --porcelain --untracked-files=no)" ]] || {
  echo "Immutable worktree contains tracked changes." >&2; exit 72;
}
for required in \
  scripts/origin_fold9_ablation_common.py \
  scripts/submit_origin_fold9_ablation_resume_array.sh \
  scripts/submit_origin_fold9_ablation_outer_release.sh \
  scripts/submit_origin_fold9_ablation_aggregate.sh; do
  [[ -f "${SNAPSHOT_ROOT}/${required}" ]] || {
    echo "Launch snapshot lacks retry component ${required}." >&2; exit 73;
  }
done

RETRY_VARIANTS="$(IFS=:; echo "${REQUESTED_VARIANTS[*]}")"

# Fail closed before submitting anything.  This verifies the signed protocol,
# the exact fold/config in every resume checkpoint, the complete set of failed
# variants, and that no outer-test release has begun.
cd "${SNAPSHOT_ROOT}"
ORIGIN_RETRY_PROTOCOL="${PROTOCOL}" \
ORIGIN_RETRY_COMMIT="${LAUNCH_COMMIT}" \
ORIGIN_RETRY_ROOT="${EXPERIMENT_ROOT}" \
ORIGIN_RETRY_DATA="${DATA_ROOT}" \
ORIGIN_RETRY_WORKTREE="${SNAPSHOT_ROOT}" \
ORIGIN_RETRY_VARIANTS="${RETRY_VARIANTS}" \
PYTHONPATH="${SNAPSHOT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}" \
python - <<'PY'
import os
from pathlib import Path
from typing import Mapping

import torch

from scripts.origin_fold9_ablation_common import (
    EXPECTED_SPLIT_SIGNATURE,
    FOLD,
    TRAIN_MARKER_SCHEMA,
    read_json,
    validate_split_manifest,
    validate_variant_config,
    verify_checksummed_payload,
    verify_protocol,
)
from training.origin_ablation_trainer import origin_ablation_implementation_signature

protocol_path = Path(os.environ["ORIGIN_RETRY_PROTOCOL"]).resolve()
root = Path(os.environ["ORIGIN_RETRY_ROOT"]).resolve()
worktree = Path(os.environ["ORIGIN_RETRY_WORKTREE"]).resolve()
protocol = read_json(protocol_path)
selected = verify_protocol(
    protocol,
    expected_launch_commit=os.environ["ORIGIN_RETRY_COMMIT"],
    expected_root=root,
    expected_data_root=os.environ["ORIGIN_RETRY_DATA"],
)
if Path(protocol["immutable_worktree"]).resolve() != worktree:
    raise RuntimeError("immutable worktree differs from the signed protocol")
if protocol_path != root / "LOCKED_PROTOCOL.json":
    raise RuntimeError("retry must use the experiment's canonical protocol path")

requested = tuple(filter(None, os.environ["ORIGIN_RETRY_VARIANTS"].split(":")))
if not requested or len(requested) != len(set(requested)):
    raise RuntimeError("retry variants must be a non-empty duplicate-free list")
unknown = sorted(set(requested) - set(selected))
if unknown:
    raise RuntimeError(f"retry variants are outside the locked protocol: {unknown}")

for forbidden in (
    root / "release",
    root / "OUTER_RELEASE_COMPLETE.json",
    root / "FOLD9_ABLATION_RESULTS.json",
    root / "FOLD9_ABLATION_RESULTS.csv",
    root / "FOLD9_ABLATION_RESULTS.md",
):
    if forbidden.exists():
        raise RuntimeError(f"outer release/aggregation already exists: {forbidden}")
staging = sorted(root.glob(".release.staging.*"))
if staging:
    raise RuntimeError(f"outer-release staging exists and requires audit: {staging}")
for retry_record in sorted(root.glob("RETRY_SUBMISSION_*.json")):
    verify_checksummed_payload(
        read_json(retry_record),
        schema="origin-fold9-ablation-retry-submission-v1",
    )

missing = []
for variant in selected:
    fold_dir = root / "workers" / variant / f"fold{FOLD}"
    marker_path = fold_dir / "ABLATION_TRAIN_COMPLETE.json"
    if marker_path.is_file():
        marker = read_json(marker_path)
        verify_checksummed_payload(marker, schema=TRAIN_MARKER_SCHEMA)
        if marker.get("variant") != variant:
            raise RuntimeError(f"completion marker identity changed for {variant}")
        continue
    missing.append(variant)

if set(missing) != set(requested):
    raise RuntimeError(
        "the retry list must equal all variants without completion markers; "
        f"missing={missing}, requested={list(requested)}"
    )

for variant in requested:
    fold_dir = root / "workers" / variant / f"fold{FOLD}"
    marker_path = fold_dir / "ABLATION_TRAIN_COMPLETE.json"
    if marker_path.exists():
        raise RuntimeError(f"completed variant cannot be resumed: {variant}")
    last_path = fold_dir / "last.pth"
    split_path = fold_dir / "split_manifest.json"
    if not last_path.is_file():
        raise RuntimeError(f"resume checkpoint is missing for {variant}: {last_path}")
    if not split_path.is_file():
        raise RuntimeError(f"split manifest is missing for {variant}: {split_path}")
    split = read_json(split_path)
    validate_split_manifest(split)
    if split.get("ablation_variant") != variant:
        raise RuntimeError(f"split variant mismatch for {variant}")
    state = torch.load(last_path, map_location="cpu", weights_only=False)
    if state.get("schema") != "origin-checkpoint-v3":
        raise RuntimeError(f"unexpected resume checkpoint schema for {variant}")
    if state.get("fold") != FOLD or state.get("split_signature") != EXPECTED_SPLIT_SIGNATURE:
        raise RuntimeError(f"resume checkpoint fold/split mismatch for {variant}")
    active_implementation = origin_ablation_implementation_signature()
    if state.get("implementation_signature") != active_implementation:
        raise RuntimeError(
            f"resume checkpoint implementation differs from the immutable launch snapshot for {variant}"
        )
    config = state.get("config")
    if not isinstance(config, Mapping):
        raise RuntimeError(f"resume checkpoint configuration is missing for {variant}")
    validate_variant_config(config, variant)
    expected_run_dir = root / "workers" / variant
    if Path(str(config.get("run_dir"))).resolve() != expected_run_dir.resolve():
        raise RuntimeError(f"resume checkpoint run directory changed for {variant}")

print("resume_gate_verified", requested)
PY

# Refuse to race an original release/aggregate job that has not yet left the
# queue.  SUBMISSION.json is operational metadata only; the scientific
# protocol remains untouched.
if [[ -f "${EXPERIMENT_ROOT}/SUBMISSION.json" ]]; then
  mapfile -t ORIGINAL_POST_JOBS < <(python - "${EXPERIMENT_ROOT}/SUBMISSION.json" <<'PY'
import json
import pathlib
import sys
with pathlib.Path(sys.argv[1]).open(encoding="utf-8") as stream:
    jobs = json.load(stream).get("jobs", {})
for key in ("single_outer_release", "afterany_aggregate"):
    value = jobs.get(key)
    if value is not None:
        print(str(value))
PY
  )
  for job in "${ORIGINAL_POST_JOBS[@]}"; do
    if [[ -n "$(squeue -h -j "${job}" -o '%i' 2>/dev/null)" ]]; then
      echo "Original post-training job ${job} is still queued/running; wait for it to fail closed or cancel it before retrying." >&2
      exit 74
    fi
  done
fi

# A prior retry record is allowed only after every job it submitted has left
# the queue.  Together with retry_submission.lock this prevents duplicate
# resume arrays from being launched in the gap before a worker acquires its
# per-variant lock.
for record in "${EXPERIMENT_ROOT}"/RETRY_SUBMISSION_*.json; do
  [[ -e "${record}" ]] || continue
  mapfile -t PRIOR_RETRY_JOBS < <(python - "${record}" <<'PY'
import json
import pathlib
import sys
with pathlib.Path(sys.argv[1]).open(encoding="utf-8") as stream:
    jobs = json.load(stream).get("jobs", {})
for key in ("resume_array", "single_outer_release", "afterany_aggregate"):
    value = jobs.get(key)
    if value is not None:
        print(str(value))
PY
  )
  for job in "${PRIOR_RETRY_JOBS[@]}"; do
    if [[ -n "$(squeue -h -j "${job}" -o '%i' 2>/dev/null)" ]]; then
      echo "Prior retry job ${job} from ${record} is still queued/running." >&2
      exit 75
    fi
  done
done

# A non-blocking lock probe prevents racing a worker that is still running.
for variant in "${REQUESTED_VARIANTS[@]}"; do
  exec {lock_fd}>"${EXPERIMENT_ROOT}/locks/${variant}.lock"
  flock -n "${lock_fd}" || {
    echo "Variant ${variant} is still owned by another worker." >&2
    exit 76
  }
  flock -u "${lock_fd}"
  exec {lock_fd}>&-
done

EXPORTS="ALL,ORIGIN_REPO_ROOT=${SNAPSHOT_ROOT},ORIGIN_LAUNCH_COMMIT=${LAUNCH_COMMIT},ORIGIN_FOLD9_ABLATION_ROOT=${EXPERIMENT_ROOT},ORIGIN_DATA_ROOT=${DATA_ROOT},ORIGIN_FOLD9_ABLATION_PROTOCOL=${PROTOCOL},ORIGIN_FOLD9_RETRY_VARIANTS=${RETRY_VARIANTS}"
TASK_COUNT=${#REQUESTED_VARIANTS[@]}
RESUME_JOB="$(sbatch --parsable \
  --array="0-$((TASK_COUNT - 1))%${ORIGIN_FOLD9_ABLATION_RETRY_CONCURRENCY:-3}" \
  --export="${EXPORTS}" \
  "${SNAPSHOT_ROOT}/scripts/submit_origin_fold9_ablation_resume_array.sh" | cut -d';' -f1)"
RELEASE_JOB="$(sbatch --parsable --dependency="afterany:${RESUME_JOB}" \
  --export="${EXPORTS}" \
  "${SNAPSHOT_ROOT}/scripts/submit_origin_fold9_ablation_outer_release.sh" | cut -d';' -f1)"
AGGREGATE_JOB="$(sbatch --parsable --dependency="afterany:${RELEASE_JOB}" \
  --export="${EXPORTS}" \
  "${SNAPSHOT_ROOT}/scripts/submit_origin_fold9_ablation_aggregate.sh" | cut -d';' -f1)"

RETRY_TAG="$(date -u +%Y%m%dT%H%M%SZ)"
RETRY_RECORD="${EXPERIMENT_ROOT}/RETRY_SUBMISSION_${RETRY_TAG}.json"
ORIGIN_RETRY_RECORD="${RETRY_RECORD}" \
ORIGIN_RETRY_PROTOCOL="${PROTOCOL}" \
ORIGIN_RETRY_VARIANTS="${RETRY_VARIANTS}" \
ORIGIN_RETRY_RESUME_JOB="${RESUME_JOB}" \
ORIGIN_RETRY_RELEASE_JOB="${RELEASE_JOB}" \
ORIGIN_RETRY_AGGREGATE_JOB="${AGGREGATE_JOB}" \
ORIGIN_RETRY_COMMIT="${LAUNCH_COMMIT}" \
PYTHONPATH="${SNAPSHOT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}" \
python - <<'PY'
import os
from datetime import datetime, timezone
from scripts.origin_fold9_ablation_common import (
    canonical_sha256,
    file_sha256,
    write_json_atomic,
)
payload = {
    "schema": "origin-fold9-ablation-retry-submission-v1",
    "created_at_utc": datetime.now(timezone.utc).isoformat(),
    "launch_commit": os.environ["ORIGIN_RETRY_COMMIT"],
    "protocol_path": os.environ["ORIGIN_RETRY_PROTOCOL"],
    "protocol_file_sha256": file_sha256(os.environ["ORIGIN_RETRY_PROTOCOL"]),
    "resumed_variants": os.environ["ORIGIN_RETRY_VARIANTS"].split(":"),
    "jobs": {
        "resume_array": os.environ["ORIGIN_RETRY_RESUME_JOB"],
        "single_outer_release": os.environ["ORIGIN_RETRY_RELEASE_JOB"],
        "afterany_aggregate": os.environ["ORIGIN_RETRY_AGGREGATE_JOB"],
    },
}
payload["content_checksum_sha256"] = canonical_sha256(payload)
write_json_atomic(os.environ["ORIGIN_RETRY_RECORD"], payload)
PY

echo "Locked fold-9 ablation retry submitted without changing the protocol."
echo "  variants:          ${REQUESTED_VARIANTS[*]}"
echo "  resume array:      ${RESUME_JOB}"
echo "  outer release:     ${RELEASE_JOB}"
echo "  aggregate/audit:   ${AGGREGATE_JOB}"
echo "  experiment root:   ${EXPERIMENT_ROOT}"
echo "  retry record:      ${RETRY_RECORD}"
echo "Monitor: squeue -u \"${USER}\""
