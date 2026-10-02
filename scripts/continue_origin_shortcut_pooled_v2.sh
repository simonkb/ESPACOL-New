#!/bin/bash
# Submit exact-protocol continuation for canceled/console-failed pooled tasks.
# Usage: bash scripts/continue_origin_shortcut_pooled_v2.sh SUITE_ROOT

set -euo pipefail
[[ $# -eq 1 ]] || { echo "Usage: $0 SUITE_ROOT" >&2; exit 64; }

SOURCE_ROOT="$(git rev-parse --show-toplevel)"
cd "${SOURCE_ROOT}"
RECOVERY_COMMIT="$(git rev-parse HEAD)"
[[ -z "$(git status --porcelain --untracked-files=no)" ]] || {
  echo "Commit tracked recovery changes before continuation." >&2; exit 2;
}
ENV_PYTHON="${HOME}/.conda/envs/${ORIGIN_CONDA_ENV:-G}/bin/python"
[[ -x "${ENV_PYTHON}" ]] || { echo "Missing ${ENV_PYTHON}." >&2; exit 3; }
command -v sbatch >/dev/null || { echo "sbatch unavailable." >&2; exit 4; }
command -v squeue >/dev/null || { echo "squeue unavailable." >&2; exit 5; }
command -v scontrol >/dev/null || { echo "scontrol unavailable." >&2; exit 5; }

SUITE_ROOT="$("${ENV_PYTHON}" -c 'import pathlib,sys; print(pathlib.Path(sys.argv[1]).resolve())' "$1")"
PROTOCOL="${SUITE_ROOT}/PROTOCOL_V2.json"
SUBMISSION="${SUITE_ROOT}/SUBMISSION.json"
[[ -f "${PROTOCOL}" && -f "${SUBMISSION}" ]] || {
  echo "Suite lacks PROTOCOL_V2.json/SUBMISSION.json: ${SUITE_ROOT}." >&2; exit 6;
}
[[ ! -e "${SUITE_ROOT}/COMPARATOR_SHORTCUT_RESULTS_V2.json" ]] || {
  echo "Comparator aggregate already exists." >&2; exit 7;
}
[[ ! -e "${SUITE_ROOT}/POOLED_CONTINUATION_SUBMISSION.json" ]] || {
  echo "Pooled continuation was already submitted." >&2; exit 8;
}

mapfile -t META < <("${ENV_PYTHON}" - "${PROTOCOL}" "${SUBMISSION}" "${SUITE_ROOT}" <<'PY'
import hashlib, json, pathlib, sys
protocol_path, submission_path, expected_root = map(pathlib.Path, sys.argv[1:])
protocol = json.loads(protocol_path.read_text())
recorded = protocol.pop("content_checksum_sha256")
canonical = lambda value: hashlib.sha256(json.dumps(
    value, sort_keys=True, separators=(",", ":"), allow_nan=False
).encode()).hexdigest()
if recorded != canonical(protocol):
    raise SystemExit("protocol checksum mismatch")
if pathlib.Path(protocol.get("suite_root", "")).resolve() != expected_root.resolve():
    raise SystemExit("suite root mismatch")
submission = json.loads(submission_path.read_text())
submitted_checksum = submission.pop("content_checksum_sha256")
if submitted_checksum != canonical(submission):
    raise SystemExit("submission checksum mismatch")
if submission.get("launch_commit") != protocol.get("launch_commit"):
    raise SystemExit("submission/protocol commit mismatch")
for value in (
    protocol["immutable_worktree"], protocol["launch_commit"], protocol["data_root"],
    protocol["origin_reference_root"], submission["jobs"]["comparator_worker_array"],
    submission["jobs"]["expanded_aggregate"],
):
    print(value)
PY
)
[[ ${#META[@]} -eq 6 ]] || { echo "Could not resolve protocol metadata." >&2; exit 9; }
SNAPSHOT_ROOT="${META[0]}"
LAUNCH_COMMIT="${META[1]}"
DATA_ROOT="${META[2]}"
REFERENCE_ROOT="${META[3]}"
ORIGINAL_WORKER_JOB="${META[4]}"
ORIGINAL_AGGREGATE_JOB="${META[5]}"

[[ "$(git -C "${SNAPSHOT_ROOT}" rev-parse HEAD)" == "${LAUNCH_COMMIT}" ]] || exit 10
[[ -z "$(git -C "${SNAPSHOT_ROOT}" status --porcelain --untracked-files=no)" ]] || exit 11
# Import protocol semantics only from the original immutable launch snapshot.
# Python gives the current directory precedence over PYTHONPATH, so an explicit
# chdir is part of the fail-closed provenance contract.
cd "${SNAPSHOT_ROOT}"
"${ENV_PYTHON}" - "${PROTOCOL}" "${SUITE_ROOT}" <<'PY'
import json, pathlib, sys
from scripts.origin_shortcut_comparator_common import PROTOCOL_CORE_SHA256, task_at
protocol = json.loads(pathlib.Path(sys.argv[1]).read_text())
if protocol.get("protocol_core_sha256") != PROTOCOL_CORE_SHA256:
    raise SystemExit("protocol core mismatch against immutable launch snapshot")
suite = pathlib.Path(sys.argv[2]).resolve()
for task_id in range(27, 42):
    task = task_at(task_id)
    fold = (
        suite / "workers" / task.model_variant / task.arm / task.family
        / f"seed{task.training_seed}" / "fold0"
    )
    if fold.exists() and any(fold.iterdir()):
        raise SystemExit(
            f"original canceled task {task_id} has a non-empty fold directory: {fold}"
        )
for task_id in range(21, 42):
    recovery = suite / "recovery" / f"task{task_id}.json"
    if recovery.exists():
        raise SystemExit(f"task {task_id} already has a recovery record: {recovery}")
PY

# Queueing must not expose workers to a mutable recovery checkout.  Seal the
# recovery implementation in its own detached worktree at the exact commit
# that is recorded in continuation provenance.
RECOVERY_SHORT="${RECOVERY_COMMIT:0:12}"
RECOVERY_ROOT="${ORIGIN_SHORTCUT_RECOVERY_WORKTREE:-${SOURCE_ROOT}-origin-shortcut-recovery-${RECOVERY_SHORT}}"
if [[ -e "${RECOVERY_ROOT}" ]]; then
  [[ -d "${RECOVERY_ROOT}" ]] || { echo "Recovery path is not a directory: ${RECOVERY_ROOT}" >&2; exit 12; }
else
  git -C "${SOURCE_ROOT}" worktree add --detach "${RECOVERY_ROOT}" "${RECOVERY_COMMIT}"
fi
[[ "$(git -C "${RECOVERY_ROOT}" rev-parse HEAD)" == "${RECOVERY_COMMIT}" ]] || {
  echo "Detached recovery worktree commit mismatch: ${RECOVERY_ROOT}" >&2; exit 12;
}
[[ -z "$(git -C "${RECOVERY_ROOT}" status --porcelain --untracked-files=no)" ]] || {
  echo "Detached recovery worktree contains tracked changes: ${RECOVERY_ROOT}" >&2; exit 12;
}
if [[ -n "$(squeue -h -j "${ORIGINAL_AGGREGATE_JOB}" -o '%i' 2>/dev/null)" ]]; then
  echo "Original blocked aggregate ${ORIGINAL_AGGREGATE_JOB} is still queued." >&2
  echo "Run: scancel ${ORIGINAL_AGGREGATE_JOB}" >&2
  echo "Then rerun this continuation launcher." >&2
  exit 13
fi
[[ -f "${REFERENCE_ROOT}/APTOS_SHORTCUT_PILOT_RESULTS.json" ]] || {
  echo "Sealed ORIGIN reference aggregate is missing." >&2; exit 14;
}

# Tasks 27..41 were canceled while only represented by the parent array's
# compressed queue expression.  This cluster creates no individual Slurm
# object for those canceled-before-start elements, so depending on nonexistent
# JOB_TASK ids can remain pending forever.  Prove that state explicitly before
# submitting fresh replacements; do not infer it merely from missing files.
scontrol show job "${ORIGINAL_WORKER_JOB}" >/dev/null 2>&1 || {
  echo "Original comparator parent array is not reachable: ${ORIGINAL_WORKER_JOB}." >&2
  exit 15
}
ORIGINAL_ARRAY_SQUEUE_ROWS="$(squeue -h -r -j "${ORIGINAL_WORKER_JOB}" -o '%F|%K|%T')"
ORIGIN_RECOVERY_SQUEUE_ROWS="${ORIGINAL_ARRAY_SQUEUE_ROWS}" \
"${ENV_PYTHON}" - "${ORIGINAL_WORKER_JOB}" <<'PY'
import os, sys
job = sys.argv[1]
for line in os.environ["ORIGIN_RECOVERY_SQUEUE_ROWS"].splitlines():
    fields = line.strip().split("|", 2)
    if len(fields) != 3 or fields[0] != job or not fields[1].isdigit():
        raise SystemExit(f"unexpected expanded squeue row: {line!r}")
    task_id = int(fields[1])
    if 27 <= task_id <= 41:
        raise SystemExit(
            f"original task {task_id} still exists in expanded squeue: {line}"
        )
PY
for task_id in $(seq 27 41); do
  if scontrol show job "${ORIGINAL_WORKER_JOB}_${task_id}" >/dev/null 2>&1; then
    echo "Original task ${ORIGINAL_WORKER_JOB}_${task_id} still has a Slurm object." >&2
    exit 16
  fi
done

EXPORTS="ALL,ORIGIN_REPO_ROOT=${SNAPSHOT_ROOT},ORIGIN_RECOVERY_REPO_ROOT=${RECOVERY_ROOT},ORIGIN_RECOVERY_COMMIT=${RECOVERY_COMMIT},ORIGIN_LAUNCH_COMMIT=${LAUNCH_COMMIT},ORIGIN_ORIGINAL_WORKER_JOB=${ORIGINAL_WORKER_JOB},ORIGIN_SHORTCUT_COMPARATOR_ROOT=${SUITE_ROOT},ORIGIN_SHORTCUT_COMPARATOR_PROTOCOL=${PROTOCOL},ORIGIN_SHORTCUT_REFERENCE_ROOT=${REFERENCE_ROOT},ORIGIN_DATA_ROOT=${DATA_ROOT}"
RUNNING_POOLED_DEPENDENCY="afterany:${ORIGINAL_WORKER_JOB}_23:${ORIGINAL_WORKER_JOB}_24:${ORIGINAL_WORKER_JOB}_25:${ORIGINAL_WORKER_JOB}_26"
CONTINUATION_JOB="$(sbatch --parsable --array="21-41%${ORIGIN_SHORTCUT_POOLED_CONTINUATION_CONCURRENCY:-8}" \
  --dependency="${RUNNING_POOLED_DEPENDENCY}" --export="${EXPORTS}" \
  "${RECOVERY_ROOT}/scripts/submit_origin_shortcut_pooled_continuation_v2.sh" | cut -d';' -f1)"
AGGREGATE_JOB="$(sbatch --parsable \
  --dependency="afterok:${CONTINUATION_JOB},afterany:${ORIGINAL_WORKER_JOB}" \
  --export="ALL,ORIGIN_REPO_ROOT=${SNAPSHOT_ROOT},ORIGIN_LAUNCH_COMMIT=${LAUNCH_COMMIT},ORIGIN_SHORTCUT_COMPARATOR_ROOT=${SUITE_ROOT},ORIGIN_SHORTCUT_REFERENCE_ROOT=${REFERENCE_ROOT}" \
  "${SNAPSHOT_ROOT}/scripts/submit_origin_shortcut_comparator_aggregate_v2.sh" | cut -d';' -f1)"

ORIGIN_RECOVERY_OUTPUT="${SUITE_ROOT}/POOLED_CONTINUATION_SUBMISSION.json" \
ORIGIN_RECOVERY_PROTOCOL="${PROTOCOL}" \
ORIGIN_RECOVERY_COMMIT="${RECOVERY_COMMIT}" \
ORIGIN_RECOVERY_ORIGINAL_COMMIT="${LAUNCH_COMMIT}" \
ORIGIN_RECOVERY_WORKTREE="${RECOVERY_ROOT}" \
ORIGIN_RECOVERY_ORIGINAL_WORKER="${ORIGINAL_WORKER_JOB}" \
ORIGIN_RECOVERY_SQUEUE_ROWS="${ORIGINAL_ARRAY_SQUEUE_ROWS}" \
ORIGIN_RECOVERY_WORKER="${CONTINUATION_JOB}" \
ORIGIN_RECOVERY_AGGREGATE="${AGGREGATE_JOB}" \
"${ENV_PYTHON}" - <<'PY'
import hashlib, json, os
from datetime import datetime, timezone
from pathlib import Path
from scripts.origin_shortcut_comparator_common import canonical_sha256
p = {
    "schema": "origin-shortcut-pooled-continuation-submission-v1",
    "created_at_utc": datetime.now(timezone.utc).isoformat(),
    "protocol": os.environ["ORIGIN_RECOVERY_PROTOCOL"],
    "protocol_modified": False,
    "original_launch_commit": os.environ["ORIGIN_RECOVERY_ORIGINAL_COMMIT"],
    "recovery_implementation_commit": os.environ["ORIGIN_RECOVERY_COMMIT"],
    "immutable_recovery_worktree": os.environ["ORIGIN_RECOVERY_WORKTREE"],
    "task_ids": list(range(21, 42)),
    "expected_task_dispositions": {
        **{str(task_id): "reused_complete_original_training" for task_id in range(21, 27)},
        **{
            str(task_id): "fresh_training_after_original_task_cancellation"
            for task_id in range(27, 42)
        },
    },
    "original_array_state_proof": {
        "parent_array_reachable": True,
        "expanded_squeue_checked": True,
        "expanded_squeue_sha256": hashlib.sha256(
            os.environ["ORIGIN_RECOVERY_SQUEUE_ROWS"].encode()
        ).hexdigest(),
        "individual_slurm_objects_absent": list(range(27, 42)),
        "protocol_mapped_fold_directories_verified_empty": list(range(27, 42)),
        "preexisting_recovery_records_absent": list(range(21, 42)),
        "dependency_waits_for_original_task_ids": [23, 24, 25, 26],
    },
    "jobs": {
        "original_comparator_array": os.environ["ORIGIN_RECOVERY_ORIGINAL_WORKER"],
        "pooled_continuation_array": os.environ["ORIGIN_RECOVERY_WORKER"],
        "replacement_aggregate": os.environ["ORIGIN_RECOVERY_AGGREGATE"],
    },
}
p["content_checksum_sha256"] = canonical_sha256(p)
with Path(os.environ["ORIGIN_RECOVERY_OUTPUT"]).open("x", encoding="utf-8") as stream:
    json.dump(p, stream, indent=2, sort_keys=True, allow_nan=False)
    stream.write("\n")
PY

echo "Pooled comparator continuation submitted without changing PROTOCOL_V2.json."
echo "  active pooled dependency:   ${RUNNING_POOLED_DEPENDENCY}"
echo "  continuation array:         ${CONTINUATION_JOB} (tasks 21..41)"
echo "  replacement aggregate:      ${AGGREGATE_JOB} (also waits afterany ${ORIGINAL_WORKER_JOB})"
echo "  immutable recovery tree:    ${RECOVERY_ROOT}"
echo "  suite root:                 ${SUITE_ROOT}"
