#!/bin/bash
# Submit exact-protocol continuation for canceled/console-failed pooled tasks.
# Usage: bash scripts/continue_origin_shortcut_pooled_v2.sh SUITE_ROOT

set -euo pipefail
[[ $# -eq 1 ]] || { echo "Usage: $0 SUITE_ROOT" >&2; exit 64; }

RECOVERY_ROOT="$(git rev-parse --show-toplevel)"
cd "${RECOVERY_ROOT}"
RECOVERY_COMMIT="$(git rev-parse HEAD)"
[[ -z "$(git status --porcelain --untracked-files=no)" ]] || {
  echo "Commit tracked recovery changes before continuation." >&2; exit 2;
}
ENV_PYTHON="${HOME}/.conda/envs/${ORIGIN_CONDA_ENV:-G}/bin/python"
[[ -x "${ENV_PYTHON}" ]] || { echo "Missing ${ENV_PYTHON}." >&2; exit 3; }
command -v sbatch >/dev/null || { echo "sbatch unavailable." >&2; exit 4; }
command -v squeue >/dev/null || { echo "squeue unavailable." >&2; exit 5; }

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
from scripts.origin_shortcut_comparator_common import PROTOCOL_CORE_SHA256, canonical_sha256
protocol_path, submission_path, expected_root = map(pathlib.Path, sys.argv[1:])
protocol = json.loads(protocol_path.read_text())
recorded = protocol.pop("content_checksum_sha256")
if recorded != canonical_sha256(protocol):
    raise SystemExit("protocol checksum mismatch")
if protocol.get("protocol_core_sha256") != PROTOCOL_CORE_SHA256:
    raise SystemExit("protocol core mismatch")
if pathlib.Path(protocol.get("suite_root", "")).resolve() != expected_root.resolve():
    raise SystemExit("suite root mismatch")
submission = json.loads(submission_path.read_text())
submitted_checksum = submission.pop("content_checksum_sha256")
if submitted_checksum != canonical_sha256(submission):
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
if [[ -n "$(squeue -h -j "${ORIGINAL_AGGREGATE_JOB}" -o '%i' 2>/dev/null)" ]]; then
  echo "Original blocked aggregate ${ORIGINAL_AGGREGATE_JOB} is still queued." >&2
  echo "Run: scancel ${ORIGINAL_AGGREGATE_JOB}" >&2
  echo "Then rerun this continuation launcher." >&2
  exit 12
fi
[[ -f "${REFERENCE_ROOT}/APTOS_SHORTCUT_PILOT_RESULTS.json" ]] || {
  echo "Sealed ORIGIN reference aggregate is missing." >&2; exit 13;
}

EXPORTS="ALL,ORIGIN_REPO_ROOT=${SNAPSHOT_ROOT},ORIGIN_RECOVERY_REPO_ROOT=${RECOVERY_ROOT},ORIGIN_RECOVERY_COMMIT=${RECOVERY_COMMIT},ORIGIN_LAUNCH_COMMIT=${LAUNCH_COMMIT},ORIGIN_SHORTCUT_COMPARATOR_ROOT=${SUITE_ROOT},ORIGIN_SHORTCUT_COMPARATOR_PROTOCOL=${PROTOCOL},ORIGIN_SHORTCUT_REFERENCE_ROOT=${REFERENCE_ROOT},ORIGIN_DATA_ROOT=${DATA_ROOT}"
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
ORIGIN_RECOVERY_WORKER="${CONTINUATION_JOB}" \
ORIGIN_RECOVERY_AGGREGATE="${AGGREGATE_JOB}" \
"${ENV_PYTHON}" - <<'PY'
import json, os
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
    "task_ids": list(range(21, 42)),
    "jobs": {
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
echo "  suite root:                 ${SUITE_ROOT}"
