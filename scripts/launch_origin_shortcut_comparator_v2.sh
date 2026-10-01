#!/bin/bash
# Freeze and submit the 84-worker matched shortcut-comparator follow-up.
set -euo pipefail

REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "${REPO_ROOT}"
ENV_PYTHON="${HOME}/.conda/envs/${ORIGIN_CONDA_ENV:-G}/bin/python"
[[ -x "${ENV_PYTHON}" ]] || {
  echo "Conda environment Python is missing or not executable: ${ENV_PYTHON}" >&2
  exit 1
}
"${ENV_PYTHON}" -c 'import sys; assert sys.version_info >= (3, 11)' || {
  echo "Comparator launcher requires Python >=3.11 in ${ENV_PYTHON}." >&2
  exit 1
}
BRANCH="$(git branch --show-current)"
[[ "${BRANCH}" == "origin-acceptance-revision" ]] || {
  echo "Expected origin-acceptance-revision, got ${BRANCH}." >&2; exit 2;
}
[[ -z "$(git status --porcelain --untracked-files=no)" ]] || {
  echo "Commit tracked changes before freezing comparator protocol v2." >&2; exit 3;
}

LAUNCH_COMMIT="$(git rev-parse HEAD)"
SHORT_COMMIT="${LAUNCH_COMMIT:0:12}"
TAG="$(date -u +%Y%m%dT%H%M%SZ)"
DATA_ROOT="${ORIGIN_DATA_ROOT:-${REPO_ROOT}/Datasets/aptos2019-blindness-detection}"
REFERENCE_ROOT="${ORIGIN_SHORTCUT_REFERENCE_ROOT:?Set ORIGIN_SHORTCUT_REFERENCE_ROOT to the sealed 21-job ORIGIN-v1 root}"
SUITE_ROOT="${ORIGIN_SHORTCUT_COMPARATOR_ROOT:-${REPO_ROOT}/runs/origin_shortcut_comparators_v2_${TAG}}"
SNAPSHOT_ROOT="${ORIGIN_SHORTCUT_COMPARATOR_WORKTREE:-${REPO_ROOT}-origin-shortcut-comparators-${SHORT_COMMIT}}"
PROTOCOL="${SUITE_ROOT}/PROTOCOL_V2.json"

[[ -f "${REFERENCE_ROOT}/PROTOCOL.json" && -f "${REFERENCE_ROOT}/SUBMISSION.json" ]] || {
  echo "Reference root lacks sealed v1 PROTOCOL.json/SUBMISSION.json." >&2; exit 4;
}
[[ -d "${DATA_ROOT}" ]] || { echo "APTOS data root missing: ${DATA_ROOT}" >&2; exit 5; }
[[ ! -e "${SUITE_ROOT}" ]] || { echo "Fresh suite root already exists." >&2; exit 6; }
if [[ -e "${SNAPSHOT_ROOT}" ]]; then
  [[ "$(git -C "${SNAPSHOT_ROOT}" rev-parse HEAD)" == "${LAUNCH_COMMIT}" ]] || exit 7
  [[ -z "$(git -C "${SNAPSHOT_ROOT}" status --porcelain --untracked-files=no)" ]] || exit 8
else
  git worktree add --detach "${SNAPSHOT_ROOT}" "${LAUNCH_COMMIT}"
fi

mkdir -p "${SUITE_ROOT}" "${REPO_ROOT}/origin_shortcut_logs"
ORIGIN_PROTOCOL_OUTPUT="${PROTOCOL}" \
ORIGIN_PROTOCOL_COMMIT="${LAUNCH_COMMIT}" \
ORIGIN_PROTOCOL_WORKTREE="${SNAPSHOT_ROOT}" \
ORIGIN_PROTOCOL_ROOT="${SUITE_ROOT}" \
ORIGIN_PROTOCOL_DATA="${DATA_ROOT}" \
ORIGIN_PROTOCOL_REFERENCE="${REFERENCE_ROOT}" "${ENV_PYTHON}" - <<'PY'
import hashlib, json, os, sys
from datetime import datetime, timezone
from pathlib import Path
snapshot = Path(os.environ["ORIGIN_PROTOCOL_WORKTREE"]).resolve()
sys.path.insert(0, str(snapshot))
from scripts.origin_shortcut_comparator_common import (
    PROTOCOL_CORE_SHA256, canonical_sha256, protocol_core,
)
reference = Path(os.environ["ORIGIN_PROTOCOL_REFERENCE"]).resolve()
payload = {
    **protocol_core(),
    "created_at_utc": datetime.now(timezone.utc).isoformat(),
    "launch_commit": os.environ["ORIGIN_PROTOCOL_COMMIT"],
    "immutable_worktree": str(snapshot),
    "suite_root": str(Path(os.environ["ORIGIN_PROTOCOL_ROOT"]).resolve()),
    "data_root": str(Path(os.environ["ORIGIN_PROTOCOL_DATA"]).resolve()),
    "origin_reference_root": str(reference),
    "origin_reference_protocol_sha256": hashlib.sha256(
        (reference / "PROTOCOL.json").read_bytes()
    ).hexdigest(),
    "origin_reference_submission_sha256": hashlib.sha256(
        (reference / "SUBMISSION.json").read_bytes()
    ).hexdigest(),
    "protocol_core_sha256": PROTOCOL_CORE_SHA256,
}
payload["content_checksum_sha256"] = canonical_sha256(payload)
with Path(os.environ["ORIGIN_PROTOCOL_OUTPUT"]).open("x", encoding="utf-8") as stream:
    json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
    stream.write("\n")
PY

EXPORTS="ALL,ORIGIN_REPO_ROOT=${SNAPSHOT_ROOT},ORIGIN_LAUNCH_COMMIT=${LAUNCH_COMMIT},ORIGIN_SHORTCUT_COMPARATOR_ROOT=${SUITE_ROOT},ORIGIN_SHORTCUT_COMPARATOR_PROTOCOL=${PROTOCOL},ORIGIN_SHORTCUT_REFERENCE_ROOT=${REFERENCE_ROOT},ORIGIN_DATA_ROOT=${DATA_ROOT}"
REFERENCE_DEPENDENCY=""
if [[ -n "${ORIGIN_SHORTCUT_REFERENCE_AGGREGATE_JOB:-}" ]]; then
  [[ "${ORIGIN_SHORTCUT_REFERENCE_AGGREGATE_JOB}" =~ ^[0-9]+$ ]] || {
    echo "Reference aggregate job must be a numeric Slurm job id." >&2; exit 9;
  }
  REFERENCE_DEPENDENCY=":${ORIGIN_SHORTCUT_REFERENCE_AGGREGATE_JOB}"
elif [[ ! -f "${REFERENCE_ROOT}/APTOS_SHORTCUT_PILOT_RESULTS.json" ]]; then
  echo "Reference results are incomplete and no reference aggregate job was supplied." >&2
  exit 9
fi
PREFLIGHT_JOB="$(sbatch --parsable --export="${EXPORTS}" \
  "${SNAPSHOT_ROOT}/scripts/submit_origin_shortcut_comparator_preflight_v2.sh" | cut -d';' -f1)"
WORKER_JOB="$(sbatch --parsable --array="0-83%${ORIGIN_SHORTCUT_COMPARATOR_CONCURRENCY:-8}" \
  --dependency="afterok:${PREFLIGHT_JOB}" \
  --export="${EXPORTS}" \
  "${SNAPSHOT_ROOT}/scripts/submit_origin_shortcut_comparator_v2.sh" | cut -d';' -f1)"
DEPENDENCY="afterok:${WORKER_JOB}${REFERENCE_DEPENDENCY}"
AGGREGATE_JOB="$(sbatch --parsable --dependency="${DEPENDENCY}" \
  --export="${EXPORTS}" \
  "${SNAPSHOT_ROOT}/scripts/submit_origin_shortcut_comparator_aggregate_v2.sh" | cut -d';' -f1)"

ORIGIN_SUBMISSION_OUTPUT="${SUITE_ROOT}/SUBMISSION.json" \
ORIGIN_SUBMISSION_COMMIT="${LAUNCH_COMMIT}" \
ORIGIN_SUBMISSION_PREFLIGHT="${PREFLIGHT_JOB}" \
ORIGIN_SUBMISSION_WORKER="${WORKER_JOB}" \
ORIGIN_SUBMISSION_AGGREGATE="${AGGREGATE_JOB}" "${ENV_PYTHON}" - <<'PY'
import hashlib, json, os
from datetime import datetime, timezone
from pathlib import Path
p = {
    "schema": "origin-ordinal-shortcut-comparator-submission-v2",
    "created_at_utc": datetime.now(timezone.utc).isoformat(),
    "launch_commit": os.environ["ORIGIN_SUBMISSION_COMMIT"],
    "additional_training_worker_count": 84,
    "jobs": {
        "cpu_preflight": os.environ["ORIGIN_SUBMISSION_PREFLIGHT"],
        "comparator_worker_array": os.environ["ORIGIN_SUBMISSION_WORKER"],
        "expanded_aggregate": os.environ["ORIGIN_SUBMISSION_AGGREGATE"],
    },
}
p["content_checksum_sha256"] = hashlib.sha256(
    json.dumps(p, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
).hexdigest()
with Path(os.environ["ORIGIN_SUBMISSION_OUTPUT"]).open("x", encoding="utf-8") as stream:
    json.dump(p, stream, indent=2, sort_keys=True, allow_nan=False)
    stream.write("\n")
PY

echo "Shortcut comparator follow-up v2 submitted."
echo "  commit:             ${LAUNCH_COMMIT}"
echo "  CPU preflight:      ${PREFLIGHT_JOB}"
echo "  additional workers: ${WORKER_JOB} (84 exact tasks)"
echo "  aggregate:          ${AGGREGATE_JOB}"
echo "  root:               ${SUITE_ROOT}"
