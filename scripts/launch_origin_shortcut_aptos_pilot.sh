#!/bin/bash
# Freeze and submit the complete APTOS controlled ordinal-shortcut pilot.

set -euo pipefail

REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "${REPO_ROOT}"
BRANCH="$(git branch --show-current)"
LAUNCH_COMMIT="$(git rev-parse HEAD)"
SHORT_COMMIT="$(git rev-parse --short=12 HEAD)"
[[ "${BRANCH}" == "origin-acceptance-revision" ]] || {
  echo "Refusing shortcut launch from '${BRANCH}'; expected origin-acceptance-revision." >&2
  exit 2
}
[[ -z "$(git status --porcelain --untracked-files=no)" ]] || {
  echo "Tracked changes are forbidden for the controlled benchmark launch." >&2
  git status --short --untracked-files=no >&2
  exit 3
}
command -v sbatch >/dev/null || { echo "sbatch is unavailable." >&2; exit 4; }

set +u
source /etc/profile.d/lmod.sh || exit 1
module load miniconda/3 || exit 1
source activate "${ORIGIN_CONDA_ENV:-G}" || exit 1
set -u

TAG="${ORIGIN_SHORTCUT_TAG:-$(date -u +%Y%m%dT%H%M%SZ)}"
PILOT_ROOT="${ORIGIN_SHORTCUT_ROOT:-${REPO_ROOT}/runs/origin_shortcut_aptos_${TAG}}"
DATA_ROOT="${ORIGIN_APTOS_ROOT:-${REPO_ROOT}/Datasets/aptos2019-blindness-detection}"
SNAPSHOT_ROOT="${ORIGIN_SHORTCUT_WORKTREE:-${REPO_ROOT}-origin-shortcut-${SHORT_COMMIT}}"
[[ -d "${DATA_ROOT}" ]] || { echo "Missing APTOS root: ${DATA_ROOT}." >&2; exit 5; }
[[ ! -e "${PILOT_ROOT}" ]] || {
  echo "Fresh-only protocol refuses existing pilot root: ${PILOT_ROOT}." >&2; exit 6;
}

if [[ -e "${SNAPSHOT_ROOT}" ]]; then
  [[ "$(git -C "${SNAPSHOT_ROOT}" rev-parse HEAD)" == "${LAUNCH_COMMIT}" ]] || {
    echo "Existing shortcut worktree has the wrong commit." >&2; exit 7;
  }
  [[ -z "$(git -C "${SNAPSHOT_ROOT}" status --porcelain --untracked-files=no)" ]] || {
    echo "Existing shortcut worktree contains tracked changes." >&2; exit 8;
  }
else
  git worktree add --detach "${SNAPSHOT_ROOT}" "${LAUNCH_COMMIT}"
fi

python -m pytest -q "${SNAPSHOT_ROOT}/tests/test_origin_shortcut_benchmark.py"
ORIGIN_PREFLIGHT_DATA="${DATA_ROOT}" python - <<'PY'
import os
from Datasets.origin_data import load_aptos_items
items = load_aptos_items(os.environ["ORIGIN_PREFLIGHT_DATA"])
if len(items) != 3662:
    raise RuntimeError(f"APTOS identity changed: {len(items)} images")
print("APTOS preflight", len(items))
PY

mkdir -p "${PILOT_ROOT}/workers" "${PILOT_ROOT}/locks" \
  "${REPO_ROOT}/origin_shortcut_logs"
PROTOCOL="${PILOT_ROOT}/LOCKED_PROTOCOL.json"
ORIGIN_PROTOCOL_OUTPUT="${PROTOCOL}" \
ORIGIN_PROTOCOL_ROOT="${PILOT_ROOT}" \
ORIGIN_PROTOCOL_DATA="${DATA_ROOT}" \
ORIGIN_PROTOCOL_REPO="${SNAPSHOT_ROOT}" \
ORIGIN_PROTOCOL_COMMIT="${LAUNCH_COMMIT}" python - <<'PY'
import hashlib, json, os
from datetime import datetime, timezone
from pathlib import Path
p = {
    "schema": "origin-ordinal-shortcut-pilot-protocol-v1",
    "created_at_utc": datetime.now(timezone.utc).isoformat(),
    "development_status": "frozen_before_result_inspection",
    "launch_commit": os.environ["ORIGIN_PROTOCOL_COMMIT"],
    "immutable_worktree": str(Path(os.environ["ORIGIN_PROTOCOL_REPO"]).resolve()),
    "pilot_root": str(Path(os.environ["ORIGIN_PROTOCOL_ROOT"]).resolve()),
    "data_root": str(Path(os.environ["ORIGIN_PROTOCOL_DATA"]).resolve()),
    "dataset": "aptos",
    "aptos_images": 3662,
    "fold": 0,
    "split_seed": 42,
    "shortcut_seed": 271828,
    "training_seeds": [1701, 2603, 3907],
    "families": ["localized", "border", "diffuse"],
    "arms": ["shortcut", "cue_only", "clean"],
    "conditions": [
        "aligned", "neutral", "inverted", "missing", "boundary_swapped",
        "conflicting", "clean", "cue_only"
    ],
    "unseen_test_domain": {"position": "unseen", "appearance": "unseen"},
    "factorial_marker_states": 16,
    "localization_permutations": 999,
    "effect_bootstrap_replicates": 2000,
}
p["content_checksum_sha256"] = hashlib.sha256(
    json.dumps(p, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
).hexdigest()
path = Path(os.environ["ORIGIN_PROTOCOL_OUTPUT"])
with path.open("x", encoding="utf-8") as stream:
    json.dump(p, stream, indent=2, sort_keys=True, allow_nan=False)
    stream.write("\n")
PY

EXPORTS="ALL,ORIGIN_REPO_ROOT=${SNAPSHOT_ROOT},ORIGIN_LAUNCH_COMMIT=${LAUNCH_COMMIT},ORIGIN_SHORTCUT_ROOT=${PILOT_ROOT},ORIGIN_DATA_ROOT=${DATA_ROOT},ORIGIN_SHORTCUT_PROTOCOL=${PROTOCOL}"
WORKER_JOB="$(sbatch --parsable \
  --array="0-20%${ORIGIN_SHORTCUT_CONCURRENCY:-5}" \
  --export="${EXPORTS}" \
  "${SNAPSHOT_ROOT}/scripts/submit_origin_shortcut_aptos_pilot.sh" | cut -d';' -f1)"
AGGREGATE_JOB="$(sbatch --parsable --dependency="afterok:${WORKER_JOB}" \
  --export="${EXPORTS}" \
  "${SNAPSHOT_ROOT}/scripts/submit_origin_shortcut_aptos_aggregate.sh" | cut -d';' -f1)"

ORIGIN_SUBMISSION_OUTPUT="${PILOT_ROOT}/SUBMISSION.json" \
ORIGIN_SUBMISSION_COMMIT="${LAUNCH_COMMIT}" \
ORIGIN_SUBMISSION_WORKER="${WORKER_JOB}" \
ORIGIN_SUBMISSION_AGGREGATE="${AGGREGATE_JOB}" python - <<'PY'
import hashlib, json, os
from datetime import datetime, timezone
from pathlib import Path
p = {
    "schema": "origin-ordinal-shortcut-pilot-submission-v1",
    "created_at_utc": datetime.now(timezone.utc).isoformat(),
    "launch_commit": os.environ["ORIGIN_SUBMISSION_COMMIT"],
    "jobs": {
        "worker_array": os.environ["ORIGIN_SUBMISSION_WORKER"],
        "aggregate": os.environ["ORIGIN_SUBMISSION_AGGREGATE"],
    },
}
p["content_checksum_sha256"] = hashlib.sha256(
    json.dumps(p, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
).hexdigest()
with Path(os.environ["ORIGIN_SUBMISSION_OUTPUT"]).open("x", encoding="utf-8") as stream:
    json.dump(p, stream, indent=2, sort_keys=True, allow_nan=False)
    stream.write("\n")
PY

echo "APTOS controlled-shortcut pilot submitted."
echo "  commit:   ${LAUNCH_COMMIT}"
echo "  workers:  ${WORKER_JOB}"
echo "  aggregate:${AGGREGATE_JOB}"
echo "  root:     ${PILOT_ROOT}"
echo "Monitor: squeue -u \"${USER}\""
