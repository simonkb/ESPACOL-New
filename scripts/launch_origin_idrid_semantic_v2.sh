#!/usr/bin/env bash
# Freeze and submit the corrective IDRiD v2 audit from an immutable worktree.

set -euo pipefail

EXPECTED_BRANCH="origin-acceptance-revision"
SOURCE_REPO="$(git rev-parse --show-toplevel)"
cd "$SOURCE_REPO"
[[ "$(git branch --show-current)" == "$EXPECTED_BRANCH" ]] || {
  echo "Expected branch $EXPECTED_BRANCH." >&2; exit 2;
}
[[ -z "$(git status --porcelain --untracked-files=no)" ]] || {
  echo "Commit tracked changes before freezing the IDRiD v2 audit." >&2; exit 3;
}
for command_name in git sbatch scancel python sha256sum; do
  command -v "$command_name" >/dev/null || {
    echo "$command_name is unavailable." >&2; exit 4;
  }
done

LAUNCH_COMMIT="$(git rev-parse HEAD)"
SHORT_COMMIT="${LAUNCH_COMMIT:0:12}"
SNAPSHOT_ROOT="${ORIGIN_IDRID_V2_WORKTREE:-${SOURCE_REPO}-idrid-v2-${SHORT_COMMIT}}"
CV_ROOT="${ORIGIN_V3_CV_ROOT:-${SOURCE_REPO}/runs/origin_v3_full_cv_20260922T084547Z}"
ACCEPTANCE_ROOT="${ORIGIN_ACCEPTANCE_ROOT:-${SOURCE_REPO}/runs/origin_acceptance}"
IDRID_ROOT="${ORIGIN_IDRID_ROOT:-${SOURCE_REPO}/Datasets/IDRiD}"
OUTPUT_ROOT="${ORIGIN_IDRID_V2_OUTPUT_ROOT:-${ACCEPTANCE_ROOT}/idrid_semantics_v2_bidirectional}"
LOG_ROOT="${ORIGIN_IDRID_V2_LOG_ROOT:-${SOURCE_REPO}/origin_idrid_v2_logs}"
SUBMISSION_RECORD="${OUTPUT_ROOT}.SUBMISSION.json"
RESERVATION_DIR="${OUTPUT_ROOT}.launch-reservation"

[[ -d "$IDRID_ROOT" ]] || { echo "Missing IDRiD root: $IDRID_ROOT" >&2; exit 5; }
for fold in $(seq 0 9); do
  checkpoint="$CV_ROOT/dr/workers/fold${fold}/fold${fold}/best.pth"
  [[ -f "$checkpoint" ]] || { echo "Missing checkpoint: $checkpoint" >&2; exit 6; }
done
[[ ! -e "$OUTPUT_ROOT" ]] || {
  echo "Corrective output already exists: $OUTPUT_ROOT" >&2; exit 7;
}
[[ ! -e "$SUBMISSION_RECORD" ]] || {
  echo "Submission record already exists: $SUBMISSION_RECORD" >&2; exit 8;
}
mkdir -p "$(dirname "$OUTPUT_ROOT")" "$LOG_ROOT"
mkdir "$RESERVATION_DIR" 2>/dev/null || {
  echo "Another v2 launch already reserved: $RESERVATION_DIR" >&2; exit 9;
}

JOB_ID=""
LAUNCH_COMMITTED=0
cleanup_launch() {
  status=$?
  if [[ "$LAUNCH_COMMITTED" != "1" ]]; then
    if [[ -n "$JOB_ID" ]]; then
      scancel "$JOB_ID" >/dev/null 2>&1 || true
    fi
    rmdir "$RESERVATION_DIR" >/dev/null 2>&1 || true
  fi
  exit "$status"
}
trap cleanup_launch EXIT

if [[ -e "$SNAPSHOT_ROOT" ]]; then
  [[ "$(git -C "$SNAPSHOT_ROOT" rev-parse HEAD)" == "$LAUNCH_COMMIT" ]] || {
    echo "Existing IDRiD v2 worktree has the wrong commit." >&2; exit 10;
  }
  [[ -z "$(git -C "$SNAPSHOT_ROOT" branch --show-current)" ]] || {
    echo "Existing IDRiD v2 worktree is not detached." >&2; exit 11;
  }
  [[ -z "$(git -C "$SNAPSHOT_ROOT" status --porcelain --untracked-files=no)" ]] || {
    echo "Existing IDRiD v2 worktree contains tracked changes." >&2; exit 12;
  }
else
  git worktree add --detach "$SNAPSHOT_ROOT" "$LAUNCH_COMMIT"
fi

set +u
source /etc/profile.d/lmod.sh || exit 1
module load miniconda/3 || exit 1
source activate "${ORIGIN_CONDA_ENV:-G}" || exit 1
set -u
PYTHONPATH="$SNAPSHOT_ROOT${PYTHONPATH:+:$PYTHONPATH}" \
  python -m pytest -q \
  "$SNAPSHOT_ROOT/tests/test_origin_idrid_semantics.py" \
  "$SNAPSHOT_ROOT/tests/test_origin_idrid_semantic_statistics.py" \
  "$SNAPSHOT_ROOT/tests/test_origin_idrid_semantics_v2.py"

PROTOCOL="$SNAPSHOT_ROOT/scripts/protocols/origin_idrid_semantic_statistics_protocol_v2.json"
PROTOCOL_SHA256="$(sha256sum "$PROTOCOL" | awk '{print $1}')"
SUBMIT_SHA256="$(sha256sum "$SNAPSHOT_ROOT/scripts/submit_origin_idrid_semantic_v2.sh" | awk '{print $1}')"
EXPORTS="ALL,ORIGIN_REPO_ROOT=${SNAPSHOT_ROOT},ORIGIN_LAUNCH_COMMIT=${LAUNCH_COMMIT},ORIGIN_V3_CV_ROOT=${CV_ROOT},ORIGIN_ACCEPTANCE_ROOT=${ACCEPTANCE_ROOT},ORIGIN_IDRID_ROOT=${IDRID_ROOT},ORIGIN_IDRID_V2_OUTPUT_ROOT=${OUTPUT_ROOT}"
JOB_ID="$(sbatch --parsable \
  --chdir="$SOURCE_REPO" \
  --output="$LOG_ROOT/idrid_v2_%j.out" \
  --error="$LOG_ROOT/idrid_v2_%j.err" \
  --export="$EXPORTS" \
  "$SNAPSHOT_ROOT/scripts/submit_origin_idrid_semantic_v2.sh" | cut -d';' -f1)"

ORIGIN_SUBMISSION_RECORD="$SUBMISSION_RECORD" \
ORIGIN_SUBMISSION_JOB="$JOB_ID" \
ORIGIN_SUBMISSION_COMMIT="$LAUNCH_COMMIT" \
ORIGIN_SUBMISSION_WORKTREE="$SNAPSHOT_ROOT" \
ORIGIN_SUBMISSION_PROTOCOL="$PROTOCOL" \
ORIGIN_SUBMISSION_PROTOCOL_SHA256="$PROTOCOL_SHA256" \
ORIGIN_SUBMISSION_WORKER_SHA256="$SUBMIT_SHA256" \
ORIGIN_SUBMISSION_CV_ROOT="$CV_ROOT" \
ORIGIN_SUBMISSION_IDRID_ROOT="$IDRID_ROOT" \
ORIGIN_SUBMISSION_OUTPUT_ROOT="$OUTPUT_ROOT" \
python - <<'PY'
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

path = Path(os.environ["ORIGIN_SUBMISSION_RECORD"])
payload = {
    "schema": "origin-idrid-semantic-v2-submission-v1",
    "created_at_utc": datetime.now(timezone.utc).isoformat(),
    "job_id": int(os.environ["ORIGIN_SUBMISSION_JOB"]),
    "launch_commit": os.environ["ORIGIN_SUBMISSION_COMMIT"],
    "immutable_worktree": os.environ["ORIGIN_SUBMISSION_WORKTREE"],
    "protocol_path": os.environ["ORIGIN_SUBMISSION_PROTOCOL"],
    "protocol_file_sha256": os.environ["ORIGIN_SUBMISSION_PROTOCOL_SHA256"],
    "worker_script_sha256": os.environ["ORIGIN_SUBMISSION_WORKER_SHA256"],
    "cv_root": os.environ["ORIGIN_SUBMISSION_CV_ROOT"],
    "idrid_root": os.environ["ORIGIN_SUBMISSION_IDRID_ROOT"],
    "output_root": os.environ["ORIGIN_SUBMISSION_OUTPUT_ROOT"],
    "analysis_status_at_launch": "pending_no_result_assumed",
}
encoded = json.dumps(
    payload, sort_keys=True, separators=(",", ":"), allow_nan=False
).encode()
payload["content_checksum_sha256"] = hashlib.sha256(encoded).hexdigest()
temporary = path.with_suffix(path.suffix + ".tmp")
temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
temporary.replace(path)
PY

rmdir "$RESERVATION_DIR"
LAUNCH_COMMITTED=1
echo "Corrective IDRiD v2 audit submitted."
echo "  job:             $JOB_ID"
echo "  launch commit:   $LAUNCH_COMMIT"
echo "  immutable tree:  $SNAPSHOT_ROOT"
echo "  output root:     $OUTPUT_ROOT"
echo "  submission:      $SUBMISSION_RECORD"
