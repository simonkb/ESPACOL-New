#!/bin/bash
# Launch the frozen, final ORIGIN-v3 outer-CV protocol.
#
# The launcher creates an immutable detached Git worktree, locks one protocol
# manifest per dataset, and submits: preflight -> two arrays -> two aggregators.
# Untracked cluster logs in the main checkout are ignored; tracked changes fail.

set -euo pipefail

BASE_COMMIT="b0f4609b8847d862554c84b29dec59bc562d256f"
EXPECTED_IMPLEMENTATION="0d9734495c4aeccb2a0038f2b9672f88c445a13b276d7d1073cae1cc0c86909b"

REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "${REPO_ROOT}"
LAUNCH_COMMIT="$(git rev-parse HEAD)"
SHORT_COMMIT="$(git rev-parse --short=12 HEAD)"
BRANCH="$(git branch --show-current)"

[[ "${BRANCH}" == "origin-v3-full-cv" ]] || {
  echo "Refusing launch from branch '${BRANCH}'; expected origin-v3-full-cv." >&2
  exit 2
}
git merge-base --is-ancestor "${BASE_COMMIT}" "${LAUNCH_COMMIT}" || {
  echo "Launch commit is not descended from the audited V3 base ${BASE_COMMIT}." >&2
  exit 3
}
[[ -z "$(git status --porcelain --untracked-files=no)" ]] || {
  echo "Tracked worktree changes are forbidden for the final CV protocol." >&2
  git status --short --untracked-files=no >&2
  exit 4
}

ALLOWED_CHANGES=(
  scripts/aggregate_origin_v3_cv.py
  scripts/audit_origin_v3_cv_fold.py
  scripts/export_origin_v3_outer_predictions.py
  scripts/launch_origin_v3_full_cv.sh
  scripts/origin_v3_cv_common.py
  scripts/submit_origin_v3_cv_aggregate.sh
  scripts/submit_origin_v3_cv_aptos_array.sh
  scripts/submit_origin_v3_cv_dr_array.sh
  tests/test_origin_v3_cv_protocol.py
)
while IFS= read -r changed; do
  allowed=0
  for candidate in "${ALLOWED_CHANGES[@]}"; do
    [[ "${changed}" == "${candidate}" ]] && allowed=1 && break
  done
  [[ "${allowed}" == "1" ]] || {
    echo "Scientific implementation drift relative to ${BASE_COMMIT}: ${changed}" >&2
    exit 5
  }
done < <(git diff --name-only "${BASE_COMMIT}..${LAUNCH_COMMIT}")

set +u
source /etc/profile.d/lmod.sh || exit 1
module load miniconda/3 || exit 1
source activate "${ORIGIN_CONDA_ENV:-G}" || exit 1
set -u

OBSERVED_IMPLEMENTATION="$(python - <<'PY'
from training.origin_trainer import origin_implementation_signature
print(origin_implementation_signature())
PY
)"
[[ "${OBSERVED_IMPLEMENTATION}" == "${EXPECTED_IMPLEMENTATION}" ]] || {
  echo "ORIGIN implementation signature drift: ${OBSERVED_IMPLEMENTATION}." >&2
  exit 6
}
command -v sbatch >/dev/null || { echo "sbatch is unavailable." >&2; exit 7; }

TAG="${ORIGIN_V3_CV_TAG:-$(date -u +%Y%m%dT%H%M%SZ)}"
MASTER_ROOT="${ORIGIN_V3_CV_MASTER_ROOT:-${REPO_ROOT}/runs/origin_v3_full_cv_${TAG}}"
APTOS_ROOT="${MASTER_ROOT}/aptos"
DR_ROOT="${MASTER_ROOT}/dr"
APTOS_DATA_ROOT="${ORIGIN_APTOS_ROOT:-${REPO_ROOT}/Datasets/aptos2019-blindness-detection}"
DR_DATA_ROOT="${ORIGIN_DR_ROOT:-${REPO_ROOT}/Datasets/DR}"
SNAPSHOT_ROOT="${ORIGIN_V3_CV_WORKTREE:-${REPO_ROOT}-origin-v3-cv-${SHORT_COMMIT}}"

[[ -d "${APTOS_DATA_ROOT}" ]] || { echo "Missing APTOS root: ${APTOS_DATA_ROOT}." >&2; exit 8; }
[[ -d "${DR_DATA_ROOT}" ]] || { echo "Missing EyePACS root: ${DR_DATA_ROOT}." >&2; exit 9; }
[[ ! -e "${MASTER_ROOT}" ]] || {
  echo "Fresh-only protocol refuses an existing master root: ${MASTER_ROOT}." >&2
  exit 10
}

if [[ -e "${SNAPSHOT_ROOT}" ]]; then
  [[ "$(git -C "${SNAPSHOT_ROOT}" rev-parse HEAD)" == "${LAUNCH_COMMIT}" ]] || {
    echo "Existing immutable worktree has the wrong commit: ${SNAPSHOT_ROOT}." >&2
    exit 11
  }
  [[ -z "$(git -C "${SNAPSHOT_ROOT}" status --porcelain --untracked-files=no)" ]] || {
    echo "Existing immutable worktree contains tracked changes." >&2; exit 12;
  }
else
  git worktree add --detach "${SNAPSHOT_ROOT}" "${LAUNCH_COMMIT}"
fi

mkdir -p "${APTOS_ROOT}/workers" "${APTOS_ROOT}/locks" \
  "${DR_ROOT}/workers" "${DR_ROOT}/locks" \
  "${REPO_ROOT}/origin_v3_cv_logs"

create_protocol() {
  local dataset="$1" data_root="$2" cv_root="$3" output="$4"
  ORIGIN_PROTOCOL_DATASET="${dataset}" \
  ORIGIN_PROTOCOL_DATA_ROOT="${data_root}" \
  ORIGIN_PROTOCOL_CV_ROOT="${cv_root}" \
  ORIGIN_PROTOCOL_OUTPUT="${output}" \
  ORIGIN_PROTOCOL_LAUNCH_COMMIT="${LAUNCH_COMMIT}" \
  ORIGIN_PROTOCOL_REPO_ROOT="${SNAPSHOT_ROOT}" \
  python - <<'PY'
import os
from datetime import datetime, timezone
from pathlib import Path
from scripts.origin_v3_cv_common import (
    ARCHITECTURE_SIGNATURE, BASE_COMMIT, DATASET_SPECS,
    IMPLEMENTATION_SIGNATURE, PINNED_CONFIG, canonical_sha256,
    write_json_atomic,
)
dataset = os.environ["ORIGIN_PROTOCOL_DATASET"]
spec = DATASET_SPECS[dataset]
config = dict(PINNED_CONFIG)
config.update({
    "dataset": dataset,
    "n_folds": spec["n_folds"],
    "epochs": spec["epochs"],
    "early_stopping_patience": spec["early_stopping_patience"],
})
payload = {
    "schema": "origin-v3-full-cv-protocol-v1",
    "created_at_utc": datetime.now(timezone.utc).isoformat(),
    "development_status": "frozen_final_outer_evaluation_do_not_tune_on_results",
    "dataset": dataset,
    "n_folds": spec["n_folds"],
    "n_images": spec["n_images"],
    "evaluation_scope": "outer_test_after_selection",
    "base_commit": BASE_COMMIT,
    "launch_commit": os.environ["ORIGIN_PROTOCOL_LAUNCH_COMMIT"],
    "implementation_signature": IMPLEMENTATION_SIGNATURE,
    "architecture_signature": ARCHITECTURE_SIGNATURE,
    "config_signature": spec["config_signature"],
    "frozen_config": config,
    "data_root": str(Path(os.environ["ORIGIN_PROTOCOL_DATA_ROOT"]).resolve()),
    "cv_root": str(Path(os.environ["ORIGIN_PROTOCOL_CV_ROOT"]).resolve()),
    "immutable_worktree": str(Path(os.environ["ORIGIN_PROTOCOL_REPO_ROOT"]).resolve()),
}
payload["content_checksum_sha256"] = canonical_sha256(payload)
write_json_atomic(os.environ["ORIGIN_PROTOCOL_OUTPUT"], payload)
PY
}

APTOS_PROTOCOL="${APTOS_ROOT}/LOCKED_PROTOCOL.json"
DR_PROTOCOL="${DR_ROOT}/LOCKED_PROTOCOL.json"
create_protocol aptos "${APTOS_DATA_ROOT}" "${APTOS_ROOT}" "${APTOS_PROTOCOL}"
create_protocol dr "${DR_DATA_ROOT}" "${DR_ROOT}" "${DR_PROTOCOL}"

export_string() {
  local dataset="$1" cv_root="$2" data_root="$3" protocol="$4"
  printf 'ALL,ORIGIN_REPO_ROOT=%s,ORIGIN_LAUNCH_COMMIT=%s,ORIGIN_DATASET=%s,ORIGIN_V3_CV_ROOT=%s,ORIGIN_DATA_ROOT=%s,ORIGIN_V3_PROTOCOL=%s' \
    "${SNAPSHOT_ROOT}" "${LAUNCH_COMMIT}" "${dataset}" "${cv_root}" "${data_root}" "${protocol}"
}

PREFLIGHT_JOB="$(sbatch --parsable \
  --export="ALL,ORIGIN_REPO_ROOT=${SNAPSHOT_ROOT}" \
  "${SNAPSHOT_ROOT}/scripts/submit_origin_v3_preflight.sh" | cut -d';' -f1)"
APTOS_JOB="$(sbatch --parsable --dependency="afterok:${PREFLIGHT_JOB}" \
  --array="0-4%${ORIGIN_APTOS_CV_CONCURRENCY:-5}" \
  --export="$(export_string aptos "${APTOS_ROOT}" "${APTOS_DATA_ROOT}" "${APTOS_PROTOCOL}")" \
  "${SNAPSHOT_ROOT}/scripts/submit_origin_v3_cv_aptos_array.sh" | cut -d';' -f1)"
DR_JOB="$(sbatch --parsable --dependency="afterok:${PREFLIGHT_JOB}" \
  --array="0-9%${ORIGIN_DR_CV_CONCURRENCY:-3}" \
  --export="$(export_string dr "${DR_ROOT}" "${DR_DATA_ROOT}" "${DR_PROTOCOL}")" \
  "${SNAPSHOT_ROOT}/scripts/submit_origin_v3_cv_dr_array.sh" | cut -d';' -f1)"
APTOS_AGG_JOB="$(sbatch --parsable --dependency="afterok:${APTOS_JOB}" \
  --export="$(export_string aptos "${APTOS_ROOT}" "${APTOS_DATA_ROOT}" "${APTOS_PROTOCOL}")" \
  "${SNAPSHOT_ROOT}/scripts/submit_origin_v3_cv_aggregate.sh" | cut -d';' -f1)"
DR_AGG_JOB="$(sbatch --parsable --dependency="afterok:${DR_JOB}" \
  --export="$(export_string dr "${DR_ROOT}" "${DR_DATA_ROOT}" "${DR_PROTOCOL}")" \
  "${SNAPSHOT_ROOT}/scripts/submit_origin_v3_cv_aggregate.sh" | cut -d';' -f1)"

ORIGIN_SUBMISSION_OUTPUT="${MASTER_ROOT}/SUBMISSION.json" \
ORIGIN_SUBMISSION_TAG="${TAG}" \
ORIGIN_SUBMISSION_COMMIT="${LAUNCH_COMMIT}" \
ORIGIN_SUBMISSION_PREFLIGHT="${PREFLIGHT_JOB}" \
ORIGIN_SUBMISSION_APTOS="${APTOS_JOB}" \
ORIGIN_SUBMISSION_DR="${DR_JOB}" \
ORIGIN_SUBMISSION_APTOS_AGG="${APTOS_AGG_JOB}" \
ORIGIN_SUBMISSION_DR_AGG="${DR_AGG_JOB}" \
ORIGIN_SUBMISSION_APTOS_ROOT="${APTOS_ROOT}" \
ORIGIN_SUBMISSION_DR_ROOT="${DR_ROOT}" \
python - <<'PY'
import os
from datetime import datetime, timezone
from scripts.origin_v3_cv_common import canonical_sha256, write_json_atomic
p = {
    "schema": "origin-v3-full-cv-submission-v1",
    "created_at_utc": datetime.now(timezone.utc).isoformat(),
    "tag": os.environ["ORIGIN_SUBMISSION_TAG"],
    "launch_commit": os.environ["ORIGIN_SUBMISSION_COMMIT"],
    "jobs": {
        "preflight": os.environ["ORIGIN_SUBMISSION_PREFLIGHT"],
        "aptos_array": os.environ["ORIGIN_SUBMISSION_APTOS"],
        "dr_array": os.environ["ORIGIN_SUBMISSION_DR"],
        "aptos_aggregate": os.environ["ORIGIN_SUBMISSION_APTOS_AGG"],
        "dr_aggregate": os.environ["ORIGIN_SUBMISSION_DR_AGG"],
    },
    "roots": {
        "aptos": os.environ["ORIGIN_SUBMISSION_APTOS_ROOT"],
        "dr": os.environ["ORIGIN_SUBMISSION_DR_ROOT"],
    },
}
p["content_checksum_sha256"] = canonical_sha256(p)
write_json_atomic(os.environ["ORIGIN_SUBMISSION_OUTPUT"], p)
PY

echo "Frozen ORIGIN-v3 full CV submitted."
echo "  tag:              ${TAG}"
echo "  launch commit:    ${LAUNCH_COMMIT}"
echo "  preflight:        ${PREFLIGHT_JOB}"
echo "  APTOS array:      ${APTOS_JOB}"
echo "  EyePACS array:    ${DR_JOB}"
echo "  APTOS aggregate:  ${APTOS_AGG_JOB}"
echo "  EyePACS aggregate:${DR_AGG_JOB}"
echo "  master root:      ${MASTER_ROOT}"
echo "Monitor: squeue -u \"${USER}\""
