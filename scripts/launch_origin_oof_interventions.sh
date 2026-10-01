#!/bin/bash
# Launch the frozen, read-only ORIGIN OOF intervention census.

set -euo pipefail

EXPECTED_BRANCH="origin-oof-runtime"
REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "${REPO_ROOT}"
LAUNCH_COMMIT="$(git rev-parse HEAD)"
SHORT_COMMIT="$(git rev-parse --short=12 HEAD)"
[[ "$(git branch --show-current)" == "${EXPECTED_BRANCH}" ]] || {
  echo "Expected branch ${EXPECTED_BRANCH}." >&2; exit 2;
}
[[ -z "$(git status --porcelain --untracked-files=no)" ]] || {
  echo "Tracked changes are forbidden." >&2; exit 3;
}

CV_MASTER="${ORIGIN_V3_CV_MASTER_ROOT:-${REPO_ROOT}/runs/origin_v3_full_cv_20260922T084547Z}"
SNAPSHOT_ROOT="${ORIGIN_OOF_WORKTREE:-${REPO_ROOT}-origin-oof-${SHORT_COMMIT}}"
LOG_ROOT="${ORIGIN_OOF_LOG_ROOT:-${REPO_ROOT}/origin_oof_intervention_logs}"
APTOS_DATA_ROOT="${ORIGIN_APTOS_ROOT:-${REPO_ROOT}/Datasets/aptos2019-blindness-detection}"
DR_DATA_ROOT="${ORIGIN_DR_ROOT:-${REPO_ROOT}/Datasets/DR}"

for required in \
  "${CV_MASTER}/aptos/LOCKED_PROTOCOL.json" \
  "${CV_MASTER}/dr/LOCKED_PROTOCOL.json" \
  "${APTOS_DATA_ROOT}" "${DR_DATA_ROOT}"; do
  [[ -e "${required}" ]] || { echo "Missing required path: ${required}." >&2; exit 4; }
done

if [[ -e "${SNAPSHOT_ROOT}" ]]; then
  [[ "$(git -C "${SNAPSHOT_ROOT}" rev-parse HEAD)" == "${LAUNCH_COMMIT}" ]] || {
    echo "Existing audit worktree has the wrong commit." >&2; exit 5;
  }
else
  git worktree add --detach "${SNAPSHOT_ROOT}" "${LAUNCH_COMMIT}"
fi
mkdir -p "${LOG_ROOT}"

set +u
source /etc/profile.d/lmod.sh || exit 1
module load miniconda/3 || exit 1
source activate "${ORIGIN_CONDA_ENV:-G}" || exit 1
set -u
PYTHONPATH="${SNAPSHOT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}" \
  python -m pytest -q \
  "${SNAPSHOT_ROOT}/tests/test_origin_oof_interventions.py" \
  "${SNAPSHOT_ROOT}/tests/test_origin_core.py" \
  "${SNAPSHOT_ROOT}/tests/test_origin_v3_cv_protocol.py"

submit_dataset() {
  local dataset="$1" folds="$2" concurrency="$3" data_root="$4"
  local cv_root="${CV_MASTER}/${dataset}"
  local cv_protocol="${cv_root}/LOCKED_PROTOCOL.json"
  local exports
  exports="ALL,ORIGIN_REPO_ROOT=${SNAPSHOT_ROOT},ORIGIN_LAUNCH_COMMIT=${LAUNCH_COMMIT},ORIGIN_DATASET=${dataset},ORIGIN_V3_CV_ROOT=${cv_root},ORIGIN_DATA_ROOT=${data_root},ORIGIN_V3_PROTOCOL=${cv_protocol}"
  local array_job aggregate_job
  array_job="$(sbatch --parsable \
    --array="${folds}%${concurrency}" \
    --export="${exports}" \
    "${SNAPSHOT_ROOT}/scripts/submit_origin_oof_intervention_fold.sh" | cut -d';' -f1)"
  aggregate_job="$(sbatch --parsable \
    --dependency="afterany:${array_job}" \
    --export="${exports}" \
    "${SNAPSHOT_ROOT}/scripts/submit_origin_oof_intervention_aggregate.sh" | cut -d';' -f1)"
  printf '%s %s\n' "${array_job}" "${aggregate_job}"
}

read -r APTOS_ARRAY APTOS_AGG < <(
  submit_dataset aptos 0-4 "${ORIGIN_OOF_APTOS_CONCURRENCY:-5}" "${APTOS_DATA_ROOT}"
)
read -r DR_ARRAY DR_AGG < <(
  submit_dataset dr 0-9 "${ORIGIN_OOF_DR_CONCURRENCY:-4}" "${DR_DATA_ROOT}"
)

echo "Frozen OOF intervention census submitted."
echo "  launch commit:    ${LAUNCH_COMMIT}"
echo "  immutable tree:   ${SNAPSHOT_ROOT}"
echo "  APTOS array/agg:  ${APTOS_ARRAY} / ${APTOS_AGG}"
echo "  EyePACS array/agg:${DR_ARRAY} / ${DR_AGG}"
