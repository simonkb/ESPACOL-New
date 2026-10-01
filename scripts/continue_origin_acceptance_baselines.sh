#!/bin/bash
# Continue a sealed acceptance-baseline experiment after an infrastructure-only
# canary-audit failure. The completed canary is reused; an explicitly supplied
# replacement audit job gates every newly submitted downstream stage.

set -euo pipefail

EXPECTED_BRANCH="origin-acceptance-revision"
SOURCE_REPO="$(git rev-parse --show-toplevel)"
cd "${SOURCE_REPO}"
LAUNCH_COMMIT="$(git rev-parse HEAD)"
SHORT_COMMIT="$(git rev-parse --short=12 HEAD)"
SOURCE_BRANCH="$(git branch --show-current)"

[[ "${SOURCE_BRANCH}" == "${EXPECTED_BRANCH}" ]] || {
  echo "Refusing continuation from branch '${SOURCE_BRANCH}'; expected ${EXPECTED_BRANCH}." >&2
  exit 2
}
[[ -z "$(git status --porcelain --untracked-files=no)" ]] || {
  echo "Tracked changes are forbidden when freezing a continuation." >&2
  git status --short --untracked-files=no >&2
  exit 3
}

EXPERIMENT_ROOT="${ORIGIN_ACCEPTANCE_ROOT:?ORIGIN_ACCEPTANCE_ROOT is required}"
EXPECTED_PARENT_SHA="${ORIGIN_PARENT_SUBMISSION_SHA256:?ORIGIN_PARENT_SUBMISSION_SHA256 is required}"
PARENT_CANARY_JOB="${ORIGIN_PARENT_CANARY_JOB:?ORIGIN_PARENT_CANARY_JOB is required}"
FAILED_PARENT_AUDIT_JOB="${ORIGIN_PARENT_FAILED_CANARY_AUDIT_JOB:?ORIGIN_PARENT_FAILED_CANARY_AUDIT_JOB is required}"
CANARY_REAUDIT_JOB="${ORIGIN_CANARY_AUDIT_JOB:?ORIGIN_CANARY_AUDIT_JOB is required}"
APTOS_DATA_ROOT="${ORIGIN_APTOS_ROOT:-${SOURCE_REPO}/Datasets/aptos2019-blindness-detection}"
DR_DATA_ROOT="${ORIGIN_DR_ROOT:-${SOURCE_REPO}/Datasets/DR}"
SNAPSHOT_ROOT="${ORIGIN_ACCEPTANCE_CONTINUATION_WORKTREE:-${SOURCE_REPO}-origin-acceptance-continuation-${SHORT_COMMIT}}"
LOG_ROOT="${ORIGIN_ACCEPTANCE_LOG_ROOT:-${SOURCE_REPO}/origin_acceptance_logs}"
CONDA_ENV="${ORIGIN_CONDA_ENV:-G}"
RESTART_MANIFEST="${EXPERIMENT_ROOT}/RESTART_SUBMISSION.json"

for value in "${PARENT_CANARY_JOB}" "${FAILED_PARENT_AUDIT_JOB}" "${CANARY_REAUDIT_JOB}"; do
  [[ "${value}" =~ ^[0-9]+$ ]] || { echo "Scheduler job ids must be decimal integers." >&2; exit 4; }
done
[[ "${CANARY_REAUDIT_JOB}" != "${FAILED_PARENT_AUDIT_JOB}" ]] || {
  echo "Replacement canary audit job equals the failed parent audit job." >&2
  exit 5
}
[[ -d "${APTOS_DATA_ROOT}" && -d "${DR_DATA_ROOT}" ]] || {
  echo "One or both sealed data roots are missing." >&2
  exit 6
}
[[ ! -e "${RESTART_MANIFEST}" ]] || {
  echo "A continuation is already recorded: ${RESTART_MANIFEST}" >&2
  exit 7
}

set +u
source /etc/profile.d/lmod.sh || exit 1
module load miniconda/3 || exit 1
source activate "${CONDA_ENV}" || exit 1
set -u
command -v sbatch >/dev/null || { echo "sbatch is unavailable." >&2; exit 8; }
command -v sacct >/dev/null || { echo "sacct is unavailable." >&2; exit 9; }
command -v scancel >/dev/null || { echo "scancel is unavailable." >&2; exit 10; }

job_state() {
  local job_id="$1" state
  state="$(sacct -X -n -P -j "${job_id}" --format=JobIDRaw,State 2>/dev/null \
    | awk -F'|' -v wanted="${job_id}" '$1 == wanted {print $2; exit}')"
  if [[ -z "${state}" ]]; then
    state="$(squeue -h -j "${job_id}" -o '%T' 2>/dev/null | head -n1)"
  fi
  state="${state%%+*}"
  [[ -n "${state}" ]] || { echo "Cannot resolve scheduler state for job ${job_id}." >&2; return 1; }
  printf '%s\n' "${state}"
}

PARENT_CANARY_STATE="$(job_state "${PARENT_CANARY_JOB}")"
PARENT_AUDIT_STATE="$(job_state "${FAILED_PARENT_AUDIT_JOB}")"
REAUDIT_STATE="$(job_state "${CANARY_REAUDIT_JOB}")"
[[ "${PARENT_CANARY_STATE}" == "COMPLETED" ]] || {
  echo "Parent canary array is not completed: ${PARENT_CANARY_STATE}." >&2
  exit 11
}
case "${PARENT_AUDIT_STATE}" in
  FAILED|CANCELLED|TIMEOUT|OUT_OF_MEMORY|NODE_FAIL|PREEMPTED|BOOT_FAIL|DEADLINE) ;;
  *) echo "Parent canary audit is not in a terminal failure state: ${PARENT_AUDIT_STATE}." >&2; exit 12 ;;
esac
case "${REAUDIT_STATE}" in
  PENDING|CONFIGURING|RUNNING|COMPLETING|COMPLETED) ;;
  *) echo "Replacement canary audit is not usable: ${REAUDIT_STATE}." >&2; exit 13 ;;
esac

if [[ -e "${SNAPSHOT_ROOT}" ]]; then
  git -C "${SNAPSHOT_ROOT}" rev-parse --is-inside-work-tree >/dev/null 2>&1 || {
    echo "Existing continuation snapshot is not a Git worktree." >&2; exit 14;
  }
  [[ "$(git -C "${SNAPSHOT_ROOT}" rev-parse HEAD)" == "${LAUNCH_COMMIT}" ]] || {
    echo "Existing continuation snapshot commit mismatch." >&2; exit 15;
  }
  [[ -z "$(git -C "${SNAPSHOT_ROOT}" branch --show-current)" ]] || {
    echo "Continuation snapshot is not detached." >&2; exit 16;
  }
  [[ -z "$(git -C "${SNAPSHOT_ROOT}" status --porcelain --untracked-files=no)" ]] || {
    echo "Continuation snapshot has tracked changes." >&2; exit 17;
  }
else
  git worktree add --detach "${SNAPSHOT_ROOT}" "${LAUNCH_COMMIT}"
fi

mkdir -p "${EXPERIMENT_ROOT}/locks" "${LOG_ROOT}"
exec 9>"${EXPERIMENT_ROOT}/locks/acceptance_continuation.lock"
flock -n 9 || { echo "Another continuation launcher owns this experiment." >&2; exit 18; }

PREFLIGHT_DIR="$(mktemp -d "${TMPDIR:-/tmp}/origin-acceptance-continuation.XXXXXX")"
PREFLIGHT_RECORD="${PREFLIGHT_DIR}/preflight.json"
cleanup_preflight() {
  rm -f "${PREFLIGHT_RECORD}"
  rmdir "${PREFLIGHT_DIR}" 2>/dev/null || true
}
trap cleanup_preflight EXIT

cd "${SNAPSHOT_ROOT}"
export PYTHONPATH="${SNAPSHOT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
python -m pytest -q tests/test_origin_acceptance_baselines.py
python -m py_compile scripts/validate_origin_acceptance_continuation.py
bash -n \
  scripts/continue_origin_acceptance_baselines.sh \
  scripts/submit_origin_acceptance_canary_audit.sh \
  scripts/submit_origin_acceptance_full_array.sh \
  scripts/submit_origin_acceptance_full_audit.sh \
  scripts/submit_origin_acceptance_outer_release.sh \
  scripts/submit_origin_acceptance_release_aggregate.sh

python -m scripts.validate_origin_acceptance_continuation validate \
  --experiment-root "${EXPERIMENT_ROOT}" \
  --expected-submission-sha256 "${EXPECTED_PARENT_SHA}" \
  --expected-canary-job "${PARENT_CANARY_JOB}" \
  --expected-failed-canary-audit-job "${FAILED_PARENT_AUDIT_JOB}" \
  --source-repository "${SOURCE_REPO}" \
  --aptos-root "${APTOS_DATA_ROOT}" \
  --dr-root "${DR_DATA_ROOT}" \
  --output "${PREFLIGHT_RECORD}"

if [[ "${REAUDIT_STATE}" == "COMPLETED" && ! -f "${EXPERIMENT_ROOT}/canary/CANARY_PASSED.json" ]]; then
  echo "Completed replacement audit did not produce CANARY_PASSED.json." >&2
  exit 19
fi

if [[ "${REAUDIT_STATE}" == "COMPLETED" ]]; then
  # Slurm may purge a completed job from the active-controller dependency
  # window (MinJobAge).  The validator above has independently replayed and
  # matched the exact CANARY_PASSED gate, so no scheduler dependency is needed.
  REAUDIT_DEPENDENCY_MODE="verified_completed_no_dependency"
else
  REAUDIT_DEPENDENCY_MODE="afterok_live_reaudit"
fi

EXPORTS="ALL,ORIGIN_REPO_ROOT=${SNAPSHOT_ROOT},ORIGIN_ACCEPTANCE_ROOT=${EXPERIMENT_ROOT},ORIGIN_APTOS_ROOT=${APTOS_DATA_ROOT},ORIGIN_DR_ROOT=${DR_DATA_ROOT},ORIGIN_LAUNCH_COMMIT=${LAUNCH_COMMIT},ORIGIN_CONDA_ENV=${CONDA_ENV}"
NEW_JOBS=()
cancel_partial_submission() {
  if ((${#NEW_JOBS[@]})); then
    echo "Continuation submission failed; cancelling newly submitted jobs: ${NEW_JOBS[*]}" >&2
    scancel "${NEW_JOBS[@]}" || true
  fi
}
trap 'cancel_partial_submission; cleanup_preflight' ERR

if [[ "${REAUDIT_STATE}" == "COMPLETED" ]]; then
  FULL_JOB="$(sbatch --parsable \
    --export="${EXPORTS}" \
    --output="${LOG_ROOT}/full_%A_%a.out" --error="${LOG_ROOT}/full_%A_%a.err" \
    "${SNAPSHOT_ROOT}/scripts/submit_origin_acceptance_full_array.sh" | cut -d';' -f1)"
else
  FULL_JOB="$(sbatch --parsable --dependency="afterok:${CANARY_REAUDIT_JOB}" \
    --export="${EXPORTS}" \
    --output="${LOG_ROOT}/full_%A_%a.out" --error="${LOG_ROOT}/full_%A_%a.err" \
    "${SNAPSHOT_ROOT}/scripts/submit_origin_acceptance_full_array.sh" | cut -d';' -f1)"
fi
NEW_JOBS+=("${FULL_JOB}")
FULL_AUDIT_JOB="$(sbatch --parsable --dependency="afterok:${FULL_JOB}" \
  --export="${EXPORTS}" \
  --output="${LOG_ROOT}/full_audit_%j.out" --error="${LOG_ROOT}/full_audit_%j.err" \
  "${SNAPSHOT_ROOT}/scripts/submit_origin_acceptance_full_audit.sh" | cut -d';' -f1)"
NEW_JOBS+=("${FULL_AUDIT_JOB}")
RELEASE_JOB="$(sbatch --parsable --dependency="afterok:${FULL_AUDIT_JOB}" \
  --export="${EXPORTS}" \
  --output="${LOG_ROOT}/release_%A_%a.out" --error="${LOG_ROOT}/release_%A_%a.err" \
  "${SNAPSHOT_ROOT}/scripts/submit_origin_acceptance_outer_release.sh" | cut -d';' -f1)"
NEW_JOBS+=("${RELEASE_JOB}")
AGGREGATE_JOB="$(sbatch --parsable --dependency="afterok:${RELEASE_JOB}" \
  --export="${EXPORTS}" \
  --output="${LOG_ROOT}/aggregate_%j.out" --error="${LOG_ROOT}/aggregate_%j.err" \
  "${SNAPSHOT_ROOT}/scripts/submit_origin_acceptance_release_aggregate.sh" | cut -d';' -f1)"
NEW_JOBS+=("${AGGREGATE_JOB}")

python -m scripts.validate_origin_acceptance_continuation write-manifest \
  --output "${RESTART_MANIFEST}" --preflight "${PREFLIGHT_RECORD}" \
  --launch-commit "${LAUNCH_COMMIT}" --immutable-worktree "${SNAPSHOT_ROOT}" \
  --conda-environment "${CONDA_ENV}" \
  --job-canary-reaudit "${CANARY_REAUDIT_JOB}" \
  --job-full-training-array "${FULL_JOB}" \
  --job-full-freeze-audit "${FULL_AUDIT_JOB}" \
  --job-coordinated-outer-release-array "${RELEASE_JOB}" \
  --job-release-aggregate "${AGGREGATE_JOB}" \
  --parent-canary-state "${PARENT_CANARY_STATE}" \
  --parent-canary-audit-state "${PARENT_AUDIT_STATE}" \
  --reaudit-dependency-mode "${REAUDIT_DEPENDENCY_MODE}"

trap cleanup_preflight EXIT
printf 'Acceptance baseline continuation submitted.\n'
printf '  experiment root: %s\n' "${EXPERIMENT_ROOT}"
printf '  reused canary:   %s\n' "${PARENT_CANARY_JOB}"
printf '  canary re-audit: %s\n' "${CANARY_REAUDIT_JOB}"
printf '  re-audit mode:   %s\n' "${REAUDIT_DEPENDENCY_MODE}"
printf '  full array:      %s\n' "${FULL_JOB}"
printf '  full audit:      %s\n' "${FULL_AUDIT_JOB}"
printf '  outer release:   %s\n' "${RELEASE_JOB}"
printf '  aggregate:       %s\n' "${AGGREGATE_JOB}"
printf '  manifest:        %s\n' "${RESTART_MANIFEST}"
