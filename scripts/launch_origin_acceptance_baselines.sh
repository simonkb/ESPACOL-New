#!/bin/bash
# Submit the acceptance suite from one immutable detached launch snapshot:
# canary -> canary audit -> full training -> freeze audit -> outer release.

set -euo pipefail

EXPECTED_BRANCH="origin-acceptance-revision"
SOURCE_REPO="$(git rev-parse --show-toplevel)"
cd "${SOURCE_REPO}"
LAUNCH_COMMIT="$(git rev-parse HEAD)"
SHORT_COMMIT="$(git rev-parse --short=12 HEAD)"
SOURCE_BRANCH="$(git branch --show-current)"

[[ "${SOURCE_BRANCH}" == "${EXPECTED_BRANCH}" ]] || {
  echo "Refusing launch from branch '${SOURCE_BRANCH}'; expected ${EXPECTED_BRANCH}." >&2
  exit 2
}
[[ -z "$(git status --porcelain --untracked-files=no)" ]] || {
  echo "Tracked changes are forbidden when freezing the acceptance suite." >&2
  git status --short --untracked-files=no >&2
  exit 3
}

TAG="${ORIGIN_ACCEPTANCE_TAG:-$(date -u +%Y%m%dT%H%M%SZ)}"
EXPERIMENT_ROOT="${ORIGIN_ACCEPTANCE_ROOT:-${SOURCE_REPO}/runs/origin_acceptance_${TAG}}"
APTOS_DATA_ROOT="${ORIGIN_APTOS_ROOT:-${SOURCE_REPO}/Datasets/aptos2019-blindness-detection}"
DR_DATA_ROOT="${ORIGIN_DR_ROOT:-${SOURCE_REPO}/Datasets/DR}"
SNAPSHOT_ROOT="${ORIGIN_ACCEPTANCE_WORKTREE:-${SOURCE_REPO}-origin-acceptance-${SHORT_COMMIT}}"
LOG_ROOT="${ORIGIN_ACCEPTANCE_LOG_ROOT:-${SOURCE_REPO}/origin_acceptance_logs}"
CONDA_ENV="${ORIGIN_CONDA_ENV:-G}"

[[ -d "${APTOS_DATA_ROOT}" ]] || {
  echo "Missing APTOS data root: ${APTOS_DATA_ROOT}." >&2
  exit 4
}
[[ -d "${DR_DATA_ROOT}" ]] || {
  echo "Missing EyePACS data root: ${DR_DATA_ROOT}." >&2
  exit 5
}
[[ ! -e "${EXPERIMENT_ROOT}" ]] || {
  echo "Fresh launch refuses existing experiment root: ${EXPERIMENT_ROOT}." >&2
  exit 6
}

# Login nodes expose an obsolete system Python on the target cluster. Activate
# the exact environment before creating the preflight result or invoking
# sbatch, and export its name to every worker.
set +u
source /etc/profile.d/lmod.sh || exit 1
module load miniconda/3 || exit 1
source activate "${CONDA_ENV}" || exit 1
set -u
command -v sbatch >/dev/null || { echo "sbatch is unavailable." >&2; exit 7; }

if [[ -e "${SNAPSHOT_ROOT}" ]]; then
  git -C "${SNAPSHOT_ROOT}" rev-parse --is-inside-work-tree >/dev/null 2>&1 || {
    echo "Existing snapshot path is not a Git worktree: ${SNAPSHOT_ROOT}." >&2
    exit 8
  }
  [[ "$(git -C "${SNAPSHOT_ROOT}" rev-parse HEAD)" == "${LAUNCH_COMMIT}" ]] || {
    echo "Existing immutable worktree has the wrong commit: ${SNAPSHOT_ROOT}." >&2
    exit 9
  }
  [[ -z "$(git -C "${SNAPSHOT_ROOT}" branch --show-current)" ]] || {
    echo "Existing launch worktree is not detached: ${SNAPSHOT_ROOT}." >&2
    exit 10
  }
  [[ -z "$(git -C "${SNAPSHOT_ROOT}" status --porcelain --untracked-files=no)" ]] || {
    echo "Existing immutable worktree contains tracked changes." >&2
    exit 11
  }
else
  git worktree add --detach "${SNAPSHOT_ROOT}" "${LAUNCH_COMMIT}"
fi

mkdir -p "${LOG_ROOT}"
cd "${SNAPSHOT_ROOT}"

# Targeted local preflight runs from the detached snapshot. Thus successful
# tests certify exactly the source tree the scheduler will execute.
python -m pytest -q tests/test_origin_acceptance_baselines.py
python -m py_compile \
  models/origin_acceptance_baselines.py \
  configs/origin_acceptance_baseline_config.py \
  losses/origin_acceptance_baselines.py \
  training/origin_acceptance_baseline_trainer.py \
  train_origin_acceptance_baseline.py \
  release_origin_acceptance_baselines.py \
  scripts/audit_origin_acceptance_baselines.py \
  scripts/aggregate_origin_acceptance_release.py
bash -n \
  scripts/launch_origin_acceptance_baselines.sh \
  scripts/submit_origin_acceptance_aptos_f0_canary.sh \
  scripts/submit_origin_acceptance_canary_audit.sh \
  scripts/submit_origin_acceptance_full_array.sh \
  scripts/submit_origin_acceptance_full_audit.sh \
  scripts/submit_origin_acceptance_outer_release.sh \
  scripts/submit_origin_acceptance_release_aggregate.sh

mkdir -p "${EXPERIMENT_ROOT}" "${LOG_ROOT}"
EXPORTS="ALL,ORIGIN_REPO_ROOT=${SNAPSHOT_ROOT},ORIGIN_ACCEPTANCE_ROOT=${EXPERIMENT_ROOT},ORIGIN_APTOS_ROOT=${APTOS_DATA_ROOT},ORIGIN_DR_ROOT=${DR_DATA_ROOT},ORIGIN_LAUNCH_COMMIT=${LAUNCH_COMMIT},ORIGIN_CONDA_ENV=${CONDA_ENV}"

CANARY_JOB="$(sbatch --parsable \
  --export="${EXPORTS}" \
  --output="${LOG_ROOT}/canary_%A_%a.out" \
  --error="${LOG_ROOT}/canary_%A_%a.err" \
  "${SNAPSHOT_ROOT}/scripts/submit_origin_acceptance_aptos_f0_canary.sh" | cut -d';' -f1)"
CANARY_AUDIT_JOB="$(sbatch --parsable --dependency="afterok:${CANARY_JOB}" \
  --export="${EXPORTS}" \
  --output="${LOG_ROOT}/canary_audit_%j.out" \
  --error="${LOG_ROOT}/canary_audit_%j.err" \
  "${SNAPSHOT_ROOT}/scripts/submit_origin_acceptance_canary_audit.sh" | cut -d';' -f1)"
FULL_JOB="$(sbatch --parsable --dependency="afterok:${CANARY_AUDIT_JOB}" \
  --export="${EXPORTS}" \
  --output="${LOG_ROOT}/full_%A_%a.out" \
  --error="${LOG_ROOT}/full_%A_%a.err" \
  "${SNAPSHOT_ROOT}/scripts/submit_origin_acceptance_full_array.sh" | cut -d';' -f1)"
FULL_AUDIT_JOB="$(sbatch --parsable --dependency="afterok:${FULL_JOB}" \
  --export="${EXPORTS}" \
  --output="${LOG_ROOT}/full_audit_%j.out" \
  --error="${LOG_ROOT}/full_audit_%j.err" \
  "${SNAPSHOT_ROOT}/scripts/submit_origin_acceptance_full_audit.sh" | cut -d';' -f1)"
RELEASE_JOB="$(sbatch --parsable --dependency="afterok:${FULL_AUDIT_JOB}" \
  --export="${EXPORTS}" \
  --output="${LOG_ROOT}/release_%A_%a.out" \
  --error="${LOG_ROOT}/release_%A_%a.err" \
  "${SNAPSHOT_ROOT}/scripts/submit_origin_acceptance_outer_release.sh" | cut -d';' -f1)"
AGGREGATE_JOB="$(sbatch --parsable --dependency="afterok:${RELEASE_JOB}" \
  --export="${EXPORTS}" \
  --output="${LOG_ROOT}/aggregate_%j.out" \
  --error="${LOG_ROOT}/aggregate_%j.err" \
  "${SNAPSHOT_ROOT}/scripts/submit_origin_acceptance_release_aggregate.sh" | cut -d';' -f1)"

ORIGIN_SUBMISSION_OUTPUT="${EXPERIMENT_ROOT}/SUBMISSION.json" \
ORIGIN_SUBMISSION_TAG="${TAG}" \
ORIGIN_SUBMISSION_COMMIT="${LAUNCH_COMMIT}" \
ORIGIN_SUBMISSION_SOURCE="${SOURCE_REPO}" \
ORIGIN_SUBMISSION_SNAPSHOT="${SNAPSHOT_ROOT}" \
ORIGIN_SUBMISSION_APTOS="${APTOS_DATA_ROOT}" \
ORIGIN_SUBMISSION_DR="${DR_DATA_ROOT}" \
ORIGIN_SUBMISSION_ENV="${CONDA_ENV}" \
ORIGIN_SUBMISSION_CANARY="${CANARY_JOB}" \
ORIGIN_SUBMISSION_CANARY_AUDIT="${CANARY_AUDIT_JOB}" \
ORIGIN_SUBMISSION_FULL="${FULL_JOB}" \
ORIGIN_SUBMISSION_FULL_AUDIT="${FULL_AUDIT_JOB}" \
ORIGIN_SUBMISSION_RELEASE="${RELEASE_JOB}" \
ORIGIN_SUBMISSION_AGGREGATE="${AGGREGATE_JOB}" \
PYTHONPATH="${SNAPSHOT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}" \
python - <<'PY'
import os
from datetime import datetime, timezone
from scripts.origin_acceptance_baseline_common import (
    PROTOCOL_ID,
    PROTOCOL_SHA256,
    canonical_sha256,
)
from train_origin import write_json_atomic

payload = {
    "schema": "origin-acceptance-submission-v1",
    "created_at_utc": datetime.now(timezone.utc).isoformat(),
    "tag": os.environ["ORIGIN_SUBMISSION_TAG"],
    "protocol_id": PROTOCOL_ID,
    "protocol_sha256": PROTOCOL_SHA256,
    "launch_commit": os.environ["ORIGIN_SUBMISSION_COMMIT"],
    "source_repository": os.environ["ORIGIN_SUBMISSION_SOURCE"],
    "immutable_worktree": os.environ["ORIGIN_SUBMISSION_SNAPSHOT"],
    "conda_environment": os.environ["ORIGIN_SUBMISSION_ENV"],
    "data_roots": {
        "aptos": os.environ["ORIGIN_SUBMISSION_APTOS"],
        "dr": os.environ["ORIGIN_SUBMISSION_DR"],
    },
    "preflight": "targeted_tests_py_compile_and_bash_parse_passed_before_submission",
    "jobs": {
        "aptos_fold0_canary_array": os.environ["ORIGIN_SUBMISSION_CANARY"],
        "canary_gate_audit": os.environ["ORIGIN_SUBMISSION_CANARY_AUDIT"],
        "full_training_array": os.environ["ORIGIN_SUBMISSION_FULL"],
        "full_freeze_audit": os.environ["ORIGIN_SUBMISSION_FULL_AUDIT"],
        "coordinated_outer_release_array": os.environ["ORIGIN_SUBMISSION_RELEASE"],
        "release_aggregate": os.environ["ORIGIN_SUBMISSION_AGGREGATE"],
    },
}
payload["content_checksum_sha256"] = canonical_sha256(payload)
write_json_atomic(os.environ["ORIGIN_SUBMISSION_OUTPUT"], payload)
PY

printf 'Acceptance baseline suite submitted from immutable snapshot.\n'
printf '  tag:             %s\n' "${TAG}"
printf '  launch commit:   %s\n' "${LAUNCH_COMMIT}"
printf '  snapshot:        %s\n' "${SNAPSHOT_ROOT}"
printf '  experiment root: %s\n' "${EXPERIMENT_ROOT}"
printf '  canary:          %s\n' "${CANARY_JOB}"
printf '  canary audit:    %s\n' "${CANARY_AUDIT_JOB}"
printf '  full array:      %s\n' "${FULL_JOB}"
printf '  full audit:      %s\n' "${FULL_AUDIT_JOB}"
printf '  outer release:   %s\n' "${RELEASE_JOB}"
printf '  aggregate:       %s\n' "${AGGREGATE_JOB}"
printf 'Monitor: squeue -u %q\n' "${USER:-unknown}"
