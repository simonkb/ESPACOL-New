#!/bin/bash
# Launch a frozen retrospective EyePACS fold-9 mechanism ablation.
#
# Stages are deliberately separated:
#   structural preflight -> validation-only training array
#   -> one all-variant outer release -> after-any aggregation/audit.

set -euo pipefail

ABLATION_BASE_COMMIT="29925b9"
EXPECTED_BRANCH="origin-v3-fold9-ablations"

REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "${REPO_ROOT}"
LAUNCH_COMMIT="$(git rev-parse HEAD)"
SHORT_COMMIT="$(git rev-parse --short=12 HEAD)"
BRANCH="$(git branch --show-current)"

[[ "${BRANCH}" == "${EXPECTED_BRANCH}" ]] || {
  echo "Refusing launch from branch '${BRANCH}'; expected ${EXPECTED_BRANCH}." >&2
  exit 2
}
git merge-base --is-ancestor "${ABLATION_BASE_COMMIT}" "${LAUNCH_COMMIT}" || {
  echo "Launch commit is not descended from ${ABLATION_BASE_COMMIT}." >&2
  exit 3
}
[[ -z "$(git status --porcelain --untracked-files=no)" ]] || {
  echo "Tracked worktree changes are forbidden for the ablation protocol." >&2
  git status --short --untracked-files=no >&2
  exit 4
}

# Login nodes expose an obsolete system Python by default.  Build and sign the
# protocol with the same Python environment used by every Slurm stage.
set +u
source /etc/profile.d/lmod.sh || exit 1
module load miniconda/3 || exit 1
source activate "${ORIGIN_CONDA_ENV:-G}" || exit 1
set -u
command -v sbatch >/dev/null || { echo "sbatch is unavailable." >&2; exit 5; }

GROUP="${ORIGIN_FOLD9_ABLATION_GROUP:-full}"
case "${GROUP}" in
  full) VARIANT_COUNT=10 ;;
  core) VARIANT_COUNT=5 ;;
  *) echo "ORIGIN_FOLD9_ABLATION_GROUP must be 'full' or 'core'." >&2; exit 6 ;;
esac

TAG="${ORIGIN_FOLD9_ABLATION_TAG:-$(date -u +%Y%m%dT%H%M%SZ)}"
EXPERIMENT_ROOT="${ORIGIN_FOLD9_ABLATION_ROOT:-${REPO_ROOT}/runs/origin_fold9_ablation_${GROUP}_${TAG}}"
DATA_ROOT="${ORIGIN_DR_ROOT:-${REPO_ROOT}/Datasets/DR}"
SNAPSHOT_ROOT="${ORIGIN_FOLD9_ABLATION_WORKTREE:-${REPO_ROOT}-origin-fold9-ablation-${SHORT_COMMIT}}"
PROTOCOL="${EXPERIMENT_ROOT}/LOCKED_PROTOCOL.json"
LOG_ROOT="${REPO_ROOT}/origin_fold9_ablation_logs"

[[ -d "${DATA_ROOT}" ]] || { echo "Missing EyePACS root: ${DATA_ROOT}." >&2; exit 7; }
[[ ! -e "${EXPERIMENT_ROOT}" ]] || {
  echo "Fresh-only protocol refuses existing root: ${EXPERIMENT_ROOT}." >&2
  exit 8
}

if [[ -e "${SNAPSHOT_ROOT}" ]]; then
  [[ "$(git -C "${SNAPSHOT_ROOT}" rev-parse HEAD)" == "${LAUNCH_COMMIT}" ]] || {
    echo "Existing immutable worktree has the wrong commit: ${SNAPSHOT_ROOT}." >&2
    exit 9
  }
  [[ -z "$(git -C "${SNAPSHOT_ROOT}" status --porcelain --untracked-files=no)" ]] || {
    echo "Existing immutable worktree contains tracked changes." >&2
    exit 10
  }
else
  git worktree add --detach "${SNAPSHOT_ROOT}" "${LAUNCH_COMMIT}"
fi

mkdir -p "${EXPERIMENT_ROOT}/workers" "${EXPERIMENT_ROOT}/locks" "${LOG_ROOT}"

# From this point onward every imported protocol module must come from the
# immutable launch snapshot, not from a possibly changing caller worktree.
cd "${SNAPSHOT_ROOT}"

ORIGIN_ABLATION_GROUP="${GROUP}" \
ORIGIN_ABLATION_COMMIT="${LAUNCH_COMMIT}" \
ORIGIN_ABLATION_ROOT="${EXPERIMENT_ROOT}" \
ORIGIN_ABLATION_DATA_ROOT="${DATA_ROOT}" \
ORIGIN_ABLATION_WORKTREE="${SNAPSHOT_ROOT}" \
ORIGIN_ABLATION_TAG="${TAG}" \
ORIGIN_ABLATION_PROTOCOL="${PROTOCOL}" \
PYTHONPATH="${SNAPSHOT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}" \
python - <<'PY'
import os
from scripts.origin_fold9_ablation_common import (
    make_protocol_payload,
    variants_for_group,
    write_json_atomic,
)
group = os.environ["ORIGIN_ABLATION_GROUP"]
payload = make_protocol_payload(
    selected_variants=variants_for_group(group),
    group=group,
    launch_commit=os.environ["ORIGIN_ABLATION_COMMIT"],
    experiment_root=os.environ["ORIGIN_ABLATION_ROOT"],
    data_root=os.environ["ORIGIN_ABLATION_DATA_ROOT"],
    immutable_worktree=os.environ["ORIGIN_ABLATION_WORKTREE"],
    tag=os.environ["ORIGIN_ABLATION_TAG"],
)
write_json_atomic(os.environ["ORIGIN_ABLATION_PROTOCOL"], payload)
PY

EXPORTS="ALL,ORIGIN_REPO_ROOT=${SNAPSHOT_ROOT},ORIGIN_LAUNCH_COMMIT=${LAUNCH_COMMIT},ORIGIN_FOLD9_ABLATION_ROOT=${EXPERIMENT_ROOT},ORIGIN_DATA_ROOT=${DATA_ROOT},ORIGIN_FOLD9_ABLATION_PROTOCOL=${PROTOCOL}"

PREFLIGHT_JOB="$(sbatch --parsable \
  --export="${EXPORTS}" \
  "${SNAPSHOT_ROOT}/scripts/submit_origin_fold9_ablation_preflight.sh" | cut -d';' -f1)"
# Use after-any plus an authenticated preflight marker. On this cluster an
# after-ok child can otherwise remain DependencyNeverSatisfied indefinitely;
# workers fail immediately and visibly when preflight did not complete.
TRAIN_JOB="$(sbatch --parsable --dependency="afterany:${PREFLIGHT_JOB}" \
  --array="0-$((VARIANT_COUNT - 1))%${ORIGIN_FOLD9_ABLATION_CONCURRENCY:-3}" \
  --export="${EXPORTS}" \
  "${SNAPSHOT_ROOT}/scripts/submit_origin_fold9_ablation_train_array.sh" | cut -d';' -f1)"
# Run the release gate after any array outcome.  Its all-marker audit remains
# the only unlock condition and therefore produces an actionable failure
# instead of a DependencyNeverSatisfied job when one worker fails.
RELEASE_JOB="$(sbatch --parsable --dependency="afterany:${TRAIN_JOB}" \
  --export="${EXPORTS}" \
  "${SNAPSHOT_ROOT}/scripts/submit_origin_fold9_ablation_outer_release.sh" | cut -d';' -f1)"
AGGREGATE_JOB="$(sbatch --parsable --dependency="afterany:${RELEASE_JOB}" \
  --export="${EXPORTS}" \
  "${SNAPSHOT_ROOT}/scripts/submit_origin_fold9_ablation_aggregate.sh" | cut -d';' -f1)"

ORIGIN_SUBMISSION_OUTPUT="${EXPERIMENT_ROOT}/SUBMISSION.json" \
ORIGIN_SUBMISSION_TAG="${TAG}" \
ORIGIN_SUBMISSION_GROUP="${GROUP}" \
ORIGIN_SUBMISSION_COMMIT="${LAUNCH_COMMIT}" \
ORIGIN_SUBMISSION_ROOT="${EXPERIMENT_ROOT}" \
ORIGIN_SUBMISSION_PROTOCOL="${PROTOCOL}" \
ORIGIN_SUBMISSION_PREFLIGHT="${PREFLIGHT_JOB}" \
ORIGIN_SUBMISSION_TRAIN="${TRAIN_JOB}" \
ORIGIN_SUBMISSION_RELEASE="${RELEASE_JOB}" \
ORIGIN_SUBMISSION_AGGREGATE="${AGGREGATE_JOB}" \
PYTHONPATH="${SNAPSHOT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}" \
python - <<'PY'
import os
from datetime import datetime, timezone
from scripts.origin_fold9_ablation_common import (
    canonical_sha256, file_sha256, variants_for_group, write_json_atomic,
)
p = {
    "schema": "origin-fold9-ablation-submission-v1",
    "created_at_utc": datetime.now(timezone.utc).isoformat(),
    "tag": os.environ["ORIGIN_SUBMISSION_TAG"],
    "ablation_group": os.environ["ORIGIN_SUBMISSION_GROUP"],
    "selected_variants": list(variants_for_group(os.environ["ORIGIN_SUBMISSION_GROUP"])),
    "launch_commit": os.environ["ORIGIN_SUBMISSION_COMMIT"],
    "experiment_root": os.environ["ORIGIN_SUBMISSION_ROOT"],
    "protocol": os.environ["ORIGIN_SUBMISSION_PROTOCOL"],
    "protocol_sha256": file_sha256(os.environ["ORIGIN_SUBMISSION_PROTOCOL"]),
    "jobs": {
        "preflight": os.environ["ORIGIN_SUBMISSION_PREFLIGHT"],
        "validation_training_array": os.environ["ORIGIN_SUBMISSION_TRAIN"],
        "single_outer_release": os.environ["ORIGIN_SUBMISSION_RELEASE"],
        "afterany_aggregate": os.environ["ORIGIN_SUBMISSION_AGGREGATE"],
    },
}
p["content_checksum_sha256"] = canonical_sha256(p)
write_json_atomic(os.environ["ORIGIN_SUBMISSION_OUTPUT"], p)
PY

echo "Locked ORIGIN fold-9 ablation submitted."
echo "  group:             ${GROUP} (${VARIANT_COUNT} variants)"
echo "  tag:               ${TAG}"
echo "  launch commit:     ${LAUNCH_COMMIT}"
echo "  preflight:         ${PREFLIGHT_JOB}"
echo "  validation array:  ${TRAIN_JOB}"
echo "  outer release:     ${RELEASE_JOB}"
echo "  aggregate/audit:   ${AGGREGATE_JOB}"
echo "  experiment root:   ${EXPERIMENT_ROOT}"
echo "Monitor: squeue -u \"${USER}\""
