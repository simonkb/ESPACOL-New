#!/bin/bash
# Run after one or more OOF intervention aggregate jobs have completed.
# Submit with --dependency=afterok:<aggregate-job>[:<aggregate-job>...], or run
# independently when every referenced aggregate manifest already exists.
#SBATCH --job-name=origin_oof_stats
#SBATCH --partition=prod
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=0-12:00:00
#SBATCH --output=/dpc/kuin0170/ESPACOL-New/origin_oof_intervention_logs/statistics_%j.out
#SBATCH --error=/dpc/kuin0170/ESPACOL-New/origin_oof_intervention_logs/statistics_%j.err
#SBATCH --account=kuin0170

set -euo pipefail
set +u
source /etc/profile.d/lmod.sh || exit 1
module load miniconda/3 || exit 1
source activate "${ORIGIN_CONDA_ENV:-G}" || exit 1
set -u

REPO_ROOT="${ORIGIN_REPO_ROOT:?ORIGIN_REPO_ROOT is required}"
LAUNCH_COMMIT="${ORIGIN_LAUNCH_COMMIT:?ORIGIN_LAUNCH_COMMIT is required}"
MANIFEST_SPEC="${ORIGIN_OOF_AGGREGATE_MANIFESTS:?comma-separated DATASET=PATH aggregate manifests are required}"
OUTPUT_DIR="${ORIGIN_OOF_STATISTICS_OUTPUT:?ORIGIN_OOF_STATISTICS_OUTPUT is required}"
PROTOCOL="${ORIGIN_OOF_STATISTICS_PROTOCOL:-${REPO_ROOT}/scripts/protocols/origin_oof_statistics_protocol.json}"

cd "${REPO_ROOT}"
[[ "$(git rev-parse HEAD)" == "${LAUNCH_COMMIT}" ]] || {
  echo "Immutable launch commit mismatch." >&2; exit 2;
}
[[ -z "$(git status --porcelain --untracked-files=no)" ]] || {
  echo "Tracked files changed in immutable statistics worktree." >&2; exit 3;
}
[[ ! -f "${OUTPUT_DIR}/statistics_manifest.json" ]] || {
  echo "Completed statistics manifest already exists; refusing overwrite." >&2; exit 4;
}

IFS=',' read -r -a MANIFESTS <<< "${MANIFEST_SPEC}"
ARGS=()
for specification in "${MANIFESTS[@]}"; do
  [[ "${specification}" == *=* ]] || {
    echo "Invalid manifest specification: ${specification}" >&2; exit 5;
  }
  path="${specification#*=}"
  [[ -f "${path}" ]] || {
    echo "Aggregate manifest does not exist: ${path}" >&2; exit 6;
  }
  ARGS+=(--aggregate-manifest "${specification}")
done

echo "=== ORIGIN complete OOF proper-score, calibration, and intervention statistics ==="
date --iso-8601=seconds
python scripts/analyze_origin_oof_statistics.py \
  "${ARGS[@]}" --protocol "${PROTOCOL}" --output-dir "${OUTPUT_DIR}"

echo "Statistics manifest: ${OUTPUT_DIR}/statistics_manifest.json"
