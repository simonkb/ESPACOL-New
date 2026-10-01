#!/bin/bash
# Aggregate only after every per-fold exact intervention census succeeds.
#SBATCH --job-name=origin_oof_sum
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=0-12:00:00
#SBATCH --output=/dpc/kuin0170/ESPACOL-New/origin_oof_intervention_logs/aggregate_%j.out
#SBATCH --error=/dpc/kuin0170/ESPACOL-New/origin_oof_intervention_logs/aggregate_%j.err
#SBATCH --account=kuin0170

set -euo pipefail
set +u
source /etc/profile.d/lmod.sh || exit 1
module load miniconda/3 || exit 1
source activate "${ORIGIN_CONDA_ENV:-G}" || exit 1
set -u

REPO_ROOT="${ORIGIN_REPO_ROOT:?ORIGIN_REPO_ROOT is required}"
DATASET="${ORIGIN_DATASET:?ORIGIN_DATASET must be dr or aptos}"
CV_ROOT="${ORIGIN_V3_CV_ROOT:?ORIGIN_V3_CV_ROOT is required}"
CV_PROTOCOL="${ORIGIN_V3_PROTOCOL:?ORIGIN_V3_PROTOCOL is required}"
LAUNCH_COMMIT="${ORIGIN_LAUNCH_COMMIT:?ORIGIN_LAUNCH_COMMIT is required}"
AUDIT_PROTOCOL="${ORIGIN_OOF_AUDIT_PROTOCOL:-${REPO_ROOT}/scripts/protocols/origin_oof_intervention_protocol.json}"
OUTPUT_DIR="${CV_ROOT}/oof_intervention_aggregate"

[[ "${DATASET}" == "dr" || "${DATASET}" == "aptos" ]] || {
  echo "Unsupported dataset: ${DATASET}." >&2; exit 2;
}
cd "${REPO_ROOT}"
[[ "$(git rev-parse HEAD)" == "${LAUNCH_COMMIT}" ]] || {
  echo "Immutable launch commit mismatch." >&2; exit 3;
}
[[ -z "$(git status --porcelain --untracked-files=no)" ]] || {
  echo "Tracked files changed in immutable audit worktree." >&2; exit 4;
}
[[ ! -f "${OUTPUT_DIR}/audit_manifest.json" ]] || {
  echo "Completed aggregate already exists; refusing overwrite." >&2; exit 5;
}

echo "=== ORIGIN complete OOF intervention aggregation: ${DATASET} ==="
date --iso-8601=seconds
python scripts/aggregate_origin_oof_interventions.py \
  --dataset "${DATASET}" --cv-root "${CV_ROOT}" \
  --cv-protocol "${CV_PROTOCOL}" --audit-protocol "${AUDIT_PROTOCOL}" \
  --output-dir "${OUTPUT_DIR}"

echo "Aggregate audit manifest: ${OUTPUT_DIR}/audit_manifest.json"
