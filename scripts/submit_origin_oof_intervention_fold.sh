#!/bin/bash
# Read-only grouped delete/retain census for one completed ORIGIN-v3 outer fold.
# Submit once per fold, setting ORIGIN_DATASET and ORIGIN_FOLD.
#SBATCH --job-name=origin_oof_int
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH --output=/dpc/kuin0170/ESPACOL-New/origin_oof_intervention_logs/fold_%j.out
#SBATCH --error=/dpc/kuin0170/ESPACOL-New/origin_oof_intervention_logs/fold_%j.err
#SBATCH --account=kuin0170

set -euo pipefail
set +u
source /etc/profile.d/lmod.sh || exit 1
module load miniconda/3 || exit 1
module load cuda/12.6 || exit 1
source activate "${ORIGIN_CONDA_ENV:-G}" || exit 1
set -u

REPO_ROOT="${ORIGIN_REPO_ROOT:?ORIGIN_REPO_ROOT is required}"
DATASET="${ORIGIN_DATASET:?ORIGIN_DATASET must be dr or aptos}"
FOLD="${ORIGIN_FOLD:-${SLURM_ARRAY_TASK_ID:-}}"
[[ -n "${FOLD}" ]] || { echo "ORIGIN_FOLD or SLURM_ARRAY_TASK_ID is required." >&2; exit 2; }
CV_ROOT="${ORIGIN_V3_CV_ROOT:?ORIGIN_V3_CV_ROOT is required}"
DATA_ROOT="${ORIGIN_DATA_ROOT:?ORIGIN_DATA_ROOT is required}"
CV_PROTOCOL="${ORIGIN_V3_PROTOCOL:?ORIGIN_V3_PROTOCOL is required}"
LAUNCH_COMMIT="${ORIGIN_LAUNCH_COMMIT:?ORIGIN_LAUNCH_COMMIT is required}"
AUDIT_PROTOCOL="${ORIGIN_OOF_AUDIT_PROTOCOL:-${REPO_ROOT}/scripts/protocols/origin_oof_intervention_protocol.json}"

[[ "${DATASET}" == "dr" || "${DATASET}" == "aptos" ]] || {
  echo "Unsupported dataset: ${DATASET}." >&2; exit 2;
}
[[ "${FOLD}" =~ ^[0-9]+$ ]] || { echo "Invalid fold: ${FOLD}." >&2; exit 2; }
if [[ "${DATASET}" == "dr" ]]; then
  (( FOLD < 10 )) || { echo "EyePACS fold must be in 0..9." >&2; exit 2; }
else
  (( FOLD < 5 )) || { echo "APTOS fold must be in 0..4." >&2; exit 2; }
fi

cd "${REPO_ROOT}"
[[ "$(git rev-parse HEAD)" == "${LAUNCH_COMMIT}" ]] || {
  echo "Immutable launch commit mismatch." >&2; exit 3;
}
[[ -z "$(git status --porcelain --untracked-files=no)" ]] || {
  echo "Tracked files changed in immutable audit worktree." >&2; exit 4;
}
FOLD_DIR="${CV_ROOT}/workers/fold${FOLD}/fold${FOLD}"
OUTPUT_DIR="${FOLD_DIR}/oof_intervention_audit"
[[ -f "${FOLD_DIR}/CV_FOLD_COMPLETE.json" ]] || {
  echo "Fold completion marker is missing: ${FOLD_DIR}." >&2; exit 5;
}
[[ -f "${AUDIT_PROTOCOL}" ]] || { echo "Audit protocol is missing." >&2; exit 5; }
[[ ! -f "${OUTPUT_DIR}/audit_manifest.json" ]] || {
  echo "Completed audit already exists; refusing overwrite." >&2; exit 6;
}

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
echo "=== ORIGIN exact OOF intervention census: ${DATASET} fold ${FOLD} ==="
date --iso-8601=seconds
python scripts/audit_origin_oof_interventions.py \
  --dataset "${DATASET}" --fold "${FOLD}" \
  --cv-root "${CV_ROOT}" --data-root "${DATA_ROOT}" \
  --cv-protocol "${CV_PROTOCOL}" --audit-protocol "${AUDIT_PROTOCOL}" \
  --output-dir "${OUTPUT_DIR}" --batch-size 2 --num-workers 8 --device cuda

echo "Fold audit manifest: ${OUTPUT_DIR}/audit_manifest.json"
