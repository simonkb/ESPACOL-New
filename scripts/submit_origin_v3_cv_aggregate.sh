#!/bin/bash
# Aggregate one frozen ORIGIN-v3 dataset only after its complete Slurm array succeeds.
#SBATCH --job-name=ov3_cv_sum
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=0-12:00:00
#SBATCH --output=/dpc/kuin0170/ESPACOL-New/origin_v3_cv_logs/aggregate_%j.out
#SBATCH --error=/dpc/kuin0170/ESPACOL-New/origin_v3_cv_logs/aggregate_%j.err
#SBATCH --account=kuin0170

set -euo pipefail
set +u
source /etc/profile.d/lmod.sh || exit 1
module load miniconda/3 || exit 1
module load cuda/12.6 || exit 1
source activate "${ORIGIN_CONDA_ENV:-G}" || exit 1
set -u

REPO_ROOT="${ORIGIN_REPO_ROOT:?ORIGIN_REPO_ROOT is required}"
DATASET="${ORIGIN_DATASET:?ORIGIN_DATASET is required}"
CV_ROOT="${ORIGIN_V3_CV_ROOT:?ORIGIN_V3_CV_ROOT is required}"
DATA_ROOT="${ORIGIN_DATA_ROOT:?ORIGIN_DATA_ROOT is required}"
PROTOCOL="${ORIGIN_V3_PROTOCOL:?ORIGIN_V3_PROTOCOL is required}"
LAUNCH_COMMIT="${ORIGIN_LAUNCH_COMMIT:?ORIGIN_LAUNCH_COMMIT is required}"
[[ "${DATASET}" == "aptos" || "${DATASET}" == "dr" ]] || {
  echo "Unsupported aggregation dataset: ${DATASET}." >&2; exit 2;
}

cd "${REPO_ROOT}"
[[ "$(git rev-parse HEAD)" == "${LAUNCH_COMMIT}" ]] || {
  echo "Immutable launch commit mismatch." >&2; exit 3;
}
[[ -z "$(git status --porcelain --untracked-files=no)" ]] || {
  echo "Tracked files changed in the immutable worktree." >&2; exit 4;
}
[[ ! -e "${CV_ROOT}/CV_SUMMARY.json" ]] || {
  echo "Refusing to overwrite existing final CV summary." >&2; exit 5;
}

echo "=== Frozen ORIGIN-v3 ${DATASET} full-CV aggregation ==="
date --iso-8601=seconds
python scripts/aggregate_origin_v3_cv.py \
  --dataset "${DATASET}" --cv-root "${CV_ROOT}" \
  --data-root "${DATA_ROOT}" --protocol "${PROTOCOL}"
echo "Summary: ${CV_ROOT}/CV_SUMMARY.json"
