#!/bin/bash
# ORIGIN-v6 explicit regional-relation pilot: hash-bound v3 APTOS fold 0
# warm start, epoch-0 no-regression floor, and relation-only optimization.
#SBATCH --job-name=origin_v6_ra0
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH --output=/dpc/kuin0170/ESPACOL-New/origin_v6_relation_aptos_f0_%j.out
#SBATCH --error=/dpc/kuin0170/ESPACOL-New/origin_v6_relation_aptos_f0_%j.err
#SBATCH --account=kuin0170

set -eo pipefail
set +u
source /etc/profile.d/lmod.sh || exit 1
module load miniconda/3 || exit 1
module load cuda/12.6 || exit 1
source activate "${ORIGIN_CONDA_ENV:-G}" || exit 1
set -u

REPO_ROOT="${ORIGIN_REPO_ROOT:-/dpc/kuin0170/ESPACOL-New}"
cd "${REPO_ROOT}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

RUN_DIR="${ORIGIN_RUN_DIR:-runs/origin_aptos_f0_v6_relation_only}"
DATA_ROOT="${ORIGIN_APTOS_ROOT:-Datasets/aptos2019-blindness-detection}"
SOURCE_CHECKPOINT="${ORIGIN_V3_CHECKPOINT:-runs/origin_aptos_f0_v3_bounded/fold0/best.pth}"
: "${ORIGIN_V3_CHECKPOINT_SHA256:?Set this to: sha256sum ${SOURCE_CHECKPOINT} | cut -d' ' -f1}"
FOLD_DIR="${RUN_DIR}/fold0"
mkdir -p "${FOLD_DIR}"
exec 9>"${FOLD_DIR}/.writer.lock"
flock -n 9 || { echo "Another process is writing ${FOLD_DIR}." >&2; exit 2; }

RESUME_ARGS=()
if [[ "${ORIGIN_RESUME:-0}" == "1" ]]; then
  [[ -f "${FOLD_DIR}/last.pth" && ! -f "${FOLD_DIR}/result.json" ]] || {
    echo "Cannot resume missing or completed ORIGIN-v6 run: ${FOLD_DIR}." >&2
    exit 3
  }
  RESUME_ARGS+=(--resume)
elif [[ -e "${FOLD_DIR}/last.pth" || -e "${FOLD_DIR}/history.csv" ]]; then
  echo "Fresh ORIGIN-v6 run refused: artifacts exist in ${FOLD_DIR}." >&2
  exit 4
fi

echo "=== ORIGIN-v6 relation-only APTOS fold 0 / inner validation only ==="
echo "controlled_change=bounded_cumulative_pair_rate_log_odds;v3_base=frozen"
date --iso-8601=seconds
git rev-parse HEAD
git status --short
python --version
python -m pytest -q tests/test_origin*.py

TRAIN_ARGS=(
  --dataset aptos --data_root "${DATA_ROOT}" --run_dir "${RUN_DIR}" --folds 0 --seed 42
  --image_size 640 --encoder convnext_tiny --scales s4,s8,s16,s32
  --projection_dim 128 --reference_count 4096 --atom_rate_init 1e-6
  --prior_rate_init 1e-4 --boundary_scale_init 1.0
  --total_rate_cap 64.0 --prior_rate_cap 1.0 --boundary_scale_cap 2.0
  --rate_roundoff_margin 1.0 --atom_mode cumulative --decision_rule class_map
  --relation_enabled --relation_source_scale s8 --relation_grid_size 10
  --relation_dim 64 --relation_head_dim 16 --relation_delta_cap 2.0
  --warm_start_checkpoint "${SOURCE_CHECKPOINT}"
  --warm_start_sha256 "${ORIGIN_V3_CHECKPOINT_SHA256}"
  --warm_start_metric_floor_tolerance 1e-6
  --relation_only_epochs 15
  --batch_size 8 --epochs 15 --num_workers "${ORIGIN_NUM_WORKERS:-8}"
  --lr 1e-4 --head_lr 5e-4 --weight_decay 1e-5 --freeze_encoder_epochs 2
  --scheduler plateau --lr_factor 0.2 --lr_patience 5
  --early_stopping_patience 15 --rps_weight 0.25
  --evidence_budget_weight 0.0 --class_weighting none
  --amp_unfreeze_scale 256 --skip_test
)
if (( ${#RESUME_ARGS[@]} )); then TRAIN_ARGS+=("${RESUME_ARGS[@]}"); fi
printf 'training_command:'; printf ' %q' python train_origin.py "${TRAIN_ARGS[@]}"; printf '\n'
python train_origin.py "${TRAIN_ARGS[@]}"
