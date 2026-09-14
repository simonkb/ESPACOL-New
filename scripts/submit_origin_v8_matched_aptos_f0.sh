#!/bin/bash
# Three-way, parameter-matched ORIGIN-v8 APTOS fold-0 development experiment.
# Array index is the only source of the registered relation-variant change.
# No outer-test evaluation and no full cross-validation are authorized here.
#SBATCH --job-name=origin_v8_a0
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH --array=0-2
#SBATCH --output=/dpc/kuin0170/ESPACOL-New/origin_v8_aptos_f0_%A_%a.out
#SBATCH --error=/dpc/kuin0170/ESPACOL-New/origin_v8_aptos_f0_%A_%a.err
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

: "${ORIGIN_EXPECTED_COMMIT:?Set ORIGIN_EXPECTED_COMMIT to the preflighted git commit}"
: "${ORIGIN_V3_CHECKPOINT_SHA256:?Set the SHA-256 of the immutable APTOS v3 best.pth}"
OBSERVED_COMMIT="$(git rev-parse HEAD)"
[[ "${OBSERVED_COMMIT}" == "${ORIGIN_EXPECTED_COMMIT}" ]] || {
  echo "Commit mismatch: observed ${OBSERVED_COMMIT}, expected ${ORIGIN_EXPECTED_COMMIT}." >&2
  exit 2
}

VARIANTS=(
  identified_conserved_witness_v1
  additive_conserved_witness_control_v1
  shuffled_conserved_witness_control_v1
)
RUN_NAMES=(
  origin_aptos_f0_v8_cmwa_target
  origin_aptos_f0_v8_cmwa_endpoint_control
  origin_aptos_f0_v8_cmwa_shuffled_control
)
CONTROL_SEMANTICS=(
  identified_nonadditive_spatial_pair_witnesses
  endpoint_main_effect_capacity_null
  fixed_shuffled_endpoint_to_geometry_correspondence_null
)

TASK_ID="${SLURM_ARRAY_TASK_ID:-}"
[[ "${TASK_ID}" =~ ^[0-2]$ ]] || {
  echo "This script must run as the registered Slurm array 0-2." >&2
  exit 3
}
RELATION_VARIANT="${VARIANTS[${TASK_ID}]}"
RUN_NAME="${RUN_NAMES[${TASK_ID}]}"
CONTROL_SEMANTIC="${CONTROL_SEMANTICS[${TASK_ID}]}"

RUN_ROOT="${ORIGIN_V8_RUN_ROOT:-runs}"
RUN_DIR="${RUN_ROOT}/${RUN_NAME}"
DATA_ROOT="${ORIGIN_APTOS_ROOT:-Datasets/aptos2019-blindness-detection}"
SOURCE_CHECKPOINT="${ORIGIN_V3_CHECKPOINT:-runs/origin_aptos_f0_v3_bounded/fold0/best.pth}"
FOLD_DIR="${RUN_DIR}/fold0"

[[ -f "${SOURCE_CHECKPOINT}" ]] || {
  echo "Missing immutable v3 source checkpoint: ${SOURCE_CHECKPOINT}" >&2
  exit 4
}
OBSERVED_SOURCE_SHA="$(sha256sum "${SOURCE_CHECKPOINT}" | cut -d' ' -f1)"
[[ "${OBSERVED_SOURCE_SHA}" == "${ORIGIN_V3_CHECKPOINT_SHA256}" ]] || {
  echo "v3 source checkpoint SHA-256 mismatch." >&2
  exit 5
}

mkdir -p "${FOLD_DIR}"
exec 9>"${FOLD_DIR}/.writer.lock"
flock -n 9 || { echo "Another process is writing ${FOLD_DIR}." >&2; exit 6; }

RESUME_ARGS=()
if [[ "${ORIGIN_RESUME:-0}" == "1" ]]; then
  [[ -f "${FOLD_DIR}/last.pth" && ! -f "${FOLD_DIR}/result.json" ]] || {
    echo "Cannot resume missing or completed ORIGIN-v8 run: ${FOLD_DIR}." >&2
    exit 7
  }
  RESUME_ARGS+=(--resume)
else
  for artifact in best.pth best_learned.pth last.pth history.csv result.json; do
    [[ ! -e "${FOLD_DIR}/${artifact}" ]] || {
      echo "Fresh ORIGIN-v8 run refused: ${FOLD_DIR}/${artifact} exists." >&2
      exit 8
    }
  done
fi

echo "=== ORIGIN-v8 conserved-witness matched APTOS fold 0 (${RELATION_VARIANT}) ==="
echo "evaluation_scope=inner_validation_only"
echo "control_semantics=${CONTROL_SEMANTIC}"
echo "outer_test_authorized=false"
echo "full_cross_validation_authorized=false"
date --iso-8601=seconds
git rev-parse HEAD
git status --short
python --version

# Every array member receives this exact argument vector except the registered
# relation variant and its output directory. The v3 base remains frozen for
# the entire fixed 15-epoch budget.
TRAIN_ARGS=(
  --dataset aptos --data_root "${DATA_ROOT}" --run_dir "${RUN_DIR}" --folds 0 --seed 42
  --image_size 640 --encoder convnext_tiny --scales s4,s8,s16,s32
  --projection_dim 128 --reference_count 4096 --atom_rate_init 1e-6
  --prior_rate_init 1e-4 --boundary_scale_init 1.0
  --total_rate_cap 64.0 --prior_rate_cap 1.0 --boundary_scale_cap 2.0
  --rate_roundoff_margin 1.0 --atom_mode cumulative --decision_rule class_map
  --relation_enabled --relation_source_scale s8 --relation_grid_size 10
  --relation_dim 64 --relation_head_dim 16 --relation_delta_cap 2.0
  --relation_variant "${RELATION_VARIANT}" --relation_edge_budget 8
  --relation_allocation_temperature 1.0 --relation_permutation_seed 617
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

[[ -f "${FOLD_DIR}/best_learned.pth" ]] || {
  echo "Completed run did not produce best_learned.pth." >&2
  exit 9
}
[[ -f "${FOLD_DIR}/result.json" ]] || {
  echo "Completed run did not produce result.json." >&2
  exit 10
}
