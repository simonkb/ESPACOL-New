#!/bin/bash
# Full 2-dataset x fold x 3-seed x 5-arm training array. It is fail-closed
# until the APTOS fold-0 canary audit has passed.
#SBATCH --job-name=oa_full
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=3-00:00:00
#SBATCH --array=0-224%12
#SBATCH --output=/dpc/kuin0170/ESPACOL-New/origin_acceptance_logs/full_%A_%a.out
#SBATCH --error=/dpc/kuin0170/ESPACOL-New/origin_acceptance_logs/full_%A_%a.err
#SBATCH --account=kuin0170

set -euo pipefail
set +u
source /etc/profile.d/lmod.sh
module load miniconda/3
module load cuda/12.6
source activate "${ORIGIN_CONDA_ENV:-espacol}"
set -u

REPO_ROOT="${ORIGIN_REPO_ROOT:?ORIGIN_REPO_ROOT is required}"
EXPERIMENT_ROOT="${ORIGIN_ACCEPTANCE_ROOT:?ORIGIN_ACCEPTANCE_ROOT is required}"
LAUNCH_COMMIT="${ORIGIN_LAUNCH_COMMIT:?ORIGIN_LAUNCH_COMMIT is required}"
TASK_ID="${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}"
cd "${REPO_ROOT}"
[[ "$(git rev-parse HEAD)" == "${LAUNCH_COMMIT}" ]] || { echo "commit mismatch" >&2; exit 2; }
[[ -f "${EXPERIMENT_ROOT}/canary/CANARY_PASSED.json" ]] || { echo "canary gate missing" >&2; exit 3; }
[[ -z "$(git status --porcelain --untracked-files=no)" ]] || { echo "tracked worktree is dirty" >&2; exit 4; }
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

TASK_INFO="$(python -c 'import sys; from scripts.origin_acceptance_baseline_common import task_at; t=task_at("full", int(sys.argv[1])); print(t.key, t.dataset)' "${TASK_ID}")"
read -r TASK_KEY DATASET <<<"${TASK_INFO}"
if [[ "${DATASET}" == "aptos" ]]; then
  DATA_ROOT="${ORIGIN_APTOS_ROOT:?ORIGIN_APTOS_ROOT is required}"
  N_FOLDS=5
  EPOCHS="${ORIGIN_APTOS_EPOCHS:-35}"
else
  DATA_ROOT="${ORIGIN_DR_ROOT:?ORIGIN_DR_ROOT is required}"
  N_FOLDS=10
  EPOCHS="${ORIGIN_DR_EPOCHS:-75}"
fi
RUN_DIR="${EXPERIMENT_ROOT}/full/training/${TASK_KEY}"
mkdir -p "${EXPERIMENT_ROOT}/locks" "${RUN_DIR}"
exec 9>"${EXPERIMENT_ROOT}/locks/full_${TASK_ID}.lock"
flock -n 9 || { echo "full task already owned" >&2; exit 5; }
RESUME_ARGS=()
if [[ "${ORIGIN_ACCEPTANCE_RESUME:-0}" == "1" ]]; then RESUME_ARGS+=(--resume); fi

python train_origin_acceptance_baseline.py \
  --protocol_scope full --task_index "${TASK_ID}" \
  --data_root "${DATA_ROOT}" --run_dir "${RUN_DIR}" \
  --seed 42 --n_folds "${N_FOLDS}" --skip_test \
  --image_size "${ORIGIN_IMAGE_SIZE:-640}" --scales s4,s8,s16,s32 \
  --projection_dim 128 --reference_count 4096 --atom_rate_init 1e-6 \
  --prior_rate_init 1e-4 --boundary_scale_init 1.0 --atom_mode cumulative \
  --decision_rule class_map --batch_size "${ORIGIN_BATCH_SIZE:-8}" \
  --epochs "${EPOCHS}" --num_workers "${ORIGIN_NUM_WORKERS:-8}" \
  --lr 1e-4 --head_lr 5e-4 --weight_decay 1e-5 --freeze_encoder_epochs 2 \
  --scheduler plateau --lr_factor 0.2 --lr_patience 5 \
  --early_stopping_patience 12 --checkpoint_selection acc_then_qwk \
  --selection_qwk_weight 0.1 --rps_weight 0.25 --evidence_budget_weight 0.0 \
  --class_weighting none --amp --amp_init_scale 4096 --amp_unfreeze_scale 256 \
  --amp_growth_interval 2000 --amp_max_consecutive_skips 8 "${RESUME_ARGS[@]}"
