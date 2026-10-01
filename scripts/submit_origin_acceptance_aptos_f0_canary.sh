#!/bin/bash
# Four-way APTOS fold-0 canary. This is the only training stage allowed before
# its audit writes CANARY_PASSED.json.
#SBATCH --job-name=oa_canary
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --array=0-3%4
#SBATCH --output=/dpc/kuin0170/ESPACOL-New/origin_acceptance_logs/canary_%A_%a.out
#SBATCH --error=/dpc/kuin0170/ESPACOL-New/origin_acceptance_logs/canary_%A_%a.err
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
DATA_ROOT="${ORIGIN_APTOS_ROOT:?ORIGIN_APTOS_ROOT is required}"
LAUNCH_COMMIT="${ORIGIN_LAUNCH_COMMIT:?ORIGIN_LAUNCH_COMMIT is required}"
TASK_ID="${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}"
cd "${REPO_ROOT}"
[[ "$(git rev-parse HEAD)" == "${LAUNCH_COMMIT}" ]] || { echo "commit mismatch" >&2; exit 2; }
[[ -z "$(git status --porcelain --untracked-files=no)" ]] || { echo "tracked worktree is dirty" >&2; exit 3; }
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

TASK_KEY="$(python -c 'import sys; from scripts.origin_acceptance_baseline_common import task_at; print(task_at("canary", int(sys.argv[1])).key)' "${TASK_ID}")"
RUN_DIR="${EXPERIMENT_ROOT}/canary/training/${TASK_KEY}"
mkdir -p "${EXPERIMENT_ROOT}/locks" "${RUN_DIR}"
exec 9>"${EXPERIMENT_ROOT}/locks/canary_${TASK_ID}.lock"
flock -n 9 || { echo "canary task already owned" >&2; exit 4; }

RESUME_ARGS=()
if [[ "${ORIGIN_ACCEPTANCE_RESUME:-0}" == "1" ]]; then RESUME_ARGS+=(--resume); fi
python -m pytest -q tests/test_origin_acceptance_baselines.py
python train_origin_acceptance_baseline.py \
  --protocol_scope canary --task_index "${TASK_ID}" \
  --data_root "${DATA_ROOT}" --run_dir "${RUN_DIR}" \
  --seed 42 --n_folds 5 --skip_test \
  --image_size "${ORIGIN_IMAGE_SIZE:-640}" --scales s4,s8,s16,s32 \
  --projection_dim 128 --reference_count 4096 --atom_rate_init 1e-6 \
  --prior_rate_init 1e-4 --boundary_scale_init 1.0 --atom_mode cumulative \
  --decision_rule class_map --batch_size "${ORIGIN_BATCH_SIZE:-8}" \
  --epochs "${ORIGIN_APTOS_EPOCHS:-35}" --num_workers "${ORIGIN_NUM_WORKERS:-8}" \
  --lr 1e-4 --head_lr 5e-4 --weight_decay 1e-5 --freeze_encoder_epochs 2 \
  --scheduler plateau --lr_factor 0.2 --lr_patience 5 \
  --early_stopping_patience 12 --checkpoint_selection acc_then_qwk \
  --selection_qwk_weight 0.1 --rps_weight 0.25 --evidence_budget_weight 0.0 \
  --class_weighting none --amp --amp_init_scale 4096 --amp_unfreeze_scale 256 \
  --amp_growth_interval 2000 --amp_max_consecutive_skips 8 "${RESUME_ARGS[@]}"
