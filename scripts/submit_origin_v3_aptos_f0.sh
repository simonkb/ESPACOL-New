#!/bin/bash
# ORIGIN-v3 bounded-rate development gate: APTOS fold 0, inner validation
# only. The outer fold is constructed for integrity checks but never evaluated.
#SBATCH --job-name=origin_v3_a0
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH --output=/dpc/kuin0170/ESPACOL-New/origin_v3_aptos_f0_%j.out
#SBATCH --error=/dpc/kuin0170/ESPACOL-New/origin_v3_aptos_f0_%j.err
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
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

RUN_DIR="${ORIGIN_RUN_DIR:-runs/origin_aptos_f0_v3_bounded}"
DATA_ROOT="${ORIGIN_APTOS_ROOT:-Datasets/aptos2019-blindness-detection}"
FOLD_DIR="${RUN_DIR}/fold0"
mkdir -p "${FOLD_DIR}"
exec 9>"${FOLD_DIR}/.writer.lock"
flock -n 9 || { echo "Another process is writing ${FOLD_DIR}." >&2; exit 2; }

RESUME_ARGS=()
if [[ "${ORIGIN_RESUME:-0}" == "1" ]]; then
  [[ -f "${FOLD_DIR}/last.pth" && ! -f "${FOLD_DIR}/result.json" ]] || {
    echo "Cannot resume missing or completed ORIGIN-v3 run: ${FOLD_DIR}." >&2
    exit 3
  }
  RESUME_ARGS+=(--resume)
elif [[ -e "${FOLD_DIR}/last.pth" || -e "${FOLD_DIR}/history.csv" ]]; then
  echo "Fresh ORIGIN-v3 run refused: artifacts exist in ${FOLD_DIR}." >&2
  exit 4
fi

echo "=== ORIGIN-v3 bounded-rate APTOS fold 0 / inner validation only ==="
date --iso-8601=seconds
git rev-parse HEAD
git status --short
python --version
python -m pytest -q tests/test_origin*.py

ORIGIN_DATA_ROOT="${DATA_ROOT}" python - <<'PY'
import os
import torch
from Datasets.origin_data import aptos_fold, class_histogram, load_aptos_items, split_paths_are_disjoint
items = load_aptos_items(os.environ["ORIGIN_DATA_ROOT"])
train, validation, locked_test = aptos_fold(items, 0, n_folds=5, seed=42)
if len(items) != 3662 or (len(train), len(validation), len(locked_test)) != (2636, 293, 733):
    raise RuntimeError("APTOS fold-0 identity changed")
if class_histogram(train) != [1300, 266, 719, 139, 212]:
    raise RuntimeError("APTOS training histogram changed")
if class_histogram(validation) != [144, 30, 80, 15, 24]:
    raise RuntimeError("APTOS validation histogram changed")
if not split_paths_are_disjoint(train, validation, locked_test):
    raise RuntimeError("APTOS split contains duplicate paths")
if not torch.cuda.is_available():
    raise RuntimeError("CUDA is required")
print("gpu", torch.cuda.get_device_name(0))
print("aptos", len(items), class_histogram(items))
print("fold0", len(train), len(validation), len(locked_test))
PY

TRAIN_ARGS=(
  --dataset aptos --data_root "${DATA_ROOT}" --run_dir "${RUN_DIR}" --folds 0 --seed 42
  --image_size "${ORIGIN_IMAGE_SIZE:-640}" --encoder convnext_tiny
  --scales "${ORIGIN_SCALES:-s4,s8,s16,s32}"
  --projection_dim "${ORIGIN_PROJECTION_DIM:-128}"
  --reference_count "${ORIGIN_REFERENCE_COUNT:-4096}"
  --atom_rate_init "${ORIGIN_ATOM_RATE_INIT:-1e-6}"
  --prior_rate_init "${ORIGIN_PRIOR_RATE_INIT:-1e-4}"
  --boundary_scale_init "${ORIGIN_BOUNDARY_SCALE_INIT:-1.0}"
  --total_rate_cap "${ORIGIN_TOTAL_RATE_CAP:-64.0}"
  --prior_rate_cap "${ORIGIN_PRIOR_RATE_CAP:-1.0}"
  --boundary_scale_cap "${ORIGIN_BOUNDARY_SCALE_CAP:-2.0}"
  --rate_roundoff_margin "${ORIGIN_RATE_ROUNDOFF_MARGIN:-1.0}"
  --atom_mode "${ORIGIN_ATOM_MODE:-cumulative}" --decision_rule class_map
  --batch_size "${ORIGIN_BATCH_SIZE:-8}" --epochs "${ORIGIN_EPOCHS:-35}"
  --num_workers "${ORIGIN_NUM_WORKERS:-8}"
  --lr "${ORIGIN_ENCODER_LR:-1e-4}" --head_lr "${ORIGIN_HEAD_LR:-5e-4}"
  --weight_decay "${ORIGIN_WEIGHT_DECAY:-1e-5}"
  --scheduler plateau --lr_factor 0.2 --lr_patience 5 --early_stopping_patience 12
  --rps_weight "${ORIGIN_RPS_WEIGHT:-0.25}"
  --evidence_budget_weight "${ORIGIN_BUDGET_WEIGHT:-0.0}"
  --class_weighting none --amp_unfreeze_scale "${ORIGIN_AMP_UNFREEZE_SCALE:-256}"
  --skip_test
)
if (( ${#RESUME_ARGS[@]} )); then TRAIN_ARGS+=("${RESUME_ARGS[@]}"); fi
printf 'training_command:'; printf ' %q' python train_origin.py "${TRAIN_ARGS[@]}"; printf '\n'
python train_origin.py "${TRAIN_ARGS[@]}"
