#!/bin/bash
# First validation-only EyePACS/Kaggle-DR MOSAIC run: fold 0, up to epoch 75.
# Both eyes of every patient remain in one split and the outer test is locked.
#SBATCH --job-name=mosaic_dr_f0_e75
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --output=/dpc/kuin0170/ESPACOL-New/mosaic_dr_f0_e75_%j.out
#SBATCH --error=/dpc/kuin0170/ESPACOL-New/mosaic_dr_f0_e75_%j.err
#SBATCH --account=kuin0170

source /etc/profile.d/lmod.sh || exit 1
module load miniconda/3 || exit 1
module load cuda/12.6 || exit 1
source activate G || exit 1

set -euo pipefail
cd /dpc/kuin0170/ESPACOL-New

export HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

RUN_DIR="${MOSAIC_RUN_DIR:-runs/mosaic_dr_fold0_e75}"
FOLD=0
TOTAL_EPOCHS=75
FOLD_DIR="${RUN_DIR}/fold${FOLD}"
mkdir -p "${FOLD_DIR}"

exec 9>"${FOLD_DIR}/.writer.lock"
if ! flock -n 9; then
  echo "Another process is already writing ${FOLD_DIR}." >&2
  exit 2
fi
RESUME_ARGS=()
if [[ "${MOSAIC_RESUME:-0}" == "1" ]]; then
  if [[ ! -f "${FOLD_DIR}/last.pth" ]]; then
    echo "Cannot resume: ${FOLD_DIR}/last.pth does not exist." >&2
    exit 3
  fi
  RESUME_ARGS+=(--resume)
elif [[ -e "${FOLD_DIR}/last.pth" || -e "${FOLD_DIR}/history.csv" ]]; then
  echo "Fresh run refused because ${FOLD_DIR} already contains training artifacts." >&2
  echo "Choose a new MOSAIC_RUN_DIR or set MOSAIC_RESUME=1 deliberately." >&2
  exit 4
fi

echo "=== MOSAIC EyePACS fold 0 / 75 epochs (validation only) ==="
date --iso-8601=seconds
git rev-parse HEAD
git status --short
python --version

python - <<'PY'
import torch
from Datasets.mosaic_data import (
    class_histogram,
    eyepacs_fold,
    load_eyepacs_items,
)

items = load_eyepacs_items("Datasets/DR")
if len(items) != 35126:
    raise RuntimeError(f"expected 35126 EyePACS images, found {len(items)}")
train, validation, test = eyepacs_fold(items, 0, n_folds=10, seed=42)
if not torch.cuda.is_available():
    raise RuntimeError("CUDA is required")
print("gpu", torch.cuda.get_device_name(0))
print("eyepacs_images", len(items), "classes", class_histogram(items))
print(
    "fold0",
    {"train": len(train), "validation": len(validation), "locked_test": len(test)},
    "train_classes",
    class_histogram(train),
)
PY

python tools/audit_mosaic_shortcuts.py \
  --dataset dr \
  --data_root Datasets/DR \
  --folds "${FOLD}" \
  --image_size 896 \
  --output_stride 8 \
  --json_out "${RUN_DIR}/shortcut_audit.json"

python train_mosaic.py \
  --dataset dr \
  --data_root Datasets/DR \
  --run_dir "${RUN_DIR}" \
  --folds "${FOLD}" \
  "${RESUME_ARGS[@]}" \
  --skip_test \
  --local_stage rf_medium \
  --image_size 896 \
  --batch_size 16 \
  --num_workers 8 \
  --epochs "${TOTAL_EPOCHS}" \
  --early_stop_patience "${TOTAL_EPOCHS}" \
  --max_count 32 \
  --dense_warmup_epochs 4 \
  --proof_ramp_epochs 4 \
  --proof_epsilon 0.02 \
  --necessity_fraction 0.5 \
  --amp_init_scale 8192 \
  --amp_growth_interval 2000 \
  --amp_max_consecutive_skips 8
