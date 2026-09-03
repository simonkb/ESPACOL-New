#!/bin/bash
# Fresh, validation-only APTOS fold-1 MOSAIC run through epoch 100.
# This is an exploratory robustness run; the outer test fold remains untouched.
#SBATCH --job-name=mosaic_aptos_f1_e100
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH --output=/dpc/kuin0170/ESPACOL-New/mosaic_aptos_f1_e100_%j.out
#SBATCH --error=/dpc/kuin0170/ESPACOL-New/mosaic_aptos_f1_e100_%j.err
#SBATCH --account=kuin0170

# Lmod inspects unset PBS variables on this cluster, so enable strict shell
# handling only after the established module/Conda bootstrap has completed.
source /etc/profile.d/lmod.sh || exit 1
module load miniconda/3 || exit 1
module load cuda/12.6 || exit 1
source activate G || exit 1

set -euo pipefail
cd /dpc/kuin0170/ESPACOL-New

export HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

RUN_DIR="${MOSAIC_RUN_DIR:-runs/mosaic_aptos_fold1_e100}"
FOLD=1
TOTAL_EPOCHS=100
DECISION_RULE="${MOSAIC_DECISION_RULE:-posterior_median}"
FOLD_DIR="${RUN_DIR}/fold${FOLD}"
mkdir -p "${FOLD_DIR}"

# Prevent accidental concurrent writers.  Resume is supported only when it is
# explicitly requested and a complete last-epoch checkpoint exists.
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

echo "=== MOSAIC APTOS fold 1 / 100 epochs (validation only) ==="
date --iso-8601=seconds
git rev-parse HEAD
git status --short
python --version

python - <<'PY'
import torch
from Datasets.mosaic_data import class_histogram, load_aptos_items

items = load_aptos_items("Datasets/aptos2019-blindness-detection")
if len(items) != 3662:
    raise RuntimeError(f"expected 3662 APTOS images, found {len(items)}")
if not torch.cuda.is_available():
    raise RuntimeError("CUDA is required")
print("gpu", torch.cuda.get_device_name(0))
print("aptos_images", len(items), "classes", class_histogram(items))
PY

# Audit only train and inner-validation acquisition metadata.  The outer test
# images are not opened unless --include_test is explicitly supplied (it is not).
python tools/audit_mosaic_shortcuts.py \
  --dataset aptos \
  --data_root Datasets/aptos2019-blindness-detection \
  --folds "${FOLD}" \
  --image_size 896 \
  --output_stride 8 \
  --json_out "${RUN_DIR}/shortcut_audit.json"

python train_mosaic.py \
  --dataset aptos \
  --data_root Datasets/aptos2019-blindness-detection \
  --run_dir "${RUN_DIR}" \
  --folds "${FOLD}" \
  "${RESUME_ARGS[@]}" \
  --skip_test \
  --local_stage rf_medium \
  --image_size 896 \
  --batch_size 4 \
  --num_workers 8 \
  --epochs "${TOTAL_EPOCHS}" \
  --early_stop_patience "${TOTAL_EPOCHS}" \
  --decision_rule "${DECISION_RULE}" \
  --transition_reduction boundary_mean \
  --max_count 32 \
  --dense_warmup_epochs 4 \
  --proof_ramp_epochs 4 \
  --proof_epsilon 0.02 \
  --necessity_fraction 0.5 \
  --amp_init_scale 8192 \
  --amp_growth_interval 2000 \
  --amp_max_consecutive_skips 8
