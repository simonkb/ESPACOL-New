#!/bin/bash
# Validation-only workers for the locked EyePACS fold-9 ablation.
#SBATCH --job-name=of9_ab_tr
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --array=0-9%3
#SBATCH --output=/dpc/kuin0170/ESPACOL-New/origin_fold9_ablation_logs/train_%A_%a.out
#SBATCH --error=/dpc/kuin0170/ESPACOL-New/origin_fold9_ablation_logs/train_%A_%a.err
#SBATCH --account=kuin0170

set -euo pipefail
set +u
source /etc/profile.d/lmod.sh || exit 1
module load miniconda/3 || exit 1
module load cuda/12.6 || exit 1
source activate "${ORIGIN_CONDA_ENV:-G}" || exit 1
set -u

REPO_ROOT="${ORIGIN_REPO_ROOT:?ORIGIN_REPO_ROOT is required}"
LAUNCH_COMMIT="${ORIGIN_LAUNCH_COMMIT:?ORIGIN_LAUNCH_COMMIT is required}"
EXPERIMENT_ROOT="${ORIGIN_FOLD9_ABLATION_ROOT:?ORIGIN_FOLD9_ABLATION_ROOT is required}"
DATA_ROOT="${ORIGIN_DATA_ROOT:?ORIGIN_DATA_ROOT is required}"
PROTOCOL="${ORIGIN_FOLD9_ABLATION_PROTOCOL:?ORIGIN_FOLD9_ABLATION_PROTOCOL is required}"
TASK_INDEX="${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}"

cd "${REPO_ROOT}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
[[ "$(git rev-parse HEAD)" == "${LAUNCH_COMMIT}" ]] || {
  echo "Immutable launch commit mismatch." >&2; exit 2;
}
[[ -z "$(git status --porcelain --untracked-files=no)" ]] || {
  echo "Tracked files changed in the immutable worktree." >&2; exit 3;
}

ORIGIN_WORKER_PREFLIGHT="${EXPERIMENT_ROOT}/PREFLIGHT_COMPLETE.json" \
ORIGIN_WORKER_PROTOCOL="${PROTOCOL}" \
ORIGIN_WORKER_COMMIT="${LAUNCH_COMMIT}" \
python - <<'PY'
import os
from scripts.origin_fold9_ablation_common import (
    PREFLIGHT_MARKER_SCHEMA,
    read_json,
    verify_checksummed_payload,
    verify_protocol,
)
from training.origin_ablation_trainer import origin_ablation_implementation_signature

protocol = read_json(os.environ["ORIGIN_WORKER_PROTOCOL"])
selected = verify_protocol(
    protocol,
    expected_launch_commit=os.environ["ORIGIN_WORKER_COMMIT"],
)
marker = read_json(os.environ["ORIGIN_WORKER_PREFLIGHT"])
verify_checksummed_payload(marker, schema=PREFLIGHT_MARKER_SCHEMA)
expected = {
    "launch_commit": os.environ["ORIGIN_WORKER_COMMIT"],
    "protocol_checksum_sha256": protocol["content_checksum_sha256"],
    "selected_variants": list(selected),
    "implementation_signature": origin_ablation_implementation_signature(),
    "gpu_canaries_completed": True,
}
for key, value in expected.items():
    if marker.get(key) != value:
        raise RuntimeError(f"preflight marker {key} mismatch")
print("authenticated_preflight_marker", os.environ["ORIGIN_WORKER_PREFLIGHT"])
PY

VARIANT="$(ORIGIN_WORKER_PROTOCOL="${PROTOCOL}" \
  ORIGIN_WORKER_COMMIT="${LAUNCH_COMMIT}" \
  ORIGIN_WORKER_ROOT="${EXPERIMENT_ROOT}" \
  ORIGIN_WORKER_DATA="${DATA_ROOT}" \
  ORIGIN_WORKER_INDEX="${TASK_INDEX}" \
  python - <<'PY'
import os
from scripts.origin_fold9_ablation_common import read_json, verify_protocol
p = read_json(os.environ["ORIGIN_WORKER_PROTOCOL"])
variants = verify_protocol(
    p,
    expected_launch_commit=os.environ["ORIGIN_WORKER_COMMIT"],
    expected_root=os.environ["ORIGIN_WORKER_ROOT"],
    expected_data_root=os.environ["ORIGIN_WORKER_DATA"],
)
index = int(os.environ["ORIGIN_WORKER_INDEX"])
if index < 0 or index >= len(variants):
    raise RuntimeError(f"array index {index} is outside 0..{len(variants)-1}")
print(variants[index])
PY
)"

TASK_RUN_DIR="${EXPERIMENT_ROOT}/workers/${VARIANT}"
FOLD_DIR="${TASK_RUN_DIR}/fold9"
mkdir -p "${EXPERIMENT_ROOT}/locks"
exec 9>"${EXPERIMENT_ROOT}/locks/${VARIANT}.lock"
flock -n 9 || { echo "Another process owns variant ${VARIANT}." >&2; exit 4; }
if [[ -e "${FOLD_DIR}/best.pth" || -e "${FOLD_DIR}/last.pth" || \
      -e "${FOLD_DIR}/history.csv" || -e "${FOLD_DIR}/result.json" || \
      -e "${FOLD_DIR}/ABLATION_TRAIN_COMPLETE.json" ]]; then
  echo "Fresh-only protocol refuses existing artifacts: ${FOLD_DIR}." >&2
  exit 5
fi

echo "=== ORIGIN fold-9 validation-only ablation: ${VARIANT} ==="
date --iso-8601=seconds
echo "commit=${LAUNCH_COMMIT}"
echo "array_index=${TASK_INDEX}"
python --version
python - <<'PY'
import torch
print("torch", torch.__version__)
if not torch.cuda.is_available():
    raise RuntimeError("CUDA is required")
print("gpu", torch.cuda.get_device_name(0))
PY

TRAIN_ARGS=(
  --variant "${VARIANT}"
  --dataset dr --data_root "${DATA_ROOT}" --run_dir "${TASK_RUN_DIR}"
  --n_folds 10 --folds 9 --seed 42 --val_fraction 0.1
  --image_size 640 --encoder convnext_tiny --scales s4,s8,s16,s32
  --projection_dim 128 --reference_count 4096
  --atom_rate_init 1e-6 --prior_rate_init 1e-4 --boundary_scale_init 1.0
  --total_rate_cap 64.0 --prior_rate_cap 1.0 --boundary_scale_cap 2.0
  --rate_roundoff_margin 1.0 --atom_mode cumulative --decision_rule class_map
  --batch_size 8 --epochs 75 --num_workers 8
  --lr 1e-4 --head_lr 5e-4 --weight_decay 1e-5
  --freeze_encoder_epochs 2 --scheduler plateau --lr_factor 0.2 --lr_patience 5
  --early_stopping_patience 15 --checkpoint_selection acc_then_qwk
  --selection_qwk_weight 0.1 --rps_weight 0.25 --evidence_budget_weight 0.0
  --class_weighting none --amp --amp_init_scale 4096 --amp_unfreeze_scale 256
  --amp_growth_interval 2000 --amp_max_consecutive_skips 8 --skip_test
)
printf 'training_command:'; printf ' %q' python train_origin_ablation.py "${TRAIN_ARGS[@]}"; printf '\n'
python train_origin_ablation.py "${TRAIN_ARGS[@]}"

python scripts/audit_origin_fold9_ablation_training.py \
  --variant "${VARIANT}" --task-run-dir "${TASK_RUN_DIR}" --protocol "${PROTOCOL}"

echo "Validation-only training and audit completed for ${VARIANT}."
