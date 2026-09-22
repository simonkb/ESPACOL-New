#!/bin/bash
# Frozen ORIGIN-v3 EyePACS 10-fold patient-grouped outer-CV evaluation.
#SBATCH --job-name=ov3_dr_cv
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --array=0-9%3
#SBATCH --output=/dpc/kuin0170/ESPACOL-New/origin_v3_cv_logs/dr_%A_%a.out
#SBATCH --error=/dpc/kuin0170/ESPACOL-New/origin_v3_cv_logs/dr_%A_%a.err
#SBATCH --account=kuin0170

set -euo pipefail
set +u
source /etc/profile.d/lmod.sh || exit 1
module load miniconda/3 || exit 1
module load cuda/12.6 || exit 1
source activate "${ORIGIN_CONDA_ENV:-G}" || exit 1
set -u

REPO_ROOT="${ORIGIN_REPO_ROOT:?ORIGIN_REPO_ROOT is required}"
CV_ROOT="${ORIGIN_V3_CV_ROOT:?ORIGIN_V3_CV_ROOT is required}"
DATA_ROOT="${ORIGIN_DATA_ROOT:?ORIGIN_DATA_ROOT is required}"
PROTOCOL="${ORIGIN_V3_PROTOCOL:?ORIGIN_V3_PROTOCOL is required}"
LAUNCH_COMMIT="${ORIGIN_LAUNCH_COMMIT:?ORIGIN_LAUNCH_COMMIT is required}"
FOLD="${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}"
[[ "${FOLD}" =~ ^[0-9]$ ]] || { echo "Invalid EyePACS fold ${FOLD}." >&2; exit 2; }

cd "${REPO_ROOT}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
[[ "$(git rev-parse HEAD)" == "${LAUNCH_COMMIT}" ]] || {
  echo "Immutable launch commit mismatch." >&2; exit 3;
}
[[ -z "$(git status --porcelain --untracked-files=no)" ]] || {
  echo "Tracked files changed in the immutable worktree." >&2; exit 4;
}

ORIGIN_PROTOCOL="${PROTOCOL}" ORIGIN_EXPECTED_COMMIT="${LAUNCH_COMMIT}" \
ORIGIN_EXPECTED_DATA_ROOT="${DATA_ROOT}" ORIGIN_EXPECTED_CV_ROOT="${CV_ROOT}" python - <<'PY'
import os
from pathlib import Path
from scripts.origin_v3_cv_common import (
    ARCHITECTURE_SIGNATURE, BASE_COMMIT, DATASET_SPECS, IMPLEMENTATION_SIGNATURE,
    read_json, validate_pinned_config, verify_checksummed_payload,
)
from training.origin_trainer import origin_implementation_signature
p = read_json(os.environ["ORIGIN_PROTOCOL"])
verify_checksummed_payload(p, schema="origin-v3-full-cv-protocol-v1")
expected = {
    "dataset": "dr",
    "n_folds": 10,
    "n_images": 35126,
    "base_commit": BASE_COMMIT,
    "launch_commit": os.environ["ORIGIN_EXPECTED_COMMIT"],
    "implementation_signature": IMPLEMENTATION_SIGNATURE,
    "architecture_signature": ARCHITECTURE_SIGNATURE,
    "config_signature": DATASET_SPECS["dr"]["config_signature"],
    "evaluation_scope": "outer_test_after_selection",
}
for key, value in expected.items():
    if p.get(key) != value:
        raise RuntimeError(f"protocol {key} mismatch: {p.get(key)!r} != {value!r}")
if Path(p["data_root"]).resolve() != Path(os.environ["ORIGIN_EXPECTED_DATA_ROOT"]).resolve():
    raise RuntimeError("exported EyePACS data root differs from the locked protocol")
if Path(p["cv_root"]).resolve() != Path(os.environ["ORIGIN_EXPECTED_CV_ROOT"]).resolve():
    raise RuntimeError("exported EyePACS CV root differs from the locked protocol")
validate_pinned_config(p.get("frozen_config", {}), "dr")
observed = origin_implementation_signature()
if observed != IMPLEMENTATION_SIGNATURE:
    raise RuntimeError(f"ORIGIN implementation drift: {observed}")
PY

TASK_RUN_DIR="${CV_ROOT}/workers/fold${FOLD}"
FOLD_DIR="${TASK_RUN_DIR}/fold${FOLD}"
mkdir -p "${CV_ROOT}/locks"
exec 9>"${CV_ROOT}/locks/fold${FOLD}.lock"
flock -n 9 || { echo "Another process owns EyePACS fold ${FOLD}." >&2; exit 5; }
if [[ -e "${FOLD_DIR}/best.pth" || -e "${FOLD_DIR}/last.pth" || -e "${FOLD_DIR}/history.csv" || -e "${FOLD_DIR}/result.json" ]]; then
  echo "Fresh-only protocol refuses existing fold artifacts: ${FOLD_DIR}." >&2
  exit 6
fi

echo "=== Frozen ORIGIN-v3 EyePACS outer CV fold ${FOLD}/9 ==="
date --iso-8601=seconds
echo "commit=${LAUNCH_COMMIT}"
echo "implementation_signature=0d9734495c4aeccb2a0038f2b9672f88c445a13b276d7d1073cae1cc0c86909b"
python --version
python - <<'PY'
import torch, torchvision
print("torch", torch.__version__, "torchvision", torchvision.__version__)
if not torch.cuda.is_available():
    raise RuntimeError("CUDA is required")
print("gpu", torch.cuda.get_device_name(0))
PY

TRAIN_ARGS=(
  --dataset dr --data_root "${DATA_ROOT}" --run_dir "${TASK_RUN_DIR}"
  --n_folds 10 --folds "${FOLD}" --seed 42
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
  --amp_growth_interval 2000 --amp_max_consecutive_skips 8 --include_test
)
printf 'training_command:'; printf ' %q' python train_origin.py "${TRAIN_ARGS[@]}"; printf '\n'
python train_origin.py "${TRAIN_ARGS[@]}"
python scripts/export_origin_v3_outer_predictions.py \
  --checkpoint "${FOLD_DIR}/best.pth" \
  --result "${FOLD_DIR}/result.json" \
  --split-manifest "${FOLD_DIR}/split_manifest.json" \
  --data-root "${DATA_ROOT}" --protocol "${PROTOCOL}" \
  --output-csv "${FOLD_DIR}/outer_predictions.csv" \
  --output-manifest "${FOLD_DIR}/outer_predictions_manifest.json" \
  --num-workers 8
python scripts/audit_origin_v3_cv_fold.py \
  --dataset dr --fold "${FOLD}" --task-run-dir "${TASK_RUN_DIR}" \
  --protocol "${PROTOCOL}"

echo "Frozen ORIGIN-v3 EyePACS fold ${FOLD} completed and audited."
