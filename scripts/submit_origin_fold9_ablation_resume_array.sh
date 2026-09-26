#!/bin/bash
# Resume-only workers for failed variants in an immutable fold-9 protocol.
#SBATCH --job-name=of9_ab_re
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --output=/dpc/kuin0170/ESPACOL-New/origin_fold9_ablation_logs/retry_%A_%a.out
#SBATCH --error=/dpc/kuin0170/ESPACOL-New/origin_fold9_ablation_logs/retry_%A_%a.err
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
RETRY_VARIANTS="${ORIGIN_FOLD9_RETRY_VARIANTS:?ORIGIN_FOLD9_RETRY_VARIANTS is required}"
TASK_INDEX="${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}"

cd "${REPO_ROOT}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
[[ "$(git rev-parse HEAD)" == "${LAUNCH_COMMIT}" ]] || {
  echo "Immutable launch commit mismatch." >&2; exit 2;
}
[[ -z "$(git status --porcelain --untracked-files=no)" ]] || {
  echo "Tracked files changed in the immutable worktree." >&2; exit 3;
}

VARIANT="$(ORIGIN_WORKER_PROTOCOL="${PROTOCOL}" \
  ORIGIN_WORKER_COMMIT="${LAUNCH_COMMIT}" \
  ORIGIN_WORKER_ROOT="${EXPERIMENT_ROOT}" \
  ORIGIN_WORKER_DATA="${DATA_ROOT}" \
  ORIGIN_WORKER_RETRY_VARIANTS="${RETRY_VARIANTS}" \
  ORIGIN_WORKER_INDEX="${TASK_INDEX}" \
  python - <<'PY'
import os
from scripts.origin_fold9_ablation_common import read_json, verify_protocol
p = read_json(os.environ["ORIGIN_WORKER_PROTOCOL"])
selected = verify_protocol(
    p,
    expected_launch_commit=os.environ["ORIGIN_WORKER_COMMIT"],
    expected_root=os.environ["ORIGIN_WORKER_ROOT"],
    expected_data_root=os.environ["ORIGIN_WORKER_DATA"],
)
retry = tuple(filter(None, os.environ["ORIGIN_WORKER_RETRY_VARIANTS"].split(":")))
if not retry or len(retry) != len(set(retry)) or not set(retry).issubset(selected):
    raise RuntimeError("invalid retry-variant set")
index = int(os.environ["ORIGIN_WORKER_INDEX"])
if index < 0 or index >= len(retry):
    raise RuntimeError(f"array index {index} is outside 0..{len(retry)-1}")
print(retry[index])
PY
)"

TASK_RUN_DIR="${EXPERIMENT_ROOT}/workers/${VARIANT}"
FOLD_DIR="${TASK_RUN_DIR}/fold9"
mkdir -p "${EXPERIMENT_ROOT}/locks"
exec 9>"${EXPERIMENT_ROOT}/locks/${VARIANT}.lock"
flock -n 9 || { echo "Another process owns variant ${VARIANT}." >&2; exit 4; }

[[ ! -e "${FOLD_DIR}/ABLATION_TRAIN_COMPLETE.json" ]] || {
  echo "Completed variant cannot be resumed: ${VARIANT}." >&2; exit 5;
}
[[ -f "${FOLD_DIR}/last.pth" ]] || {
  echo "Resume checkpoint is missing: ${FOLD_DIR}/last.pth." >&2; exit 6;
}
[[ -f "${FOLD_DIR}/split_manifest.json" ]] || {
  echo "Resume split manifest is missing: ${FOLD_DIR}/split_manifest.json." >&2; exit 7;
}
[[ ! -e "${EXPERIMENT_ROOT}/release" && \
   ! -e "${EXPERIMENT_ROOT}/OUTER_RELEASE_COMPLETE.json" ]] || {
  echo "Outer release already exists; resume is forbidden." >&2; exit 8;
}
if compgen -G "${EXPERIMENT_ROOT}/.release.staging.*" >/dev/null; then
  echo "Outer-release staging exists; preserve and audit it before any retry." >&2
  exit 9
fi

# Recheck the split and checkpoint in the GPU worker immediately before use.
ORIGIN_RESUME_PROTOCOL="${PROTOCOL}" \
ORIGIN_RESUME_COMMIT="${LAUNCH_COMMIT}" \
ORIGIN_RESUME_ROOT="${EXPERIMENT_ROOT}" \
ORIGIN_RESUME_DATA="${DATA_ROOT}" \
ORIGIN_RESUME_VARIANT="${VARIANT}" \
python - <<'PY'
import os
from pathlib import Path
from typing import Mapping
import torch
from scripts.origin_fold9_ablation_common import (
    EXPECTED_SPLIT_SIGNATURE, FOLD, read_json, validate_split_manifest,
    validate_variant_config, verify_protocol,
)
from training.origin_ablation_trainer import origin_ablation_implementation_signature
root = Path(os.environ["ORIGIN_RESUME_ROOT"]).resolve()
variant = os.environ["ORIGIN_RESUME_VARIANT"]
protocol = read_json(os.environ["ORIGIN_RESUME_PROTOCOL"])
selected = verify_protocol(
    protocol,
    expected_launch_commit=os.environ["ORIGIN_RESUME_COMMIT"],
    expected_root=root,
    expected_data_root=os.environ["ORIGIN_RESUME_DATA"],
)
if variant not in selected:
    raise RuntimeError("resume variant is outside the locked protocol")
fold_dir = root / "workers" / variant / f"fold{FOLD}"
split = read_json(fold_dir / "split_manifest.json")
validate_split_manifest(split)
if split.get("ablation_variant") != variant:
    raise RuntimeError("resume split variant mismatch")
state = torch.load(fold_dir / "last.pth", map_location="cpu", weights_only=False)
if state.get("schema") != "origin-checkpoint-v3":
    raise RuntimeError("resume checkpoint schema mismatch")
if state.get("fold") != FOLD or state.get("split_signature") != EXPECTED_SPLIT_SIGNATURE:
    raise RuntimeError("resume checkpoint fold/split mismatch")
if state.get("implementation_signature") != origin_ablation_implementation_signature():
    raise RuntimeError("resume checkpoint implementation differs from launch snapshot")
config = state.get("config")
if not isinstance(config, Mapping):
    raise RuntimeError("resume checkpoint configuration is missing")
validate_variant_config(config, variant)
if Path(str(config.get("run_dir"))).resolve() != (root / "workers" / variant).resolve():
    raise RuntimeError("resume checkpoint run directory mismatch")
print("resume_checkpoint_verified", variant, "epoch", state.get("epoch"))
PY

echo "=== ORIGIN fold-9 validation-only ablation resume: ${VARIANT} ==="
date --iso-8601=seconds
echo "commit=${LAUNCH_COMMIT}"
echo "array_index=${TASK_INDEX}"
python --version
python - <<'PY'
import torch
if not torch.cuda.is_available():
    raise RuntimeError("CUDA is required")
print("torch", torch.__version__)
print("gpu", torch.cuda.get_device_name(0))
PY

# Byte-for-byte scientific arguments from the fresh worker, plus the sole
# invocation-policy change --resume.  Resume itself is excluded from ORIGIN's
# numerical configuration signature.
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
  --resume
)
printf 'resume_training_command:'; printf ' %q' python train_origin_ablation.py "${TRAIN_ARGS[@]}"; printf '\n'
python train_origin_ablation.py "${TRAIN_ARGS[@]}"

python scripts/audit_origin_fold9_ablation_training.py \
  --variant "${VARIANT}" --task-run-dir "${TASK_RUN_DIR}" --protocol "${PROTOCOL}"

echo "Validation-only resume and audit completed for ${VARIANT}."
