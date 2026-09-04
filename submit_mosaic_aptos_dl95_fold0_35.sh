#!/bin/bash
# Controlled MOSAIC-DL95 experiment: APTOS fold 0, inner validation only.
# The encoder tap is the sole architectural change from the established run.
#SBATCH --job-name=mosaic_a_f0_dl95
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH --output=/dpc/kuin0170/ESPACOL-New/mosaic_aptos_dl95_f0_%j.out
#SBATCH --error=/dpc/kuin0170/ESPACOL-New/mosaic_aptos_dl95_f0_%j.err
#SBATCH --account=kuin0170

# Lmod and Conda may inspect unset scheduler variables, so initialize them
# before enabling nounset.
source /etc/profile.d/lmod.sh || exit 1
module load miniconda/3 || exit 1
module load cuda/12.6 || exit 1
source activate G || exit 1

set -euo pipefail
cd /dpc/kuin0170/ESPACOL-New

export HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

RUN_DIR="${MOSAIC_RUN_DIR:-runs/mosaic_aptos_f0_dl95_v1}"
FOLD=0
TOTAL_EPOCHS=35
FOLD_DIR="${RUN_DIR}/fold${FOLD}"
mkdir -p "${FOLD_DIR}"

exec 9>"${FOLD_DIR}/.writer.lock"
if ! flock -n 9; then
  echo "Another process is already writing ${FOLD_DIR}." >&2
  exit 2
fi

if [[ -e "${FOLD_DIR}/last.pth" || -e "${FOLD_DIR}/history.csv" ]]; then
  echo "Fresh DL95 run refused: ${FOLD_DIR} contains training artifacts." >&2
  echo "Set MOSAIC_RUN_DIR to a new directory; this experiment must not resume an rf_medium checkpoint." >&2
  exit 3
fi

echo "=== MOSAIC-DL95 APTOS fold 0 / 35 epochs (inner validation only) ==="
date --iso-8601=seconds
git rev-parse HEAD
git status --short
python --version

# Fail before loading the dataset if the new encoder contract or CUDA setup is
# wrong.  This also verifies that pretrained weights are present in the
# cluster's offline cache.
python - <<'PY'
import torch
from models.local_efficientnet import LocalEfficientNetV2S

if not torch.cuda.is_available():
    raise RuntimeError("CUDA is required for the DL95 experiment")
encoder = LocalEfficientNetV2S(
    tap="dl95",
    local_dim=128,
    pretrained=True,
    image_is_normalized=True,
)
assert encoder.output_stride == 32, encoder.output_stride
assert encoder.receptive_field == 95, encoder.receptive_field
print("gpu", torch.cuda.get_device_name(0))
print(
    "dl95_contract",
    {
        "tap": encoder.tap,
        "stride": encoder.output_stride,
        "receptive_field": encoder.receptive_field,
    },
)
PY

# Focused structural tests catch an accidental receptive-field or CLI
# regression before the job spends time loading augmented 896px batches.
if ! python -c 'import pytest' >/dev/null 2>&1; then
  echo "Missing pytest in conda environment G." >&2
  echo "Install once from the repository root:" >&2
  echo "  python -m pip install -r requirements-dev.txt" >&2
  exit 4
fi
python -m pytest -q \
  tests/test_local_efficientnet.py \
  tests/test_mosaic_integration.py

# Exercise the actual AMP + checkpointed DL95 training path once at the
# configured image and batch sizes.  This fails early on non-finite activations,
# gradients, or insufficient GPU memory instead of losing an experiment hours
# into training.
python - <<'PY'
import torch

from losses.mosaic import MosaicLoss
from models.mosaic_model import build_mosaic_model
from utils.spatial_mask import centered_ellipse_mask

labels = torch.tensor([0, 1, 3, 4], device="cuda")
train_labels = [0] * 1300 + [1] * 266 + [2] * 719 + [3] * 139 + [4] * 212
model = build_mosaic_model(
    num_classes=5,
    image_size=896,
    local_stage="dl95",
    local_dim=128,
    pretrained=True,
    grad_checkpoint=True,
    max_count=32,
    sufficiency_tolerance=0.0,
    complement_suppression=0.5,
).cuda().train()
criterion = MosaicLoss.from_training_labels(
    train_labels,
    5,
    weight_method="effective_num",
    weight_beta=0.999,
    max_transition_weight=10.0,
    dense_weight=0.1,
    transition_reduction="boundary_mean",
).cuda()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
image = torch.zeros(4, 3, 896, 896, device="cuda")
mask = centered_ellipse_mask(
    896, 896, batch_size=4, device=torch.device("cuda")
)
scaler = torch.amp.GradScaler("cuda", init_scale=8192.0, growth_interval=2000)
optimizer.zero_grad(set_to_none=True)
with torch.amp.autocast("cuda"):
    output = model(image, mask, project=True)
    loss, _ = criterion(
        output.transitions,
        labels,
        projected_stop_probabilities=output.stop_probabilities,
        projected_log_stop_probabilities=output.log_stop_probabilities,
        dense_transitions=output.dense_transitions,
        dense_stop_probabilities=output.dense_stop_probabilities,
        dense_log_stop_probabilities=output.dense_log_stop_probabilities,
    )
if not torch.isfinite(loss):
    raise RuntimeError(f"non-finite DL95 smoke loss: {float(loss.detach())}")
scaler.scale(loss).backward()
scaler.unscale_(optimizer)
grad_norm = torch.nn.utils.clip_grad_norm_(
    model.parameters(), 5.0, error_if_nonfinite=True
)
print(
    "dl95_training_smoke",
    {
        "loss": float(loss.detach()),
        "grad_norm": float(grad_norm),
        "valid_cells": model.expected_valid_cells,
        "peak_cuda_gib": torch.cuda.max_memory_allocated() / 2**30,
    },
)
PY

# Audit uses the actual DL95 evidence stride; the outer test split remains
# locked throughout architecture selection.
python tools/audit_mosaic_shortcuts.py \
  --dataset aptos \
  --data_root Datasets/aptos2019-blindness-detection \
  --folds "${FOLD}" \
  --image_size 896 \
  --output_stride 32 \
  --json_out "${RUN_DIR}/shortcut_audit.json"

python train_mosaic.py \
  --dataset aptos \
  --data_root Datasets/aptos2019-blindness-detection \
  --run_dir "${RUN_DIR}" \
  --folds "${FOLD}" \
  --skip_test \
  --local_stage dl95 \
  --image_size 896 \
  --batch_size 4 \
  --num_workers 8 \
  --grad_checkpoint \
  --epochs "${TOTAL_EPOCHS}" \
  --early_stop_patience "${TOTAL_EPOCHS}" \
  --decision_rule posterior_median \
  --transition_reduction boundary_mean \
  --max_count 32 \
  --dense_warmup_epochs 4 \
  --proof_ramp_epochs 4 \
  --proof_epsilon 0.02 \
  --necessity_fraction 0.5 \
  --lr 1e-4 \
  --head_lr 5e-4 \
  --weight_decay 1e-5 \
  --amp_init_scale 8192 \
  --amp_growth_interval 2000 \
  --amp_max_consecutive_skips 8
