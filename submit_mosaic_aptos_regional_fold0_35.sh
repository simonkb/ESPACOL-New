#!/bin/bash
# Controlled MOSAIC-v3 experiment: RF-medium + fixed 8x8 smooth regional events.
# APTOS fold 0, inner validation only. The outer test split remains locked.
#SBATCH --job-name=mosaic_a_f0_reg8
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH --output=/dpc/kuin0170/ESPACOL-New/mosaic_aptos_reg8_f0_%j.out
#SBATCH --error=/dpc/kuin0170/ESPACOL-New/mosaic_aptos_reg8_f0_%j.err
#SBATCH --account=kuin0170

source /etc/profile.d/lmod.sh || exit 1
module load miniconda/3 || exit 1
module load cuda/12.6 || exit 1
source activate G || exit 1

set -euo pipefail
cd /dpc/kuin0170/ESPACOL-New

export HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

RUN_DIR="${MOSAIC_RUN_DIR:-runs/mosaic_aptos_f0_regional_v3}"
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
  echo "Fresh regional run refused: ${FOLD_DIR} contains training artifacts." >&2
  echo "Set MOSAIC_RUN_DIR to a new directory; do not resume an incompatible checkpoint." >&2
  exit 3
fi

echo "=== MOSAIC-v3 regional APTOS fold 0 / 35 epochs (inner validation only) ==="
date --iso-8601=seconds
git rev-parse HEAD
git status --short
python --version

if ! python -c 'import pytest' >/dev/null 2>&1; then
  echo "Missing pytest in conda environment G." >&2
  echo "Install once: python -m pip install -r requirements-dev.txt" >&2
  exit 4
fi

# Structural proof, regional-envelope, dead-positive-gradient, integration,
# and certificate tests must pass before allocating the training budget.
python -m pytest -q \
  tests/test_mosaic.py \
  tests/test_mosaic_loss.py \
  tests/test_mosaic_decoder.py \
  tests/test_local_efficientnet.py \
  tests/test_mosaic_certificate.py \
  tests/test_mosaic_certificate_export.py \
  tests/test_mosaic_integration.py

# Exercise the exact 896px architecture and optimizer policy once. This also
# verifies the source/proof geometry and catches a CUDA-only gradient failure.
python - <<'PY'
import torch

from losses.mosaic import MosaicLoss
from models.mosaic_model import build_mosaic_model
from utils.spatial_mask import centered_ellipse_mask

if not torch.cuda.is_available():
    raise RuntimeError("CUDA is required")
model = build_mosaic_model(
    num_classes=5,
    image_size=896,
    local_stage="rf_medium",
    local_dim=128,
    pretrained=True,
    max_count=32,
    sufficiency_tolerance=0.02,
    complement_suppression=0.5,
    region_grid_size=8,
    region_pool_temperature=0.25,
).cuda().train()
labels = torch.tensor([0, 1, 3, 4], device="cuda")
train_labels = [0] * 1300 + [1] * 266 + [2] * 719 + [3] * 139 + [4] * 212
criterion = MosaicLoss.from_training_labels(
    train_labels,
    5,
    weight_method="effective_num",
    weight_beta=0.999,
    max_transition_weight=10.0,
    dense_weight=0.1,
    transition_reduction="boundary_mean",
).cuda()
optimizer = torch.optim.AdamW(
    [
        {"params": model.encoder.trunk.parameters(), "lr": 1e-4},
        {"params": [
            parameter for name, parameter in model.named_parameters()
            if not name.startswith("encoder.trunk.")
        ], "lr": 5e-4},
    ],
    weight_decay=1e-5,
)
images = torch.zeros(4, 3, 896, 896, device="cuda")
masks = centered_ellipse_mask(896, 896, batch_size=4, device=torch.device("cuda"))
optimizer.zero_grad(set_to_none=True)
with torch.amp.autocast("cuda"):
    output = model(images, masks, project=True)
    loss, diagnostics = criterion(
        output.transitions,
        labels,
        projected_stop_probabilities=output.stop_probabilities,
        projected_log_transition_probabilities=output.log_transition_probabilities,
        projected_log_stop_probabilities=output.log_stop_probabilities,
        dense_transitions=output.dense_transitions,
        dense_stop_probabilities=output.dense_stop_probabilities,
        dense_log_transition_probabilities=output.dense_log_transition_probabilities,
        dense_log_stop_probabilities=output.dense_log_stop_probabilities,
        proof_sizes=output.proof.proof_size,
    )
if output.source_lattice is None or output.source_lattice.lattice_size != (112, 112):
    raise RuntimeError("regional source lattice contract failed")
if output.lattice.lattice_size != (8, 8):
    raise RuntimeError("regional proof lattice contract failed")
if output.evidence.witness_probabilities.shape[1:] != (64, 4):
    raise RuntimeError("regional event tensor contract failed")
if not torch.isfinite(loss):
    raise RuntimeError("regional canary loss is non-finite")
loss.backward()
grad_norm = torch.nn.utils.clip_grad_norm_(
    model.parameters(), 5.0, error_if_nonfinite=True
)
optimizer.step()
print(
    "regional_canary",
    {
        "gpu": torch.cuda.get_device_name(0),
        "source_valid_cells": model.expected_valid_cells,
        "valid_regional_events": model.expected_valid_regions,
        "region_pool_temperature": model.region_pool_temperature,
        "proof_stride": model.proof_output_stride,
        "proof_receptive_field": model.proof_receptive_field,
        "loss": float(loss.detach()),
        "grad_norm": float(grad_norm),
        "dead_positive_count": float(diagnostics["dead_positive_count"]),
    },
)
PY

python train_mosaic.py \
  --dataset aptos \
  --data_root Datasets/aptos2019-blindness-detection \
  --run_dir "${RUN_DIR}" \
  --folds "${FOLD}" \
  --skip_test \
  --local_stage rf_medium \
  --region_grid_size 8 \
  --region_pool_temperature 0.25 \
  --image_size 896 \
  --batch_size 4 \
  --num_workers 8 \
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
