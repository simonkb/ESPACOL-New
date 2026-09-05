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

RUN_DIR="${MOSAIC_RUN_DIR:-runs/mosaic_aptos_f0_dl95_v2}"
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

# Exercise the exact DL95 precision and optimizer policy on real augmented
# fundus images.  The old one-step, all-zero smoke was the special case for
# which spatial kernel summation is exact; it also never executed AdamW.step().
# This multi-step canary would have caught the epoch-1 activation runaway before
# launching the failed 15-epoch experiment.
python - <<'PY'
import random

import torch
from PIL import Image

from Datasets.mosaic_data import (
    MosaicFundusTransform,
    aptos_fold,
    load_aptos_items,
)
from losses.mosaic import MosaicLoss
from models.mosaic_model import build_mosaic_model

torch.manual_seed(31)
random.seed(31)
root = "Datasets/aptos2019-blindness-detection"
items = load_aptos_items(root)
train_items, _, _ = aptos_fold(items, fold=0, n_folds=5, val_fraction=0.1, seed=42)
by_grade = {
    grade: [item for item in train_items if item[1] == grade][:4]
    for grade in range(5)
}
canary_items = [item for grade in range(5) for item in by_grade[grade]]
if any(len(by_grade[grade]) < 4 for grade in range(5)):
    raise RuntimeError("APTOS canary requires at least four images per grade")
transform = MosaicFundusTransform(896, augment=True)

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
    [label for _, label in train_items],
    5,
    weight_method="effective_num",
    weight_beta=0.999,
    max_transition_weight=10.0,
    dense_weight=0.1,
    transition_reduction="boundary_mean",
).cuda()
encoder_parameters = list(model.encoder.trunk.parameters())
encoder_ids = {id(parameter) for parameter in encoder_parameters}
head_parameters = [
    parameter for parameter in model.parameters()
    if id(parameter) not in encoder_ids
]
optimizer = torch.optim.AdamW(
    [
        {"params": encoder_parameters, "lr": 1e-4},
        {"params": head_parameters, "lr": 5e-4},
    ],
    weight_decay=1e-5,
)

stage_peaks = {str(index): 0.0 for index in range(len(model.encoder.trunk))}

def stage_hook(name):
    def check(_module, _inputs, output):
        if not torch.isfinite(output).all():
            raise RuntimeError(f"non-finite DL95 canary stage {name}")
        stage_peaks[name] = max(stage_peaks[name], float(output.detach().abs().max()))
    return check

handles = [
    stage.register_forward_hook(stage_hook(str(index)))
    for index, stage in enumerate(model.encoder.trunk)
]

for step in range(16):
    selected = [canary_items[(4 * step + offset) % len(canary_items)] for offset in range(4)]
    images = []
    masks = []
    labels = []
    for path, label in selected:
        with Image.open(path) as source:
            image, mask = transform(source.convert("RGB"))
        images.append(image)
        masks.append(mask)
        labels.append(label)
    image_batch = torch.stack(images).cuda(non_blocking=True)
    mask_batch = torch.stack(masks).cuda(non_blocking=True)
    label_batch = torch.tensor(labels, device="cuda")

    optimizer.zero_grad(set_to_none=True)
    output = model(image_batch, mask_batch, project=False)
    loss, _ = criterion(
        output.transitions,
        label_batch,
        projected_stop_probabilities=output.stop_probabilities,
        projected_log_stop_probabilities=output.log_stop_probabilities,
        dense_transitions=output.dense_transitions,
        dense_stop_probabilities=output.dense_stop_probabilities,
        dense_log_stop_probabilities=output.dense_log_stop_probabilities,
    )
    if not torch.isfinite(loss):
        raise RuntimeError(f"non-finite DL95 canary loss at step {step + 1}")
    loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(
        model.parameters(), 5.0, error_if_nonfinite=True
    )
    optimizer.step()

    for name, parameter in model.named_parameters():
        if not torch.isfinite(parameter).all():
            raise RuntimeError(f"non-finite DL95 parameter after step {step + 1}: {name}")
    for _parameter, state in optimizer.state.items():
        for name, value in state.items():
            if torch.is_tensor(value) and not torch.isfinite(value).all():
                raise RuntimeError(
                    f"non-finite DL95 AdamW state after step {step + 1}: {name}"
                )
    print(
        "dl95_canary_step",
        step + 1,
        "loss",
        float(loss.detach()),
        "grad_norm",
        float(grad_norm),
    )

for handle in handles:
    handle.remove()
print(
    "dl95_real_training_canary",
    {
        "steps": 16,
        "stage_peak_abs": stage_peaks,
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
  --no_amp \
  --amp_init_scale 8192 \
  --amp_growth_interval 2000 \
  --amp_max_consecutive_skips 8
