#!/bin/bash
# APTOS fold-0 end-to-end MOSAIC viability pilot.
# This is deliberately NOT the formal cached-map, matched-baseline Gate 1.
#SBATCH --job-name=mosaic_aptos_pilot
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH --output=/dpc/kuin0170/ESPACOL-New/mosaic_aptos_pilot_%j.out
#SBATCH --error=/dpc/kuin0170/ESPACOL-New/mosaic_aptos_pilot_%j.err
#SBATCH --account=kuin0170

# Match the bootstrap order used by the project's proven SLURM launchers.
# Lmod and Conda are third-party shell code and are not safe under ``set -u``
# (Lmod may inspect PBS_NODEFILE even in a Slurm job), so strict mode starts
# only after both have initialized.
source /etc/profile.d/lmod.sh || {
  echo "Failed to initialize Lmod from /etc/profile.d/lmod.sh" >&2
  exit 1
}
module load miniconda/3 || exit 1
module load cuda/12.6 || exit 1
source activate G || {
  echo "Failed to activate conda environment G" >&2
  exit 1
}

set -euo pipefail
cd /dpc/kuin0170/ESPACOL-New

export HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

RUN_DIR="${MOSAIC_RUN_DIR:-runs/mosaic_aptos_pilot}"
TOTAL_EPOCHS="${MOSAIC_EPOCHS:-50}"
EARLY_STOP_PATIENCE="${MOSAIC_EARLY_STOP_PATIENCE:-30}"
DECISION_RULE="${MOSAIC_DECISION_RULE:-deweighted_class_map}"
RESUME_ARGS=()
if [[ "${MOSAIC_RESUME:-0}" == "1" ]]; then
  RESUME_ARGS+=(--resume)
elif [[ -e "${RUN_DIR}/fold0/last.pth" || -e "${RUN_DIR}/fold0/history.csv" ]]; then
  echo "Refusing to mix a fresh pilot with existing fold-0 artifacts in ${RUN_DIR}." >&2
  echo "Choose MOSAIC_RUN_DIR=<new-dir>, or set MOSAIC_RESUME=1 deliberately." >&2
  exit 2
fi
mkdir -p "${RUN_DIR}"

echo "=== MOSAIC APTOS viability pilot ==="
date --iso-8601=seconds
git rev-parse HEAD
git status --short
python --version

# The established G environment already supplies the CUDA runtime packages.
# Pytest is intentionally a development dependency because it is required by
# Gate 0, not by inference/training itself.  Do not silently mutate the shared
# Conda environment from a compute job; print the exact one-time repair.
if ! python -c 'import pytest' >/dev/null 2>&1; then
  echo "Missing pytest in conda environment G." >&2
  echo "Install once from the repository root:" >&2
  echo "  python -m pip install -r requirements-dev.txt" >&2
  exit 3
fi
python -c 'import pytest; print("pytest", pytest.__version__)'

python - <<'PY'
import torch, torchvision
from models.local_efficientnet import LocalEfficientNetV2S

print("torch", torch.__version__)
print("torchvision", torchvision.__version__)
print("cuda_runtime", torch.version.cuda)
print("cuda_available", torch.cuda.is_available())
if not torch.cuda.is_available():
    raise RuntimeError("CUDA is required for the cluster pilot")
print("gpu", torch.cuda.get_device_name(0))
# Fail before the experiment if the ImageNet weights are not cached/available.
LocalEfficientNetV2S(
    tap="rf_medium", local_dim=8, pretrained=True
)
print("efficientnet_v2_s_pretrained", "available")
PY

# Quantify acquisition-format shortcuts on train -> inner validation only.
# The canonical mask-count channel must collapse to the majority baseline;
# no outer-test image header is opened without an explicit opt-in.
python tools/audit_mosaic_shortcuts.py \
  --dataset aptos \
  --data_root Datasets/aptos2019-blindness-detection \
  --folds 0 \
  --image_size 896 \
  --output_stride 8 \
  --json_out "${RUN_DIR}/shortcut_audit.json"

python - <<'PY'
from Datasets.mosaic_data import load_aptos_items

items = load_aptos_items("Datasets/aptos2019-blindness-detection")
if len(items) != 3662:
    raise RuntimeError(f"expected 3662 APTOS images, found {len(items)}")
print("aptos_images", len(items))
PY

# Structural tests are part of Gate 0, not optional developer checks.  The
# CUDA-only certificate test also checks GPU-forward -> CPU-replay tolerance.
python -m pytest -q \
  tests/test_mosaic.py \
  tests/test_mosaic_loss.py \
  tests/test_mosaic_decoder.py \
  tests/test_mosaic_decoder_audit.py \
  tests/test_local_efficientnet.py \
  tests/test_mosaic_certificate.py \
  tests/test_mosaic_certificate_export.py \
  tests/test_mosaic_integration.py

# The calibrated case must produce nonzero local and cardinality gradients.
# A random saturated field is timed separately as a worst-case proof stress.
python tools/benchmark_mosaic_core.py \
  --cells 12544 \
  --batch_size 4 \
  --max_count 32 \
  --json_out "${RUN_DIR}/gate0.json"

# A configured-batch 896x896 model backward catches encoder OOM before the run.
python - <<'PY'
import torch
from losses.mosaic import MosaicLoss
from models.mosaic_model import build_mosaic_model
from utils.spatial_mask import centered_ellipse_mask

labels = torch.tensor([0, 1, 3, 4], device="cuda")
train_labels = [0] * 1300 + [1] * 266 + [2] * 719 + [3] * 139 + [4] * 212
for project in (False, True):
    model = build_mosaic_model(
        num_classes=5,
        image_size=896,
        local_stage="rf_medium",
        local_dim=128,
        pretrained=True,
        max_count=32,
        sufficiency_tolerance=0.0,
        complement_suppression=0.5,
    ).cuda().train()
    image = torch.zeros(4, 3, 896, 896, device="cuda")
    mask = centered_ellipse_mask(
        896, 896, batch_size=4, device=torch.device("cuda")
    )
    criterion = MosaicLoss.from_training_labels(
        train_labels,
        5,
        weight_method="effective_num",
        weight_beta=0.999,
        max_transition_weight=10.0,
        dense_weight=0.1,
    ).cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scaler = torch.amp.GradScaler(
        "cuda", init_scale=8192.0, growth_interval=2000
    )
    optimizer.zero_grad(set_to_none=True)
    with torch.amp.autocast("cuda"):
        output = model(image, mask, project=project)
        loss, _ = criterion(
            output.transitions,
            labels,
            projected_stop_probabilities=output.stop_probabilities,
            projected_log_stop_probabilities=output.log_stop_probabilities,
            dense_transitions=output.dense_transitions,
            dense_stop_probabilities=output.dense_stop_probabilities,
            dense_log_stop_probabilities=output.dense_log_stop_probabilities,
        )
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    grad_norm = torch.nn.utils.clip_grad_norm_(
        model.parameters(), 5.0, error_if_nonfinite=True
    )
    offenders = [
        name for name, parameter in model.named_parameters()
        if parameter.grad is not None and not torch.isfinite(parameter.grad).all()
    ]
    if offenders:
        raise RuntimeError(f"non-finite scaled smoke gradients: {offenders}")
    scaler.step(optimizer)
    scaler.update()
    print(
        "image_smoke",
        {"project": project, "loss": float(loss.detach()),
         "grad_norm": float(grad_norm), "scale": float(scaler.get_scale())},
    )
print("peak_cuda_memory_gib", torch.cuda.max_memory_allocated() / 2**30)
PY

# First falsification run: one honest APTOS fold, one seed, end-to-end from
# epoch 1 with differential encoder/head learning rates.  Architecture choices
# must be made from best_validation_metrics.json, not the held-out test result.
python train_mosaic.py \
  --dataset aptos \
  --data_root Datasets/aptos2019-blindness-detection \
  --run_dir "${RUN_DIR}" \
  --folds 0 \
  "${RESUME_ARGS[@]}" \
  --skip_test \
  --local_stage rf_medium \
  --image_size 896 \
  --batch_size 4 \
  --epochs "${TOTAL_EPOCHS}" \
  --early_stop_patience "${EARLY_STOP_PATIENCE}" \
  --decision_rule "${DECISION_RULE}" \
  --max_count 32 \
  --dense_warmup_epochs 4 \
  --proof_ramp_epochs 4 \
  --proof_epsilon 0.02 \
  --necessity_fraction 0.5
