#!/bin/bash
#SBATCH --job-name=origin_preflight
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=24G
#SBATCH --time=00:30:00
#SBATCH --output=/dpc/kuin0170/ESPACOL-New/origin_preflight_%j.out
#SBATCH --error=/dpc/kuin0170/ESPACOL-New/origin_preflight_%j.err
#SBATCH --account=kuin0170

set -eo pipefail
# lmod inspects scheduler-specific variables that are not always defined.
set +u
source /etc/profile.d/lmod.sh || exit 1
module load miniconda/3 || exit 1
module load cuda/12.6 || exit 1
source activate "${ORIGIN_CONDA_ENV:-G}" || exit 1
set -u

REPO_ROOT="${ORIGIN_REPO_ROOT:-/dpc/kuin0170/ESPACOL-New}"
cd "${REPO_ROOT}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "=== ORIGIN structural preflight ==="
date --iso-8601=seconds
git rev-parse HEAD
git status --short
python --version
python - <<'PY'
import json
from dataclasses import asdict
from configs.origin_config import OriginConfig
print("default_config=" + json.dumps(asdict(OriginConfig()), sort_keys=True))
PY

if ! python -c 'import pytest' >/dev/null 2>&1; then
  echo "Missing pytest in conda environment ${ORIGIN_CONDA_ENV:-G}." >&2
  echo "Install once: python -m pip install -r requirements-dev.txt" >&2
  exit 2
fi

python -m pytest -q tests/test_origin*.py

python - <<'PY'
import torch
from losses.origin import OriginLoss
from models.origin import build_origin_model

if not torch.cuda.is_available():
    raise RuntimeError("ORIGIN preflight requires CUDA")
device = torch.device("cuda")
model = build_origin_model(
    num_classes=5,
    encoder_name="convnext_tiny",
    # This intentionally exercises the exact pretrained initialization used
    # by training.  A missing/corrupt weight cache must fail here; there is no
    # silent random-initialization fallback.
    pretrained=True,
    evidence_scales=("s4", "s8", "s16", "s32"),
    projection_dim=32,
    reference_count=4096.0,
    atom_rate_init=1e-6,
    prior_rate_init=1e-4,
    atom_mode="cumulative",
).to(device).train()
criterion = OriginLoss(num_classes=5, rps_weight=0.25).to(device)
images = torch.zeros(2, 3, 128, 128, device=device)
valid = torch.ones(2, 1, 128, 128, dtype=torch.bool, device=device)
labels = torch.tensor([0, 4], device=device)
output = model(images, valid, force_decoder_fp32=True)
loss, diagnostics = criterion(output, labels, epoch=0)
if not torch.isfinite(loss):
    raise RuntimeError(f"non-finite ORIGIN canary loss: {loss}")
loss.backward()
finite_gradients = all(
    parameter.grad is None or bool(torch.isfinite(parameter.grad).all())
    for parameter in model.parameters()
)
if not finite_gradients:
    raise RuntimeError("ORIGIN canary produced a non-finite gradient")
print("gpu", torch.cuda.get_device_name(0))
print("canary_loss", float(loss.detach()), "diagnostics", diagnostics)
print("class_probs", output.class_probs.detach().cpu().tolist())
PY

echo "ORIGIN preflight passed."
