#!/bin/bash
#SBATCH --job-name=origin_v3_pre
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=24G
#SBATCH --time=00:30:00
#SBATCH --output=/dpc/kuin0170/ESPACOL-New/origin_v3_preflight_%j.out
#SBATCH --error=/dpc/kuin0170/ESPACOL-New/origin_v3_preflight_%j.err
#SBATCH --account=kuin0170

set -eo pipefail
set +u
source /etc/profile.d/lmod.sh || exit 1
module load miniconda/3 || exit 1
module load cuda/12.6 || exit 1
source activate "${ORIGIN_CONDA_ENV:-G}" || exit 1
set -u

REPO_ROOT="${ORIGIN_REPO_ROOT:-/dpc/kuin0170/ESPACOL-New}"
cd "${REPO_ROOT}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "=== ORIGIN-v3 bounded-rate structural preflight ==="
date --iso-8601=seconds
git rev-parse HEAD
git status --short
python --version

if ! python -c 'import pytest' >/dev/null 2>&1; then
  echo "Missing pytest. Install requirements-dev.txt first." >&2
  exit 2
fi
python -m pytest -q tests/test_origin*.py

python - <<'PY'
import json
from dataclasses import asdict
import torch
from torch.amp import GradScaler, autocast

from configs.origin_config import OriginConfig
from losses.origin import OriginLoss
from models.origin import build_origin_model

cfg = OriginConfig()
print("default_config=" + json.dumps(asdict(cfg), sort_keys=True))
assert cfg.total_rate_cap == 64.0
assert cfg.prior_rate_cap == 1.0
assert cfg.boundary_scale_cap == 2.0
assert cfg.rate_roundoff_margin == 1.0
if not torch.cuda.is_available():
    raise RuntimeError("ORIGIN-v3 preflight requires CUDA")
device = torch.device("cuda")
model = build_origin_model(
    num_classes=5,
    encoder_name="convnext_tiny",
    pretrained=True,
    evidence_scales=("s4", "s8", "s16", "s32"),
    projection_dim=32,
    reference_count=cfg.reference_count,
    atom_rate_init=cfg.atom_rate_init,
    prior_rate_init=cfg.prior_rate_init,
    boundary_scale_init=cfg.boundary_scale_init,
    total_rate_cap=cfg.total_rate_cap,
    prior_rate_cap=cfg.prior_rate_cap,
    boundary_scale_cap=cfg.boundary_scale_cap,
    rate_roundoff_margin=cfg.rate_roundoff_margin,
    atom_mode="cumulative",
).to(device).train()
metadata = model.architecture_metadata()
if metadata.get("rate_parameterization") != "bounded_null_simplex_v1":
    raise RuntimeError(f"unexpected ORIGIN-v3 parameterization: {metadata}")
for key, expected in {
    "total_rate_cap": 64.0,
    "prior_rate_cap": 1.0,
    "boundary_scale_cap": 2.0,
    "rate_roundoff_margin": 1.0,
}.items():
    if float(metadata.get(key, float("nan"))) != expected:
        raise RuntimeError(f"architecture metadata omitted {key}={expected}")

criterion = OriginLoss(num_classes=5, rps_weight=0.25).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
scaler = GradScaler("cuda", enabled=True, init_scale=256.0, growth_interval=2000)
images = torch.zeros(2, 3, 128, 128, device=device)
valid = torch.ones(2, 1, 128, 128, dtype=torch.bool, device=device)
labels = torch.tensor([0, 4], device=device)
optimizer.zero_grad(set_to_none=True)
with autocast(device_type="cuda", enabled=True):
    output = model(images, valid, force_decoder_fp64=True)
if float(output.total_rates.max()) > cfg.total_rate_cap:
    raise RuntimeError("ORIGIN-v3 violated total_rate_cap")
with autocast(device_type="cuda", enabled=False):
    loss, diagnostics = criterion(output, labels, epoch=0)
if not torch.isfinite(loss):
    raise RuntimeError("ORIGIN-v3 canary loss is non-finite")
scaler.scale(loss).backward()
scaler.unscale_(optimizer)
if not all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in model.parameters()):
    raise RuntimeError("ORIGIN-v3 canary produced a non-finite gradient")
scaler.step(optimizer)
scaler.update()
print("gpu", torch.cuda.get_device_name(0))
print("architecture_metadata", json.dumps(metadata, sort_keys=True))
print("canary_loss", float(loss.detach()), "diagnostics", diagnostics)
print("total_rates", output.total_rates.detach().cpu().tolist())
PY

echo "ORIGIN-v3 bounded-rate preflight passed."
