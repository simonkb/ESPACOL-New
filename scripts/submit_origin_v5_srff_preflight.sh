#!/bin/bash
# Structural and numerical canary for the fixed-window ORIGIN-v5 SRFF model.
# It consumes no dataset or outer-test image.
#SBATCH --job-name=origin_v5_spre
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=/dpc/kuin0170/ESPACOL-New/origin_v5_srff_preflight_%j.out
#SBATCH --error=/dpc/kuin0170/ESPACOL-New/origin_v5_srff_preflight_%j.err
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
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

echo "=== ORIGIN-v5 SRFF structural preflight ==="
date --iso-8601=seconds
git rev-parse HEAD
git status --short
python --version

if ! python -c 'import pytest' >/dev/null 2>&1; then
  echo "Missing pytest. Install requirements-dev.txt inside the active environment." >&2
  exit 2
fi
python -m pytest -q tests/test_origin*.py

python - <<'PY'
import json
import torch
from torch.amp import GradScaler, autocast

from configs.origin_config import OriginConfig
from losses.origin import OriginLoss
from models.origin import build_origin_model

if not torch.cuda.is_available():
    raise RuntimeError("ORIGIN-v5 SRFF preflight requires CUDA")

cfg = OriginConfig(
    encoder="convnext_tiny_srff",
    evidence_scales=("s4", "s8", "s128"),
)
if cfg.img_size != 640 or cfg.evidence_scales != ("s4", "s8", "s128"):
    raise RuntimeError(f"unexpected SRFF configuration: {cfg}")

device = torch.device("cuda")
model = build_origin_model(
    num_classes=5,
    encoder_name=cfg.encoder,
    pretrained=True,
    evidence_scales=cfg.evidence_scales,
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
expected_metadata = {
    "evidence_dependency_policy": "spatial_receptive_field_firewall_v1",
    "srff_window_cells": 16,
    "max_evidence_receptive_field": 344,
}
for key, expected in expected_metadata.items():
    if metadata.get(key) != expected:
        raise RuntimeError(
            f"SRFF architecture metadata {key}={metadata.get(key)!r}; "
            f"expected {expected!r}"
        )
if metadata.get("evidence_scales") != ["s4", "s8", "s128"]:
    raise RuntimeError(f"unexpected SRFF evidence scales: {metadata}")
if metadata.get("sealed_source_masking") is not True:
    raise RuntimeError("SRFF checkpoint does not declare immutable source masking")
spatial_contract = metadata.get("encoder_spatial_contract", {})
for key, expected in {
    "kind": "nonoverlapping_feature_window_firewall_v1",
    "source_scale": "s8",
    "window_cells": 16,
    "window_overlap_cells": 0,
    "sealed_scale": "s128",
    "sealed_receptive_field": 344,
    "source_masking_inside_trunk": True,
}.items():
    if spatial_contract.get(key) != expected:
        raise RuntimeError(
            f"SRFF spatial contract {key}={spatial_contract.get(key)!r}; "
            f"expected {expected!r}"
        )
if not metadata.get("no_classifier_bypass", False):
    raise RuntimeError("SRFF model omitted the no-classifier-bypass invariant")

criterion = OriginLoss(num_classes=5, rps_weight=0.25).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
scaler = GradScaler("cuda", enabled=True, init_scale=256.0, growth_interval=2000)
images = torch.zeros(1, 3, 640, 640, device=device)
valid = torch.ones(1, 1, 640, 640, dtype=torch.bool, device=device)
labels = torch.tensor([2], device=device)
optimizer.zero_grad(set_to_none=True)
with autocast(device_type="cuda", enabled=True):
    output = model(images, valid, force_decoder_fp64=True)

if tuple(output.local_rate_maps) != ("s4", "s8", "s128"):
    raise RuntimeError(f"unexpected SRFF ledger: {tuple(output.local_rate_maps)}")
if tuple(output.local_rate_maps["s128"].shape) != (1, 5, 5, 4):
    raise RuntimeError(
        f"SRFF 640-pixel regional ledger has shape "
        f"{tuple(output.local_rate_maps['s128'].shape)}, expected (1,5,5,4)"
    )
if tuple(output.local_rate_maps_channels_first["s128"].shape) != (1, 4, 5, 5):
    raise RuntimeError(
        "SRFF internal channels-first regional ledger violated its "
        "(N,K-1,H,W) contract"
    )
regional = output.metadata["s128"]
geometry = (
    int(regional.output_stride),
    int(regional.receptive_field),
    float(regional.center_offset),
    tuple(regional.lattice_size),
)
if geometry != (128, 344, 64.0, (5, 5)):
    raise RuntimeError(f"unexpected SRFF regional geometry: {geometry}")
if bool(regional.globally_mixed):
    raise RuntimeError("SRFF regional evidence was marked globally mixed")
if float(output.total_rates.max()) > cfg.total_rate_cap:
    raise RuntimeError("ORIGIN-v5 SRFF violated the architectural rate cap")

with autocast(device_type="cuda", enabled=False):
    loss, diagnostics = criterion(output, labels, epoch=0)
if not bool(torch.isfinite(loss)):
    raise RuntimeError("ORIGIN-v5 SRFF canary loss is non-finite")
scaler.scale(loss).backward()
scaler.unscale_(optimizer)
if not all(
    parameter.grad is None or bool(torch.isfinite(parameter.grad).all())
    for parameter in model.parameters()
):
    raise RuntimeError("ORIGIN-v5 SRFF canary produced a non-finite gradient")
scaler.step(optimizer)
scaler.update()

print("gpu", torch.cuda.get_device_name(0))
print("architecture_metadata", json.dumps(metadata, sort_keys=True))
print("s128_geometry", geometry)
print("canary_loss", float(loss.detach()), "diagnostics", diagnostics)
print("total_rates", output.total_rates.detach().cpu().tolist())
PY

echo "ORIGIN-v5 SRFF preflight passed."
