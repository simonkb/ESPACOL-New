#!/bin/bash
#SBATCH --job-name=paths_pre
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=24G
#SBATCH --time=00:30:00
#SBATCH --output=/dpc/kuin0170/ESPACOL-New/paths_preflight_%j.out
#SBATCH --error=/dpc/kuin0170/ESPACOL-New/paths_preflight_%j.err
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

# Git index flags such as assume-unchanged can hide a stale file after branch
# switching. Compare the executed implementation bytes with the recorded HEAD
# blobs; a scientific run must never mix PATHS with another branch's trainer.
for PATHS_FILE in \
  configs/paths_config.py configs/origin_config.py \
  Datasets/origin_data.py Datasets/mosaic_data.py Datasets/dataloaders.py \
  models/origin_encoder.py models/origin.py models/paths.py \
  losses/origin.py losses/paths.py \
  training/origin_trainer.py training/paths_trainer.py \
  train_paths.py utils/spatial_mask.py; do
  EXPECTED_BLOB="$(git rev-parse "HEAD:${PATHS_FILE}")"
  OBSERVED_BLOB="$(git hash-object "${PATHS_FILE}")"
  [[ "${EXPECTED_BLOB}" == "${OBSERVED_BLOB}" ]] || {
    echo "Tracked implementation differs from HEAD: ${PATHS_FILE}" >&2
    echo "Clear any assume-unchanged/skip-worktree flag and restore this file." >&2
    exit 2
  }
done

echo "=== PATHS structural preflight ==="
date --iso-8601=seconds
git rev-parse HEAD
git status --short
python --version

if ! python -c 'import pytest' >/dev/null 2>&1; then
  echo "Missing pytest in conda environment ${ORIGIN_CONDA_ENV:-G}." >&2
  echo "Install once: python -m pip install -r requirements-dev.txt" >&2
  exit 3
fi
python -m pytest -q tests/test_paths*.py tests/test_origin_core.py tests/test_origin_loss.py

python - <<'PY'
import torch
from torch.amp import GradScaler, autocast

from losses.paths import PathsLoss
from models.paths import PathsOutput, build_paths_model

if not torch.cuda.is_available():
    raise RuntimeError("PATHS preflight requires CUDA")
device = torch.device("cuda")
model = build_paths_model(
    num_classes=5,
    encoder_name="convnext_tiny",
    pretrained=False,
    evidence_scales=("s4", "s8", "s16", "s32"),
    projection_dim=32,
    reference_count=4096.0,
    atom_mode="cumulative",
    atom_rate_init=1e-6,
    prior_rate_init=1e-4,
    boundary_scale_init=1.0,
    total_rate_cap=64.0,
    prior_rate_cap=1.0,
    boundary_scale_cap=2.0,
    rate_roundoff_margin=1.0,
    paths_probe_z=(0.05, 0.20, 0.50, 0.80),
    paths_correction_cap=3.0,
    paths_gain_init=0.05,
    paths_strength=1.0,
).to(device).train()
criterion = PathsLoss(
    5,
    label_counts=[1300, 266, 719, 139, 212],
    risk_set_power=0.5,
    rps_weight=0.25,
).to(device)
optimizer = torch.optim.AdamW(
    [
        {"params": list(model.encoder.parameters()), "lr": 1e-5, "name": "encoder"},
        {
            "params": list(model.generator.parameters()),
            "lr": 5e-5,
            "name": "origin_v3_generator",
        },
        {
            "params": list(model.paths_refiner.parameters()),
            "lr": 5e-4,
            "name": "paths_refiner",
        },
    ],
    weight_decay=1e-5,
)

# A spatially constant, all-black tensor is a pathological ConvNeXt gradient
# test: with random weights and the extreme grade-4 target it sends the same
# reference-count-amplified signal into every stem location, producing a stem
# bias gradient around 1e10 even in FP32.  Scaling that by 256 must overflow
# FP16, although neither real fundus batches nor the staged PATHS warm start
# have this degeneracy.  Use a fixed, non-constant signal and a fundus-shaped
# valid mask so the canary exercises the actual mixed-precision path.
generator = torch.Generator(device="cpu").manual_seed(1729)
images = torch.rand(2, 3, 128, 128, generator=generator).to(device)
axis = torch.linspace(-1.0, 1.0, 128, device=device)
yy, xx = torch.meshgrid(axis, axis, indexing="ij")
valid = ((xx.square() + yy.square()) <= 0.95**2)[None, None].expand(2, 1, -1, -1)
labels = torch.tensor([0, 4], device=device)


def assert_finite_gradients(stage: str) -> float:
    offenders = []
    finite_gradients = 0
    maximum = 0.0
    for name, parameter in model.named_parameters():
        if parameter.grad is None:
            continue
        if not bool(torch.isfinite(parameter.grad).all()):
            offenders.append(name)
            continue
        finite_gradients += 1
        maximum = max(maximum, float(parameter.grad.detach().abs().max()))
    if offenders:
        raise RuntimeError(
            f"PATHS {stage} produced non-finite gradients in "
            + ", ".join(offenders[:16])
        )
    if finite_gradients == 0 or maximum == 0.0:
        raise RuntimeError(f"PATHS {stage} produced no nonzero finite gradient")
    print(f"{stage}_max_unscaled_gradient", maximum)
    return maximum


def forward_loss(*, amp: bool):
    with autocast(device_type="cuda", enabled=amp):
        output = model(images, valid, force_decoder_fp64=True)
    if not isinstance(output, PathsOutput):
        raise RuntimeError("PATHS strength-one canary bypassed the refiner")
    with autocast(device_type="cuda", enabled=False):
        loss, diagnostics = criterion(output, labels, epoch=0)
    if not bool(torch.isfinite(loss)):
        raise RuntimeError("PATHS canary loss is non-finite")
    return loss, diagnostics


# First validate the complete differentiable computation in FP32/FP64.  This
# distinguishes mathematical NaNs from mixed-precision range overflows.
for parameter in model.parameters():
    parameter.requires_grad_(True)
optimizer.zero_grad(set_to_none=True)
fp32_loss, _ = forward_loss(amp=False)
fp32_loss.backward()
assert_finite_gradients("full_precision_backward")

# Mirror epochs 1--3: the hash-bound V3 base is frozen and only the PATHS
# focality law is adapted.  The trainer begins this phase with scale 4096.
for name, parameter in model.named_parameters():
    parameter.requires_grad_(name.startswith("paths_refiner."))
optimizer.zero_grad(set_to_none=True)
frozen_scaler = GradScaler(
    "cuda", enabled=True, init_scale=4096.0, growth_interval=2000
)
frozen_loss, _ = forward_loss(amp=True)
frozen_scaler.scale(frozen_loss).backward()
frozen_scaler.unscale_(optimizer)
assert_finite_gradients("correction_only_amp_backward")
frozen_scaler.step(optimizer)
frozen_scaler.update()

# Mirror the first joint epoch: all groups become trainable and the trainer
# deliberately resets the AMP scale to 256 before any joint update.
for parameter in model.parameters():
    parameter.requires_grad_(True)
optimizer.zero_grad(set_to_none=True)
joint_scaler = GradScaler(
    "cuda", enabled=True, init_scale=256.0, growth_interval=2000
)
loss, diagnostics = forward_loss(amp=True)
joint_scaler.scale(loss).backward()
joint_scaler.unscale_(optimizer)
assert_finite_gradients("joint_amp_backward")
joint_scaler.step(optimizer)
joint_scaler.update()
print("gpu", torch.cuda.get_device_name(0))
print("canary_loss", float(loss.detach()), diagnostics)
print("boundary_weights", criterion.boundary_weights.detach().cpu().tolist())
print("architecture", model.architecture_metadata())
PY

echo "PATHS structural preflight passed."
