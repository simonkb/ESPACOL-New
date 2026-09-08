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
from torch.amp import GradScaler, autocast
from losses.origin import OriginLoss
from models.origin import build_origin_model, decode_pure_birth_rates

if not torch.cuda.is_available():
    raise RuntimeError("ORIGIN preflight requires CUDA")
device = torch.device("cuda")

# Reproduce the exact high/unequal-rate regime that invalidated the v1
# experiment, and verify both the CUDA forward law and its derivatives against
# independent 100-decimal reference values.
audit_rates = torch.tensor(
    [[141.475647, 426.073364, 31.953876, 70.849700]],
    dtype=torch.float64,
    device=device,
    requires_grad=True,
)
audit = decode_pure_birth_rates(audit_rates, force_fp64=True)
reference_probs = torch.tensor(
    [[
        3.613326317285435e-62,
        1.796211452251686e-62,
        1.852020049106147e-14,
        1.521480017974467e-14,
        0.9999999999999663,
    ]],
    dtype=torch.float64,
    device=device,
)
torch.testing.assert_close(audit.class_probs, reference_probs, atol=0.0, rtol=2e-12)
reference_gradients = torch.tensor(
    [
        [1.0, 0.0, 0.0, 0.0],
        [0.9894179140547460, 0.003513731629828921, 0.0, 0.0],
        [0.002062250411556591, 0.0001902879338250475, 0.9883320936920318, 0.0],
        [0.002062250411556591, 0.0001902879338250475, 0.9313272848391561,
         0.02570970086660201],
    ],
    dtype=torch.float64,
    device=device,
)
for target in range(4):
    gradient = torch.autograd.grad(
        -audit.log_class_probs[0, target],
        audit_rates,
        retain_graph=True,
    )[0]
    torch.testing.assert_close(
        gradient,
        reference_gradients[target : target + 1],
        atol=2e-12,
        rtol=2e-11,
    )
audit_rates_fp32 = audit_rates.detach().float().requires_grad_(True)
audit_fp32 = decode_pure_birth_rates(audit_rates_fp32, force_fp64=True)
gradient_fp32 = torch.autograd.grad(
    -audit_fp32.log_class_probs[0, 1], audit_rates_fp32
)[0]
torch.testing.assert_close(
    gradient_fp32,
    reference_gradients[1:2].float(),
    atol=2e-6,
    rtol=2e-5,
)
print("high_rate_decoder_audit", "passed")

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
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
scaler = GradScaler(
    "cuda",
    enabled=True,
    init_scale=256.0,
    growth_interval=2000,
)
images = torch.zeros(2, 3, 128, 128, device=device)
valid = torch.ones(2, 1, 128, 128, dtype=torch.bool, device=device)
labels = torch.tensor([0, 4], device=device)
optimizer.zero_grad(set_to_none=True)
with autocast(device_type="cuda", enabled=True):
    output = model(images, valid, force_decoder_fp64=True)
with autocast(device_type="cuda", enabled=False):
    loss, diagnostics = criterion(output, labels, epoch=0)
if not torch.isfinite(loss):
    raise RuntimeError(f"non-finite ORIGIN canary loss: {loss}")
scaler.scale(loss).backward()
scaler.unscale_(optimizer)
finite_gradients = all(
    parameter.grad is None or bool(torch.isfinite(parameter.grad).all())
    for parameter in model.parameters()
)
if not finite_gradients:
    raise RuntimeError("ORIGIN canary produced a non-finite gradient")
scaler.step(optimizer)
scaler.update()
print("gpu", torch.cuda.get_device_name(0))
print("canary_loss", float(loss.detach()), "diagnostics", diagnostics)
print("canary_amp_scale", float(scaler.get_scale()))
print("class_probs", output.class_probs.detach().cpu().tolist())
PY

echo "ORIGIN preflight passed."
