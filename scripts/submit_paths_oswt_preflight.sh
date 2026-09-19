#!/bin/bash
# Structural, provenance, and CUDA-gradient preflight for PATHS-OSWT.
#SBATCH --job-name=oswt_pre
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=24G
#SBATCH --time=00:30:00
#SBATCH --output=/dpc/kuin0170/ESPACOL-New/paths_oswt_preflight_%j.out
#SBATCH --error=/dpc/kuin0170/ESPACOL-New/paths_oswt_preflight_%j.err
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

: "${PATHS_OSWT_APTOS_V3_SHA256:?export the audited APTOS V3 SHA-256}"
: "${PATHS_OSWT_DR_V3_SHA256:?export the audited EyePACS V3 SHA-256}"
APTOS_V3="${PATHS_OSWT_APTOS_V3_CHECKPOINT:-runs/origin_aptos_f0_v3_bounded/fold0/best.pth}"
DR_V3="${PATHS_OSWT_DR_V3_CHECKPOINT:-runs/origin_dr_f0_v3_bounded/fold0/best.pth}"

[[ -f "${APTOS_V3}" && -f "${DR_V3}" ]] || {
  echo "Both audited V3 checkpoints must exist." >&2
  exit 2
}
[[ "$(sha256sum "${APTOS_V3}" | awk '{print $1}')" == "${PATHS_OSWT_APTOS_V3_SHA256}" ]] || {
  echo "APTOS V3 SHA-256 mismatch." >&2
  exit 2
}
[[ "$(sha256sum "${DR_V3}" | awk '{print $1}')" == "${PATHS_OSWT_DR_V3_SHA256}" ]] || {
  echo "EyePACS V3 SHA-256 mismatch." >&2
  exit 2
}

OSWT_FILES=(
  configs/paths_oswt_config.py configs/paths_config.py configs/origin_config.py
  Datasets/origin_data.py Datasets/mosaic_data.py Datasets/dataloaders.py
  models/origin_encoder.py models/origin.py models/paths.py models/paths_oswt.py
  losses/origin.py losses/paths.py
  training/origin_trainer.py training/paths_trainer.py training/paths_oswt_trainer.py
  train_paths_oswt.py train_origin.py utils/spatial_mask.py
  scripts/submit_paths_oswt_preflight.sh scripts/submit_paths_oswt_aptos_f0.sh
  scripts/submit_paths_oswt_aptos_ungated_f0.sh
  scripts/submit_paths_oswt_promotion_gate.sh scripts/submit_paths_oswt_dr_f0.sh
  scripts/launch_paths_oswt_v8_aptos_f0.sh
)
for path in "${OSWT_FILES[@]}"; do
  [[ "$(git rev-parse "HEAD:${path}")" == "$(git hash-object "${path}")" ]] || {
    echo "Tracked implementation differs from HEAD: ${path}" >&2
    exit 3
  }
done

echo "=== PATHS-V8 OSWT structural preflight ==="
date --iso-8601=seconds
git rev-parse HEAD
git status --short
python --version
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

python -c 'import pytest' >/dev/null 2>&1 || {
  echo "pytest missing in ${ORIGIN_CONDA_ENV:-G}; install requirements-dev.txt once." >&2
  exit 4
}
for script in scripts/submit_paths_oswt_*.sh scripts/launch_paths_oswt_v8_aptos_f0.sh; do
  bash -n "${script}"
done
python -m pytest -q \
  tests/test_paths_oswt_core.py tests/test_paths_oswt_integration.py \
  tests/test_origin_core.py tests/test_origin_loss.py tests/test_paths_loss.py

python - <<'PY'
import torch
import torch.nn.functional as F

from models.paths_oswt import apply_shell_warranted_transport

if not torch.cuda.is_available():
    raise RuntimeError("OSWT preflight requires CUDA")
device = torch.device("cuda")
base_logits = torch.tensor(
    [[-1.1, -0.8, 1.4, 0.2, -0.5], [-1.2, -0.7, -0.3, 0.1, 1.3]],
    device=device, dtype=torch.float64,
)
left_logits = torch.tensor(
    [[0.8, 0.4, -1.4, 1.3], [0.8, 0.4, -1.4, 1.3]],
    device=device, dtype=torch.float64, requires_grad=True,
)
right_logits = torch.tensor(
    [[-0.5, 0.2, 1.5, -1.2], [-0.5, 0.2, 1.5, -1.2]],
    device=device, dtype=torch.float64, requires_grad=True,
)
raw_beta = torch.zeros(4, device=device, dtype=torch.float64, requires_grad=True)
raw_tau = torch.full((4,), -2.0, device=device, dtype=torch.float64, requires_grad=True)
result = apply_shell_warranted_transport(
    base_logits.softmax(-1), left_logits.sigmoid(), right_logits.sigmoid(),
    F.softplus(raw_beta), F.softplus(raw_tau),
)
loss = -result.log_class_probs[:, 3].mean()
loss.backward()
for name, value in (
    ("left_logits", left_logits), ("right_logits", right_logits),
    ("raw_beta", raw_beta), ("raw_tau", raw_tau)
):
    if value.grad is None or not bool(torch.isfinite(value.grad).all()):
        raise RuntimeError(f"OSWT CUDA canary has invalid {name} gradient")
if not bool(torch.allclose(result.class_probs.sum(-1), torch.ones(2, device=device, dtype=torch.float64))):
    raise RuntimeError("OSWT CUDA canary posterior is not normalized")
print("gpu", torch.cuda.get_device_name(0))
print("canary_loss", float(loss.detach()))
print("log_odds_increment", result.log_odds_increment.detach().cpu().tolist())
print("signed_flow", result.signed_boundary_flow.detach().cpu().tolist())
PY

echo "PATHS-V8 OSWT structural preflight passed."
