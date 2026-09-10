#!/bin/bash
# Full inner-validation structural-faithfulness audit. This job encodes only
# validation images; its interventions replay the stored evidence ledger.
# Outer-test files are opaque-byte hashed for split provenance, never decoded.
#SBATCH --job-name=origin_v3_audit
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=0-04:00:00
#SBATCH --output=/dpc/kuin0170/ESPACOL-New/origin_v3_audit_%j.out
#SBATCH --error=/dpc/kuin0170/ESPACOL-New/origin_v3_audit_%j.err
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

CHECKPOINT="${ORIGIN_CHECKPOINT:-runs/origin_dr_f0_v3_bounded/fold0/best.pth}"
DATA_ROOT="${ORIGIN_DR_ROOT:-Datasets/DR}"
OUTPUT="${ORIGIN_AUDIT_OUTPUT:-runs/origin_dr_f0_v3_bounded/fold0/audits/full_validation_audit_v2.json}"

[[ -f "${CHECKPOINT}" ]] || { echo "Missing checkpoint: ${CHECKPOINT}" >&2; exit 2; }
[[ -f "$(dirname "${CHECKPOINT}")/split_manifest.json" ]] || {
  echo "Missing authoritative split manifest beside checkpoint." >&2
  exit 3
}
[[ ! -e "${OUTPUT}" ]] || {
  echo "Refusing to overwrite existing audit: ${OUTPUT}" >&2
  exit 4
}

if [[ -n "${ORIGIN_EXPECTED_SCALES:-}" ]]; then
  ORIGIN_AUDIT_CHECKPOINT="${CHECKPOINT}" python - <<'PY'
import os
import torch

checkpoint = os.environ["ORIGIN_AUDIT_CHECKPOINT"]
expected = tuple(
    part.strip()
    for part in os.environ["ORIGIN_EXPECTED_SCALES"].split(",")
    if part.strip()
)
state = torch.load(checkpoint, map_location="cpu", weights_only=False)
actual = tuple(state.get("config", {}).get("evidence_scales", ()))
if actual != expected:
    raise RuntimeError(
        f"audit checkpoint evidence scales {actual} do not match expected {expected}"
    )
print("expected_evidence_scales", expected)
PY
fi

echo "=== ORIGIN-v3 EyePACS full inner-validation structural audit ==="
date --iso-8601=seconds
git rev-parse HEAD
git status --short
python --version
python -m pytest -q tests/test_origin_validation_audit.py tests/test_origin_intervention.py

python audit_origin_validation.py \
  --checkpoint "${CHECKPOINT}" \
  --data_root "${DATA_ROOT}" \
  --output "${OUTPUT}" \
  --batch_size 8 \
  --num_workers "${ORIGIN_AUDIT_NUM_WORKERS:-8}" \
  --top_ks "${ORIGIN_AUDIT_TOP_KS:-1,5,10}" \
  --certificates_per_grade "${ORIGIN_AUDIT_CERTIFICATES_PER_GRADE:-2}"
