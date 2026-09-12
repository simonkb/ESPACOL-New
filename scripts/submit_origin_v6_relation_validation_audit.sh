#!/bin/bash
# Full inner-validation unary-and-relation structural audit for ORIGIN-v6.
# Defaults to the EyePACS pilot; override checkpoint/data/output for APTOS.
#SBATCH --job-name=origin_v6_raud
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=0-06:00:00
#SBATCH --output=/dpc/kuin0170/ESPACOL-New/origin_v6_relation_audit_%j.out
#SBATCH --error=/dpc/kuin0170/ESPACOL-New/origin_v6_relation_audit_%j.err
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

RUN_DIR="${ORIGIN_V6_RUN_DIR:-runs/origin_dr_f0_v6_relation_only}"
CHECKPOINT="${ORIGIN_CHECKPOINT:-${RUN_DIR}/fold0/best.pth}"
DATA_ROOT="${ORIGIN_DATA_ROOT:-Datasets/DR}"
OUTPUT="${ORIGIN_AUDIT_OUTPUT:-${RUN_DIR}/fold0/audits/full_validation_audit_v6.json}"

[[ -f "${CHECKPOINT}" ]] || { echo "Missing checkpoint: ${CHECKPOINT}" >&2; exit 2; }
[[ -f "$(dirname "${CHECKPOINT}")/result.json" ]] || {
  echo "Audit requires a completed fold result.json." >&2
  exit 3
}
[[ ! -e "${OUTPUT}" ]] || {
  echo "Refusing to overwrite existing audit: ${OUTPUT}" >&2
  exit 4
}

ORIGIN_AUDIT_CHECKPOINT="${CHECKPOINT}" python - <<'PY'
import os
import torch

state = torch.load(os.environ["ORIGIN_AUDIT_CHECKPOINT"], map_location="cpu", weights_only=False)
if state.get("schema") != "origin-checkpoint-v6":
    raise RuntimeError("relation audit requires origin-checkpoint-v6")
config = state.get("config", {})
if not config.get("relation_enabled", False):
    raise RuntimeError("checkpoint does not enable the relation ledger")
declared = state.get("architecture", {}).get("declared", {})
contract = declared.get("relation_contract")
if not isinstance(contract, dict) or contract.get("kind") != "cumulative_ordinal_pair_rate_log_odds_v1":
    raise RuntimeError("checkpoint lacks the registered v6 relation contract")
floor = state.get("warm_start_metric_safety_floor")
eligibility = state.get("candidate_metric_safety_floor_evaluation")
if not isinstance(floor, dict):
    raise RuntimeError("checkpoint lacks the registered v3 multi-metric safety floor")
if not isinstance(eligibility, dict) or not eligibility.get("checkpoint_eligible", False):
    raise RuntimeError("selected checkpoint did not pass its registered safety floor")
print("relation_contract", contract)
print("warm_start_provenance", state.get("warm_start_provenance"))
print("warm_start_metric_safety_floor", floor)
print("selected_checkpoint_safety_floor_evaluation", eligibility)
PY

echo "=== ORIGIN-v6 full inner-validation relation audit ==="
date --iso-8601=seconds
git rev-parse HEAD
git status --short
python --version
python -m pytest -q tests/test_origin_validation_audit.py tests/test_origin_intervention.py tests/test_origin_relations.py

python audit_origin_validation.py \
  --checkpoint "${CHECKPOINT}" \
  --data_root "${DATA_ROOT}" \
  --output "${OUTPUT}" \
  --batch_size 8 \
  --num_workers "${ORIGIN_AUDIT_NUM_WORKERS:-8}" \
  --top_ks "${ORIGIN_AUDIT_TOP_KS:-1,5,10}" \
  --certificates_per_grade "${ORIGIN_AUDIT_CERTIFICATES_PER_GRADE:-2}"
