#!/bin/bash
# Full inner-validation structural and intervention audit for an eligible
# ORIGIN-v7 identified-sparse checkpoint. Defaults to EyePACS fold 0.
#SBATCH --job-name=origin_v7_aud
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=0-06:00:00
#SBATCH --output=/dpc/kuin0170/ESPACOL-New/origin_v7_audit_%j.out
#SBATCH --error=/dpc/kuin0170/ESPACOL-New/origin_v7_audit_%j.err
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

RUN_DIR="${ORIGIN_V7_RUN_DIR:-runs/origin_dr_f0_v7_identified_sparse_k8}"
CHECKPOINT="${ORIGIN_CHECKPOINT:-${RUN_DIR}/fold0/best.pth}"
DATA_ROOT="${ORIGIN_DATA_ROOT:-Datasets/DR}"
OUTPUT="${ORIGIN_AUDIT_OUTPUT:-${RUN_DIR}/fold0/audits/full_validation_audit_v7.json}"

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
import json
import os
import torch

state = torch.load(os.environ["ORIGIN_AUDIT_CHECKPOINT"], map_location="cpu", weights_only=False)
result_path = os.path.join(os.path.dirname(os.environ["ORIGIN_AUDIT_CHECKPOINT"]), "result.json")
with open(result_path) as stream:
    result = json.load(stream)
if state.get("schema") != "origin-checkpoint-v7":
    raise RuntimeError("identified-sparse audit requires origin-checkpoint-v7")
if state.get("relation_protocol_version") != "v7":
    raise RuntimeError("checkpoint does not declare relation protocol v7")
config = state.get("config", {})
if config.get("relation_variant") != "identified_sparse_v1":
    raise RuntimeError("this audit wrapper accepts only the v7 target, not a control")
declared = state.get("architecture", {}).get("declared", {})
contract = declared.get("relation_contract")
if not isinstance(contract, dict) or contract.get("kind") != (
    "identified_sparse_cumulative_ordinal_pair_log_odds_v1"
):
    raise RuntimeError("checkpoint lacks the registered v7 relation contract")
if contract.get("incremental_compilation") != (
    "fixed_budget_linear_stored_edge_log_odds"
):
    raise RuntimeError("checkpoint does not use the registered linear edge compiler")
if contract.get("pair_identification") != (
    "pre_geometry_anova_then_geometry_then_post_geometry_anova"
):
    raise RuntimeError("checkpoint does not use the registered two-stage identification")
if int(contract.get("edge_budget", -1)) != int(config.get("relation_edge_budget", -2)):
    raise RuntimeError("checkpoint relation contract and configuration budgets differ")
floor = state.get("warm_start_metric_safety_floor")
eligibility = state.get("candidate_metric_safety_floor_evaluation")
if not isinstance(floor, dict) or floor.get("schema") != (
    "origin-v7-v3-multimetric-safety-floor-v1"
):
    raise RuntimeError("checkpoint lacks the registered v3 safety floor")
if not isinstance(eligibility, dict) or not eligibility.get("checkpoint_eligible", False):
    raise RuntimeError("selected checkpoint did not pass its registered safety floor")
if int(state.get("epoch", 0)) <= 0 or int(result.get("best_epoch", 0)) <= 0:
    raise RuntimeError("v7 target audit refuses the epoch-0 v3 fallback")
if bool(result.get("best_checkpoint_is_hash_bound_v3_floor", True)):
    raise RuntimeError("v7 target audit requires a learned checkpoint above the v3 floor")
print("relation_contract", contract)
print("warm_start_provenance", state.get("warm_start_provenance"))
print("warm_start_metric_safety_floor", floor)
print("selected_checkpoint_safety_floor_evaluation", eligibility)
PY

echo "=== ORIGIN-v7 full inner-validation structural audit ==="
date --iso-8601=seconds
git rev-parse HEAD
git status --short
python --version
python -m pytest -q \
  tests/test_origin_validation_audit.py \
  tests/test_origin_intervention.py \
  tests/test_origin_relations.py \
  tests/test_origin_v7_protocol.py \
  tests/test_origin_v7_validation_audit.py

python audit_origin_validation.py \
  --checkpoint "${CHECKPOINT}" \
  --data_root "${DATA_ROOT}" \
  --output "${OUTPUT}" \
  --batch_size 8 \
  --num_workers "${ORIGIN_AUDIT_NUM_WORKERS:-8}" \
  --top_ks "${ORIGIN_AUDIT_TOP_KS:-1,5,8}" \
  --certificates_per_grade "${ORIGIN_AUDIT_CERTIFICATES_PER_GRADE:-2}"
