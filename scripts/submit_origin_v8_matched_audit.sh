#!/bin/bash
# Production full-inner-validation audit for each completed member of the
# ORIGIN-v8 APTOS fold-0 matched experiment. Each array task audits the exact
# best_learned.pth selected by its corresponding training task; no outer-test
# images are loaded and no full-CV work is authorized.
#SBATCH --job-name=origin_v8_audit
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --array=0-2
#SBATCH --output=/dpc/kuin0170/ESPACOL-New/origin_v8_audit_%A_%a.out
#SBATCH --error=/dpc/kuin0170/ESPACOL-New/origin_v8_audit_%A_%a.err
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
: "${ORIGIN_EXPECTED_COMMIT:?Set ORIGIN_EXPECTED_COMMIT to the training commit}"
[[ "$(git rev-parse HEAD)" == "${ORIGIN_EXPECTED_COMMIT}" ]] || {
  echo "Audit commit differs from the registered training commit." >&2
  exit 2
}

RUN_NAMES=(
  origin_aptos_f0_v8_cmwa_r1_target
  origin_aptos_f0_v8_cmwa_r1_endpoint_control
  origin_aptos_f0_v8_cmwa_r1_shuffled_control
)
TASK_ID="${SLURM_ARRAY_TASK_ID:-}"
[[ "${TASK_ID}" =~ ^[0-2]$ ]] || {
  echo "This script must run as the registered Slurm audit array 0-2." >&2
  exit 3
}

RUN_ROOT="${ORIGIN_V8_RUN_ROOT:-runs}"
FOLD_DIR="${RUN_ROOT}/${RUN_NAMES[${TASK_ID}]}/fold0"
CHECKPOINT="${FOLD_DIR}/best_learned.pth"
RESULT="${FOLD_DIR}/result.json"
AUDIT_DIR="${FOLD_DIR}/audits"
AUDIT_OUTPUT="${AUDIT_DIR}/full_validation_audit_v8_best_learned.json"
DATA_ROOT="${ORIGIN_APTOS_ROOT:-Datasets/aptos2019-blindness-detection}"

[[ -f "${CHECKPOINT}" && -f "${RESULT}" ]] || {
  echo "Audit requires completed best-learned artifacts in ${FOLD_DIR}." >&2
  exit 4
}
[[ ! -e "${AUDIT_OUTPUT}" ]] || {
  echo "Refusing to overwrite registered audit: ${AUDIT_OUTPUT}" >&2
  exit 5
}
mkdir -p "${AUDIT_DIR}"

echo "=== ORIGIN-v8 APTOS fold-0 best-learned structural audit ==="
echo "evaluation_scope=inner_validation_only"
echo "outer_test_authorized=false"
echo "full_cross_validation_authorized=false"
echo "checkpoint=${CHECKPOINT}"
echo "audit_output=${AUDIT_OUTPUT}"
date --iso-8601=seconds
git rev-parse HEAD
python --version

python audit_origin_validation.py \
  --checkpoint "${CHECKPOINT}" \
  --data_root "${DATA_ROOT}" \
  --output "${AUDIT_OUTPUT}" \
  --batch_size 8 \
  --num_workers "${ORIGIN_NUM_WORKERS:-8}" \
  --top_ks 1,5,10 \
  --certificates_per_grade 2

python - "${AUDIT_OUTPUT}" "${CHECKPOINT}" <<'PY'
import hashlib
import json
import pathlib
import sys

audit_path = pathlib.Path(sys.argv[1])
checkpoint_path = pathlib.Path(sys.argv[2])
with audit_path.open(encoding="utf-8") as stream:
    audit = json.load(stream)
digest = hashlib.sha256()
with checkpoint_path.open("rb") as stream:
    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
        digest.update(chunk)
checkpoint_hash = digest.hexdigest()
assert audit["schema"] == "origin-full-validation-audit-v8"
assert audit["scope"] == "inner_validation_only"
assert audit["checkpoint_role"] == "best_learned"
assert audit["checkpoint_sha256"] == checkpoint_hash
assert audit["metric_reproduction"] == {
    "checkpoint": True,
    "completed_result": True,
}
print("registered_v8_audit_verified", audit_path)
PY
