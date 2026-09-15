#!/bin/bash
# Compare the three completed and independently audited APTOS fold-0
# best-learned checkpoints only. The utility exits non-zero unless the target
# passes its V3 metric floor, passes its individual-witness strength gate, and
# strictly beats both controls in accuracy.
#SBATCH --job-name=origin_v8_cmp
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --output=/dpc/kuin0170/ESPACOL-New/origin_v8_compare_%j.out
#SBATCH --error=/dpc/kuin0170/ESPACOL-New/origin_v8_compare_%j.err
#SBATCH --account=kuin0170

set -eo pipefail
set +u
source /etc/profile.d/lmod.sh || exit 1
module load miniconda/3 || exit 1
source activate "${ORIGIN_CONDA_ENV:-G}" || exit 1
set -u

REPO_ROOT="${ORIGIN_REPO_ROOT:-/dpc/kuin0170/ESPACOL-New}"
cd "${REPO_ROOT}"
: "${ORIGIN_EXPECTED_COMMIT:?Set ORIGIN_EXPECTED_COMMIT to the training commit}"
[[ "$(git rev-parse HEAD)" == "${ORIGIN_EXPECTED_COMMIT}" ]] || {
  echo "Comparison commit differs from the registered training commit." >&2
  exit 2
}

RUN_ROOT="${ORIGIN_V8_RUN_ROOT:-runs}"
TARGET="${ORIGIN_V8_TARGET_RUN:-${RUN_ROOT}/origin_aptos_f0_v8_cmwa_r1_target}"
ADDITIVE="${ORIGIN_V8_ADDITIVE_RUN:-${RUN_ROOT}/origin_aptos_f0_v8_cmwa_r1_endpoint_control}"
SHUFFLED="${ORIGIN_V8_SHUFFLED_RUN:-${RUN_ROOT}/origin_aptos_f0_v8_cmwa_r1_shuffled_control}"
OUTPUT="${ORIGIN_V8_COMPARISON_OUTPUT:-${RUN_ROOT}/origin_aptos_f0_v8_cmwa_r1_matched_comparison.json}"

[[ ! -e "${OUTPUT}" ]] || {
  echo "Refusing to overwrite comparison: ${OUTPUT}" >&2
  exit 3
}
python -m pytest -q tests/test_compare_origin_v8_controls.py
python compare_origin_v8_controls.py \
  --target "${TARGET}" \
  --additive_endpoint_control "${ADDITIVE}" \
  --shuffled_geometry_control "${SHUFFLED}" \
  --output "${OUTPUT}"
