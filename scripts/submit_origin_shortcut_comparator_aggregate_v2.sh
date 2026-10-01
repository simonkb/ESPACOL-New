#!/bin/bash
# Aggregate sealed ORIGIN-v1 plus comparator-follow-up-v2 artifacts.
#SBATCH --job-name=origin_sc_cmp_sum
#SBATCH --partition=prod
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=/dpc/kuin0170/ESPACOL-New/origin_shortcut_logs/comparator_aggregate_%j.out
#SBATCH --error=/dpc/kuin0170/ESPACOL-New/origin_shortcut_logs/comparator_aggregate_%j.err
#SBATCH --account=kuin0170

set -euo pipefail
set +u
source /etc/profile.d/lmod.sh || exit 1
module load miniconda/3 || exit 1
source activate "${ORIGIN_CONDA_ENV:-G}" || exit 1
set -u

REPO_ROOT="${ORIGIN_REPO_ROOT:?ORIGIN_REPO_ROOT is required}"
SUITE_ROOT="${ORIGIN_SHORTCUT_COMPARATOR_ROOT:?ORIGIN_SHORTCUT_COMPARATOR_ROOT is required}"
REFERENCE_ROOT="${ORIGIN_SHORTCUT_REFERENCE_ROOT:?ORIGIN_SHORTCUT_REFERENCE_ROOT is required}"
LAUNCH_COMMIT="${ORIGIN_LAUNCH_COMMIT:?ORIGIN_LAUNCH_COMMIT is required}"
cd "${REPO_ROOT}"
[[ "$(git rev-parse HEAD)" == "${LAUNCH_COMMIT}" ]] || exit 2
[[ -z "$(git status --porcelain --untracked-files=no)" ]] || exit 3

python -m pytest -q tests/test_origin_shortcut_benchmark.py
OUTPUT="${SUITE_ROOT}/COMPARATOR_SHORTCUT_RESULTS_V2.json"
[[ ! -e "${OUTPUT}" ]] || { echo "Refusing to overwrite ${OUTPUT}." >&2; exit 4; }
python scripts/aggregate_origin_shortcut_pilot.py \
  --audit-root "${SUITE_ROOT}/workers" \
  --reference-root "${REFERENCE_ROOT}/workers" \
  --output "${OUTPUT}" --seeds 1701,2603,3907
echo "Expanded shortcut comparator aggregate: ${OUTPUT}"
