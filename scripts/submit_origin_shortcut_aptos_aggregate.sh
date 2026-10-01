#!/bin/bash
# Aggregate the APTOS three-seed controlled-shortcut pilot after all workers pass.
#SBATCH --job-name=origin_shortcut_sum
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=/dpc/kuin0170/ESPACOL-New/origin_shortcut_logs/aggregate_%j.out
#SBATCH --error=/dpc/kuin0170/ESPACOL-New/origin_shortcut_logs/aggregate_%j.err
#SBATCH --account=kuin0170

set -euo pipefail
set +u
source /etc/profile.d/lmod.sh || exit 1
module load miniconda/3 || exit 1
source activate "${ORIGIN_CONDA_ENV:-G}" || exit 1
set -u

REPO_ROOT="${ORIGIN_REPO_ROOT:?ORIGIN_REPO_ROOT is required}"
PILOT_ROOT="${ORIGIN_SHORTCUT_ROOT:?ORIGIN_SHORTCUT_ROOT is required}"
LAUNCH_COMMIT="${ORIGIN_LAUNCH_COMMIT:?ORIGIN_LAUNCH_COMMIT is required}"
cd "${REPO_ROOT}"
[[ "$(git rev-parse HEAD)" == "${LAUNCH_COMMIT}" ]] || {
  echo "Immutable launch commit mismatch." >&2; exit 2;
}
[[ -z "$(git status --porcelain --untracked-files=no)" ]] || {
  echo "Tracked files changed in immutable shortcut worktree." >&2; exit 3;
}
OUTPUT="${PILOT_ROOT}/APTOS_SHORTCUT_PILOT_RESULTS.json"
[[ ! -e "${OUTPUT}" ]] || { echo "Refusing to overwrite ${OUTPUT}." >&2; exit 4; }
python scripts/aggregate_origin_shortcut_pilot.py \
  --audit-root "${PILOT_ROOT}/workers" --output "${OUTPUT}" \
  --seeds 1701,2603,3907
echo "Shortcut pilot aggregate: ${OUTPUT}"
