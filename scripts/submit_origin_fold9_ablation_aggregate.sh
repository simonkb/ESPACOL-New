#!/bin/bash
# Aggregate after *any* release outcome so a failed release leaves a clear log.
#SBATCH --job-name=of9_ab_sum
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=/dpc/kuin0170/ESPACOL-New/origin_fold9_ablation_logs/aggregate_%j.out
#SBATCH --error=/dpc/kuin0170/ESPACOL-New/origin_fold9_ablation_logs/aggregate_%j.err
#SBATCH --account=kuin0170

set -euo pipefail
set +u
source /etc/profile.d/lmod.sh || exit 1
module load miniconda/3 || exit 1
module load cuda/12.6 || exit 1
source activate "${ORIGIN_CONDA_ENV:-G}" || exit 1
set -u

REPO_ROOT="${ORIGIN_REPO_ROOT:?ORIGIN_REPO_ROOT is required}"
LAUNCH_COMMIT="${ORIGIN_LAUNCH_COMMIT:?ORIGIN_LAUNCH_COMMIT is required}"
EXPERIMENT_ROOT="${ORIGIN_FOLD9_ABLATION_ROOT:?ORIGIN_FOLD9_ABLATION_ROOT is required}"
DATA_ROOT="${ORIGIN_DATA_ROOT:?ORIGIN_DATA_ROOT is required}"
PROTOCOL="${ORIGIN_FOLD9_ABLATION_PROTOCOL:?ORIGIN_FOLD9_ABLATION_PROTOCOL is required}"

cd "${REPO_ROOT}"
[[ "$(git rev-parse HEAD)" == "${LAUNCH_COMMIT}" ]] || {
  echo "Immutable launch commit mismatch." >&2; exit 2;
}
[[ -z "$(git status --porcelain --untracked-files=no)" ]] || {
  echo "Tracked files changed in the immutable worktree." >&2; exit 3;
}

ORIGIN_AGG_PROTOCOL="${PROTOCOL}" \
ORIGIN_AGG_COMMIT="${LAUNCH_COMMIT}" \
ORIGIN_AGG_ROOT="${EXPERIMENT_ROOT}" \
ORIGIN_AGG_DATA="${DATA_ROOT}" \
python - <<'PY'
import os
from pathlib import Path
from scripts.origin_fold9_ablation_common import read_json, verify_protocol
p = read_json(os.environ["ORIGIN_AGG_PROTOCOL"])
verify_protocol(
    p,
    expected_launch_commit=os.environ["ORIGIN_AGG_COMMIT"],
    expected_root=os.environ["ORIGIN_AGG_ROOT"],
    expected_data_root=os.environ["ORIGIN_AGG_DATA"],
)
marker = Path(os.environ["ORIGIN_AGG_ROOT"]) / "release" / "OUTER_RELEASE_COMPLETE.json"
if not marker.is_file():
    raise RuntimeError(
        "outer release did not complete; aggregation is intentionally blocked. "
        f"Missing {marker}"
    )
PY

for artifact in FOLD9_ABLATION_RESULTS.json FOLD9_ABLATION_RESULTS.csv FOLD9_ABLATION_RESULTS.md; do
  [[ ! -e "${EXPERIMENT_ROOT}/${artifact}" ]] || {
    echo "Refusing to overwrite existing aggregate: ${EXPERIMENT_ROOT}/${artifact}." >&2
    exit 4
  }
done

echo "=== ORIGIN fold-9 paired ablation aggregation ==="
date --iso-8601=seconds
python scripts/aggregate_origin_fold9_ablations.py \
  --root "${EXPERIMENT_ROOT}" --protocol "${PROTOCOL}" \
  --bootstrap-samples 10000 \
  --bootstrap-seed 26092026
echo "Results: ${EXPERIMENT_ROOT}/FOLD9_ABLATION_RESULTS.json"
