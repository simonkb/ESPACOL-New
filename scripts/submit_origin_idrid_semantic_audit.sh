#!/usr/bin/env bash
#SBATCH --job-name=org_idrid
#SBATCH --partition=gpu
#SBATCH --account=kuin0170
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --output=origin_idrid_semantic_%j.out
#SBATCH --error=origin_idrid_semantic_%j.err

set -euo pipefail

REPO_ROOT="${ORIGIN_REPO_ROOT:-/dpc/kuin0170/ESPACOL-New}"
CV_ROOT="${ORIGIN_V3_CV_ROOT:-$REPO_ROOT/runs/origin_v3_full_cv_20260922T084547Z}"
OUTPUT_ROOT="${ORIGIN_ACCEPTANCE_ROOT:-$REPO_ROOT/runs/origin_acceptance}"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
set +u
source /etc/profile.d/lmod.sh || exit 1
module load miniconda/3 || exit 1
module load cuda/12.6 || exit 1
source activate "${ORIGIN_CONDA_ENV:-G}" || exit 1
set -u

printf '=== ORIGIN external IDRiD semantic audit ===\n'
date --iso-8601=seconds
git rev-parse HEAD
python -m pytest -q tests/test_origin_idrid_semantics.py

CHECKPOINTS=()
for fold in $(seq 0 9); do
  checkpoint="$CV_ROOT/dr/workers/fold${fold}/fold${fold}/best.pth"
  test -f "$checkpoint" || { echo "missing checkpoint: $checkpoint" >&2; exit 1; }
  CHECKPOINTS+=("$checkpoint")
done

python tools/audit_origin_idrid_semantics.py \
  --idrid-root "$REPO_ROOT/Datasets/IDRiD" \
  --checkpoints "${CHECKPOINTS[@]}" \
  --output-dir "$OUTPUT_ROOT/idrid_semantics" \
  --batch-size 2 \
  --random-repeats 20 \
  --seed 20261001
