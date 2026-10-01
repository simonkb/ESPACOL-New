#!/usr/bin/env bash
#SBATCH --job-name=org_num_audit
#SBATCH --partition=gpu
#SBATCH --account=kuin0170
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:45:00
#SBATCH --output=origin_accept_numeric_%j.out
#SBATCH --error=origin_accept_numeric_%j.err

set -euo pipefail

REPO_ROOT="${ORIGIN_REPO_ROOT:-/dpc/kuin0170/ESPACOL-New}"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
set +u
source /etc/profile.d/lmod.sh || exit 1
module load miniconda/3 || exit 1
module load cuda/12.6 || exit 1
source activate "${ORIGIN_CONDA_ENV:-G}" || exit 1
set -u

printf '=== ORIGIN independent decoder numerical audit ===\n'
date --iso-8601=seconds
git rev-parse HEAD
python -m pytest -q tests/test_origin_decoder_numeric_audit.py tests/test_origin_core.py
python tools/audit_origin_decoder_numeric.py \
  --samples 128 \
  --gradient-samples 12 \
  --seed 20261001 \
  --rate-cap 64 \
  --dps 100 \
  --output runs/origin_acceptance/numerical_decoder_audit.json
