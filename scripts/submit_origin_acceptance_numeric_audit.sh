#!/usr/bin/env bash
#SBATCH --job-name=org_num_audit
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:45:00
#SBATCH --output=origin_accept_numeric_%j.out
#SBATCH --error=origin_accept_numeric_%j.err

set -euo pipefail

REPO_ROOT="${ORIGIN_REPO_ROOT:-/dpc/kuin0170/ESPACOL-New}"
cd "$REPO_ROOT"
source /home/kunet.ae/100067950/.conda/etc/profile.d/conda.sh
conda activate G

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
