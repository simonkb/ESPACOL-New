#!/usr/bin/env bash
#SBATCH --job-name=org_artifact
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=origin_accept_manifest_%j.out
#SBATCH --error=origin_accept_manifest_%j.err

set -euo pipefail

REPO_ROOT="${ORIGIN_REPO_ROOT:-/dpc/kuin0170/ESPACOL-New}"
CV_ROOT="${ORIGIN_V3_CV_ROOT:-$REPO_ROOT/runs/origin_v3_full_cv_20260922T084547Z}"
OUTPUT_ROOT="${ORIGIN_ACCEPTANCE_ROOT:-$REPO_ROOT/runs/origin_acceptance}"
cd "$REPO_ROOT"
source /home/kunet.ae/100067950/.conda/etc/profile.d/conda.sh
conda activate G

printf '=== ORIGIN acceptance artifact manifest ===\n'
date --iso-8601=seconds
git rev-parse HEAD
python -m pytest -q tests/test_origin_acceptance_manifest.py tests/test_origin_v3_cv_protocol.py
python tools/build_origin_acceptance_manifest.py \
  --cv-root "$CV_ROOT" \
  --dr-root "$REPO_ROOT/Datasets/DR" \
  --aptos-root "$REPO_ROOT/Datasets/aptos2019-blindness-detection" \
  --output-dir "$OUTPUT_ROOT/artifact_manifest"
