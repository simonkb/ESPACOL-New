#!/bin/bash
# RF-sealed ORIGIN controlled experiment: APTOS fold 0. This wrapper changes
# exactly one scientific factor relative to the bounded v3 run: only the s4
# and s8 evidence ledgers are active. All training and split logic is delegated
# to the audited v3 launcher.
#SBATCH --job-name=origin_v4_la0
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH --output=/dpc/kuin0170/ESPACOL-New/origin_v4_local_aptos_f0_%j.out
#SBATCH --error=/dpc/kuin0170/ESPACOL-New/origin_v4_local_aptos_f0_%j.err
#SBATCH --account=kuin0170

set -eo pipefail

REPO_ROOT="${ORIGIN_REPO_ROOT:-/dpc/kuin0170/ESPACOL-New}"
cd "${REPO_ROOT}"
export ORIGIN_RUN_DIR="${ORIGIN_LOCAL_APTOS_RUN_DIR:-runs/origin_aptos_f0_v4_local_s4s8}"
export ORIGIN_SCALES="s4,s8"
export ORIGIN_IMAGE_SIZE="640"
export ORIGIN_PROJECTION_DIM="128"
export ORIGIN_REFERENCE_COUNT="4096"
export ORIGIN_ATOM_RATE_INIT="1e-6"
export ORIGIN_PRIOR_RATE_INIT="1e-4"
export ORIGIN_BOUNDARY_SCALE_INIT="1.0"
export ORIGIN_TOTAL_RATE_CAP="64.0"
export ORIGIN_PRIOR_RATE_CAP="1.0"
export ORIGIN_BOUNDARY_SCALE_CAP="2.0"
export ORIGIN_RATE_ROUNDOFF_MARGIN="1.0"
export ORIGIN_ATOM_MODE="cumulative"
export ORIGIN_BATCH_SIZE="8"
export ORIGIN_EPOCHS="35"
export ORIGIN_NUM_WORKERS="8"
export ORIGIN_ENCODER_LR="1e-4"
export ORIGIN_HEAD_LR="5e-4"
export ORIGIN_WEIGHT_DECAY="1e-5"
export ORIGIN_RPS_WEIGHT="0.25"
export ORIGIN_BUDGET_WEIGHT="0.0"
export ORIGIN_AMP_UNFREEZE_SCALE="256"

echo "=== ORIGIN-v4 RF-sealed local-evidence APTOS fold 0 ==="
echo "controlled_change=evidence_scales:s4,s8"
exec bash scripts/submit_origin_v3_aptos_f0.sh
