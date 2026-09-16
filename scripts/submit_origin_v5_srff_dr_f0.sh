#!/bin/bash
# ORIGIN-v5 SRFF EyePACS fold-0 development run. Relative to the bounded v3
# control, only the encoder dependency policy and exposed evidence scales
# change: full-support s16/s32 cells are replaced by the RF-bounded s128
# sealed-regional ledger. The delegated launcher owns the split and training
# protocol and keeps the outer test locked and unevaluated.
#SBATCH --job-name=origin_v5_sdr0
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --output=/dpc/kuin0170/ESPACOL-New/origin_v5_srff_dr_f0_%j.out
#SBATCH --error=/dpc/kuin0170/ESPACOL-New/origin_v5_srff_dr_f0_%j.err
#SBATCH --account=kuin0170

set -eo pipefail

REPO_ROOT="${ORIGIN_REPO_ROOT:-/dpc/kuin0170/ESPACOL-New}"
cd "${REPO_ROOT}"

# Prospective v5 architecture identity.
export ORIGIN_RUN_DIR="${ORIGIN_SRFF_DR_RUN_DIR:-runs/origin_dr_f0_v5_srff_w16}"
export ORIGIN_ENCODER="convnext_tiny_srff"
export ORIGIN_SCALES="s4,s8,s128"

# Frozen v3 scientific controls. Infrastructure paths and resume authorization
# remain explicit environment inputs; none of these values may be inherited.
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
export ORIGIN_EPOCHS="75"
export ORIGIN_NUM_WORKERS="8"
export ORIGIN_ENCODER_LR="1e-4"
export ORIGIN_HEAD_LR="5e-4"
export ORIGIN_WEIGHT_DECAY="1e-5"
export ORIGIN_RPS_WEIGHT="0.25"
export ORIGIN_BUDGET_WEIGHT="0.0"
export ORIGIN_AMP_UNFREEZE_SCALE="256"

echo "=== ORIGIN-v5 SRFF EyePACS fold 0 / inner validation only ==="
echo "controlled_change=encoder:convnext_tiny_srff,evidence_scales:s4,s8,s128,window_cells:16"
exec bash scripts/submit_origin_v3_dr_f0.sh
