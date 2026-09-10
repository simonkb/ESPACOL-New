#!/bin/bash
# Full inner-validation audit for the RF-sealed s4+s8 EyePACS checkpoint.
# Submit only after submit_origin_v4_local_dr_f0.sh completes successfully.
#SBATCH --job-name=origin_v4_laud
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=0-04:00:00
#SBATCH --output=/dpc/kuin0170/ESPACOL-New/origin_v4_local_audit_%j.out
#SBATCH --error=/dpc/kuin0170/ESPACOL-New/origin_v4_local_audit_%j.err
#SBATCH --account=kuin0170

set -eo pipefail

REPO_ROOT="${ORIGIN_REPO_ROOT:-/dpc/kuin0170/ESPACOL-New}"
cd "${REPO_ROOT}"
RUN_DIR="${ORIGIN_LOCAL_DR_RUN_DIR:-runs/origin_dr_f0_v4_local_s4s8}"
export ORIGIN_CHECKPOINT="${RUN_DIR}/fold0/best.pth"
export ORIGIN_AUDIT_OUTPUT="${RUN_DIR}/fold0/audits/full_validation_audit_v2.json"
export ORIGIN_EXPECTED_SCALES="s4,s8"
export ORIGIN_AUDIT_NUM_WORKERS="8"
export ORIGIN_AUDIT_TOP_KS="1,5,10"
export ORIGIN_AUDIT_CERTIFICATES_PER_GRADE="2"

echo "=== ORIGIN-v4 RF-sealed EyePACS structural audit ==="
exec bash scripts/submit_origin_v3_dr_validation_audit.sh
