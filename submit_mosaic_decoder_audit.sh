#!/bin/bash
# Validation-only audit of fixed MOSAIC proof-to-grade decoders.
# Usage: sbatch submit_mosaic_decoder_audit.sh aptos|dr <checkpoint>
# The utility reconstructs the checkpoint's inner validation split and never
# creates a loader for the locked outer test split.
#SBATCH --job-name=mosaic_decoder_audit
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/dpc/kuin0170/ESPACOL-New/mosaic_decoder_audit_%j.out
#SBATCH --error=/dpc/kuin0170/ESPACOL-New/mosaic_decoder_audit_%j.err
#SBATCH --account=kuin0170

source /etc/profile.d/lmod.sh || exit 1
module load miniconda/3 || exit 1
module load cuda/12.6 || exit 1
source activate G || exit 1

set -euo pipefail
cd /dpc/kuin0170/ESPACOL-New

export HF_HUB_OFFLINE=1

DATASET="${1:-}"
CHECKPOINT="${2:-}"
if [[ -z "${DATASET}" || -z "${CHECKPOINT}" ]]; then
  echo "Usage: sbatch $0 aptos|dr <checkpoint>" >&2
  exit 2
fi
if [[ ! -f "${CHECKPOINT}" ]]; then
  echo "Checkpoint not found: ${CHECKPOINT}" >&2
  exit 3
fi

case "${DATASET}" in
  aptos)
    DATA_ROOT="Datasets/aptos2019-blindness-detection"
    ;;
  dr)
    DATA_ROOT="Datasets/DR"
    ;;
  *)
    echo "Dataset must be 'aptos' or 'dr', got: ${DATASET}" >&2
    exit 4
    ;;
esac

echo "=== MOSAIC proof-decoder audit (inner validation only) ==="
date --iso-8601=seconds
git rev-parse HEAD
git status --short
python --version
echo "dataset=${DATASET}"
echo "checkpoint=${CHECKPOINT}"

python tools/audit_mosaic_decoders.py \
  --checkpoint "${CHECKPOINT}" \
  --data_root "${DATA_ROOT}" \
  --num_workers 8
