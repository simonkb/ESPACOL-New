#!/bin/bash
#SBATCH --job-name=dr_gamma
#SBATCH --account=kuin0170
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=3-00:00:00
#SBATCH --array=0-9
#SBATCH --output=/dpc/kuin0170/u100066980/runs/dr_gamma/slurm-%A_%a.out

# DR 10-fold CV with the gamma image-text loss (EfficientNet-V2S backbone).
# Job ARRAY: one fold per task -> runs in parallel across the idle V100 nodes,
# each well within the 3-day wall. Tasks share run_dir; fold dirs are unique so
# checkpoints never collide, and each fold writes a race-safe test_result.json.
#
# RAM image cache (no disk cache) so parallel tasks never race on shared .pt
# files: each task decodes the 35K images into its own node RAM (~5 min, 16 thr).
set -e
source /home/kunet.ae/100066980/ESPACOL-New/slurm/_env.sh
cd "$REPO"

nvidia-smi || true
# Fail fast if CUDA isn't usable (else training silently falls back to CPU,
# which would never finish 35K images in the wall-time).
python -c "import torch,sys; ok=torch.cuda.is_available(); print('cuda available:', ok, '| cuda build:', torch.version.cuda); sys.exit(0 if ok else 1)" \
    || { echo "ERROR: CUDA not available on this node — aborting."; exit 1; }
echo "Starting DR gamma training fold ${SLURM_ARRAY_TASK_ID}: $(date)"

python train_dr.py \
    --dr_root  "$DR_ROOT" \
    --run_dir  "$RUNS/dr_gamma" \
    --cache_dir "" \
    --folds    "${SLURM_ARRAY_TASK_ID}"

echo "Done fold ${SLURM_ARRAY_TASK_ID}: $(date)"
