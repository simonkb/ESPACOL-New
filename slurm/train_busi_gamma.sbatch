#!/bin/bash
#SBATCH --job-name=busi_gamma
#SBATCH --account=kuin0170
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=/dpc/kuin0170/u100066980/runs/busi_gamma/slurm-%j.out

# BUSI 5-fold CV with the gamma image-text loss (EfficientNet-V2S backbone).
# All 5 folds run in this single job (BUSI is small: 780 images).
set -e
source /home/kunet.ae/100066980/ESPACOL-New/slurm/_env.sh
cd "$REPO"

nvidia-smi || true
# Fail fast if CUDA isn't usable (else training silently falls back to CPU).
python -c "import torch,sys; ok=torch.cuda.is_available(); print('cuda available:', ok, '| cuda build:', torch.version.cuda); sys.exit(0 if ok else 1)" \
    || { echo "ERROR: CUDA not available on this node — aborting."; exit 1; }
echo "Starting BUSI gamma training: $(date)"

python train_busi.py \
    --busi_root "$BUSI_ROOT" \
    --run_dir   "$RUNS/busi_gamma" \
    --folds all

echo "Done: $(date)"
