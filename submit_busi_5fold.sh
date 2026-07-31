#!/bin/bash
#SBATCH --job-name=espacol_busi_multitile_5fold
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=220G
#SBATCH --time=2-00:00:00
#SBATCH --array=0-4
#SBATCH --output=/dpc/kuin0170/ESPACOL-New/slurm_logs/busi_multitile_fold%a_%j.out
#SBATCH --error=/dpc/kuin0170/ESPACOL-New/slurm_logs/busi_multitile_fold%a_%j.err
#SBATCH --account=kuin0170

mkdir -p /dpc/kuin0170/ESPACOL-New/slurm_logs

source /etc/profile.d/lmod.sh
module load miniconda/3
module load cuda/12.6
source activate G

cd /dpc/kuin0170/ESPACOL-New

export HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# 3x3 tile grid: same architecture as DR multi-tile run.
# crop_fundus=False (BUSI is ultrasound, no fundus circle).
# Disk cache written to Datasets/BUSI/cache_tiles_900 on first run.
python train_busi.py \
    --busi_root Datasets/BUSI \
    --run_dir runs/busi_multitile_5fold \
    --folds ${SLURM_ARRAY_TASK_ID} \
    --use_multi_tile \
    --tile_grid 3 \
    --batch_size 24 \
    --grad_checkpoint
