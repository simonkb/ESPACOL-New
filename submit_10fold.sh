#!/bin/bash
#SBATCH --job-name=espacol_multitile_3x3_10fold
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=220G
#SBATCH --time=2-00:00:00
#SBATCH --array=0-9
#SBATCH --output=/dpc/kuin0170/ESPACOL-New/slurm_logs/multitile_3x3_fold%a_%j.out
#SBATCH --error=/dpc/kuin0170/ESPACOL-New/slurm_logs/multitile_3x3_fold%a_%j.err
#SBATCH --account=kuin0170

mkdir -p /dpc/kuin0170/ESPACOL-New/slurm_logs

source /etc/profile.d/lmod.sh
module load miniconda/3
module load cuda/12.6
source activate G

cd /dpc/kuin0170/ESPACOL-New

export HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# 3x3 tile grid: 900x900 -> 9 local 300x300 tiles + 1 global = T=10
# This is the config that achieved 84.25% on fold 0.
python train_dr.py \
    --dr_root Datasets/DR \
    --run_dir runs/dr_multitile_3x3_10fold \
    --folds ${SLURM_ARRAY_TASK_ID} \
    --use_multi_tile \
    --tile_grid 3 \
    --batch_size 24 \
    --grad_checkpoint \
    --epochs 75
