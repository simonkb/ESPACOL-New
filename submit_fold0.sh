#!/bin/bash
#SBATCH --job-name=espacol_ct_fold0
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=220G
#SBATCH --time=2-00:00:00
#SBATCH --output=/dpc/kuin0170/ESPACOL-New/slurm_logs/ct_fold0_%j.out
#SBATCH --error=/dpc/kuin0170/ESPACOL-New/slurm_logs/ct_fold0_%j.err
#SBATCH --account=kuin0170

mkdir -p /dpc/kuin0170/ESPACOL-New/slurm_logs

source /etc/profile.d/lmod.sh
module load miniconda/3
module load cuda/12.6
source activate G

cd /dpc/kuin0170/ESPACOL-New

export HF_HUB_OFFLINE=1

python train_dr.py \
    --dr_root Datasets/DR \
    --run_dir runs/ct4x4_fold0 \
    --folds 0 \
    --use_multi_tile \
    --tile_grid 4 \
    --use_tile_transformer \
    --use_ordinal_head \
    --batch_size 24 \
    --grad_checkpoint \
    --epochs 75