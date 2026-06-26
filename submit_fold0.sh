#!/bin/bash
#SBATCH --job-name=espacol_dr_fold0
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=220G
#SBATCH --time=2-00:00:00
#SBATCH --output=slurm_logs/dr_fold0_%j.out
#SBATCH --error=slurm_logs/dr_fold0_%j.err
#SBATCH --account=YOUR_PROJECT_ACCOUNT

mkdir -p slurm_logs

source "$HOME/anaconda3/bin/activate" G

cd "$HOME/ESPACOL-New"

export HF_HUB_OFFLINE=1

python train_dr.py \
    --dr_root Datasets/DR \
    --run_dir runs/dr_hpc_fold0 \
    --folds 0 \
    --use_multi_tile \
    --use_concept_spine \
    --batch_size 24 \
    --grad_checkpoint \
    --epochs 100
