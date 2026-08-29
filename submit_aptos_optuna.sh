#!/bin/bash
# SLURM array job — Optuna hyperparameter search for OPTIC-C on APTOS 2019.
# No wandb/internet needed. Workers coordinate via a shared SQLite study file.
#
# Usage:
#   sbatch --array=0-4 submit_aptos_optuna.sh
#
# Each worker runs --n_trials 5, giving 25 total trials across 5 workers.
# Adjust --array and --n_trials to scale up or down.
#
# After the sweep, inspect results from the login node:
#   python train_aptos_optuna.py --report
#   (or load it in Python: optuna.load_study(study_name="aptos_optic_v9", storage="sqlite:///..."))
#
#SBATCH --job-name=aptos_optuna
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=0-12:00:00
#SBATCH --output=/dpc/kuin0170/ESPACOL-New/slurm_logs/aptos_optuna_%a_%j.out
#SBATCH --error=/dpc/kuin0170/ESPACOL-New/slurm_logs/aptos_optuna_%a_%j.err
#SBATCH --account=kuin0170

mkdir -p /dpc/kuin0170/ESPACOL-New/slurm_logs

source /etc/profile.d/lmod.sh
module load miniconda/3
module load cuda/12.6
source activate G

cd /dpc/kuin0170/ESPACOL-New

export HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Install optuna into user site-packages (no write access to the shared conda env)
pip install -q --user optuna

echo "Worker ${SLURM_ARRAY_TASK_ID} starting on $(hostname)"

python train_aptos_optuna.py \
    --aptos_root  Datasets/aptos2019-blindness-detection \
    --study_dir   runs/aptos_optic_optuna \
    --n_trials    5

echo "Worker ${SLURM_ARRAY_TASK_ID} done."
