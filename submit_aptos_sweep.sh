#!/bin/bash
# SLURM sweep agent for OPTIC-C APTOS hyperparameter search.
#
# Usage:
#   1. Create the sweep ONCE on the cluster (or locally):
#        wandb sweep sweep_config_aptos_optic.yaml --project espacol-new
#        → prints something like: entity/espacol-new/abc12345
#
#   2. Submit N parallel agents (each agent runs --count trials):
#        sbatch --array=0-4 submit_aptos_sweep.sh entity/espacol-new/abc12345
#
#   Each agent requests new trials from the sweep controller independently.
#   Adjust --array range to control how many agents run in parallel.
#   Adjust --count below to control trials per agent (total = agents × count).
#
#SBATCH --job-name=aptos_sweep
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=0-12:00:00
#SBATCH --output=/dpc/kuin0170/ESPACOL-New/slurm_logs/aptos_sweep_%a_%j.out
#SBATCH --error=/dpc/kuin0170/ESPACOL-New/slurm_logs/aptos_sweep_%a_%j.err
#SBATCH --account=kuin0170

SWEEP_ID="${1}"   # pass as first positional arg: sbatch submit_aptos_sweep.sh <sweep_id>

if [ -z "${SWEEP_ID}" ]; then
    echo "ERROR: No sweep_id provided."
    echo "Usage: sbatch --array=0-4 submit_aptos_sweep.sh entity/project/sweep_id"
    exit 1
fi

mkdir -p /dpc/kuin0170/ESPACOL-New/slurm_logs

source /etc/profile.d/lmod.sh
module load miniconda/3
module load cuda/12.6
source activate G

cd /dpc/kuin0170/ESPACOL-New

export HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WANDB_PROJECT=espacol-new

echo "Agent ${SLURM_ARRAY_TASK_ID} starting — sweep: ${SWEEP_ID}"

python train_aptos_sweep.py \
    --aptos_root  Datasets/aptos2019-blindness-detection \
    --run_dir     runs/aptos_optic_sweep \
    --sweep_id    "${SWEEP_ID}" \
    --count       5

echo "Agent ${SLURM_ARRAY_TASK_ID} done."
