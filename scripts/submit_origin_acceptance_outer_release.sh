#!/bin/bash
# Coordinated post-freeze outer release. Every worker requires the common
# TRAINING_FROZEN gate; no worker can feed results back into model selection.
#SBATCH --job-name=oa_release
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH --array=0-179%12
#SBATCH --output=/dpc/kuin0170/ESPACOL-New/origin_acceptance_logs/release_%A_%a.out
#SBATCH --error=/dpc/kuin0170/ESPACOL-New/origin_acceptance_logs/release_%A_%a.err
#SBATCH --account=kuin0170

set -euo pipefail
set +u
source /etc/profile.d/lmod.sh
module load miniconda/3
module load cuda/12.6
source activate "${ORIGIN_CONDA_ENV:-espacol}"
set -u
cd "${ORIGIN_REPO_ROOT:?ORIGIN_REPO_ROOT is required}"
[[ "$(git rev-parse HEAD)" == "${ORIGIN_LAUNCH_COMMIT:?ORIGIN_LAUNCH_COMMIT is required}" ]] || exit 2
python release_origin_acceptance_baselines.py \
  --experiment_root "${ORIGIN_ACCEPTANCE_ROOT:?ORIGIN_ACCEPTANCE_ROOT is required}" \
  --task_index "${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}" \
  --aptos_root "${ORIGIN_APTOS_ROOT:?ORIGIN_APTOS_ROOT is required}" \
  --dr_root "${ORIGIN_DR_ROOT:?ORIGIN_DR_ROOT is required}" \
  --num_workers "${ORIGIN_NUM_WORKERS:-8}"
