#!/bin/bash
# Shared environment for all ESPACOL gamma SLURM jobs.
# Sourced by the .sbatch scripts. Activates the conda env and points the model
# caches at the pre-populated workspace dir, in offline mode (GPU nodes have no
# internet — weights were pre-cached on the login node by the pre-flight step).
source /etc/profile.d/lmod.sh
module purge 2>/dev/null || true
module load miniconda/3
source /apps/ku/miniconda/3/etc/profile.d/conda.sh
conda activate /dpc/kuin0170/u100066980/envs/espacol

export TORCH_HOME=/dpc/kuin0170/u100066980/.cache/torch
export HF_HOME=/dpc/kuin0170/u100066980/.cache/hf
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

export REPO=/home/kunet.ae/100066980/ESPACOL-New
export RUNS=/dpc/kuin0170/u100066980/runs
export DR_ROOT=/dpc/kuin0170/u100066980/DR
export BUSI_ROOT=/dpc/kuin0170/u100066980/BUSI
