#!/bin/bash
# Shared environment for all ESPACOL gamma SLURM jobs.
# Sourced by the .sbatch scripts. Activates the conda env and points the model
# caches at the pre-populated workspace dir, in offline mode (GPU nodes have no
# internet — weights were pre-cached on the login node by the pre-flight step).
source /etc/profile.d/lmod.sh
module purge 2>/dev/null || true
# KU cluster migrated (V100 → H200, 2026-06-24): the old `miniconda/3` module and
# /apps/ku/miniconda are gone, so `conda activate` is unavailable. The conda env
# survives in /dpc — activate it directly. Its python is self-contained and
# torch 2.12.0+cu126 ships sm_90 kernels, so it runs on the H200 gpu partition.
export CONDA_PREFIX=/dpc/kuin0170/u100066980/envs/espacol
export PATH="$CONDA_PREFIX/bin:$PATH"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"

export TORCH_HOME=/dpc/kuin0170/u100066980/.cache/torch
export HF_HOME=/dpc/kuin0170/u100066980/.cache/hf
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

export REPO=/home/kunet.ae/100066980/ESPACOL-New
export RUNS=/dpc/kuin0170/u100066980/runs
export DR_ROOT=/dpc/kuin0170/u100066980/DR
export BUSI_ROOT=/dpc/kuin0170/u100066980/BUSI
