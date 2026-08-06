#!/bin/bash
#SBATCH --job-name=optic_dr_fold0
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=220G
#SBATCH --time=2-00:00:00
#SBATCH --output=/dpc/kuin0170/ESPACOL-New/slurm_logs/optic_fold0_%j.out
#SBATCH --error=/dpc/kuin0170/ESPACOL-New/slurm_logs/optic_fold0_%j.err
#SBATCH --account=kuin0170

mkdir -p /dpc/kuin0170/ESPACOL-New/slurm_logs

source /etc/profile.d/lmod.sh
module load miniconda/3
module load cuda/12.6
source activate G

cd /dpc/kuin0170/ESPACOL-New

export HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# OPTIC v4: CTOT + CORAL only — stripped down to core novel components.
# Removed GPA/OSD/TCL auxiliary losses: they created competing gradient noise
# (v3: 79.05% with all 3 losses; v2: 80.59% before GPA was added).
# Baseline (no CTOT, no CORAL) = 84.25%. Target: >= 84.25%
#
# Hypothesis: the oscillation in v3 (13%→79%→13%...) was caused by GPA+OSD+TCL
# adding variance. Clean signal = CORAL + PCOL + SCOLw + IT only.
python train_dr.py \
    --dr_root Datasets/DR \
    --run_dir runs/optic_dr_fold0_v4 \
    --folds 0 \
    --use_multi_tile \
    --tile_grid 3 \
    --batch_size 24 \
    --grad_checkpoint \
    --epochs 75 \
    --use_tile_transformer \
    --use_ordinal_head \
    --new_component_lr_mult 2.5 \
    --lr_patience 15
