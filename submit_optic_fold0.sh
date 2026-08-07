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

# OPTIC v5: backbone freeze + longer schedule.
#
# Root cause diagnosis (v3 and v4): crashes (12-14% val_acc) are caused by CTOT
# itself. CTOT has 6.5M random params; without freezing, the backbone fine-tunes
# through unstable CTOT outputs, causing attention-pattern collapses. Removing
# GPA/OSD/TCL (v4) did NOT fix the crashes — confirming CTOT is the culprit.
#
# Fix: freeze EfficientNet backbone for first 25 epochs so CTOT+GPA initialise
# on stable pretrained features. Backbone unfreezes at epoch 26 for joint FT.
#
# Schedule (125 epochs):
#   Ep  1-19: backbone frozen, CTOT+GPA+ODH learn on fixed features
#   Ep 20-25: text encoder FT starts (proven to stabilise representations)
#   Ep    26: backbone unfreezes — joint fine-tuning begins
#   LR drop : patience=20 → ~ep80+ (after best val_loss), leaving ~45 FT epochs
#   ES      : patience=30 on val_acc → fires ~ep100+ if needed
#
# GPA re-added with lambda=0.1: required for grade-prototype explainability.
# OSD and TCL remain OFF: they added gradient noise without accuracy benefit.
#
# Baseline (multi-tile + AttentionPool, no freeze): 84.25%. Target: >= 85%.
python train_dr.py \
    --dr_root Datasets/DR \
    --run_dir runs/optic_dr_fold0_v5 \
    --folds 0 \
    --use_multi_tile \
    --tile_grid 3 \
    --batch_size 24 \
    --grad_checkpoint \
    --epochs 125 \
    --use_tile_transformer \
    --use_grade_prototypes \
    --use_ordinal_head \
    --backbone_freeze_epochs 25 \
    --new_component_lr_mult 2.5 \
    --lambda_gpa 0.1 \
    --lr_patience 20 \
    --early_stop_patience 30
