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

# OPTIC v6: fix ES/LR patience mismatch from v5.
#
# v5 post-mortem: best val_acc = 77.80% at ep93.
#   LR patience=20 → LR drops at ep113 (ep93+20)
#   ES  patience=30 → ES fires at ep123 (ep93+30)
#   Post-LR consolidation window: only 10 epochs (ep114–123)
# The model was still recovering when ES killed it: ep120=74.5% → ep123=75.8%
# (rising, not stalled). 10 epochs is far too short after an LR drop.
#
# Fix: reduce lr_patience to 15 (LR drops sooner) AND widen es_patience to 55
# (ES waits longer). This gives ~40 post-LR epochs instead of 10, plus a
# possible second LR drop (4e-05 → 8e-06) at ~peak+30 for final convergence.
#
# Schedule (155 epochs):
#   Ep  1-25: backbone frozen — CTOT/GPA initialise on fixed pretrained features
#   Ep    26: backbone unfreezes — joint fine-tuning begins
#   Ep    ~X: peak val_acc reached
#   Ep  X+15: first LR drop (2e-04 → 4e-05) — consolidation begins
#   Ep  X+30: second LR drop (4e-05 → 8e-06) if no val_loss improvement
#   Ep  X+55: ES fires if val_acc never improves — ~40 epochs of reduced-LR FT
#
# Baseline (multi-tile + AttentionPool, no freeze): 84.25%. Target: >= 83%.
python train_dr.py \
    --dr_root Datasets/DR \
    --run_dir runs/optic_dr_fold0_v6 \
    --folds 0 \
    --use_multi_tile \
    --tile_grid 3 \
    --batch_size 24 \
    --grad_checkpoint \
    --epochs 155 \
    --use_tile_transformer \
    --use_grade_prototypes \
    --use_ordinal_head \
    --backbone_freeze_epochs 25 \
    --new_component_lr_mult 2.5 \
    --lambda_gpa 0.1 \
    --lr_patience 15 \
    --early_stop_patience 55
