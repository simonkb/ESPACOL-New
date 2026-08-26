#!/bin/bash
#SBATCH --job-name=optic_busi_v2
#SBATCH --partition=gpu
#SBATCH --array=0-4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=0-08:00:00
#SBATCH --output=/dpc/kuin0170/ESPACOL-New/slurm_logs/optic_busi_v2_%a_%j.out
#SBATCH --error=/dpc/kuin0170/ESPACOL-New/slurm_logs/optic_busi_v2_%a_%j.err
#SBATCH --account=kuin0170

mkdir -p /dpc/kuin0170/ESPACOL-New/slurm_logs

source /etc/profile.d/lmod.sh
module load miniconda/3
module load cuda/12.6
source activate G

cd /dpc/kuin0170/ESPACOL-New

export HF_HUB_OFFLINE=1
export PYTHONPATH=/dpc/kuin0170/ESPACOL-New:${PYTHONPATH}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# OPTIC-C v2 BUSI — full 5-fold CV (SLURM array job, folds 0-4)
#
# Fixes vs v1 (v1 mean=84.27% with fold 1 failing at 61.78%):
#
#   v1 root cause: default lr=5e-4 × new_component_lr_mult=2.5 gave
#   new components a 1.25e-3 starting LR — too high, causing wild val_acc
#   oscillation (27%→57%→52%→60%…). lr_patience=12 then triggered an LR
#   drop inside the 15-epoch frozen phase, so the backbone entered fine-tuning
#   at an already-reduced LR that differed fold-to-fold, producing high variance.
#
#   Fixes:
#     --lr 2e-4          New components start at 2e-4 × 2.5 = 5e-4 (matches
#                        DR v9, proven stable). Backbone unfreezes at 2e-4.
#     --backbone_freeze_epochs 25
#                        25 frozen epochs give new components enough time to
#                        converge before the backbone joins training.
#     --lr_patience 30   Higher than freeze epochs → NO LR drop occurs during
#                        the frozen phase. Every fold enters fine-tuning at the
#                        same backbone LR (2e-4), eliminating fold-to-fold variance.
#     --use_cosine_lr    After unfreeze, switch to CosineAnnealingLR decaying
#                        from 2e-4 → lr_min over 75 epochs. Eliminates
#                        ReduceLROnPlateau timing luck entirely post-unfreeze.
#     --lr_min 1e-6      Floor for cosine decay.
#
python train_busi.py \
    --busi_root Datasets/BUSI \
    --run_dir runs/optic_concept_busi_v2 \
    --folds ${SLURM_ARRAY_TASK_ID} \
    --use_multi_tile \
    --tile_grid 3 \
    --batch_size 16 \
    --grad_checkpoint \
    --epochs 100 \
    --lr 2e-4 \
    --use_tile_transformer \
    --use_grade_prototypes \
    --use_ordinal_head \
    --use_concept_prototype \
    --backbone_freeze_epochs 25 \
    --new_component_lr_mult 2.5 \
    --use_cosine_lr \
    --lr_min 1e-6 \
    --alpha 0 \
    --beta 0 \
    --lambda_gpa 0.1 \
    --lambda_proto_ce 1.0 \
    --lambda_tile_concept 0.5 \
    --proto_temperature 0.15 \
    --proto_label_smoothing 0.07 \
    --lr_patience 30 \
    --early_stop_patience 35 \
    --weight_decay 1e-5
