#!/bin/bash
#SBATCH --job-name=optic_v7_cv
#SBATCH --partition=gpu
#SBATCH --array=0-9
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=220G
#SBATCH --time=2-00:00:00
#SBATCH --output=/dpc/kuin0170/ESPACOL-New/slurm_logs/optic_v7_cv_%a_%j.out
#SBATCH --error=/dpc/kuin0170/ESPACOL-New/slurm_logs/optic_v7_cv_%a_%j.err
#SBATCH --account=kuin0170

mkdir -p /dpc/kuin0170/ESPACOL-New/slurm_logs

source /etc/profile.d/lmod.sh
module load miniconda/3
module load cuda/12.6
source activate G

cd /dpc/kuin0170/ESPACOL-New

export HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# OPTIC-C v7 — full 10-fold CV (SLURM array job, folds 0-9)
#
# Changes vs v6:
#
#   1. Reverted cosine annealing back to ReduceLROnPlateau.
#      v6 showed cosine is the wrong tool: it decays LR immediately at
#      unfreeze, killing the high-LR feature adaptation phase that v4 "good"
#      folds depended on (full lr=2e-4 for ~25 post-unfreeze epochs before
#      first drop). All v6 folds scored 77-83% vs v4's 83-85%.
#
#   2. lr_patience reduced from 15 → 8.
#      With patience=15, noisy folds (v4: 2/5/7/9) occasionally beat their
#      best by a tiny margin, resetting the counter and never triggering the
#      second LR drop within 100 epochs. With patience=8, oscillations must
#      peak every <8 epochs to prevent a drop — much harder for noisy folds.
#      Guarantees 2 LR drops comfortably within the 75 post-unfreeze epochs.
#
#   3. Early stopping counter now resets at backbone unfreeze.
#      v6 fold 2 stopped at epoch 34: a lucky val_acc=41.98% spike at epoch 4
#      (frozen phase) set the baseline, the post-unfreeze disruption dropped
#      it to ~12%, and the counter ran out before the model could recover.
#      Now the early stopping best and counter reset at unfreeze, matching the
#      LR scheduler reset that was already there.
python train_dr.py \
    --dr_root Datasets/DR \
    --run_dir runs/optic_concept_cv_v7 \
    --folds ${SLURM_ARRAY_TASK_ID} \
    --use_multi_tile \
    --tile_grid 3 \
    --batch_size 24 \
    --grad_checkpoint \
    --epochs 100 \
    --use_tile_transformer \
    --use_grade_prototypes \
    --use_ordinal_head \
    --use_concept_prototype \
    --backbone_freeze_epochs 25 \
    --new_component_lr_mult 2.5 \
    --alpha 0 \
    --beta 0 \
    --gamma 0 \
    --lambda_osd 0 \
    --lambda_tcl 0 \
    --lambda_gpa 0.1 \
    --lambda_proto_ce 1.0 \
    --lambda_concept_align 0.5 \
    --lambda_tile_concept 0.5 \
    --proto_temperature 0.15 \
    --proto_label_smoothing 0.07 \
    --lr_patience 8 \
    --early_stop_patience 30 \
    --weight_decay 1e-5
