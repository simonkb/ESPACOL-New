#!/bin/bash
#SBATCH --job-name=optic_v6_cv
#SBATCH --partition=gpu
#SBATCH --array=0-9
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=220G
#SBATCH --time=2-00:00:00
#SBATCH --output=/dpc/kuin0170/ESPACOL-New/slurm_logs/optic_v6_cv_%a_%j.out
#SBATCH --error=/dpc/kuin0170/ESPACOL-New/slurm_logs/optic_v6_cv_%a_%j.err
#SBATCH --account=kuin0170

mkdir -p /dpc/kuin0170/ESPACOL-New/slurm_logs

source /etc/profile.d/lmod.sh
module load miniconda/3
module load cuda/12.6
source activate G

cd /dpc/kuin0170/ESPACOL-New

export HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# OPTIC-C v6 — full 10-fold CV (SLURM array job, folds 0-9)
#
# Changes vs v5:
#   - CosineAnnealingLR replaces ReduceLROnPlateau after backbone unfreeze.
#     At epoch 26, LRs reset to base (backbone=2e-4, optic_new=5e-4) and
#     decay deterministically to lr_min=1e-6 over the remaining 75 epochs.
#     In v4, folds 2/5/7/9 stayed at lr=4e-5 for all 100 epochs because
#     val_acc oscillations reset the patience counter — those folds scored
#     77-80% vs 83-85% for folds that got a second LR drop. Cosine gives
#     every fold the same LR trajectory regardless of validation noise.
#   - All other hyperparameters identical to v5.
python train_dr.py \
    --dr_root Datasets/DR \
    --run_dir runs/optic_concept_cv_v6 \
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
    --early_stop_patience 30 \
    --weight_decay 1e-5 \
    --use_cosine_lr
