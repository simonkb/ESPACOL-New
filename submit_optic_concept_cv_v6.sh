#!/bin/bash
#SBATCH --job-name=optic_concept_cv_v6
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=220G
#SBATCH --time=10-00:00:00
#SBATCH --output=/dpc/kuin0170/ESPACOL-New/slurm_logs/optic_concept_cv_v6_%j.out
#SBATCH --error=/dpc/kuin0170/ESPACOL-New/slurm_logs/optic_concept_cv_v6_%j.err
#SBATCH --account=kuin0170

mkdir -p /dpc/kuin0170/ESPACOL-New/slurm_logs

source /etc/profile.d/lmod.sh
module load miniconda/3
module load cuda/12.6
source activate G

cd /dpc/kuin0170/ESPACOL-New

export HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# OPTIC-C v6 — full 10-fold CV
#
# Changes vs v5:
#
#   1. CosineAnnealingLR replaces ReduceLROnPlateau after backbone unfreeze.
#      At epoch 26 (unfreeze), LRs reset to their base values (backbone=2e-4,
#      optic_new=5e-4) and a cosine schedule decays them to lr_min=1e-6 over
#      the remaining 75 epochs. This is deterministic — every fold gets the
#      same LR trajectory regardless of how noisy its validation accuracy is.
#
#      In v4, 4 of 9 folds (2, 5, 7, 9) never received their second LR drop
#      within 100 epochs because val_acc oscillations kept resetting the
#      lr_patience=15 counter. Those folds plateaued at lr=4e-5 and scored
#      77-80% vs 83-85% for folds that did get a second drop. Cosine removes
#      this timing dependency entirely.
#
#      ReduceLROnPlateau is still used during the frozen phase (epochs 1-25)
#      as a safety net, but in practice never triggers (25 < patience=15+
#      sufficient improvement epochs).
#
#   2. All other hyperparameters identical to v5 (ordinal CE penalty, concept
#      alignment active, image-text head removed).
python train_dr.py \
    --dr_root Datasets/DR \
    --run_dir runs/optic_concept_cv_v6 \
    --folds all \
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
