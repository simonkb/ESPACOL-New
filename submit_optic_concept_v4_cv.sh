#!/bin/bash
#SBATCH --job-name=optic_v4_cv
#SBATCH --partition=gpu
#SBATCH --array=1-9
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=220G
#SBATCH --time=2-00:00:00
#SBATCH --output=/dpc/kuin0170/ESPACOL-New/slurm_logs/optic_v4_cv_%a_%j.out
#SBATCH --error=/dpc/kuin0170/ESPACOL-New/slurm_logs/optic_v4_cv_%a_%j.err
#SBATCH --account=kuin0170

mkdir -p /dpc/kuin0170/ESPACOL-New/slurm_logs

source /etc/profile.d/lmod.sh
module load miniconda/3
module load cuda/12.6
source activate G

cd /dpc/kuin0170/ESPACOL-New

export HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# OPTIC-C v4 — full 10-fold cross-validation (SLURM array job, folds 1-9)
# Fold 0 already done: runs/optic_concept_fold0_v4 → TEST acc=84.82%
#
# v4 changes vs v3:
#   - LR scheduler reset at unfreeze (trainer.py): backbone fine-tunes at
#     intended 2e-4, not accidental 4e-5 caused by freeze-phase plateau drop.
#     This is the primary fix for inter-fold variance (v3 CV had 70s-84%).
#   - Stronger augmentation (dataloaders.py): rotation 10°→45°, jitter 0.1→0.2
#   - proto_label_smoothing 0.1→0.07, proto_temperature 0.2→0.15
#   - weight_decay 1e-6→1e-5
python train_dr.py \
    --dr_root Datasets/DR \
    --run_dir runs/optic_concept_v4_cv \
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
    --lr_patience 15 \
    --early_stop_patience 30 \
    --weight_decay 1e-5
