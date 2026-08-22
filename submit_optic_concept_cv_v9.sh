#!/bin/bash
#SBATCH --job-name=optic_v9_cv
#SBATCH --partition=gpu
#SBATCH --array=0-9
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=220G
#SBATCH --time=2-00:00:00
#SBATCH --output=/dpc/kuin0170/ESPACOL-New/slurm_logs/optic_v9_cv_%a_%j.out
#SBATCH --error=/dpc/kuin0170/ESPACOL-New/slurm_logs/optic_v9_cv_%a_%j.err
#SBATCH --account=kuin0170

mkdir -p /dpc/kuin0170/ESPACOL-New/slurm_logs

source /etc/profile.d/lmod.sh
module load miniconda/3
module load cuda/12.6
source activate G

cd /dpc/kuin0170/ESPACOL-New

export HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# OPTIC-C v9 — full 10-fold CV (SLURM array job, folds 0-9)
#
# Changes vs v8:
#
#   Fixed critical bug: LR restored to full base (2e-4) was too high for backbone.
#
#   In v8, at unfreeze the optimizer LR was restored to the full base values
#   (backbone=2e-4, optic_new=5e-4). This was 5x too aggressive for a pretrained
#   backbone receiving its first gradient update after 25 frozen epochs. The result
#   was that all 9 completed folds scored only 77-79%, worse than v4's bad folds.
#
#   The best ever result (v4 fold 8: 85.04%) entered fine-tuning at backbone=4e-5
#   (one LR drop had occurred during the frozen phase). Now at unfreeze we restore
#   to lr_factor * base = 0.2 * 2e-4 = 4e-5, giving every fold the same consistent
#   starting LR as v4's best fold.
#
#   lr_patience changed 8 -> 12:
#     patience=8 with 2e-4 caused 3+ drops in 75 epochs (too aggressive).
#     patience=12 gives 2 reliable drops from 4e-5 within 75 epochs, matching
#     the v4 fold 8 pattern (62 epochs at 4e-5, then 12 at 8e-6).
#
#   Also fixed NFS race condition in setup_logging makedirs (fold 8 crash in v8).
python train_dr.py \
    --dr_root Datasets/DR \
    --run_dir runs/optic_concept_cv_v9 \
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
    --lr_patience 12 \
    --early_stop_patience 30 \
    --weight_decay 1e-5
