#!/bin/bash
#SBATCH --job-name=optic_v10_cv
#SBATCH --partition=gpu
#SBATCH --array=0-9
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=220G
#SBATCH --time=2-00:00:00
#SBATCH --output=/dpc/kuin0170/ESPACOL-New/slurm_logs/optic_v10_cv_%a_%j.out
#SBATCH --error=/dpc/kuin0170/ESPACOL-New/slurm_logs/optic_v10_cv_%a_%j.err
#SBATCH --account=kuin0170

mkdir -p /dpc/kuin0170/ESPACOL-New/slurm_logs

source /etc/profile.d/lmod.sh
module load miniconda/3
module load cuda/12.6
source activate G

cd /dpc/kuin0170/ESPACOL-New

export HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# OPTIC-C v10 — full 10-fold CV (SLURM array job, folds 0-9)
#
# Single change vs v9: lr_min raised from 1e-6 to 8e-6.
#
# In v9, ReduceLROnPlateau could cascade through 3-4 drops during 75 post-
# unfreeze epochs: 4e-5 → 8e-6 → 1.6e-6 → 1e-6 (dead LR). Folds 0, 5, 6
# spent 12-18 epochs at 1e-6 (near-zero gradient updates) which capped them
# at 84%. Folds 1 and 4 avoided this by luck of timing (their drops fired
# late), and both exceeded 85%.
#
# With lr_min=8e-6, ReduceLROnPlateau can only execute one fine-tuning drop
# (4e-5 → 8e-6, then the floor is hit). Every fold trains at 4e-5 until the
# first plateau, then at 8e-6 for the rest — matching v4 fold 8's schedule
# (85.04%) for all folds deterministically. The 85%+ folds from v9 are
# unaffected: fold 4 had only 2 epochs below 8e-6, fold 1's 16 epochs at
# 1.6e-6 are replaced by 8e-6 (strictly more gradient signal).
python train_dr.py \
    --dr_root Datasets/DR \
    --run_dir runs/optic_concept_cv_v10 \
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
    --lr_min 8e-6 \
    --early_stop_patience 30 \
    --weight_decay 1e-5
