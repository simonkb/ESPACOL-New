#!/bin/bash
#SBATCH --job-name=optic_v8_cv
#SBATCH --partition=gpu
#SBATCH --array=0-9
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=220G
#SBATCH --time=2-00:00:00
#SBATCH --output=/dpc/kuin0170/ESPACOL-New/slurm_logs/optic_v8_cv_%a_%j.out
#SBATCH --error=/dpc/kuin0170/ESPACOL-New/slurm_logs/optic_v8_cv_%a_%j.err
#SBATCH --account=kuin0170

mkdir -p /dpc/kuin0170/ESPACOL-New/slurm_logs

source /etc/profile.d/lmod.sh
module load miniconda/3
module load cuda/12.6
source activate G

cd /dpc/kuin0170/ESPACOL-New

export HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# OPTIC-C v8 — full 10-fold CV (SLURM array job, folds 0-9)
#
# Changes vs v7:
#
#   Fixed critical bug: ReduceLROnPlateau fires during frozen phase.
#
#   In v7 (and all prior versions), scheduler.step(val_loss) was called every
#   epoch, including the 25 frozen epochs. Val_loss during the frozen phase is
#   noisy (backbone is fixed, only CTOT/GPA train), so with patience=8 the
#   scheduler drops LR twice before the backbone ever trains:
#     - Ep 11: 2e-4 → 4e-5  (after 9 noisy frozen epochs)
#     - Ep 24: 4e-5 → 8e-6  (another 9 frozen epochs)
#   At unfreeze (ep 26) the scheduler patience counter was reset but the
#   optimizer LR was NOT restored — all 75 post-unfreeze epochs ran at
#   minimum LR (8e-6 instead of 2e-4). v7 fold 8 confirmed this: val_acc
#   was only 73% at ep 49, still climbing at minimum LR.
#
#   Fix: at backbone unfreeze, restore all optimizer param group LRs to
#   their original base values (self._base_lrs), identical to what the
#   cosine branch already did. No other changes from v7.
python train_dr.py \
    --dr_root Datasets/DR \
    --run_dir runs/optic_concept_cv_v8 \
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
