#!/bin/bash
#SBATCH --job-name=optic_concept_fold0_v5
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=220G
#SBATCH --time=2-00:00:00
#SBATCH --output=/dpc/kuin0170/ESPACOL-New/slurm_logs/optic_concept_fold0_v5_%j.out
#SBATCH --error=/dpc/kuin0170/ESPACOL-New/slurm_logs/optic_concept_fold0_v5_%j.err
#SBATCH --account=kuin0170

mkdir -p /dpc/kuin0170/ESPACOL-New/slurm_logs

source /etc/profile.d/lmod.sh
module load miniconda/3
module load cuda/12.6
source activate G

cd /dpc/kuin0170/ESPACOL-New

export HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# OPTIC-C v5 changes vs v4:
#
#   1. proto_CE gains ordinal denominator penalty (OrdinalPrototypeLoss):
#      |y - k| is added to class k's logit before softmax. True class
#      unchanged (dist=0); farther wrong grades inflate the denominator more.
#      Same strategy as PCOL/SCOLw r_{a,n}. Loss scale and label_smoothing
#      unchanged so training dynamics are broadly preserved.
#
#   2. Concept alignment loss now active (was silently zero in all prior runs):
#      grade_text_embeds were only fetched when use_image_text=True, which is
#      never set in OPTIC-C. Fixed in trainer.py — embeds fetched whenever
#      use_concept_prototype=True. lambda_concept_align=0.5 now takes effect.
#      Watch train_loss_concept_align in logs; if it starts large (>0.5) and
#      dominates, lower lambda_concept_align.
#
#   3. Image-text head fully removed from architecture (was already inactive).
python train_dr.py \
    --dr_root Datasets/DR \
    --run_dir runs/optic_concept_fold0_v5 \
    --folds 0 \
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
