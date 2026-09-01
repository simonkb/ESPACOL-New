#!/bin/bash
#SBATCH --job-name=optic_aptos_cv
#SBATCH --partition=gpu
#SBATCH --array=0-4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=0-18:00:00
#SBATCH --output=/dpc/kuin0170/ESPACOL-New/slurm_logs/optic_aptos_cv_%a_%j.out
#SBATCH --error=/dpc/kuin0170/ESPACOL-New/slurm_logs/optic_aptos_cv_%a_%j.err
#SBATCH --account=kuin0170

mkdir -p /dpc/kuin0170/ESPACOL-New/slurm_logs

source /etc/profile.d/lmod.sh
module load miniconda/3
module load cuda/12.6
source activate G

cd /dpc/kuin0170/ESPACOL-New

export HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# OPTIC-C APTOS 2019 5-fold CV
#
# Purpose: cross-dataset DR generalization benchmark (Table 1 APTOS column).
# APTOS 2019 (Aravind Eye Hospital, India): 3,662 training images, DR grades 0-4.
# Different patient population and equipment from Kaggle DR — tests domain shift.
#
# Design: 5-fold stratified CV on training set (test labels not public).
# Same OPTIC-C hyperparameters as Kaggle DR v10 (lr_min=8e-6, patience=12).
# 18h limit: APTOS has 3,662 images (~10% of Kaggle DR); each fold should
# finish in ~4h, leaving margin for 5 folds in the array.
python train_dr.py \
    --dataset aptos \
    --dr_root Datasets/aptos2019-blindness-detection \
    --run_dir runs/optic_concept_aptos_cv_3loss \
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
    --lambda_proto_ce 0 \
    --lambda_tile_concept 0.5 \
    --proto_temperature 0.15 \
    --proto_label_smoothing 0.07 \
    --lr_patience 12 \
    --lr_min 8e-6 \
    --early_stop_patience 30 \
    --weight_decay 1e-5
