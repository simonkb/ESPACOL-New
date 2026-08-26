#!/bin/bash
#SBATCH --job-name=optic_idrid_cv
#SBATCH --partition=gpu
#SBATCH --array=0-4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=0-12:00:00
#SBATCH --output=/dpc/kuin0170/ESPACOL-New/slurm_logs/optic_idrid_cv_%a_%j.out
#SBATCH --error=/dpc/kuin0170/ESPACOL-New/slurm_logs/optic_idrid_cv_%a_%j.err
#SBATCH --account=kuin0170

mkdir -p /dpc/kuin0170/ESPACOL-New/slurm_logs

source /etc/profile.d/lmod.sh
module load miniconda/3
module load cuda/12.6
source activate G

cd /dpc/kuin0170/ESPACOL-New

export HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# OPTIC-C IDRiD 5-fold CV
#
# Same OPTIC-C architecture and hyperparameters as v10 (Kaggle DR).
# IDRiD has 516 images (413 train + 103 test combined into 5-fold CV).
# Used for:
#   1. Cross-dataset DR grading benchmark (Table 1 IDRiD column)
#   2. Explainability evaluation via pixel-level lesion masks (Pointing Game, IoU)
#
# Key differences from DR v10:
#   --dataset idrid           : switches loader to IDRiD CSV-based loader
#   --dr_root Datasets/IDRiD  : IDRiD root directory
#   --mem 64G                 : smaller dataset, no need for 220G
#   --array=0-4               : 5 folds (not 10)
python train_dr.py \
    --dataset idrid \
    --dr_root Datasets/IDRiD \
    --run_dir runs/optic_concept_idrid_cv \
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
    --lambda_tile_concept 0.5 \
    --proto_temperature 0.15 \
    --proto_label_smoothing 0.07 \
    --lr_patience 12 \
    --lr_min 8e-6 \
    --early_stop_patience 30 \
    --weight_decay 1e-5
