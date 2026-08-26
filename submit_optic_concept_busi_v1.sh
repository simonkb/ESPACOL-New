#!/bin/bash
#SBATCH --job-name=optic_busi_v1
#SBATCH --partition=gpu
#SBATCH --array=0-4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=0-08:00:00
#SBATCH --output=/dpc/kuin0170/ESPACOL-New/slurm_logs/optic_busi_v1_%a_%j.out
#SBATCH --error=/dpc/kuin0170/ESPACOL-New/slurm_logs/optic_busi_v1_%a_%j.err
#SBATCH --account=kuin0170

mkdir -p /dpc/kuin0170/ESPACOL-New/slurm_logs

source /etc/profile.d/lmod.sh
module load miniconda/3
module load cuda/12.6
source activate G

cd /dpc/kuin0170/ESPACOL-New

export HF_HUB_OFFLINE=1
export PYTHONPATH=/dpc/kuin0170/ESPACOL-New:${PYTHONPATH}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# OPTIC-C v1 BUSI — full 5-fold CV (SLURM array job, folds 0-4)
#
# Architecture mirrors DR v9 (same CTOT + GPA + ODH + CGPM stack).
# BUSI-specific adaptations:
#   - 5 folds (vs 10 for DR), 3 classes (normal/benign/malignant)
#   - backbone_freeze_epochs=15 (shorter: BUSI is ~624 train images, converges fast)
#   - batch_size=16 (smaller dataset; 24 caused empty-class batches in early folds)
#   - epochs=100, lr_patience=12, early_stop_patience=30 (same as DR v9)
#   - alpha=0, beta=0: disable PCOL/SCOLw; novel losses dominate
#   - lambda_proto_ce=1.0, lambda_tile_concept=0.5 (same as DR v9)
#   - lambda_gpa=0.1 (same as DR v9)
#   - n_concepts=10 (BUSI_CONCEPTS has 10 entries; set in BUSIConfig)
#
python train_busi.py \
    --busi_root Datasets/BUSI \
    --run_dir runs/optic_concept_busi_v1 \
    --folds ${SLURM_ARRAY_TASK_ID} \
    --use_multi_tile \
    --tile_grid 3 \
    --batch_size 16 \
    --grad_checkpoint \
    --epochs 100 \
    --use_tile_transformer \
    --use_grade_prototypes \
    --use_ordinal_head \
    --use_concept_prototype \
    --backbone_freeze_epochs 15 \
    --new_component_lr_mult 2.5 \
    --alpha 0 \
    --beta 0 \
    --lambda_gpa 0.1 \
    --lambda_proto_ce 1.0 \
    --lambda_tile_concept 0.5 \
    --proto_temperature 0.15 \
    --proto_label_smoothing 0.07 \
    --lr_patience 12 \
    --early_stop_patience 30 \
    --weight_decay 1e-5
