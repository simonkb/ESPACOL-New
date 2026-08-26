#!/bin/bash
#SBATCH --job-name=optic_idrid
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=0-04:00:00
#SBATCH --output=/dpc/kuin0170/ESPACOL-New/slurm_logs/optic_idrid_%j.out
#SBATCH --error=/dpc/kuin0170/ESPACOL-New/slurm_logs/optic_idrid_%j.err
#SBATCH --account=kuin0170

mkdir -p /dpc/kuin0170/ESPACOL-New/slurm_logs

source /etc/profile.d/lmod.sh
module load miniconda/3
module load cuda/12.6
source activate G

cd /dpc/kuin0170/ESPACOL-New

export HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# OPTIC-C IDRiD — official challenge split (413 train / 103 test)
#
# Uses the official IDRiD Disease Grading split rather than CV because
# the test labels are publicly available.
# val_fraction=0.2 → ~330 training images, 83 val, 103 test.
# Larger val set gives more stable early stopping on this small dataset.
# Single run; no SLURM array needed.
python train_dr.py \
    --dataset idrid \
    --dr_root Datasets/IDRiD \
    --run_dir runs/optic_concept_idrid_v2 \
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
    --weight_decay 1e-5 \
    --val_fraction 0.2
