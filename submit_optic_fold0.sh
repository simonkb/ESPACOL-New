#!/bin/bash
#SBATCH --job-name=optic_dr_fold0
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=220G
#SBATCH --time=2-00:00:00
#SBATCH --output=/dpc/kuin0170/ESPACOL-New/slurm_logs/optic_fold0_%j.out
#SBATCH --error=/dpc/kuin0170/ESPACOL-New/slurm_logs/optic_fold0_%j.err
#SBATCH --account=kuin0170

mkdir -p /dpc/kuin0170/ESPACOL-New/slurm_logs

source /etc/profile.d/lmod.sh
module load miniconda/3
module load cuda/12.6
source activate G

cd /dpc/kuin0170/ESPACOL-New

export HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# OPTIC fold-0 validation run.
# Enables: CTOT + GradePrototypes + OrdinalHead + OSD + TileConsistency
# Baseline achieves 84.25% on fold 0. Target: >= 85.5%
#
# Key tuning vs previous runs:
#   lambda_gpa 0.1  (was 0.5 → caused val_acc oscillations 13%→69%→18%→69%)
#   lr_patience 15  (was 8  → LR was dropping before CTOT had time to converge)
python train_dr.py \
    --dr_root Datasets/DR \
    --run_dir runs/optic_dr_fold0_v3 \
    --folds 0 \
    --use_multi_tile \
    --tile_grid 3 \
    --batch_size 24 \
    --grad_checkpoint \
    --epochs 75 \
    --use_tile_transformer \
    --use_grade_prototypes \
    --use_ordinal_head \
    --use_osd_loss \
    --lambda_osd 0.5 \
    --osd_margin 0.1 \
    --use_tile_consistency \
    --lambda_tcl 0.1 \
    --new_component_lr_mult 2.5 \
    --lambda_gpa 0.1 \
    --lr_patience 15
