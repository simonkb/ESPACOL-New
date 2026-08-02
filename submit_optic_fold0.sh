#!/bin/bash
#SBATCH --job-name=optic_dr_fold0
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=220G
#SBATCH --time=0-12:00:00
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
python train_dr.py \
    --dr_root Datasets/DR \
    --run_dir runs/optic_dr_fold0 \
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
    --use_tile_consistency \
    --lambda_tcl 0.1
