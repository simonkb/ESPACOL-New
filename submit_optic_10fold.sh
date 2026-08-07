#!/bin/bash
#SBATCH --job-name=optic_dr_10fold
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=220G
#SBATCH --time=2-00:00:00
#SBATCH --array=0-9
#SBATCH --output=/dpc/kuin0170/ESPACOL-New/slurm_logs/optic_fold%a_%j.out
#SBATCH --error=/dpc/kuin0170/ESPACOL-New/slurm_logs/optic_fold%a_%j.err
#SBATCH --account=kuin0170

mkdir -p /dpc/kuin0170/ESPACOL-New/slurm_logs

source /etc/profile.d/lmod.sh
module load miniconda/3
module load cuda/12.6
source activate G

cd /dpc/kuin0170/ESPACOL-New

export HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Full OPTIC v5 10-fold cross-validation on DR.
# Run submit_optic_fold0.sh first to validate >= 85% before launching this.
python train_dr.py \
    --dr_root Datasets/DR \
    --run_dir runs/optic_dr_10fold_v5 \
    --folds ${SLURM_ARRAY_TASK_ID} \
    --use_multi_tile \
    --tile_grid 3 \
    --batch_size 24 \
    --grad_checkpoint \
    --epochs 125 \
    --use_tile_transformer \
    --use_grade_prototypes \
    --use_ordinal_head \
    --backbone_freeze_epochs 25 \
    --new_component_lr_mult 2.5 \
    --lambda_gpa 0.1 \
    --lr_patience 20 \
    --early_stop_patience 30
