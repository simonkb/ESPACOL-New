#!/bin/bash
#SBATCH --job-name=optic_concept_fold0_v3
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=220G
#SBATCH --time=2-00:00:00
#SBATCH --output=/dpc/kuin0170/ESPACOL-New/slurm_logs/optic_concept_fold0_v3_%j.out
#SBATCH --error=/dpc/kuin0170/ESPACOL-New/slurm_logs/optic_concept_fold0_v3_%j.err
#SBATCH --account=kuin0170

mkdir -p /dpc/kuin0170/ESPACOL-New/slurm_logs

source /etc/profile.d/lmod.sh
module load miniconda/3
module load cuda/12.6
source activate G

cd /dpc/kuin0170/ESPACOL-New

export HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# OPTIC-C v3: targeted fixes based on v2 convergence analysis.
#
# Root cause identified in v2: proto_CE collapsed to 0.005 on training data
# while val_acc plateaued at 84.28% — proto_CE was overfitting.
#
# Changes from v2:
#   1. proto_label_smoothing 0.0 → 0.1:
#      proto_CE cannot collapse to near-zero; regularises the dominant loss.
#      Expected gain: +0.3–0.5%
#
#   2. proto_temperature 0.1 → 0.2:
#      Softer cosine similarity → model cannot hyper-specialise on training prototypes.
#      Complements label smoothing.
#
#   3. lambda_proto_ce 1.5 → 1.0:
#      The overfitting loss gets less dominance; CORAL and tile_concept get
#      relatively more signal.
#
#   4. lambda_tile_concept 0.3 → 0.5:
#      tc loss barely moved in v2 (0.414 → 0.397 over 60 epochs); starved of gradient.
#
#   5. gamma=0 (removed IT loss):
#      it=1.387 barely moved in v2; redundant with concept_align; saves compute.
#
#   6. epochs 155 → 100, early_stop_patience 55 → 30:
#      v2 peaked at ep73; ES would fire ep128 anyway. 100 epochs is generous.
python train_dr.py \
    --dr_root Datasets/DR \
    --run_dir runs/optic_concept_fold0_v3 \
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
    --proto_temperature 0.2 \
    --proto_label_smoothing 0.1 \
    --lr_patience 15 \
    --early_stop_patience 30
