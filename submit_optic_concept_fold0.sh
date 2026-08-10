#!/bin/bash
#SBATCH --job-name=optic_concept_fold0
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=220G
#SBATCH --time=2-00:00:00
#SBATCH --output=/dpc/kuin0170/ESPACOL-New/slurm_logs/optic_concept_fold0_%j.out
#SBATCH --error=/dpc/kuin0170/ESPACOL-New/slurm_logs/optic_concept_fold0_%j.err
#SBATCH --account=kuin0170

mkdir -p /dpc/kuin0170/ESPACOL-New/slurm_logs

source /etc/profile.d/lmod.sh
module load miniconda/3
module load cuda/12.6
source activate G

cd /dpc/kuin0170/ESPACOL-New

export HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# OPTIC-C: Concept-Grounded Grade Prototype Architecture (fold 0 validation run)
#
# Key design: novel losses dominate ~98% of gradient signal.
# SCOLw (alpha=0, beta=0) and PCOL removed — they are not our contributions.
# Novel contributions:
#   - L_proto_CE (1.5x):  cosine prototype CrossEntropy — dominant signal
#   - L_CORAL (1.0x):     ordinal distribution head (CORAL) — ordinal regression
#   - L_concept_align (0.5x): grade prototype ↔ grade text cosine alignment
#   - L_tile_concept (0.3x):  per-tile concept BCE vs clinical grade-concept targets
#   - L_GPA (0.1x):       GPA tile evidence supervision
#   - L_IT (gamma=0.05x): BiomedCLIP image-text alignment
#
# Dual explainability:
#   1. Spatial heatmap: GPA tile_weights for predicted grade
#   2. Concept report:  per-tile concept scores (microaneurysms, hemorrhages, etc.)
#
# Architecture: CTOT + GPA + CORAL + ConceptGradePrototypeModule
#   Backbone: EfficientNetV2S (frozen ep1-25, joint fine-tuning ep26+)
#   Tile grid: 3×3 = 9 local + 1 global = 10 tiles @ 300×300
#   CTOT: d=512, nhead=8, 2-layer pre-LN Transformer
#   GPA: K=5 grade prototypes (512-dim)
#   CORAL: 4 threshold heads P(Y>k)
#   ConceptModule: K=5 grade prototypes (1280-dim) + concept_projection (1280→128)
#
# Target: >85% val_acc, val_mae < 0.1933
# Baseline (multi-tile + AttentionPool): 84.25%
# v6 OPTIC best: 83.63% (still converging at ep88)
python train_dr.py \
    --dr_root Datasets/DR \
    --run_dir runs/optic_concept_fold0_v2 \
    --folds 0 \
    --use_multi_tile \
    --tile_grid 3 \
    --batch_size 24 \
    --grad_checkpoint \
    --epochs 155 \
    --use_tile_transformer \
    --use_grade_prototypes \
    --use_ordinal_head \
    --use_concept_prototype \
    --backbone_freeze_epochs 25 \
    --new_component_lr_mult 2.5 \
    --alpha 0 \
    --beta 0 \
    --lambda_osd 0 \
    --lambda_tcl 0 \
    --lambda_gpa 0.1 \
    --lambda_proto_ce 1.5 \
    --lambda_concept_align 0.5 \
    --lambda_tile_concept 0.3 \
    --proto_temperature 0.1 \
    --lr_patience 15 \
    --early_stop_patience 55
