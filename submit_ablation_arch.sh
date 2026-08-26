#!/bin/bash
#SBATCH --job-name=optic_ablation
#SBATCH --partition=gpu
#SBATCH --array=0-5            # 6 variants × fold 0 only
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=220G
#SBATCH --time=2-00:00:00
#SBATCH --output=/dpc/kuin0170/ESPACOL-New/slurm_logs/ablation_%a_%j.out
#SBATCH --error=/dpc/kuin0170/ESPACOL-New/slurm_logs/ablation_%a_%j.err
#SBATCH --account=kuin0170

mkdir -p /dpc/kuin0170/ESPACOL-New/slurm_logs

source /etc/profile.d/lmod.sh
module load miniconda/3
module load cuda/12.6
source activate G

cd /dpc/kuin0170/ESPACOL-New

export HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Array index = variant (0-5), all run on fold 0.
#   Task 0 → V0: Eff-V2S + CORAL, single-tile
#   Task 1 → V1: +Multi-tile (AttentionPool)
#   Task 2 → V2: +CTOT
#   Task 3 → V3: +GPA
#   Task 4 → V4: +Proto CE (tile concept off)
#   Task 5 → V5: Full OPTIC-C (identical to v10)
VARIANT=${SLURM_ARRAY_TASK_ID}
FOLD=0

# ── Hyperparameters held constant across ALL variants ───────────────────────
# These match the v10 full-run settings exactly so the only thing that changes
# between rows is which components are enabled.
BASE_ARGS="
    --dr_root Datasets/DR
    --folds ${FOLD}
    --batch_size 24
    --grad_checkpoint
    --epochs 100
    --backbone_freeze_epochs 25
    --new_component_lr_mult 2.5
    --alpha 0
    --beta 0
    --gamma 0
    --lambda_osd 0
    --lambda_tcl 0
    --lr_patience 12
    --lr_min 8e-6
    --early_stop_patience 30
    --weight_decay 1e-5
"

case ${VARIANT} in

0)
    # V0: EfficientNetV2S + CORAL ordinal head, single tile
    # No multi-tile, no CTOT, no GPA, no concept prototype.
    # Establishes how much the base ordinal formulation contributes.
    python train_dr.py ${BASE_ARGS} \
        --run_dir runs/ablation/v0_single_tile/fold${FOLD} \
        --use_ordinal_head \
        --lambda_gpa 0 \
        --lambda_proto_ce 0 \
        --lambda_tile_concept 0
    ;;

1)
    # V1: +Multi-tile (3×3 grid with AttentionPool, no Transformer)
    # Tests whether tiling alone — without cross-tile reasoning — improves over
    # a single-crop view of the fundus.
    python train_dr.py ${BASE_ARGS} \
        --run_dir runs/ablation/v1_multitile/fold${FOLD} \
        --use_multi_tile \
        --tile_grid 3 \
        --use_ordinal_head \
        --lambda_gpa 0 \
        --lambda_proto_ce 0 \
        --lambda_tile_concept 0
    ;;

2)
    # V2: +CTOT (CrossTileOrdinalTransformer replaces AttentionPool)
    # Tests whether cross-tile ordinal reasoning via the [GRADE] token adds
    # value beyond naive attention pooling.
    python train_dr.py ${BASE_ARGS} \
        --run_dir runs/ablation/v2_ctot/fold${FOLD} \
        --use_multi_tile \
        --tile_grid 3 \
        --use_tile_transformer \
        --use_ordinal_head \
        --lambda_gpa 0 \
        --lambda_proto_ce 0 \
        --lambda_tile_concept 0
    ;;

3)
    # V3: +GPA (GradePrototypeAttention + L_gpa supervision)
    # Tests whether per-tile grade evidence maps improve grading beyond
    # CTOT alone (GPA adds a spatial explainability inductive bias).
    python train_dr.py ${BASE_ARGS} \
        --run_dir runs/ablation/v3_gpa/fold${FOLD} \
        --use_multi_tile \
        --tile_grid 3 \
        --use_tile_transformer \
        --use_grade_prototypes \
        --use_ordinal_head \
        --lambda_gpa 0.1 \
        --lambda_proto_ce 0 \
        --lambda_tile_concept 0
    ;;

4)
    # V4: +Proto CE (ConceptGradePrototypeModule with L_proto_CE only)
    # Tile concept BCE is off (lambda_tile_concept=0): tests whether the
    # cosine prototype cross-entropy loss alone — without clinical concept
    # supervision — provides the accuracy benefit, or whether tile concept
    # guidance (V5) is also necessary.
    python train_dr.py ${BASE_ARGS} \
        --run_dir runs/ablation/v4_proto_ce/fold${FOLD} \
        --use_multi_tile \
        --tile_grid 3 \
        --use_tile_transformer \
        --use_grade_prototypes \
        --use_ordinal_head \
        --use_concept_prototype \
        --lambda_gpa 0.1 \
        --lambda_proto_ce 1.0 \
        --lambda_tile_concept 0 \
        --proto_temperature 0.15 \
        --proto_label_smoothing 0.07
    ;;

5)
    # V5: Full OPTIC-C (all components, all losses) — identical to v10
    # This is the complete model and the top row of Table 1.
    python train_dr.py ${BASE_ARGS} \
        --run_dir runs/ablation/v5_full_optic_c/fold${FOLD} \
        --use_multi_tile \
        --tile_grid 3 \
        --use_tile_transformer \
        --use_grade_prototypes \
        --use_ordinal_head \
        --use_concept_prototype \
        --lambda_gpa 0.1 \
        --lambda_proto_ce 1.0 \
        --lambda_tile_concept 0.5 \
        --proto_temperature 0.15 \
        --proto_label_smoothing 0.07
    ;;

*)
    echo "Unknown variant ${VARIANT} (array task ${SLURM_ARRAY_TASK_ID})" >&2
    exit 1
    ;;
esac
