#!/bin/bash
#SBATCH --job-name=concept_pointing
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=0-01:00:00
#SBATCH --output=/dpc/kuin0170/ESPACOL-New/slurm_logs/concept_pointing_%j.out
#SBATCH --error=/dpc/kuin0170/ESPACOL-New/slurm_logs/concept_pointing_%j.err
#SBATCH --account=kuin0170

mkdir -p /dpc/kuin0170/ESPACOL-New/slurm_logs

source /etc/profile.d/lmod.sh
module load miniconda/3
module load cuda/12.6
source activate G

cd /dpc/kuin0170/ESPACOL-New

export HF_HUB_OFFLINE=1

# ── Step 1: Run inference on all 81 IDRiD segmentation images ─────────────────
# Uses the v10 fold-1 checkpoint (best grading accuracy: 85.35%).
# If the outputs file already exists from a previous run, this step can be
# skipped by passing --seg_split all (it will overwrite with the same data).
#
# MODEL_DIR should point to the directory containing fold1_best.pth.
# Adjust the path below if the v10 checkpoint lives elsewhere.
MODEL_DIR=runs/optic_concept_cv_v10

python explainability/inference_idrid.py \
    --idrid_root  Datasets/IDRiD \
    --model_dir   ${MODEL_DIR} \
    --output_dir  explainability/idrid_outputs \
    --seg_split   all

echo ""
echo "========== Inference complete =========="
echo ""

# ── Step 2: Concept Pointing Game (quantitative) ──────────────────────────────
# Evaluates whether the max-scoring tile for each concept overlaps the IDRiD
# lesion mask. Reports per-concept and overall Pointing Game accuracy.
# Results also written to explainability/idrid_outputs/concept_pointing_results.json

python explainability/eval_concept_pointing.py \
    --npz         explainability/idrid_outputs/official_outputs.npz \
    --idrid_root  Datasets/IDRiD \
    --tile_grid   3 \
    --tile_size   300

echo ""
echo "========== Concept Pointing Game complete =========="
echo ""

# ── Step 3: Qualitative concept figure ────────────────────────────────────────
# Produces paper/optic_c_concept_qual.png:
#   4 rows (Grade 1–4) × 3 cols (original | concept heatmap | IDRiD mask)

python explainability/make_concept_figure.py \
    --npz         explainability/idrid_outputs/official_outputs.npz \
    --idrid_root  Datasets/IDRiD \
    --out         paper/optic_c_concept_qual.png \
    --tile_grid   3 \
    --tile_size   300 \
    --dpi         300

echo ""
echo "========== Concept figure saved to paper/optic_c_concept_qual.png =========="
