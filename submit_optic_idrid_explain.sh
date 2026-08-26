#!/bin/bash
#SBATCH --job-name=optic_idrid_explain
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=0-02:00:00
#SBATCH --output=/dpc/kuin0170/ESPACOL-New/slurm_logs/optic_idrid_explain_%j.out
#SBATCH --error=/dpc/kuin0170/ESPACOL-New/slurm_logs/optic_idrid_explain_%j.err
#SBATCH --account=kuin0170

mkdir -p /dpc/kuin0170/ESPACOL-New/slurm_logs

source /etc/profile.d/lmod.sh
module load miniconda/3
module load cuda/12.6
source activate G

cd /dpc/kuin0170/ESPACOL-New

export HF_HUB_OFFLINE=1

MODEL_DIR=runs/optic_concept_idrid_v2/official
IDRID_ROOT=Datasets/IDRiD
OUTPUT_DIR=explainability/idrid_outputs
FIGURE_DIR=explainability/figures

# ── Step 1: Run inference on IDRiD segmentation TEST images (27 images) ──────
# Uses seg_split=test to avoid evaluating on images seen during training.
python explainability/inference_idrid.py \
    --idrid_root  $IDRID_ROOT \
    --model_dir   $MODEL_DIR \
    --output_dir  $OUTPUT_DIR \
    --seg_split   test

# ── Step 2: GPA Pointing Game ─────────────────────────────────────────────────
python explainability/gpa_pointing_game.py \
    --idrid_root    $IDRID_ROOT \
    --inference_dir $OUTPUT_DIR \
    --output_csv    explainability/pointing_game_results.csv

# ── Step 3: Tile Occlusion Faithfulness ───────────────────────────────────────
python explainability/tile_occlusion.py \
    --idrid_root    $IDRID_ROOT \
    --model_dir     $MODEL_DIR \
    --inference_dir $OUTPUT_DIR \
    --seg_split     test \
    --output_csv    explainability/tile_occlusion_results.csv

# ── Step 4: Concept Score Analysis ────────────────────────────────────────────
python explainability/concept_score_analysis.py \
    --inference_dir $OUTPUT_DIR \
    --output_csv    explainability/concept_scores_per_grade.csv

# ── Step 5: Qualitative figures (heatmaps + mask overlays) ───────────────────
python explainability/make_qual_figure.py \
    --idrid_root    $IDRID_ROOT \
    --inference_dir $OUTPUT_DIR \
    --output_dir    $FIGURE_DIR

echo "=== Explainability pipeline complete. ==="
echo "Figures: $FIGURE_DIR/optic_c_qualitative.pdf"
echo "Numbers: explainability/pointing_game_results.csv"
echo "         explainability/tile_occlusion_results.csv"
echo "         explainability/concept_scores_per_grade.csv"
