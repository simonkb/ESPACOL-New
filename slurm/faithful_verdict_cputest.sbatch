#!/bin/bash
# CPU-only validation of the full-dataset verdict CODE path while the GPU job
# queues. Tests: env activation on the migrated (H200) cluster, torch import,
# held-out fold reconstruction (BUSI + DR validators), per-image train/heldout
# tagging, 3-way partitioned stats + per-partition BH-FDR, report/plots, and the
# self-check. NOT a GPU/H200 test and NOT a meaningful verdict — a tiny per-class
# on CPU just to catch runtime bugs in the new code before the long GPU run.
#
# Submit:  sbatch slurm/faithful_verdict_cputest.sbatch
#SBATCH -J espacol_faith_cputest
#SBATCH -A kuin0170
#SBATCH -p prod
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH -t 00:40:00
#SBATCH -o /dpc/kuin0170/u100066980/runs/faithful_verdict_cputest/slurm-%j.out

set -e
source /home/kunet.ae/100066980/ESPACOL-New/slurm/_env.sh
cd "$REPO"

OUT=${OUT:-/dpc/kuin0170/u100066980/runs/faithful_verdict_cputest}
mkdir -p "$OUT"

# Fail fast on a Python syntax error (cheap, ~1s).
python -m py_compile Hady/eval_faithfulness_verdict.py

# CPU node: report torch/cuda status but do NOT assert CUDA (there is no GPU here).
python -c "import torch; print('torch', torch.__version__, '| cuda avail:', torch.cuda.is_available())"

PER_CLASS=${PER_CLASS:-5}
DATASETS=${DATASETS:-both}
echo "######## CPU code-path test (datasets=$DATASETS, per-class=$PER_CLASS) ########"
python Hady/eval_faithfulness_verdict.py \
    --datasets "$DATASETS" --per-class "$PER_CLASS" --controls 4 --seed 0 \
    --occlusion blur --blur-sigma 21 --encoder biomedclip \
    --busi_root "$BUSI_ROOT" --dr_root "$DR_ROOT" \
    --out "$OUT"

echo "######## DONE — outputs in $OUT ########"
ls -la "$OUT"
