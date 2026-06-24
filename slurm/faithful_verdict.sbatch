#!/bin/bash
# Faithfulness VERDICT run — scaled, controlled, statistically-tested.
# Runs Hady/eval_faithfulness_verdict.py over BUSI (3 classes) + DR (5 grades),
# up to --per-class images per group, with the on-vs-off-target specificity control
# on every image, then writes per-image/per-group CSVs, a markdown verdict report,
# and distribution plots. Single GPU job, everything on HPC (login = submit only).
#
# Submit:  export SBATCH_EXPORT=NONE; sbatch slurm/faithful_verdict.sbatch
#SBATCH -J espacol_faith_verdict
#SBATCH -A kuin0170
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=48G
#SBATCH -t 10:00:00
#SBATCH -o /dpc/kuin0170/u100066980/runs/faithful_verdict_full/slurm-%j.out

set -e
source /home/kunet.ae/100066980/ESPACOL-New/slurm/_env.sh
cd "$REPO"

nvidia-smi || true

# Fail fast on a Python syntax error before doing any heavy work (cheap, ~1s).
python -m py_compile Hady/eval_faithfulness_verdict.py

OUT=${OUT:-/dpc/kuin0170/u100066980/runs/faithful_verdict_full}
mkdir -p "$OUT"

# ── CUDA fail-fast (nodes have driver 560 = CUDA 12.6; torch must be cu126) ──
python -c "import torch; assert torch.cuda.is_available(), 'NO CUDA — wrong torch build?'; print('cuda build:', torch.version.cuda)"

# Per-group sample size. 0 (or -1) ⇒ the FULL dataset, no sampling: every BUSI
# image (780) + every DR image (35,126) = ~35,906 across 8 groups (~3.5–5 h on one
# V100, 9 forward passes/image). Each image is tagged train vs held-out for its
# checkpoint's CV fold so the verdict is reported overall / train / held-out.
# Set PER_CLASS=20 for a quick structural smoke test, or PER_CLASS=100 for the
# old balanced sample. DATASETS lets a smoke run hit one dataset (e.g. busi).
PER_CLASS=${PER_CLASS:-0}
DATASETS=${DATASETS:-both}

echo "######## Faithfulness verdict run (datasets=$DATASETS, per-class=$PER_CLASS) ########"
python Hady/eval_faithfulness_verdict.py \
    --datasets "$DATASETS" --per-class "$PER_CLASS" --controls 4 --seed 0 \
    --occlusion blur --blur-sigma 21 --encoder biomedclip \
    --busi_root "$BUSI_ROOT" --dr_root "$DR_ROOT" \
    --out "$OUT"

echo "######## DONE — outputs in $OUT ########"
ls -la "$OUT"
