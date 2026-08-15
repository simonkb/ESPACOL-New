#!/bin/bash
#SBATCH --job-name=faith_on
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=220G
#SBATCH --time=2-00:00:00
#SBATCH --output=/dpc/kuin0170/ESPACOL-New/slurm_logs/faith_on_%j.out
#SBATCH --error=/dpc/kuin0170/ESPACOL-New/slurm_logs/faith_on_%j.err
#SBATCH --account=kuin0170

# ─────────────────────────────────────────────────────────────────────────────
# FAITHFULNESS ON — DR fold 0
#
# The faithfulness arm of the paired experiment. Identical to submit_faith_off.sh
# in every respect except nu (the L_faith weight), which is left at the DRConfig
# default (peak 0.1, ramped 0 -> 0.1 across epochs 25-35).
#
# PCOL (alpha), SCOLw (beta) and the image-text loss (gamma) are all zeroed so
# L_faith is not competing with the losses that previously dominated the
# gradient. The text encoder still loads despite gamma=0 — the concept spine
# needs it to encode concept phrases — but L_IT contributes no gradient, and
# --no_finetune_text_encoder keeps it frozen for the whole run so ~15M params do
# not unfreeze at epoch 20, five epochs before the L_faith ramp starts at 25.
#
# Watch in train.log:  drop / dir / spec / occ / mask
#   occ ~ 0 with a healthy mask  => occlusion lands but concepts don't respond
#   mask ~ 0                     => CAMs too diffuse for the 0.5 threshold;
#                                   rerun with --faith_threshold 0.3
# ─────────────────────────────────────────────────────────────────────────────

mkdir -p /dpc/kuin0170/ESPACOL-New/slurm_logs

source /etc/profile.d/lmod.sh
module load miniconda/3
module load cuda/12.6

# ── FILL IN: conda environment name or full path ─────────────────────────────
CONDA_ENV=G                        # <-- EDIT ME before submitting
source activate "$CONDA_ENV"

cd /dpc/kuin0170/ESPACOL-New

export HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

RUN_DIR=runs/faith_on

# Never reuse a run_dir — a resubmit into an existing one would overwrite
# train.log and the fold checkpoints, silently mixing two runs together.
if [ -e "$RUN_DIR" ]; then
    echo "ERROR: $RUN_DIR already exists. Use a fresh run_dir (e.g. ${RUN_DIR}_v2)." >&2
    exit 1
fi

python train_dr.py \
    --dr_root Datasets/DR \
    --run_dir "$RUN_DIR" \
    --folds 0 \
    --epochs 75 \
    --use_multi_tile \
    --use_concept_spine \
    --batch_size 24 \
    --grad_checkpoint \
    --alpha 0 \
    --beta 0 \
    --gamma 0 \
    --no_finetune_text_encoder
