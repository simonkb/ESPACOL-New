#!/bin/bash
#SBATCH --job-name=faith_off
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=220G
#SBATCH --time=2-00:00:00
#SBATCH --output=/dpc/kuin0170/ESPACOL-New/slurm_logs/faith_off_%j.out
#SBATCH --error=/dpc/kuin0170/ESPACOL-New/slurm_logs/faith_off_%j.err
#SBATCH --account=kuin0170

# ─────────────────────────────────────────────────────────────────────────────
# FAITHFULNESS OFF — DR fold 0  (control arm)
#
# Byte-for-byte identical to submit_faith_on.sh except for `--nu 0`, which
# disables the faithfulness loop entirely: with nu_peak = 0 the trainer never
# enters the occlusion branch, so no CAM pass, no second forward, no L_faith.
#
# The concept spine itself stays ON (L_PIC + L_cons still train), so the
# difference between this run and faith_on isolates L_faith alone rather than
# the whole spine.
#
# drop / dir / spec / occ / mask will all log as 0.0000 here — expected, since
# they are averaged over faithfulness batches and there are none.
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

RUN_DIR=runs/faith_off

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
    --no_finetune_text_encoder \
    --nu 0
