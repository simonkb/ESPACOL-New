#!/bin/bash
#SBATCH --job-name=optic_concept_fold0_v4
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=220G
#SBATCH --time=2-00:00:00
#SBATCH --output=/dpc/kuin0170/ESPACOL-New/slurm_logs/optic_concept_fold0_v4_%j.out
#SBATCH --error=/dpc/kuin0170/ESPACOL-New/slurm_logs/optic_concept_fold0_v4_%j.err
#SBATCH --account=kuin0170

mkdir -p /dpc/kuin0170/ESPACOL-New/slurm_logs

source /etc/profile.d/lmod.sh
module load miniconda/3
module load cuda/12.6
source activate G

cd /dpc/kuin0170/ESPACOL-New

export HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# OPTIC-C v4: fix backbone LR starvation and tighten proto_CE floor.
#
# Root cause found in v3 numbers:
#
#   1. ReduceLROnPlateau fired at ep17 DURING freeze (ep1-25).
#      Val_loss couldn't improve because backbone was frozen — not a real plateau.
#      Result: backbone unfroze at ep26 with LR=4e-5 instead of intended 2e-4.
#      Fix (trainer.py): reset scheduler.best and num_bad_epochs at unfreeze so
#      freeze-phase history is discarded. lr_patience=15 now applies only to
#      post-unfreeze epochs, which is the correct behaviour.
#
#   2. proto_ce ended at 0.396; label smoothing floor (ε=0.1, 5 classes) ≈ 0.325.
#      Only 0.07 headroom left — model is constrained by the floor, not by capacity.
#      Fix: ε=0.07 drops floor to ~0.285, giving 0.11 more headroom for discrimination.
#      Collapse risk is manageable now that we understand the dynamics.
#
#   3. Augmentation was too conservative for fundus images (rotation ±10°, jitter 0.1).
#      Fundus has no natural orientation; lesion color varies across cameras.
#      Fix (in dataloaders.py): rotation 10°→45°, brightness/contrast 0.1→0.2, +saturation=0.1.
#
#   4. proto_temperature 0.2→0.15: with smoothing floor lowered, sharper boundaries affordable.
#      weight_decay 1e-6→1e-5: counterbalances higher backbone LR.
python train_dr.py \
    --dr_root Datasets/DR \
    --run_dir runs/optic_concept_fold0_v4 \
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
    --proto_temperature 0.15 \
    --proto_label_smoothing 0.07 \
    --lr_patience 15 \
    --early_stop_patience 30 \
    --weight_decay 1e-5
