#!/bin/bash
# Recover the gamma text projections (Interpreter B) for BUSI + DR and render
# gamma-aligned explainability demo figures. Single GPU job, everything on HPC.
#
# Submit:  export SBATCH_EXPORT=NONE; sbatch slurm/recover_explain.sbatch
#SBATCH -J espacol_explain
#SBATCH -A kuin0170
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=48G
#SBATCH -t 04:00:00
#SBATCH -o /dpc/kuin0170/u100066980/runs/explain/slurm-%j.out

set -e
source /home/kunet.ae/100066980/ESPACOL-New/slurm/_env.sh
cd "$REPO"

nvidia-smi || true

OUT=/dpc/kuin0170/u100066980/runs/explain
mkdir -p "$OUT"

# ── CUDA fail-fast (nodes have driver 560 = CUDA 12.6; torch must be cu126) ──
python -c "import torch; assert torch.cuda.is_available(), 'NO CUDA — wrong torch build?'; print('cuda build:', torch.version.cuda)"

echo "######## Recover BUSI text projection ########"
python Hady/recover_text_projection.py --dataset busi \
    --ckpt Hady/BUSI_best_fold4.pth --busi_root "$BUSI_ROOT" \
    --out Hady/BUSI_text_projection.pth --zit_cache "$OUT/z_it_busi.pt"

echo "######## Recover DR text projection (best-acc fold2) ########"
python Hady/recover_text_projection.py --dataset dr \
    --ckpt Hady/DR_best_acc_fold2.pth --dr_root "$DR_ROOT" \
    --out Hady/DR_text_projection.pth --zit_cache "$OUT/z_it_dr.pt"

# ── Demo figures (headless PNGs into $OUT) ──────────────────────────────────
echo "######## BUSI demo figures ########"
for s in 0 1 2; do
  python Hady/demo_explain_busi.py --seed "$s" --busi_root "$BUSI_ROOT" \
      --save "$OUT/busi_explain_seed${s}.png" --no-show
done
# Old-vs-new comparison: force the random-orthogonal bridge (no recovered proj).
python Hady/demo_explain_busi.py --seed 1 --busi_root "$BUSI_ROOT" \
    --text_projection /nonexistent_force_fallback \
    --save "$OUT/busi_explain_seed1_OLD_bridge.png" --no-show

echo "######## DR demo figures (one per severity grade) ########"
for g in 0 2 3 4; do
  python Hady/demo_explain_dr.py --grade "$g" --seed 1 --dr_root "$DR_ROOT" \
      --save "$OUT/dr_explain_grade${g}.png" --no-show
done
python Hady/demo_explain_dr.py --grade 4 --seed 1 --dr_root "$DR_ROOT" \
    --text_projection /nonexistent_force_fallback \
    --save "$OUT/dr_explain_grade4_OLD_bridge.png" --no-show

echo "######## DONE — outputs in $OUT ########"
ls -la "$OUT"
