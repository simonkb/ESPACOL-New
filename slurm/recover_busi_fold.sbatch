#!/bin/bash
# Recover BUSI's training-time CV fold (broken by the storage-migration os.listdir
# reorder) from the pre-migration z_it cache. CPU is fine (~780 images) and the prod
# partition schedules instantly, so this doesn't compete with GPU jobs.
#
# Submit:  sbatch slurm/recover_busi_fold.sbatch
#SBATCH -J espacol_recover_busi
#SBATCH -A kuin0170
#SBATCH -p prod
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH -t 00:40:00
#SBATCH -o /dpc/kuin0170/u100066980/runs/recover_busi/slurm-%j.out

set -e
source /home/kunet.ae/100066980/ESPACOL-New/slurm/_env.sh
cd "$REPO"

mkdir -p /dpc/kuin0170/u100066980/runs/recover_busi

python -m py_compile Hady/recover_busi_fold.py
python -c "import torch; print('torch', torch.__version__, '| cuda avail:', torch.cuda.is_available())"

echo "######## Recovering BUSI fold4 held-out set ########"
# The pre-migration z_it cache lives in runs/explain (where recover_text_projection
# wrote it on Jun 21), not Hady/.
python Hady/recover_busi_fold.py --busi_root "$BUSI_ROOT" \
    --zit_cache "$RUNS/explain/z_it_busi.pt"

echo "######## DONE ########"
ls -la Hady/busi_fold4_heldout.json
