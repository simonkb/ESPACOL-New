#!/bin/bash
# Re-tag BUSI with the recovered fold and rebuild the verdict artifacts in place,
# from a finished run's per_image.csv (no model re-run). CPU/prod.
#
# Submit:  RUN=/dpc/.../runs/faithful_verdict_full sbatch slurm/retag_busi_rebuild.sbatch
#SBATCH -J espacol_retag_busi
#SBATCH -A kuin0170
#SBATCH -p prod
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH -t 00:20:00
#SBATCH -o /dpc/kuin0170/u100066980/runs/recover_busi/retag-%j.out

set -e
source /home/kunet.ae/100066980/ESPACOL-New/slurm/_env.sh
cd "$REPO"

RUN=${RUN:-/dpc/kuin0170/u100066980/runs/faithful_verdict_full}
python -m py_compile Hady/retag_busi_rebuild.py
echo "######## Re-tagging BUSI + rebuilding verdict in $RUN ########"
python Hady/retag_busi_rebuild.py --run "$RUN"
echo "######## DONE ########"
