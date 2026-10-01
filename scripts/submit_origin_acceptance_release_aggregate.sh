#!/bin/bash
#SBATCH --job-name=oa_aggregate
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=/dpc/kuin0170/ESPACOL-New/origin_acceptance_logs/aggregate_%j.out
#SBATCH --error=/dpc/kuin0170/ESPACOL-New/origin_acceptance_logs/aggregate_%j.err
#SBATCH --account=kuin0170

set -euo pipefail
set +u
source /etc/profile.d/lmod.sh
module load miniconda/3
source activate "${ORIGIN_CONDA_ENV:-espacol}"
set -u
cd "${ORIGIN_REPO_ROOT:?ORIGIN_REPO_ROOT is required}"
[[ "$(git rev-parse HEAD)" == "${ORIGIN_LAUNCH_COMMIT:?ORIGIN_LAUNCH_COMMIT is required}" ]] || exit 2
python scripts/aggregate_origin_acceptance_release.py \
  --experiment_root "${ORIGIN_ACCEPTANCE_ROOT:?ORIGIN_ACCEPTANCE_ROOT is required}" \
  --bootstrap_samples 10000 --bootstrap_seed 20261001
