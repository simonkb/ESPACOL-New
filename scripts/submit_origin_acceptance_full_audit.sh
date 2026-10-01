#!/bin/bash
#SBATCH --job-name=oa_full_audit
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=/dpc/kuin0170/ESPACOL-New/origin_acceptance_logs/full_audit_%j.out
#SBATCH --error=/dpc/kuin0170/ESPACOL-New/origin_acceptance_logs/full_audit_%j.err
#SBATCH --account=kuin0170

set -euo pipefail
set +u
source /etc/profile.d/lmod.sh
module load miniconda/3
source activate "${ORIGIN_CONDA_ENV:-espacol}"
set -u
REPO_ROOT="${ORIGIN_REPO_ROOT:?ORIGIN_REPO_ROOT is required}"
cd "${REPO_ROOT}"
[[ "$(git rev-parse HEAD)" == "${ORIGIN_LAUNCH_COMMIT:?ORIGIN_LAUNCH_COMMIT is required}" ]] || exit 2
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
python -m scripts.audit_origin_acceptance_baselines \
  --experiment_root "${ORIGIN_ACCEPTANCE_ROOT:?ORIGIN_ACCEPTANCE_ROOT is required}" \
  --scope full --write_gate
