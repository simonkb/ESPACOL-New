#!/bin/bash
# R-vs-V comparison only (no GPU, no checkpoint needed).
# For the full eval with a checkpoint, pass it as the first argument:
#   sbatch submit_sanity_check.sh /path/to/fold0_best.pth
#
#SBATCH --job-name=tscgp_sanity
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --mem=8G
#SBATCH --time=0-00:30:00
#SBATCH --output=/dpc/kuin0170/ESPACOL-New/slurm_logs/sanity_check_%j.out
#SBATCH --error=/dpc/kuin0170/ESPACOL-New/slurm_logs/sanity_check_%j.err
#SBATCH --account=kuin0170

mkdir -p /dpc/kuin0170/ESPACOL-New/slurm_logs

source /etc/profile.d/lmod.sh
module load miniconda/3
source activate G

cd /dpc/kuin0170/ESPACOL-New

export HF_HUB_OFFLINE=1

CHECKPOINT="${1}"

if [ -z "${CHECKPOINT}" ]; then
    echo "=== R vs V comparison only (no checkpoint) ==="
    python sanity_check_tscgp.py
else
    echo "=== Full eval with checkpoint: ${CHECKPOINT} ==="
    python sanity_check_tscgp.py \
        --checkpoint "${CHECKPOINT}" \
        --dataset    dr \
        --data_root  Datasets/DR \
        --fold       0
fi
