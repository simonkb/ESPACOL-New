#!/bin/bash
#SBATCH --job-name=espacol_nofaith
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=220G
#SBATCH --time=2-00:00:00
#SBATCH --array=0,1,2,3,4,5,8,13,17,24,28,32,36,42,45,47
#SBATCH --output=/dpc/kuin0170/ESPACOL-New/slurm_logs/nofaith_%A_%a.out
#SBATCH --error=/dpc/kuin0170/ESPACOL-New/slurm_logs/nofaith_%A_%a.err
#SBATCH --account=kuin0170

mkdir -p /dpc/kuin0170/ESPACOL-New/slurm_logs

source /etc/profile.d/lmod.sh
module load miniconda/3
module load cuda/12.6
source activate G

cd /dpc/kuin0170/ESPACOL-New

export HF_HUB_OFFLINE=1

python scripts/run_config.py $SLURM_ARRAY_TASK_ID
