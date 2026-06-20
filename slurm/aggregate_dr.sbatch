#!/bin/bash
#SBATCH --job-name=dr_aggr
#SBATCH --account=kuin0170
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:10:00
#SBATCH --output=/dpc/kuin0170/u100066980/runs/dr_gamma/slurm-aggregate-%j.out

# Combine the DR job-array per-fold results into final_results.csv.
# Submit with: --dependency=afterok:<dr_array_jobid>
set -e
source /home/kunet.ae/100066980/ESPACOL-New/slurm/_env.sh
cd "$REPO"
python slurm/aggregate_dr.py "$RUNS/dr_gamma"
