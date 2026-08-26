#!/bin/bash
#SBATCH --job-name=qual_figure
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=0-00:30:00
#SBATCH --output=/dpc/kuin0170/ESPACOL-New/slurm_logs/qual_figure_%j.out
#SBATCH --error=/dpc/kuin0170/ESPACOL-New/slurm_logs/qual_figure_%j.err
#SBATCH --account=kuin0170

mkdir -p /dpc/kuin0170/ESPACOL-New/slurm_logs

source /etc/profile.d/lmod.sh
module load miniconda/3
module load cuda/12.6
source activate G

cd /dpc/kuin0170/ESPACOL-New

python explainability/make_qual_figure.py \
    --idrid_root    Datasets/IDRiD \
    --inference_dir explainability/idrid_outputs \
    --output_dir    explainability/figures
