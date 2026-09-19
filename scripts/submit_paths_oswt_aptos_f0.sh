#!/bin/bash
# Registered risk-gated OSWT treatment on APTOS fold 0, validation only.
#SBATCH --job-name=oswt_a0
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=/dpc/kuin0170/ESPACOL-New/paths_oswt_aptos_f0_%j.out
#SBATCH --error=/dpc/kuin0170/ESPACOL-New/paths_oswt_aptos_f0_%j.err
#SBATCH --account=kuin0170

set -eo pipefail
set +u
source /etc/profile.d/lmod.sh || exit 1
module load miniconda/3 || exit 1
module load cuda/12.6 || exit 1
source activate "${ORIGIN_CONDA_ENV:-G}" || exit 1
set -u

REPO_ROOT="${ORIGIN_REPO_ROOT:-/dpc/kuin0170/ESPACOL-New}"
cd "${REPO_ROOT}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
for path in configs/paths_oswt_config.py configs/paths_config.py configs/origin_config.py Datasets/origin_data.py Datasets/mosaic_data.py Datasets/dataloaders.py models/origin_encoder.py models/origin.py models/paths.py models/paths_oswt.py losses/origin.py losses/paths.py training/origin_trainer.py training/paths_trainer.py training/paths_oswt_trainer.py train_paths_oswt.py train_origin.py utils/spatial_mask.py scripts/submit_paths_oswt_preflight.sh scripts/submit_paths_oswt_aptos_f0.sh scripts/submit_paths_oswt_aptos_ungated_f0.sh scripts/submit_paths_oswt_promotion_gate.sh scripts/submit_paths_oswt_dr_f0.sh scripts/launch_paths_oswt_v8_aptos_f0.sh; do
  [[ "$(git rev-parse "HEAD:${path}")" == "$(git hash-object "${path}")" ]] || {
    echo "Tracked implementation differs from HEAD: ${path}" >&2; exit 2;
  }
done
: "${PATHS_OSWT_APTOS_V3_SHA256:?export the audited APTOS V3 SHA-256}"
V3="${PATHS_OSWT_APTOS_V3_CHECKPOINT:-runs/origin_aptos_f0_v3_bounded/fold0/best.pth}"
[[ "$(sha256sum "${V3}" | awk '{print $1}')" == "${PATHS_OSWT_APTOS_V3_SHA256}" ]] || {
  echo "APTOS V3 SHA-256 mismatch." >&2; exit 2;
}

echo "=== PATHS-V8 risk-gated OSWT APTOS fold 0 / inner validation only ==="
date --iso-8601=seconds
git rev-parse HEAD
git status --short
python --version

RESUME_ARGS=()
if [[ "${PATHS_OSWT_RESUME:-0}" == "1" ]]; then
  RESUME_ARGS+=(--resume)
fi

PATHS_OSWT_RUN_GIT_COMMIT="$(git rev-parse HEAD)" python train_paths_oswt.py \
  --dataset aptos \
  --data_root "${PATHS_OSWT_APTOS_DATA_ROOT:-${REPO_ROOT}/Datasets/aptos2019-blindness-detection}" \
  --run_dir "${PATHS_OSWT_APTOS_RUN_DIR:-runs/paths_oswt_aptos_f0_v8_gated}" \
  --folds 0 --seed 42 \
  --v3_checkpoint "${V3}" --v3_sha256 "${PATHS_OSWT_APTOS_V3_SHA256}" \
  --oswt_variant shell_warranted \
  --image_size 640 --encoder convnext_tiny --scales s4,s8,s16,s32 \
  --projection_dim 128 --reference_count 4096 \
  --atom_rate_init 1e-6 --prior_rate_init 1e-4 --boundary_scale_init 1.0 \
  --total_rate_cap 64 --prior_rate_cap 1 --boundary_scale_cap 2 \
  --rate_roundoff_margin 1 --decision_rule class_map \
  --pgf_probes 0.05,0.20,0.50,0.80 \
  --oswt_beta_init 0.1 --oswt_tau_init 0.05 \
  --oswt_beta_cap 3.0 --oswt_tau_floor 1e-3 --oswt_tau_cap 0.5 \
  --oswt_strength 1.0 \
  --risk_set_alpha 0.5 --rps_weight 0.25 \
  --batch_size 8 --epochs 35 --num_workers 8 \
  --paths_refiner_lr 5e-4 --weight_decay 0 \
  --lr_factor 0.2 --lr_patience 5 --early_stopping_patience 12 \
  --certificate_samples 8 --certificate_shortlist_size 8 \
  "${RESUME_ARGS[@]}" --skip_test
