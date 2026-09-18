#!/bin/bash
# No-training, inner-validation-only amplitude/direction audit for the selected
# PATHS-v3 SAPT APTOS checkpoint.
#SBATCH --job-name=paths_audit
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/dpc/kuin0170/ESPACOL-New/paths_transport_audit_%j.out
#SBATCH --error=/dpc/kuin0170/ESPACOL-New/paths_transport_audit_%j.err
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

for AUDIT_FILE in \
  audit_paths_transport.py models/origin_encoder.py models/origin.py models/paths.py \
  Datasets/origin_data.py Datasets/mosaic_data.py Datasets/dataloaders.py \
  training/origin_trainer.py training/paths_trainer.py train_origin.py \
  utils/spatial_mask.py \
  tests/test_paths_transport_audit.py scripts/submit_paths_transport_audit.sh; do
  [[ "$(git rev-parse "HEAD:${AUDIT_FILE}")" == "$(git hash-object "${AUDIT_FILE}")" ]] || {
    echo "Tracked audit implementation differs from HEAD: ${AUDIT_FILE}" >&2
    exit 2
  }
done

TREATMENT_DIR="${PATHS_APTOS_FOLD_DIR:-runs/paths_aptos_f0_v3_sapt_9a860c8/fold0}"
CONTROL_DIR="${PATHS_APTOS_CONTROL_FOLD_DIR:-runs/paths_aptos_f0_v3_risk_control_9a860c8/fold0}"
DATA_ROOT="${ORIGIN_APTOS_ROOT:-Datasets/aptos2019-blindness-detection}"
OUTPUT_DIR="${PATHS_TRANSPORT_AUDIT_DIR:-${TREATMENT_DIR}/audits/transport_margin_v1}"
for RUN_FILE in \
  "${TREATMENT_DIR}/best.pth" "${TREATMENT_DIR}/result.json" \
  "${TREATMENT_DIR}/v3_strength_zero_control.json" \
  "${CONTROL_DIR}/best.pth" "${CONTROL_DIR}/result.json" \
  "${CONTROL_DIR}/v3_strength_zero_control.json"; do
  [[ -f "${RUN_FILE}" ]] || { echo "Missing ${RUN_FILE}" >&2; exit 3; }
done
[[ ! -e "${OUTPUT_DIR}/summary.json" && ! -e "${OUTPUT_DIR}/per_sample.csv" ]] || {
  echo "Audit output already exists in ${OUTPUT_DIR}; choose a new directory." >&2
  exit 4
}

echo "=== PATHS-v3 SAPT no-retraining margin/flow audit ==="
date --iso-8601=seconds
git rev-parse HEAD
python --version
echo "treatment_fold_dir=${TREATMENT_DIR}"
echo "control_fold_dir=${CONTROL_DIR}"
echo "output_dir=${OUTPUT_DIR}"

python -m pytest -q tests/test_paths_transport_audit.py

python audit_paths_transport.py \
  --treatment_fold_dir "${TREATMENT_DIR}" \
  --control_fold_dir "${CONTROL_DIR}" \
  --data_root "${DATA_ROOT}" \
  --output_dir "${OUTPUT_DIR}" \
  --batch_size 8 \
  --num_workers 8 \
  --device cuda \
  --multiplier_grid 0,0.25,0.5,1,2,4,8,16,32,64

echo "PATHS transport audit completed: ${OUTPUT_DIR}/summary.json"
