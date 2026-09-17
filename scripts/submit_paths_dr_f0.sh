#!/bin/bash
# PATHS-v3 SAPT EyePACS pilot. This script refuses to run before the paired
# APTOS treatment/control gate passes under the same implementation commit.
#SBATCH --job-name=paths_dr0
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --output=/dpc/kuin0170/ESPACOL-New/paths_dr_f0_%j.out
#SBATCH --error=/dpc/kuin0170/ESPACOL-New/paths_dr_f0_%j.err
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
export PATHS_RUN_GIT_COMMIT="$(git rev-parse HEAD)"

for PATHS_FILE in \
  configs/paths_config.py configs/origin_config.py \
  Datasets/origin_data.py Datasets/mosaic_data.py Datasets/dataloaders.py \
  models/origin_encoder.py models/origin.py models/paths.py \
  losses/origin.py losses/paths.py \
  training/origin_trainer.py training/paths_trainer.py \
  train_paths.py utils/spatial_mask.py \
  scripts/submit_paths_preflight.sh scripts/submit_paths_aptos_f0.sh \
  scripts/submit_paths_aptos_control_f0.sh scripts/submit_paths_aptos_gate.sh \
  scripts/submit_paths_dr_f0.sh; do
  [[ "$(git rev-parse "HEAD:${PATHS_FILE}")" == "$(git hash-object "${PATHS_FILE}")" ]] || {
    echo "Tracked implementation differs from HEAD: ${PATHS_FILE}" >&2
    exit 2
  }
done

APTOS_GATE="${PATHS_APTOS_GATE:-runs/paths_aptos_f0_v3_sapt/fold0/PROMOTED_TO_EYEPACS.json}"
[[ -f "${APTOS_GATE}" ]] || {
  echo "PATHS EyePACS is blocked until the APTOS promotion gate exists: ${APTOS_GATE}" >&2
  exit 3
}
PATHS_GATE="${APTOS_GATE}" PATHS_HEAD="$(git rev-parse HEAD)" python - <<'PY'
import json, os, pathlib
from configs.paths_config import PATHS_PROTOCOL_VERSION
from training.paths_trainer import paths_implementation_signature
gate = json.loads(pathlib.Path(os.environ["PATHS_GATE"]).read_text())
valid = (
    gate.get("schema") == "paths-v3-sapt-aptos-paired-gate-v1"
    and gate.get("passed") is True
    and all(gate.get("checks", {}).values())
    and gate.get("git_commit") == os.environ["PATHS_HEAD"]
    and gate.get("protocol") == PATHS_PROTOCOL_VERSION
    and gate.get("implementation_signature") == paths_implementation_signature()
)
if not valid:
    raise SystemExit("APTOS promotion gate did not pass")
print("aptos_gate", json.dumps(gate["checks"], sort_keys=True))
PY

RUN_DIR="${PATHS_DR_RUN_DIR:-runs/paths_dr_f0_v3_sapt}"
DATA_ROOT="${ORIGIN_DR_ROOT:-Datasets/DR}"
SOURCE_CHECKPOINT="${PATHS_V3_DR_CHECKPOINT:-runs/origin_dr_f0_v3_bounded/fold0/best.pth}"
SOURCE_SHA="${PATHS_V3_DR_SHA256:-}"
[[ -f "${SOURCE_CHECKPOINT}" ]] || { echo "Missing ${SOURCE_CHECKPOINT}" >&2; exit 3; }
[[ "${SOURCE_SHA}" =~ ^[0-9a-fA-F]{64}$ ]] || {
  echo "Set PATHS_V3_DR_SHA256 to the audited V3 checkpoint SHA-256." >&2
  exit 3
}
FOLD_DIR="${RUN_DIR}/fold0"
mkdir -p "${FOLD_DIR}"
exec 9>"${FOLD_DIR}/.writer.lock"
flock -n 9 || { echo "Another process is writing ${FOLD_DIR}." >&2; exit 4; }
RESUME_ARGS=()
if [[ "${PATHS_RESUME:-0}" == "1" ]]; then
  [[ -f "${FOLD_DIR}/last.pth" && ! -f "${FOLD_DIR}/result.json" ]] || exit 5
  RESUME_ARGS+=(--resume)
elif [[ -e "${FOLD_DIR}/last.pth" || -e "${FOLD_DIR}/history.csv" ]]; then
  echo "Fresh PATHS run refused: artifacts exist in ${FOLD_DIR}." >&2
  exit 6
fi

echo "=== PATHS-v3 SAPT EyePACS fold 0 / inner validation only ==="
date --iso-8601=seconds
git rev-parse HEAD
git status --short
python --version

ORIGIN_DATA_ROOT="${DATA_ROOT}" python - <<'PY'
import os
from Datasets.origin_data import class_histogram, eyepacs_fold, load_eyepacs_items, split_paths_are_disjoint
items = load_eyepacs_items(os.environ["ORIGIN_DATA_ROOT"])
train, validation, test = eyepacs_fold(items, 0, n_folds=10, seed=42)
assert len(items) == 35126 and (len(train), len(validation), len(test)) == (28446, 3162, 3518)
assert class_histogram(train) == [20918, 1979, 4271, 702, 576]
assert class_histogram(validation) == [2318, 216, 481, 83, 64]
assert split_paths_are_disjoint(train, validation, test)
PY

ARGS=(
  --dataset dr --data_root "${DATA_ROOT}" --run_dir "${RUN_DIR}" --folds 0 --seed 42
  --v3_checkpoint "${SOURCE_CHECKPOINT}" --v3_sha256 "${SOURCE_SHA,,}"
  --image_size 640 --encoder convnext_tiny --scales s4,s8,s16,s32
  --projection_dim 128 --reference_count 4096 --atom_rate_init 1e-6
  --prior_rate_init 1e-4 --boundary_scale_init 1.0 --total_rate_cap 64.0
  --prior_rate_cap 1.0 --boundary_scale_cap 2.0 --rate_roundoff_margin 1.0
  --decision_rule class_map --paths_variant signed_transport
  --pgf_probes 0.05,0.20,0.50,0.80
  --transport_gain_cap 1.0 --transport_gain_init 0.05
  --transport_threshold_init 0.5 --transport_slope_init 2.0
  --transport_slope_cap 8.0 --transport_strength 1.0
  --risk_set_alpha 0.5 --rps_weight 0.25 --batch_size 8 --epochs 25
  --num_workers 8 --paths_encoder_lr 1e-5 --paths_base_lr 1e-5
  --paths_refiner_lr 5e-4 --weight_decay 1e-5
  --correction_only_epochs 0 --freeze_encoder_epochs 25
  --lr_factor 0.2 --lr_patience 4 --early_stopping_patience 8
  --amp_init_scale 256 --amp_unfreeze_scale 256 --skip_test
)
if (( ${#RESUME_ARGS[@]} )); then ARGS+=("${RESUME_ARGS[@]}"); fi
printf 'training_command:'; printf ' %q' python train_paths.py "${ARGS[@]}"; printf '\n'
python train_paths.py "${ARGS[@]}"
