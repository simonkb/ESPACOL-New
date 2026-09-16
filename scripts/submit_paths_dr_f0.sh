#!/bin/bash
# PATHS EyePACS pilot. This script refuses to run before the APTOS gate passes.
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

APTOS_GATE="${PATHS_APTOS_GATE:-runs/paths_aptos_f0_v2/fold0/PROMOTED_TO_EYEPACS.json}"
[[ -f "${APTOS_GATE}" ]] || {
  echo "PATHS EyePACS is blocked until the APTOS promotion gate exists: ${APTOS_GATE}" >&2
  exit 2
}
PATHS_GATE="${APTOS_GATE}" python - <<'PY'
import json, os, pathlib
gate = json.loads(pathlib.Path(os.environ["PATHS_GATE"]).read_text())
if gate.get("passed") is not True or not all(gate.get("checks", {}).values()):
    raise SystemExit("APTOS promotion gate did not pass")
print("aptos_gate", json.dumps(gate["checks"], sort_keys=True))
PY

RUN_DIR="${PATHS_DR_RUN_DIR:-runs/paths_dr_f0_v2}"
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

echo "=== PATHS EyePACS fold 0 / inner validation only ==="
date --iso-8601=seconds
git rev-parse HEAD
git status --short
python --version

ARGS=(
  --dataset dr --data_root "${DATA_ROOT}" --run_dir "${RUN_DIR}" --folds 0 --seed 42
  --v3_checkpoint "${SOURCE_CHECKPOINT}" --v3_sha256 "${SOURCE_SHA,,}"
  --image_size 640 --encoder convnext_tiny --scales s4,s8,s16,s32
  --projection_dim 128 --reference_count 4096 --atom_rate_init 1e-6
  --prior_rate_init 1e-4 --boundary_scale_init 1.0 --total_rate_cap 64.0
  --prior_rate_cap 1.0 --boundary_scale_cap 2.0 --rate_roundoff_margin 1.0
  --decision_rule class_map --pgf_probes 0.05,0.20,0.50,0.80
  --correction_cap 3.0 --correction_gain_init 0.05 --correction_strength 1.0
  --risk_set_alpha 0.5 --rps_weight 0.25 --batch_size 8 --epochs 75
  --num_workers 8 --paths_encoder_lr 1e-5 --paths_base_lr 5e-5
  --paths_refiner_lr 5e-4 --weight_decay 1e-5
  --correction_only_epochs 3 --freeze_encoder_epochs 3
  --lr_factor 0.2 --lr_patience 5 --early_stopping_patience 15
  --amp_unfreeze_scale 256 --skip_test
)
if (( ${#RESUME_ARGS[@]} )); then ARGS+=("${RESUME_ARGS[@]}"); fi
printf 'training_command:'; printf ' %q' python train_paths.py "${ARGS[@]}"; printf '\n'
python train_paths.py "${ARGS[@]}"
