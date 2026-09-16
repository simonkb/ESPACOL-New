#!/bin/bash
# PATHS first gate: APTOS fold 0, inner validation only.
#SBATCH --job-name=paths_a0
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH --output=/dpc/kuin0170/ESPACOL-New/paths_aptos_f0_%j.out
#SBATCH --error=/dpc/kuin0170/ESPACOL-New/paths_aptos_f0_%j.err
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

for PATHS_FILE in \
  configs/paths_config.py configs/origin_config.py \
  Datasets/origin_data.py Datasets/mosaic_data.py Datasets/dataloaders.py \
  models/origin_encoder.py models/origin.py models/paths.py \
  losses/origin.py losses/paths.py \
  training/origin_trainer.py training/paths_trainer.py \
  train_paths.py utils/spatial_mask.py; do
  [[ "$(git rev-parse "HEAD:${PATHS_FILE}")" == "$(git hash-object "${PATHS_FILE}")" ]] || {
    echo "Tracked implementation differs from HEAD: ${PATHS_FILE}" >&2
    exit 2
  }
done

RUN_DIR="${PATHS_APTOS_RUN_DIR:-runs/paths_aptos_f0_v2}"
DATA_ROOT="${ORIGIN_APTOS_ROOT:-Datasets/aptos2019-blindness-detection}"
SOURCE_CHECKPOINT="${PATHS_V3_APTOS_CHECKPOINT:-runs/origin_aptos_f0_v3_bounded/fold0/best.pth}"
SOURCE_SHA="${PATHS_V3_APTOS_SHA256:-}"
[[ -f "${SOURCE_CHECKPOINT}" ]] || { echo "Missing ${SOURCE_CHECKPOINT}" >&2; exit 3; }
[[ "${SOURCE_SHA}" =~ ^[0-9a-fA-F]{64}$ ]] || {
  echo "Set PATHS_V3_APTOS_SHA256 to the audited V3 checkpoint SHA-256." >&2
  exit 3
}

FOLD_DIR="${RUN_DIR}/fold0"
mkdir -p "${FOLD_DIR}"
exec 9>"${FOLD_DIR}/.writer.lock"
flock -n 9 || { echo "Another process is writing ${FOLD_DIR}." >&2; exit 3; }
RESUME_ARGS=()
if [[ "${PATHS_RESUME:-0}" == "1" ]]; then
  [[ -f "${FOLD_DIR}/last.pth" && ! -f "${FOLD_DIR}/result.json" ]] || exit 4
  RESUME_ARGS+=(--resume)
elif [[ -e "${FOLD_DIR}/last.pth" || -e "${FOLD_DIR}/history.csv" ]]; then
  echo "Fresh PATHS run refused: artifacts exist in ${FOLD_DIR}." >&2
  exit 5
fi

echo "=== PATHS APTOS fold 0 / inner validation only ==="
date --iso-8601=seconds
git rev-parse HEAD
git status --short
python --version

ORIGIN_DATA_ROOT="${DATA_ROOT}" python - <<'PY'
import os
from Datasets.origin_data import aptos_fold, class_histogram, load_aptos_items, split_paths_are_disjoint
items = load_aptos_items(os.environ["ORIGIN_DATA_ROOT"])
train, validation, test = aptos_fold(items, 0, n_folds=5, seed=42)
assert len(items) == 3662 and (len(train), len(validation), len(test)) == (2636, 293, 733)
assert class_histogram(train) == [1300, 266, 719, 139, 212]
assert class_histogram(validation) == [144, 30, 80, 15, 24]
assert split_paths_are_disjoint(train, validation, test)
PY

ARGS=(
  --dataset aptos --data_root "${DATA_ROOT}" --run_dir "${RUN_DIR}" --folds 0 --seed 42
  --v3_checkpoint "${SOURCE_CHECKPOINT}" --v3_sha256 "${SOURCE_SHA,,}"
  --image_size 640 --encoder convnext_tiny --scales s4,s8,s16,s32
  --projection_dim 128 --reference_count 4096 --atom_rate_init 1e-6
  --prior_rate_init 1e-4 --boundary_scale_init 1.0 --total_rate_cap 64.0
  --prior_rate_cap 1.0 --boundary_scale_cap 2.0 --rate_roundoff_margin 1.0
  --decision_rule class_map --pgf_probes 0.05,0.20,0.50,0.80
  --correction_cap 3.0 --correction_gain_init 0.05 --correction_strength 1.0
  --risk_set_alpha 0.5 --rps_weight 0.25 --batch_size 8 --epochs 35
  --num_workers 8 --paths_encoder_lr 1e-5 --paths_base_lr 5e-5
  --paths_refiner_lr 5e-4 --weight_decay 1e-5
  --correction_only_epochs 3 --freeze_encoder_epochs 3
  --lr_factor 0.2 --lr_patience 5 --early_stopping_patience 12
  --amp_unfreeze_scale 256 --skip_test
)
if (( ${#RESUME_ARGS[@]} )); then ARGS+=("${RESUME_ARGS[@]}"); fi
printf 'training_command:'; printf ' %q' python train_paths.py "${ARGS[@]}"; printf '\n'
python train_paths.py "${ARGS[@]}"

# Pre-registered promotion gate.  A failing pilot exits non-zero, so an
# afterok-dependent EyePACS job will not start.
PATHS_RESULT="${FOLD_DIR}/result.json" PATHS_SOURCE="${SOURCE_CHECKPOINT}" python - <<'PY'
import json, os, pathlib, torch
result = json.loads(pathlib.Path(os.environ["PATHS_RESULT"]).read_text())
source = torch.load(os.environ["PATHS_SOURCE"], map_location="cpu", weights_only=False)
metrics = result["best_validation"]
base = source["metrics"]
confusion = metrics["confusion"]
correct = sum(int(confusion[i][i]) for i in range(len(confusion)))
checks = {
    "correct_at_least_253_of_293": correct >= 253,
    "qwk_at_least_0_9295": float(metrics["qwk"]) >= 0.9295,
    "mae_non_regression": float(metrics["mae"]) <= float(base["mae"]),
    "balanced_acc_non_regression": float(metrics["balanced_acc"]) >= float(base["balanced_acc"]),
    "macro_f1_non_regression": float(metrics["macro_f1"]) >= float(base["macro_f1"]),
    "grade3_at_least_4_of_15": int(confusion[3][3]) >= 4,
}
certificate = json.loads(pathlib.Path(result["validation_certificate_path"]).read_text())
checks["joint_paths_certificate"] = (
    certificate.get("certifies_final_paths_prediction") is True
    and certificate.get("base_only_certificate") is False
    and certificate["joint_replay_audit"]["final_correction_max_abs"] > 1e-10
)
payload = {"schema": "paths-aptos-promotion-gate-v2", "checks": checks, "passed": all(checks.values()), "correct": correct, "metrics": metrics}
gate = pathlib.Path(os.environ["PATHS_RESULT"]).with_name("PROMOTED_TO_EYEPACS.json")
gate.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
print(json.dumps(payload, indent=2, sort_keys=True))
if not payload["passed"]:
    raise SystemExit("PATHS APTOS promotion gate failed; EyePACS remains blocked")
PY

echo "PATHS APTOS gate passed."
