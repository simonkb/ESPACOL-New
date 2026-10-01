#!/bin/bash
# Four comparator variants x the frozen 21-cell shortcut task layout = 84 jobs.
#SBATCH --job-name=origin_sc_cmp
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --array=0-83%8
#SBATCH --output=/dpc/kuin0170/ESPACOL-New/origin_shortcut_logs/comparator_%A_%a.out
#SBATCH --error=/dpc/kuin0170/ESPACOL-New/origin_shortcut_logs/comparator_%A_%a.err
#SBATCH --account=kuin0170

set -euo pipefail
set +u
source /etc/profile.d/lmod.sh || exit 1
module load miniconda/3 || exit 1
module load cuda/12.6 || exit 1
source activate "${ORIGIN_CONDA_ENV:-G}" || exit 1
set -u

REPO_ROOT="${ORIGIN_REPO_ROOT:?ORIGIN_REPO_ROOT is required}"
SUITE_ROOT="${ORIGIN_SHORTCUT_COMPARATOR_ROOT:?ORIGIN_SHORTCUT_COMPARATOR_ROOT is required}"
DATA_ROOT="${ORIGIN_DATA_ROOT:?ORIGIN_DATA_ROOT is required}"
PROTOCOL="${ORIGIN_SHORTCUT_COMPARATOR_PROTOCOL:?ORIGIN_SHORTCUT_COMPARATOR_PROTOCOL is required}"
LAUNCH_COMMIT="${ORIGIN_LAUNCH_COMMIT:?ORIGIN_LAUNCH_COMMIT is required}"
TASK_ID="${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}"

cd "${REPO_ROOT}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
[[ "$(git rev-parse HEAD)" == "${LAUNCH_COMMIT}" ]] || {
  echo "Immutable comparator commit mismatch." >&2; exit 2;
}
[[ -z "$(git status --porcelain --untracked-files=no)" ]] || {
  echo "Tracked files changed in immutable comparator worktree." >&2; exit 3;
}

TASK_JSON="$(python - "${TASK_ID}" "${PROTOCOL}" "${LAUNCH_COMMIT}" <<'PY'
import hashlib, json, sys
from pathlib import Path
from scripts.origin_shortcut_comparator_common import (
    PROTOCOL_CORE_SHA256, TASK_COUNT, task_at,
)
index = int(sys.argv[1])
if index < 0 or index >= TASK_COUNT:
    raise ValueError(f"task {index} outside 0..{TASK_COUNT - 1}")
path = Path(sys.argv[2])
payload = json.loads(path.read_text())
recorded = payload.pop("content_checksum_sha256")
observed = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
if recorded != observed:
    raise RuntimeError("comparator protocol checksum mismatch")
if payload.get("protocol_core_sha256") != PROTOCOL_CORE_SHA256:
    raise RuntimeError("comparator protocol core mismatch")
if payload.get("launch_commit") != sys.argv[3]:
    raise RuntimeError("comparator protocol launch commit mismatch")
print(json.dumps(task_at(index).__dict__, sort_keys=True))
PY
)"
readarray -t TASK_FIELDS < <(python - "${TASK_JSON}" <<'PY'
import json, sys
p = json.loads(sys.argv[1])
for key in ("model_variant", "arm", "family", "training_seed"):
    print(p[key])
PY
)
MODEL_VARIANT="${TASK_FIELDS[0]}"
ARM="${TASK_FIELDS[1]}"
FAMILY="${TASK_FIELDS[2]}"
TRAIN_SEED="${TASK_FIELDS[3]}"
RUN_DIR="${SUITE_ROOT}/workers/${MODEL_VARIANT}/${ARM}/${FAMILY}/seed${TRAIN_SEED}"
FOLD_DIR="${RUN_DIR}/fold0"

mkdir -p "${SUITE_ROOT}/locks" "${RUN_DIR}"
exec 9>"${SUITE_ROOT}/locks/task${TASK_ID}.lock"
flock -n 9 || { echo "Another worker owns comparator task ${TASK_ID}." >&2; exit 4; }
if [[ -e "${FOLD_DIR}/best_learned.pth" || -e "${FOLD_DIR}/last_learned.pth" ]]; then
  echo "Fresh-only comparator worker refuses existing artifacts: ${FOLD_DIR}." >&2
  exit 5
fi

echo "=== shortcut comparator v2 task ${TASK_ID}: ${MODEL_VARIANT}/${ARM}/${FAMILY}/seed${TRAIN_SEED} ==="
date --iso-8601=seconds
echo "commit=${LAUNCH_COMMIT}"
python --version
python -m pytest -q tests/test_origin_shortcut_benchmark.py tests/test_origin_acceptance_baselines.py

TRAIN_ARGS=(
  --dataset aptos --data_root "${DATA_ROOT}" --run_dir "${RUN_DIR}"
  --n_folds 5 --folds 0 --split_seed 42 --seed "${TRAIN_SEED}"
  --model_variant "${MODEL_VARIANT}"
  --shortcut_seed 271828 --shortcut_arm "${ARM}" --shortcut_family "${FAMILY}"
  --shortcut_strength 0.30 --marker_radius_fraction 0.040
  --image_size 640 --encoder convnext_tiny --scales s4,s8,s16,s32
  --projection_dim 128 --reference_count 4096 --atom_rate_init 1e-6
  --prior_rate_init 1e-4 --boundary_scale_init 1.0 --total_rate_cap 64.0
  --prior_rate_cap 1.0 --boundary_scale_cap 2.0 --rate_roundoff_margin 1.0
  --atom_mode cumulative --decision_rule class_map --batch_size 8 --epochs 35
  --num_workers 8 --lr 1e-4 --head_lr 5e-4 --weight_decay 1e-5
  --freeze_encoder_epochs 2 --scheduler plateau --lr_factor 0.2 --lr_patience 5
  --early_stopping_patience 12 --checkpoint_selection acc_then_qwk
  --selection_qwk_weight 0.1 --rps_weight 0.25 --evidence_budget_weight 0.0
  --class_weighting none --amp --amp_init_scale 4096 --amp_unfreeze_scale 256
  --amp_growth_interval 2000 --amp_max_consecutive_skips 8 --skip_test
)
printf 'training_command:'; printf ' %q' python train_origin_shortcut.py "${TRAIN_ARGS[@]}"; printf '\n'
python train_origin_shortcut.py "${TRAIN_ARGS[@]}"

CHECKPOINT="${FOLD_DIR}/best_learned.pth"
[[ -f "${CHECKPOINT}" && -f "${FOLD_DIR}/result.json" ]] || {
  echo "Comparator training did not produce required learned artifacts." >&2; exit 6;
}
run_audit() {
  local audit_family="$1" output="$2"
  [[ ! -e "${output}" ]] || { echo "Refusing to overwrite ${output}." >&2; return 7; }
  python scripts/audit_origin_shortcut.py \
    --checkpoint "${CHECKPOINT}" --data-root "${DATA_ROOT}" \
    --family "${audit_family}" --output "${output}" --batch-size 4 \
    --audit-grid 64 --audit-seed 7193 --permutations 999 --bootstrap-replicates 2000
}
if [[ "${ARM}" == "clean" ]]; then
  for audit_family in localized border diffuse; do
    run_audit "${audit_family}" "${FOLD_DIR}/shortcut_audit_${audit_family}.json"
  done
else
  run_audit "${FAMILY}" "${FOLD_DIR}/shortcut_audit.json"
fi
echo "Shortcut comparator task ${TASK_ID} completed and blind-audited."
