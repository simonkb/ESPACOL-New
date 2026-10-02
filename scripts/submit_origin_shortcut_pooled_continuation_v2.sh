#!/bin/bash
# Exact-protocol continuation for comparator-v2 pooled tasks 21..41.
# Model training and scientific audit are executed from the immutable original
# snapshot.  The newer checkout supplies only a read-only artifact validator.
#SBATCH --job-name=origin_sc_pool_fix
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --array=21-41%8
#SBATCH --output=/dpc/kuin0170/ESPACOL-New/origin_shortcut_logs/pooled_continue_%A_%a.out
#SBATCH --error=/dpc/kuin0170/ESPACOL-New/origin_shortcut_logs/pooled_continue_%A_%a.err
#SBATCH --account=kuin0170

set -euo pipefail
set +u
source /etc/profile.d/lmod.sh || exit 1
module load miniconda/3 || exit 1
module load cuda/12.6 || exit 1
source activate "${ORIGIN_CONDA_ENV:-G}" || exit 1
set -u

REPO_ROOT="${ORIGIN_REPO_ROOT:?immutable original worktree is required}"
RECOVERY_ROOT="${ORIGIN_RECOVERY_REPO_ROOT:?recovery checkout is required}"
RECOVERY_COMMIT="${ORIGIN_RECOVERY_COMMIT:?recovery commit is required}"
SUITE_ROOT="${ORIGIN_SHORTCUT_COMPARATOR_ROOT:?suite root is required}"
DATA_ROOT="${ORIGIN_DATA_ROOT:?data root is required}"
PROTOCOL="${ORIGIN_SHORTCUT_COMPARATOR_PROTOCOL:?protocol is required}"
LAUNCH_COMMIT="${ORIGIN_LAUNCH_COMMIT:?original launch commit is required}"
TASK_ID="${SLURM_ARRAY_TASK_ID:?array task id is required}"

[[ "${TASK_ID}" =~ ^[0-9]+$ && "${TASK_ID}" -ge 21 && "${TASK_ID}" -le 41 ]] || {
  echo "Pooled continuation accepts only original task ids 21..41." >&2; exit 2;
}
[[ "$(git -C "${REPO_ROOT}" rev-parse HEAD)" == "${LAUNCH_COMMIT}" ]] || {
  echo "Immutable comparator commit mismatch." >&2; exit 3;
}
[[ -z "$(git -C "${REPO_ROOT}" status --porcelain --untracked-files=no)" ]] || {
  echo "Immutable comparator worktree contains tracked changes." >&2; exit 4;
}
[[ "$(git -C "${RECOVERY_ROOT}" rev-parse HEAD)" == "${RECOVERY_COMMIT}" ]] || {
  echo "Recovery implementation commit mismatch." >&2; exit 5;
}
[[ -z "$(git -C "${RECOVERY_ROOT}" status --porcelain --untracked-files=no)" ]] || {
  echo "Recovery checkout contains tracked changes." >&2; exit 6;
}

cd "${REPO_ROOT}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

TASK_JSON="$(python - "${TASK_ID}" "${PROTOCOL}" "${LAUNCH_COMMIT}" "${SUITE_ROOT}" <<'PY'
import hashlib, json, sys
from pathlib import Path
from scripts.origin_shortcut_comparator_common import PROTOCOL_CORE_SHA256, task_at
index = int(sys.argv[1])
task = task_at(index)
if task.model_variant != "pooled_conditional":
    raise RuntimeError("continuation task is not pooled_conditional")
path = Path(sys.argv[2])
payload = json.loads(path.read_text())
recorded = payload.pop("content_checksum_sha256")
observed = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
if recorded != observed:
    raise RuntimeError("comparator protocol checksum mismatch")
if payload.get("protocol_core_sha256") != PROTOCOL_CORE_SHA256:
    raise RuntimeError("comparator protocol core mismatch")
if payload.get("launch_commit") != sys.argv[3]:
    raise RuntimeError("comparator launch commit mismatch")
if Path(payload.get("suite_root", "")).resolve() != Path(sys.argv[4]).resolve():
    raise RuntimeError("comparator suite root mismatch")
print(json.dumps(task.__dict__, sort_keys=True))
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
RECOVERY_RECORD="${SUITE_ROOT}/recovery/task${TASK_ID}.json"

mkdir -p "${SUITE_ROOT}/locks" "${RUN_DIR}" "${SUITE_ROOT}/recovery"
exec 9>"${SUITE_ROOT}/locks/task${TASK_ID}.lock"
flock -n 9 || { echo "Original/recovery worker still owns task ${TASK_ID}." >&2; exit 7; }
[[ ! -e "${RECOVERY_RECORD}" ]] || {
  echo "Recovery record already exists for task ${TASK_ID}." >&2; exit 8;
}
[[ ! -e "${SUITE_ROOT}/COMPARATOR_SHORTCUT_RESULTS_V2.json" ]] || {
  echo "Comparator aggregate already exists; continuation is forbidden." >&2; exit 9;
}

echo "=== comparator-v2 pooled continuation task ${TASK_ID}: ${ARM}/${FAMILY}/seed${TRAIN_SEED} ==="
date --iso-8601=seconds
echo "original_commit=${LAUNCH_COMMIT}"
echo "recovery_commit=${RECOVERY_COMMIT}"

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

if [[ -f "${FOLD_DIR}/best_learned.pth" && -f "${FOLD_DIR}/result.json" ]]; then
  echo "Reusing completed immutable-snapshot training artifacts."
elif [[ -f "${FOLD_DIR}/last_learned.pth" && -f "${FOLD_DIR}/shortcut_protocol.json" ]]; then
  printf 'resume_training_command:'; printf ' %q' python train_origin_shortcut.py "${TRAIN_ARGS[@]}" --resume; printf '\n'
  python train_origin_shortcut.py "${TRAIN_ARGS[@]}" --resume
else
  if [[ -d "${FOLD_DIR}" ]]; then
    if [[ -n "$(find "${FOLD_DIR}" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
      echo "Partial artifacts exist without a resumable checkpoint: ${FOLD_DIR}." >&2
      exit 10
    fi
  fi
  printf 'fresh_training_command:'; printf ' %q' python train_origin_shortcut.py "${TRAIN_ARGS[@]}"; printf '\n'
  python train_origin_shortcut.py "${TRAIN_ARGS[@]}"
fi

CHECKPOINT="${FOLD_DIR}/best_learned.pth"
[[ -f "${CHECKPOINT}" && -f "${FOLD_DIR}/result.json" ]] || {
  echo "Pooled training artifacts are incomplete." >&2; exit 11;
}

run_or_reuse_historical_audit() {
  local audit_family="$1" output="$2"
  if [[ -f "${output}" ]]; then
    echo "Reusing atomically written historical audit ${output}."
    return 0
  fi
  local prediction="${output%.json}_predictions.jsonl"
  [[ ! -e "${prediction}" ]] || {
    echo "Orphan prediction artifact without audit report: ${prediction}." >&2
    return 12
  }
  set +e
  python scripts/audit_origin_shortcut.py \
    --checkpoint "${CHECKPOINT}" --data-root "${DATA_ROOT}" \
    --family "${audit_family}" --output "${output}" --batch-size 4 \
    --audit-grid 64 --audit-seed 7193 --permutations 999 --bootstrap-replicates 2000
  local status=$?
  set -e
  # The original launch snapshot is expected to exit nonzero only in its final
  # pooled console summary.  Never trust the status override itself: the
  # read-only validator below must reproduce all hashes, identities and
  # non-applicability contracts before this worker can succeed.
  echo "historical_audit_exit_status=${status} family=${audit_family}"
  [[ -f "${output}" ]] || return 13
}

if [[ "${ARM}" == "clean" ]]; then
  for audit_family in localized border diffuse; do
    run_or_reuse_historical_audit \
      "${audit_family}" "${FOLD_DIR}/shortcut_audit_${audit_family}.json"
  done
else
  run_or_reuse_historical_audit "${FAMILY}" "${FOLD_DIR}/shortcut_audit.json"
fi

cd "${RECOVERY_ROOT}"
python scripts/validate_origin_shortcut_comparator_recovery.py \
  --suite-root "${SUITE_ROOT}" --protocol "${PROTOCOL}" \
  --task-ids "${TASK_ID}" --output "${RECOVERY_RECORD}"
echo "Pooled continuation task ${TASK_ID} completed and validated read-only."
