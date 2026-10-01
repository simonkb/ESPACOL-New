#!/bin/bash
# APTOS controlled ordinal-shortcut pilot: 9 benchmark, 9 cue-only, 3 clean jobs.
#SBATCH --job-name=origin_shortcut
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --array=0-20%5
#SBATCH --output=/dpc/kuin0170/ESPACOL-New/origin_shortcut_logs/worker_%A_%a.out
#SBATCH --error=/dpc/kuin0170/ESPACOL-New/origin_shortcut_logs/worker_%A_%a.err
#SBATCH --account=kuin0170

set -euo pipefail
set +u
source /etc/profile.d/lmod.sh || exit 1
module load miniconda/3 || exit 1
module load cuda/12.6 || exit 1
source activate "${ORIGIN_CONDA_ENV:-G}" || exit 1
set -u

REPO_ROOT="${ORIGIN_REPO_ROOT:?ORIGIN_REPO_ROOT is required}"
PILOT_ROOT="${ORIGIN_SHORTCUT_ROOT:?ORIGIN_SHORTCUT_ROOT is required}"
DATA_ROOT="${ORIGIN_DATA_ROOT:?ORIGIN_DATA_ROOT is required}"
PROTOCOL="${ORIGIN_SHORTCUT_PROTOCOL:?ORIGIN_SHORTCUT_PROTOCOL is required}"
LAUNCH_COMMIT="${ORIGIN_LAUNCH_COMMIT:?ORIGIN_LAUNCH_COMMIT is required}"
TASK_ID="${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}"
[[ "${TASK_ID}" =~ ^([0-9]|1[0-9]|20)$ ]] || {
  echo "Invalid shortcut task ${TASK_ID}; expected 0..20." >&2; exit 2;
}

SEEDS=(1701 2603 3907)
FAMILIES=(localized border diffuse)
if (( TASK_ID < 9 )); then
  ARM="shortcut"
  CELL="${TASK_ID}"
  FAMILY="${FAMILIES[$(( CELL / 3 ))]}"
  SEED_INDEX="$(( CELL % 3 ))"
  RUN_DIR="${PILOT_ROOT}/workers/${ARM}/${FAMILY}/seed${SEEDS[${SEED_INDEX}]}"
elif (( TASK_ID < 18 )); then
  ARM="cue_only"
  CELL="$(( TASK_ID - 9 ))"
  FAMILY="${FAMILIES[$(( CELL / 3 ))]}"
  SEED_INDEX="$(( CELL % 3 ))"
  RUN_DIR="${PILOT_ROOT}/workers/${ARM}/${FAMILY}/seed${SEEDS[${SEED_INDEX}]}"
else
  ARM="clean"
  FAMILY="localized"
  SEED_INDEX="$(( TASK_ID - 18 ))"
  RUN_DIR="${PILOT_ROOT}/workers/${ARM}/seed${SEEDS[${SEED_INDEX}]}"
fi
TRAIN_SEED="${SEEDS[${SEED_INDEX}]}"

cd "${REPO_ROOT}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
[[ "$(git rev-parse HEAD)" == "${LAUNCH_COMMIT}" ]] || {
  echo "Immutable launch commit mismatch." >&2; exit 3;
}
[[ -z "$(git status --porcelain --untracked-files=no)" ]] || {
  echo "Tracked files changed in immutable shortcut worktree." >&2; exit 4;
}

ORIGIN_CHECK_PROTOCOL="${PROTOCOL}" \
ORIGIN_CHECK_ROOT="${PILOT_ROOT}" \
ORIGIN_CHECK_DATA="${DATA_ROOT}" \
ORIGIN_CHECK_COMMIT="${LAUNCH_COMMIT}" python - <<'PY'
import hashlib, json, os
from pathlib import Path
p = json.loads(Path(os.environ["ORIGIN_CHECK_PROTOCOL"]).read_text())
recorded = p.pop("content_checksum_sha256")
observed = hashlib.sha256(json.dumps(p, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
if observed != recorded:
    raise RuntimeError("shortcut protocol checksum mismatch")
expected = {
    "schema": "origin-ordinal-shortcut-pilot-protocol-v1",
    "launch_commit": os.environ["ORIGIN_CHECK_COMMIT"],
    "aptos_images": 3662,
    "fold": 0,
    "split_seed": 42,
    "shortcut_seed": 271828,
    "training_seeds": [1701, 2603, 3907],
    "families": ["localized", "border", "diffuse"],
    "arms": ["shortcut", "cue_only", "clean"],
}
for key, value in expected.items():
    if p.get(key) != value:
        raise RuntimeError(f"shortcut protocol {key} mismatch")
if Path(p["pilot_root"]).resolve() != Path(os.environ["ORIGIN_CHECK_ROOT"]).resolve():
    raise RuntimeError("shortcut pilot root mismatch")
if Path(p["data_root"]).resolve() != Path(os.environ["ORIGIN_CHECK_DATA"]).resolve():
    raise RuntimeError("shortcut data root mismatch")
PY

mkdir -p "${PILOT_ROOT}/locks" "${RUN_DIR}"
exec 9>"${PILOT_ROOT}/locks/task${TASK_ID}.lock"
flock -n 9 || { echo "Another worker owns shortcut task ${TASK_ID}." >&2; exit 5; }

FOLD_DIR="${RUN_DIR}/fold0"
RESUME_ARGS=()
SKIP_TRAIN=0
if [[ "${ORIGIN_SHORTCUT_RESUME:-0}" == "1" ]]; then
  if [[ -f "${FOLD_DIR}/result.json" && -f "${FOLD_DIR}/best.pth" ]]; then
    SKIP_TRAIN=1
  elif [[ -f "${FOLD_DIR}/last.pth" && -f "${FOLD_DIR}/shortcut_protocol.json" ]]; then
    RESUME_ARGS+=(--resume)
  else
    echo "Resume requested without a resumable or completed run: ${FOLD_DIR}." >&2
    exit 6
  fi
elif [[ -e "${FOLD_DIR}/best.pth" || -e "${FOLD_DIR}/last.pth" || -e "${FOLD_DIR}/result.json" ]]; then
  echo "Fresh-only shortcut task refuses existing artifacts: ${FOLD_DIR}." >&2
  exit 7
fi

echo "=== APTOS controlled shortcut task ${TASK_ID}: ${ARM}/${FAMILY}/seed${TRAIN_SEED} ==="
date --iso-8601=seconds
echo "commit=${LAUNCH_COMMIT}"
python --version
python - <<'PY'
import torch, torchvision
if not torch.cuda.is_available():
    raise RuntimeError("CUDA is required")
print("torch", torch.__version__, "torchvision", torchvision.__version__)
print("gpu", torch.cuda.get_device_name(0))
PY
python -m pytest -q tests/test_origin_shortcut_benchmark.py

if (( SKIP_TRAIN == 0 )); then
  TRAIN_ARGS=(
    --dataset aptos --data_root "${DATA_ROOT}" --run_dir "${RUN_DIR}"
    --n_folds 5 --folds 0 --split_seed 42 --seed "${TRAIN_SEED}"
    --shortcut_seed 271828 --shortcut_arm "${ARM}" --shortcut_family "${FAMILY}"
    --shortcut_strength "${ORIGIN_SHORTCUT_STRENGTH:-0.30}"
    --marker_radius_fraction "${ORIGIN_SHORTCUT_RADIUS_FRACTION:-0.040}"
    --image_size "${ORIGIN_SHORTCUT_IMAGE_SIZE:-640}" --encoder convnext_tiny
    --scales s4,s8,s16,s32 --projection_dim 128 --reference_count 4096
    --atom_rate_init 1e-6 --prior_rate_init 1e-4 --boundary_scale_init 1.0
    --total_rate_cap 64.0 --prior_rate_cap 1.0 --boundary_scale_cap 2.0
    --rate_roundoff_margin 1.0 --atom_mode cumulative --decision_rule class_map
    --batch_size "${ORIGIN_SHORTCUT_BATCH_SIZE:-8}"
    --epochs "${ORIGIN_SHORTCUT_EPOCHS:-35}"
    --num_workers "${ORIGIN_SHORTCUT_NUM_WORKERS:-8}"
    --lr 1e-4 --head_lr 5e-4 --weight_decay 1e-5
    --freeze_encoder_epochs 2 --scheduler plateau --lr_factor 0.2 --lr_patience 5
    --early_stopping_patience 12 --checkpoint_selection acc_then_qwk
    --selection_qwk_weight 0.1 --rps_weight 0.25 --evidence_budget_weight 0.0
    --class_weighting none --amp --amp_init_scale 4096 --amp_unfreeze_scale 256
    --amp_growth_interval 2000 --amp_max_consecutive_skips 8 --include_test
  )
  if (( ${#RESUME_ARGS[@]} )); then TRAIN_ARGS+=("${RESUME_ARGS[@]}"); fi
  printf 'training_command:'; printf ' %q' python train_origin_shortcut.py "${TRAIN_ARGS[@]}"; printf '\n'
  python train_origin_shortcut.py "${TRAIN_ARGS[@]}"
fi

CHECKPOINT="${FOLD_DIR}/best.pth"
[[ -f "${CHECKPOINT}" && -f "${FOLD_DIR}/result.json" ]] || {
  echo "Shortcut training did not produce a selected checkpoint and result." >&2; exit 8;
}

run_audit() {
  local audit_family="$1" output="$2"
  [[ ! -e "${output}" ]] || {
    echo "Refusing to overwrite shortcut audit: ${output}." >&2; return 9;
  }
  python scripts/audit_origin_shortcut.py \
    --checkpoint "${CHECKPOINT}" --data-root "${DATA_ROOT}" \
    --family "${audit_family}" --output "${output}" \
    --batch-size "${ORIGIN_SHORTCUT_AUDIT_BATCH_SIZE:-4}" \
    --audit-grid "${ORIGIN_SHORTCUT_AUDIT_GRID:-64}" \
    --audit-seed 7193 --permutations 999 --bootstrap-replicates 2000
}

if [[ "${ARM}" == "clean" ]]; then
  for audit_family in "${FAMILIES[@]}"; do
    output="${FOLD_DIR}/shortcut_audit_${audit_family}.json"
    if [[ "${ORIGIN_SHORTCUT_RESUME:-0}" == "1" && -f "${output}" ]]; then
      continue
    fi
    run_audit "${audit_family}" "${output}"
  done
else
  output="${FOLD_DIR}/shortcut_audit.json"
  if [[ "${ORIGIN_SHORTCUT_RESUME:-0}" != "1" || ! -f "${output}" ]]; then
    run_audit "${FAMILY}" "${output}"
  fi
fi

echo "Controlled shortcut task ${TASK_ID} completed and audited."
