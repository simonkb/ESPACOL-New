#!/bin/bash
# One non-adaptive outer-test release after every requested worker is sealed.
#SBATCH --job-name=of9_ab_out
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH --output=/dpc/kuin0170/ESPACOL-New/origin_fold9_ablation_logs/release_%j.out
#SBATCH --error=/dpc/kuin0170/ESPACOL-New/origin_fold9_ablation_logs/release_%j.err
#SBATCH --account=kuin0170

set -euo pipefail
set +u
source /etc/profile.d/lmod.sh || exit 1
module load miniconda/3 || exit 1
module load cuda/12.6 || exit 1
source activate "${ORIGIN_CONDA_ENV:-G}" || exit 1
set -u

REPO_ROOT="${ORIGIN_REPO_ROOT:?ORIGIN_REPO_ROOT is required}"
LAUNCH_COMMIT="${ORIGIN_LAUNCH_COMMIT:?ORIGIN_LAUNCH_COMMIT is required}"
EXPERIMENT_ROOT="${ORIGIN_FOLD9_ABLATION_ROOT:?ORIGIN_FOLD9_ABLATION_ROOT is required}"
DATA_ROOT="${ORIGIN_DATA_ROOT:?ORIGIN_DATA_ROOT is required}"
PROTOCOL="${ORIGIN_FOLD9_ABLATION_PROTOCOL:?ORIGIN_FOLD9_ABLATION_PROTOCOL is required}"

cd "${REPO_ROOT}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
[[ "$(git rev-parse HEAD)" == "${LAUNCH_COMMIT}" ]] || {
  echo "Immutable launch commit mismatch." >&2; exit 2;
}
[[ -z "$(git status --porcelain --untracked-files=no)" ]] || {
  echo "Tracked files changed in the immutable worktree." >&2; exit 3;
}

exec 9>"${EXPERIMENT_ROOT}/locks/outer_release.lock"
flock -n 9 || { echo "Another process owns the outer release." >&2; exit 4; }
if [[ -e "${EXPERIMENT_ROOT}/OUTER_RELEASE_COMPLETE.json" || \
      -e "${EXPERIMENT_ROOT}/release/OUTER_RELEASE_COMPLETE.json" ]]; then
  echo "Fresh-only protocol refuses to repeat the outer release." >&2
  exit 5
fi

ORIGIN_RELEASE_PROTOCOL="${PROTOCOL}" \
ORIGIN_RELEASE_COMMIT="${LAUNCH_COMMIT}" \
ORIGIN_RELEASE_ROOT="${EXPERIMENT_ROOT}" \
ORIGIN_RELEASE_DATA="${DATA_ROOT}" \
python - <<'PY'
import os
from pathlib import Path
from scripts.origin_fold9_ablation_common import (
    DATASET, EXPECTED_SPLIT_SIGNATURE, FOLD, TRAIN_MARKER_SCHEMA,
    file_sha256, read_json, verify_checksummed_payload, verify_protocol,
)

protocol = read_json(os.environ["ORIGIN_RELEASE_PROTOCOL"])
variants = verify_protocol(
    protocol,
    expected_launch_commit=os.environ["ORIGIN_RELEASE_COMMIT"],
    expected_root=os.environ["ORIGIN_RELEASE_ROOT"],
    expected_data_root=os.environ["ORIGIN_RELEASE_DATA"],
)
root = Path(os.environ["ORIGIN_RELEASE_ROOT"])
for variant in variants:
    marker_path = root / "workers" / variant / f"fold{FOLD}" / "ABLATION_TRAIN_COMPLETE.json"
    marker = read_json(marker_path)
    verify_checksummed_payload(marker, schema=TRAIN_MARKER_SCHEMA)
    expected = {
        "dataset": DATASET,
        "fold": FOLD,
        "variant": variant,
        "evaluation_scope": "inner_validation_only_outer_locked",
        "outer_test_evaluated": False,
        "protocol_checksum_sha256": protocol["content_checksum_sha256"],
        "launch_commit": protocol["launch_commit"],
        "split_signature": EXPECTED_SPLIT_SIGNATURE,
    }
    for key, value in expected.items():
        if marker.get(key) != value:
            raise RuntimeError(f"{variant} training marker {key} mismatch")
    paths = marker.get("artifact_paths", {})
    hashes = marker.get("artifact_sha256", {})
    if set(paths) != set(hashes):
        raise RuntimeError(f"{variant} marker artifact sets differ")
    for name, recorded in hashes.items():
        artifact = Path(paths[name])
        if file_sha256(artifact) != recorded:
            raise RuntimeError(f"{variant} artifact changed after training audit: {name}")
print("all_training_markers_verified", variants)
PY

echo "=== ORIGIN fold-9 single outer-test release ==="
date --iso-8601=seconds
python scripts/release_origin_fold9_ablation_outer.py \
  --root "${EXPERIMENT_ROOT}" --protocol "${PROTOCOL}" \
  --data-root "${DATA_ROOT}" --num-workers 8
echo "Single outer release completed."
