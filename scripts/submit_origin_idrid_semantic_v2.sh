#!/usr/bin/env bash
# New v2 IDRiD audit.  It never reads from or writes into the frozen v1 output.
# The job performs GPU ledger extraction followed by CPU-light cluster analysis.
#SBATCH --job-name=org_idrid_v2
#SBATCH --partition=gpu
#SBATCH --account=kuin0170
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=10:00:00
#SBATCH --output=origin_idrid_semantic_v2_%j.out
#SBATCH --error=origin_idrid_semantic_v2_%j.err

set -euo pipefail

SHARED_ROOT="${ORIGIN_SHARED_ROOT:-/dpc/kuin0170/ESPACOL-New}"
REPO_ROOT="${ORIGIN_REPO_ROOT:-$SHARED_ROOT}"
CV_ROOT="${ORIGIN_V3_CV_ROOT:-$SHARED_ROOT/runs/origin_v3_full_cv_20260922T084547Z}"
ACCEPTANCE_ROOT="${ORIGIN_ACCEPTANCE_ROOT:-$SHARED_ROOT/runs/origin_acceptance}"
IDRID_ROOT="${ORIGIN_IDRID_ROOT:-$SHARED_ROOT/Datasets/IDRiD}"
OUTPUT_ROOT="${ORIGIN_IDRID_V2_OUTPUT_ROOT:-$ACCEPTANCE_ROOT/idrid_semantics_v2_bidirectional}"
AUDIT_DIR="$OUTPUT_ROOT/audit"
STATISTICS_DIR="$OUTPUT_ROOT/statistics"
PROTOCOL="$REPO_ROOT/scripts/protocols/origin_idrid_semantic_statistics_protocol_v2.json"

cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
set +u
source /etc/profile.d/lmod.sh || exit 1
module load miniconda/3 || exit 1
module load cuda/12.6 || exit 1
source activate "${ORIGIN_CONDA_ENV:-G}" || exit 1
set -u

if [[ -z "${ORIGIN_LAUNCH_COMMIT:-}" ]]; then
  echo "ORIGIN_LAUNCH_COMMIT is required for the immutable v2 audit." >&2
  exit 2
fi
if [[ "$(git rev-parse HEAD)" != "${ORIGIN_LAUNCH_COMMIT}" ]]; then
  echo "Immutable launch commit mismatch." >&2
  exit 3
fi
[[ -z "$(git status --porcelain --untracked-files=no)" ]] || {
  echo "Tracked files changed in the IDRiD v2 worktree." >&2; exit 4;
}
[[ -d "$IDRID_ROOT/A. Segmentation" ]] || {
  echo "IDRiD root is unavailable (detached worktrees do not contain datasets): $IDRID_ROOT" >&2
  exit 5
}

printf '=== ORIGIN external IDRiD semantic audit v2: bidirectional scale matching ===\n'
date --iso-8601=seconds
git rev-parse HEAD
printf 'audit_dir=%s\nstatistics_dir=%s\n' "$AUDIT_DIR" "$STATISTICS_DIR"
printf 'idrid_root=%s\n' "$IDRID_ROOT"

python -m pytest -q \
  tests/test_origin_idrid_semantics.py \
  tests/test_origin_idrid_semantic_statistics.py \
  tests/test_origin_idrid_semantics_v2.py

SUMMARY="$AUDIT_DIR/idrid_semantic_summary_v2.json"
if [[ ! -f "$SUMMARY" ]]; then
  if [[ -e "$AUDIT_DIR" ]]; then
    echo "Partial v2 audit directory exists without a summary; refusing overwrite: $AUDIT_DIR" >&2
    exit 6
  fi
  CHECKPOINTS=()
  for fold in $(seq 0 9); do
    checkpoint="$CV_ROOT/dr/workers/fold${fold}/fold${fold}/best.pth"
    test -f "$checkpoint" || { echo "missing checkpoint: $checkpoint" >&2; exit 7; }
    CHECKPOINTS+=("$checkpoint")
  done
  python - "$PROTOCOL" "${CHECKPOINTS[@]}" <<'PY'
import hashlib
import json
from pathlib import Path
import sys

protocol = json.loads(Path(sys.argv[1]).read_text())
expected = protocol["expected_checkpoint_sha256_by_fold"]
for fold, raw_path in enumerate(sys.argv[2:]):
    path = Path(raw_path)
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    observed = digest.hexdigest()
    if observed != expected[str(fold)]:
        raise SystemExit(
            f"sealed checkpoint mismatch for fold {fold}: {observed} != {expected[str(fold)]}"
        )
print("All ten checkpoint identities match the sealed v2 protocol.")
PY
  python tools/audit_origin_idrid_semantics_v2.py \
    --idrid-root "$IDRID_ROOT" \
    --checkpoints "${CHECKPOINTS[@]}" \
    --output-dir "$AUDIT_DIR" \
    --batch-size 2 \
    --paired-repeats 20 \
    --seed 20261011 \
    --source-commit "$ORIGIN_LAUNCH_COMMIT"
else
  echo "Reusing completed, checksum-verified-on-analysis v2 audit: $SUMMARY"
fi

MANIFEST="$STATISTICS_DIR/idrid_statistics_manifest_v2.json"
if [[ -f "$MANIFEST" ]]; then
  echo "Completed v2 statistics already exist; refusing overwrite: $MANIFEST" >&2
  exit 8
fi
python tools/analyze_origin_idrid_semantics_v2.py \
  --summary "$SUMMARY" \
  --protocol "$PROTOCOL" \
  --output-dir "$STATISTICS_DIR" \
  --expected-source-commit "$ORIGIN_LAUNCH_COMMIT"

echo "V2 statistics manifest: $MANIFEST"
