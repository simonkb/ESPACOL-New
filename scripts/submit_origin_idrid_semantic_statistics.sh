#!/usr/bin/env bash
# Postprocess an already completed IDRiD semantic audit.  No model is loaded.
# May be submitted independently now, or with --dependency=afterok:<semantic-job>.
#SBATCH --job-name=org_idrid_stats
#SBATCH --partition=prod
#SBATCH --account=kuin0170
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=origin_idrid_semantic_statistics_%j.out
#SBATCH --error=origin_idrid_semantic_statistics_%j.err

set -euo pipefail

REPO_ROOT="${ORIGIN_REPO_ROOT:-/dpc/kuin0170/ESPACOL-New}"
ACCEPTANCE_ROOT="${ORIGIN_ACCEPTANCE_ROOT:-$REPO_ROOT/runs/origin_acceptance}"
SUMMARY="${ORIGIN_IDRID_SEMANTIC_SUMMARY:-$ACCEPTANCE_ROOT/idrid_semantics/idrid_semantic_summary.json}"
OUTPUT_DIR="${ORIGIN_IDRID_STATISTICS_OUTPUT:-$ACCEPTANCE_ROOT/idrid_semantics/statistics}"
PROTOCOL="${ORIGIN_IDRID_STATISTICS_PROTOCOL:-$REPO_ROOT/scripts/protocols/origin_idrid_semantic_statistics_protocol.json}"

cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
ENV_NAME="${ORIGIN_CONDA_ENV:-G}"
ENV_PYTHON="${ORIGIN_PYTHON:-${HOME}/.conda/envs/${ENV_NAME}/bin/python}"
[[ -x "$ENV_PYTHON" ]] || {
  echo "Missing cluster Python environment: $ENV_PYTHON" >&2; exit 6;
}
export PATH="$(dirname "$ENV_PYTHON"):$PATH"

if [[ -n "${ORIGIN_LAUNCH_COMMIT:-}" && "$(git rev-parse HEAD)" != "${ORIGIN_LAUNCH_COMMIT}" ]]; then
  echo "Immutable launch commit mismatch." >&2
  exit 2
fi
[[ -z "$(git status --porcelain --untracked-files=no)" ]] || {
  echo "Tracked files changed in the IDRiD statistics worktree." >&2; exit 3;
}
[[ -f "$SUMMARY" ]] || { echo "Missing semantic summary: $SUMMARY" >&2; exit 4; }
[[ ! -f "$OUTPUT_DIR/idrid_statistics_manifest.json" ]] || {
  echo "Completed IDRiD statistics already exist; refusing overwrite." >&2; exit 5;
}

printf '=== ORIGIN IDRiD image-cluster semantic statistics ===\n'
date --iso-8601=seconds
git rev-parse HEAD
python -m pytest -q \
  tests/test_origin_idrid_semantics.py \
  tests/test_origin_idrid_semantic_statistics.py
python tools/analyze_origin_idrid_semantics.py \
  --summary "$SUMMARY" --protocol "$PROTOCOL" --output-dir "$OUTPUT_DIR"

echo "Statistics manifest: $OUTPUT_DIR/idrid_statistics_manifest.json"
