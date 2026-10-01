#!/bin/bash
# Assemble the final ORIGIN evidence package after every prerequisite succeeds.
# The semantic input must be the corrective, bidirectionally matched IDRiD v2
# manifest; the assembler deliberately rejects the historical v1 control.
# This is a CPU-only validation/manifest job and must run from an immutable
# detached worktree.  No raw image, checkpoint, prediction, or identifier is
# copied into the resulting package. Privacy-safe outer predictions and
# anonymous memberships are bundled; checkpoint tensors require an external
# stable archive reference before public reproducibility can be claimed.
#SBATCH --job-name=origin_pkg
#SBATCH --partition=prod
#SBATCH --account=kuin0170
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=0-04:00:00
#SBATCH --output=origin_acceptance_package_%j.out
#SBATCH --error=origin_acceptance_package_%j.err

set -euo pipefail

REPO_ROOT="${ORIGIN_REPO_ROOT:?ORIGIN_REPO_ROOT is required}"
LAUNCH_COMMIT="${ORIGIN_LAUNCH_COMMIT:?ORIGIN_LAUNCH_COMMIT is required}"
OUTPUT_DIR="${ORIGIN_PACKAGE_OUTPUT:?ORIGIN_PACKAGE_OUTPUT is required}"
NUMERICAL="${ORIGIN_NUMERICAL_AUDIT:?ORIGIN_NUMERICAL_AUDIT is required}"
DECODER="${ORIGIN_DECODER_CONTRACT_AUDIT:?ORIGIN_DECODER_CONTRACT_AUDIT is required}"
ARTIFACTS="${ORIGIN_ARTIFACT_MANIFEST:?ORIGIN_ARTIFACT_MANIFEST is required}"
IDRID="${ORIGIN_IDRID_STATISTICS_MANIFEST:?ORIGIN_IDRID_STATISTICS_MANIFEST is required}"
OOF="${ORIGIN_OOF_STATISTICS_MANIFEST:?ORIGIN_OOF_STATISTICS_MANIFEST is required}"
SHORTCUT="${ORIGIN_SHORTCUT_AGGREGATE:?ORIGIN_SHORTCUT_AGGREGATE is required}"
BASELINES="${ORIGIN_BASELINE_AGGREGATE:?ORIGIN_BASELINE_AGGREGATE is required}"
CHECKPOINT_ARCHIVE_URI="${ORIGIN_CHECKPOINT_ARCHIVE_URI:-}"
CHECKPOINT_ARCHIVE_SHA256="${ORIGIN_CHECKPOINT_ARCHIVE_SHA256:-}"
CHECKPOINT_ARCHIVE_BYTES="${ORIGIN_CHECKPOINT_ARCHIVE_BYTES:-}"
EVIDENCE_ARCHIVE_URI="${ORIGIN_EVIDENCE_ARCHIVE_URI:-}"
EVIDENCE_ARCHIVE_SHA256="${ORIGIN_EVIDENCE_ARCHIVE_SHA256:-}"
EVIDENCE_ARCHIVE_BYTES="${ORIGIN_EVIDENCE_ARCHIVE_BYTES:-}"
# A negative scientific gate remains a reproducible result and is packaged
# with claims explicitly withheld.  Set this to 1 only for a stricter
# submission-readiness marker that requires both gates to pass.
REQUIRE_GATES="${ORIGIN_REQUIRE_GATES_PASS:-0}"

ENV_NAME="${ORIGIN_CONDA_ENV:-G}"
ENV_PYTHON="${ORIGIN_PYTHON:-${HOME}/.conda/envs/${ENV_NAME}/bin/python}"
[[ -x "${ENV_PYTHON}" ]] || {
  echo "Missing cluster Python environment: ${ENV_PYTHON}" >&2; exit 7;
}
export PATH="$(dirname "${ENV_PYTHON}"):${PATH}"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

cd "${REPO_ROOT}"
[[ "$(git rev-parse HEAD)" == "${LAUNCH_COMMIT}" ]] || {
  echo "Immutable package commit mismatch." >&2; exit 2;
}
[[ -z "$(git status --porcelain --untracked-files=no)" ]] || {
  echo "Tracked files changed in immutable package worktree." >&2; exit 3;
}
for required in \
  "${NUMERICAL}" "${DECODER}" "${ARTIFACTS}" "${IDRID}" \
  "${OOF}" "${SHORTCUT}" "${BASELINES}"; do
  [[ -f "${required}" ]] || {
    echo "Required acceptance artifact is missing: ${required}" >&2; exit 4;
  }
done
[[ ! -e "${OUTPUT_DIR}/ORIGIN_ACCEPTANCE_PACKAGE_MANIFEST.json" ]] || {
  echo "Acceptance package already exists; refusing overwrite." >&2; exit 5;
}

echo "=== ORIGIN fail-closed acceptance package assembly ==="
date --iso-8601=seconds
git rev-parse HEAD
"${ENV_PYTHON}" -m pytest -q \
  tests/test_origin_acceptance_package.py \
  tests/test_origin_idrid_semantics_v2.py

ARGS=(
  --repo-root "${REPO_ROOT}"
  --output-dir "${OUTPUT_DIR}"
  --numerical-audit "${NUMERICAL}"
  --decoder-contract-audit "${DECODER}"
  --artifact-manifest "${ARTIFACTS}"
  --idrid-manifest "${IDRID}"
  --oof-manifest "${OOF}"
  --shortcut-aggregate "${SHORTCUT}"
  --baseline-aggregate "${BASELINES}"
  --expected-commit "${LAUNCH_COMMIT}"
)
case "${REQUIRE_GATES}" in
  1|true|TRUE|yes|YES) ARGS+=(--require-gates-pass) ;;
  0|false|FALSE|no|NO) ;;
  *) echo "ORIGIN_REQUIRE_GATES_PASS must be a boolean." >&2; exit 6 ;;
esac
if [[ -n "${CHECKPOINT_ARCHIVE_URI}" || -n "${CHECKPOINT_ARCHIVE_SHA256}" || -n "${CHECKPOINT_ARCHIVE_BYTES}" ]]; then
  [[ -n "${CHECKPOINT_ARCHIVE_URI}" && -n "${CHECKPOINT_ARCHIVE_SHA256}" ]] || {
    echo "Checkpoint archive URI and SHA-256 must be supplied together." >&2; exit 8;
  }
  ARGS+=(
    --checkpoint-archive-uri "${CHECKPOINT_ARCHIVE_URI}"
    --checkpoint-archive-sha256 "${CHECKPOINT_ARCHIVE_SHA256}"
  )
  [[ -z "${CHECKPOINT_ARCHIVE_BYTES}" ]] || ARGS+=(
    --checkpoint-archive-bytes "${CHECKPOINT_ARCHIVE_BYTES}"
  )
fi
if [[ -n "${EVIDENCE_ARCHIVE_URI}" || -n "${EVIDENCE_ARCHIVE_SHA256}" || -n "${EVIDENCE_ARCHIVE_BYTES}" ]]; then
  [[ -n "${EVIDENCE_ARCHIVE_URI}" && -n "${EVIDENCE_ARCHIVE_SHA256}" ]] || {
    echo "Per-image evidence archive URI and SHA-256 must be supplied together." >&2; exit 9;
  }
  ARGS+=(
    --evidence-archive-uri "${EVIDENCE_ARCHIVE_URI}"
    --evidence-archive-sha256 "${EVIDENCE_ARCHIVE_SHA256}"
  )
  [[ -z "${EVIDENCE_ARCHIVE_BYTES}" ]] || ARGS+=(
    --evidence-archive-bytes "${EVIDENCE_ARCHIVE_BYTES}"
  )
fi

"${ENV_PYTHON}" tools/assemble_origin_acceptance_package.py "${ARGS[@]}"
sha256sum \
  "${OUTPUT_DIR}/ORIGIN_ACCEPTANCE_PACKAGE_MANIFEST.json" \
  "${OUTPUT_DIR}/ORIGIN_ACCEPTANCE_PACKAGE_COMPLETE.json"
