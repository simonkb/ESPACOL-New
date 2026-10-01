#!/bin/bash
# Freeze and dependency-gate the final CPU-only acceptance package job.
#
# All seven upstream jobs are explicit so the package can never race a pending
# or failed audit.  Slurm afterok semantics prevent launch if any prerequisite
# fails.  Completed negative Gate A/B results still count as successful jobs;
# the assembler records them and withholds the corresponding claims.

set -euo pipefail

SOURCE_REPO="$(git rev-parse --show-toplevel)"
cd "${SOURCE_REPO}"
[[ "$(git branch --show-current)" == "origin-acceptance-revision" ]] || {
  echo "Expected origin-acceptance-revision." >&2; exit 2;
}
[[ -z "$(git status --porcelain --untracked-files=no)" ]] || {
  echo "Commit tracked changes before freezing package assembly." >&2; exit 3;
}
command -v sbatch >/dev/null || { echo "sbatch is unavailable." >&2; exit 4; }

LAUNCH_COMMIT="$(git rev-parse HEAD)"
SHORT_COMMIT="${LAUNCH_COMMIT:0:12}"
SNAPSHOT_ROOT="${ORIGIN_PACKAGE_WORKTREE:-${SOURCE_REPO}-origin-package-${SHORT_COMMIT}}"
OUTPUT_DIR="${ORIGIN_PACKAGE_OUTPUT:?ORIGIN_PACKAGE_OUTPUT is required}"

# Authoritative artifact locations.
NUMERICAL="${ORIGIN_NUMERICAL_AUDIT:?ORIGIN_NUMERICAL_AUDIT is required}"
DECODER="${ORIGIN_DECODER_CONTRACT_AUDIT:?ORIGIN_DECODER_CONTRACT_AUDIT is required}"
ARTIFACTS="${ORIGIN_ARTIFACT_MANIFEST:?ORIGIN_ARTIFACT_MANIFEST is required}"
IDRID="${ORIGIN_IDRID_STATISTICS_MANIFEST:?ORIGIN_IDRID_STATISTICS_MANIFEST is required}"
OOF="${ORIGIN_OOF_STATISTICS_MANIFEST:?ORIGIN_OOF_STATISTICS_MANIFEST is required}"
SHORTCUT="${ORIGIN_SHORTCUT_AGGREGATE:?ORIGIN_SHORTCUT_AGGREGATE is required}"
BASELINES="${ORIGIN_BASELINE_AGGREGATE:?ORIGIN_BASELINE_AGGREGATE is required}"

# Matching upstream job IDs.  afterok is the execution barrier; the assembler
# independently revalidates every file and checksum after the barrier opens.
NUMERICAL_JOB="${ORIGIN_NUMERICAL_AUDIT_JOB:?ORIGIN_NUMERICAL_AUDIT_JOB is required}"
DECODER_JOB="${ORIGIN_DECODER_CONTRACT_JOB:?ORIGIN_DECODER_CONTRACT_JOB is required}"
ARTIFACT_JOB="${ORIGIN_ARTIFACT_MANIFEST_JOB:?ORIGIN_ARTIFACT_MANIFEST_JOB is required}"
IDRID_JOB="${ORIGIN_IDRID_STATISTICS_JOB:?ORIGIN_IDRID_STATISTICS_JOB is required}"
OOF_JOB="${ORIGIN_OOF_GATE_A_JOB:?ORIGIN_OOF_GATE_A_JOB is required}"
SHORTCUT_JOB="${ORIGIN_SHORTCUT_GATE_B_JOB:?ORIGIN_SHORTCUT_GATE_B_JOB is required}"
BASELINE_JOB="${ORIGIN_BASELINE_AGGREGATE_JOB:?ORIGIN_BASELINE_AGGREGATE_JOB is required}"

for job in \
  "${NUMERICAL_JOB}" "${DECODER_JOB}" "${ARTIFACT_JOB}" "${IDRID_JOB}" \
  "${OOF_JOB}" "${SHORTCUT_JOB}" "${BASELINE_JOB}"; do
  [[ "${job}" =~ ^[0-9]+$ ]] || { echo "Invalid Slurm job ID: ${job}" >&2; exit 5; }
done
[[ ! -e "${OUTPUT_DIR}" ]] || {
  echo "Fresh package output already exists: ${OUTPUT_DIR}" >&2; exit 6;
}

if [[ -e "${SNAPSHOT_ROOT}" ]]; then
  [[ "$(git -C "${SNAPSHOT_ROOT}" rev-parse HEAD)" == "${LAUNCH_COMMIT}" ]] || exit 7
  [[ -z "$(git -C "${SNAPSHOT_ROOT}" status --porcelain --untracked-files=no)" ]] || exit 8
else
  git worktree add --detach "${SNAPSHOT_ROOT}" "${LAUNCH_COMMIT}"
fi

DEPENDENCY="afterok:${NUMERICAL_JOB}:${DECODER_JOB}:${ARTIFACT_JOB}:${IDRID_JOB}:${OOF_JOB}:${SHORTCUT_JOB}:${BASELINE_JOB}"
EXPORTS="ALL,ORIGIN_REPO_ROOT=${SNAPSHOT_ROOT},ORIGIN_LAUNCH_COMMIT=${LAUNCH_COMMIT},ORIGIN_PACKAGE_OUTPUT=${OUTPUT_DIR},ORIGIN_NUMERICAL_AUDIT=${NUMERICAL},ORIGIN_DECODER_CONTRACT_AUDIT=${DECODER},ORIGIN_ARTIFACT_MANIFEST=${ARTIFACTS},ORIGIN_IDRID_STATISTICS_MANIFEST=${IDRID},ORIGIN_OOF_STATISTICS_MANIFEST=${OOF},ORIGIN_SHORTCUT_AGGREGATE=${SHORTCUT},ORIGIN_BASELINE_AGGREGATE=${BASELINES},ORIGIN_REQUIRE_GATES_PASS=${ORIGIN_REQUIRE_GATES_PASS:-0},ORIGIN_CONDA_ENV=${ORIGIN_CONDA_ENV:-G},ORIGIN_CHECKPOINT_ARCHIVE_URI=${ORIGIN_CHECKPOINT_ARCHIVE_URI:-},ORIGIN_CHECKPOINT_ARCHIVE_SHA256=${ORIGIN_CHECKPOINT_ARCHIVE_SHA256:-},ORIGIN_CHECKPOINT_ARCHIVE_BYTES=${ORIGIN_CHECKPOINT_ARCHIVE_BYTES:-},ORIGIN_EVIDENCE_ARCHIVE_URI=${ORIGIN_EVIDENCE_ARCHIVE_URI:-},ORIGIN_EVIDENCE_ARCHIVE_SHA256=${ORIGIN_EVIDENCE_ARCHIVE_SHA256:-},ORIGIN_EVIDENCE_ARCHIVE_BYTES=${ORIGIN_EVIDENCE_ARCHIVE_BYTES:-}"

PACKAGE_JOB="$(sbatch --parsable --dependency="${DEPENDENCY}" \
  --export="${EXPORTS}" \
  "${SNAPSHOT_ROOT}/scripts/submit_origin_acceptance_package.sh" | cut -d';' -f1)"

echo "ORIGIN acceptance package job submitted."
echo "  commit:      ${LAUNCH_COMMIT}"
echo "  dependency:  ${DEPENDENCY}"
echo "  package job: ${PACKAGE_JOB}"
echo "  output:      ${OUTPUT_DIR}"
