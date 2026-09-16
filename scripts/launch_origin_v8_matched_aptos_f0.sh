#!/bin/bash
# Submit the V8 preflight, three-member APTOS training array, corresponding
# full-validation audit array, and fail-closed comparison as one dependency
# chain. Run this from the cluster login node after pulling the intended
# commit. It does not submit outer-test or full-CV work.

set -euo pipefail

REPO_ROOT="${ORIGIN_REPO_ROOT:-/dpc/kuin0170/ESPACOL-New}"
cd "${REPO_ROOT}"
EXPECTED_COMMIT="${ORIGIN_EXPECTED_COMMIT:-$(git rev-parse HEAD)}"
[[ "$(git rev-parse HEAD)" == "${EXPECTED_COMMIT}" ]] || {
  echo "Current commit differs from ORIGIN_EXPECTED_COMMIT." >&2
  exit 2
}

SOURCE_CHECKPOINT="${ORIGIN_V3_CHECKPOINT:-runs/origin_aptos_f0_v3_bounded/fold0/best.pth}"
[[ -f "${SOURCE_CHECKPOINT}" ]] || {
  echo "Missing immutable APTOS v3 checkpoint: ${SOURCE_CHECKPOINT}" >&2
  exit 3
}
SOURCE_SHA="${ORIGIN_V3_CHECKPOINT_SHA256:-$(sha256sum "${SOURCE_CHECKPOINT}" | cut -d' ' -f1)}"
[[ "$(sha256sum "${SOURCE_CHECKPOINT}" | cut -d' ' -f1)" == "${SOURCE_SHA}" ]] || {
  echo "ORIGIN_V3_CHECKPOINT_SHA256 does not match ${SOURCE_CHECKPOINT}." >&2
  exit 4
}

EXPORTS="ALL,ORIGIN_EXPECTED_COMMIT=${EXPECTED_COMMIT},ORIGIN_V3_CHECKPOINT=${SOURCE_CHECKPOINT},ORIGIN_V3_CHECKPOINT_SHA256=${SOURCE_SHA}"
PREFLIGHT_JOB="$(
  sbatch --parsable --export="${EXPORTS}" \
    scripts/submit_origin_v8_matched_preflight.sh | cut -d';' -f1
)"
ARRAY_JOB="$(
  sbatch --parsable --dependency="afterok:${PREFLIGHT_JOB}" --export="${EXPORTS}" \
    scripts/submit_origin_v8_matched_aptos_f0.sh | cut -d';' -f1
)"
AUDIT_JOB="$(
  sbatch --parsable --dependency="afterok:${ARRAY_JOB}" --export="${EXPORTS}" \
    scripts/submit_origin_v8_matched_audit.sh | cut -d';' -f1
)"

echo "preflight_job=${PREFLIGHT_JOB}"
echo "matched_array_job=${ARRAY_JOB} (tasks 0=target, 1=additive endpoint, 2=shuffled geometry)"
echo "matched_audit_job=${AUDIT_JOB} (same task-to-arm mapping)"

if [[ "${ORIGIN_SUBMIT_COMPARISON:-1}" == "1" ]]; then
  COMPARISON_JOB="$(
    sbatch --parsable --dependency="afterok:${AUDIT_JOB}" --export="${EXPORTS}" \
      scripts/submit_origin_v8_matched_comparison.sh | cut -d';' -f1
  )"
  echo "comparison_job=${COMPARISON_JOB}"
fi
