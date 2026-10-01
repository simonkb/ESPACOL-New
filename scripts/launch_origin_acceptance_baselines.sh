#!/bin/bash
# Submit canary -> audits -> full training -> one coordinated outer release.
set -euo pipefail
: "${ORIGIN_REPO_ROOT:?ORIGIN_REPO_ROOT is required}"
: "${ORIGIN_ACCEPTANCE_ROOT:?ORIGIN_ACCEPTANCE_ROOT is required}"
: "${ORIGIN_APTOS_ROOT:?ORIGIN_APTOS_ROOT is required}"
: "${ORIGIN_DR_ROOT:?ORIGIN_DR_ROOT is required}"
: "${ORIGIN_LAUNCH_COMMIT:?ORIGIN_LAUNCH_COMMIT is required}"
mkdir -p /dpc/kuin0170/ESPACOL-New/origin_acceptance_logs

CANARY_JOB="$(sbatch --parsable scripts/submit_origin_acceptance_aptos_f0_canary.sh)"
CANARY_AUDIT_JOB="$(sbatch --parsable --dependency="afterok:${CANARY_JOB}" scripts/submit_origin_acceptance_canary_audit.sh)"
FULL_JOB="$(sbatch --parsable --dependency="afterok:${CANARY_AUDIT_JOB}" scripts/submit_origin_acceptance_full_array.sh)"
FULL_AUDIT_JOB="$(sbatch --parsable --dependency="afterok:${FULL_JOB}" scripts/submit_origin_acceptance_full_audit.sh)"
RELEASE_JOB="$(sbatch --parsable --dependency="afterok:${FULL_AUDIT_JOB}" scripts/submit_origin_acceptance_outer_release.sh)"
AGGREGATE_JOB="$(sbatch --parsable --dependency="afterok:${RELEASE_JOB}" scripts/submit_origin_acceptance_release_aggregate.sh)"
printf 'canary=%s\ncanary_audit=%s\nfull=%s\nfull_audit=%s\nrelease=%s\naggregate=%s\n' \
  "${CANARY_JOB}" "${CANARY_AUDIT_JOB}" "${FULL_JOB}" "${FULL_AUDIT_JOB}" \
  "${RELEASE_JOB}" "${AGGREGATE_JOB}"
