#!/bin/bash
# Freeze and dependency-gate the final CPU-only acceptance package job.
#
# This launcher deliberately does not put already-completed jobs in a new
# Slurm dependency expression. On clusters with a short MinJobAge, afterok on
# an old completed job can become DependencyNeverSatisfied even though the
# corresponding artifact is valid. Each job is classified from squeue/sacct:
# live jobs remain afterok prerequisites, completed jobs must already have a
# non-empty artifact, and every failure/unknown state aborts.

set -euo pipefail

SOURCE_REPO="$(git rev-parse --show-toplevel)"
cd "${SOURCE_REPO}"
[[ "$(git branch --show-current)" == "origin-acceptance-revision" ]] || {
  echo "Expected origin-acceptance-revision." >&2; exit 2;
}
[[ -z "$(git status --porcelain --untracked-files=no)" ]] || {
  echo "Commit tracked changes before freezing package assembly." >&2; exit 3;
}
for command_name in sbatch squeue sacct scontrol scancel git python sha256sum; do
  command -v "${command_name}" >/dev/null || {
    echo "${command_name} is unavailable." >&2; exit 4;
  }
done

LAUNCH_COMMIT="$(git rev-parse HEAD)"
SHORT_COMMIT="${LAUNCH_COMMIT:0:12}"
SNAPSHOT_ROOT="${ORIGIN_PACKAGE_WORKTREE:-${SOURCE_REPO}-origin-package-${SHORT_COMMIT}}"
OUTPUT_DIR="${ORIGIN_PACKAGE_OUTPUT:?ORIGIN_PACKAGE_OUTPUT is required}"
SUBMISSION_RECORD="${OUTPUT_DIR}.SUBMISSION.json"
RESERVATION_DIR="${OUTPUT_DIR}.launch-reservation"
LAUNCHER_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"

# Authoritative artifact locations and matching producer job IDs. The IDRiD
# entry is the corrective bidirectionally matched v2 statistics manifest; a
# legacy v1 manifest will pass the scheduler barrier but fail closed in the
# immutable package worker before any release is written. The package
# worker independently revalidates contents/checksums after all live barriers
# open; this launcher establishes scheduler state and artifact existence.
UPSTREAM_LABELS=(numerical decoder artifacts idrid oof shortcut baselines)
UPSTREAM_ARTIFACTS=(
  "${ORIGIN_NUMERICAL_AUDIT:?ORIGIN_NUMERICAL_AUDIT is required}"
  "${ORIGIN_DECODER_CONTRACT_AUDIT:?ORIGIN_DECODER_CONTRACT_AUDIT is required}"
  "${ORIGIN_ARTIFACT_MANIFEST:?ORIGIN_ARTIFACT_MANIFEST is required}"
  "${ORIGIN_IDRID_STATISTICS_MANIFEST:?ORIGIN_IDRID_STATISTICS_MANIFEST is required}"
  "${ORIGIN_OOF_STATISTICS_MANIFEST:?ORIGIN_OOF_STATISTICS_MANIFEST is required}"
  "${ORIGIN_SHORTCUT_AGGREGATE:?ORIGIN_SHORTCUT_AGGREGATE is required}"
  "${ORIGIN_BASELINE_AGGREGATE:?ORIGIN_BASELINE_AGGREGATE is required}"
)
UPSTREAM_JOBS=(
  "${ORIGIN_NUMERICAL_AUDIT_JOB:?ORIGIN_NUMERICAL_AUDIT_JOB is required}"
  "${ORIGIN_DECODER_CONTRACT_JOB:?ORIGIN_DECODER_CONTRACT_JOB is required}"
  "${ORIGIN_ARTIFACT_MANIFEST_JOB:?ORIGIN_ARTIFACT_MANIFEST_JOB is required}"
  "${ORIGIN_IDRID_STATISTICS_JOB:?ORIGIN_IDRID_STATISTICS_JOB is required}"
  "${ORIGIN_OOF_GATE_A_JOB:?ORIGIN_OOF_GATE_A_JOB is required}"
  "${ORIGIN_SHORTCUT_GATE_B_JOB:?ORIGIN_SHORTCUT_GATE_B_JOB is required}"
  "${ORIGIN_BASELINE_AGGREGATE_JOB:?ORIGIN_BASELINE_AGGREGATE_JOB is required}"
)

normalize_state() {
  local value="$1"
  value="${value%% *}"
  value="${value%%+}"
  printf '%s' "${value}"
}

state_class() {
  case "$1" in
    PENDING|CONFIGURING|RUNNING|COMPLETING|SUSPENDED|RESIZING|REQUEUED|REQUEUE_FED|REQUEUE_HOLD|SIGNALING|STAGE_OUT|STOPPED)
      printf 'live' ;;
    COMPLETED)
      printf 'completed' ;;
    BOOT_FAIL|CANCELLED|DEADLINE|FAILED|NODE_FAIL|OUT_OF_MEMORY|PREEMPTED|REVOKED|SPECIAL_EXIT|TIMEOUT)
      printf 'failure' ;;
    *)
      printf 'unknown' ;;
  esac
}

# Print CLASS|NORMALIZED_STATE. squeue is authoritative while a job is live;
# sacct is consulted only after the job disappears from the queue. Exact raw
# job IDs prevent a successful .batch step from masking a failed allocation.
query_job() {
  local job="$1"
  local queue_output accounting_output raw_state normalized classification
  # Slurm returns a non-zero status ("Invalid job id specified") once a job
  # ages out of squeue, even while its durable allocation record remains in
  # sacct. Treat that exactly like an empty queue lookup and fall through to
  # the exact JobIDRaw accounting check below. A real scheduler outage still
  # fails closed unless sacct can independently establish the job state.
  if ! queue_output="$(squeue -h -j "${job}" -o '%T' 2>&1)"; then
    echo "squeue has no usable record for job ${job}; consulting sacct: ${queue_output}" >&2
    queue_output=""
  fi
  if [[ -n "${queue_output//[[:space:]]/}" ]]; then
    normalized=""
    while IFS= read -r raw_state; do
      [[ -n "${raw_state//[[:space:]]/}" ]] || continue
      raw_state="$(normalize_state "${raw_state}")"
      classification="$(state_class "${raw_state}")"
      [[ "${classification}" == "live" ]] || {
        echo "Queued job ${job} has non-live state ${raw_state}." >&2; return 91;
      }
      if [[ -z "${normalized}" ]]; then
        normalized="${raw_state}"
      elif [[ ",${normalized}," != *",${raw_state},"* ]]; then
        normalized="${normalized},${raw_state}"
      fi
    done <<< "${queue_output}"
    [[ -n "${normalized}" ]] || {
      echo "squeue returned no usable state for job ${job}." >&2; return 92;
    }
    printf 'live|%s\n' "${normalized}"
    return 0
  fi

  accounting_output="$(sacct -n -X -P -j "${job}" --format=JobIDRaw,State)" || {
    echo "sacct failed while inspecting job ${job}." >&2; return 93;
  }
  raw_state=""
  while IFS='|' read -r candidate_id candidate_state _; do
    [[ "${candidate_id//[[:space:]]/}" == "${job}" ]] || continue
    [[ -z "${raw_state}" ]] || {
      echo "sacct returned multiple allocation records for job ${job}." >&2
      return 94
    }
    raw_state="${candidate_state}"
  done <<< "${accounting_output}"
  [[ -n "${raw_state//[[:space:]]/}" ]] || {
    echo "No exact scheduler record exists for job ${job}." >&2; return 95;
  }
  normalized="$(normalize_state "${raw_state}")"
  classification="$(state_class "${normalized}")"
  printf '%s|%s\n' "${classification}" "${normalized}"
}

for job in "${UPSTREAM_JOBS[@]}"; do
  [[ "${job}" =~ ^[0-9]+$ ]] || {
    echo "Invalid Slurm job ID: ${job}" >&2; exit 5;
  }
done
[[ ! -e "${OUTPUT_DIR}" ]] || {
  echo "Fresh package output already exists: ${OUTPUT_DIR}" >&2; exit 6;
}
[[ ! -e "${SUBMISSION_RECORD}" ]] || {
  echo "Package submission record already exists: ${SUBMISSION_RECORD}" >&2; exit 7;
}

mkdir -p "$(dirname "${OUTPUT_DIR}")" "$(dirname "${SUBMISSION_RECORD}")"
mkdir "${RESERVATION_DIR}" 2>/dev/null || {
  echo "Another package launch already reserved: ${RESERVATION_DIR}" >&2; exit 8;
}

STATUS_FILE="$(mktemp "${TMPDIR:-/tmp}/origin-package-upstreams.XXXXXX")"
SUPERSEDED_FILE="$(mktemp "${TMPDIR:-/tmp}/origin-package-superseded.XXXXXX")"
PACKAGE_JOB=""
LAUNCH_COMMITTED=0
cleanup_launch() {
  cleanup_status=$?
  rm -f "${STATUS_FILE}" "${SUPERSEDED_FILE}"
  if [[ "${LAUNCH_COMMITTED}" != "1" ]]; then
    if [[ -n "${PACKAGE_JOB}" ]]; then
      scancel "${PACKAGE_JOB}" >/dev/null 2>&1 || true
    fi
    rmdir "${RESERVATION_DIR}" >/dev/null 2>&1 || true
  fi
  exit "${cleanup_status}"
}
trap cleanup_launch EXIT

LIVE_JOBS=()
for index in 0 1 2 3 4 5 6; do
  label="${UPSTREAM_LABELS[$index]}"
  job="${UPSTREAM_JOBS[$index]}"
  artifact="${UPSTREAM_ARTIFACTS[$index]}"
  result="$(query_job "${job}")" || exit $?
  classification="${result%%|*}"
  scheduler_state="${result#*|}"
  case "${classification}" in
    live)
      LIVE_JOBS+=("${job}")
      ;;
    completed)
      [[ -s "${artifact}" ]] || {
        echo "Completed upstream ${label} job ${job} has no non-empty artifact: ${artifact}" >&2
        exit 9
      }
      ;;
    failure)
      echo "Upstream ${label} job ${job} failed with state ${scheduler_state}." >&2
      exit 10
      ;;
    *)
      echo "Upstream ${label} job ${job} has unsupported state ${scheduler_state}." >&2
      exit 11
      ;;
  esac
  printf '%s\t%s\t%s\t%s\t%s\n' \
    "${label}" "${job}" "${artifact}" "${classification}" "${scheduler_state}" \
    >> "${STATUS_FILE}"
done

# Recovery launches may name package jobs that they supersede. They must have
# reached CANCELLED and disappeared from squeue before a replacement is
# admitted, which prevents the old worker from later writing a competing tree.
SUPERSEDED_RAW="${ORIGIN_SUPERSEDED_PACKAGE_JOBS:-}"
SUPERSEDED_RAW="${SUPERSEDED_RAW//,/ }"
SUPERSEDED_RAW="${SUPERSEDED_RAW//:/ }"
if [[ -n "${SUPERSEDED_RAW// /}" ]]; then
  read -r -a SUPERSEDED_JOBS <<< "${SUPERSEDED_RAW}"
  for job in "${SUPERSEDED_JOBS[@]}"; do
    [[ -n "${job}" ]] || continue
    [[ "${job}" =~ ^[0-9]+$ ]] || {
      echo "Invalid superseded package job ID: ${job}" >&2; exit 12;
    }
    result="$(query_job "${job}")" || exit $?
    classification="${result%%|*}"
    scheduler_state="${result#*|}"
    [[ "${classification}" == "failure" && "${scheduler_state}" == "CANCELLED" ]] || {
      echo "Superseded package job ${job} is not terminal CANCELLED (state=${scheduler_state})." >&2
      exit 13
    }
    printf '%s\t%s\t%s\n' "${job}" "${classification}" "${scheduler_state}" \
      >> "${SUPERSEDED_FILE}"
  done
fi

if [[ -e "${SNAPSHOT_ROOT}" ]]; then
  git -C "${SNAPSHOT_ROOT}" rev-parse --is-inside-work-tree >/dev/null 2>&1 || {
    echo "Existing snapshot is not a Git worktree: ${SNAPSHOT_ROOT}" >&2; exit 14;
  }
  [[ "$(git -C "${SNAPSHOT_ROOT}" rev-parse HEAD)" == "${LAUNCH_COMMIT}" ]] || {
    echo "Existing package snapshot has the wrong commit." >&2; exit 15;
  }
  [[ -z "$(git -C "${SNAPSHOT_ROOT}" branch --show-current)" ]] || {
    echo "Existing package snapshot is not detached." >&2; exit 16;
  }
  [[ -z "$(git -C "${SNAPSHOT_ROOT}" status --porcelain --untracked-files=no)" ]] || {
    echo "Existing package snapshot contains tracked changes." >&2; exit 17;
  }
else
  git worktree add --detach "${SNAPSHOT_ROOT}" "${LAUNCH_COMMIT}"
fi

DEPENDENCY=""
if [[ "${#LIVE_JOBS[@]}" -gt 0 ]]; then
  DEPENDENCY="afterok:$(IFS=:; echo "${LIVE_JOBS[*]}")"
fi
EXPORTS="ALL,ORIGIN_REPO_ROOT=${SNAPSHOT_ROOT},ORIGIN_LAUNCH_COMMIT=${LAUNCH_COMMIT},ORIGIN_PACKAGE_OUTPUT=${OUTPUT_DIR},ORIGIN_NUMERICAL_AUDIT=${UPSTREAM_ARTIFACTS[0]},ORIGIN_DECODER_CONTRACT_AUDIT=${UPSTREAM_ARTIFACTS[1]},ORIGIN_ARTIFACT_MANIFEST=${UPSTREAM_ARTIFACTS[2]},ORIGIN_IDRID_STATISTICS_MANIFEST=${UPSTREAM_ARTIFACTS[3]},ORIGIN_OOF_STATISTICS_MANIFEST=${UPSTREAM_ARTIFACTS[4]},ORIGIN_SHORTCUT_AGGREGATE=${UPSTREAM_ARTIFACTS[5]},ORIGIN_BASELINE_AGGREGATE=${UPSTREAM_ARTIFACTS[6]},ORIGIN_REQUIRE_GATES_PASS=${ORIGIN_REQUIRE_GATES_PASS:-0},ORIGIN_CONDA_ENV=${ORIGIN_CONDA_ENV:-G},ORIGIN_CHECKPOINT_ARCHIVE_URI=${ORIGIN_CHECKPOINT_ARCHIVE_URI:-},ORIGIN_CHECKPOINT_ARCHIVE_SHA256=${ORIGIN_CHECKPOINT_ARCHIVE_SHA256:-},ORIGIN_CHECKPOINT_ARCHIVE_BYTES=${ORIGIN_CHECKPOINT_ARCHIVE_BYTES:-},ORIGIN_EVIDENCE_ARCHIVE_URI=${ORIGIN_EVIDENCE_ARCHIVE_URI:-},ORIGIN_EVIDENCE_ARCHIVE_SHA256=${ORIGIN_EVIDENCE_ARCHIVE_SHA256:-},ORIGIN_EVIDENCE_ARCHIVE_BYTES=${ORIGIN_EVIDENCE_ARCHIVE_BYTES:-}"

# Hold the new job until its immutable, checksummed submission record exists.
# If recording or releasing fails, the EXIT trap cancels the held job.
SBATCH_ARGS=(--parsable --hold --export="${EXPORTS}")
[[ -z "${DEPENDENCY}" ]] || SBATCH_ARGS+=(--dependency="${DEPENDENCY}")
PACKAGE_JOB="$(sbatch "${SBATCH_ARGS[@]}" \
  "${SNAPSHOT_ROOT}/scripts/submit_origin_acceptance_package.sh" | cut -d';' -f1)"
[[ "${PACKAGE_JOB}" =~ ^[0-9]+$ ]] || {
  echo "sbatch returned an invalid package job ID: ${PACKAGE_JOB}" >&2; exit 18;
}

ORIGIN_RECORD_OUTPUT="${SUBMISSION_RECORD}" \
ORIGIN_RECORD_STATUS="${STATUS_FILE}" \
ORIGIN_RECORD_SUPERSEDED="${SUPERSEDED_FILE}" \
ORIGIN_RECORD_COMMIT="${LAUNCH_COMMIT}" \
ORIGIN_RECORD_SOURCE="${SOURCE_REPO}" \
ORIGIN_RECORD_SNAPSHOT="${SNAPSHOT_ROOT}" \
ORIGIN_RECORD_PACKAGE_OUTPUT="${OUTPUT_DIR}" \
ORIGIN_RECORD_RESERVATION="${RESERVATION_DIR}" \
ORIGIN_RECORD_PACKAGE_JOB="${PACKAGE_JOB}" \
ORIGIN_RECORD_DEPENDENCY="${DEPENDENCY}" \
ORIGIN_RECORD_LAUNCHER="${LAUNCHER_PATH}" \
ORIGIN_RECORD_WORKER="${SNAPSHOT_ROOT}/scripts/submit_origin_acceptance_package.sh" \
python - <<'PY'
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


upstreams = []
with Path(os.environ["ORIGIN_RECORD_STATUS"]).open(encoding="utf-8") as stream:
    for line in stream:
        label, job_id, artifact_raw, state_class_value, scheduler_state = line.rstrip("\n").split("\t")
        artifact = Path(artifact_raw)
        exists = artifact.is_file()
        size = artifact.stat().st_size if exists else None
        upstreams.append({
            "label": label,
            "job_id": job_id,
            "scheduler_state": scheduler_state,
            "state_class": state_class_value,
            "included_in_afterok": state_class_value == "live",
            "artifact": {
                "path": str(artifact),
                "exists_at_launch": exists,
                "bytes_at_launch": size,
                "sha256_at_launch": file_sha256(artifact) if exists and size else None,
            },
        })

superseded = []
with Path(os.environ["ORIGIN_RECORD_SUPERSEDED"]).open(encoding="utf-8") as stream:
    for line in stream:
        job_id, state_class_value, scheduler_state = line.rstrip("\n").split("\t")
        superseded.append({
            "job_id": job_id,
            "state_class": state_class_value,
            "scheduler_state": scheduler_state,
        })

launcher = Path(os.environ["ORIGIN_RECORD_LAUNCHER"])
worker = Path(os.environ["ORIGIN_RECORD_WORKER"])
payload = {
    "schema": "origin-acceptance-package-submission-v2",
    "created_at_utc": datetime.now(timezone.utc).isoformat(),
    "launch_commit": os.environ["ORIGIN_RECORD_COMMIT"],
    "source_repository": os.environ["ORIGIN_RECORD_SOURCE"],
    "immutable_worktree": os.environ["ORIGIN_RECORD_SNAPSHOT"],
    "package_output": os.environ["ORIGIN_RECORD_PACKAGE_OUTPUT"],
    "launch_reservation": os.environ["ORIGIN_RECORD_RESERVATION"],
    "package_job_id": os.environ["ORIGIN_RECORD_PACKAGE_JOB"],
    "dependency_expression": os.environ["ORIGIN_RECORD_DEPENDENCY"] or None,
    "submission_control": "sbatch_hold_record_then_scontrol_release",
    "launcher_sha256": file_sha256(launcher),
    "worker_sha256": file_sha256(worker),
    "upstreams": upstreams,
    "superseded_package_jobs": superseded,
}
payload["content_checksum_sha256"] = canonical_sha256(payload)
destination = Path(os.environ["ORIGIN_RECORD_OUTPUT"])
temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
with temporary.open("x", encoding="utf-8") as stream:
    json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
    stream.write("\n")
os.replace(temporary, destination)
PY

scontrol release "${PACKAGE_JOB}"
LAUNCH_COMMITTED=1
rm -f "${STATUS_FILE}" "${SUPERSEDED_FILE}"
trap - EXIT

echo "ORIGIN acceptance package job submitted from a held, provenance-sealed launch."
echo "  commit:             ${LAUNCH_COMMIT}"
echo "  dependency:         ${DEPENDENCY:-none (all upstream artifacts complete)}"
echo "  package job:        ${PACKAGE_JOB}"
echo "  output:             ${OUTPUT_DIR}"
echo "  submission record:  ${SUBMISSION_RECORD}"
echo "  launch reservation: ${RESERVATION_DIR}"
