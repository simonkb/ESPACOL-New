#!/bin/bash
# CPU-only fail-closed preflight before the 84 GPU comparator workers fan out.
#SBATCH --job-name=origin_sc_cmp_check
#SBATCH --partition=prod
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=/dpc/kuin0170/ESPACOL-New/origin_shortcut_logs/comparator_preflight_%j.out
#SBATCH --error=/dpc/kuin0170/ESPACOL-New/origin_shortcut_logs/comparator_preflight_%j.err
#SBATCH --account=kuin0170

set -euo pipefail

ENV_NAME="${ORIGIN_CONDA_ENV:-G}"
ENV_PYTHON="${ORIGIN_PYTHON:-${HOME}/.conda/envs/${ENV_NAME}/bin/python}"
[[ -x "${ENV_PYTHON}" ]] || {
  echo "Conda environment Python is missing or not executable: ${ENV_PYTHON}" >&2
  exit 7
}
export PATH="$(dirname "${ENV_PYTHON}"):${PATH}"

REPO_ROOT="${ORIGIN_REPO_ROOT:?ORIGIN_REPO_ROOT is required}"
PROTOCOL="${ORIGIN_SHORTCUT_COMPARATOR_PROTOCOL:?ORIGIN_SHORTCUT_COMPARATOR_PROTOCOL is required}"
REFERENCE_ROOT="${ORIGIN_SHORTCUT_REFERENCE_ROOT:?ORIGIN_SHORTCUT_REFERENCE_ROOT is required}"
LAUNCH_COMMIT="${ORIGIN_LAUNCH_COMMIT:?ORIGIN_LAUNCH_COMMIT is required}"
cd "${REPO_ROOT}"
[[ "$(git rev-parse HEAD)" == "${LAUNCH_COMMIT}" ]] || exit 2
[[ -z "$(git status --porcelain --untracked-files=no)" ]] || exit 3

"${ENV_PYTHON}" - "${PROTOCOL}" "${REFERENCE_ROOT}" "${LAUNCH_COMMIT}" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

from scripts.origin_shortcut_comparator_common import (
    PROTOCOL_CORE_SHA256,
    TASK_COUNT,
    all_tasks,
)

protocol_path = Path(sys.argv[1]).resolve()
reference = Path(sys.argv[2]).resolve()
expected_commit = sys.argv[3]
payload = json.loads(protocol_path.read_text())
recorded = payload.pop("content_checksum_sha256", None)
observed = hashlib.sha256(
    json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
).hexdigest()
if recorded != observed:
    raise RuntimeError("comparator protocol checksum mismatch")
if payload.get("protocol_core_sha256") != PROTOCOL_CORE_SHA256:
    raise RuntimeError("comparator protocol core mismatch")
if payload.get("launch_commit") != expected_commit:
    raise RuntimeError("comparator launch commit mismatch")
if TASK_COUNT != 84 or len(all_tasks()) != 84:
    raise RuntimeError("comparator task registry is not exactly 84 workers")
for filename, key in (
    ("LOCKED_PROTOCOL.json", "origin_reference_protocol_sha256"),
    ("SUBMISSION.json", "origin_reference_submission_sha256"),
):
    path = reference / filename
    if not path.is_file():
        raise FileNotFoundError(path)
    if hashlib.sha256(path.read_bytes()).hexdigest() != payload.get(key):
        raise RuntimeError(f"sealed ORIGIN reference changed: {filename}")
print("Comparator protocol and sealed ORIGIN-v1 reference verified.")
PY

"${ENV_PYTHON}" -m pytest -q \
  tests/test_origin_shortcut_benchmark.py \
  tests/test_origin_acceptance_baselines.py
echo "Shortcut comparator v2 preflight passed."
