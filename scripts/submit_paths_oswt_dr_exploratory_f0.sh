#!/bin/bash
# Exploratory EyePACS fold 0 after the registered APTOS gate failed.
#
# This job is intentionally separate from submit_paths_oswt_dr_f0.sh.  It does
# not represent prospective promotion and must not be reported as such.
#SBATCH --job-name=oswt_xdr0
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=/dpc/kuin0170/ESPACOL-New/paths_oswt_exploratory_dr_f0_%j.out
#SBATCH --error=/dpc/kuin0170/ESPACOL-New/paths_oswt_exploratory_dr_f0_%j.err
#SBATCH --account=kuin0170

set -eo pipefail
set +u
source /etc/profile.d/lmod.sh || exit 1
module load miniconda/3 || exit 1
module load cuda/12.6 || exit 1
source activate "${ORIGIN_CONDA_ENV:-G}" || exit 1
set -u

REPO_ROOT="$(realpath -e -- "${ORIGIN_REPO_ROOT:-/dpc/kuin0170/ESPACOL-New}")"
cd "${REPO_ROOT}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

EXPLORATORY_ACK="I_ACKNOWLEDGE_APTOS_GATE_FAILED_AND_THIS_EYEPACS_RUN_IS_EXPLORATORY_ONLY"
[[ "${PATHS_OSWT_EXPLORATORY_ACK:-}" == "${EXPLORATORY_ACK}" ]] || {
  echo "Refusing launch: export PATHS_OSWT_EXPLORATORY_ACK=${EXPLORATORY_ACK}" >&2
  exit 2
}

# These are the exact prediction/training files covered by the implementation
# signature stored in the completed APTOS gate.  The exploratory launcher is
# additionally bound to its own committed bytes.
TRACKED_FILES=(
  configs/paths_oswt_config.py configs/paths_config.py configs/origin_config.py
  Datasets/origin_data.py Datasets/mosaic_data.py Datasets/dataloaders.py
  models/origin_encoder.py models/origin.py models/paths.py models/paths_oswt.py
  losses/origin.py losses/paths.py
  training/origin_trainer.py training/paths_trainer.py training/paths_oswt_trainer.py
  train_paths_oswt.py train_origin.py utils/spatial_mask.py
  scripts/submit_paths_oswt_preflight.sh scripts/submit_paths_oswt_aptos_f0.sh
  scripts/submit_paths_oswt_aptos_ungated_f0.sh
  scripts/submit_paths_oswt_promotion_gate.sh scripts/submit_paths_oswt_dr_f0.sh
  scripts/launch_paths_oswt_v8_aptos_f0.sh
  scripts/submit_paths_oswt_dr_exploratory_f0.sh
  scripts/launch_paths_oswt_v8_dr_exploratory_f0.sh
)
for path in "${TRACKED_FILES[@]}"; do
  [[ "$(git rev-parse "HEAD:${path}")" == "$(git hash-object "${path}")" ]] || {
    echo "Tracked implementation differs from HEAD: ${path}" >&2
    exit 3
  }
done

: "${PATHS_OSWT_DR_V3_SHA256:?export the audited EyePACS V3 SHA-256}"
V3="${PATHS_OSWT_DR_V3_CHECKPOINT:-runs/origin_dr_f0_v3_bounded/fold0/best.pth}"
[[ -f "${V3}" ]] || { echo "EyePACS V3 checkpoint is absent: ${V3}" >&2; exit 4; }
[[ "$(sha256sum "${V3}" | awk '{print $1}')" == "${PATHS_OSWT_DR_V3_SHA256}" ]] || {
  echo "EyePACS V3 SHA-256 mismatch." >&2
  exit 4
}

FAILED_GATE="${PATHS_OSWT_APTOS_GATE_RESULT:-runs/paths_oswt_aptos_f0_v8_gated/fold0/PROMOTED_TO_EYEPACS.json}"
[[ -f "${FAILED_GATE}" ]] || {
  echo "The completed APTOS gate report is absent: ${FAILED_GATE}" >&2
  exit 5
}
EXPECTED_FAILED_GATE_FILE_SHA256="47c12db8ebf448e7ee318b4a532b96065de77ddd0a532b9b87ea87fa69de851b"
[[ "$(sha256sum "${FAILED_GATE}" | awk '{print $1}')" == "${EXPECTED_FAILED_GATE_FILE_SHA256}" ]] || {
  echo "APTOS gate is not the exact audited failed-gate artifact." >&2
  exit 5
}

RUNS_ROOT="$(realpath -m -- "${REPO_ROOT}/runs")"
RUN_ROOT="$(realpath -m -- "${PATHS_OSWT_DR_EXPLORATORY_RUN_DIR:-${RUNS_ROOT}/paths_oswt_dr_f0_v8_exploratory_after_failed_aptos_gate}")"
[[ "${RUN_ROOT}" == "${RUNS_ROOT}/"* ]] || {
  echo "Exploratory run_dir must resolve strictly beneath ${RUNS_ROOT}." >&2
  exit 6
}
[[ "${RUN_ROOT}" == *"exploratory_after_failed_aptos"* ]] || {
  echo "Exploratory run_dir must contain 'exploratory_after_failed_aptos'." >&2
  exit 6
}
for forbidden in \
  "${RUNS_ROOT}/paths_oswt_dr_f0_v8_gated" \
  "${RUNS_ROOT}/paths_oswt_aptos_f0_v8_gated" \
  "${RUNS_ROOT}/paths_oswt_aptos_f0_v8_ungated"; do
  forbidden="$(realpath -m -- "${forbidden}")"
  [[ "${RUN_ROOT}" != "${forbidden}" && "${RUN_ROOT}" != "${forbidden}/"* ]] || {
    echo "Refusing registered treatment/control run_dir: ${RUN_ROOT}" >&2
    exit 6
  }
done
[[ "${PATHS_OSWT_EXPLORATORY_RESUME:-0}" == "0" ]] || {
  echo "Resume is disabled for this one-off exploratory experiment." >&2
  exit 6
}
if [[ -e "${RUN_ROOT}" ]]; then
  echo "Exploratory run already exists; choose a fresh run_dir." >&2
  exit 6
fi

TRAIN_ARGS=(
  python train_paths_oswt.py
  --dataset dr
  --data_root "${PATHS_OSWT_DR_DATA_ROOT:-${REPO_ROOT}/Datasets/DR}"
  --run_dir "${RUN_ROOT}"
  --folds 0 --seed 42
  --v3_checkpoint "${V3}" --v3_sha256 "${PATHS_OSWT_DR_V3_SHA256}"
  --oswt_variant shell_warranted
  --image_size 640 --encoder convnext_tiny --scales s4,s8,s16,s32
  --projection_dim 128 --reference_count 4096
  --atom_rate_init 1e-6 --prior_rate_init 1e-4 --boundary_scale_init 1.0
  --total_rate_cap 64 --prior_rate_cap 1 --boundary_scale_cap 2
  --rate_roundoff_margin 1 --decision_rule class_map
  --pgf_probes 0.05,0.20,0.50,0.80
  --oswt_beta_init 0.1 --oswt_tau_init 0.05
  --oswt_beta_cap 3.0 --oswt_tau_floor 1e-3 --oswt_tau_cap 0.5
  --oswt_strength 1.0
  --risk_set_alpha 0.5 --rps_weight 0.25
  --batch_size 8 --epochs 35 --num_workers 8
  --paths_refiner_lr 5e-4 --weight_decay 0
  --lr_factor 0.2 --lr_patience 5 --early_stopping_patience 12
  --certificate_samples 8 --certificate_shortlist_size 8
  --skip_test
)
printf -v PATHS_OSWT_EXPLORATORY_TRAIN_COMMAND '%q ' "${TRAIN_ARGS[@]}"
export PATHS_OSWT_EXPLORATORY_TRAIN_COMMAND

# Validate and record the failed prospective gate.  The implementation-byte
# signature must still match even though this launcher lives in a later commit.
python - "${FAILED_GATE}" "${RUN_ROOT}" "${V3}" "${PATHS_OSWT_DR_V3_SHA256}" \
  "$(git rev-parse HEAD)" "${EXPLORATORY_ACK}" "${SLURM_JOB_ID:-not_running_under_slurm}" \
  "${EXPECTED_FAILED_GATE_FILE_SHA256}" <<'PY'
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

from training.paths_oswt_trainer import paths_oswt_implementation_signature


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


gate_path = Path(sys.argv[1])
run_root = Path(sys.argv[2])
v3_path = Path(sys.argv[3])
v3_sha256 = sys.argv[4].lower()
current_commit = sys.argv[5]
acknowledgment = sys.argv[6]
slurm_job_id = sys.argv[7]
expected_gate_file_sha256 = sys.argv[8]
gate = json.loads(gate_path.read_text())

if file_sha256(gate_path) != expected_gate_file_sha256:
    raise SystemExit("APTOS gate file SHA-256 differs from the audited artifact")

unsigned = dict(gate)
recorded_checksum = unsigned.pop("content_checksum_sha256", None)
observed_checksum = hashlib.sha256(
    json.dumps(
        unsigned,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()
).hexdigest()
if recorded_checksum != observed_checksum:
    raise SystemExit("APTOS gate content checksum mismatch")
if gate.get("schema") != "paths-v8-oswt-complementary-aptos-paired-gate-v2":
    raise SystemExit("unexpected APTOS gate schema")
if gate.get("passed") is not False:
    raise SystemExit("this launcher is only for an explicitly failed APTOS gate")
if gate.get("protocol") != "paths-v8-oswt-complementary-v2":
    raise SystemExit("unexpected APTOS gate protocol")
expected_gate_identity = {
    "content_checksum_sha256": "2747445a0fd5209b2b2c059c2d6dd3a156b9575373a3d3a03a64af9841e26ef8",
    "git_commit": "6e85ea9b5ce44279f87453d36ddfa39e67cb34d5",
    "split_signature": "0679d0f1f31cd4ceeef1db74f83fefd23a28ae72873a95b7365cce24febbd177",
    "source_checkpoint_sha256": "fa8979a919e8c1e7709224c4c554c31975816de2377686c0d07523c45bf99365",
    "treatment_checkpoint_sha256": "9409f4d67cbe4f63ea83b138638358aaeaa1852890c82491a343c525033530e9",
    "ungated_checkpoint_sha256": "13f6405a1bb3dc9bcfc184f5b511065bccb369780bdf94c83eed26d37664f866",
    "implementation_signature": "fdaa08402ad694c7e718d4e3c1e070726e3388a29b996a6ea09be3c5d43596b6",
}
for key, expected in expected_gate_identity.items():
    if gate.get(key) != expected:
        raise SystemExit(f"APTOS gate identity mismatch: {key}")
expected_failed_checks = {
    "exact_deletion_certificates",
    "exact_grade3_2_to_3_correction",
    "grade3_2_to_3_is_positive_boundary2_transport",
}
failed_checks = {
    key for key, passed in gate.get("checks", {}).items() if passed is not True
}
if failed_checks != expected_failed_checks:
    raise SystemExit(
        "APTOS failed-check set differs from the audited V8 result: "
        + repr(sorted(failed_checks))
    )
current_signature = paths_oswt_implementation_signature()
if gate.get("implementation_signature") != current_signature:
    raise SystemExit("OSWT implementation bytes differ from the completed APTOS experiment")
gate_commit = gate.get("git_commit")
if not isinstance(gate_commit, str) or len(gate_commit) != 40:
    raise SystemExit("APTOS gate commit is absent or malformed")
allowed_post_gate_files = {
    "scripts/submit_paths_oswt_dr_exploratory_f0.sh",
    "scripts/launch_paths_oswt_v8_dr_exploratory_f0.sh",
}
changed_files = {
    line.strip()
    for line in subprocess.check_output(
        ["git", "diff", "--name-only", f"{gate_commit}..{current_commit}"],
        text=True,
    ).splitlines()
    if line.strip()
}
if not changed_files or not changed_files.issubset(allowed_post_gate_files):
    raise SystemExit(
        "post-gate commit range contains non-launcher changes: "
        + repr(sorted(changed_files))
    )
if file_sha256(v3_path) != v3_sha256:
    raise SystemExit("EyePACS V3 checkpoint checksum changed during launch")

record = {
    "schema": "paths-oswt-exploratory-eyepacs-after-failed-gate-v2",
    "experiment_status": "exploratory_after_failed_aptos_gate",
    "acknowledgment": acknowledgment,
    "confirmatory": False,
    "promotion_eligible": False,
    "registered_gate_bypassed": True,
    "dataset": "dr",
    "fold": 0,
    "seed": 42,
    "evaluation_scope": "inner_validation_only",
    "locked_test_evaluated": False,
    "aptos_gate_passed": False,
    "failed_gate_checks": sorted(failed_checks),
    "aptos_gate_path": str(gate_path),
    "aptos_gate_file_sha256": file_sha256(gate_path),
    "aptos_gate_content_checksum_sha256": recorded_checksum,
    "aptos_gate_git_commit": gate.get("git_commit"),
    "aptos_split_signature": gate.get("split_signature"),
    "aptos_source_checkpoint_sha256": gate.get("source_checkpoint_sha256"),
    "aptos_treatment_checkpoint_sha256": gate.get("treatment_checkpoint_sha256"),
    "aptos_ungated_checkpoint_sha256": gate.get("ungated_checkpoint_sha256"),
    "launch_git_commit": current_commit,
    "implementation_signature": current_signature,
    "source_v3_checkpoint": str(v3_path),
    "source_v3_checkpoint_sha256": v3_sha256,
    "run_dir": str(run_root),
    "slurm_job_id": slurm_job_id,
    "command": os.environ["PATHS_OSWT_EXPLORATORY_TRAIN_COMMAND"],
    "command_contract": "unchanged_v8_gated_hyperparameters_fold0_skip_test",
    "created_at_utc": datetime.now(timezone.utc).isoformat(),
}
payload = dict(record)
payload["content_checksum_sha256"] = hashlib.sha256(
    json.dumps(
        record,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()
).hexdigest()
run_root.mkdir(parents=True, exist_ok=True)
destination = run_root / "EXPLORATORY_ONLY.json"
temporary = destination.with_suffix(".json.tmp")
temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
temporary.replace(destination)
(run_root / "DO_NOT_USE_FOR_PROMOTION").write_text(
    "Exploratory EyePACS run after a failed APTOS gate.\n"
)
print(json.dumps(payload, indent=2, sort_keys=True))
PY

echo "=== PATHS-V8 OSWT exploratory EyePACS fold 0 ==="
echo "STATUS: exploratory after failed APTOS gate; not prospectively promoted"
date --iso-8601=seconds
git rev-parse HEAD
git status --short
python --version

PATHS_OSWT_RUN_GIT_COMMIT="$(git rev-parse HEAD)" "${TRAIN_ARGS[@]}"

# Bind the completed exploratory result to the launch manifest and every
# scientific artifact needed for later audit.  This file is deliberately not
# a promotion marker.
python - "${RUN_ROOT}" "$(git rev-parse HEAD)" <<'PY'
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys


def file_sha256(path: Path) -> str:
    if not path.is_file():
        raise FileNotFoundError(path)
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


run_root = Path(sys.argv[1])
current_commit = sys.argv[2]
fold_root = run_root / "fold0"
manifest_path = run_root / "EXPLORATORY_ONLY.json"
manifest = json.loads(manifest_path.read_text())
manifest_unsigned = dict(manifest)
manifest_checksum = manifest_unsigned.pop("content_checksum_sha256", None)
observed_manifest_checksum = hashlib.sha256(
    json.dumps(
        manifest_unsigned,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()
).hexdigest()
if manifest_checksum != observed_manifest_checksum:
    raise SystemExit("exploratory launch manifest checksum mismatch after training")
result_path = fold_root / "result.json"
result = json.loads(result_path.read_text())

expected_manifest = {
    "schema": "paths-oswt-exploratory-eyepacs-after-failed-gate-v2",
    "experiment_status": "exploratory_after_failed_aptos_gate",
    "acknowledgment": "I_ACKNOWLEDGE_APTOS_GATE_FAILED_AND_THIS_EYEPACS_RUN_IS_EXPLORATORY_ONLY",
    "confirmatory": False,
    "promotion_eligible": False,
    "registered_gate_bypassed": True,
    "dataset": "dr",
    "fold": 0,
    "seed": 42,
    "evaluation_scope": "inner_validation_only",
    "locked_test_evaluated": False,
    "aptos_gate_passed": False,
    "failed_gate_checks": [
        "exact_deletion_certificates",
        "exact_grade3_2_to_3_correction",
        "grade3_2_to_3_is_positive_boundary2_transport",
    ],
    "aptos_gate_file_sha256": "47c12db8ebf448e7ee318b4a532b96065de77ddd0a532b9b87ea87fa69de851b",
    "aptos_gate_content_checksum_sha256": "2747445a0fd5209b2b2c059c2d6dd3a156b9575373a3d3a03a64af9841e26ef8",
    "aptos_gate_git_commit": "6e85ea9b5ce44279f87453d36ddfa39e67cb34d5",
    "aptos_split_signature": "0679d0f1f31cd4ceeef1db74f83fefd23a28ae72873a95b7365cce24febbd177",
    "aptos_source_checkpoint_sha256": "fa8979a919e8c1e7709224c4c554c31975816de2377686c0d07523c45bf99365",
    "aptos_treatment_checkpoint_sha256": "9409f4d67cbe4f63ea83b138638358aaeaa1852890c82491a343c525033530e9",
    "aptos_ungated_checkpoint_sha256": "13f6405a1bb3dc9bcfc184f5b511065bccb369780bdf94c83eed26d37664f866",
    "launch_git_commit": current_commit,
    "implementation_signature": "fdaa08402ad694c7e718d4e3c1e070726e3388a29b996a6ea09be3c5d43596b6",
    "source_v3_checkpoint_sha256": os.environ["PATHS_OSWT_DR_V3_SHA256"].lower(),
    "run_dir": str(run_root),
    "slurm_job_id": os.environ.get("SLURM_JOB_ID", "not_running_under_slurm"),
    "command": os.environ["PATHS_OSWT_EXPLORATORY_TRAIN_COMMAND"],
    "command_contract": "unchanged_v8_gated_hyperparameters_fold0_skip_test",
}
for key, expected in expected_manifest.items():
    if manifest.get(key) != expected:
        raise SystemExit(f"exploratory launch manifest contract mismatch: {key}")

if result.get("test_evaluated") is not False:
    raise SystemExit("exploratory run unexpectedly evaluated the locked test split")
if result.get("fold") != 0:
    raise SystemExit("exploratory run is not fold 0")
if result.get("protocol") != "paths-v8-oswt-complementary-v2":
    raise SystemExit("unexpected OSWT result protocol")
if result.get("oswt_variant") != "shell_warranted":
    raise SystemExit("unexpected exploratory OSWT variant")
if result.get("run_git_commit") != current_commit:
    raise SystemExit("result commit differs from exploratory launcher commit")
if result.get("implementation_signature") != manifest.get("implementation_signature"):
    raise SystemExit("result implementation differs from launch manifest")
if result.get("source_v3_base_state_sha256") != result.get("selected_v3_base_state_sha256"):
    raise SystemExit("frozen V3 state changed during the exploratory run")

artifact_paths = {
    "launch_manifest": manifest_path,
    "do_not_promote_sentinel": run_root / "DO_NOT_USE_FOR_PROMOTION",
    "result": result_path,
    "history": fold_root / "history.csv",
    "split_manifest": fold_root / "split_manifest.json",
    "identity_control": fold_root / "hash_bound_v3_identity_control.json",
    "structural_audit": fold_root / "validation_oswt_audit.json",
    "certificates": fold_root / "validation_certificates.json",
    "best_checkpoint": fold_root / "best.pth",
    "best_learned_checkpoint": fold_root / "best_learned.pth",
    "last_checkpoint": fold_root / "last.pth",
    "final_results_json": run_root / "final_results_folds_0.json",
    "final_results_csv": run_root / "final_results_folds_0.csv",
}
artifact_sha256 = {
    name: file_sha256(path) for name, path in artifact_paths.items()
}
if artifact_sha256["best_checkpoint"] != artifact_sha256["best_learned_checkpoint"]:
    raise SystemExit("best_learned.pth is not the byte-exact selected checkpoint")

record = {
    "schema": "paths-oswt-exploratory-result-binding-v1",
    "experiment_status": "exploratory_after_failed_aptos_gate",
    "confirmatory": False,
    "promotion_eligible": False,
    "registered_gate_bypassed": True,
    "dataset": "dr",
    "fold": 0,
    "evaluation_scope": "inner_validation_only",
    "locked_test_evaluated": False,
    "run_dir": str(run_root),
    "launch_git_commit": current_commit,
    "implementation_signature": result.get("implementation_signature"),
    "config_signature": result.get("config_signature"),
    "split_signature": result.get("split_signature"),
    "source_checkpoint_sha256": manifest.get("source_v3_checkpoint_sha256"),
    "aptos_failed_gate_file_sha256": manifest.get("aptos_gate_file_sha256"),
    "aptos_failed_gate_content_checksum_sha256": manifest.get(
        "aptos_gate_content_checksum_sha256"
    ),
    "source_v3_base_state_sha256": result.get("source_v3_base_state_sha256"),
    "selected_v3_base_state_sha256": result.get("selected_v3_base_state_sha256"),
    "best_epoch": result.get("best_epoch"),
    "artifact_sha256": artifact_sha256,
    "completed_at_utc": datetime.now(timezone.utc).isoformat(),
}
payload = dict(record)
payload["content_checksum_sha256"] = hashlib.sha256(
    json.dumps(
        record,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()
).hexdigest()
destination = run_root / "EXPLORATORY_RESULT_BINDING.json"
temporary = destination.with_suffix(".json.tmp")
temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
temporary.replace(destination)
print(json.dumps(payload, indent=2, sort_keys=True))
PY
