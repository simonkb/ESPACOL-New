#!/bin/bash
# Paired promotion audit for SAPT treatment versus exact V3 and the identical
# risk-objective V3 control.  This job performs no training.
#SBATCH --job-name=paths_gate
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --output=/dpc/kuin0170/ESPACOL-New/paths_aptos_gate_%j.out
#SBATCH --error=/dpc/kuin0170/ESPACOL-New/paths_aptos_gate_%j.err
#SBATCH --account=kuin0170

set -eo pipefail
set +u
source /etc/profile.d/lmod.sh || exit 1
module load miniconda/3 || exit 1
module load cuda/12.6 || exit 1
source activate "${ORIGIN_CONDA_ENV:-G}" || exit 1
set -u

REPO_ROOT="${ORIGIN_REPO_ROOT:-/dpc/kuin0170/ESPACOL-New}"
cd "${REPO_ROOT}"
TREATMENT_DIR="${PATHS_APTOS_RUN_DIR:-runs/paths_aptos_f0_v3_sapt}/fold0"
CONTROL_DIR="${PATHS_APTOS_CONTROL_RUN_DIR:-runs/paths_aptos_f0_v3_risk_control}/fold0"
HEAD_COMMIT="$(git rev-parse HEAD)"

for PATHS_FILE in \
  configs/paths_config.py configs/origin_config.py \
  Datasets/origin_data.py Datasets/mosaic_data.py Datasets/dataloaders.py \
  models/origin_encoder.py models/origin.py models/paths.py \
  losses/origin.py losses/paths.py \
  training/origin_trainer.py training/paths_trainer.py \
  train_paths.py utils/spatial_mask.py \
  scripts/submit_paths_preflight.sh scripts/submit_paths_aptos_f0.sh \
  scripts/submit_paths_aptos_control_f0.sh scripts/submit_paths_aptos_gate.sh \
  scripts/submit_paths_dr_f0.sh; do
  [[ "$(git rev-parse "HEAD:${PATHS_FILE}")" == "$(git hash-object "${PATHS_FILE}")" ]] || {
    echo "Tracked implementation differs from HEAD: ${PATHS_FILE}" >&2
    exit 2
  }
done

echo "=== PATHS-v3 SAPT paired APTOS promotion gate ==="
date --iso-8601=seconds
echo "${HEAD_COMMIT}"

PATHS_HEAD="${HEAD_COMMIT}" PATHS_TREATMENT="${TREATMENT_DIR}" \
PATHS_CONTROL="${CONTROL_DIR}" python - <<'PY'
import hashlib
import json
import os
from pathlib import Path

import torch

from configs.paths_config import PATHS_PROTOCOL_VERSION
from training.paths_trainer import paths_implementation_signature


def load(path: Path):
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text())


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def correct(metrics: dict) -> int:
    confusion = metrics["confusion"]
    return sum(int(confusion[index][index]) for index in range(len(confusion)))


def content_checksum(payload: dict) -> str:
    unsigned = dict(payload)
    unsigned.pop("content_checksum_sha256", None)
    encoded = json.dumps(
        unsigned, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def matched_config(payload: dict) -> dict:
    result = dict(payload)
    result.pop("paths_variant", None)
    result.pop("transport_strength", None)
    return result


def matched_static_metrics(left: dict, right: dict, tolerance: float) -> bool:
    if left.get("confusion") != right.get("confusion"):
        return False
    scalar_keys = ("acc", "qwk", "mae", "balanced_acc", "macro_f1", "ece")
    return all(
        abs(float(left[key]) - float(right[key])) <= tolerance
        for key in scalar_keys
    )


def checkpoint_matches_result(state: dict, result: dict) -> bool:
    return (
        state.get("schema") == "paths-checkpoint-v3-sapt"
        and state.get("paths_protocol_version") == result.get("protocol")
        and state.get("paths_variant") == result.get("paths_variant")
        and state.get("run_git_commit") == result.get("run_git_commit")
        and state.get("split_signature") == result.get("split_signature")
        and state.get("implementation_signature") == result.get("implementation_signature")
        and state.get("architecture_signature") == result.get("architecture_signature")
        and state.get("config_signature") == result.get("config_signature")
        and state.get("critical_config") == result.get("critical_config")
        and state.get("metrics") == result.get("best_learned_validation")
        and int(state.get("epoch", -1)) == int(result.get("best_learned_epoch", -2))
    )


treatment_dir = Path(os.environ["PATHS_TREATMENT"])
control_dir = Path(os.environ["PATHS_CONTROL"])
treatment = load(treatment_dir / "result.json")
control = load(control_dir / "result.json")
base = load(treatment_dir / "v3_strength_zero_control.json")
control_base = load(control_dir / "v3_strength_zero_control.json")
certificate_path = Path(treatment["validation_certificate_path"])
certificate = load(certificate_path)

treatment_metrics = treatment["best_learned_validation"]
control_metrics = control["best_learned_validation"]
base_metrics = base["metrics"]
treatment_correct = correct(treatment_metrics)
control_correct = correct(control_metrics)
base_correct = correct(base_metrics)
implementation = paths_implementation_signature()
best_learned = Path(treatment["best_learned_checkpoint"])
control_best_learned = Path(control["best_learned_checkpoint"])
treatment_state = torch.load(
    treatment_dir / "best.pth", map_location="cpu", weights_only=False
)
control_state = torch.load(
    control_dir / "best.pth", map_location="cpu", weights_only=False
)
audit = certificate["joint_replay_audit"]
transport = treatment["validation_transport_summary"]
tolerance = float(certificate["replay_tolerance"])
base_match_tolerance = max(
    float(base.get("reproduction_tolerance", 0.0)),
    float(control_base.get("reproduction_tolerance", 0.0)),
)

identity_checks = {
    "protocol_v3_sapt": treatment.get("protocol") == control.get("protocol") == PATHS_PROTOCOL_VERSION,
    "treatment_variant": treatment.get("paths_variant") == "signed_transport",
    "control_variant": control.get("paths_variant") == "risk_objective_v3",
    "same_run_commit": treatment.get("run_git_commit") == control.get("run_git_commit") == os.environ["PATHS_HEAD"],
    "same_split": treatment.get("split_signature") == control.get("split_signature") == base.get("split_signature") == control_base.get("split_signature"),
    "same_source_v3": (
        treatment["warm_start_provenance"]["source_checkpoint_sha256"]
        == control["warm_start_provenance"]["source_checkpoint_sha256"]
        == base["source_checkpoint_sha256"]
        == control_base["source_checkpoint_sha256"]
    ),
    "matched_training_configuration": matched_config(treatment["critical_config"]) == matched_config(control["critical_config"]),
    "implementation_bound": treatment.get("implementation_signature") == control.get("implementation_signature") == implementation,
    "certificate_implementation_bound": certificate.get("implementation_signature") == implementation,
    "certificate_architecture_bound": certificate.get("architecture_signature") == treatment.get("architecture_signature"),
    "certificate_experiment_bound": (
        certificate.get("split_signature") == treatment.get("split_signature")
        and certificate.get("warm_start_checkpoint_sha256") == base.get("source_checkpoint_sha256")
        and certificate.get("protocol") == PATHS_PROTOCOL_VERSION
        and certificate.get("paths_variant") == "signed_transport"
    ),
    "certificate_path_bound": certificate_path.resolve() == (treatment_dir / "validation_certificates.json").resolve(),
    "certificate_content_bound": (
        content_checksum(certificate)
        == certificate.get("content_checksum_sha256")
        == treatment.get("validation_certificate_checksum")
    ),
    "certificate_checkpoint_bound": (
        certificate.get("checkpoint_sha256")
        == treatment.get("best_checkpoint_sha256")
        == treatment.get("best_learned_checkpoint_sha256")
        == sha256(treatment_dir / "best.pth")
        == sha256(best_learned)
        and int(certificate.get("checkpoint_epoch", -1)) == int(treatment.get("best_epoch", -2))
        and treatment.get("best_learned_is_byte_exact_best_alias") is True
    ),
    "control_checkpoint_hash_bound": (
        sha256(control_dir / "best.pth")
        == sha256(control_best_learned)
        == control.get("best_checkpoint_sha256")
        == control.get("best_learned_checkpoint_sha256")
        and control.get("best_learned_is_byte_exact_best_alias") is True
    ),
    "checkpoint_metadata_bound_to_results": (
        checkpoint_matches_result(treatment_state, treatment)
        and checkpoint_matches_result(control_state, control)
    ),
    "both_validation_only": treatment.get("test_evaluated") is False and control.get("test_evaluated") is False,
    "exact_v3_reproduced": base.get("reproduced") is True and control_base.get("reproduced") is True,
    "identical_static_v3_predictions": (
        base.get("ordered_prediction_checksum_sha256")
        == control_base.get("ordered_prediction_checksum_sha256")
        and matched_static_metrics(
            base.get("metrics", {}),
            control_base.get("metrics", {}),
            base_match_tolerance,
        )
    ),
}

checks = {
    **identity_checks,
    "correct_at_least_253_of_293": treatment_correct >= 253,
    "strictly_more_correct_than_v3": treatment_correct > base_correct,
    "strictly_more_correct_than_risk_control": treatment_correct > control_correct,
    "qwk_at_least_v3_plus_0_003": float(treatment_metrics["qwk"]) >= float(base_metrics["qwk"]) + 0.003,
    "qwk_above_risk_control": float(treatment_metrics["qwk"]) > float(control_metrics["qwk"]),
    "mae_non_regression": float(treatment_metrics["mae"]) <= float(base_metrics["mae"]),
    "balanced_acc_non_regression": float(treatment_metrics["balanced_acc"]) >= float(base_metrics["balanced_acc"]),
    "macro_f1_non_regression": float(treatment_metrics["macro_f1"]) >= float(base_metrics["macro_f1"]),
    "grade3_at_least_4_of_15": int(treatment_metrics["confusion"][3][3]) >= 4,
    "joint_sapt_certificate": (
        certificate.get("schema") == "paths-exact-joint-validation-certificates-v3-sapt"
        and certificate.get("protocol") == PATHS_PROTOCOL_VERSION
        and certificate.get("paths_variant") == "signed_transport"
        and certificate.get("certifies_final_paths_prediction") is True
        and certificate.get("base_only_certificate") is False
        and float(audit["transport_effect_max_abs"]) > 1e-10
        and float(audit["rate_replay_error"]) <= tolerance
        and float(audit["concentration_replay_error"]) <= tolerance
        and float(audit["transport_matrix_row_sum_error"]) <= tolerance
        and float(audit["transport_matrix_minimum"]) >= -tolerance
        and float(audit["transport_non_adjacent_max_abs"]) <= tolerance
        and float(audit["baseline_transport_posterior_error"]) <= tolerance
        and float(audit["replayed_transport_posterior_error"]) <= tolerance
        and float(audit["baseline_flow_reconstruction_error"]) <= tolerance
        and float(audit["replayed_flow_reconstruction_error"]) <= tolerance
    ),
    "full_validation_transport_is_structural": (
        transport.get("schema") == "paths-v3-sapt-validation-transport-audit-v1"
        and transport.get("scope") == "full_inner_validation"
        and int(transport.get("sample_count", -1)) == 293
        and transport.get("transport_active") is True
        and float(transport["transport_effect_max_abs"]) > 1e-10
        and float(transport["transport_matrix_row_sum_error"]) <= tolerance
        and float(transport["transport_matrix_minimum"]) >= -tolerance
        and float(transport["transport_non_adjacent_max_abs"]) <= tolerance
        and float(transport["posterior_mass_error"]) <= tolerance
    ),
    "full_validation_bidirectional_transport": (
        transport.get("bidirectional_flow_observed") is True
        and int(transport.get("positive_upward_flow_count", 0)) > 0
        and int(transport.get("negative_downward_flow_count", 0)) > 0
    ),
}

payload = {
    "schema": "paths-v3-sapt-aptos-paired-gate-v1",
    "git_commit": os.environ["PATHS_HEAD"],
    "protocol": PATHS_PROTOCOL_VERSION,
    "implementation_signature": implementation,
    "architecture_signature": treatment["architecture_signature"],
    "split_signature": treatment["split_signature"],
    "source_checkpoint_sha256": base["source_checkpoint_sha256"],
    "treatment_best_learned_sha256": treatment["best_learned_checkpoint_sha256"],
    "control_best_learned_sha256": control["best_learned_checkpoint_sha256"],
    "checks": checks,
    "passed": all(checks.values()),
    "correct": {
        "signed_transport": treatment_correct,
        "risk_objective_v3": control_correct,
        "static_v3": base_correct,
    },
    "metrics": {
        "signed_transport": treatment_metrics,
        "risk_objective_v3": control_metrics,
        "static_v3": base_metrics,
    },
    "validation_transport_summary": transport,
}
gate = treatment_dir / "PROMOTED_TO_EYEPACS.json"
gate.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")
print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))
if not payload["passed"]:
    raise SystemExit("PATHS-v3 SAPT APTOS paired gate failed; EyePACS remains blocked")
PY

echo "PATHS-v3 SAPT paired APTOS gate passed."
