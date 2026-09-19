#!/bin/bash
# Prospective paired APTOS gate. This job performs inference but no training.
#SBATCH --job-name=oswt_gate
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:1
#SBATCH --mem=40G
#SBATCH --time=02:00:00
#SBATCH --output=/dpc/kuin0170/ESPACOL-New/paths_oswt_gate_%j.out
#SBATCH --error=/dpc/kuin0170/ESPACOL-New/paths_oswt_gate_%j.err
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
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
for path in configs/paths_oswt_config.py configs/paths_config.py configs/origin_config.py Datasets/origin_data.py Datasets/mosaic_data.py Datasets/dataloaders.py models/origin_encoder.py models/origin.py models/paths.py models/paths_oswt.py losses/origin.py losses/paths.py training/origin_trainer.py training/paths_trainer.py training/paths_oswt_trainer.py train_paths_oswt.py train_origin.py utils/spatial_mask.py scripts/submit_paths_oswt_preflight.sh scripts/submit_paths_oswt_aptos_f0.sh scripts/submit_paths_oswt_aptos_ungated_f0.sh scripts/submit_paths_oswt_promotion_gate.sh scripts/submit_paths_oswt_dr_f0.sh scripts/launch_paths_oswt_v8_aptos_f0.sh; do
  [[ "$(git rev-parse "HEAD:${path}")" == "$(git hash-object "${path}")" ]] || {
    echo "Tracked implementation differs from HEAD: ${path}" >&2; exit 2;
  }
done
export PATHS_OSWT_TREATMENT_DIR="${PATHS_OSWT_APTOS_RUN_DIR:-runs/paths_oswt_aptos_f0_v8_gated}/fold0"
export PATHS_OSWT_CONTROL_DIR="${PATHS_OSWT_APTOS_UNGATED_RUN_DIR:-runs/paths_oswt_aptos_f0_v8_ungated}/fold0"
export PATHS_OSWT_APTOS_ROOT="${PATHS_OSWT_APTOS_DATA_ROOT:-${REPO_ROOT}/Datasets/aptos2019-blindness-detection}"
export PATHS_OSWT_GATE_HEAD="$(git rev-parse HEAD)"

echo "=== PATHS-V8 OSWT prospective paired APTOS promotion gate ==="
date --iso-8601=seconds
echo "${PATHS_OSWT_GATE_HEAD}"
git status --short
python --version

python - <<'PY'
from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping
from pathlib import Path

import torch
from torch.amp import autocast

from configs.paths_oswt_config import (
    PATHS_OSWT_CHECKPOINT_SCHEMA,
    PATHS_OSWT_PROTOCOL_VERSION,
)
from Datasets.origin_data import load_origin_items, make_origin_loaders, split_origin_items
from models.paths_oswt import PathsOSWTOutput, build_paths_oswt_model
from train_origin import split_signature
from training.origin_trainer import evaluate_origin_predictions
from training.paths_oswt_trainer import (
    _v3_base_state_sha256,
    paths_oswt_implementation_signature,
)


def read(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text())


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def content_hash(payload: dict) -> str:
    unsigned = dict(payload)
    unsigned.pop("content_checksum_sha256", None)
    return hashlib.sha256(
        json.dumps(unsigned, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def correct(metrics: dict) -> int:
    return sum(int(row[index]) for index, row in enumerate(metrics["confusion"]))


def build_from_state(state: dict, *, gated: bool):
    cfg = state["config"]
    model = build_paths_oswt_model(
        num_classes=int(cfg["n_classes"]),
        encoder_name=cfg["encoder"],
        pretrained=False,
        evidence_scales=tuple(cfg["evidence_scales"]),
        projection_dim=int(cfg["projection_dim"]),
        reference_count=float(cfg["reference_count"]),
        atom_mode=cfg["atom_mode"],
        hybrid_cumulative_init=float(cfg["hybrid_cumulative_init"]),
        atom_rate_init=float(cfg["atom_rate_init"]),
        prior_rate_init=float(cfg["prior_rate_init"]),
        boundary_scale_init=float(cfg["boundary_scale_init"]),
        total_rate_cap=float(cfg["total_rate_cap"]),
        prior_rate_cap=float(cfg["prior_rate_cap"]),
        boundary_scale_cap=float(cfg["boundary_scale_cap"]),
        rate_roundoff_margin=float(cfg["rate_roundoff_margin"]),
        evidence_dropout=float(cfg["evidence_dropout"]),
        mask_valid_fraction=float(cfg["mask_valid_fraction"]),
        grad_checkpoint=bool(cfg["grad_checkpoint"]),
        paths_oswt_probe_z=tuple(cfg["pgf_probes"]),
        paths_oswt_beta_init=float(cfg["oswt_beta_init"]),
        paths_oswt_tau_init=float(cfg["oswt_tau_init"]),
        paths_oswt_beta_cap=float(cfg["oswt_beta_cap"]),
        paths_oswt_tau_floor=float(cfg["oswt_tau_floor"]),
        paths_oswt_tau_cap=float(cfg["oswt_tau_cap"]),
        paths_oswt_strength=float(cfg["oswt_strength"]),
        paths_oswt_risk_gate=gated,
    )
    model.load_state_dict(state["model_state"], strict=True)
    return model


def infer(state: dict, *, gated: bool, loader, device):
    model = build_from_state(state, gated=gated).to(device).eval()
    labels, indices, base_pred, final_pred, probabilities, signed_flow = [], [], [], [], [], []
    with torch.no_grad():
        for images, valid, target, index in loader:
            images = images.to(device, non_blocking=True)
            valid = valid.to(device, non_blocking=True)
            with autocast(device_type="cuda", enabled=True):
                output = model(images, valid, force_decoder_fp64=True)
            if not isinstance(output, PathsOSWTOutput):
                raise RuntimeError("learned OSWT checkpoint bypassed shell transport")
            labels.append(target.long())
            indices.append(index.long())
            base_pred.append(output.base_output.class_map.detach().cpu().long())
            final_pred.append(output.class_map.detach().cpu().long())
            probabilities.append(output.class_probs.detach().cpu().float())
            signed_flow.append(output.signed_boundary_flow.detach().cpu().double())
    del model
    torch.cuda.empty_cache()
    target = torch.cat(labels)
    predicted = torch.cat(final_pred)
    base = torch.cat(base_pred)
    metric = evaluate_origin_predictions(torch.cat(probabilities), predicted, target)
    return {
        "labels": target,
        "indices": torch.cat(indices),
        "base": base,
        "predicted": predicted,
        "signed_flow": torch.cat(signed_flow),
        "metrics": metric,
    }


treatment_dir = Path(os.environ["PATHS_OSWT_TREATMENT_DIR"])
control_dir = Path(os.environ["PATHS_OSWT_CONTROL_DIR"])
treatment_result = read(treatment_dir / "result.json")
control_result = read(control_dir / "result.json")
treatment_checkpoint = treatment_dir / "best.pth"
control_checkpoint = control_dir / "best.pth"
treatment_alias = treatment_dir / "best_learned.pth"
control_alias = control_dir / "best_learned.pth"
treatment_state = torch.load(treatment_checkpoint, map_location="cpu", weights_only=False)
control_state = torch.load(control_checkpoint, map_location="cpu", weights_only=False)
treatment_certificate = read(Path(treatment_result["validation_certificate_path"]))
control_certificate = read(Path(control_result["validation_certificate_path"]))
treatment_structure = read(Path(treatment_result["validation_structural_audit_path"]))
control_structure = read(Path(control_result["validation_structural_audit_path"]))
identity = treatment_result["v3_identity_control"]
control_identity = control_result["v3_identity_control"]

items = load_origin_items("aptos", os.environ["PATHS_OSWT_APTOS_ROOT"])
train_items, validation_items, test_items = split_origin_items(
    "aptos", items, fold=0, n_folds=5, val_fraction=0.1, seed=42
)
expected_split = split_signature(
    ("train", train_items), ("validation", validation_items), ("locked_test", test_items)
)
_, validation_loader, _ = make_origin_loaders(
    train_items, validation_items, test_items,
    image_size=640, batch_size=8, num_workers=4, pin_memory=True,
    seed=42, fundus=True, stratified=False,
)
device = torch.device("cuda")
treatment = infer(treatment_state, gated=True, loader=validation_loader, device=device)
control = infer(control_state, gated=False, loader=validation_loader, device=device)

if not torch.equal(treatment["indices"], control["indices"]):
    raise AssertionError("paired OSWT validation orders differ")
if not torch.equal(treatment["labels"], control["labels"]):
    raise AssertionError("paired OSWT validation labels differ")
if not torch.equal(treatment["base"], control["base"]):
    raise AssertionError("paired checkpoints do not reproduce the same V3 predictions")

labels = treatment["labels"]
source = treatment["base"]
treat_pred = treatment["predicted"]
control_pred = control["predicted"]
flow = treatment["signed_flow"]
grade3 = labels == 3
upward_grade3 = grade3 & (source == 2) & (treat_pred == 3)
downward_grade3 = grade3 & (source == 4) & (treat_pred == 3)
flow_tolerance = 1e-12
paired = {
    "sample_count": int(labels.numel()),
    "source_correct": int((source == labels).sum()),
    "treatment_correct": int((treat_pred == labels).sum()),
    "ungated_correct": int((control_pred == labels).sum()),
    "treatment_help_vs_source": int(((treat_pred == labels) & (source != labels)).sum()),
    "treatment_harm_vs_source": int(((treat_pred != labels) & (source == labels)).sum()),
    "treatment_help_vs_ungated": int(((treat_pred == labels) & (control_pred != labels)).sum()),
    "treatment_harm_vs_ungated": int(((treat_pred != labels) & (control_pred == labels)).sum()),
    "grade3_source_2_to_treatment_3": int(upward_grade3.sum()),
    "grade3_source_4_to_treatment_3": int(downward_grade3.sum()),
    "grade3_2_to_3_with_positive_boundary2_flow": int((upward_grade3 & (flow[:, 2] > flow_tolerance)).sum()),
    "grade3_4_to_3_with_negative_boundary3_flow": int((downward_grade3 & (flow[:, 3] < -flow_tolerance)).sum()),
    "grade3_source_correct_harmed": int((grade3 & (source == 3) & (treat_pred != 3)).sum()),
}
paired_ids = {
    "grade3_source_2_to_treatment_3": [
        Path(validation_items[int(index)][0]).name
        for index in treatment["indices"][upward_grade3]
    ],
    "grade3_source_4_to_treatment_3": [
        Path(validation_items[int(index)][0]).name
        for index in treatment["indices"][downward_grade3]
    ],
    "grade3_2_to_3_with_positive_boundary2_flow": [
        Path(validation_items[int(index)][0]).name
        for index in treatment["indices"][upward_grade3 & (flow[:, 2] > flow_tolerance)]
    ],
    "grade3_4_to_3_with_negative_boundary3_flow": [
        Path(validation_items[int(index)][0]).name
        for index in treatment["indices"][downward_grade3 & (flow[:, 3] < -flow_tolerance)]
    ],
    "grade3_source_correct_harmed": [
        Path(validation_items[int(index)][0]).name
        for index in treatment["indices"][grade3 & (source == 3) & (treat_pred != 3)]
    ],
}

implementation = paths_oswt_implementation_signature()
treatment_metrics = treatment["metrics"]
control_metrics = control["metrics"]
source_metrics = identity["metrics"]
tolerance = max(
    float(treatment_state["config"].get("certificate_replay_tolerance", 2e-5)),
    float(control_state["config"].get("certificate_replay_tolerance", 2e-5)),
)

def metric_replay(saved: dict, observed: dict) -> bool:
    return saved["confusion"] == observed["confusion"] and all(
        abs(float(saved[key]) - float(observed[key])) <= 2e-5
        for key in ("acc", "qwk", "mae", "balanced_acc", "macro_f1", "ece")
    )


REQUIRED_FORWARD_ERRORS = {
    "adjacent_log_odds_replay_error",
    "base_boundary_probability_formula_error",
    "boundary_correction_reconstruction_error",
    "boundary_coverage_formula_error",
    "boundary_gate_formula_error",
    "boundary_scale_weight_simplex_error",
    "complementary_partition_mass_error",
    "grade_log_correction_reconstruction_error",
    "left_baseline_mass_reconstruction_error",
    "left_boundary_warrant_ledger_error",
    "left_local_boundary_warrant_reconstruction_error",
    "left_local_warrant_spectrum_formula_error",
    "left_mixed_warrant_reconstruction_error",
    "left_partition_map_reconstruction_error",
    "left_partition_share_mass_error",
    "left_partition_share_reconstruction_error",
    "left_phi_spectrum_reconstruction_error",
    "left_surviving_mass_bound_error",
    "left_surviving_mass_reconstruction_error",
    "left_warrant_spectrum_reduction_error",
    "log_odds_increment_formula_error",
    "ordinal_shell_reconstruction_error",
    "ordinal_shell_simplex_error",
    "posterior_from_grade_correction_error",
    "posterior_normalization_error",
    "probe_weight_simplex_error",
    "right_baseline_mass_reconstruction_error",
    "right_boundary_warrant_ledger_error",
    "right_local_boundary_warrant_reconstruction_error",
    "right_local_warrant_spectrum_formula_error",
    "right_mixed_warrant_reconstruction_error",
    "right_partition_map_reconstruction_error",
    "right_partition_share_mass_error",
    "right_partition_share_reconstruction_error",
    "right_phi_spectrum_reconstruction_error",
    "right_surviving_mass_bound_error",
    "right_surviving_mass_reconstruction_error",
    "right_warrant_spectrum_reduction_error",
    "shell_contrast_formula_error",
    "signed_flow_reconstruction_error",
    "wasserstein1_reconstruction_error",
}
REQUIRED_DELETION_ERRORS = {
    "base_rate_intervention_consistency_error",
    "continuation_correction_deletion_replay_error",
    "left_boundary_warrant_deletion_replay_error",
    "log_odds_increment_deletion_replay_error",
    "origin_rate_deletion_replay_error",
    "right_boundary_warrant_deletion_replay_error",
    "signed_flow_deletion_replay_error",
}
CERTIFICATE_COUNT = 8
CERTIFICATE_SELECTION_RULE = (
    "max_predicted_grade_margin_drop_then_target_log_probability_"
    "drop_then_scale_then_flat_index"
)


def normalized_provenance(payload: object) -> dict:
    if not isinstance(payload, dict):
        return {}
    normalized = dict(payload)
    # Invocation paths are not content identity and may differ across mounts.
    normalized.pop("source_checkpoint", None)
    return normalized


def finite_number(value: object) -> bool:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return False
    return bool(torch.isfinite(torch.tensor(number, dtype=torch.float64)))


def exact_shortlist_row_ok(row: dict) -> bool:
    candidates = row.get("candidate_replays")
    if not isinstance(candidates, list) or len(candidates) != CERTIFICATE_COUNT:
        return False
    if (
        row.get("selection_scope") != "exact_best_within_prespecified_shortlist"
        or row.get("selection_rule") != CERTIFICATE_SELECTION_RULE
        or row.get("positive_margin_drop_required_for_promotion") is not True
        or row.get("positive_margin_drop") is not True
        or int(row.get("candidate_shortlist_limit", -1)) != CERTIFICATE_COUNT
        or int(row.get("candidate_shortlist_size", -1)) != CERTIFICATE_COUNT
        or not finite_number(row.get("predicted_grade_margin_drop"))
        or float(row["predicted_grade_margin_drop"]) <= 0.0
    ):
        return False
    required = {
        "shortlist_rank",
        "scale",
        "spatial_index",
        "flat_index",
        "local_intervention_score",
        "predicted_grade_margin_drop",
        "target_log_probability_drop",
    }
    if [candidate.get("shortlist_rank") for candidate in candidates] != list(
        range(CERTIFICATE_COUNT)
    ):
        return False
    for candidate in candidates:
        if not required.issubset(candidate):
            return False
        if candidate["scale"] not in {"s4", "s8", "s16", "s32"}:
            return False
        if not all(
            finite_number(candidate[key])
            for key in (
                "local_intervention_score",
                "predicted_grade_margin_drop",
                "target_log_probability_drop",
            )
        ):
            return False
    scale_rank = {"s4": 0, "s8": 1, "s16": 2, "s32": 3}
    winner = max(
        candidates,
        key=lambda candidate: (
            float(candidate["predicted_grade_margin_drop"]),
            float(candidate["target_log_probability_drop"]),
            -scale_rank[candidate["scale"]],
            -int(candidate["flat_index"]),
        ),
    )
    selected_fields = (
        "shortlist_rank",
        "scale",
        "spatial_index",
        "flat_index",
        "local_intervention_score",
        "predicted_grade_margin_drop",
        "target_log_probability_drop",
    )
    if any(row.get(key) != winner.get(key) for key in selected_fields):
        return False
    if int(row.get("scale_rank", -1)) != scale_rank[winner["scale"]]:
        return False
    return abs(
        float(row.get("baseline_predicted_grade_margin", float("nan")))
        - float(row.get("replayed_predicted_grade_margin", float("nan")))
        - float(row["predicted_grade_margin_drop"])
    ) <= tolerance


def structural_ok(
    payload: dict,
    *,
    variant: str,
    architecture: str,
    checkpoint_hash: str,
    checkpoint_epoch: int,
    base_hash: str,
) -> bool:
    maxima = payload.get("maximum_identity_errors", {})
    audited = set(payload.get("forward_identity_error_keys", []))
    return (
        payload.get("schema") == "paths-oswt-full-validation-structural-audit-v2"
        and payload.get("scope") == "inner_validation_only"
        and int(payload.get("fold", -1)) == 0
        and int(payload.get("sample_count", -1)) == 293
        and int(payload.get("checkpoint_epoch", -1)) == checkpoint_epoch
        and payload.get("selected_checkpoint_sha256") == checkpoint_hash
        and payload.get("paths_oswt_protocol_version") == PATHS_OSWT_PROTOCOL_VERSION
        and payload.get("oswt_variant") == variant
        and payload.get("implementation_signature") == implementation
        and payload.get("architecture_signature") == architecture
        and payload.get("split_signature") == expected_split
        and payload.get("source_v3_base_state_sha256") == base_hash
        and payload.get("final_v3_base_state_sha256") == base_hash
        and payload.get("selected_v3_base_state_sha256") == base_hash
        and payload.get("final_v3_base_matches_source") is True
        and payload.get("selected_v3_base_matches_source") is True
        and content_hash(payload) == payload.get("content_checksum_sha256")
        and payload.get("all_forward_posterior_identities_audited") is True
        and payload.get("deletion_replay_scope") == "separate_exact_certificate_subset"
        and REQUIRED_FORWARD_ERRORS.issubset(audited)
        and audited.issubset(set(maxima))
        and all(
            float(value) <= tolerance
            for key, value in maxima.items()
            if key.endswith("_error")
        )
        and float(maxima.get("transport_effect_max_abs", 0.0)) > 1e-12
    )


def certificate_ok(
    payload: dict,
    *,
    variant: str,
    architecture: str,
    checkpoint_hash: str,
    checkpoint_epoch: int,
    source_checkpoint_hash: str,
    base_hash: str,
) -> bool:
    audit = payload.get("audit", {})
    replay_keys = set(payload.get("deletion_replay_error_keys", []))
    rows = payload.get("certificates", [])
    return (
        payload.get("schema") == "paths-oswt-exact-validation-certificates-v2"
        and payload.get("scope") == "inner_validation_only"
        and int(payload.get("fold", -1)) == 0
        and int(payload.get("checkpoint_epoch", -1)) == checkpoint_epoch
        and payload.get("selected_checkpoint_sha256") == checkpoint_hash
        and payload.get("paths_oswt_protocol_version") == PATHS_OSWT_PROTOCOL_VERSION
        and payload.get("oswt_variant") == variant
        and payload.get("implementation_signature") == implementation
        and payload.get("architecture_signature") == architecture
        and payload.get("split_signature") == expected_split
        and payload.get("source_checkpoint_sha256") == source_checkpoint_hash
        and payload.get("source_v3_base_state_sha256") == base_hash
        and payload.get("final_v3_base_state_sha256") == base_hash
        and payload.get("selected_v3_base_state_sha256") == base_hash
        and payload.get("final_v3_base_matches_source") is True
        and payload.get("selected_v3_base_matches_source") is True
        and content_hash(payload) == payload.get("content_checksum_sha256")
        and payload.get("prediction_is_transported_posterior") is True
        and payload.get("sample_selection") == "deterministic_grade_round_robin_full_validation"
        and int(payload.get("certificate_shortlist_size", -1)) == CERTIFICATE_COUNT
        and payload.get("witness_selection_scope") == "exact_best_within_prespecified_shortlist"
        and payload.get("deletion_replay_audited") is True
        and int(payload.get("certificate_count", -1)) == CERTIFICATE_COUNT
        and len(rows) == CERTIFICATE_COUNT
        and int(payload.get("positive_witness_count", -1)) == CERTIFICATE_COUNT
        and float(payload.get("positive_witness_rate", -1.0)) == 1.0
        and payload.get("all_witnesses_positive") is True
        and REQUIRED_DELETION_ERRORS == replay_keys
        and REQUIRED_DELETION_ERRORS.issubset(set(audit))
        and REQUIRED_FORWARD_ERRORS.issubset(set(audit))
        and {f"replayed_{key}" for key in REQUIRED_FORWARD_ERRORS}.issubset(set(audit))
        and all(
            float(value) <= tolerance
            for key, value in audit.items()
            if key.endswith("_error")
        )
        and float(audit.get("posterior_minimum", -1.0)) >= -tolerance
        and float(audit.get("replayed_posterior_minimum", -1.0)) >= -tolerance
        and float(audit.get("transport_effect_max_abs", 0.0)) > 1e-12
        and [row.get("certificate_rank") for row in rows]
        == list(range(CERTIFICATE_COUNT))
        and len({row.get("validation_position") for row in rows})
        == CERTIFICATE_COUNT
        and len({str(row.get("sample_id")) for row in rows})
        == CERTIFICATE_COUNT
        and {int(row.get("label", -1)) for row in rows} == {0, 1, 2, 3, 4}
        and all(exact_shortlist_row_ok(row) for row in rows)
    )


def matched_training_config(left: dict, right: dict) -> bool:
    left = dict(left)
    right = dict(right)
    for key in ("oswt_variant", "run_dir"):
        left.pop(key, None)
        right.pop(key, None)
    return left == right


def refiner_only_optimizer_contract(payload: object) -> bool:
    return (
        isinstance(payload, list)
        and len(payload) == 1
        and payload[0].get("name") == "paths_oswt_refiner"
        and float(payload[0].get("weight_decay", -1.0)) == 0.0
        and int(payload[0].get("parameter_count", 0)) > 0
    )


def identity_control_ok(
    payload: dict,
    *,
    variant: str,
    architecture: str,
    source_checkpoint_hash: str,
    base_hash: str,
) -> bool:
    return (
        payload.get("schema") == "paths-oswt-v3-identity-control-v1"
        and payload.get("scope") == "inner_validation_only"
        and int(payload.get("fold", -1)) == 0
        and payload.get("split_signature") == expected_split
        and payload.get("paths_oswt_protocol_version") == PATHS_OSWT_PROTOCOL_VERSION
        and payload.get("oswt_variant") == variant
        and payload.get("implementation_signature") == implementation
        and payload.get("architecture_signature") == architecture
        and payload.get("source_checkpoint_sha256") == source_checkpoint_hash
        and payload.get("source_v3_base_state_sha256") == base_hash
        and payload.get("control_v3_base_state_sha256") == base_hash
        and payload.get("control_v3_base_matches_source") is True
        and payload.get("reproduced") is True
        and content_hash(payload) == payload.get("content_checksum_sha256")
    )


def checkpoint_binding_ok(
    *,
    result: dict,
    state: dict,
    checkpoint: Path,
    alias: Path,
    certificate: dict,
    structure: dict,
    variant: str,
) -> bool:
    checkpoint_hash = sha256(checkpoint)
    alias_hash = sha256(alias)
    epoch = int(result.get("best_learned_epoch", -1))
    model_state = state.get("model_state")
    if not isinstance(model_state, Mapping):
        return False
    base_hash = _v3_base_state_sha256(model_state)
    return (
        checkpoint_hash == alias_hash
        and checkpoint_hash == result.get("best_checkpoint_sha256")
        and checkpoint_hash == result.get("best_learned_checkpoint_sha256")
        and result.get("best_learned_is_byte_exact_best_alias") is True
        and int(result.get("best_epoch", -2)) == epoch
        and int(state.get("epoch", -3)) == epoch
        and state.get("schema") == PATHS_OSWT_CHECKPOINT_SCHEMA
        and state.get("paths_oswt_protocol_version") == PATHS_OSWT_PROTOCOL_VERSION
        and state.get("oswt_variant") == variant
        and state.get("run_git_commit") == os.environ["PATHS_OSWT_GATE_HEAD"]
        and state.get("implementation_signature") == implementation
        and state.get("architecture_signature") == result.get("architecture_signature")
        and state.get("checkpoint_role") == "oswt_selected_learned"
        and state.get("warm_start_provenance") is None
        and state.get("source_v3_base_state_sha256") == base_hash
        and state.get("checkpoint_v3_base_state_sha256") == base_hash
        and state.get("checkpoint_v3_base_matches_source") is True
        and certificate.get("selected_checkpoint_sha256") == checkpoint_hash
        and structure.get("selected_checkpoint_sha256") == checkpoint_hash
        and int(certificate.get("checkpoint_epoch", -4)) == epoch
        and int(structure.get("checkpoint_epoch", -5)) == epoch
    )


def frozen_base_provenance_ok(
    *,
    result: dict,
    state: dict,
    identity_payload: dict,
    certificate: dict,
    structure: dict,
) -> bool:
    provenance = result.get("warm_start_provenance")
    state_provenance = state.get("paths_oswt_warm_start_provenance")
    if not isinstance(provenance, dict) or not isinstance(state_provenance, dict):
        return False
    model_state = state.get("model_state")
    if not isinstance(model_state, Mapping):
        return False
    source_checkpoint = Path(str(provenance.get("source_checkpoint", "")))
    source_checkpoint_hash = provenance.get("source_checkpoint_sha256")
    base_hash = _v3_base_state_sha256(model_state)
    recorded_base_hashes = (
        result.get("source_v3_base_state_sha256"),
        result.get("final_v3_base_state_sha256"),
        result.get("selected_v3_base_state_sha256"),
        state.get("source_v3_base_state_sha256"),
        state.get("checkpoint_v3_base_state_sha256"),
        identity_payload.get("source_v3_base_state_sha256"),
        identity_payload.get("control_v3_base_state_sha256"),
        certificate.get("source_v3_base_state_sha256"),
        certificate.get("final_v3_base_state_sha256"),
        certificate.get("selected_v3_base_state_sha256"),
        structure.get("source_v3_base_state_sha256"),
        structure.get("final_v3_base_state_sha256"),
        structure.get("selected_v3_base_state_sha256"),
    )
    return (
        normalized_provenance(provenance)
        == normalized_provenance(state_provenance)
        and source_checkpoint.is_file()
        and sha256(source_checkpoint) == source_checkpoint_hash
        and identity_payload.get("source_checkpoint_sha256") == source_checkpoint_hash
        and certificate.get("source_checkpoint_sha256") == source_checkpoint_hash
        and all(value == base_hash for value in recorded_base_hashes)
        and result.get("final_v3_base_matches_source") is True
        and result.get("selected_v3_base_matches_source") is True
        and state.get("checkpoint_v3_base_matches_source") is True
        and identity_payload.get("control_v3_base_matches_source") is True
        and certificate.get("final_v3_base_matches_source") is True
        and certificate.get("selected_v3_base_matches_source") is True
        and structure.get("final_v3_base_matches_source") is True
        and structure.get("selected_v3_base_matches_source") is True
    )


treatment_checkpoint_hash = sha256(treatment_checkpoint)
control_checkpoint_hash = sha256(control_checkpoint)
treatment_base_hash = _v3_base_state_sha256(treatment_state["model_state"])
control_base_hash = _v3_base_state_sha256(control_state["model_state"])
source_checkpoint_hash = identity.get("source_checkpoint_sha256")

checks = {
    "registered_protocol": treatment_result.get("protocol") == control_result.get("protocol") == PATHS_OSWT_PROTOCOL_VERSION,
    "registered_variants": treatment_result.get("oswt_variant") == "shell_warranted" and control_result.get("oswt_variant") == "shell_warranted_ungated",
    "validation_only": treatment_result.get("test_evaluated") is False and control_result.get("test_evaluated") is False,
    "same_git_commit": treatment_result.get("run_git_commit") == control_result.get("run_git_commit") == os.environ["PATHS_OSWT_GATE_HEAD"],
    "same_implementation": treatment_result.get("implementation_signature") == control_result.get("implementation_signature") == implementation,
    "same_split": treatment_result.get("split_signature") == control_result.get("split_signature") == expected_split,
    "same_hash_bound_source": (
        source_checkpoint_hash == control_identity.get("source_checkpoint_sha256")
        and normalized_provenance(treatment_result.get("warm_start_provenance"))
        == normalized_provenance(control_result.get("warm_start_provenance"))
        and treatment_base_hash == control_base_hash
    ),
    "matched_training_configuration": (
        matched_training_config(treatment_state["config"], control_state["config"])
        and matched_training_config(
            treatment_result.get("critical_config", {}),
            control_result.get("critical_config", {}),
        )
    ),
    "source_reproduced_twice": identity.get("reproduced") is True and control_identity.get("reproduced") is True,
    "identical_source_predictions": identity.get("ordered_prediction_checksum_sha256") == control_identity.get("ordered_prediction_checksum_sha256"),
    "v3_frozen_refiner_only_contract": (
        refiner_only_optimizer_contract(treatment_result.get("optimizer_group_contract"))
        and refiner_only_optimizer_contract(control_result.get("optimizer_group_contract"))
        and treatment_result.get("optimizer_group_contract")
        == treatment_state.get("optimizer_group_contract")
        and control_result.get("optimizer_group_contract")
        == control_state.get("optimizer_group_contract")
        and treatment_state.get("training_phase") == "oswt_refiner_only"
        and control_state.get("training_phase") == "oswt_refiner_only"
    ),
    "checkpoint_schema": treatment_state.get("schema") == control_state.get("schema") == PATHS_OSWT_CHECKPOINT_SCHEMA,
    "checkpoint_hashes_bound": checkpoint_binding_ok(
        result=treatment_result,
        state=treatment_state,
        checkpoint=treatment_checkpoint,
        alias=treatment_alias,
        certificate=treatment_certificate,
        structure=treatment_structure,
        variant="shell_warranted",
    ) and checkpoint_binding_ok(
        result=control_result,
        state=control_state,
        checkpoint=control_checkpoint,
        alias=control_alias,
        certificate=control_certificate,
        structure=control_structure,
        variant="shell_warranted_ungated",
    ),
    "frozen_v3_hash_chain_and_provenance": frozen_base_provenance_ok(
        result=treatment_result,
        state=treatment_state,
        identity_payload=identity,
        certificate=treatment_certificate,
        structure=treatment_structure,
    ) and frozen_base_provenance_ok(
        result=control_result,
        state=control_state,
        identity_payload=control_identity,
        certificate=control_certificate,
        structure=control_structure,
    ),
    "identity_controls_content_bound": identity_control_ok(
        identity,
        variant="shell_warranted",
        architecture=treatment_result.get("architecture_signature"),
        source_checkpoint_hash=source_checkpoint_hash,
        base_hash=treatment_base_hash,
    ) and identity_control_ok(
        control_identity,
        variant="shell_warranted_ungated",
        architecture=control_result.get("architecture_signature"),
        source_checkpoint_hash=source_checkpoint_hash,
        base_hash=control_base_hash,
    ),
    "audit_artifacts_content_bound": (
        treatment_result.get("validation_certificate_checksum")
        == treatment_certificate.get("content_checksum_sha256")
        and control_result.get("validation_certificate_checksum")
        == control_certificate.get("content_checksum_sha256")
        and treatment_result.get("validation_structural_audit") == treatment_structure
        and control_result.get("validation_structural_audit") == control_structure
    ),
    "certificate_experiment_bound": (
        treatment_certificate.get("implementation_signature") == implementation
        and control_certificate.get("implementation_signature") == implementation
        and treatment_certificate.get("architecture_signature") == treatment_result.get("architecture_signature")
        and control_certificate.get("architecture_signature") == control_result.get("architecture_signature")
        and treatment_certificate.get("split_signature") == control_certificate.get("split_signature") == expected_split
        and int(treatment_certificate.get("checkpoint_epoch", -1)) == int(treatment_result.get("best_learned_epoch", -2))
        and int(control_certificate.get("checkpoint_epoch", -1)) == int(control_result.get("best_learned_epoch", -2))
    ),
    "exact_deletion_certificates": certificate_ok(
        treatment_certificate,
        variant="shell_warranted",
        architecture=treatment_result.get("architecture_signature"),
        checkpoint_hash=treatment_checkpoint_hash,
        checkpoint_epoch=int(treatment_result.get("best_learned_epoch", -1)),
        source_checkpoint_hash=source_checkpoint_hash,
        base_hash=treatment_base_hash,
    ) and certificate_ok(
        control_certificate,
        variant="shell_warranted_ungated",
        architecture=control_result.get("architecture_signature"),
        checkpoint_hash=control_checkpoint_hash,
        checkpoint_epoch=int(control_result.get("best_learned_epoch", -1)),
        source_checkpoint_hash=source_checkpoint_hash,
        base_hash=control_base_hash,
    ),
    "full_validation_forward_identities": structural_ok(
        treatment_structure,
        variant="shell_warranted",
        architecture=treatment_result.get("architecture_signature"),
        checkpoint_hash=treatment_checkpoint_hash,
        checkpoint_epoch=int(treatment_result.get("best_learned_epoch", -1)),
        base_hash=treatment_base_hash,
    ) and structural_ok(
        control_structure,
        variant="shell_warranted_ungated",
        architecture=control_result.get("architecture_signature"),
        checkpoint_hash=control_checkpoint_hash,
        checkpoint_epoch=int(control_result.get("best_learned_epoch", -1)),
        base_hash=control_base_hash,
    ),
    "paired_metric_replay": metric_replay(treatment_result["best_learned_validation"], treatment_metrics) and metric_replay(control_result["best_learned_validation"], control_metrics),
    "paired_source_correct_replay": paired["source_correct"] == correct(source_metrics),
    "at_least_254_of_293": paired["treatment_correct"] >= 254,
    "strictly_beats_v3": paired["treatment_correct"] > paired["source_correct"],
    "strictly_beats_ungated": paired["treatment_correct"] > paired["ungated_correct"],
    "qwk_at_least_0_9265": float(treatment_metrics["qwk"]) >= 0.9265,
    "mae_at_most_0_1741": float(treatment_metrics["mae"]) <= 0.1741,
    "balanced_accuracy_non_regression": float(treatment_metrics["balanced_acc"]) >= float(source_metrics["balanced_acc"]),
    "macro_f1_non_regression": float(treatment_metrics["macro_f1"]) >= float(source_metrics["macro_f1"]),
    "exact_grade3_2_to_3_correction": paired["grade3_source_2_to_treatment_3"] >= 1,
    "exact_grade3_4_to_3_correction": paired["grade3_source_4_to_treatment_3"] >= 1,
    "grade3_2_to_3_is_positive_boundary2_transport": paired["grade3_2_to_3_with_positive_boundary2_flow"] >= 1,
    "grade3_4_to_3_is_negative_boundary3_transport": paired["grade3_4_to_3_with_negative_boundary3_flow"] >= 1,
    "no_correct_grade3_source_harmed": paired["grade3_source_correct_harmed"] == 0,
}

payload = {
    "schema": "paths-v8-oswt-complementary-aptos-paired-gate-v2",
    "protocol": PATHS_OSWT_PROTOCOL_VERSION,
    "git_commit": os.environ["PATHS_OSWT_GATE_HEAD"],
    "implementation_signature": implementation,
    "split_signature": expected_split,
    "source_checkpoint_sha256": identity["source_checkpoint_sha256"],
    "source_v3_base_state_sha256": treatment_base_hash,
    "treatment_checkpoint_sha256": treatment_checkpoint_hash,
    "ungated_checkpoint_sha256": control_checkpoint_hash,
    "treatment_certificate_checksum_sha256": treatment_certificate[
        "content_checksum_sha256"
    ],
    "ungated_certificate_checksum_sha256": control_certificate[
        "content_checksum_sha256"
    ],
    "treatment_structural_audit_checksum_sha256": treatment_structure[
        "content_checksum_sha256"
    ],
    "ungated_structural_audit_checksum_sha256": control_structure[
        "content_checksum_sha256"
    ],
    "checks": checks,
    "passed": all(checks.values()),
    "paired_exact_counts": paired,
    "paired_exact_sample_ids": paired_ids,
    "metrics": {
        "risk_gated_oswt": treatment_metrics,
        "ungated_shell_control": control_metrics,
        "static_v3": source_metrics,
    },
}
payload["content_checksum_sha256"] = content_hash(payload)
gate = treatment_dir / "PROMOTED_TO_EYEPACS.json"
gate.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")
print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))
if not payload["passed"]:
    raise SystemExit("PATHS-V8 OSWT APTOS gate failed; EyePACS remains blocked")
PY

echo "PATHS-V8 OSWT paired APTOS gate passed."
