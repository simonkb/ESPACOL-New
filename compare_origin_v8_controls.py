#!/usr/bin/env python3
"""Fail-closed comparison of the three matched ORIGIN-v8 APTOS pilots.

This utility compares the independently selected ``best_learned.pth``
checkpoint from the registered target, additive-endpoint null, and shuffled
endpoint-to-geometry null.  It deliberately refuses deployment ``best.pth``
artifacts: those are protected by the v3 safety floor and can therefore still
be the epoch-0 v3 model, which would not be a learned-control comparison.

The command is validation-only.  It never loads images and cannot evaluate the
outer test split.  Every arm must first have a production full-validation audit
of its exact ``best_learned.pth`` artifact.  The final development gate is
fail-closed: the target must pass the registered V3 metric floor, pass the
audited individual-witness strength gate, and achieve strictly higher
validation accuracy than both matched controls.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch


SCHEMA = "origin-v8-matched-control-comparison-v1"
EXPECTED_CHECKPOINT_SCHEMA = "origin-checkpoint-v8"
EXPECTED_AUDIT_SCHEMA = "origin-full-validation-audit-v8"
EXPECTED_AUDIT_FILENAME = "full_validation_audit_v8_best_learned.json"
EXPECTED_PROTOCOL = "v8"
EXPECTED_VARIANTS = {
    "target": "identified_conserved_witness_v1",
    "additive_endpoint_control": "additive_conserved_witness_control_v1",
    "shuffled_geometry_control": "shuffled_conserved_witness_control_v1",
}
EXPECTED_CONTROL_SEMANTICS = {
    "target": "identified_nonadditive_spatial_pair_witnesses",
    "additive_endpoint_control": "endpoint_main_effect_capacity_null",
    "shuffled_geometry_control": (
        "fixed_shuffled_endpoint_to_geometry_correspondence_null"
    ),
}
ALLOWED_CONFIG_DIFFERENCES = frozenset(("relation_variant", "run_dir"))
PRIMARY_METRICS = (
    "acc",
    "qwk",
    "mae",
    "balanced_acc",
    "macro_f1",
    "ece",
    "loss",
)
AUDIT_REPRODUCED_METRICS = (
    "acc",
    "qwk",
    "mae",
    "balanced_acc",
    "macro_f1",
    "ece",
)
EXPECTED_RELATION_AGGREGATION = "conserved_sparse_witness_allocation_v1"
EXPECTED_RELATION_CONTRACT = (
    "stored_edge_log_odds_contributions_to_cumulative_transition_rate_log_odds"
)
EXPECTED_INTERPRETATION_SCOPE = (
    "exact_stored_unary_and_pair_ledger_deletion_not_causal_pixel_masking"
)
V8_TRACE_ERROR_TOLERANCE = 1.0e-4
EXACT_REPLAY_ERROR_TOLERANCE = 2.0e-10


class ComparisonContractError(ValueError):
    """Raised when runs are not a valid matched-control comparison."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical(value: Any) -> Any:
    """Convert checkpoint/config values into a stable JSON-compatible form."""

    if isinstance(value, Mapping):
        return {str(key): _canonical(item) for key, item in sorted(value.items())}
    if isinstance(value, (tuple, list)):
        return [_canonical(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, bool, int)) or value is None:
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ComparisonContractError("comparison metadata contains non-finite floats")
        return value
    if torch.is_tensor(value):
        if value.numel() != 1:
            raise ComparisonContractError(
                "only scalar tensors are valid in comparison metadata"
            )
        return _canonical(value.detach().cpu().item())
    # NumPy scalar types expose ``item`` without requiring NumPy here.
    item = getattr(value, "item", None)
    if callable(item):
        return _canonical(item())
    raise ComparisonContractError(
        f"unsupported comparison metadata type: {type(value).__name__}"
    )


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        _canonical(value), sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _resolve_fold_dir(path: str | Path) -> Path:
    candidate = Path(path).expanduser().resolve()
    if (candidate / "result.json").is_file():
        return candidate
    fold0 = candidate / "fold0"
    if (fold0 / "result.json").is_file():
        return fold0
    raise FileNotFoundError(
        f"{candidate} is neither a completed fold directory nor a run with fold0"
    )


def _load_json(path: Path) -> Mapping[str, Any]:
    with path.open(encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, Mapping):
        raise ComparisonContractError(f"{path} must contain a JSON object")
    return value


def _require_mapping(value: Any, *, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ComparisonContractError(f"{path} must be a JSON object")
    return value


def _values_close(observed: Any, expected: Any, *, path: str) -> None:
    if isinstance(observed, Mapping) and isinstance(expected, Mapping):
        if set(observed) != set(expected):
            raise ComparisonContractError(
                f"{path} keys differ: observed={sorted(observed)}, expected={sorted(expected)}"
            )
        for key in observed:
            _values_close(observed[key], expected[key], path=f"{path}.{key}")
        return
    if isinstance(observed, Sequence) and not isinstance(observed, (str, bytes)):
        if not isinstance(expected, Sequence) or isinstance(expected, (str, bytes)):
            raise ComparisonContractError(f"{path} type differs")
        if len(observed) != len(expected):
            raise ComparisonContractError(f"{path} length differs")
        for index, (left, right) in enumerate(zip(observed, expected)):
            _values_close(left, right, path=f"{path}[{index}]")
        return
    if isinstance(observed, (int, float)) and isinstance(expected, (int, float)):
        if not math.isclose(
            float(observed), float(expected), rel_tol=1.0e-7, abs_tol=1.0e-8
        ):
            raise ComparisonContractError(
                f"{path} differs: observed={observed!r}, expected={expected!r}"
            )
        return
    if observed != expected:
        raise ComparisonContractError(
            f"{path} differs: observed={observed!r}, expected={expected!r}"
        )


def _metrics_close(
    observed: Mapping[str, Any], expected: Mapping[str, Any], *, path: str
) -> None:
    """Match the independent CUDA replay tolerance used by the production audit."""

    for name in AUDIT_REPRODUCED_METRICS:
        if name not in observed or name not in expected:
            raise ComparisonContractError(f"{path} omits required metric {name!r}")
        left = float(observed[name])
        right = float(expected[name])
        if not math.isfinite(left) or not math.isfinite(right) or not math.isclose(
            left, right, rel_tol=1.0e-5, abs_tol=1.0e-5
        ):
            raise ComparisonContractError(
                f"{path}.{name} does not reproduce: observed={left}, expected={right}"
            )


def _verify_content_checksum(payload: Mapping[str, Any], *, path: Path) -> str:
    claimed = payload.get("content_checksum_sha256")
    if not isinstance(claimed, str) or len(claimed) != 64:
        raise ComparisonContractError(f"{path} has no valid content checksum")
    body = dict(payload)
    del body["content_checksum_sha256"]
    observed = _canonical_sha256(body)
    if observed != claimed:
        raise ComparisonContractError(f"{path} content checksum does not reproduce")
    return claimed


def _verify_v8_audit(
    *,
    role: str,
    fold_dir: Path,
    checkpoint_sha256: str,
    state: Mapping[str, Any],
    result: Mapping[str, Any],
    metrics: Mapping[str, Any],
    config: Mapping[str, Any],
    history_budget: Mapping[str, Any],
) -> Mapping[str, Any]:
    """Verify the independent full-validation replay for one learned arm."""

    audit_path = fold_dir / "audits" / EXPECTED_AUDIT_FILENAME
    if not audit_path.is_file():
        raise FileNotFoundError(
            f"{role} has no required V8 best-learned audit: {audit_path}"
        )
    audit = _load_json(audit_path)
    checksum = _verify_content_checksum(audit, path=audit_path)
    if audit.get("schema") != EXPECTED_AUDIT_SCHEMA:
        raise ComparisonContractError(
            f"{role} audit schema={audit.get('schema')!r}; "
            f"expected {EXPECTED_AUDIT_SCHEMA!r}"
        )
    if audit.get("scope") != "inner_validation_only":
        raise ComparisonContractError(f"{role} audit is not inner-validation-only")
    if int(audit.get("fold", -1)) != 0:
        raise ComparisonContractError(f"{role} audit is not APTOS fold 0")
    if audit.get("checkpoint_role") != "best_learned":
        raise ComparisonContractError(
            f"{role} audit did not evaluate the best_learned checkpoint role"
        )
    if audit.get("checkpoint_sha256") != checkpoint_sha256:
        raise ComparisonContractError(
            f"{role} audit checkpoint hash differs from best_learned.pth"
        )
    if int(audit.get("checkpoint_epoch", -1)) != int(state.get("epoch", -2)):
        raise ComparisonContractError(f"{role} audit checkpoint epoch differs")
    if audit.get("split_signature") != state.get("split_signature"):
        raise ComparisonContractError(f"{role} audit split signature differs")
    for key in (
        "architecture_signature",
        "implementation_signature",
        "checkpoint_config_signature",
    ):
        state_key = "config_signature" if key == "checkpoint_config_signature" else key
        if audit.get(key) != state.get(state_key):
            raise ComparisonContractError(f"{role} audit {key} differs from checkpoint")

    expected_audit_code = _sha256_file(
        Path(__file__).resolve().parent / "audit_origin_validation.py"
    )
    if audit.get("audit_implementation_sha256") != expected_audit_code:
        raise ComparisonContractError(
            f"{role} audit implementation hash differs from this registered commit"
        )
    reproduction = _require_mapping(
        audit.get("metric_reproduction"), path=f"{role}.audit.metric_reproduction"
    )
    if reproduction.get("checkpoint") is not True or reproduction.get(
        "completed_result"
    ) is not True:
        raise ComparisonContractError(
            f"{role} audit did not reproduce checkpoint and result metrics"
        )
    audit_metrics = _require_mapping(
        audit.get("metrics"), path=f"{role}.audit.metrics"
    )
    _metrics_close(audit_metrics, metrics, path=f"{role}.audit.metrics")
    if int(audit.get("n", -1)) != int(history_budget["validation_samples_per_epoch"]):
        raise ComparisonContractError(f"{role} audit validation sample count differs")
    sample_ids = audit.get("validation_sample_ids")
    if (
        not isinstance(sample_ids, list)
        or len(sample_ids) != int(audit["n"])
        or len(set(sample_ids)) != len(sample_ids)
        or sorted(sample_ids) != list(range(int(audit["n"])))
    ):
        raise ComparisonContractError(
            f"{role} audit validation sample IDs are missing or non-unique"
        )
    if audit.get("decision_rule") != config.get("decision_rule"):
        raise ComparisonContractError(f"{role} audit decision rule differs")

    floor_evaluation = _require_mapping(
        state.get("candidate_metric_safety_floor_evaluation"),
        path=f"{role}.candidate_metric_safety_floor_evaluation",
    )
    result_floor_evaluation = _require_mapping(
        result.get("best_learned_metric_safety_floor_evaluation"),
        path=f"{role}.best_learned_metric_safety_floor_evaluation",
    )
    _values_close(
        result_floor_evaluation,
        floor_evaluation,
        path=f"{role}.best_learned_metric_safety_floor_evaluation",
    )
    audit_floor_evaluation = _require_mapping(
        audit.get("selected_checkpoint_metric_safety_floor_evaluation"),
        path=f"{role}.audit.selected_checkpoint_metric_safety_floor_evaluation",
    )
    _values_close(
        audit_floor_evaluation,
        floor_evaluation,
        path=f"{role}.audit.selected_checkpoint_metric_safety_floor_evaluation",
    )
    passes_floor = bool(floor_evaluation.get("passes_metric_floor", False))
    if bool(result.get("best_learned_passes_v3_safety_floor", False)) != passes_floor:
        raise ComparisonContractError(
            f"{role} result safety-floor flag differs from its checkpoint evaluation"
        )
    _values_close(
        audit.get("warm_start_metric_safety_floor"),
        state.get("warm_start_metric_safety_floor"),
        path=f"{role}.audit.warm_start_metric_safety_floor",
    )
    if audit.get("selected_checkpoint_training_phase") != state.get("training_phase"):
        raise ComparisonContractError(f"{role} audit training phase differs")

    if audit.get("interpretation_scope") != EXPECTED_INTERPRETATION_SCOPE:
        raise ComparisonContractError(f"{role} audit interpretation scope differs")
    diagnostics = _require_mapping(
        audit.get("relation_interaction_diagnostics"),
        path=f"{role}.audit.relation_interaction_diagnostics",
    )
    if diagnostics.get("variant") != EXPECTED_VARIANTS[role]:
        raise ComparisonContractError(f"{role} audit relation variant differs")
    if diagnostics.get("aggregation_kind") != EXPECTED_RELATION_AGGREGATION:
        raise ComparisonContractError(f"{role} audit aggregation contract differs")
    if diagnostics.get("contract") != EXPECTED_RELATION_CONTRACT:
        raise ComparisonContractError(f"{role} audit ledger contract differs")
    if not math.isclose(
        float(diagnostics.get("cumulative_log_odds_bound", math.nan)),
        float(config.get("relation_delta_cap", math.nan)),
        rel_tol=0.0,
        abs_tol=0.0,
    ):
        raise ComparisonContractError(f"{role} audited relation bound differs")
    sparse = _require_mapping(
        diagnostics.get("sparse_support_and_identifiability"),
        path=f"{role}.audit.sparse_support_and_identifiability",
    )
    if int(sparse.get("edge_budget_per_boundary", -1)) != 8:
        raise ComparisonContractError(f"{role} audited witness cap differs from 8")
    expected_trace_errors = (
        "max_topq_ranking_violation",
        "max_allocation_sum_error",
        "max_entmax_allocation_replay_error",
        "max_effective_witness_count_identity_error",
        "max_identified_projection_replay_error",
        "max_identified_residual_row_or_column_sum",
        "max_message_identity_error",
        "max_edge_contribution_identity_error",
        "max_sparse_aggregation_identity_error",
        "max_allocated_capacity_identity_error",
        "max_l1_usage_identity_error",
        "max_per_boundary_budget_violation",
        "max_cumulative_budget_violation",
    )
    trace_errors: dict[str, float] = {}
    for name in expected_trace_errors:
        value = float(sparse.get(name, math.nan))
        if not math.isfinite(value) or abs(value) > V8_TRACE_ERROR_TOLERANCE:
            raise ComparisonContractError(
                f"{role} audited V8 invariant {name} failed: {value}"
            )
        trace_errors[name] = value
    exact_replay_names = (
        "max_exact_source_ledger_invariance_error",
        "max_exact_edge_partition_error",
        "max_exact_relation_reaggregation_error",
        "max_exact_relation_merge_error",
        "max_all_edges_removed_to_base_rate_error",
    )
    exact_replay_errors: dict[str, float] = {}
    for name in exact_replay_names:
        value = float(diagnostics.get(name, math.nan))
        if not math.isfinite(value) or abs(value) > EXACT_REPLAY_ERROR_TOLERANCE:
            raise ComparisonContractError(
                f"{role} audited exact replay invariant {name} failed: {value}"
            )
        exact_replay_errors[name] = value
    for name in exact_replay_names[:2]:
        if exact_replay_errors[name] != 0.0:
            raise ComparisonContractError(
                f"{role} audited exact invariant {name} is not identically zero"
            )
    total_rate_replay_error = float(
        audit.get("max_exact_total_rate_replay_error", math.nan)
    )
    if (
        not math.isfinite(total_rate_replay_error)
        or abs(total_rate_replay_error) > EXACT_REPLAY_ERROR_TOLERANCE
    ):
        raise ComparisonContractError(
            f"{role} unary-ledger total-rate replay invariant failed"
        )
    for summary_name in (
        "all_edge_deletion_effects",
        "most_pivotal_selected_edge_deletion_effects",
    ):
        summary = _require_mapping(
            diagnostics.get(summary_name), path=f"{role}.audit.{summary_name}"
        )
        if int(summary.get("n", -1)) != int(audit["n"]):
            raise ComparisonContractError(
                f"{role} audit {summary_name} does not cover validation set"
            )
        if summary_name == "all_edge_deletion_effects":
            effect_total = sum(
                int(summary.get(name, -int(audit["n"])))
                for name in (
                    "positive_effect_count_above_1e-5",
                    "negative_effect_count_below_minus_1e-5",
                    "near_zero_effect_count_at_1e-5",
                )
            )
            if effect_total != int(audit["n"]):
                raise ComparisonContractError(
                    f"{role} audit collective effect partition is incomplete"
                )
    ablation = _require_mapping(
        diagnostics.get("full_vs_no_relation_ablation"),
        path=f"{role}.audit.full_vs_no_relation_ablation",
    )
    if int(ablation.get("n", -1)) != int(audit["n"]):
        raise ComparisonContractError(
            f"{role} no-relation ablation does not cover validation set"
        )
    no_relation_metrics = _require_mapping(
        ablation.get("no_relation_metrics"),
        path=f"{role}.audit.full_vs_no_relation_ablation.no_relation_metrics",
    )
    grade_metrics = _require_mapping(
        audit.get("grade_stratified_metrics"),
        path=f"{role}.audit.grade_stratified_metrics",
    )
    if sum(int(item["support"]) for item in grade_metrics.values()) != int(audit["n"]):
        raise ComparisonContractError(f"{role} audited grade supports do not sum to n")
    strength_gate = _require_mapping(
        diagnostics.get("individual_certificate_strength_gate"),
        path=f"{role}.audit.individual_certificate_strength_gate",
    )
    strength_checks = _require_mapping(
        strength_gate.get("checks"),
        path=f"{role}.audit.individual_certificate_strength_gate.checks",
    )
    reproduced_strength_pass = bool(strength_checks) and all(
        value is True for value in strength_checks.values()
    )
    if strength_gate.get("passed") is not reproduced_strength_pass:
        raise ComparisonContractError(
            f"{role} audit individual-certificate strength gate is inconsistent"
        )
    _require_mapping(
        diagnostics.get("conserved_witness_allocation"),
        path=f"{role}.audit.conserved_witness_allocation",
    )
    return {
        "path": str(audit_path),
        "content_checksum_sha256": checksum,
        "audit_implementation_sha256": expected_audit_code,
        "metrics": dict(audit_metrics),
        "sample_ids_sha256": _canonical_sha256(sample_ids),
        "audit_settings": dict(
            _require_mapping(
                audit.get("audit_settings"), path=f"{role}.audit.audit_settings"
            )
        ),
        "trace_replay_errors": trace_errors,
        "exact_deletion_replay_errors": exact_replay_errors,
        "max_exact_total_rate_replay_error": total_rate_replay_error,
        "candidate_edge_count_summary_sha256": _canonical_sha256(
            sparse.get("candidate_edge_count_quantiles")
        ),
        "no_relation_metrics": dict(no_relation_metrics),
        "individual_certificate_strength_gate": {
            "passed": reproduced_strength_pass,
            "checks": dict(strength_checks),
        },
        "passes_v3_safety_floor": passes_floor,
    }


def _relation_state_signature(state: Mapping[str, Any]) -> tuple[tuple[str, tuple[int, ...], str], ...]:
    model_state = state.get("model_state")
    if not isinstance(model_state, Mapping):
        raise ComparisonContractError("checkpoint model_state is missing")
    signature = []
    for name, tensor in model_state.items():
        if not str(name).startswith("generator.relation_field."):
            continue
        if not torch.is_tensor(tensor):
            raise ComparisonContractError(f"relation state {name!r} is not a tensor")
        signature.append((str(name), tuple(tensor.shape), str(tensor.dtype)))
    if not signature:
        raise ComparisonContractError("checkpoint has no relation-field state")
    return tuple(sorted(signature))


def _history_budget(fold_dir: Path, config: Mapping[str, Any]) -> Mapping[str, Any]:
    path = fold_dir / "history.csv"
    if not path.is_file():
        raise FileNotFoundError(f"matched comparison requires {path}")
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ComparisonContractError(f"{path} contains no epochs")
    epochs = [int(row["epoch"]) for row in rows]
    configured_epochs = int(config["epochs"])
    expected_epochs = list(range(1, configured_epochs + 1))
    if epochs != expected_epochs:
        raise ComparisonContractError(
            f"{path} did not consume the fixed epoch budget: "
            f"observed={epochs}, expected={expected_epochs}"
        )
    final = rows[-1]
    skipped = int(float(final.get("train_amp_total_skipped_steps", "0") or 0))
    if skipped != 0:
        raise ComparisonContractError(
            f"{path} recorded {skipped} skipped optimizer steps; "
            "the strict matched-control comparison requires zero"
        )
    required = ("train_n", "val_n")
    sample_counts: dict[str, int] = {}
    for name in required:
        values = {int(float(row[name])) for row in rows if row.get(name, "") != ""}
        if len(values) != 1:
            raise ComparisonContractError(f"{path} has inconsistent {name}")
        sample_counts[name] = values.pop()
    return {
        "epochs": configured_epochs,
        "batch_size": int(config["batch_size"]),
        "train_samples_per_epoch": sample_counts["train_n"],
        "validation_samples_per_epoch": sample_counts["val_n"],
        "amp_total_skipped_steps": skipped,
    }


def _load_run(role: str, supplied_path: str | Path) -> Mapping[str, Any]:
    fold_dir = _resolve_fold_dir(supplied_path)
    result_path = fold_dir / "result.json"
    checkpoint_path = fold_dir / "best_learned.pth"
    result = _load_json(result_path)
    if bool(result.get("test_evaluated", False)):
        raise ComparisonContractError(
            f"{role} evaluated the outer test; this development comparison refuses it"
        )
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"{role} has no best_learned.pth; deployment best.pth is not a valid control"
        )
    checkpoint_sha256 = _sha256_file(checkpoint_path)
    expected_hash = result.get("best_learned_checkpoint_sha256")
    if not isinstance(expected_hash, str) or checkpoint_sha256 != expected_hash:
        raise ComparisonContractError(
            f"{role} best_learned checkpoint hash does not match result.json"
        )
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(state, Mapping):
        raise ComparisonContractError(f"{role} checkpoint payload is not a mapping")
    if state.get("schema") != EXPECTED_CHECKPOINT_SCHEMA:
        raise ComparisonContractError(
            f"{role} schema={state.get('schema')!r}; expected {EXPECTED_CHECKPOINT_SCHEMA!r}"
        )
    if state.get("relation_protocol_version") != EXPECTED_PROTOCOL:
        raise ComparisonContractError(f"{role} is not relation protocol v8")
    expected_variant = EXPECTED_VARIANTS[role]
    variant = str(state.get("relation_variant", ""))
    config = state.get("config")
    if not isinstance(config, Mapping):
        raise ComparisonContractError(f"{role} checkpoint config is missing")
    if (
        variant != expected_variant
        or result.get("relation_variant") != expected_variant
        or config.get("relation_variant") != expected_variant
    ):
        raise ComparisonContractError(
            f"{role} variant mismatch: checkpoint={variant!r}, "
            f"result={result.get('relation_variant')!r}, "
            f"config={config.get('relation_variant')!r}, expected={expected_variant!r}"
        )
    if state.get("checkpoint_role") != "best_learned":
        raise ComparisonContractError(
            f"{role} checkpoint_role must be 'best_learned', got "
            f"{state.get('checkpoint_role')!r}"
        )
    epoch = int(state.get("epoch", 0))
    if epoch <= 0 or int(result.get("best_learned_epoch", -1)) != epoch:
        raise ComparisonContractError(
            f"{role} best learned epoch is missing, zero, or inconsistent"
        )
    if int(state.get("best_learned_epoch", -1)) != epoch:
        raise ComparisonContractError(
            f"{role} checkpoint does not identify itself as the selected learned epoch"
        )
    if state.get("best_learned_selection_key") is None:
        raise ComparisonContractError(f"{role} best learned selector state is missing")
    if state.get("early_stopping_track") != "learned_epoch_selector":
        raise ComparisonContractError(f"{role} used the wrong early-stopping track")
    metrics = result.get("best_learned_validation")
    if not isinstance(metrics, Mapping):
        raise ComparisonContractError(f"{role} result omits best_learned_validation")
    state_metrics = state.get("metrics")
    if not isinstance(state_metrics, Mapping):
        raise ComparisonContractError(f"{role} checkpoint omits validation metrics")
    _values_close(metrics, state_metrics, path=f"{role}.best_learned_validation")
    for metric in PRIMARY_METRICS:
        value = float(metrics[metric])
        if not math.isfinite(value):
            raise ComparisonContractError(f"{role} metric {metric} is non-finite")

    if str(config.get("dataset")) != "aptos" or int(state.get("fold", -1)) != 0:
        raise ComparisonContractError("V8 matched pilot must be APTOS fold 0")
    if not bool(config.get("relation_enabled", False)):
        raise ComparisonContractError(f"{role} relation field is disabled")
    if int(config.get("relation_edge_budget", -1)) != 8:
        raise ComparisonContractError(f"{role} witness cap must equal 8")
    if not math.isclose(
        float(config.get("relation_allocation_temperature", math.nan)),
        1.0,
        rel_tol=0.0,
        abs_tol=0.0,
    ):
        raise ComparisonContractError(f"{role} allocation temperature must equal 1.0")
    if int(config.get("relation_only_epochs", -1)) != int(config.get("epochs", -2)):
        raise ComparisonContractError(f"{role} must freeze the v3 base for every pilot epoch")
    provenance = state.get("warm_start_provenance")
    if not isinstance(provenance, Mapping):
        raise ComparisonContractError(f"{role} warm-start provenance is missing")
    source_hash = provenance.get("source_checkpoint_sha256")
    if not isinstance(source_hash, str) or len(source_hash) != 64:
        raise ComparisonContractError(f"{role} v3 source hash is invalid")
    if config.get("warm_start_sha256") != source_hash:
        raise ComparisonContractError(f"{role} config and provenance v3 hashes differ")
    if provenance.get("target_protocol") != "v8":
        raise ComparisonContractError(f"{role} warm start does not target protocol v8")
    if provenance.get("target_relation_variant") != expected_variant:
        raise ComparisonContractError(
            f"{role} warm-start provenance relation variant differs"
        )
    if provenance.get("source_split_signature") != state.get("split_signature"):
        raise ComparisonContractError(
            f"{role} v3 source and target split signatures differ"
        )

    history_budget = _history_budget(fold_dir, config)
    audit = _verify_v8_audit(
        role=role,
        fold_dir=fold_dir,
        checkpoint_sha256=checkpoint_sha256,
        state=state,
        result=result,
        metrics=metrics,
        config=config,
        history_budget=history_budget,
    )

    return {
        "role": role,
        "control_semantics": EXPECTED_CONTROL_SEMANTICS[role],
        "fold_dir": str(fold_dir),
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_sha256,
        "epoch": epoch,
        "metrics": dict(metrics),
        "state": state,
        "config": dict(config),
        "source_hash": source_hash,
        "split_signature": str(state.get("split_signature")),
        "implementation_signature": str(state.get("implementation_signature", "")),
        "selection_policy": str(state.get("selection_policy", "")),
        "selection_qwk_weight": float(state.get("selection_qwk_weight", math.nan)),
        "relation_state_signature": _relation_state_signature(state),
        "history_budget": history_budget,
        "audit": audit,
        "passes_v3_safety_floor": audit["passes_v3_safety_floor"],
    }


def _shared_config(config: Mapping[str, Any]) -> Mapping[str, Any]:
    return {
        key: value
        for key, value in config.items()
        if key not in ALLOWED_CONFIG_DIFFERENCES
    }


def compare_runs(
    target: str | Path,
    additive_endpoint_control: str | Path,
    shuffled_geometry_control: str | Path,
) -> Mapping[str, Any]:
    runs = {
        "target": _load_run("target", target),
        "additive_endpoint_control": _load_run(
            "additive_endpoint_control", additive_endpoint_control
        ),
        "shuffled_geometry_control": _load_run(
            "shuffled_geometry_control", shuffled_geometry_control
        ),
    }
    target_run = runs["target"]
    for role, run in runs.items():
        if run["split_signature"] != target_run["split_signature"]:
            raise ComparisonContractError(f"{role} split signature is not matched")
        if run["source_hash"] != target_run["source_hash"]:
            raise ComparisonContractError(f"{role} v3 source checkpoint hash is not matched")
        if (
            not run["implementation_signature"]
            or run["implementation_signature"]
            != target_run["implementation_signature"]
        ):
            raise ComparisonContractError(f"{role} implementation signature is not matched")
        if (
            run["selection_policy"] != target_run["selection_policy"]
            or run["selection_qwk_weight"] != target_run["selection_qwk_weight"]
        ):
            raise ComparisonContractError(f"{role} checkpoint selector is not matched")
        if run["relation_state_signature"] != target_run["relation_state_signature"]:
            raise ComparisonContractError(
                f"{role} relation-field state names/shapes/dtypes are not parameter matched"
            )
        if run["history_budget"] != target_run["history_budget"]:
            raise ComparisonContractError(f"{role} executed training budget is not matched")
        if run["audit"]["audit_settings"] != target_run["audit"]["audit_settings"]:
            raise ComparisonContractError(f"{role} audit settings are not matched")
        if run["audit"]["sample_ids_sha256"] != target_run["audit"]["sample_ids_sha256"]:
            raise ComparisonContractError(
                f"{role} audited validation sample identities are not matched"
            )
        if (
            run["audit"]["candidate_edge_count_summary_sha256"]
            != target_run["audit"]["candidate_edge_count_summary_sha256"]
        ):
            raise ComparisonContractError(
                f"{role} audited candidate-edge population is not matched"
            )
        _metrics_close(
            run["audit"]["no_relation_metrics"],
            target_run["audit"]["no_relation_metrics"],
            path=f"{role}.audit.no_relation_metrics",
        )
        observed_shared = _canonical(_shared_config(run["config"]))
        expected_shared = _canonical(_shared_config(target_run["config"]))
        if observed_shared != expected_shared:
            differing = sorted(
                key
                for key in set(observed_shared) | set(expected_shared)
                if observed_shared.get(key) != expected_shared.get(key)
            )
            raise ComparisonContractError(
                f"{role} hyperparameters differ outside the registered variant: {differing}"
            )

    serialized_runs: dict[str, Any] = {}
    for role, run in runs.items():
        serialized_runs[role] = {
            key: _canonical(value)
            for key, value in run.items()
            if key not in {"state", "config", "relation_state_signature"}
        }
        serialized_runs[role]["variant"] = EXPECTED_VARIANTS[role]
        serialized_runs[role]["relation_state_signature_sha256"] = _canonical_sha256(
            run["relation_state_signature"]
        )

    target_metrics = runs["target"]["metrics"]
    deltas: dict[str, Any] = {}
    for role in ("additive_endpoint_control", "shuffled_geometry_control"):
        control_metrics = runs[role]["metrics"]
        deltas[role] = {
            f"target_minus_control_{metric}": (
                float(target_metrics[metric]) - float(control_metrics[metric])
            )
            for metric in PRIMARY_METRICS
        }
    gate = {
        "target_passes_v3_safety_floor": bool(
            runs["target"]["passes_v3_safety_floor"]
        ),
        "target_passes_individual_certificate_strength_gate": bool(
            runs["target"]["audit"]["individual_certificate_strength_gate"][
                "passed"
            ]
        ),
        "target_accuracy_strictly_above_additive_control": (
            float(target_metrics["acc"])
            > float(runs["additive_endpoint_control"]["metrics"]["acc"])
        ),
        "target_accuracy_strictly_above_shuffled_control": (
            float(target_metrics["acc"])
            > float(runs["shuffled_geometry_control"]["metrics"]["acc"])
        ),
    }
    gate["target_accuracy_above_both_controls"] = (
        gate["target_accuracy_strictly_above_additive_control"]
        and gate["target_accuracy_strictly_above_shuffled_control"]
    )
    gate["passed"] = (
        gate["target_passes_v3_safety_floor"]
        and gate["target_passes_individual_certificate_strength_gate"]
        and gate["target_accuracy_above_both_controls"]
    )
    return {
        "schema": SCHEMA,
        "scope": "aptos_fold0_inner_validation_only",
        "outer_test_evaluated": False,
        "registered_variants": EXPECTED_VARIANTS,
        "allowed_config_differences": sorted(ALLOWED_CONFIG_DIFFERENCES),
        "shared_contract": {
            "split_signature": target_run["split_signature"],
            "v3_source_checkpoint_sha256": target_run["source_hash"],
            "shared_config_sha256": _canonical_sha256(
                _shared_config(target_run["config"])
            ),
            "relation_state_signature_sha256": _canonical_sha256(
                target_run["relation_state_signature"]
            ),
            "executed_training_budget": target_run["history_budget"],
            "witness_cap": 8,
            "allocation_temperature": 1.0,
        },
        "runs": serialized_runs,
        "target_minus_control": deltas,
        "development_gate": gate,
        "interpretation": (
            "The shuffled control nulls fixed endpoint-to-geometry correspondence; "
            "it does not remove or independently shuffle the complete set of "
            "within-image content pairs. This is a validation-only development "
            "comparison, not an outer-test or full-cross-validation claim. "
            "Passing requires the target learned checkpoint to pass its V3 "
            "metric floor, pass the audited individual-witness strength gate, "
            "and strictly exceed both controls in validation accuracy."
        ),
    }


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", required=True, help="target run or fold0 directory")
    parser.add_argument(
        "--additive_endpoint_control", required=True, help="additive-control run"
    )
    parser.add_argument(
        "--shuffled_geometry_control", required=True, help="shuffled-control run"
    )
    parser.add_argument("--output", default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    payload = compare_runs(
        args.target,
        args.additive_endpoint_control,
        args.shuffled_geometry_control,
    )
    if args.output:
        output = Path(args.output).expanduser().resolve()
        if output.exists() and not args.overwrite:
            raise FileExistsError(f"refusing to overwrite comparison: {output}")
        _write_json_atomic(output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))
    if not payload["development_gate"]["passed"]:
        raise SystemExit(10)


if __name__ == "__main__":
    main()
