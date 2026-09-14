from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import pytest
import torch

from compare_origin_v8_controls import (
    ComparisonContractError,
    EXPECTED_AUDIT_FILENAME,
    EXPECTED_VARIANTS,
    compare_runs,
)


ROOT = Path(__file__).resolve().parents[1]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_history(fold_dir: Path, *, epochs: int = 2, skipped: int = 0) -> None:
    with (fold_dir / "history.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=(
                "epoch",
                "train_n",
                "val_n",
                "train_amp_total_skipped_steps",
            ),
        )
        writer.writeheader()
        for epoch in range(1, epochs + 1):
            writer.writerow(
                {
                    "epoch": epoch,
                    "train_n": 2636,
                    "val_n": 293,
                    "train_amp_total_skipped_steps": skipped,
                }
            )


def _write_audit(
    fold_dir: Path,
    *,
    state: dict[str, object],
    metrics: dict[str, float],
    checkpoint_sha256: str,
    audit_updates: dict[str, object] | None = None,
) -> None:
    config = state["config"]
    assert isinstance(config, dict)
    variant = str(state["relation_variant"])
    trace_errors = {
        name: 0.0
        for name in (
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
    }
    exact_errors = {
        name: 0.0
        for name in (
            "max_exact_source_ledger_invariance_error",
            "max_exact_edge_partition_error",
            "max_exact_relation_reaggregation_error",
            "max_exact_relation_merge_error",
            "max_all_edges_removed_to_base_rate_error",
        )
    }
    audit: dict[str, object] = {
        "schema": "origin-full-validation-audit-v8",
        "scope": "inner_validation_only",
        "n": 293,
        "fold": 0,
        "decision_rule": config["decision_rule"],
        "split_signature": state["split_signature"],
        "checkpoint_epoch": state["epoch"],
        "checkpoint_role": "best_learned",
        "checkpoint_sha256": checkpoint_sha256,
        "architecture_signature": state["architecture_signature"],
        "implementation_signature": state["implementation_signature"],
        "checkpoint_config_signature": state["config_signature"],
        "warm_start_metric_safety_floor": state[
            "warm_start_metric_safety_floor"
        ],
        "selected_checkpoint_metric_safety_floor_evaluation": state[
            "candidate_metric_safety_floor_evaluation"
        ],
        "selected_checkpoint_training_phase": state["training_phase"],
        "audit_implementation_sha256": _sha256(ROOT / "audit_origin_validation.py"),
        "audit_settings": {
            "batch_size": config["batch_size"],
            "num_workers": 8,
            "top_ks": [1, 5, 10],
            "certificates_per_grade": 2,
            "amp": True,
            "decoder_precision": "fp64",
        },
        "metrics": {
            name: metrics[name]
            for name in (
                "acc",
                "qwk",
                "mae",
                "balanced_acc",
                "macro_f1",
                "ece",
            )
        },
        "grade_stratified_metrics": {
            "0": {"support": 100},
            "1": {"support": 50},
            "2": {"support": 60},
            "3": {"support": 40},
            "4": {"support": 43},
        },
        "metric_reproduction": {"checkpoint": True, "completed_result": True},
        "validation_sample_ids": list(range(293)),
        "max_exact_total_rate_replay_error": 0.0,
        "interpretation_scope": (
            "exact_stored_unary_and_pair_ledger_deletion_not_causal_pixel_masking"
        ),
        "relation_interaction_diagnostics": {
            "contract": (
                "stored_edge_log_odds_contributions_to_cumulative_transition_rate_log_odds"
            ),
            "variant": variant,
            "aggregation_kind": "conserved_sparse_witness_allocation_v1",
            "cumulative_log_odds_bound": config["relation_delta_cap"],
            "sparse_support_and_identifiability": {
                "edge_budget_per_boundary": 8,
                "candidate_edge_count_quantiles": {"q0": 90.0, "q100": 90.0},
                **trace_errors,
            },
            "conserved_witness_allocation": {"semantics": "unit capacity"},
            "all_edge_deletion_effects": {
                "n": 293,
                "positive_effect_count_above_1e-5": 100,
                "negative_effect_count_below_minus_1e-5": 100,
                "near_zero_effect_count_at_1e-5": 93,
            },
            "most_pivotal_selected_edge_deletion_effects": {"n": 293},
            "full_vs_no_relation_ablation": {
                "n": 293,
                "no_relation_metrics": {
                    "acc": 84.0,
                    "qwk": 0.90,
                    "mae": 0.20,
                    "balanced_acc": 62.0,
                    "macro_f1": 0.61,
                    "ece": 0.04,
                },
            },
            "individual_certificate_strength_gate": {
                "checks": {
                    "at_least_one_top_edge_changes_a_prediction": True,
                    "median_top_edge_absolute_expected_grade_effect_at_least_1e-3": True,
                    "q90_top_edge_absolute_expected_grade_effect_at_least_1e-2": True,
                    "at_least_5pct_collective_positive_effects": True,
                    "at_least_5pct_collective_negative_effects": True,
                },
                "passed": True,
            },
            **exact_errors,
        },
    }
    if audit_updates:
        audit.update(audit_updates)
    audit["content_checksum_sha256"] = hashlib.sha256(
        json.dumps(
            audit, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
    ).hexdigest()
    audit_dir = fold_dir / "audits"
    audit_dir.mkdir()
    with (audit_dir / EXPECTED_AUDIT_FILENAME).open("w", encoding="utf-8") as stream:
        json.dump(audit, stream)


def _rewrite_audit(path: Path, update) -> None:
    with path.open(encoding="utf-8") as stream:
        audit = json.load(stream)
    audit.pop("content_checksum_sha256", None)
    update(audit)
    audit["content_checksum_sha256"] = hashlib.sha256(
        json.dumps(
            audit, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
    ).hexdigest()
    with path.open("w", encoding="utf-8") as stream:
        json.dump(audit, stream)


def _make_run(
    root: Path,
    role: str,
    *,
    metric_offset: float = 0.0,
    config_updates: dict[str, object] | None = None,
    state_updates: dict[str, object] | None = None,
    result_updates: dict[str, object] | None = None,
    passes_floor: bool = True,
    audit_updates: dict[str, object] | None = None,
    skipped: int = 0,
) -> Path:
    fold_dir = root / role / "fold0"
    fold_dir.mkdir(parents=True)
    variant = EXPECTED_VARIANTS[role]
    config: dict[str, object] = {
        "dataset": "aptos",
        "run_dir": str(root / role),
        "seed": 42,
        "n_classes": 5,
        "n_folds": 5,
        "epochs": 2,
        "relation_only_epochs": 2,
        "batch_size": 8,
        "lr": 1e-4,
        "head_lr": 5e-4,
        "weight_decay": 1e-5,
        "scheduler": "plateau",
        "lr_factor": 0.2,
        "lr_patience": 5,
        "rps_weight": 0.25,
        "class_weighting": "none",
        "relation_enabled": True,
        "relation_variant": variant,
        "relation_edge_budget": 8,
        "relation_allocation_temperature": 1.0,
        "relation_permutation_seed": 617,
        "relation_delta_cap": 2.0,
        "decision_rule": "class_map",
        "warm_start_checkpoint": "/immutable/v3/best.pth",
        "warm_start_sha256": "a" * 64,
        "resume": False,
    }
    if config_updates:
        config.update(config_updates)
    metrics = {
        "acc": 84.0 + metric_offset,
        "qwk": 0.90 + metric_offset / 100.0,
        "mae": 0.20 - metric_offset / 100.0,
        "balanced_acc": 62.0 + metric_offset,
        "macro_f1": 0.61 + metric_offset / 100.0,
        "ece": 0.04,
        "loss": 0.70,
    }
    state: dict[str, object] = {
        "schema": "origin-checkpoint-v8",
        "relation_protocol_version": "v8",
        "relation_variant": variant,
        "checkpoint_role": "best_learned",
        "epoch": 2,
        "training_phase": "relation_only",
        "best_learned_epoch": 2,
        "best_learned_selection_key": (84.0 + metric_offset, 0.9, -0.7),
        "early_stopping_track": "learned_epoch_selector",
        "fold": 0,
        "split_signature": "matched-split-signature",
        "architecture_signature": "v8-architecture-signature",
        "implementation_signature": "v8-implementation-signature",
        "config_signature": "v8-config-signature",
        "selection_policy": "acc_then_qwk",
        "selection_qwk_weight": 0.1,
        "metrics": metrics,
        "config": config,
        "warm_start_provenance": {
            "source_checkpoint_sha256": "a" * 64,
            "source_split_signature": "matched-split-signature",
            "target_protocol": "v8",
            "target_relation_variant": variant,
        },
        "warm_start_metric_safety_floor": {
            "requirements": {"acc": 84.0, "qwk": 0.90}
        },
        "candidate_metric_safety_floor_evaluation": {
            "passes_metric_floor": passes_floor,
            "checkpoint_eligible": passes_floor,
        },
        "model_state": {
            "generator.relation_field.token_projection.weight": torch.zeros(64, 192),
            "generator.relation_field.raw_relation_strength": torch.zeros(4),
        },
    }
    if state_updates:
        state.update(state_updates)
    checkpoint = fold_dir / "best_learned.pth"
    torch.save(state, checkpoint)
    result: dict[str, object] = {
        "fold": 0,
        "relation_variant": variant,
        "test_evaluated": False,
        "best_learned_epoch": int(state.get("epoch", 2)),
        "best_learned_validation": metrics,
        "best_learned_checkpoint_sha256": _sha256(checkpoint),
        "best_learned_checkpoint_role": "best_learned",
        "best_learned_training_phase": "relation_only",
        "best_learned_metric_safety_floor_evaluation": state[
            "candidate_metric_safety_floor_evaluation"
        ],
        "best_learned_passes_v3_safety_floor": passes_floor,
    }
    if result_updates:
        result.update(result_updates)
    with (fold_dir / "result.json").open("w", encoding="utf-8") as stream:
        json.dump(result, stream)
    _write_history(fold_dir, epochs=int(config["epochs"]), skipped=skipped)
    _write_audit(
        fold_dir,
        state=state,
        metrics=metrics,
        checkpoint_sha256=_sha256(checkpoint),
        audit_updates=audit_updates,
    )
    return root / role


def _three_runs(tmp_path: Path):
    target = _make_run(tmp_path, "target", metric_offset=1.0)
    additive = _make_run(tmp_path, "additive_endpoint_control")
    shuffled = _make_run(tmp_path, "shuffled_geometry_control", metric_offset=0.5)
    return target, additive, shuffled


def test_valid_best_learned_three_way_comparison(tmp_path: Path) -> None:
    target, additive, shuffled = _three_runs(tmp_path)
    payload = compare_runs(target, additive, shuffled)
    assert payload["schema"] == "origin-v8-matched-control-comparison-v1"
    assert payload["outer_test_evaluated"] is False
    assert payload["development_gate"]["target_accuracy_above_both_controls"] is True
    assert payload["development_gate"]["target_passes_v3_safety_floor"] is True
    assert (
        payload["development_gate"][
            "target_passes_individual_certificate_strength_gate"
        ]
        is True
    )
    assert payload["development_gate"]["passed"] is True
    assert payload["shared_contract"]["witness_cap"] == 8
    assert payload["shared_contract"]["allocation_temperature"] == 1.0
    assert (
        payload["runs"]["shuffled_geometry_control"]["control_semantics"]
        == "fixed_shuffled_endpoint_to_geometry_correspondence_null"
    )


@pytest.mark.parametrize(
    ("kind", "updates", "message"),
    (
        ("config", {"seed": 43}, "hyperparameters differ"),
        ("config", {"head_lr": 1e-3}, "hyperparameters differ"),
        ("config", {"relation_edge_budget": 7}, "witness cap must equal 8"),
        ("config", {"relation_allocation_temperature": 0.5}, "temperature must equal 1.0"),
        ("state", {"split_signature": "other-split"}, "source and target split"),
    ),
)
def test_rejects_unmatched_contract(
    tmp_path: Path,
    kind: str,
    updates: dict[str, object],
    message: str,
) -> None:
    target = _make_run(tmp_path, "target")
    kwargs = {f"{kind}_updates": updates}
    additive = _make_run(tmp_path, "additive_endpoint_control", **kwargs)
    shuffled = _make_run(tmp_path, "shuffled_geometry_control")
    with pytest.raises(ComparisonContractError, match=message):
        compare_runs(target, additive, shuffled)


def test_rejects_different_v3_source_hash(tmp_path: Path) -> None:
    target = _make_run(tmp_path, "target")
    additive = _make_run(
        tmp_path,
        "additive_endpoint_control",
        config_updates={"warm_start_sha256": "b" * 64},
        state_updates={
            "warm_start_provenance": {
                "source_checkpoint_sha256": "b" * 64,
                "source_split_signature": "matched-split-signature",
                "target_protocol": "v8",
                "target_relation_variant": EXPECTED_VARIANTS[
                    "additive_endpoint_control"
                ],
            }
        },
    )
    shuffled = _make_run(tmp_path, "shuffled_geometry_control")
    with pytest.raises(ComparisonContractError, match="source checkpoint hash"):
        compare_runs(target, additive, shuffled)


def test_rejects_non_learned_or_missing_checkpoint(tmp_path: Path) -> None:
    target, additive, shuffled = _three_runs(tmp_path)
    checkpoint = additive / "fold0" / "best_learned.pth"
    checkpoint.unlink()
    # A deployable artifact must never be silently substituted.
    torch.save({}, additive / "fold0" / "best.pth")
    with pytest.raises(FileNotFoundError, match="best_learned"):
        compare_runs(target, additive, shuffled)


def test_rejects_skipped_optimizer_steps(tmp_path: Path) -> None:
    target = _make_run(tmp_path, "target")
    additive = _make_run(tmp_path, "additive_endpoint_control", skipped=1)
    shuffled = _make_run(tmp_path, "shuffled_geometry_control")
    with pytest.raises(ComparisonContractError, match="skipped optimizer steps"):
        compare_runs(target, additive, shuffled)


def test_rejects_missing_best_learned_audit(tmp_path: Path) -> None:
    target, additive, shuffled = _three_runs(tmp_path)
    (additive / "fold0" / "audits" / EXPECTED_AUDIT_FILENAME).unlink()
    with pytest.raises(FileNotFoundError, match="best-learned audit"):
        compare_runs(target, additive, shuffled)


@pytest.mark.parametrize(
    ("updates", "message"),
    (
        ({"schema": "origin-full-validation-audit-v7"}, "audit schema"),
        ({"scope": "outer_test"}, "inner-validation-only"),
        ({"checkpoint_role": "deployable_selected"}, "best_learned"),
        ({"checkpoint_sha256": "b" * 64}, "checkpoint hash differs"),
        ({"split_signature": "wrong-split"}, "audit split signature differs"),
        (
            {"metric_reproduction": {"checkpoint": True, "completed_result": False}},
            "did not reproduce",
        ),
    ),
)
def test_rejects_invalid_production_audit_contract(
    tmp_path: Path, updates: dict[str, object], message: str
) -> None:
    target = _make_run(tmp_path, "target", audit_updates=updates)
    additive = _make_run(tmp_path, "additive_endpoint_control")
    shuffled = _make_run(tmp_path, "shuffled_geometry_control")
    with pytest.raises(ComparisonContractError, match=message):
        compare_runs(target, additive, shuffled)


def test_rejects_audit_metric_or_structural_replay_mismatch(tmp_path: Path) -> None:
    target, additive, shuffled = _three_runs(tmp_path)
    audit_path = target / "fold0" / "audits" / EXPECTED_AUDIT_FILENAME

    def corrupt(audit: dict[str, object]) -> None:
        diagnostics = audit["relation_interaction_diagnostics"]
        assert isinstance(diagnostics, dict)
        sparse = diagnostics["sparse_support_and_identifiability"]
        assert isinstance(sparse, dict)
        sparse["max_entmax_allocation_replay_error"] = 0.1

    _rewrite_audit(audit_path, corrupt)
    with pytest.raises(ComparisonContractError, match="entmax_allocation_replay"):
        compare_runs(target, additive, shuffled)


def test_development_gate_requires_target_v3_floor(tmp_path: Path) -> None:
    target = _make_run(tmp_path, "target", metric_offset=1.0, passes_floor=False)
    additive = _make_run(tmp_path, "additive_endpoint_control")
    shuffled = _make_run(tmp_path, "shuffled_geometry_control", metric_offset=0.5)
    payload = compare_runs(target, additive, shuffled)
    assert payload["development_gate"]["target_accuracy_above_both_controls"] is True
    assert payload["development_gate"]["target_passes_v3_safety_floor"] is False
    assert payload["development_gate"]["passed"] is False


def test_development_gate_requires_target_individual_witness_strength(
    tmp_path: Path,
) -> None:
    target, additive, shuffled = _three_runs(tmp_path)
    audit_path = target / "fold0" / "audits" / EXPECTED_AUDIT_FILENAME

    def weaken(audit: dict[str, object]) -> None:
        diagnostics = audit["relation_interaction_diagnostics"]
        assert isinstance(diagnostics, dict)
        gate = diagnostics["individual_certificate_strength_gate"]
        assert isinstance(gate, dict)
        checks = gate["checks"]
        assert isinstance(checks, dict)
        checks["at_least_one_top_edge_changes_a_prediction"] = False
        gate["passed"] = False

    _rewrite_audit(audit_path, weaken)
    payload = compare_runs(target, additive, shuffled)
    assert (
        payload["development_gate"][
            "target_passes_individual_certificate_strength_gate"
        ]
        is False
    )
    assert payload["development_gate"]["passed"] is False
