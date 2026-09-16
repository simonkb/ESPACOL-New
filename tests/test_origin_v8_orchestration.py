from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_v8_aptos_launcher_is_a_three_member_validation_only_array() -> None:
    text = (ROOT / "scripts" / "submit_origin_v8_matched_aptos_f0.sh").read_text()
    assert "#SBATCH --array=0-2" in text
    assert "--dataset aptos" in text
    assert "--folds 0" in text
    assert "--skip_test" in text
    assert "--include_test" not in text
    assert "--relation_edge_budget 8" in text
    assert "--relation_allocation_temperature 1.0" in text
    assert "--relation_only_epochs 15" in text
    assert "--epochs 15" in text
    for variant in (
        "identified_conserved_witness_v1",
        "additive_conserved_witness_control_v1",
        "shuffled_conserved_witness_control_v1",
    ):
        assert text.count(variant) == 1


def test_v8_orchestration_uses_fresh_distinct_run_names() -> None:
    text = (ROOT / "scripts" / "submit_origin_v8_matched_aptos_f0.sh").read_text()
    for name in (
        "origin_aptos_f0_v8_cmwa_r1_target",
        "origin_aptos_f0_v8_cmwa_r1_endpoint_control",
        "origin_aptos_f0_v8_cmwa_r1_shuffled_control",
    ):
        assert text.count(name) == 1
    assert "best_learned.pth" in text
    assert "Fresh ORIGIN-v8 run refused" in text


def test_v8_comparison_names_shuffled_null_precisely() -> None:
    utility = (ROOT / "compare_origin_v8_controls.py").read_text()
    assert "fixed_shuffled_endpoint_to_geometry_correspondence_null" in utility
    assert "does not remove or independently shuffle" in utility
    assert "checkpoint_role\") != \"best_learned\"" in utility


def test_v8_audit_array_replays_each_best_learned_checkpoint_only() -> None:
    text = (ROOT / "scripts" / "submit_origin_v8_matched_audit.sh").read_text()
    assert "#SBATCH --array=0-2" in text
    assert "best_learned.pth" in text
    assert "full_validation_audit_v8_best_learned.json" in text
    assert "audit_origin_validation.py" in text
    assert "--batch_size 8" in text
    assert "--top_ks 1,5,10" in text
    assert "--include_test" not in text
    assert "outer_test_authorized=false" in text


def test_v8_launcher_orders_preflight_training_audit_then_comparison() -> None:
    text = (ROOT / "scripts" / "launch_origin_v8_matched_aptos_f0.sh").read_text()
    assert "scripts/submit_origin_v8_matched_audit.sh" in text
    assert 'afterok:${ARRAY_JOB}' in text
    assert 'afterok:${AUDIT_JOB}' in text
    assert text.index("submit_origin_v8_matched_preflight.sh") < text.index(
        "submit_origin_v8_matched_aptos_f0.sh"
    )
    assert text.index("submit_origin_v8_matched_aptos_f0.sh") < text.index(
        "submit_origin_v8_matched_audit.sh"
    )
    assert text.index("submit_origin_v8_matched_audit.sh") < text.index(
        "submit_origin_v8_matched_comparison.sh"
    )


def test_v8_comparison_is_a_fail_closed_audited_development_gate() -> None:
    utility = (ROOT / "compare_origin_v8_controls.py").read_text()
    assert "origin-full-validation-audit-v8" in utility
    assert "full_validation_audit_v8_best_learned.json" in utility
    assert "target_passes_v3_safety_floor" in utility
    assert "target_passes_individual_certificate_strength_gate" in utility
    assert 'raise SystemExit(10)' in utility


def test_v8_preflight_registers_single_score_and_frozen_replay_contracts() -> None:
    text = (ROOT / "scripts" / "submit_origin_v8_matched_preflight.sh").read_text()
    assert (
        "same_identified_score_drives_shortlist_allocation_and_edgewise_value" in text
    )
    assert "one_zero_start_scalar_per_ordinal_boundary" in text
    assert "stored_contribution_deletion_without_reselection_or_reallocation" in text
