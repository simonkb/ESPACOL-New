from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from scripts.analyze_origin_oof_statistics import (
    GATE_A_SCHEMA,
    STATISTICS_MANIFEST_SCHEMA,
    GRADING_METRICS,
    GradeRow,
    StatisticsConfig,
    adjudicate_gate_a,
    analyze_oof_statistics,
    cluster_bootstrap_grading,
    derive_curve_endpoints,
    grading_metrics,
    load_statistics_protocol,
    paired_intervention_bootstrap,
    paired_scalar_bootstrap,
    threshold_reliability,
)
from scripts.audit_origin_oof_interventions import AtomicJsonlGzipWriter
from scripts.origin_oof_intervention_common import AGGREGATE_SCHEMA, METHODS, ROW_SCHEMA
from scripts.origin_v3_cv_common import canonical_sha256, file_sha256, write_json_atomic


def _config(*, samples: int = 200) -> StatisticsConfig:
    return StatisticsConfig(
        bootstrap_samples=samples,
        bootstrap_seed=1234,
        bootstrap_chunk_size=31,
        confidence_level=0.95,
        reliability_bins=5,
        nll_probability_floor=1e-15,
    )


def test_proper_scores_are_zero_for_a_perfect_deterministic_posterior() -> None:
    probabilities = np.eye(5, dtype=np.float64)
    labels = np.arange(5, dtype=np.int64)
    metrics = grading_metrics(probabilities, labels, labels)
    assert metrics["accuracy_percent"] == 100.0
    assert metrics["qwk"] == 1.0
    assert metrics["nll"] == pytest.approx(0.0)
    assert metrics["rps"] == pytest.approx(0.0)
    assert metrics["multiclass_brier"] == pytest.approx(0.0)
    assert metrics["expected_grade_mae"] == pytest.approx(0.0)


def test_ranked_probability_score_matches_manual_boundary_definition() -> None:
    probabilities = np.asarray([[0.1, 0.2, 0.3, 0.25, 0.15]])
    label = np.asarray([2])
    prediction = np.asarray([2])
    tails = np.asarray([0.9, 0.7, 0.4, 0.15])
    target = np.asarray([1.0, 1.0, 0.0, 0.0])
    expected = float(np.mean((tails - target) ** 2))
    assert grading_metrics(probabilities, label, prediction)["rps"] == pytest.approx(expected)


def test_threshold_reliability_reports_every_boundary_and_bin() -> None:
    probabilities = np.asarray(
        [
            [0.8, 0.1, 0.05, 0.03, 0.02],
            [0.1, 0.2, 0.4, 0.2, 0.1],
            [0.01, 0.02, 0.07, 0.2, 0.7],
        ]
    )
    metrics, rows = threshold_reliability(
        probabilities, np.asarray([0, 2, 4]), bins=5
    )
    assert len(metrics) == 4
    assert len(rows) == 4 * 5
    assert all(0.0 <= record["ece"] <= 1.0 for record in metrics)
    assert all(sum(row["n"] for row in rows if row["boundary"] == k) == 3 for k in range(4))


def test_grading_bootstrap_clusters_both_eyes_and_is_deterministic() -> None:
    rows = [
        GradeRow("dr", "left", "patient-a", 0, 0, (0.9, 0.1, 0.0, 0.0, 0.0)),
        GradeRow("dr", "right", "patient-a", 2, 2, (0.0, 0.1, 0.8, 0.1, 0.0)),
        GradeRow("dr", "single", "patient-b", 4, 3, (0.0, 0.0, 0.1, 0.6, 0.3)),
    ]
    first = cluster_bootstrap_grading(rows, config=_config(), seed_label="unit")
    second = cluster_bootstrap_grading(rows, config=_config(), seed_label="unit")
    assert first == second
    assert first["n_images"] == 3
    assert first["n_clusters"] == 2
    assert set(GRADING_METRICS).issubset(first["metrics"])
    assert "threshold_ece_y_gt_3" in first["metrics"]


def _summary_rows(delta: float = 0.2) -> list[dict]:
    result = []
    for image, cluster in (("left", "patient-a"), ("right", "patient-a"), ("x", "patient-b")):
        for method_index, method in enumerate(METHODS):
            value = 0.8 if method == "ranked_native" else 0.8 - delta - 0.01 * method_index
            result.append(
                {
                    "schema": ROW_SCHEMA,
                    "row_type": "image_boundary_curve_summary",
                    "dataset": "dr",
                    "image_key": image,
                    "cluster_id": cluster,
                    "boundary": 1,
                    "method": method,
                    "auc_deletion_tail_normalized_by_cells": value,
                    "auc_retention_tail_normalized_by_cells": value - 0.1,
                }
            )
    return result


def test_paired_intervention_cluster_bootstrap_recovers_constant_difference() -> None:
    records = paired_intervention_bootstrap(
        _summary_rows(), config=_config(), seed_label="paired"
    )
    target = next(
        row for row in records
        if row["comparator"] == METHODS[1]
        and row["metric"] == "auc_deletion_tail_normalized_by_cells"
    )
    expected = 0.2 + 0.01
    assert target["n_paired_image_boundaries"] == 3
    assert target["n_clusters"] == 2
    assert target["ranked_minus_comparator"] == pytest.approx(expected)
    assert target["ci_percentile"] == pytest.approx([expected, expected])
    assert target["bootstrap_fraction_ranked_better"] == 1.0


def test_paired_intervention_requires_every_method() -> None:
    rows = [row for row in _summary_rows() if row["method"] != METHODS[-1]]
    with pytest.raises(AssertionError, match="fully paired"):
        paired_intervention_bootstrap(rows, config=_config(), seed_label="missing")


def test_statistics_protocol_is_checksum_sealed() -> None:
    path = Path("scripts/protocols/origin_oof_statistics_protocol.json")
    payload, config = load_statistics_protocol(path)
    assert config.bootstrap_samples == 10_000
    assert config.reliability_bins == 15
    corrupt = json.loads(path.read_text())
    corrupt["config"]["bootstrap_seed"] += 1
    temporary = path.parent / ".corrupt_statistics_protocol.json"
    try:
        temporary.write_text(json.dumps(corrupt))
        with pytest.raises(ValueError, match="checksum"):
            load_statistics_protocol(temporary)
    finally:
        temporary.unlink(missing_ok=True)


def test_checksum_sealed_aggregate_runs_end_to_end(tmp_path: Path) -> None:
    image_path = tmp_path / "images.jsonl.gz"
    summary_path = tmp_path / "summaries.jsonl.gz"
    image_specs = (
        ("a", "patient-a", 0, (0.9, 0.1, 0.0, 0.0, 0.0)),
        ("b", "patient-b", 2, (0.0, 0.1, 0.8, 0.1, 0.0)),
    )
    with AtomicJsonlGzipWriter(image_path) as writer:
        for image_key, cluster, label, probabilities in image_specs:
            for boundary in range(4):
                writer.write(
                    {
                        "schema": ROW_SCHEMA,
                        "row_type": "image_boundary_census",
                        "dataset": "dr",
                        "image_key": image_key,
                        "cluster_id": cluster,
                        "true_grade": label,
                        "predicted_grade": label,
                        "boundary": boundary,
                        "full_class_probabilities": list(probabilities),
                    }
                )
    with AtomicJsonlGzipWriter(summary_path) as writer:
        for image_key, cluster, _, _ in image_specs:
            for method_index, method in enumerate(METHODS):
                writer.write(
                    {
                        "schema": ROW_SCHEMA,
                        "row_type": "image_boundary_curve_summary",
                        "dataset": "dr",
                        "image_key": image_key,
                        "cluster_id": cluster,
                        "boundary": 0,
                        "method": method,
                        "auc_deletion_tail_normalized_by_cells": 1.0 - 0.1 * method_index,
                        "auc_deletion_tail_normalized_by_nominal_stride_area": (
                            1.0 - 0.1 * method_index
                        ),
                        "auc_retention_tail_normalized_by_nominal_stride_area": (
                            1.0 - 0.1 * method_index
                        ),
                    }
                )
    aggregate = {
        "schema": AGGREGATE_SCHEMA,
        "dataset": "dr",
        "n_images": 2,
        "artifacts": {
            "image_rows": {
                "path": str(image_path.resolve()),
                "sha256": file_sha256(image_path),
                "rows": 8,
            },
            "summary_rows": {
                "path": str(summary_path.resolve()),
                "sha256": file_sha256(summary_path),
                "rows": 2 * len(METHODS),
            },
        },
    }
    aggregate["content_checksum_sha256"] = canonical_sha256(aggregate)
    aggregate_path = tmp_path / "audit_manifest.json"
    write_json_atomic(aggregate_path, aggregate)

    protocol = json.loads(
        Path("scripts/protocols/origin_oof_statistics_protocol.json").read_text()
    )
    protocol["config"]["bootstrap_samples"] = 1000
    protocol["config"]["bootstrap_chunk_size"] = 97
    protocol.pop("content_checksum_sha256")
    protocol["content_checksum_sha256"] = canonical_sha256(protocol)
    protocol_path = tmp_path / "protocol.json"
    write_json_atomic(protocol_path, protocol)

    output = tmp_path / "output"
    manifest = analyze_oof_statistics(
        aggregate_manifests={"dr": aggregate_path},
        protocol_path=protocol_path,
        output_dir=output,
    )
    assert manifest["schema"] == STATISTICS_MANIFEST_SCHEMA
    assert set(manifest["artifacts"]) == {
        "proper_score_calibration",
        "grading_bootstrap",
        "intervention_bootstrap",
        "rf_footprint_diagnostics",
        "gate_a",
    }
    unsigned = dict(manifest)
    assert unsigned.pop("content_checksum_sha256") == canonical_sha256(unsigned)
    for artifact in manifest["artifacts"].values():
        path = Path(artifact["path"])
        assert path.is_file()
        assert file_sha256(path) == artifact["sha256"]
    gate = json.loads((output / "OOF_GATE_A_ADJUDICATION.json").read_text())
    assert gate["schema"] == GATE_A_SCHEMA
    assert gate["status"] == "insufficient_data"
    assert not gate["claim_authorized"]
    inference = json.loads((output / "OOF_INTERVENTION_PAIRED_BOOTSTRAP.json").read_text())
    scope_types = {row["scope_type"] for row in inference["records"]}
    assert "dataset_correctness" in scope_types
    assert "dataset_correct_true_grade" in scope_types
    assert "dataset_boundary_outcome" in scope_types


def _curve_profile(method: str) -> dict:
    points = []
    for index, (area, effect, preserved) in enumerate(
        ((0.0, 0.0, 0.0), (0.1, 0.6, 1.0), (1.0, 1.0, 1.0))
    ):
        points.append(
            {
                "budget_index": index,
                "requested_cell_fraction": area,
                "selected_cell_fraction": area,
                "selected_nominal_stride_area_fraction": area,
                "selected_target_boundary_mass_fraction": effect,
                "deletion_tail_normalized": effect,
                "retention_tail_normalized": effect,
                "retention_map_preserved": preserved,
                "maximum_single_footprint_area_input_fraction": 0.02,
                "summed_clipped_footprint_area_input_fraction": area,
                "clipped_union_footprint_area_input_fraction": area,
                "selected_count": area * 100,
                "selected_count_s4": area * 100,
                "selected_count_s32": 0.0,
            }
        )
    return {
        "dataset": "dr",
        "image_key": "image",
        "cluster_id": "patient",
        "true_grade": 2,
        "predicted_grade": 2,
        "correct": True,
        "boundary": 1,
        "boundary_outcome": "true_positive",
        "method": method,
        "points": points,
    }


def test_curve_endpoints_report_concentration_and_global_rf_share() -> None:
    profiles = [_curve_profile(method) for method in METHODS]
    census = {
        ("image", 1): {
            "scale_geometry": {
                "s4": {"theoretical_support_class": "finite_local"},
                "s32": {"theoretical_support_class": "global_context"},
            }
        }
    }
    rows = derive_curve_endpoints(profiles, census, gate=_config().gate_a)
    ranked = next(row for row in rows if row["method"] == "ranked_native")
    assert ranked["area_to_50pct_deletion_effect"] == pytest.approx(1 / 12)
    assert ranked["deletion_effect_at_10pct_area"] == pytest.approx(0.6)
    assert ranked["retention_map_preserved_at_10pct_area"] == pytest.approx(1.0)
    assert ranked["global_context_cell_fraction_at_10pct_area"] == pytest.approx(0.0)


def test_paired_scalar_bootstrap_preserves_patient_pairing() -> None:
    rows = []
    for image, cluster in (("left", "patient-a"), ("right", "patient-a"), ("x", "patient-b")):
        for method in METHODS:
            base = 0.9 if method == "ranked_native" else 0.6
            rows.append(
                {
                    "dataset": "dr", "image_key": image, "cluster_id": cluster,
                    "boundary": 0, "method": method,
                    "area_to_50pct_deletion_effect": 1.0 - base,
                    "area_to_50pct_retention_effect": 1.0 - base,
                    "deletion_effect_at_10pct_area": base,
                    "retention_effect_at_10pct_area": base,
                    "retention_map_preserved_at_10pct_area": base,
                }
            )
    records = paired_scalar_bootstrap(
        rows, config=_config(), seed_label="derived",
        comparators=("random_scale_count_stride_area",),
    )
    preservation = next(
        row for row in records
        if row["metric"] == "retention_map_preserved_at_10pct_area"
    )
    assert preservation["n_clusters"] == 2
    assert preservation["contrast_positive_favors_ranked"] == pytest.approx(0.3)
    assert preservation["ci_percentile"] == pytest.approx([0.3, 0.3])


def _passing_gate_inputs(config: StatisticsConfig) -> tuple[list[dict], list[dict], dict]:
    auc = []
    for dataset in config.gate_a.required_datasets:
        for scope_type, scope in [
            ("dataset_correct_positive", f"{dataset}:correct_positive"),
            *(
                ("dataset_correct_true_grade", f"{dataset}:grade={grade}")
                for grade in (
                    config.gate_a.dr_positive_grades
                    if dataset == "dr" else config.gate_a.aptos_positive_grades
                )
            ),
        ]:
            for comparator in config.gate_a.matched_controls:
                for metric in (
                    config.gate_a.deletion_auc_metric,
                    config.gate_a.retention_auc_metric,
                ):
                    auc.append(
                        {
                            "scope_type": scope_type, "scope": scope,
                            "comparator": comparator, "metric": metric,
                            "ranked_minus_comparator": 0.2,
                            "ci_percentile": [0.1, 0.3], "n_clusters": 50,
                        }
                    )
    derived_inference = []
    for dataset in config.gate_a.required_datasets:
        for comparator in config.gate_a.matched_controls:
            derived_inference.append(
                {
                    "scope_type": "dataset_correct_positive",
                    "scope": f"{dataset}:correct_positive",
                    "comparator": comparator,
                    "metric": "retention_map_preserved_at_10pct_area",
                    "contrast_positive_favors_ranked": 0.1,
                    "ci_percentile": [0.02, 0.2], "n_clusters": 50,
                }
            )
    by_dataset = {}
    for dataset in config.gate_a.required_datasets:
        rows = []
        for method in METHODS:
            for index in range(100):
                rows.append(
                    {
                        "dataset": dataset,
                        "true_grade": 2,
                        "predicted_grade": 2,
                        "correct": True,
                        "method": method,
                        "area_to_50pct_deletion_effect": (
                            0.05 if method == "ranked_native" else 0.2
                        ),
                        "area_to_50pct_retention_effect": (
                            0.05 if method == "ranked_native" else 0.2
                        ),
                        "deletion_effect_at_10pct_area": 0.6,
                        "retention_effect_at_10pct_area": 0.6,
                    }
                )
        by_dataset[dataset] = rows
    return auc, derived_inference, by_dataset


def test_gate_a_passes_only_when_every_clause_passes() -> None:
    config = _config()
    auc, derived, by_dataset = _passing_gate_inputs(config)
    result = adjudicate_gate_a(
        config=config, intervention_records=auc,
        derived_inference_records=derived, derived_by_dataset=by_dataset,
        input_issues=[],
        runtime_source={"git_commit_available": True, "tracked_worktree_clean": True},
    )
    assert result["status"] == "pass"
    assert result["claim_authorized"]
    auc[0]["ranked_minus_comparator"] = 0.01
    result = adjudicate_gate_a(
        config=config, intervention_records=auc,
        derived_inference_records=derived, derived_by_dataset=by_dataset,
        input_issues=[],
        runtime_source={"git_commit_available": True, "tracked_worktree_clean": True},
    )
    assert result["status"] == "fail"
    assert not result["claim_authorized"]


def test_gate_a_withholds_claim_on_incomplete_input() -> None:
    config = _config()
    auc, derived, by_dataset = _passing_gate_inputs(config)
    result = adjudicate_gate_a(
        config=config, intervention_records=auc,
        derived_inference_records=derived, derived_by_dataset=by_dataset,
        input_issues=["missing curve artifact"],
        runtime_source={"git_commit_available": True, "tracked_worktree_clean": True},
    )
    assert result["status"] == "insufficient_data"
    assert result["auditability_claim_status"] == "withheld_incomplete"
