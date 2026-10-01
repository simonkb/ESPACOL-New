from __future__ import annotations

import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest
import torch

from scripts.aggregate_origin_oof_interventions import aggregate_manifests
from scripts.audit_origin_oof_interventions import AtomicJsonlGzipWriter, iter_jsonl_gzip
from scripts.origin_oof_intervention_common import (
    CONTROL_METHODS,
    FOLD_SCHEMA,
    METHODS,
    ROW_SCHEMA,
    InterventionAuditConfig,
    audit_image_ledger,
    load_audit_protocol,
)
from scripts.origin_v3_cv_common import (
    ARCHITECTURE_SIGNATURE,
    IMPLEMENTATION_SIGNATURE,
    canonical_sha256,
    file_sha256,
    write_json_atomic,
)


def _config(seed: int = 17) -> InterventionAuditConfig:
    return InterventionAuditConfig(
        cell_budget_fractions=(0.0, 0.5, 1.0),
        random_repeats=3,
        random_seed=seed,
        denominator_epsilon=1e-10,
        footprint_bin_fraction=0.10,
        mass_match_relative_tolerance=0.25,
        decode_chunk_size=1024,
    )


def _audit(seed: int = 17) -> dict:
    # Boundary 1 ranking is deliberately unrelated to the all-boundary sum:
    # the bottom-right s4 cell has the largest irrelevant boundary-0 rate.
    s4 = torch.tensor(
        [
            [[0.05, 0.90, 0.10], [0.10, 0.70, 0.20]],
            [[0.20, 0.30, 0.40], [5.00, 0.01, 0.01]],
        ],
        dtype=torch.float32,
    )
    s8 = torch.tensor(
        [[[0.40, 0.60, 0.30], [0.30, 0.20, 0.50]]], dtype=torch.float32
    )
    return audit_image_ledger(
        rate_maps={"s4": s4, "s8": s8},
        valid_masks={
            "s4": torch.ones(2, 2, dtype=torch.bool),
            "s8": torch.ones(1, 2, dtype=torch.bool),
        },
        prior_rates=torch.tensor([0.10, 0.10, 0.10]),
        metadata={
            "s4": SimpleNamespace(
                output_stride=4, receptive_field=6, center_offset=2.0,
                input_size=(8, 8),
            ),
            "s8": SimpleNamespace(
                output_stride=8, receptive_field=12, center_offset=4.0,
                input_size=(8, 8),
            ),
        },
        identity={
            "dataset": "toy", "fold": 0, "outer_index": 0,
            "image_key": "toy:0:0", "image_path_relative": "toy.png",
            "image_id": "toy.png", "patient_id": "p0", "cluster_id": "p0",
        },
        true_grade=2,
        config=_config(seed),
    )


@pytest.fixture(scope="module")
def census() -> dict:
    return _audit()


def _curve(census: dict, boundary: int, method: str, budget: int) -> dict:
    return next(
        row for row in census["curve_rows"]
        if row["boundary"] == boundary
        and row["method"] == method
        and row["budget_index"] == budget
    )


def test_native_boundary_ranking_delete_retain_and_three_budget_units(census: dict) -> None:
    zero = _curve(census, 1, "ranked_native", 0)
    half = _curve(census, 1, "ranked_native", 1)
    full = _curve(census, 1, "ranked_native", 2)
    assert zero["selected_count"] == 0
    assert half["selected_count"] == 3
    # Native boundary-1 top three are .9, .7, .6.  The 5.0 rate on boundary 0
    # must not make its cell enter a target-boundary rank.
    assert half["selected_target_boundary_mass"] == pytest.approx(2.2, abs=2e-7)
    assert half["selected_cell_fraction"] == pytest.approx(0.5)
    assert 0.0 < half["selected_nominal_stride_area_fraction"] < 1.0
    assert 0.0 < half["selected_target_boundary_mass_fraction"] < 1.0
    assert zero["deletion_tail_normalized"] == pytest.approx(0.0)
    assert zero["retention_tail_normalized"] == pytest.approx(0.0)
    assert full["deletion_tail_normalized"] == pytest.approx(1.0)
    assert full["retention_tail_normalized"] == pytest.approx(1.0)
    assert zero["deletion_expected_grade_raw"] == pytest.approx(0.0)
    assert full["deletion_posterior_tv_normalized"] == pytest.approx(1.0)
    assert full["retention_posterior_tv_normalized"] == pytest.approx(1.0)


def test_matched_controls_are_deterministic_and_image_level(census: dict) -> None:
    assert _audit() == census
    assert _audit(seed=18) != census
    ranked = _curve(census, 1, "ranked_native", 1)
    count_matched = _curve(census, 1, "random_scale_count_stride_area", 1)
    footprint_matched = _curve(census, 1, "random_scale_footprint", 1)
    mass_matched = _curve(census, 1, "random_boundary_mass", 1)
    permuted = _curve(census, 1, "coordinate_permuted", 1)
    for row in (count_matched, footprint_matched, mass_matched, permuted):
        assert row["repeat_count"] == 3
        assert row["ranked_reference_deletion_tail_raw"] == pytest.approx(
            ranked["deletion_tail_raw"]
        )
        assert row["ranked_advantage_deletion_tail_raw"] is not None
        assert row["deletion_tail_raw_sd"] is not None
    # Exact scale counts imply exact nominal stride area at every repeat.
    assert count_matched["selected_nominal_stride_area_px2"] == pytest.approx(
        ranked["selected_nominal_stride_area_px2"]
    )
    assert count_matched["selected_nominal_stride_area_px2_sd"] == pytest.approx(0.0)
    assert footprint_matched["selected_nominal_stride_area_px2"] == pytest.approx(
        ranked["selected_nominal_stride_area_px2"]
    )
    assert 0.0 <= mass_matched["mass_match_within_tolerance_rate"] <= 1.0


def test_strata_thresholds_concentration_and_bootstrap_summaries_are_serializable(
    census: dict,
) -> None:
    assert len(census["image_rows"]) == 3
    assert len(census["curve_rows"]) == 3 * len(METHODS) * 3
    assert len(census["summary_rows"]) == 3 * len(METHODS)
    row = census["image_rows"][1]
    assert row["true_grade"] == 2
    assert isinstance(row["predicted_grade"], int)
    assert isinstance(row["correct"], bool)
    assert row["target_mass_effective_support"] > 1.0
    assert set(json.loads(row["scale_target_mass_json"])) == {"s4", "s8"}
    for prefix in (
        "min_delete_map_change",
        "min_delete_expected_drop_0p25",
        "min_retain_map_preservation",
        "min_delete_tail_effect_50pct",
        "min_retain_tail_effect_90pct",
    ):
        assert f"{prefix}_cell_fraction" in row
        assert f"{prefix}_nominal_stride_area_fraction" in row
        assert f"{prefix}_target_mass_fraction" in row
    control_summary = next(
        item for item in census["summary_rows"]
        if item["boundary"] == 1 and item["method"] == "random_scale_footprint"
    )
    assert "auc_deletion_tail_normalized_by_cells" in control_summary
    assert "ranked_advantage_auc_deletion_tail_normalized_by_cells" in control_summary
    json.dumps(census, sort_keys=True, allow_nan=False)


def test_zero_local_boundary_range_is_flagged_and_not_divided() -> None:
    result = audit_image_ledger(
        rate_maps={"s4": torch.zeros(1, 2, 2)},
        valid_masks={"s4": torch.ones(1, 2, dtype=torch.bool)},
        prior_rates=torch.tensor([0.1, 0.1]),
        metadata={
            "s4": SimpleNamespace(
                output_stride=4, receptive_field=4, center_offset=2.0,
                input_size=(4, 8),
            )
        },
        identity={"image_key": "zero"},
        true_grade=0,
        config=InterventionAuditConfig((0.0, 1.0), 1, 1, 1e-8, 0.1, 0.1, 128),
    )
    image = result["image_rows"][1]
    curve = next(
        row for row in result["curve_rows"]
        if row["boundary"] == 1 and row["method"] == "ranked_native"
    )
    assert image["tail_range_near_zero"] is True
    assert curve["deletion_tail_normalized"] is None


def test_frozen_protocol_is_checksummed_and_has_production_repeats() -> None:
    path = Path("scripts/protocols/origin_oof_intervention_protocol.json")
    payload, config = load_audit_protocol(path)
    assert config.random_repeats == 50
    assert config.cell_budget_fractions[0] == 0.0
    assert config.cell_budget_fractions[-1] == 1.0
    assert tuple(payload["controls"]) == CONTROL_METHODS
    corrupt = json.loads(path.read_text())
    corrupt["config"]["random_seed"] += 1
    temporary = path.parent / ".corrupt_protocol_for_test.json"
    try:
        temporary.write_text(json.dumps(corrupt))
        with pytest.raises(ValueError, match="checksum"):
            load_audit_protocol(temporary)
    finally:
        temporary.unlink(missing_ok=True)


def _write_fake_fold(
    root: Path, *, fold: int, image_key: str, image_path: str,
    audit_checksum: str = "audit", cv_checksum: str = "cv",
) -> Path:
    directory = root / f"fold{fold}"
    directory.mkdir(parents=True)
    identity = {
        "schema": ROW_SCHEMA,
        "dataset": "toy",
        "fold": fold,
        "image_key": image_key,
        "image_path_relative": image_path,
        "true_grade": fold,
        "predicted_grade": fold,
        "correct": True,
        "cluster_id": image_key,
    }
    paths = {
        "image_rows": directory / "images.jsonl.gz",
        "curve_rows": directory / "curves.jsonl.gz",
        "summary_rows": directory / "summaries.jsonl.gz",
    }
    with AtomicJsonlGzipWriter(paths["image_rows"]) as writer:
        writer.write({**identity, "row_type": "image_boundary_census", "boundary": 0})
    with AtomicJsonlGzipWriter(paths["curve_rows"]) as writer:
        for method in METHODS:
            for budget_index, budget in enumerate((0.0, 1.0)):
                writer.write(
                    {
                        **identity,
                        "row_type": "image_boundary_curve_point",
                        "boundary": 0,
                        "method": method,
                        "budget_index": budget_index,
                        "requested_cell_fraction": budget,
                    }
                )
    with AtomicJsonlGzipWriter(paths["summary_rows"]) as writer:
        for method_index, method in enumerate(METHODS):
            writer.write(
                {
                    **identity,
                    "row_type": "image_boundary_curve_summary",
                    "boundary": 0,
                    "method": method,
                    "auc_deletion_tail_normalized_by_cells": 1.0 - 0.1 * method_index,
                    "ranked_advantage_auc_deletion_tail_normalized_by_cells": (
                        None if method == "ranked_native" else 0.1 * method_index
                    ),
                }
            )
    manifest = {
        "schema": FOLD_SCHEMA,
        "dataset": "toy",
        "fold": fold,
        "scope": "every_locked_outer_fold_image_exact_stored_ledger_interventions",
        "n_images": 1,
        "n_boundaries": 1,
        "split_signature": f"split-{fold}",
        "audit_protocol_checksum_sha256": audit_checksum,
        "cv_protocol_checksum_sha256": cv_checksum,
        "audit_config": {"cell_budget_fractions": [0.0, 1.0]},
        "audit_config_sha256": "same-config",
        "implementation_signature": IMPLEMENTATION_SIGNATURE,
        "architecture_signature": ARCHITECTURE_SIGNATURE,
        "audit_implementation_sha256": "audit-code",
        "audit_common_implementation_sha256": "audit-common-code",
        "artifacts": {
            name: {
                "path": str(path.resolve()),
                "sha256": file_sha256(path),
                "rows": {"image_rows": 1, "curve_rows": 2 * len(METHODS), "summary_rows": len(METHODS)}[name],
                "row_schema": ROW_SCHEMA,
            }
            for name, path in paths.items()
        },
    }
    manifest["content_checksum_sha256"] = canonical_sha256(manifest)
    manifest_path = directory / "audit_manifest.json"
    write_json_atomic(manifest_path, manifest)
    return manifest_path


def test_fold_artifacts_aggregate_to_bootstrap_ready_oof_outputs(tmp_path: Path) -> None:
    manifests = [
        _write_fake_fold(tmp_path, fold=0, image_key="toy:0", image_path="a.png"),
        _write_fake_fold(tmp_path, fold=1, image_key="toy:1", image_path="b.png"),
    ]
    output = tmp_path / "aggregate"
    payload = aggregate_manifests(
        dataset="toy",
        manifest_paths=manifests,
        output_dir=output,
        expected_n_images=2,
        expected_folds=(0, 1),
        audit_protocol_checksum="audit",
        cv_protocol_checksum="cv",
    )
    assert payload["n_images"] == 2
    assert payload["schema"] == "origin-oof-intervention-aggregate-v1"
    assert len(list(iter_jsonl_gzip(payload["artifacts"]["image_rows"]["path"]))) == 2
    assert len(list(iter_jsonl_gzip(payload["artifacts"]["summary_rows"]["path"]))) == 2 * len(METHODS)
    strata = json.loads((output / "OOF_STRATIFIED_CURVE_SUMMARY.json").read_text())
    assert any(
        row["stratum_type"] == "true_grade" and row["stratum"] == "0"
        for row in strata["records"]
    )
    unsigned = dict(payload)
    assert unsigned.pop("content_checksum_sha256") == canonical_sha256(unsigned)


def test_atomic_jsonl_is_deterministic_and_slurm_jobs_are_parseable(tmp_path: Path) -> None:
    first, second = tmp_path / "a.gz", tmp_path / "b.gz"
    for path in (first, second):
        with AtomicJsonlGzipWriter(path) as writer:
            writer.write({"b": 2, "a": 1})
    assert first.read_bytes() == second.read_bytes()
    assert list(iter_jsonl_gzip(first)) == [{"a": 1, "b": 2}]
    for script in (
        Path("scripts/submit_origin_oof_intervention_fold.sh"),
        Path("scripts/submit_origin_oof_intervention_aggregate.sh"),
    ):
        subprocess.run(["bash", "-n", str(script)], check=True)
        source = script.read_text()
        assert "origin_oof_intervention_protocol.json" in source
        assert "audit_manifest.json" in source
    assert "audit_origin_oof_interventions.py" in Path(
        "scripts/submit_origin_oof_intervention_fold.sh"
    ).read_text()
    assert "aggregate_origin_oof_interventions.py" in Path(
        "scripts/submit_origin_oof_intervention_aggregate.sh"
    ).read_text()
