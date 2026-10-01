from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.analyze_origin_idrid_semantics import (
    LESIONS,
    MANIFEST_SCHEMA,
    SCALES,
    StatisticsConfig,
    analyze_idrid_semantics,
    aggregate_alignment_image_units,
    aggregate_deletion_image_units,
    cluster_bootstrap_mean,
    exact_two_sided_sign_test,
    load_protocol,
    paired_sign_flip_test,
    validate_input_records,
)
from scripts.origin_v3_cv_common import canonical_sha256, file_sha256, write_json_atomic


def _alignment_record(
    *, fold: int, image: str, grade: int, scale: str, lesion: str,
    boundary: int, ap: float,
) -> dict:
    prevalence = 0.1 if lesion == "union" else 0.05
    return {
        "kind": "alignment",
        "fold": fold,
        "image_id": image,
        "grade": grade,
        "split": "training",
        "scale": scale,
        "scale_group": "fine" if scale in ("s4", "s8") else "coarse",
        "boundary": boundary,
        "target_boundary": min(max(grade - 1, 0), 3),
        "lesion": lesion,
        "prevalence": prevalence,
        "average_precision": ap,
        "ap_over_prevalence": ap / prevalence,
        "mean_rate_ratio": 2.0,
    }


def _deletion_record(
    *, fold: int, image: str, grade: int, group: str, guided: float, random: float,
) -> dict:
    return {
        "kind": "external_mask_deletion",
        "fold": fold,
        "image_id": image,
        "grade": grade,
        "split": "training",
        "scale_group": group,
        "target_boundary": min(max(grade - 1, 0), 3),
        "deleted_cells": 4,
        "expected_grade_delta": guided,
        "random_expected_grade_delta_mean": random,
        "expected_grade_delta_lift": guided - random,
        "target_tail_delta": guided / 2,
        "random_target_tail_delta_mean": random / 2,
        "target_tail_delta_lift": (guided - random) / 2,
        "random_repeats": 20,
    }


def _complete_toy_records() -> list[dict]:
    records = []
    for fold in (0, 1):
        for image_index, (image, grade) in enumerate((("IDRiD_01", 2), ("IDRiD_02", 4))):
            for scale_index, scale in enumerate(SCALES):
                for lesion in LESIONS:
                    for boundary in range(4):
                        records.append(
                            _alignment_record(
                                fold=fold,
                                image=image,
                                grade=grade,
                                scale=scale,
                                lesion=lesion,
                                boundary=boundary,
                                ap=0.2 + 0.1 * fold + 0.01 * scale_index + 0.02 * image_index,
                            )
                        )
            for group in ("all", "fine", "coarse"):
                records.append(
                    _deletion_record(
                        fold=fold, image=image, grade=grade, group=group,
                        guided=0.3 + 0.1 * fold + 0.02 * image_index,
                        random=0.1 + 0.01 * fold,
                    )
                )
    return records


def test_raw_census_validation_rejects_pseudoreplication_or_missing_rows() -> None:
    records = _complete_toy_records()
    summary = {
        "schema": "origin-idrid-semantic-audit-v1",
        "scope": "external_zero_shot_cell_bin_alignment_and_internal_ledger_deletion",
        "idrid_images": 2,
        "random_repeats": 20,
        "checkpoints": [
            {"fold": 0, "checkpoint_sha256": "a"},
            {"fold": 1, "checkpoint_sha256": "b"},
        ],
    }
    census = validate_input_records(
        summary, records, expected_folds=(0, 1), expected_images=2,
        expected_random_repeats=20,
    )
    assert census["alignment_rows"] == 2 * 2 * 4 * 5 * 4
    assert census["deletion_rows"] == 2 * 2 * 3
    with pytest.raises(AssertionError, match="incomplete"):
        validate_input_records(
            summary, records[:-1], expected_folds=(0, 1), expected_images=2,
            expected_random_repeats=20,
        )


def test_alignment_aggregates_checkpoints_within_image_before_inference() -> None:
    units = aggregate_alignment_image_units(
        _complete_toy_records(), scale_group="fine", lesion="union",
        boundary=None, expected_folds=(0, 1),
    )
    assert len(units) == 2
    assert all(unit["checkpoints_aggregated"] == 2 for unit in units)
    assert all(unit["scales_aggregated"] == 2 for unit in units)
    # Image 1: means of fold/scale AP values (.20,.21,.30,.31).
    assert units[0]["average_precision"] == pytest.approx(0.255)
    assert units[0]["ap_minus_prevalence"] == pytest.approx(0.155)


def test_deletion_is_paired_and_aggregated_within_image() -> None:
    units = aggregate_deletion_image_units(
        _complete_toy_records(), scale_group="overall", expected_folds=(0, 1)
    )
    assert len(units) == 2
    first = units[0]
    assert first["checkpoints_aggregated"] == 2
    assert first["guided_expected_grade_delta"] == pytest.approx(0.35)
    assert first["random_expected_grade_delta"] == pytest.approx(0.105)
    assert first["expected_grade_delta_lift"] == pytest.approx(0.245)


def test_cluster_bootstrap_constant_vector_has_degenerate_interval() -> None:
    result = cluster_bootstrap_mean(
        [0.3] * 8, samples=1000, seed=7, confidence=0.95
    )
    assert result["estimate"] == pytest.approx(0.3)
    assert result["ci_percentile"] == pytest.approx([0.3, 0.3])


def test_exact_sign_and_sign_flip_tests_detect_constant_positive_pairs() -> None:
    values = [0.2] * 6
    sign = exact_two_sided_sign_test(values, zero_tolerance=1e-12)
    flip = paired_sign_flip_test(
        values, samples=999, seed=3, exact_max_n=20, zero_tolerance=1e-12
    )
    assert sign["positive"] == 6 and sign["negative"] == 0
    assert sign["p_value"] == pytest.approx(2 / (2**6))
    assert flip["method"] == "exact_paired_sign_flip"
    assert flip["two_sided_p_value"] == pytest.approx(2 / (2**6))


def test_monte_carlo_sign_flip_is_deterministic() -> None:
    values = [0.1 + 0.001 * index for index in range(30)]
    first = paired_sign_flip_test(
        values, samples=1999, seed=44, exact_max_n=20, zero_tolerance=1e-12
    )
    second = paired_sign_flip_test(
        values, samples=1999, seed=44, exact_max_n=20, zero_tolerance=1e-12
    )
    assert first == second
    assert first["method"] == "monte_carlo_paired_sign_flip"


def test_statistics_protocol_is_checksum_sealed() -> None:
    path = Path("scripts/protocols/origin_idrid_semantic_statistics_protocol.json")
    payload, config = load_protocol(path)
    assert payload["expected_images"] == 81
    assert config.bootstrap_samples == 10_000
    corrupt = json.loads(path.read_text())
    corrupt["config"]["bootstrap_seed"] += 1
    temporary = path.parent / ".corrupt_idrid_statistics_protocol.json"
    try:
        temporary.write_text(json.dumps(corrupt))
        with pytest.raises(ValueError, match="checksum"):
            load_protocol(temporary)
    finally:
        temporary.unlink(missing_ok=True)


def test_checksum_sealed_small_audit_runs_end_to_end(tmp_path: Path) -> None:
    records = _complete_toy_records()
    records_path = tmp_path / "idrid_semantic_records.jsonl"
    records_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in records)
    )
    summary = {
        "schema": "origin-idrid-semantic-audit-v1",
        "scope": "external_zero_shot_cell_bin_alignment_and_internal_ledger_deletion",
        "idrid_images": 2,
        "random_repeats": 20,
        "checkpoints": [
            {"fold": 0, "checkpoint_sha256": "a"},
            {"fold": 1, "checkpoint_sha256": "b"},
        ],
        "records": records_path.name,
        "records_sha256": file_sha256(records_path),
    }
    summary_path = tmp_path / "idrid_semantic_summary.json"
    write_json_atomic(summary_path, summary)

    protocol = json.loads(
        Path("scripts/protocols/origin_idrid_semantic_statistics_protocol.json").read_text()
    )
    protocol["expected_images"] = 2
    protocol["expected_checkpoint_folds"] = [0, 1]
    protocol["expected_segmentation_split_counts"] = {"training": 2}
    protocol["config"]["bootstrap_samples"] = 1000
    protocol["config"]["permutation_samples"] = 999
    protocol.pop("content_checksum_sha256")
    protocol["content_checksum_sha256"] = canonical_sha256(protocol)
    protocol_path = tmp_path / "protocol.json"
    write_json_atomic(protocol_path, protocol)

    output = tmp_path / "statistics"
    manifest = analyze_idrid_semantics(
        summary_path=summary_path, protocol_path=protocol_path, output_dir=output
    )
    assert manifest["schema"] == MANIFEST_SCHEMA
    assert set(manifest["artifacts"]) == {
        "alignment_statistics", "deletion_statistics", "image_level_units"
    }
    unsigned = dict(manifest)
    assert unsigned.pop("content_checksum_sha256") == canonical_sha256(unsigned)
    for artifact in manifest["artifacts"].values():
        path = Path(artifact["path"])
        assert path.is_file()
        assert file_sha256(path) == artifact["sha256"]
