from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from scripts.origin_v3_cv_common import canonical_sha256, file_sha256, write_json_atomic
from tools.analyze_origin_idrid_semantics_v2 import (
    MANIFEST_SCHEMA,
    UNIT_SCHEMA,
    analyze_idrid_semantics_v2,
    aggregate_deletion_image_units_v2,
    load_protocol,
    validate_input_records_v2,
)
from tools.audit_origin_idrid_semantics_v2 import (
    AP_CONTRACT,
    AUDIT_SCHEMA,
    AUDIT_SCOPE,
    DELETION_KIND,
    MATCHING_CONTRACT,
    _stable_seed,
    balanced_paired_cell_masks,
    summarize,
    tie_grouped_binary_average_precision,
)


SCALES = ("s4", "s8", "s16", "s32")
LESIONS = ("union", "microaneurysm", "haemorrhage", "hard_exudate", "soft_exudate")
SOURCE_COMMIT = "1" * 40
ARCHITECTURE_SIGNATURE = "2" * 64
CONFIG_SIGNATURE = "3" * 64


def test_tie_grouped_ap_constant_scores_equal_prevalence() -> None:
    scores = torch.zeros(6)
    labels = torch.tensor([True, False, True, False, False, True])
    expected = float(labels.float().mean())
    assert tie_grouped_binary_average_precision(scores, labels) == pytest.approx(expected)
    permutation = torch.tensor([5, 1, 3, 0, 4, 2])
    assert tie_grouped_binary_average_precision(
        scores[permutation], labels[permutation]
    ) == pytest.approx(expected)


def test_tie_grouped_ap_is_invariant_within_nonconstant_ties() -> None:
    scores = torch.tensor([0.9, 0.9, 0.4, 0.4, 0.4, 0.1])
    labels = torch.tensor([True, False, False, True, False, True])
    expected = tie_grouped_binary_average_precision(scores, labels)
    permutation = torch.tensor([1, 0, 4, 2, 3, 5])
    assert tie_grouped_binary_average_precision(
        scores[permutation], labels[permutation]
    ) == pytest.approx(expected)


def test_bidirectional_matcher_handles_lesion_dense_and_sparse_lattices() -> None:
    valid = torch.ones(2, 4, dtype=torch.bool)
    for lesion_count in (2, 6):
        lesion = torch.zeros_like(valid)
        lesion.flatten()[:lesion_count] = True
        first = balanced_paired_cell_masks(
            lesion, valid, generator=torch.Generator().manual_seed(17)
        )
        second = balanced_paired_cell_masks(
            lesion, valid, generator=torch.Generator().manual_seed(17)
        )
        guided, control, counts = first
        assert torch.equal(guided, second[0])
        assert torch.equal(control, second[1])
        assert counts == second[2]
        assert counts["matched_cells"] == min(lesion_count, 8 - lesion_count)
        assert int(guided.sum()) == int(control.sum()) == counts["matched_cells"]
        assert not bool((guided & control).any())
        assert not bool((guided & ~lesion).any())
        assert not bool((control & lesion).any())


def _alignment_record(
    *, fold: int, image: str, grade: int, scale: str, lesion: str, boundary: int
) -> dict:
    prevalence = 0.25
    ap = 0.4 + 0.01 * fold
    return {
        "kind": "alignment",
        "fold": fold,
        "image_id": image,
        "grade": grade,
        "split": "training",
        "scale": scale,
        "scale_group": "fine" if scale in ("s4", "s8") else "coarse",
        "average_precision_definition": AP_CONTRACT,
        "boundary": boundary,
        "target_boundary": min(max(grade - 1, 0), 3),
        "lesion": lesion,
        "prevalence": prevalence,
        "average_precision": ap,
        "ap_over_prevalence": ap / prevalence,
        "mean_rate_ratio": 2.0,
    }


def _deletion_record(
    *, fold: int, image: str, grade: int, group: str, repeat: int
) -> dict:
    included = {
        "all": set(SCALES),
        "fine": {"s4", "s8"},
        "coarse": {"s16", "s32"},
    }[group]
    scale_counts = {}
    for scale in SCALES:
        lesion_candidates = 3 if image == "IDRiD_01" and scale == "s32" else 1
        nonlesion_candidates = 4 - lesion_candidates
        matched = min(lesion_candidates, nonlesion_candidates) if scale in included else 0
        geometry = 2.0  # reference_count=8 / valid_cells=4
        multiplier = 0.5
        exposure = matched * geometry
        scale_counts[scale] = {
            "included_in_group": scale in included,
            "matching_seed": _stable_seed(
                20261011, fold, image, group, repeat, scale
            ),
            "valid_cells": 4,
            "lesion_candidates": lesion_candidates,
            "nonlesion_candidates": nonlesion_candidates,
            "matched_cells": matched,
            "guided_cells": matched,
            "control_cells": matched,
            "nominal_valid_fraction": matched / 4,
            "geometry_weight": geometry,
            "target_boundary_pre_atom_multiplier": multiplier,
            "guided_nominal_reference_exposure": exposure,
            "control_nominal_reference_exposure": exposure,
            "guided_nominal_weighted_exposure": exposure * multiplier,
            "control_nominal_weighted_exposure": exposure * multiplier,
            "guided_target_rate_removed": matched * (0.3 + 0.01 * repeat),
            "control_target_rate_removed": matched * 0.1,
        }
    cells = sum(item["matched_cells"] for item in scale_counts.values())
    budget = sum(item["guided_nominal_weighted_exposure"] for item in scale_counts.values())
    guided = 0.30 + 0.02 * fold + 0.01 * repeat
    control = 0.10 + 0.005 * repeat
    return {
        "kind": DELETION_KIND,
        "matching_contract": MATCHING_CONTRACT,
        "fold": fold,
        "image_id": image,
        "grade": grade,
        "split": "training",
        "scale_group": group,
        "target_boundary": min(max(grade - 1, 0), 3),
        "repeat": repeat,
        "paired_repeats": 1,
        "scale_counts": scale_counts,
        "guided_deleted_cells": cells,
        "control_deleted_cells": cells,
        "guided_nominal_weighted_exposure": budget,
        "control_nominal_weighted_exposure": budget,
        "guided_expected_grade_delta": guided,
        "control_expected_grade_delta": control,
        "expected_grade_delta_lift": guided - control,
        "guided_target_tail_delta": guided / 2,
        "control_target_tail_delta": control / 2,
        "target_tail_delta_lift": (guided - control) / 2,
        "guided_target_rate_removed": sum(
            item["guided_target_rate_removed"] for item in scale_counts.values()
        ),
        "control_target_rate_removed": sum(
            item["control_target_rate_removed"] for item in scale_counts.values()
        ),
    }


def _toy_records(repeats: int = 3) -> list[dict]:
    records: list[dict] = []
    for fold in (0, 1):
        for image, grade in (("IDRiD_01", 2), ("IDRiD_02", 4)):
            for scale in SCALES:
                for lesion in LESIONS:
                    for boundary in range(4):
                        records.append(
                            _alignment_record(
                                fold=fold, image=image, grade=grade, scale=scale,
                                lesion=lesion, boundary=boundary,
                            )
                        )
            for group in ("all", "fine", "coarse"):
                for repeat in range(repeats):
                    records.append(
                        _deletion_record(
                            fold=fold, image=image, grade=grade, group=group,
                            repeat=repeat,
                        )
                    )
    return records


def _summary() -> dict:
    inventory = {
        "root_label": "IDRiD",
        "files": [{"path": "toy.jpg", "bytes": 1, "sha256": "4" * 64}],
        "file_count": 1,
    }
    inventory["content_checksum_sha256"] = canonical_sha256(inventory)
    summary = {
        "schema": AUDIT_SCHEMA,
        "scope": AUDIT_SCOPE,
        "matching_contract": MATCHING_CONTRACT,
        "average_precision_definition": AP_CONTRACT,
        "idrid_images": 2,
        "paired_repeats": 3,
        "seed": 20261011,
        "checkpoints": [
            {
                "fold": 0, "checkpoint_sha256": "a" * 64,
                "reference_count": 8.0,
                "architecture_signature": ARCHITECTURE_SIGNATURE,
                "config_signature": CONFIG_SIGNATURE,
            },
            {
                "fold": 1, "checkpoint_sha256": "b" * 64,
                "reference_count": 8.0,
                "architecture_signature": ARCHITECTURE_SIGNATURE,
                "config_signature": CONFIG_SIGNATURE,
            },
        ],
        "generation": {
            "source_commit": SOURCE_COMMIT,
            "audit_implementation_sha256": file_sha256(
                Path("tools/audit_origin_idrid_semantics_v2.py")
            ),
            "v1_audit_dependency_sha256": file_sha256(
                Path("tools/audit_origin_idrid_semantics.py")
            ),
            "checkpoint_order": [0, 1],
        },
        "input_dataset_inventory": inventory,
        "summaries": {"matching_census": summarize(_toy_records())["matching_census"]},
    }
    summary["content_checksum_sha256"] = canonical_sha256(summary)
    return summary


def _validation_kwargs() -> dict:
    return {
        "expected_folds": (0, 1),
        "expected_images": 2,
        "expected_paired_repeats": 3,
        "expected_matching_seed": 20261011,
        "expected_dense_image_scale_cases": 1,
        "expected_checkpoint_sha256_by_fold": {"0": "a" * 64, "1": "b" * 64},
        "expected_architecture_signature": ARCHITECTURE_SIGNATURE,
        "expected_config_signature": CONFIG_SIGNATURE,
        "expected_source_commit": SOURCE_COMMIT,
    }


def test_v2_census_asserts_exact_per_scale_matching_and_reports_dense_cases() -> None:
    records = _toy_records()
    census = validate_input_records_v2(
        _summary(), records, **_validation_kwargs()
    )
    assert census["dense_image_scale_cases"] == 1
    assert census["dense_image_scale_case_ids"] == ["IDRiD_01:s32"]
    assert census["deletion_paired_repeat_rows"] == 2 * 2 * 3 * 3
    assert _summary()["summaries"]["matching_census"][
        "v1_one_sided_control_would_truncate_cases"
    ] == 1

    corrupt = json.loads(json.dumps(records))
    deletion = next(row for row in corrupt if row["kind"] == DELETION_KIND)
    deletion["scale_counts"]["s4"]["control_cells"] -= 1
    with pytest.raises(AssertionError, match="matched counts"):
        validate_input_records_v2(
            _summary(), corrupt, **_validation_kwargs()
        )


def test_v2_deletion_aggregation_is_repeat_then_checkpoint_then_image() -> None:
    units = aggregate_deletion_image_units_v2(
        _toy_records(), scale_group="overall", expected_folds=(0, 1),
        expected_repeats=3,
    )
    assert len(units) == 2
    first = units[0]
    assert first["paired_repeats_aggregated_per_checkpoint"] == 3
    assert first["checkpoints_aggregated"] == 2
    assert first["matched_cells_per_side_per_repeat"] == 4
    # Mean guided=.32, control=.105 after averaging folds and repeats.
    assert first["guided_expected_grade_delta"] == pytest.approx(0.32)
    assert first["control_expected_grade_delta"] == pytest.approx(0.105)
    assert first["expected_grade_delta_lift"] == pytest.approx(0.215)


def test_v2_generation_provenance_and_checkpoint_identity_are_enforced() -> None:
    records = _toy_records()
    wrong_commit = _summary()
    wrong_commit["generation"]["source_commit"] = "f" * 40
    wrong_commit.pop("content_checksum_sha256")
    wrong_commit["content_checksum_sha256"] = canonical_sha256(wrong_commit)
    with pytest.raises(AssertionError, match="different source commit"):
        validate_input_records_v2(wrong_commit, records, **_validation_kwargs())

    wrong_checkpoint = _summary()
    wrong_checkpoint["checkpoints"][0]["checkpoint_sha256"] = "c" * 64
    wrong_checkpoint.pop("content_checksum_sha256")
    wrong_checkpoint["content_checksum_sha256"] = canonical_sha256(wrong_checkpoint)
    with pytest.raises(AssertionError, match="checkpoint hashes"):
        validate_input_records_v2(wrong_checkpoint, records, **_validation_kwargs())

    stale_code = _summary()
    stale_code["generation"]["audit_implementation_sha256"] = "d" * 64
    stale_code.pop("content_checksum_sha256")
    stale_code["content_checksum_sha256"] = canonical_sha256(stale_code)
    with pytest.raises(AssertionError, match="implementation hash"):
        validate_input_records_v2(stale_code, records, **_validation_kwargs())


def test_v2_protocol_is_checksum_sealed() -> None:
    path = Path("scripts/protocols/origin_idrid_semantic_statistics_protocol_v2.json")
    payload, config = load_protocol(path)
    assert payload["expected_dense_image_scale_cases"] == 14
    assert payload["expected_paired_repeats"] == 20
    assert len(payload["expected_checkpoint_sha256_by_fold"]) == 10
    assert len(payload["expected_architecture_signature"]) == 64
    assert len(payload["expected_config_signature"]) == 64
    assert config.bootstrap_samples == 10_000
    corrupt = json.loads(path.read_text())
    corrupt["expected_paired_repeats"] = 19
    temporary = path.parent / ".corrupt_idrid_statistics_protocol_v2.json"
    try:
        temporary.write_text(json.dumps(corrupt))
        with pytest.raises(ValueError, match="checksum"):
            load_protocol(temporary)
    finally:
        temporary.unlink(missing_ok=True)


def test_v2_checksum_sealed_small_audit_runs_end_to_end(tmp_path: Path) -> None:
    records = _toy_records()
    audit_dir = tmp_path / "audit"
    audit_dir.mkdir()
    records_path = audit_dir / "idrid_semantic_records_v2.jsonl"
    records_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in records)
    )
    summary = {
        **_summary(),
        "records": records_path.name,
        "records_sha256": file_sha256(records_path),
    }
    summary.pop("content_checksum_sha256")
    summary["content_checksum_sha256"] = canonical_sha256(summary)
    summary_path = audit_dir / "idrid_semantic_summary_v2.json"
    write_json_atomic(summary_path, summary)

    protocol = json.loads(
        Path("scripts/protocols/origin_idrid_semantic_statistics_protocol_v2.json").read_text()
    )
    protocol["expected_images"] = 2
    protocol["expected_checkpoint_folds"] = [0, 1]
    protocol["expected_checkpoint_sha256_by_fold"] = {
        "0": "a" * 64,
        "1": "b" * 64,
    }
    protocol["expected_architecture_signature"] = ARCHITECTURE_SIGNATURE
    protocol["expected_config_signature"] = CONFIG_SIGNATURE
    protocol["expected_segmentation_split_counts"] = {"training": 2}
    protocol["expected_paired_repeats"] = 3
    protocol["expected_dense_image_scale_cases"] = 1
    protocol["config"]["bootstrap_samples"] = 1000
    protocol["config"]["permutation_samples"] = 999
    protocol.pop("content_checksum_sha256")
    protocol["content_checksum_sha256"] = canonical_sha256(protocol)
    protocol_path = tmp_path / "protocol_v2.json"
    write_json_atomic(protocol_path, protocol)

    output = tmp_path / "statistics"
    manifest = analyze_idrid_semantics_v2(
        summary_path=summary_path,
        protocol_path=protocol_path,
        output_dir=output,
        expected_source_commit=SOURCE_COMMIT,
    )
    assert manifest["schema"] == MANIFEST_SCHEMA
    assert manifest["input_census"]["dense_image_scale_cases"] == 1
    assert set(manifest["artifacts"]) == {
        "alignment_statistics", "deletion_statistics", "image_level_units"
    }
    unsigned = dict(manifest)
    assert unsigned.pop("content_checksum_sha256") == canonical_sha256(unsigned)
    for artifact in manifest["artifacts"].values():
        path = Path(artifact["path"])
        assert path.is_file()
        assert file_sha256(path) == artifact["sha256"]
    unit_path = Path(manifest["artifacts"]["image_level_units"]["path"])
    unit_rows = [json.loads(line) for line in unit_path.read_text().splitlines()]
    assert unit_rows
    assert {row["schema"] for row in unit_rows} == {UNIT_SCHEMA}
