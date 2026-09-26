from __future__ import annotations

import copy
import math

import pytest

import scripts.release_origin_fold9_ablation_outer as release_module
from scripts.aggregate_origin_fold9_ablations import (
    metrics_from_rows,
    paired_patient_cluster_bootstrap,
)
from scripts.origin_fold9_ablation_common import (
    make_protocol_payload,
    variants_for_group,
    write_json_atomic,
)
from scripts.origin_fold9_ablation_common import (
    DATASET,
    EXPECTED_SPLIT_COUNTS,
    EXPECTED_SPLIT_HISTOGRAMS,
    EXPECTED_SPLIT_SIGNATURE,
    FOLD,
    SPLIT_SCHEMA,
    TRAIN_MARKER_SCHEMA,
    canonical_sha256,
    file_sha256,
)
from scripts.release_origin_fold9_ablation_outer import verify_all_training_markers


def _row(
    index: int,
    *,
    variant: str,
    label: int,
    prediction: int,
    patient: str,
) -> dict[str, object]:
    probabilities = [0.025] * 5
    probabilities[prediction] = 0.9
    expected = sum(grade * probability for grade, probability in enumerate(probabilities))
    return {
        "variant": variant,
        "fold": 9,
        "outer_index": index,
        "image_path_relative": f"train/{patient}_{index % 2}.jpeg",
        "patient_id": patient,
        "true_label": label,
        "primary_prediction": prediction,
        "expected_grade": expected,
        **{f"p_grade_{grade}": probability for grade, probability in enumerate(probabilities)},
    }


def _five_grade_rows(variant: str, *, make_errors: bool) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for patient_index in range(5):
        label = patient_index
        for eye in range(2):
            index = 2 * patient_index + eye
            prediction = label
            if make_errors and eye == 1 and label > 0:
                prediction = label - 1
            rows.append(
                _row(
                    index,
                    variant=variant,
                    label=label,
                    prediction=prediction,
                    patient=f"p{patient_index}",
                )
            )
    return rows


def test_metrics_replay_class_map_and_expected_grade_mae() -> None:
    rows = _five_grade_rows("origin_full", make_errors=False)
    metrics = metrics_from_rows(rows)
    assert metrics["n"] == 10
    assert metrics["acc"] == 100.0
    assert metrics["mae"] == 0.0
    assert metrics["qwk"] == 1.0
    assert metrics["balanced_acc"] == 100.0
    assert metrics["macro_f1"] == 1.0
    assert metrics["expected_grade_mae"] > 0.0
    assert math.isfinite(metrics["ece"])


def test_metrics_reject_prediction_that_is_not_probability_argmax() -> None:
    rows = _five_grade_rows("origin_full", make_errors=False)
    rows[0]["primary_prediction"] = 1
    with pytest.raises(ValueError, match="class-MAP"):
        metrics_from_rows(rows)


def test_paired_patient_bootstrap_is_zero_for_identical_predictions() -> None:
    reference = _five_grade_rows("origin_full", make_errors=True)
    identical = copy.deepcopy(reference)
    for row in identical:
        row["variant"] = "pooled_softmax"
    result = paired_patient_cluster_bootstrap(
        {"origin_full": reference, "pooled_softmax": identical},
        samples=128,
        seed=7,
        chunk_size=17,
    )
    assert result["patient_clusters"] == 5
    assert result["images"] == 10
    for metric, payload in result["comparisons"]["pooled_softmax"].items():
        assert payload["candidate_minus_reference"] == 0.0, metric
        assert payload["ci95_percentile"] == [0.0, 0.0], metric


def test_paired_patient_bootstrap_detects_strictly_better_candidate() -> None:
    reference = _five_grade_rows("origin_full", make_errors=True)
    candidate = _five_grade_rows("pooled_softmax", make_errors=False)
    result = paired_patient_cluster_bootstrap(
        {"origin_full": reference, "pooled_softmax": candidate},
        samples=256,
        seed=19,
        chunk_size=31,
    )
    comparison = result["comparisons"]["pooled_softmax"]
    assert comparison["acc"]["candidate_minus_reference"] > 0.0
    assert comparison["mae"]["candidate_minus_reference"] < 0.0
    assert comparison["qwk"]["candidate_minus_reference"] > 0.0
    assert comparison["acc"]["bootstrap_fraction_candidate_better"] > 0.95
    assert comparison["mae"]["bootstrap_fraction_candidate_better"] > 0.95


def test_paired_patient_bootstrap_rejects_misaligned_images() -> None:
    reference = _five_grade_rows("origin_full", make_errors=True)
    candidate = _five_grade_rows("pooled_softmax", make_errors=False)
    candidate[0], candidate[1] = candidate[1], candidate[0]
    with pytest.raises(AssertionError, match="not paired"):
        paired_patient_cluster_bootstrap(
            {"origin_full": reference, "pooled_softmax": candidate},
            samples=8,
            seed=1,
        )


def test_missing_training_marker_blocks_every_outer_evaluation(
    tmp_path, monkeypatch
) -> None:
    root = tmp_path / "experiment"
    data_root = tmp_path / "data"
    worktree = tmp_path / "worktree"
    root.mkdir()
    data_root.mkdir()
    worktree.mkdir()
    protocol = make_protocol_payload(
        selected_variants=variants_for_group("core"),
        group="core",
        launch_commit="1" * 40,
        experiment_root=root,
        data_root=data_root,
        immutable_worktree=worktree,
        tag="missing-marker-test",
    )
    protocol_path = root / "LOCKED_PROTOCOL.json"
    write_json_atomic(protocol_path, protocol)
    evaluated: list[str] = []

    def forbidden_evaluation(**kwargs):
        evaluated.append(str(kwargs.get("variant")))
        raise AssertionError("outer evaluator must remain unreachable")

    monkeypatch.setattr(release_module, "_evaluate_variant", forbidden_evaluation)
    with pytest.raises(FileNotFoundError, match="outer release remains locked"):
        release_module.release(
            root=root,
            protocol_path=protocol_path,
            data_root=data_root,
            num_workers=0,
        )
    assert evaluated == []


def test_bad_late_checkpoint_blocks_every_outer_evaluation(
    tmp_path, monkeypatch
) -> None:
    root = tmp_path / "experiment"
    data_root = tmp_path / "data"
    worktree = tmp_path / "worktree"
    root.mkdir()
    data_root.mkdir()
    worktree.mkdir()
    variants = variants_for_group("core")
    protocol = make_protocol_payload(
        selected_variants=variants,
        group="core",
        launch_commit="1" * 40,
        experiment_root=root,
        data_root=data_root,
        immutable_worktree=worktree,
        tag="late-checkpoint-test",
    )
    protocol_path = root / "LOCKED_PROTOCOL.json"
    write_json_atomic(protocol_path, protocol)
    verified = {variant: {"variant": variant} for variant in variants}
    reconstructed: list[str] = []
    evaluated: list[str] = []

    monkeypatch.setattr(
        release_module,
        "verify_all_training_markers",
        lambda **_: verified,
    )

    def reconstruct_or_fail(*, variant, verified):
        reconstructed.append(variant)
        if variant == variants[-1]:
            raise ValueError("bad late checkpoint")
        return object(), object()

    def forbidden_evaluation(**kwargs):
        evaluated.append(str(kwargs.get("variant")))
        raise AssertionError("outer evaluator must remain unreachable")

    monkeypatch.setattr(
        release_module, "_reconstruct_validated_model", reconstruct_or_fail
    )
    monkeypatch.setattr(release_module, "_evaluate_variant", forbidden_evaluation)
    with pytest.raises(ValueError, match="bad late checkpoint"):
        release_module.release(
            root=root,
            protocol_path=protocol_path,
            data_root=data_root,
            num_workers=0,
        )
    assert reconstructed == list(variants)
    assert evaluated == []
    assert not (root / "release").exists()
    assert not list(root.glob(".release.staging.*"))


def _write_json(path, payload) -> None:
    import json

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_outer_gate_verifies_complete_training_marker_before_release(tmp_path) -> None:
    variant = "origin_full"
    fold_dir = tmp_path / "workers" / variant / f"fold{FOLD}"
    paths = {
        "checkpoint": fold_dir / "best.pth",
        "result": fold_dir / "result.json",
        "split_manifest": fold_dir / "split_manifest.json",
        "history": fold_dir / "history.csv",
        "validation_certificate": fold_dir / "validation_certificates.json",
    }
    split = {
        "schema": SPLIT_SCHEMA,
        "dataset": DATASET,
        "fold": FOLD,
        "ablation_variant": variant,
        "signature": EXPECTED_SPLIT_SIGNATURE,
        "evaluation_scope": "inner_validation_only_outer_locked",
        "counts": EXPECTED_SPLIT_COUNTS,
        "histograms": EXPECTED_SPLIT_HISTOGRAMS,
    }
    result = {
        "fold": FOLD,
        "test_evaluated": False,
        "test": None,
        "best_epoch": 17,
        "ablation_variant": variant,
        "parameter_count": 10,
        "trainable_parameter_count": 10,
        "training_wall_time_seconds": 1.0,
        "training_efficiency_scope": "complete_training_process",
    }
    _write_json(paths["split_manifest"], split)
    _write_json(paths["result"], result)
    paths["checkpoint"].write_bytes(b"selected-checkpoint")
    paths["history"].write_text("epoch,val_acc\n17,85.0\n", encoding="utf-8")
    _write_json(paths["validation_certificate"], {"status": "test-fixture"})
    protocol = {"content_checksum_sha256": "p" * 64, "launch_commit": "abc123"}
    marker = {
        "schema": TRAIN_MARKER_SCHEMA,
        "dataset": DATASET,
        "fold": FOLD,
        "variant": variant,
        "evaluation_scope": "inner_validation_only_outer_locked",
        "outer_test_evaluated": False,
        "protocol_checksum_sha256": protocol["content_checksum_sha256"],
        "launch_commit": protocol["launch_commit"],
        "split_signature": EXPECTED_SPLIT_SIGNATURE,
        "best_epoch": 17,
        "training_efficiency_scope": "complete_training_process",
        "artifact_paths": {name: str(path.resolve()) for name, path in paths.items()},
        "artifact_sha256": {name: file_sha256(path) for name, path in paths.items()},
    }
    marker["content_checksum_sha256"] = canonical_sha256(marker)
    _write_json(fold_dir / "ABLATION_TRAIN_COMPLETE.json", marker)

    verified = verify_all_training_markers(
        root=tmp_path, protocol=protocol, variants=(variant,)
    )
    assert tuple(verified) == (variant,)
    assert verified[variant]["result"]["best_epoch"] == 17

    paths["history"].write_text("mutated\n", encoding="utf-8")
    with pytest.raises(AssertionError, match="artifact changed"):
        verify_all_training_markers(
            root=tmp_path, protocol=protocol, variants=(variant,)
        )
