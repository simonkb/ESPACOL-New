from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from scripts.audit_origin_fold9_ablation_training import audit_training_run
from scripts.origin_fold9_ablation_common import (
    CORE_VARIANTS,
    EXPECTED_SPLIT_COUNTS,
    EXPECTED_SPLIT_HISTOGRAMS,
    EXPECTED_SPLIT_SIGNATURE,
    FULL_VARIANTS,
    SPLIT_SCHEMA,
    TRAIN_MARKER_SCHEMA,
    TRAINING_STREAM_SEED,
    canonical_sha256,
    config_signature_for_variant,
    file_sha256,
    frozen_config_for_variant,
    get_variant_spec,
    make_protocol_payload,
    read_json,
    validate_split_manifest,
    variants_for_group,
    verify_checksummed_payload,
    verify_protocol,
    write_json_atomic,
)
from scripts.origin_v3_cv_common import metrics_from_confusion


ROOT = Path(__file__).resolve().parents[1]


def test_training_audit_direct_script_fallback_import_is_complete() -> None:
    """Match the ``python scripts/audit_...py`` worker invocation exactly."""

    code = """
import pathlib
import runpy
import sys
root = pathlib.Path(sys.argv[1]).resolve()
sys.path = [str(root / 'scripts')] + [
    entry for entry in sys.path
    if pathlib.Path(entry or '.').resolve() != root
]
scope = runpy.run_path(str(root / 'scripts' / 'audit_origin_fold9_ablation_training.py'))
assert scope['TRAINING_STREAM_SEED'] == 100051
"""
    subprocess.run([sys.executable, "-c", code, str(ROOT)], check=True)


def test_variant_registry_is_ordered_complete_and_isolated() -> None:
    assert FULL_VARIANTS == (
        "origin_full",
        "pooled_softmax",
        "pooled_cumulative_logit",
        "ledger_sequential_hazard",
        "origin_simplex_direct",
        "origin_nll_only",
        "origin_fine_only",
        "origin_coarse_only",
        "origin_independent_sigmoid",
        "pooled_conditional",
    )
    assert CORE_VARIANTS == FULL_VARIANTS[:5]
    assert variants_for_group("full") == FULL_VARIANTS
    assert variants_for_group("CORE") == CORE_VARIANTS
    assert len(FULL_VARIANTS) == len(set(FULL_VARIANTS)) == 10
    with pytest.raises(ValueError):
        variants_for_group("custom")

    expected_overrides = {
        "origin_simplex_direct": {"atom_mode": "simplex_direct"},
        "origin_nll_only": {"rps_weight": 0.0},
        "origin_fine_only": {"evidence_scales": ["s4", "s8"]},
        "origin_coarse_only": {"evidence_scales": ["s16", "s32"]},
        "origin_independent_sigmoid": {"atom_mode": "independent"},
    }
    for name in FULL_VARIANTS:
        assert get_variant_spec(name)["config_overrides"] == expected_overrides.get(name, {})
    mutable = get_variant_spec("origin_fine_only")
    mutable["config_overrides"]["evidence_scales"].append("s32")
    assert get_variant_spec("origin_fine_only")["config_overrides"] == {
        "evidence_scales": ["s4", "s8"]
    }
    assert len({config_signature_for_variant(name) for name in FULL_VARIANTS}) == 10


def _make_protocol(tmp_path: Path, group: str = "core") -> tuple[Path, dict]:
    root = tmp_path / "experiment"
    data = tmp_path / "data"
    worktree = tmp_path / "worktree"
    root.mkdir()
    data.mkdir()
    worktree.mkdir()
    payload = make_protocol_payload(
        selected_variants=variants_for_group(group),
        group=group,
        launch_commit="1" * 40,
        experiment_root=root,
        data_root=data,
        immutable_worktree=worktree,
        tag="unit-test",
    )
    path = root / "LOCKED_PROTOCOL.json"
    write_json_atomic(path, payload)
    return path, payload


def test_protocol_roundtrip_and_checksum_reject_mutation(tmp_path: Path) -> None:
    path, payload = _make_protocol(tmp_path)
    assert verify_protocol(
        read_json(path),
        expected_launch_commit="1" * 40,
        expected_root=tmp_path / "experiment",
        expected_data_root=tmp_path / "data",
    ) == CORE_VARIANTS
    policy = payload["evaluation_policy"]
    assert policy["paired_patient_cluster_bootstrap_samples"] == 10_000
    assert policy["paired_patient_cluster_bootstrap_seed"] == 26_092_026
    unsigned = dict(payload)
    del unsigned["content_checksum_sha256"]
    assert payload["content_checksum_sha256"] == canonical_sha256(unsigned)
    mutated = json.loads(json.dumps(payload))
    mutated["selected_variants"].reverse()
    with pytest.raises(ValueError, match="checksum mismatch"):
        verify_checksummed_payload(mutated)


def test_split_identity_is_exact_and_outer_locked() -> None:
    split = {
        "schema": SPLIT_SCHEMA,
        "dataset": "dr",
        "fold": 9,
        "ablation_variant": "origin_full",
        "signature": EXPECTED_SPLIT_SIGNATURE,
        "evaluation_scope": "inner_validation_only_outer_locked",
        "counts": EXPECTED_SPLIT_COUNTS,
        "histograms": EXPECTED_SPLIT_HISTOGRAMS,
    }
    validate_split_manifest(split)
    changed = json.loads(json.dumps(split))
    changed["counts"]["validation"] += 1
    with pytest.raises(ValueError, match="split manifest mismatch"):
        validate_split_manifest(changed)


def test_training_audit_emits_sealed_marker(tmp_path: Path) -> None:
    protocol_path, protocol = _make_protocol(tmp_path)
    variant = "origin_full"
    task_dir = tmp_path / "experiment" / "workers" / variant
    fold_dir = task_dir / "fold9"
    fold_dir.mkdir(parents=True)

    confusion = [
        [2330, 10, 0, 0, 0],
        [20, 180, 10, 0, 0],
        [0, 20, 440, 16, 0],
        [0, 0, 15, 55, 5],
        [0, 0, 0, 10, 51],
    ]
    validation = metrics_from_confusion(confusion)
    validation["ece"] = 0.02
    config = frozen_config_for_variant(variant)
    config["run_dir"] = str(task_dir)
    checkpoint = {
        "schema": "origin-checkpoint-v3",
        "fold": 9,
        "split_signature": EXPECTED_SPLIT_SIGNATURE,
        "epoch": 7,
        "config": config,
        "metrics": validation,
        "architecture": {
            "declared": {
                "ablation_variant": variant,
                "posterior_path": "pure_birth_generator",
                "supports_exact_replay": True,
            }
        },
        "architecture_signature": "a" * 64,
        "implementation_signature": "b" * 64,
        "config_signature": "c" * 64,
    }
    torch.save(checkpoint, fold_dir / "best.pth")
    write_json_atomic(
        fold_dir / "result.json",
        {
            "fold": 9,
            "best_epoch": 7,
            "best_validation": validation,
            "test_evaluated": False,
            "run_dir": str(fold_dir),
            "ablation_variant": variant,
            "training_stream_seed": TRAINING_STREAM_SEED,
            "training_efficiency_scope": "complete_training_process",
            },
    )
    write_json_atomic(
        fold_dir / "split_manifest.json",
        {
            "schema": SPLIT_SCHEMA,
            "dataset": "dr",
            "fold": 9,
            "ablation_variant": variant,
            "signature": EXPECTED_SPLIT_SIGNATURE,
            "evaluation_scope": "inner_validation_only_outer_locked",
            "counts": EXPECTED_SPLIT_COUNTS,
            "histograms": EXPECTED_SPLIT_HISTOGRAMS,
        },
    )
    (fold_dir / "history.csv").write_text("epoch,val_acc\n1,80\n", encoding="utf-8")
    certificate = {
        "schema": "origin-ablation-certificate-status-v1",
        "scope": "inner_validation_only",
        "fold": 9,
        "split_signature": EXPECTED_SPLIT_SIGNATURE,
        "checkpoint_epoch": 7,
        "checkpoint_sha256": file_sha256(fold_dir / "best.pth"),
        "implementation_signature": "b" * 64,
        "architecture_signature": "a" * 64,
        "ablation_variant": variant,
        "certificate_status": "test_fixture",
    }
    certificate["content_checksum_sha256"] = canonical_sha256(certificate)
    write_json_atomic(fold_dir / "validation_certificates.json", certificate)

    output = audit_training_run(
        variant=variant, task_run_dir=task_dir, protocol_path=protocol_path
    )
    marker = read_json(output["marker"])
    verify_checksummed_payload(marker, schema=TRAIN_MARKER_SCHEMA)
    assert marker["outer_test_evaluated"] is False
    assert marker["protocol_checksum_sha256"] == protocol["content_checksum_sha256"]
    assert marker["validation"]["n"] == EXPECTED_SPLIT_COUNTS["validation"]
    assert set(marker["artifact_sha256"]) == {
        "checkpoint", "result", "split_manifest", "history",
        "validation_certificate",
    }


def test_slurm_pipeline_seals_outer_release_and_has_valid_shell() -> None:
    scripts = {
        name: (ROOT / "scripts" / name).read_text(encoding="utf-8")
        for name in (
            "launch_origin_fold9_ablations.sh",
            "submit_origin_fold9_ablation_preflight.sh",
            "submit_origin_fold9_ablation_train_array.sh",
            "submit_origin_fold9_ablation_outer_release.sh",
            "submit_origin_fold9_ablation_aggregate.sh",
        )
    }
    for name in scripts:
        subprocess.run(
            ["bash", "-n", str(ROOT / "scripts" / name)], check=True
        )
    launch = scripts["launch_origin_fold9_ablations.sh"]
    assert "git worktree add --detach" in launch
    assert 'GROUP="${ORIGIN_FOLD9_ABLATION_GROUP:-full}"' in launch
    assert '--dependency="afterany:${PREFLIGHT_JOB}"' in launch
    assert '--dependency="afterany:${TRAIN_JOB}"' in launch
    assert '--dependency="afterany:${RELEASE_JOB}"' in launch
    train = scripts["submit_origin_fold9_ablation_train_array.sh"]
    assert "PREFLIGHT_COMPLETE.json" in train
    assert "PREFLIGHT_MARKER_SCHEMA" in train
    assert "train_origin_ablation.py" in train
    assert "--folds 9" in train
    assert "--skip_test" in train
    assert "--include_test" not in train
    assert "audit_origin_fold9_ablation_training.py" in train
    release = scripts["submit_origin_fold9_ablation_outer_release.sh"]
    assert "all_training_markers_verified" in release
    assert "release_origin_fold9_ablation_outer.py" in release
    aggregate = scripts["submit_origin_fold9_ablation_aggregate.sh"]
    assert "aggregate_origin_fold9_ablations.py" in aggregate
    assert "10000" in aggregate and "26092026" in aggregate
    assert "ORIGIN_FOLD9_BOOTSTRAP" not in aggregate
