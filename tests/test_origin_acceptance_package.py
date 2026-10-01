from __future__ import annotations

import csv
import gzip
import hashlib
import io
import json
from pathlib import Path
import subprocess

import numpy as np
import pytest

from tools.assemble_origin_acceptance_package import (
    COMPLETION_FILENAME,
    MANIFEST_FILENAME,
    EXPECTED_SHORTCUT_VARIANTS,
    BASELINE_BOOTSTRAP_METRICS,
    BASELINE_SUMMARY_METRICS,
    _assert_privacy_safe_release,
    _validate_baseline_execution_provenance,
    assemble_package,
    canonical_sha256,
    file_sha256,
    validate_baseline_aggregate,
    validate_numerical,
)


def _write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return path


def _write_sealed(path: Path, payload: dict) -> Path:
    payload = dict(payload)
    payload["content_checksum_sha256"] = canonical_sha256(payload)
    return _write_json(path, payload)


def _reseal(payload: dict) -> dict:
    payload = dict(payload)
    payload.pop("content_checksum_sha256", None)
    payload["content_checksum_sha256"] = canonical_sha256(payload)
    return payload


def _git_repo(path: Path) -> tuple[Path, str]:
    path.mkdir()
    subprocess.run(["git", "init"], cwd=path, check=True, capture_output=True)
    subprocess.run(
        ["git", "config", "user.email", "audit@example.invalid"], cwd=path,
        check=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Audit Test"], cwd=path, check=True,
    )
    (path / "tools").mkdir()
    (path / "scripts" / "protocols").mkdir(parents=True)
    (path / "tools" / "assemble_origin_acceptance_package.py").write_text(
        Path("tools/assemble_origin_acceptance_package.py").read_text()
    )
    (path / "tools" / "analyze_origin_idrid_semantics.py").write_text("# idrid\n")
    (path / "scripts" / "analyze_origin_oof_statistics.py").write_text("# oof\n")
    (path / "scripts" / "origin_acceptance_baseline_common.py").write_text(
        "# executable protocol\n"
    )
    (path / "scripts" / "submit_origin_acceptance_full_array.sh").write_text(
        '#!/bin/bash\ncd "${ORIGIN_REPO_ROOT:?ORIGIN_REPO_ROOT is required}"\n'
        'python train_origin.py --epochs 35\n'
    )
    _write_sealed(
        path / "scripts" / "protocols" / "origin_idrid_semantic_statistics_protocol.json",
        {"schema": "origin-idrid-semantic-statistics-protocol-v1"},
    )
    _write_sealed(
        path / "scripts" / "protocols" / "origin_oof_statistics_protocol.json",
        {"schema": "origin-oof-statistics-protocol-v2"},
    )
    subprocess.run(["git", "add", "."], cwd=path, check=True)
    subprocess.run(["git", "commit", "-m", "sealed"], cwd=path, check=True)
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=path, check=True,
        capture_output=True, text=True,
    ).stdout.strip()
    return path, commit


def _numeric(root: Path) -> Path:
    return _write_sealed(
        root / "numeric.json",
        {
            "schema": "origin-decoder-independent-numerical-audit-v1",
            "independent_reference": "mpmath.expm",
            "rate_domain": [0.0, 64.0],
            "n_forward_cases": 138,
            "n_gradient_cases": 12,
            "summary": {"passed": True},
        },
    )


def _decoder(root: Path) -> Path:
    return _write_json(
        root / "decoder.json",
        {
            "schema": "origin-decoder-contract-audit-v1",
            "passed": True,
            "trials": 64,
            "maximum_ctmc_semigroup_error": 1e-15,
            "minimum_sequential_semigroup_failure": 1e-3,
        },
    )


def _anonymous_membership(path: Path, rows: int = 1) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as raw:
        with gzip.GzipFile(filename="", fileobj=raw, mode="wb", mtime=0) as compressed:
            with io.TextIOWrapper(compressed, encoding="utf-8", newline="") as text:
                writer = csv.DictWriter(
                    text,
                    fieldnames=[
                        "image_id_sha256", "patient_cluster_sha256", "label", "split"
                    ],
                    lineterminator="\n",
                )
                writer.writeheader()
                for index in range(rows):
                    writer.writerow(
                        {
                            "image_id_sha256": hashlib.sha256(f"image{index}".encode()).hexdigest(),
                            "patient_cluster_sha256": hashlib.sha256(
                                f"cluster{index}".encode()
                            ).hexdigest(),
                            "label": index % 5,
                            "split": "outer_test",
                        }
                    )


def _artifact_manifest(root: Path, repo: Path, commit: str) -> Path:
    cv = root / "cv"
    submission = _write_sealed(
        cv / "SUBMISSION.json",
        {"schema": "origin-v3-full-cv-submission-v1", "launch_commit": commit},
    )
    output = root / "artifact_manifest"
    checkpoints = []
    memberships = []
    for dataset, folds in (("aptos", 5), ("dr", 10)):
        for fold in range(folds):
            checkpoint = cv / dataset / f"fold{fold}.pth"
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            checkpoint.write_bytes(f"{dataset}-{fold}".encode())
            membership = output / "split_memberships" / f"{dataset}_fold{fold}.csv.gz"
            _anonymous_membership(membership)
            checkpoints.append(
                {
                    "dataset": dataset,
                    "fold": fold,
                    "relative_path": str(checkpoint),
                    "bytes": checkpoint.stat().st_size,
                    "sha256": file_sha256(checkpoint),
                    "checkpoint_schema": "origin-checkpoint-v3",
                    "split_signature": "1" * 64,
                    "implementation_signature": "2" * 64,
                    "architecture_signature": "3" * 64,
                    "config_signature": "4" * 64,
                }
            )
            memberships.append(
                {
                    "dataset": dataset,
                    "fold": fold,
                    "relative_path": str(membership.relative_to(output)),
                    "rows": 1,
                    "sha256": file_sha256(membership),
                }
            )
    return _write_sealed(
        output / "manifest.json",
        {
            "schema": "origin-acceptance-artifact-manifest-v1",
            "cv_root": str(cv),
            "source_provenance": {
                "audit_source_commit": commit,
                "tracked_tree_dirty": False,
            },
            "full_cv_submission": {
                "relative_path": submission.name,
                "sha256": file_sha256(submission),
                "training_launch_commit": commit,
            },
            "raw_patient_data_redistributed": False,
            "checkpoints": checkpoints,
            "split_memberships": memberships,
        },
    )


def _idrid(root: Path, repo: Path) -> Path:
    semantic = root / "idrid"
    stats = semantic / "statistics"
    summary = _write_json(semantic / "summary.json", {"schema": "raw"})
    records = semantic / "records.jsonl"
    records.write_text("{}\n")
    alignment = _write_sealed(stats / "alignment.json", {"schema": "align"})
    deletion = _write_sealed(stats / "deletion.json", {"schema": "delete"})
    units = stats / "units.jsonl"
    units.write_text("{}\n")
    protocol = json.loads(
        (repo / "scripts" / "protocols" / "origin_idrid_semantic_statistics_protocol.json").read_text()
    )
    return _write_sealed(
        stats / "idrid_statistics_manifest.json",
        {
            "schema": "origin-idrid-semantic-statistics-manifest-v1",
            "protocol_checksum_sha256": protocol["content_checksum_sha256"],
            "input_summary_path": str(summary),
            "input_summary_sha256": file_sha256(summary),
            "input_records_path": str(records),
            "input_records_sha256": file_sha256(records),
            "input_census": {"images": [f"private-{index}" for index in range(81)]},
            "analysis_implementation_sha256": file_sha256(
                repo / "tools" / "analyze_origin_idrid_semantics.py"
            ),
            "artifacts": {
                "alignment_statistics": {
                    "path": str(alignment), "sha256": file_sha256(alignment),
                    "content_checksum_sha256": json.loads(alignment.read_text())[
                        "content_checksum_sha256"
                    ],
                },
                "deletion_statistics": {
                    "path": str(deletion), "sha256": file_sha256(deletion),
                    "content_checksum_sha256": json.loads(deletion.read_text())[
                        "content_checksum_sha256"
                    ],
                },
                "image_level_units": {
                    "path": str(units), "sha256": file_sha256(units), "rows": 1,
                },
            },
        },
    )


def _oof(root: Path, repo: Path, commit: str, *, gate: str) -> Path:
    source = root / "oof_sources"
    inputs = []
    for dataset, n in (("aptos", 3662), ("dr", 35126)):
        aggregate = _write_sealed(
            source / f"{dataset}.json",
            {"schema": "origin-oof-intervention-aggregate-v1"},
        )
        payload = json.loads(aggregate.read_text())
        inputs.append(
            {
                "dataset": dataset,
                "path": str(aggregate),
                "sha256": file_sha256(aggregate),
                "content_checksum_sha256": payload["content_checksum_sha256"],
                "n_images": n,
                "model_frozen_base_commit": commit,
            }
        )
    output = root / "oof_stats"
    artifact_records = {}
    names = {
        "proper_score_calibration": "proper.json",
        "grading_bootstrap": "grading.json",
        "intervention_bootstrap": "intervention.json",
        "rf_footprint_diagnostics": "rf.json",
        "gate_a": "gate.json",
    }
    for name, filename in names.items():
        if name == "gate_a":
            nested = {
                "schema": "origin-oof-gate-a-adjudication-v1",
                "status": gate,
                "claim_authorized": gate == "pass",
                "input_completeness_issues": [] if gate == "pass" else ["failed clause"],
                "clauses": {
                    key: {"status": gate}
                    for key in (
                        "auc_advantage", "positive_grade_strata", "concentration",
                        "retention_preservation",
                    )
                },
            }
        else:
            nested = {"schema": f"toy-{name}"}
        artifact = _write_sealed(output / filename, nested)
        nested_payload = json.loads(artifact.read_text())
        artifact_records[name] = {
            "path": str(artifact),
            "sha256": file_sha256(artifact),
            "content_checksum_sha256": nested_payload["content_checksum_sha256"],
        }
        if name == "gate_a":
            artifact_records[name].update(
                {"gate_status": gate, "claim_authorized": gate == "pass"}
            )
    protocol = json.loads(
        (repo / "scripts" / "protocols" / "origin_oof_statistics_protocol.json").read_text()
    )
    implementation_hash = file_sha256(repo / "scripts" / "analyze_origin_oof_statistics.py")
    return _write_sealed(
        output / "statistics_manifest.json",
        {
            "schema": "origin-oof-statistics-manifest-v2",
            "statistics_protocol_checksum_sha256": protocol["content_checksum_sha256"],
            "statistics_runtime_source": {
                "git_commit": commit,
                "git_commit_available": True,
                "tracked_worktree_clean": True,
                "analysis_implementation_sha256": implementation_hash,
            },
            "analysis_implementation_sha256": implementation_hash,
            "inputs": inputs,
            "artifacts": artifact_records,
        },
    )


def _shortcut(root: Path, commit: str, *, passed: bool) -> Path:
    suite = root / "shortcut"
    checkpoint = suite / "checkpoint.pth"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    checkpoint.write_bytes(b"checkpoint")
    prediction = suite / "predictions.jsonl"
    prediction.write_text("{}\n")
    audits = []
    for variant in EXPECTED_SHORTCUT_VARIANTS:
        for arm in ("shortcut", "cue_only", "clean"):
            for family in ("localized", "border", "diffuse"):
                for seed in (1701, 2603, 3907):
                    audit = _write_sealed(
                        suite / "workers" / variant / arm / family / f"seed{seed}.json",
                        {
                            "schema": "origin-ordinal-shortcut-audit-v1",
                            "complete_outer_fold": True,
                            "prediction_artifact": str(prediction),
                            "prediction_artifact_sha256": file_sha256(prediction),
                            "checkpoint": str(checkpoint),
                            "checkpoint_sha256": file_sha256(checkpoint),
                            "model_variant": variant,
                            "shortcut_arm": arm,
                            "shortcut_family": family,
                            "training_seed": seed,
                        },
                    )
                    audits.append(str(audit))
    _write_sealed(
        suite / "SUBMISSION.json",
        {
            "schema": "origin-ordinal-shortcut-comparator-submission-v2",
            "launch_commit": commit,
        },
    )
    _write_sealed(
        suite / "PROTOCOL_V2.json",
        {
            "schema": "origin-ordinal-shortcut-comparator-protocol-v2",
            "launch_commit": commit,
            "additional_training_workers": 84,
        },
    )
    focal_checks = {
        name: passed
        for name in (
            "cue_only_positive_control", "cue_learned_all_seeds",
            "localization_seed_requirement", "deletion_all_seeds",
            "internal_pixel_correlation_all_seeds", "boundary_response_selectivity",
            "clean_negative_control",
        )
    }
    diffuse_checks = {
        name: passed
        for name in (
            "cue_only_positive_control", "cue_learned_all_seeds",
            "boundary_response_selectivity", "diffuse_nonfocal_support",
        )
    }
    families = {
        "localized": {"gate_b": {"checks": focal_checks, "passed": passed}},
        "border": {"gate_b": {"checks": focal_checks, "passed": passed}},
        "diffuse": {
            "gate_b": {
                "checks": diffuse_checks,
                "passed": passed,
                "gate_kind": "diffuse_nonfocal_negative_locality",
                "localization_gate_applicable": False,
                "must_not_be_reported_as_focal_lesion": True,
            }
        },
    }
    return _write_sealed(
        suite / "COMPARATOR_SHORTCUT_RESULTS_V2.json",
        {
            "schema": "origin-ordinal-shortcut-pilot-aggregate-v2",
            "expected_training_seeds": [1701, 2603, 3907],
            "model_variants": {
                variant: {
                    "official_author_implementation": False,
                    "families": families if variant == "origin_ctmc" else {
                        name: {} for name in ("localized", "border", "diffuse")
                    },
                }
                for variant in EXPECTED_SHORTCUT_VARIANTS
            },
            "audit_files": audits,
            "origin_family_gates_passed": [passed, passed, passed],
            "passed": passed,
        },
    )


def _reliability_bins(n: int) -> list[dict]:
    return [
        {
            "bin": index,
            "lower": index / 15,
            "upper": (index + 1) / 15,
            "right_edge_inclusive": index == 14,
            "count": n if index == 0 else 0,
            "mean_predicted": 0.0 if index == 0 else None,
            "empirical_frequency": 0.0 if index == 0 else None,
            "absolute_gap": 0.0 if index == 0 else None,
        }
        for index in range(15)
    ]


def _threshold_reliability(n: int, *, include_n: bool) -> dict:
    payload = {
        "bin_count": 15,
        "threshold_ece": 0.0,
        "threshold_ece_by_boundary": [0.0] * 4,
        "threshold_binary_brier": 0.0,
        "threshold_binary_brier_by_boundary": [0.0] * 4,
        "boundaries": [
            {
                "boundary": boundary,
                "event": f"Y>{boundary}",
                "ece": 0.0,
                "binary_brier": 0.0,
                "bins": _reliability_bins(n),
            }
            for boundary in range(4)
        ],
    }
    if include_n:
        payload["n"] = n
        payload["scope"] = "pooled_out_of_fold_predictions_for_one_training_seed"
    return payload


def _classwise_reliability(n: int, *, include_n: bool) -> dict:
    payload = {
        "bin_count": 15,
        "classwise_ece": 0.0,
        "classwise_ece_by_class": [0.0] * 5,
        "classes": [
            {
                "class": grade,
                "event": f"Y={grade}",
                "prevalence": 0.0,
                "ece": 0.0,
                "bins": _reliability_bins(n),
            }
            for grade in range(5)
        ],
    }
    if include_n:
        payload["n"] = n
        payload["scope"] = "pooled_out_of_fold_predictions_for_one_training_seed"
    return payload


def _baseline(root: Path, commit: str) -> Path:
    experiment = root / "baselines"
    release = experiment / "full" / "outer_release"
    protocol_sha = "a" * 64
    summary_metrics = {name: 0.0 for name in BASELINE_SUMMARY_METRICS}
    summary = {
        dataset: {
            variant: {
                "seed_cv_means": {
                    str(seed): dict(summary_metrics) for seed in (42, 31415, 27182)
                },
                "replication_summary": {
                    name: {
                        "mean_of_seed_cv_means": 0.0,
                        "sd_across_seed_cv_means": 0.0,
                        "values": [0.0, 0.0, 0.0],
                    }
                    for name in BASELINE_SUMMARY_METRICS
                },
                "threshold_reliability_by_seed": {
                    str(seed): _threshold_reliability(
                        3662 if dataset == "aptos" else 35126, include_n=True
                    )
                    for seed in (42, 31415, 27182)
                },
                "classwise_reliability_by_seed": {
                    str(seed): _classwise_reliability(
                        3662 if dataset == "aptos" else 35126, include_n=True
                    )
                    for seed in (42, 31415, 27182)
                },
            }
            for variant in EXPECTED_SHORTCUT_VARIANTS
        }
        for dataset in ("aptos", "dr")
    }
    bootstrap = {
        dataset: {
            comparator: {
                "reference_variant": "origin_ctmc",
                "comparator": comparator,
                "dataset": dataset,
                "comparisons": {
                    name: {
                        "origin_minus_comparator": 0.0,
                        "bootstrap_mean_delta": 0.0,
                        "ci95_percentile": [0.0, 0.0],
                        "higher_is_better": name in {
                            "acc", "qwk", "balanced_acc", "macro_f1"
                        } or name.startswith("per_grade_recall_"),
                    }
                    for name in BASELINE_BOOTSTRAP_METRICS
                },
            }
            for comparator in EXPECTED_SHORTCUT_VARIANTS
            if comparator != "origin_ctmc"
        }
        for dataset in ("aptos", "dr")
    }
    frozen_workers = []
    for dataset, folds in (("aptos", 5), ("dr", 10)):
        for fold in range(folds):
            for seed in (42, 31415, 27182):
                for variant in EXPECTED_SHORTCUT_VARIANTS:
                    key = f"{dataset}__fold{fold}__{variant}__seed{seed}"
                    task = {
                        "dataset": dataset,
                        "fold": fold,
                        "baseline_variant": variant,
                        "training_seed": seed,
                    }
                    fold_dir = experiment / "full" / "training" / key / f"fold{fold}"
                    checkpoint = fold_dir / "best_learned.pth"
                    checkpoint.parent.mkdir(parents=True, exist_ok=True)
                    checkpoint.write_bytes(key.encode())
                    frozen_workers.append({
                        "task": task,
                        "fold_dir": str(fold_dir),
                        "best_learned_sha256": file_sha256(checkpoint),
                        "split_signature": hashlib.sha256(
                            f"{dataset}-{fold}".encode()
                        ).hexdigest(),
                    })

                    worker = release / key
                    identifier = hashlib.sha256(key.encode()).hexdigest()
                    predictions = worker / "outer_predictions.npz"
                    predictions.parent.mkdir(parents=True, exist_ok=True)
                    np.savez_compressed(
                        predictions,
                        sample_index=np.asarray([0]),
                        image_id=np.asarray([identifier]),
                        cluster_id=np.asarray([identifier]),
                        label=np.asarray([0]),
                        prediction=np.asarray([0]),
                        expected_grade=np.asarray([0.0]),
                        class_probs=np.asarray([[1.0, 0.0, 0.0, 0.0, 0.0]]),
                        cumulative_probs=np.asarray([[0.0, 0.0, 0.0, 0.0]]),
                    )
                    scalar_metrics = {
                        name: 0.0
                        for name in (
                            "acc", "qwk", "mae", "expected_grade_mae",
                            "balanced_acc", "macro_f1", "ece", "nll", "rps",
                            "multiclass_brier", "threshold_ece",
                            "threshold_binary_brier", "classwise_ece",
                        )
                    }
                    scalar_metrics.update({
                        "n": 1,
                        "threshold_ece_by_boundary": [0.0] * 4,
                        "threshold_binary_brier_by_boundary": [0.0] * 4,
                        "classwise_ece_by_class": [0.0] * 5,
                        "per_grade_recall": [0.0] * 5,
                    })
                    metrics = _write_sealed(
                        worker / "outer_metrics.json",
                        {
                            "schema": "origin-acceptance-outer-result-v2",
                            "protocol_sha256": protocol_sha,
                            "task": task,
                            "best_learned_checkpoint_sha256": file_sha256(checkpoint),
                            "test_evaluated": True,
                            "selection_reopened": False,
                            "predictions_sha256": file_sha256(predictions),
                            "identifier_privacy": {
                                "dataset_scope": dataset,
                                "algorithm": "sha256",
                                "raw_image_paths_exported": False,
                                "raw_patient_or_cluster_identifiers_exported": False,
                            },
                            "metrics": scalar_metrics,
                            "posterior_quality": {
                                "threshold_reliability": _threshold_reliability(
                                    1, include_n=False
                                ),
                                "classwise_reliability": _classwise_reliability(
                                    1, include_n=False
                                ),
                                "exact_per_sample_probabilities": (
                                    "outer_predictions.npz:class_probs"
                                ),
                            },
                        },
                    )
                    _write_json(
                        worker / "OUTER_COMPLETE.json",
                        {
                            "schema": "origin-acceptance-outer-complete-v2",
                            "protocol_sha256": protocol_sha,
                            "task": task,
                            "outer_metrics_sha256": file_sha256(metrics),
                            "predictions_sha256": file_sha256(predictions),
                        },
                    )
    aggregate = _write_sealed(
        release / "aggregate.json",
        {
            "schema": "origin-acceptance-outer-aggregate-v2",
            "status": "complete",
            "release_worker_count": 225,
            "protocol_sha256": protocol_sha,
            "comparator_implementation_scope": {"kind": "matched_in_repo_analogues"},
            "posterior_quality_contract": {
                "multiclass_brier": "mean_sum_k_(p_k-onehot_k)^2",
                "threshold_binary_brier": (
                    "per_boundary_mean_(P(Y>k)-1[Y>k])^2_then_unweighted_boundary_mean"
                ),
                "threshold_binary_brier_identity": (
                    "unweighted_boundary_mean_equals_reported_ranked_probability_score"
                ),
                "threshold_ece": (
                    "15_equal_width_bins_per_boundary_then_unweighted_boundary_mean"
                ),
                "classwise_ece": (
                    "15_equal_width_bins_one_vs_rest_per_class_then_unweighted_class_mean"
                ),
                "reliability_scope": (
                    "pooled_out_of_fold_predictions_per_training_seed"
                ),
            },
            "summary": summary,
            "paired_cluster_bootstrap": bootstrap,
        },
    )
    metrics = release / "fold_metrics.csv"
    metrics.write_text("dataset,fold\n")
    _write_json(
        release / "OUTER_RELEASE_COMPLETE.json",
        {
            "schema": "origin-acceptance-outer-release-complete-v2",
            "aggregate_sha256": file_sha256(aggregate),
            "fold_metrics_sha256": file_sha256(metrics),
            "worker_count": 225,
        },
    )
    _write_sealed(
        experiment / "SUBMISSION.json",
        {
            "schema": "origin-acceptance-submission-v1",
            "protocol_sha256": protocol_sha,
            "launch_commit": commit,
        },
    )
    _write_sealed(
        experiment / "full" / "TRAINING_FROZEN.json",
        {
            "schema": "origin-acceptance-training-audit-v1",
            "protocol_sha256": protocol_sha,
            "status": "passed",
            "worker_count": 225,
            "workers": frozen_workers,
        },
    )
    return aggregate


def _install_restart_submission(
    *, experiment: Path, repo: Path, parent_commit: str,
    changed_path: str = "docs/continuation.md",
) -> dict:
    """Create a real two-worktree continuation provenance chain for tests."""

    parent_worktree = repo.parent / f"{repo.name}-parent-worktree"
    subprocess.run(
        ["git", "worktree", "add", "--detach", str(parent_worktree), parent_commit],
        cwd=repo, check=True, capture_output=True,
    )
    submission_path = experiment / "SUBMISSION.json"
    submission = json.loads(submission_path.read_text())
    submission.update({
        "protocol_id": "origin-acceptance-test-protocol-v1",
        "source_repository": str(repo),
        "immutable_worktree": str(parent_worktree),
        "conda_environment": "test",
        "data_roots": {
            "aptos": str(repo.parent / "aptos"),
            "dr": str(repo.parent / "dr"),
        },
        "jobs": {
            "aptos_fold0_canary_array": "100",
            "canary_gate_audit": "101",
            "full_training_array": "102",
            "full_freeze_audit": "103",
            "coordinated_outer_release_array": "104",
            "release_aggregate": "105",
        },
    })
    _write_json(submission_path, _reseal(submission))
    submission = json.loads(submission_path.read_text())

    workers = []
    worker_manifests = {}
    checkpoints = {}
    for index, variant in enumerate(EXPECTED_SHORTCUT_VARIANTS):
        manifest_digest = hashlib.sha256(f"manifest-{variant}".encode()).hexdigest()
        checkpoint_digest = hashlib.sha256(f"checkpoint-{variant}".encode()).hexdigest()
        workers.append({
            "task": {
                "dataset": "aptos",
                "fold": 0,
                "baseline_variant": variant,
                "training_seed": 42,
            },
            "fold_dir": str(experiment / "canary" / "training" / variant),
            "manifest_sha256": manifest_digest,
            "best_learned_sha256": checkpoint_digest,
            "split_signature": hashlib.sha256(b"aptos-fold0").hexdigest(),
            "best_epoch": index + 1,
        })
        worker_manifests[variant] = manifest_digest
        checkpoints[variant] = checkpoint_digest
    gate = _reseal({
        "schema": "origin-acceptance-training-audit-v1",
        "protocol_id": submission["protocol_id"],
        "protocol_sha256": submission["protocol_sha256"],
        "scope": "canary",
        "status": "passed",
        "worker_count": len(workers),
        "workers": workers,
        "outer_test_released": False,
    })
    _write_json(experiment / "canary" / "CANARY_PASSED.json", gate)

    changed = repo / changed_path
    changed.parent.mkdir(parents=True, exist_ok=True)
    changed.write_text("continuation infrastructure repair\n")
    subprocess.run(["git", "add", changed_path], cwd=repo, check=True)
    subprocess.run(
        ["git", "commit", "-m", "continuation repair"], cwd=repo,
        check=True, capture_output=True,
    )
    continuation_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repo, check=True,
        capture_output=True, text=True,
    ).stdout.strip()
    continuation_worktree = repo.parent / f"{repo.name}-continuation-worktree"
    subprocess.run(
        [
            "git", "worktree", "add", "--detach", str(continuation_worktree),
            continuation_commit,
        ],
        cwd=repo, check=True, capture_output=True,
    )

    task_keys = sorted(
        f"aptos__fold0__{variant}__seed42"
        for variant in EXPECTED_SHORTCUT_VARIANTS
    )
    preflight = _reseal({
        "schema": "origin-acceptance-continuation-preflight-v1",
        "protocol_id": submission["protocol_id"],
        "protocol_sha256": submission["protocol_sha256"],
        "experiment_root": str(experiment.resolve()),
        "parent": {
            "submission_path": str(submission_path.resolve()),
            "submission_file_sha256": file_sha256(submission_path),
            "submission_content_checksum_sha256": submission[
                "content_checksum_sha256"
            ],
            "parent_launch_commit": parent_commit,
            "parent_immutable_worktree": str(parent_worktree.resolve()),
            "parent_jobs": dict(submission["jobs"]),
        },
        "reused_canary": {
            "task_count": len(task_keys),
            "task_keys": task_keys,
            "training_audit_content_checksum_sha256": gate[
                "content_checksum_sha256"
            ],
            "valid_gate_already_present": True,
            "worker_manifest_sha256": worker_manifests,
            "best_learned_sha256": checkpoints,
        },
        "downstream_state": "absent",
    })
    restart = _reseal({
        "schema": "origin-acceptance-restart-submission-v1",
        "created_at_utc": "2026-10-01T00:00:00+00:00",
        "protocol_id": submission["protocol_id"],
        "protocol_sha256": submission["protocol_sha256"],
        "experiment_root": str(experiment.resolve()),
        "continuation_reason": "parent audit import path repair",
        "scientific_protocol_changed": False,
        "canary_training_reused": True,
        "preflight": preflight,
        "launch_commit": continuation_commit,
        "immutable_worktree": str(continuation_worktree.resolve()),
        "conda_environment": "test",
        "scheduler_parent_states": {
            "canary_array": "COMPLETED",
            "failed_canary_audit": "FAILED",
        },
        "canary_reaudit_dependency_mode": "verified_completed_no_dependency",
        "jobs": {
            "canary_reaudit": "200",
            "full_training_array": "201",
            "full_freeze_audit": "202",
            "coordinated_outer_release_array": "203",
            "release_aggregate": "204",
        },
    })
    restart_path = _write_json(experiment / "RESTART_SUBMISSION.json", restart)
    return {
        "submission": submission,
        "submission_path": submission_path,
        "restart": restart,
        "restart_path": restart_path,
        "parent_commit": parent_commit,
        "continuation_commit": continuation_commit,
        "parent_worktree": parent_worktree,
        "continuation_worktree": continuation_worktree,
    }


def _write_resealed_restart(path: Path, payload: dict) -> None:
    preflight = payload.get("preflight")
    if isinstance(preflight, dict):
        payload["preflight"] = _reseal(preflight)
    _write_json(path, _reseal(payload))


def _inputs(tmp_path: Path, *, gate_a: str = "pass", gate_b: bool = True) -> dict:
    repo, commit = _git_repo(tmp_path / "repo")
    artifacts = tmp_path / "artifacts"
    return {
        "repo_root": repo,
        "expected_commit": commit,
        "output_dir": tmp_path / "package",
        "numerical_audit": _numeric(artifacts),
        "decoder_contract_audit": _decoder(artifacts),
        "artifact_manifest": _artifact_manifest(artifacts, repo, commit),
        "idrid_manifest": _idrid(artifacts, repo),
        "oof_manifest": _oof(artifacts, repo, commit, gate=gate_a),
        "shortcut_aggregate": _shortcut(artifacts, commit, passed=gate_b),
        "baseline_aggregate": _baseline(artifacts, commit),
    }


def test_complete_package_is_checksum_sealed_and_exports_no_locations_or_ids(
    tmp_path: Path,
) -> None:
    inputs = _inputs(tmp_path)
    payload = assemble_package(**inputs, require_gates_pass=True)
    assert payload["scientific_gates"]["all_passed"] is True
    manifest = inputs["output_dir"] / MANIFEST_FILENAME
    completion = inputs["output_dir"] / COMPLETION_FILENAME
    assert manifest.is_file() and completion.is_file()
    exported = manifest.read_text() + completion.read_text()
    assert str(tmp_path) not in exported
    assert "private-0" not in exported
    assert "licensed_pixels_copied\": false" in exported
    assert payload["artifact_availability"]["status"] == (
        "incomplete_external_archives"
    )
    assert payload["artifact_availability"][
        "independent_reproduction_claim_authorized"
    ] is False
    bundle = inputs["output_dir"] / "bundle"
    assert (bundle / "BUNDLE_INDEX.json").is_file()
    assert len(list(bundle.glob("matched_baselines/workers/*/outer_predictions.npz"))) == 225
    assert len(list(bundle.glob("memberships/*.csv.gz"))) == 15
    for json_file in bundle.rglob("*.json"):
        text = json_file.read_text()
        assert str(tmp_path) not in text
        assert "private-0" not in text
        _assert_privacy_safe_release(json.loads(text))
    _assert_privacy_safe_release(json.loads(manifest.read_text()))


def test_stable_checkpoint_archive_authorizes_independent_reproduction(
    tmp_path: Path,
) -> None:
    inputs = _inputs(tmp_path)
    payload = assemble_package(
        **inputs,
        checkpoint_archive_uri="https://example.invalid/origin-checkpoints.tar.zst",
        checkpoint_archive_sha256="b" * 64,
        checkpoint_archive_bytes=12345,
        evidence_archive_uri="https://example.invalid/origin-evidence.tar.zst",
        evidence_archive_sha256="c" * 64,
        evidence_archive_bytes=67890,
    )
    availability = payload["artifact_availability"]
    assert availability["status"] == "complete_with_external_archives"
    assert availability["independent_reproduction_claim_authorized"] is True
    completion = json.loads(
        (inputs["output_dir"] / COMPLETION_FILENAME).read_text()
    )
    assert completion["independent_reproduction_claim_authorized"] is True


def test_scientific_failure_is_recorded_but_submission_ready_mode_refuses(
    tmp_path: Path,
) -> None:
    inputs = _inputs(tmp_path, gate_a="fail", gate_b=True)
    with pytest.raises(RuntimeError, match="Gate A=fail"):
        assemble_package(**inputs, require_gates_pass=True)
    assert not inputs["output_dir"].exists()
    payload = assemble_package(**inputs, require_gates_pass=False)
    assert payload["status"] == "verified_complete"
    assert payload["scientific_gates"]["claims_authorized"] is False


def test_failed_gate_b_is_preserved_and_withholds_claim(tmp_path: Path) -> None:
    inputs = _inputs(tmp_path, gate_a="pass", gate_b=False)
    payload = assemble_package(**inputs, require_gates_pass=False)
    assert payload["status"] == "verified_complete"
    assert payload["scientific_gates"]["gate_b"] == "fail"
    assert payload["scientific_gates"]["claims_authorized"] is False
    assert payload["claim_authorized"] is False


def test_missing_gate_artifact_prevents_completion(tmp_path: Path) -> None:
    inputs = _inputs(tmp_path)
    oof = json.loads(inputs["oof_manifest"].read_text())
    gate_path = Path(oof["artifacts"]["gate_a"]["path"])
    gate_path.unlink()
    with pytest.raises(FileNotFoundError):
        assemble_package(**inputs, require_gates_pass=False)
    assert not inputs["output_dir"].exists()


def test_tampered_component_is_rejected_before_any_output(tmp_path: Path) -> None:
    inputs = _inputs(tmp_path)
    with inputs["numerical_audit"].open("a", encoding="utf-8") as stream:
        stream.write(" \n")
    # Whitespace does not affect the content checksum, so mutate a value.
    payload = json.loads(inputs["numerical_audit"].read_text())
    payload["n_forward_cases"] = 999
    _write_json(inputs["numerical_audit"], payload)
    with pytest.raises(ValueError, match="checksum mismatch"):
        assemble_package(**inputs)
    assert not inputs["output_dir"].exists()


def test_dirty_assembly_worktree_is_rejected(tmp_path: Path) -> None:
    inputs = _inputs(tmp_path)
    tracked = inputs["repo_root"] / "tracked.txt"
    tracked.write_text("clean\n")
    subprocess.run(["git", "add", "tracked.txt"], cwd=inputs["repo_root"], check=True)
    subprocess.run(
        ["git", "commit", "-m", "tracked"], cwd=inputs["repo_root"], check=True
    )
    tracked.write_text("dirty\n")
    inputs["expected_commit"] = None
    with pytest.raises(RuntimeError, match="clean tracked worktree"):
        assemble_package(**inputs)


def test_numerical_validator_rejects_a_failed_audit(tmp_path: Path) -> None:
    path = _numeric(tmp_path)
    payload = json.loads(path.read_text())
    payload["summary"]["passed"] = False
    payload.pop("content_checksum_sha256")
    _write_sealed(path, payload)
    with pytest.raises(ValueError, match="did not pass"):
        validate_numerical(path)


def test_baseline_v1_contract_is_rejected(tmp_path: Path) -> None:
    inputs = _inputs(tmp_path)
    path = inputs["baseline_aggregate"]
    payload = json.loads(path.read_text())
    payload["schema"] = "origin-acceptance-outer-aggregate-v1"
    payload.pop("content_checksum_sha256")
    _write_sealed(path, payload)
    with pytest.raises(ValueError, match="outer-aggregate-v2"):
        validate_baseline_aggregate(path, inputs["repo_root"])


def test_missing_posterior_reliability_prevents_completion(tmp_path: Path) -> None:
    inputs = _inputs(tmp_path)
    path = inputs["baseline_aggregate"]
    payload = json.loads(path.read_text())
    del payload["summary"]["dr"]["origin_ctmc"][
        "classwise_reliability_by_seed"
    ]
    payload.pop("content_checksum_sha256")
    _write_sealed(path, payload)
    with pytest.raises(ValueError, match="classwise reliability"):
        assemble_package(**inputs)
    assert not inputs["output_dir"].exists()


def test_baseline_prediction_checksum_mismatch_prevents_completion(
    tmp_path: Path,
) -> None:
    inputs = _inputs(tmp_path)
    worker = inputs["baseline_aggregate"].parent / (
        "aptos__fold0__origin_ctmc__seed42"
    )
    with (worker / "outer_predictions.npz").open("ab") as stream:
        stream.write(b"tampered")
    with pytest.raises(ValueError, match="predictions SHA-256 mismatch"):
        assemble_package(**inputs)
    assert not inputs["output_dir"].exists()


def test_continued_baseline_records_both_commits_worktrees_and_job_chains(
    tmp_path: Path,
) -> None:
    inputs = _inputs(tmp_path)
    chain = _install_restart_submission(
        experiment=inputs["baseline_aggregate"].parents[2],
        repo=inputs["repo_root"],
        parent_commit=inputs["expected_commit"],
    )
    record = validate_baseline_aggregate(
        inputs["baseline_aggregate"], inputs["repo_root"]
    )
    provenance = record["execution_provenance"]
    assert record["source_commit"] == chain["continuation_commit"]
    assert record["parent_canary_source_commit"] == chain["parent_commit"]
    assert record["continuation_source_commit"] == chain["continuation_commit"]
    assert provenance["mode"] == "infrastructure_only_continuation"
    assert provenance["parent_canary_worktree"]["recorded_worktree_verified"] is True
    assert provenance["continuation_worktree"]["recorded_worktree_verified"] is True
    assert provenance["infrastructure_only_commit_delta"]["verified"] is True
    assert provenance["infrastructure_only_commit_delta"]["changed_files"] == [
        "docs/continuation.md"
    ]
    assert provenance["parent_job_chain"]["canary_gate_audit"] == "101"
    assert provenance["continuation_job_chain"]["canary_reaudit"] == "200"
    inputs["expected_commit"] = chain["continuation_commit"]
    package = assemble_package(**inputs)
    packaged_provenance = package["components"][
        "matched_baseline_outer_release"
    ]["execution_provenance"]
    assert packaged_provenance["parent_canary_source_commit"] == chain[
        "parent_commit"
    ]
    sanitized = inputs["output_dir"] / "bundle" / "matched_baselines" / (
        "RESTART_SUBMISSION_sanitized.json"
    )
    assert sanitized.is_file()
    assert str(tmp_path) not in sanitized.read_text()


def test_continuation_restart_checksum_tampering_is_rejected(tmp_path: Path) -> None:
    inputs = _inputs(tmp_path)
    chain = _install_restart_submission(
        experiment=inputs["baseline_aggregate"].parents[2],
        repo=inputs["repo_root"], parent_commit=inputs["expected_commit"],
    )
    restart = dict(chain["restart"])
    restart["scientific_protocol_changed"] = True
    _write_json(chain["restart_path"], restart)
    with pytest.raises(ValueError, match="canonical content checksum mismatch"):
        _validate_baseline_execution_provenance(
            experiment_root=inputs["baseline_aggregate"].parents[2],
            submission=chain["submission"],
            aggregate_protocol_sha256=chain["submission"]["protocol_sha256"],
            repo_root=inputs["repo_root"],
        )


def test_resealed_parent_submission_digest_tampering_is_rejected(
    tmp_path: Path,
) -> None:
    inputs = _inputs(tmp_path)
    chain = _install_restart_submission(
        experiment=inputs["baseline_aggregate"].parents[2],
        repo=inputs["repo_root"], parent_commit=inputs["expected_commit"],
    )
    restart = dict(chain["restart"])
    preflight = dict(restart["preflight"])
    parent = dict(preflight["parent"])
    parent["submission_file_sha256"] = "f" * 64
    preflight["parent"] = parent
    restart["preflight"] = preflight
    _write_resealed_restart(chain["restart_path"], restart)
    with pytest.raises(ValueError, match="parent submission file digest mismatch"):
        _validate_baseline_execution_provenance(
            experiment_root=inputs["baseline_aggregate"].parents[2],
            submission=chain["submission"],
            aggregate_protocol_sha256=chain["submission"]["protocol_sha256"],
            repo_root=inputs["repo_root"],
        )


def test_resealed_parent_job_chain_tampering_is_rejected(tmp_path: Path) -> None:
    inputs = _inputs(tmp_path)
    chain = _install_restart_submission(
        experiment=inputs["baseline_aggregate"].parents[2],
        repo=inputs["repo_root"], parent_commit=inputs["expected_commit"],
    )
    restart = dict(chain["restart"])
    preflight = dict(restart["preflight"])
    parent = dict(preflight["parent"])
    jobs = dict(parent["parent_jobs"])
    jobs["release_aggregate"] = "999"
    parent["parent_jobs"] = jobs
    preflight["parent"] = parent
    restart["preflight"] = preflight
    _write_resealed_restart(chain["restart_path"], restart)
    with pytest.raises(ValueError, match="parent job chain mismatch"):
        _validate_baseline_execution_provenance(
            experiment_root=inputs["baseline_aggregate"].parents[2],
            submission=chain["submission"],
            aggregate_protocol_sha256=chain["submission"]["protocol_sha256"],
            repo_root=inputs["repo_root"],
        )


def test_resealed_continuation_protocol_tampering_is_rejected(
    tmp_path: Path,
) -> None:
    inputs = _inputs(tmp_path)
    chain = _install_restart_submission(
        experiment=inputs["baseline_aggregate"].parents[2],
        repo=inputs["repo_root"], parent_commit=inputs["expected_commit"],
    )
    restart = dict(chain["restart"])
    restart["protocol_sha256"] = "b" * 64
    _write_resealed_restart(chain["restart_path"], restart)
    with pytest.raises(ValueError, match="restart protocol digest mismatch"):
        _validate_baseline_execution_provenance(
            experiment_root=inputs["baseline_aggregate"].parents[2],
            submission=chain["submission"],
            aggregate_protocol_sha256=chain["submission"]["protocol_sha256"],
            repo_root=inputs["repo_root"],
        )


def test_continuation_with_scientific_source_change_is_rejected(
    tmp_path: Path,
) -> None:
    inputs = _inputs(tmp_path)
    chain = _install_restart_submission(
        experiment=inputs["baseline_aggregate"].parents[2],
        repo=inputs["repo_root"], parent_commit=inputs["expected_commit"],
        changed_path="models/origin.py",
    )
    with pytest.raises(ValueError, match="changes non-infrastructure files"):
        _validate_baseline_execution_provenance(
            experiment_root=inputs["baseline_aggregate"].parents[2],
            submission=chain["submission"],
            aggregate_protocol_sha256=chain["submission"]["protocol_sha256"],
            repo_root=inputs["repo_root"],
        )


def test_continuation_wrapper_scientific_change_is_rejected(tmp_path: Path) -> None:
    inputs = _inputs(tmp_path)
    chain = _install_restart_submission(
        experiment=inputs["baseline_aggregate"].parents[2],
        repo=inputs["repo_root"], parent_commit=inputs["expected_commit"],
        changed_path="scripts/submit_origin_acceptance_full_array.sh",
    )
    with pytest.raises(ValueError, match="changes non-infrastructure files"):
        _validate_baseline_execution_provenance(
            experiment_root=inputs["baseline_aggregate"].parents[2],
            submission=chain["submission"],
            aggregate_protocol_sha256=chain["submission"]["protocol_sha256"],
            repo_root=inputs["repo_root"],
        )


def test_dirty_continuation_worktree_is_rejected(tmp_path: Path) -> None:
    inputs = _inputs(tmp_path)
    chain = _install_restart_submission(
        experiment=inputs["baseline_aggregate"].parents[2],
        repo=inputs["repo_root"], parent_commit=inputs["expected_commit"],
    )
    (chain["continuation_worktree"] / "docs" / "continuation.md").write_text(
        "dirty continuation\n"
    )
    with pytest.raises(ValueError, match="tracked modifications"):
        _validate_baseline_execution_provenance(
            experiment_root=inputs["baseline_aggregate"].parents[2],
            submission=chain["submission"],
            aggregate_protocol_sha256=chain["submission"]["protocol_sha256"],
            repo_root=inputs["repo_root"],
        )
