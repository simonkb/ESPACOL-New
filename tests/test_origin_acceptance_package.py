from __future__ import annotations

import csv
import gzip
import hashlib
import io
import json
from pathlib import Path
import subprocess

import pytest

from tools.assemble_origin_acceptance_package import (
    COMPLETION_FILENAME,
    MANIFEST_FILENAME,
    EXPECTED_SHORTCUT_VARIANTS,
    _assert_privacy_safe_release,
    assemble_package,
    canonical_sha256,
    file_sha256,
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
    return _write_sealed(
        suite / "COMPARATOR_SHORTCUT_RESULTS_V2.json",
        {
            "schema": "origin-ordinal-shortcut-pilot-aggregate-v2",
            "expected_training_seeds": [1701, 2603, 3907],
            "model_variants": {
                variant: {
                    "official_author_implementation": False,
                    "families": {name: {} for name in ("localized", "border", "diffuse")},
                }
                for variant in EXPECTED_SHORTCUT_VARIANTS
            },
            "audit_files": audits,
            "localized_family_gates_passed": [passed, passed],
            "passed": passed,
        },
    )


def _baseline(root: Path, commit: str) -> Path:
    experiment = root / "baselines"
    release = experiment / "full" / "outer_release"
    summary = {
        dataset: {
            variant: {"seed_cv_means": {str(seed): {} for seed in (42, 31415, 27182)}}
            for variant in EXPECTED_SHORTCUT_VARIANTS
        }
        for dataset in ("aptos", "dr")
    }
    aggregate = _write_sealed(
        release / "aggregate.json",
        {
            "schema": "origin-acceptance-outer-aggregate-v1",
            "status": "complete",
            "release_worker_count": 225,
            "protocol_sha256": "a" * 64,
            "comparator_implementation_scope": {"kind": "matched_in_repo_analogues"},
            "summary": summary,
        },
    )
    metrics = release / "fold_metrics.csv"
    metrics.write_text("dataset,fold\n")
    _write_json(
        release / "OUTER_RELEASE_COMPLETE.json",
        {
            "schema": "origin-acceptance-outer-release-complete-v1",
            "aggregate_sha256": file_sha256(aggregate),
            "fold_metrics_sha256": file_sha256(metrics),
            "worker_count": 225,
        },
    )
    _write_sealed(
        experiment / "SUBMISSION.json",
        {
            "schema": "origin-acceptance-submission-v1",
            "protocol_sha256": "a" * 64,
            "launch_commit": commit,
        },
    )
    _write_sealed(
        experiment / "full" / "TRAINING_FROZEN.json",
        {
            "schema": "origin-acceptance-training-audit-v1",
            "protocol_sha256": "a" * 64,
            "status": "passed",
            "worker_count": 225,
        },
    )
    return aggregate


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
    _assert_privacy_safe_release(json.loads(manifest.read_text()))


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
