#!/usr/bin/env python3
"""Assemble a fail-closed, privacy-safe ORIGIN acceptance manifest.

The assembler does not copy source images, checkpoints, per-image predictions,
patient identifiers, or filesystem locations.  It validates the authoritative
artifacts in place and exports only schemas, cryptographic digests, protocol
identifiers, aggregate census values, and the two preregistered gate outcomes.

Scientific failure is not artifact corruption: a complete package may record a
failed Gate A or Gate B, but its claims are then explicitly withheld.  Pass
``--require-gates-pass`` for a submission-readiness job that refuses to create
a completion marker unless both gates pass.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
from typing import Any, Iterable, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_SCHEMA = "origin-acceptance-reproducibility-package-v1"
COMPLETION_SCHEMA = "origin-acceptance-reproducibility-complete-v1"
MANIFEST_FILENAME = "ORIGIN_ACCEPTANCE_PACKAGE_MANIFEST.json"
COMPLETION_FILENAME = "ORIGIN_ACCEPTANCE_PACKAGE_COMPLETE.json"

EXPECTED_SHORTCUT_VARIANTS = (
    "ledger_sequential_hazard",
    "pooled_conditional",
    "ordinal_additive_mil",
    "sparse_bagnet",
    "origin_ctmc",
)
EXPECTED_DATASET_IMAGES = {"aptos": 3662, "dr": 35126}
HEX64 = re.compile(r"^[0-9a-f]{64}$")
HEX40 = re.compile(r"^[0-9a-f]{40}$")


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(encoding="utf-8") as stream:
        payload = json.load(stream)
    if not isinstance(payload, dict):
        raise TypeError(f"expected a JSON object: {path}")
    return payload


def verify_checksummed_payload(
    path: str | Path, *, schema: str | Sequence[str]
) -> dict[str, Any]:
    path = Path(path)
    payload = read_json(path)
    schemas = {schema} if isinstance(schema, str) else set(schema)
    if payload.get("schema") not in schemas:
        raise ValueError(
            f"unexpected schema {payload.get('schema')!r} in {path.name}; "
            f"expected {sorted(schemas)}"
        )
    recorded = payload.get("content_checksum_sha256")
    if not isinstance(recorded, str) or not HEX64.fullmatch(recorded):
        raise ValueError(f"missing canonical content checksum: {path}")
    unsigned = dict(payload)
    unsigned.pop("content_checksum_sha256", None)
    if canonical_sha256(unsigned) != recorded:
        raise ValueError(f"canonical content checksum mismatch: {path}")
    return payload


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    temporary.replace(path)


def _git(repo_root: Path, *arguments: str) -> str:
    return subprocess.run(
        ("git", "-C", str(repo_root), *arguments), check=True,
        capture_output=True, text=True,
    ).stdout.strip()


def _verify_commit(repo_root: Path, commit: Any, *, field: str) -> str:
    value = str(commit)
    if not HEX40.fullmatch(value):
        raise ValueError(f"{field} is not a full Git commit")
    try:
        subprocess.run(
            ("git", "-C", str(repo_root), "cat-file", "-e", f"{value}^{{commit}}"),
            check=True, capture_output=True,
        )
    except subprocess.CalledProcessError as error:
        raise ValueError(f"{field} is not present in the repository: {value}") from error
    return value


def runtime_provenance(repo_root: Path, expected_commit: str | None) -> dict[str, Any]:
    commit = _git(repo_root, "rev-parse", "HEAD")
    if expected_commit is not None and commit != expected_commit:
        raise RuntimeError(
            f"assembly commit {commit} differs from expected {expected_commit}"
        )
    dirty = _git(repo_root, "status", "--porcelain", "--untracked-files=no")
    if dirty:
        raise RuntimeError("acceptance assembly requires a clean tracked worktree")
    _verify_commit(repo_root, commit, field="assembly commit")
    return {
        "git_commit": commit,
        "tracked_worktree_clean": True,
        "assembler_sha256": file_sha256(Path(__file__).resolve()),
    }


def _resolve_declared_file(
    declaration: Any, *, relative_to: Path | None = None,
    allowed_roots: Iterable[Path] | None = None,
) -> Path:
    path = Path(str(declaration))
    if not path.is_absolute():
        if relative_to is None:
            raise ValueError(f"relative artifact has no declared base: {path}")
        path = relative_to / path
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    if allowed_roots is not None:
        roots = tuple(root.resolve() for root in allowed_roots)
        if not any(path.is_relative_to(root) for root in roots):
            raise ValueError(f"artifact escapes its declared output root: {path.name}")
    return path


def _verify_declared_hash(path: Path, expected: Any, *, label: str) -> str:
    expected = str(expected)
    if not HEX64.fullmatch(expected):
        raise ValueError(f"{label} has no valid SHA-256 digest")
    observed = file_sha256(path)
    if observed != expected:
        raise ValueError(f"{label} SHA-256 mismatch")
    return observed


def _verify_json_artifact(
    manifest_path: Path, record: Mapping[str, Any], *, allowed_root: Path,
) -> dict[str, Any]:
    artifact_path = _resolve_declared_file(
        record.get("path"), relative_to=manifest_path.parent,
        allowed_roots=(allowed_root,),
    )
    _verify_declared_hash(artifact_path, record.get("sha256"), label=artifact_path.name)
    payload = read_json(artifact_path)
    if "content_checksum_sha256" in record:
        recorded = str(record["content_checksum_sha256"])
        if payload.get("content_checksum_sha256") != recorded:
            raise ValueError(f"nested content checksum mismatch: {artifact_path.name}")
        unsigned = dict(payload)
        unsigned.pop("content_checksum_sha256", None)
        if canonical_sha256(unsigned) != recorded:
            raise ValueError(f"nested canonical checksum mismatch: {artifact_path.name}")
    return payload


def validate_numerical(path: Path) -> dict[str, Any]:
    payload = verify_checksummed_payload(
        path, schema="origin-decoder-independent-numerical-audit-v1"
    )
    summary = payload.get("summary")
    if not isinstance(summary, Mapping) or summary.get("passed") is not True:
        raise ValueError("independent decoder numerical audit did not pass")
    if payload.get("independent_reference") != "mpmath.expm":
        raise ValueError("decoder audit is not independent of the deployed exponential")
    if payload.get("rate_domain") != [0.0, 64.0]:
        raise ValueError("decoder audit did not cover the registered rate domain [0,64]")
    if int(payload.get("n_forward_cases", 0)) < 128:
        raise ValueError("decoder numerical audit has fewer than 128 random cases")
    if int(payload.get("n_gradient_cases", 0)) < 12:
        raise ValueError("decoder numerical audit has fewer than 12 gradient cases")
    return {
        "schema": payload["schema"],
        "file_sha256": file_sha256(path),
        "content_checksum_sha256": payload["content_checksum_sha256"],
        "validation_implementation_sha256": file_sha256(
            REPO_ROOT / "tools" / "audit_origin_decoder_numeric.py"
        ),
        "passed": True,
        "n_forward_cases": int(payload["n_forward_cases"]),
        "n_gradient_cases": int(payload["n_gradient_cases"]),
        "rate_cap": 64.0,
    }


def validate_decoder_contract(path: Path) -> dict[str, Any]:
    payload = read_json(path)
    if payload.get("schema") != "origin-decoder-contract-audit-v1":
        raise ValueError("unexpected decoder-contract schema")
    if payload.get("passed") is not True or int(payload.get("trials", 0)) < 64:
        raise ValueError("decoder-contract audit did not pass its registered trials")
    if float(payload.get("maximum_ctmc_semigroup_error", 1.0)) > 1e-12:
        raise ValueError("CTMC semigroup contract exceeds tolerance")
    if float(payload.get("minimum_sequential_semigroup_failure", 0.0)) < 1e-6:
        raise ValueError("sequential-hazard negative control did not separate")
    return {
        "schema": payload["schema"],
        "file_sha256": file_sha256(path),
        "validation_implementation_sha256": file_sha256(
            REPO_ROOT / "tools" / "audit_origin_decoder_contracts.py"
        ),
        "passed": True,
        "trials": int(payload["trials"]),
        "contract": "ledger_replay_shared_ctmc_semigroup_decoder_specific",
    }


def _verify_anonymous_membership(path: Path, expected_rows: int) -> None:
    with gzip.open(path, "rt", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        expected_fields = [
            "image_id_sha256", "patient_cluster_sha256", "label", "split"
        ]
        if reader.fieldnames != expected_fields:
            raise ValueError(f"anonymous membership columns changed: {path.name}")
        rows = 0
        for row in reader:
            rows += 1
            if not HEX64.fullmatch(str(row["image_id_sha256"])):
                raise ValueError("membership contains a raw or invalid image identifier")
            if not HEX64.fullmatch(str(row["patient_cluster_sha256"])):
                raise ValueError("membership contains a raw or invalid cluster identifier")
            if row["split"] not in {"train", "validation", "outer_test"}:
                raise ValueError("membership contains an unknown split")
            if int(row["label"]) not in range(5):
                raise ValueError("membership contains an invalid label")
    if rows != expected_rows:
        raise ValueError(f"membership row census changed: {path.name}")


def validate_artifact_manifest(path: Path, repo_root: Path) -> dict[str, Any]:
    payload = verify_checksummed_payload(
        path, schema="origin-acceptance-artifact-manifest-v1"
    )
    if payload.get("raw_patient_data_redistributed") is not False:
        raise ValueError("artifact manifest permits raw patient-data redistribution")
    source = payload.get("source_provenance")
    if not isinstance(source, Mapping) or source.get("tracked_tree_dirty") is not False:
        raise ValueError("artifact audit was not run from a clean source tree")
    audit_commit = _verify_commit(
        repo_root, source.get("audit_source_commit"), field="artifact audit source commit"
    )
    submission = payload.get("full_cv_submission")
    if not isinstance(submission, Mapping):
        raise ValueError("full-CV launch provenance is missing")
    training_commit = _verify_commit(
        repo_root, submission.get("training_launch_commit"),
        field="full-CV training launch commit",
    )
    cv_root = Path(str(payload.get("cv_root", ""))).resolve()
    submission_path = _resolve_declared_file(
        submission.get("relative_path"), relative_to=cv_root, allowed_roots=(cv_root,)
    )
    _verify_declared_hash(
        submission_path, submission.get("sha256"), label="full-CV submission"
    )

    checkpoints = payload.get("checkpoints")
    memberships = payload.get("split_memberships")
    if not isinstance(checkpoints, list) or len(checkpoints) != 15:
        raise ValueError("artifact manifest must contain all 15 selected checkpoints")
    if not isinstance(memberships, list) or len(memberships) != 15:
        raise ValueError("artifact manifest must contain all 15 split memberships")
    expected_folds = {"aptos": set(range(5)), "dr": set(range(10))}
    seen_checkpoints = {name: set() for name in expected_folds}
    seen_memberships = {name: set() for name in expected_folds}
    for record in checkpoints:
        dataset, fold = str(record.get("dataset")), int(record.get("fold", -1))
        if dataset not in expected_folds or fold not in expected_folds[dataset]:
            raise ValueError("checkpoint identifies an unexpected dataset/fold")
        checkpoint = _resolve_declared_file(
            record.get("relative_path"), relative_to=cv_root, allowed_roots=(cv_root,)
        )
        _verify_declared_hash(checkpoint, record.get("sha256"), label="selected checkpoint")
        if checkpoint.stat().st_size != int(record.get("bytes", -1)):
            raise ValueError("selected checkpoint size mismatch")
        seen_checkpoints[dataset].add(fold)
    for record in memberships:
        dataset, fold = str(record.get("dataset")), int(record.get("fold", -1))
        if dataset not in expected_folds or fold not in expected_folds[dataset]:
            raise ValueError("membership identifies an unexpected dataset/fold")
        membership = _resolve_declared_file(
            record.get("relative_path"), relative_to=path.parent,
            allowed_roots=(path.parent,),
        )
        _verify_declared_hash(membership, record.get("sha256"), label="split membership")
        _verify_anonymous_membership(membership, int(record.get("rows", -1)))
        seen_memberships[dataset].add(fold)
    if seen_checkpoints != expected_folds or seen_memberships != expected_folds:
        raise ValueError("artifact manifest fold coverage is incomplete")
    return {
        "schema": payload["schema"],
        "file_sha256": file_sha256(path),
        "content_checksum_sha256": payload["content_checksum_sha256"],
        "audit_source_commit": audit_commit,
        "training_launch_commit": training_commit,
        "checkpoints": 15,
        "anonymous_split_memberships": 15,
        "raw_patient_data_redistributed": False,
    }


def validate_idrid_manifest(path: Path, repo_root: Path) -> dict[str, Any]:
    payload = verify_checksummed_payload(
        path, schema="origin-idrid-semantic-statistics-manifest-v1"
    )
    implementation = repo_root / "tools" / "analyze_origin_idrid_semantics.py"
    if file_sha256(implementation) != payload.get("analysis_implementation_sha256"):
        raise ValueError("IDRiD analysis implementation differs from the sealed run")
    protocol_path = repo_root / "scripts" / "protocols" / "origin_idrid_semantic_statistics_protocol.json"
    protocol = verify_checksummed_payload(
        protocol_path, schema="origin-idrid-semantic-statistics-protocol-v1"
    )
    if protocol["content_checksum_sha256"] != payload.get("protocol_checksum_sha256"):
        raise ValueError("IDRiD statistics protocol checksum mismatch")
    census = payload.get("input_census")
    if not isinstance(census, Mapping) or len(census.get("images", [])) != 81:
        raise ValueError("IDRiD audit does not cover exactly 81 image clusters")

    # The raw summary/records are validated in place but never copied.
    semantic_root = path.parent.parent.resolve()
    for key, hash_key in (
        ("input_summary_path", "input_summary_sha256"),
        ("input_records_path", "input_records_sha256"),
    ):
        source = _resolve_declared_file(
            payload.get(key), allowed_roots=(semantic_root,)
        )
        _verify_declared_hash(source, payload.get(hash_key), label=key)
    artifacts = payload.get("artifacts")
    required = {"alignment_statistics", "deletion_statistics", "image_level_units"}
    if not isinstance(artifacts, Mapping) or set(artifacts) != required:
        raise ValueError("IDRiD statistics artifact set is incomplete")
    for name, record in artifacts.items():
        artifact = _resolve_declared_file(
            record.get("path"), relative_to=path.parent, allowed_roots=(path.parent,)
        )
        _verify_declared_hash(artifact, record.get("sha256"), label=name)
        if name != "image_level_units":
            nested = read_json(artifact)
            recorded = record.get("content_checksum_sha256")
            if nested.get("content_checksum_sha256") != recorded:
                raise ValueError(f"IDRiD nested checksum mismatch: {name}")
            unsigned = dict(nested)
            unsigned.pop("content_checksum_sha256", None)
            if canonical_sha256(unsigned) != recorded:
                raise ValueError(f"IDRiD nested canonical checksum mismatch: {name}")
        else:
            with artifact.open(encoding="utf-8") as stream:
                rows = sum(1 for line in stream if line.strip())
            if rows != int(record.get("rows", -1)):
                raise ValueError("IDRiD image-unit row census mismatch")
    return {
        "schema": payload["schema"],
        "file_sha256": file_sha256(path),
        "content_checksum_sha256": payload["content_checksum_sha256"],
        "protocol_checksum_sha256": protocol["content_checksum_sha256"],
        "image_clusters": 81,
        "checkpoint_evaluations_per_image": 10,
        "interpretation_scope": "stored_ledger_not_causal_pixel_intervention",
    }


def validate_oof_manifest(path: Path, repo_root: Path) -> tuple[dict[str, Any], str]:
    payload = verify_checksummed_payload(path, schema="origin-oof-statistics-manifest-v2")
    runtime = payload.get("statistics_runtime_source")
    if not isinstance(runtime, Mapping):
        raise ValueError("OOF statistics runtime provenance is missing")
    if runtime.get("git_commit_available") is not True or runtime.get(
        "tracked_worktree_clean"
    ) is not True:
        raise ValueError("OOF statistics were not produced from a clean Git commit")
    source_commit = _verify_commit(
        repo_root, runtime.get("git_commit"), field="OOF statistics source commit"
    )
    implementation = repo_root / "scripts" / "analyze_origin_oof_statistics.py"
    implementation_hash = file_sha256(implementation)
    if implementation_hash != payload.get("analysis_implementation_sha256"):
        raise ValueError("OOF analysis implementation differs from the sealed run")
    if implementation_hash != runtime.get("analysis_implementation_sha256"):
        raise ValueError("OOF runtime implementation hash is internally inconsistent")
    protocol_path = repo_root / "scripts" / "protocols" / "origin_oof_statistics_protocol.json"
    protocol = verify_checksummed_payload(
        protocol_path, schema="origin-oof-statistics-protocol-v2"
    )
    if protocol["content_checksum_sha256"] != payload.get(
        "statistics_protocol_checksum_sha256"
    ):
        raise ValueError("OOF Gate-A protocol checksum mismatch")

    inputs = payload.get("inputs")
    if not isinstance(inputs, list) or {row.get("dataset") for row in inputs} != {
        "aptos", "dr"
    }:
        raise ValueError("OOF statistics require complete APTOS and EyePACS inputs")
    for record in inputs:
        dataset = str(record["dataset"])
        source = _resolve_declared_file(record.get("path"))
        _verify_declared_hash(source, record.get("sha256"), label=f"{dataset} OOF aggregate")
        aggregate = verify_checksummed_payload(
            source, schema="origin-oof-intervention-aggregate-v1"
        )
        if aggregate["content_checksum_sha256"] != record.get(
            "content_checksum_sha256"
        ):
            raise ValueError(f"{dataset} OOF aggregate content checksum mismatch")
        if int(record.get("n_images", -1)) != EXPECTED_DATASET_IMAGES[dataset]:
            raise ValueError(f"{dataset} OOF image census changed")
        _verify_commit(
            repo_root, record.get("model_frozen_base_commit"),
            field=f"{dataset} frozen model commit",
        )

    artifacts = payload.get("artifacts")
    required = {
        "proper_score_calibration", "grading_bootstrap", "intervention_bootstrap",
        "rf_footprint_diagnostics", "gate_a",
    }
    if not isinstance(artifacts, Mapping) or set(artifacts) != required:
        raise ValueError("OOF statistics artifact set is incomplete")
    gate_payload: dict[str, Any] | None = None
    for name, record in artifacts.items():
        nested = _verify_json_artifact(path, record, allowed_root=path.parent)
        if name == "gate_a":
            gate_payload = nested
    if gate_payload is None or gate_payload.get("schema") != "origin-oof-gate-a-adjudication-v1":
        raise ValueError("OOF Gate-A adjudication is missing")
    gate_status = str(gate_payload.get("status"))
    if gate_status not in {"pass", "fail", "insufficient_data"}:
        raise ValueError("OOF Gate-A has an invalid status")
    authorized = gate_payload.get("claim_authorized") is True
    if authorized != (gate_status == "pass"):
        raise ValueError("OOF Gate-A authorization/status mismatch")
    if artifacts["gate_a"].get("gate_status") != gate_status:
        raise ValueError("OOF Gate-A manifest/artifact status mismatch")
    clauses = gate_payload.get("clauses")
    if not isinstance(clauses, Mapping) or set(clauses) != {
        "auc_advantage", "positive_grade_strata", "concentration", "retention_preservation"
    }:
        raise ValueError("OOF Gate-A clause census changed")
    if gate_status == "pass" and (
        gate_payload.get("input_completeness_issues")
        or any(row.get("status") != "pass" for row in clauses.values())
    ):
        raise ValueError("OOF Gate-A pass is not supported by every registered clause")
    return ({
        "schema": payload["schema"],
        "file_sha256": file_sha256(path),
        "content_checksum_sha256": payload["content_checksum_sha256"],
        "source_commit": source_commit,
        "protocol_checksum_sha256": protocol["content_checksum_sha256"],
        "datasets": ["aptos", "dr"],
        "gate_status": gate_status,
        "claim_authorized": authorized,
    }, gate_status)


def _verify_shortcut_audit(path: Path) -> tuple[str, str, str, int]:
    payload = verify_checksummed_payload(
        path, schema="origin-ordinal-shortcut-audit-v1"
    )
    if payload.get("complete_outer_fold") is not True:
        raise ValueError("shortcut aggregate contains a partial-fold audit")
    prediction = _resolve_declared_file(payload.get("prediction_artifact"))
    checkpoint = _resolve_declared_file(payload.get("checkpoint"))
    _verify_declared_hash(
        prediction, payload.get("prediction_artifact_sha256"),
        label="shortcut prediction artifact",
    )
    _verify_declared_hash(
        checkpoint, payload.get("checkpoint_sha256"), label="shortcut checkpoint"
    )
    return (
        str(payload.get("model_variant", "origin_ctmc")),
        str(payload.get("shortcut_arm")),
        str(payload.get("shortcut_family")),
        int(payload.get("training_seed", -1)),
    )


def validate_shortcut_aggregate(
    path: Path, repo_root: Path
) -> tuple[dict[str, Any], bool]:
    payload = verify_checksummed_payload(
        path, schema="origin-ordinal-shortcut-pilot-aggregate-v2"
    )
    if payload.get("expected_training_seeds") != [1701, 2603, 3907]:
        raise ValueError("shortcut training-seed set changed")
    model_reports = payload.get("model_variants")
    if not isinstance(model_reports, Mapping) or set(model_reports) != set(
        EXPECTED_SHORTCUT_VARIANTS
    ):
        raise ValueError("shortcut aggregate lacks the five matched model variants")
    for variant, record in model_reports.items():
        if record.get("official_author_implementation") is not False:
            raise ValueError(f"shortcut comparator {variant} is misattributed")
        families = record.get("families")
        if not isinstance(families, Mapping) or set(families) != {
            "localized", "border", "diffuse"
        }:
            raise ValueError(f"shortcut family coverage is incomplete for {variant}")

    audit_files = payload.get("audit_files")
    if not isinstance(audit_files, list) or len(audit_files) != 135:
        raise ValueError("shortcut aggregate must bind all 135 matched audit files")
    cells = [_verify_shortcut_audit(Path(item)) for item in audit_files]
    if len(set(cells)) != 135:
        raise ValueError("shortcut aggregate contains duplicate model/arm/family/seed audits")

    # Comparator-v2 aggregate, protocol, and immutable launch are siblings.
    submission_path = path.parent / "SUBMISSION.json"
    protocol_path = path.parent / "PROTOCOL_V2.json"
    submission = verify_checksummed_payload(
        submission_path, schema="origin-ordinal-shortcut-comparator-submission-v2"
    )
    protocol = verify_checksummed_payload(
        protocol_path, schema="origin-ordinal-shortcut-comparator-protocol-v2"
    )
    launch_commit = _verify_commit(
        repo_root, submission.get("launch_commit"), field="shortcut launch commit"
    )
    if protocol.get("launch_commit") != launch_commit:
        raise ValueError("shortcut protocol/submission launch commits differ")
    if protocol.get("additional_training_workers") != 84:
        raise ValueError("shortcut comparator protocol worker census changed")
    gate_passed = payload.get("passed") is True
    gates = payload.get("localized_family_gates_passed")
    if not isinstance(gates, list) or len(gates) != 2 or gate_passed != all(
        value is True for value in gates
    ):
        raise ValueError("shortcut Gate-B aggregate is internally inconsistent")
    return ({
        "schema": payload["schema"],
        "file_sha256": file_sha256(path),
        "content_checksum_sha256": payload["content_checksum_sha256"],
        "source_commit": launch_commit,
        "training_seeds": [1701, 2603, 3907],
        "model_variants": list(EXPECTED_SHORTCUT_VARIANTS),
        "audit_files_verified": 135,
        "gate_status": "pass" if gate_passed else "fail",
        "claim_authorized": gate_passed,
    }, gate_passed)


def validate_baseline_aggregate(path: Path, repo_root: Path) -> dict[str, Any]:
    payload = verify_checksummed_payload(
        path, schema="origin-acceptance-outer-aggregate-v1"
    )
    if payload.get("status") != "complete" or int(payload.get("release_worker_count", 0)) != 225:
        raise ValueError("matched outer-release aggregate is incomplete")
    if payload.get("comparator_implementation_scope", {}).get("kind") != (
        "matched_in_repo_analogues"
    ):
        raise ValueError("matched-baseline implementation scope changed")
    summary = payload.get("summary")
    if not isinstance(summary, Mapping) or set(summary) != {"aptos", "dr"}:
        raise ValueError("matched-baseline dataset coverage is incomplete")
    for dataset in ("aptos", "dr"):
        if set(summary[dataset]) != set(EXPECTED_SHORTCUT_VARIANTS):
            raise ValueError(f"matched-baseline model coverage is incomplete for {dataset}")
        for model in EXPECTED_SHORTCUT_VARIANTS:
            seed_means = summary[dataset][model].get("seed_cv_means", {})
            if set(seed_means) != {"42", "31415", "27182"}:
                raise ValueError("matched-baseline replication seed coverage is incomplete")

    release_root = path.parent
    marker_path = release_root / "OUTER_RELEASE_COMPLETE.json"
    marker = read_json(marker_path)
    if marker.get("schema") != "origin-acceptance-outer-release-complete-v1":
        raise ValueError("outer-release completion marker schema changed")
    if marker.get("aggregate_sha256") != file_sha256(path):
        raise ValueError("outer-release marker does not bind the aggregate")
    metrics_path = release_root / "fold_metrics.csv"
    _verify_declared_hash(
        metrics_path, marker.get("fold_metrics_sha256"), label="outer fold metrics"
    )
    if int(marker.get("worker_count", 0)) != 225:
        raise ValueError("outer-release marker worker census changed")

    experiment_root = path.parents[2]
    submission = verify_checksummed_payload(
        experiment_root / "SUBMISSION.json", schema="origin-acceptance-submission-v1"
    )
    if submission.get("protocol_sha256") != payload.get("protocol_sha256"):
        raise ValueError("matched-baseline submission/aggregate protocol mismatch")
    launch_commit = _verify_commit(
        repo_root, submission.get("launch_commit"), field="matched-baseline launch commit"
    )
    frozen = verify_checksummed_payload(
        experiment_root / "full" / "TRAINING_FROZEN.json",
        schema="origin-acceptance-training-audit-v1",
    )
    if frozen.get("status") != "passed" or int(frozen.get("worker_count", 0)) != 225:
        raise ValueError("matched-baseline training freeze audit is incomplete")
    if frozen.get("protocol_sha256") != payload.get("protocol_sha256"):
        raise ValueError("matched-baseline freeze/aggregate protocol mismatch")
    return {
        "schema": payload["schema"],
        "file_sha256": file_sha256(path),
        "content_checksum_sha256": payload["content_checksum_sha256"],
        "source_commit": launch_commit,
        "protocol_sha256": payload["protocol_sha256"],
        "release_workers": 225,
        "datasets": ["aptos", "dr"],
        "training_seeds": [42, 31415, 27182],
        "model_variants": list(EXPECTED_SHORTCUT_VARIANTS),
        "identifier_contract": "dataset_scoped_sha256_no_raw_paths_or_patient_ids",
    }


def _assert_privacy_safe_release(payload: Any, *, trail: tuple[str, ...] = ()) -> None:
    """Refuse filesystem locations and individual identifiers in exported JSON."""

    if isinstance(payload, Mapping):
        for key, value in payload.items():
            lower = str(key).lower()
            if lower == "path" or lower.endswith("_path") or lower.endswith("_paths"):
                raise ValueError(f"release payload exposes a filesystem field: {'.'.join(trail + (str(key),))}")
            if lower in {"image_id", "patient_id", "cluster_id", "sample_id"}:
                raise ValueError(f"release payload exposes an individual identifier: {lower}")
            _assert_privacy_safe_release(value, trail=trail + (str(key),))
    elif isinstance(payload, (list, tuple)):
        for index, value in enumerate(payload):
            _assert_privacy_safe_release(value, trail=trail + (str(index),))
    elif isinstance(payload, str):
        if "/" in payload or "\\" in payload or payload.startswith("file:"):
            raise ValueError(f"release payload exposes a filesystem location at {'.'.join(trail)}")


def assemble_package(
    *, repo_root: Path, output_dir: Path, numerical_audit: Path,
    decoder_contract_audit: Path, artifact_manifest: Path,
    idrid_manifest: Path, oof_manifest: Path, shortcut_aggregate: Path,
    baseline_aggregate: Path, expected_commit: str | None = None,
    require_gates_pass: bool = False,
) -> dict[str, Any]:
    repo_root = repo_root.resolve()
    output_dir = output_dir.resolve()
    manifest_path = output_dir / MANIFEST_FILENAME
    completion_path = output_dir / COMPLETION_FILENAME
    if manifest_path.exists() or completion_path.exists():
        raise FileExistsError("refusing to overwrite an existing acceptance package")
    provenance = runtime_provenance(repo_root, expected_commit)
    components: dict[str, Any] = {
        "independent_numerical_decoder_audit": validate_numerical(numerical_audit),
        "decoder_contract_audit": validate_decoder_contract(decoder_contract_audit),
        "checkpoint_split_artifact_audit": validate_artifact_manifest(
            artifact_manifest, repo_root
        ),
        "idrid_semantic_cluster_audit": validate_idrid_manifest(
            idrid_manifest, repo_root
        ),
    }
    oof_record, gate_a = validate_oof_manifest(oof_manifest, repo_root)
    shortcut_record, gate_b_passed = validate_shortcut_aggregate(
        shortcut_aggregate, repo_root
    )
    components["oof_gate_a"] = oof_record
    components["shortcut_gate_b"] = shortcut_record
    components["matched_baseline_outer_release"] = validate_baseline_aggregate(
        baseline_aggregate, repo_root
    )
    gates_pass = gate_a == "pass" and gate_b_passed
    if require_gates_pass and not gates_pass:
        raise RuntimeError(
            f"submission-readiness withheld: Gate A={gate_a}, "
            f"Gate B={'pass' if gate_b_passed else 'fail'}"
        )
    payload: dict[str, Any] = {
        "schema": PACKAGE_SCHEMA,
        "status": "verified_complete",
        "assembly_source": provenance,
        "required_component_count": 7,
        "components": components,
        "scientific_gates": {
            "gate_a": gate_a,
            "gate_b": "pass" if gate_b_passed else "fail",
            "all_passed": gates_pass,
            "claims_authorized": gates_pass,
            "claim_policy": (
                "auditability claims are withheld unless both preregistered gates pass"
            ),
        },
        "release_contract": {
            "input_artifacts_embedded": False,
            "licensed_pixels_copied": False,
            "checkpoints_copied": False,
            "raw_filesystem_locations_exported": False,
            "raw_image_or_patient_identifiers_exported": False,
            "exported_content": "schemas_digests_protocols_census_and_gate_status_only",
        },
    }
    payload["content_checksum_sha256"] = canonical_sha256(payload)
    _assert_privacy_safe_release(payload)

    output_dir.mkdir(parents=True, exist_ok=True)
    _atomic_json(manifest_path, payload)
    completion: dict[str, Any] = {
        "schema": COMPLETION_SCHEMA,
        "status": "verified_complete",
        "manifest_filename": MANIFEST_FILENAME,
        "manifest_sha256": file_sha256(manifest_path),
        "manifest_content_checksum_sha256": payload["content_checksum_sha256"],
        "all_required_components_verified": True,
        "scientific_gates_passed": gates_pass,
        "scientific_claims_authorized": gates_pass,
    }
    completion["content_checksum_sha256"] = canonical_sha256(completion)
    _assert_privacy_safe_release(completion)
    _atomic_json(completion_path, completion)
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--numerical-audit", type=Path, required=True)
    parser.add_argument("--decoder-contract-audit", type=Path, required=True)
    parser.add_argument("--artifact-manifest", type=Path, required=True)
    parser.add_argument("--idrid-manifest", type=Path, required=True)
    parser.add_argument("--oof-manifest", type=Path, required=True)
    parser.add_argument("--shortcut-aggregate", type=Path, required=True)
    parser.add_argument("--baseline-aggregate", type=Path, required=True)
    parser.add_argument("--expected-commit")
    parser.add_argument("--require-gates-pass", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    payload = assemble_package(
        repo_root=args.repo_root,
        output_dir=args.output_dir,
        numerical_audit=args.numerical_audit,
        decoder_contract_audit=args.decoder_contract_audit,
        artifact_manifest=args.artifact_manifest,
        idrid_manifest=args.idrid_manifest,
        oof_manifest=args.oof_manifest,
        shortcut_aggregate=args.shortcut_aggregate,
        baseline_aggregate=args.baseline_aggregate,
        expected_commit=args.expected_commit,
        require_gates_pass=args.require_gates_pass,
    )
    print(json.dumps({
        "status": payload["status"],
        "manifest": MANIFEST_FILENAME,
        "completion_marker": COMPLETION_FILENAME,
        "content_checksum_sha256": payload["content_checksum_sha256"],
        "scientific_gates": payload["scientific_gates"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
