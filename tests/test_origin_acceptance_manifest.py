from __future__ import annotations

import gzip
import hashlib
import json
import subprocess

from tools.build_origin_acceptance_manifest import (
    _anonymous_id,
    _audit_source_provenance,
    _canonical_sha256,
    _membership_rows,
    _validated_cv_submission,
    _write_deterministic_csv_gz,
)


def test_membership_export_is_anonymous_and_deterministic(tmp_path) -> None:
    items = [
        ("/private/data/100_left.jpeg", 1),
        ("/private/data/100_right.jpeg", 2),
    ]
    rows = _membership_rows("dr", (("outer_test", items),))
    assert len(rows) == 2
    assert all("private" not in str(row) for row in rows)
    assert rows[0]["patient_cluster_sha256"] == rows[1]["patient_cluster_sha256"]
    assert rows[0]["patient_cluster_sha256"] == _anonymous_id("dr", "100")

    first = tmp_path / "first.csv.gz"
    second = tmp_path / "second.csv.gz"
    _write_deterministic_csv_gz(first, rows)
    _write_deterministic_csv_gz(second, rows)
    assert first.read_bytes() == second.read_bytes()
    with gzip.open(first, "rt", encoding="utf-8") as stream:
        content = stream.read()
    assert "image_id_sha256" in content
    assert "100_left" not in content


def test_cv_submission_binds_training_commit_and_checksum(tmp_path) -> None:
    payload = {
        "schema": "origin-v3-full-cv-submission-v1",
        "tag": "frozen",
        "launch_commit": "a" * 40,
        "jobs": {},
        "roots": {},
        "created_at_utc": "2026-09-22T00:00:00+00:00",
    }
    payload["content_checksum_sha256"] = _canonical_sha256(payload)
    path = tmp_path / "SUBMISSION.json"
    encoded = json.dumps(payload, indent=2, sort_keys=True).encode("utf-8") + b"\n"
    path.write_bytes(encoded)

    record = _validated_cv_submission(tmp_path)
    assert record["training_launch_commit"] == "a" * 40
    assert record["content_checksum_sha256"] == payload["content_checksum_sha256"]
    assert record["sha256"] == hashlib.sha256(encoded).hexdigest()


def test_cv_submission_rejects_changed_content(tmp_path) -> None:
    payload = {
        "schema": "origin-v3-full-cv-submission-v1",
        "tag": "frozen",
        "launch_commit": "b" * 40,
        "content_checksum_sha256": "0" * 64,
    }
    (tmp_path / "SUBMISSION.json").write_text(json.dumps(payload), encoding="utf-8")
    try:
        _validated_cv_submission(tmp_path)
    except ValueError as error:
        assert "checksum" in str(error)
    else:
        raise AssertionError("tampered submission was accepted")


def test_audit_source_provenance_binds_current_clean_commit(tmp_path) -> None:
    subprocess.run(["git", "init"], cwd=tmp_path, check=True, capture_output=True)
    subprocess.run(
        ["git", "config", "user.email", "audit@example.invalid"],
        cwd=tmp_path,
        check=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Audit Test"], cwd=tmp_path, check=True
    )
    (tmp_path / "source.txt").write_text("sealed\n", encoding="utf-8")
    subprocess.run(["git", "add", "source.txt"], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-m", "seal"], cwd=tmp_path, check=True)
    provenance = _audit_source_provenance(tmp_path)
    assert len(provenance["audit_source_commit"]) == 40
    assert provenance["tracked_tree_dirty"] is False
