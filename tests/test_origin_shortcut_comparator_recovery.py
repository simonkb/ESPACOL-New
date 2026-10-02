from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import shlex
import subprocess

import pytest

from scripts.origin_shortcut_comparator_common import (
    PROTOCOL_CORE_SHA256,
    canonical_sha256,
    task_at,
)
from scripts.validate_origin_shortcut_comparator_recovery import (
    RECOVERABLE_TASKS,
    validate_console_failure_artifacts,
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")


def _clean_worktree(path: Path) -> str:
    path.mkdir()
    subprocess.run(["git", "init", "-q", str(path)], check=True)
    subprocess.run(["git", "-C", str(path), "config", "user.email", "test@example.com"], check=True)
    subprocess.run(["git", "-C", str(path), "config", "user.name", "Test"], check=True)
    (path / "identity.txt").write_text("frozen\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(path), "add", "identity.txt"], check=True)
    subprocess.run(["git", "-C", str(path), "commit", "-qm", "freeze"], check=True)
    return subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    ).stdout.strip()


def _fixture(
    tmp_path: Path, task_ids: tuple[int, ...] = (21, 22)
) -> tuple[Path, Path]:
    suite = tmp_path / "suite"
    worktree = tmp_path / "snapshot"
    commit = _clean_worktree(worktree)
    for task_id in task_ids:
        task = task_at(task_id)
        fold = (
            suite
            / "workers"
            / task.model_variant
            / task.arm
            / task.family
            / f"seed{task.training_seed}"
            / "fold0"
        )
        fold.mkdir(parents=True)
        checkpoint = fold / "best_learned.pth"
        checkpoint.write_bytes(f"checkpoint-{task_id}".encode())
        families = ("localized", "border", "diffuse") if task.arm == "clean" else (task.family,)
        for family in families:
            stem = "shortcut_audit" if task.arm != "clean" else f"shortcut_audit_{family}"
            prediction = fold / f"{stem}_predictions.jsonl"
            manifest = {
                "record_type": "manifest",
                "schema": "origin-ordinal-shortcut-predictions-v1",
                "dataset": "aptos",
                "model_variant": task.model_variant,
                "shortcut_arm": task.arm,
                "shortcut_family": family,
                "training_seed": task.training_seed,
                "fold": 0,
                "split_seed": 42,
                "checkpoint": str(checkpoint.resolve()),
                "checkpoint_sha256": _sha(checkpoint),
                "condition_prediction_records": 0,
                "factorial_prediction_records": 0,
                "internal_effect_records": 0,
            }
            prediction.write_text(
                json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n",
                encoding="utf-8",
            )
            report = {
                "schema": "origin-ordinal-shortcut-audit-v1",
                "dataset": "aptos",
                "checkpoint": str(checkpoint.resolve()),
                "checkpoint_sha256": _sha(checkpoint),
                "prediction_artifact": str(prediction.resolve()),
                "prediction_artifact_sha256": _sha(prediction),
                "model_variant": task.model_variant,
                "shortcut_arm": task.arm,
                "shortcut_family": family,
                "training_seed": task.training_seed,
                "fold": 0,
                "split_seed": 42,
                "complete_outer_fold": True,
                "official_author_implementation": False,
                "local_ledger_applicable": False,
                "localization": {"applicable": False, "reason": "not applicable"},
                "internal_pixel_effects": {"applicable": False, "reason": "not applicable"},
                "sample_count": 733,
                "condition_prediction_records": 0,
                "factorial_prediction_records": 0,
                "internal_effect_records": 0,
            }
            report["content_checksum_sha256"] = canonical_sha256(report)
            _write_json(fold / f"{stem}.json", report)

    protocol = {
        "schema": "origin-ordinal-shortcut-comparator-protocol-v2",
        "protocol_core_sha256": PROTOCOL_CORE_SHA256,
        "suite_root": str(suite.resolve()),
        "immutable_worktree": str(worktree.resolve()),
        "launch_commit": commit,
    }
    protocol["content_checksum_sha256"] = canonical_sha256(protocol)
    protocol_path = suite / "PROTOCOL_V2.json"
    _write_json(protocol_path, protocol)
    return suite, protocol_path


def test_console_failure_recovery_is_read_only_and_exactly_scoped(tmp_path: Path) -> None:
    suite, protocol = _fixture(tmp_path)
    before = {
        path: _sha(path)
        for path in suite.rglob("*")
        if path.is_file()
    }
    payload = validate_console_failure_artifacts(
        suite_root=suite,
        protocol_path=protocol,
        task_ids=(21, 22),
    )
    after = {path: _sha(path) for path in before}
    assert before == after
    assert payload["scientific_protocol_modified"] is False
    assert payload["validator_modified_scientific_artifacts"] is False
    assert payload["recovery_execution"]["scientific_artifacts_created"] is False
    assert [item["task_id"] for item in payload["validated_tasks"]] == [21, 22]

    with pytest.raises(ValueError, match="sorted, unique subset"):
        validate_console_failure_artifacts(
            suite_root=suite,
            protocol_path=protocol,
            task_ids=(20,),
        )


def test_recovery_rejects_fabricated_pooled_metric(tmp_path: Path) -> None:
    suite, protocol = _fixture(tmp_path)
    task = task_at(21)
    audit = (
        suite
        / "workers"
        / task.model_variant
        / task.arm
        / task.family
        / f"seed{task.training_seed}"
        / "fold0"
        / "shortcut_audit.json"
    )
    payload = json.loads(audit.read_text())
    payload["localization"]["macro_auprc"] = 0.99
    payload.pop("content_checksum_sha256")
    payload["content_checksum_sha256"] = canonical_sha256(payload)
    _write_json(audit, payload)
    with pytest.raises(ValueError, match="fabricated spatial metrics"):
        validate_console_failure_artifacts(
            suite_root=suite,
            protocol_path=protocol,
            task_ids=(21, 22),
        )


def test_clean_recovery_requires_and_validates_all_three_families(tmp_path: Path) -> None:
    suite, protocol = _fixture(tmp_path, task_ids=(39,))
    payload = validate_console_failure_artifacts(
        suite_root=suite, protocol_path=protocol, task_ids=(39,)
    )
    assert [row["family"] for row in payload["validated_tasks"][0]["audits"]] == [
        "localized", "border", "diffuse"
    ]


def test_recovery_rejects_wrong_checkpoint_and_prediction_paths(tmp_path: Path) -> None:
    suite, protocol = _fixture(tmp_path, task_ids=(21,))
    task = task_at(21)
    fold = (
        suite / "workers" / task.model_variant / task.arm / task.family
        / f"seed{task.training_seed}" / "fold0"
    )
    audit = fold / "shortcut_audit.json"
    payload = json.loads(audit.read_text())
    alternate = fold / "alternate.pth"
    alternate.write_bytes((fold / "best_learned.pth").read_bytes())
    payload["checkpoint"] = str(alternate.resolve())
    payload.pop("content_checksum_sha256")
    payload["content_checksum_sha256"] = canonical_sha256(payload)
    _write_json(audit, payload)
    with pytest.raises(ValueError, match="checkpoint path mismatch"):
        validate_console_failure_artifacts(
            suite_root=suite, protocol_path=protocol, task_ids=(21,)
        )


def test_recovery_rejects_prediction_record_count_mismatch(tmp_path: Path) -> None:
    suite, protocol = _fixture(tmp_path, task_ids=(21,))
    task = task_at(21)
    fold = (
        suite / "workers" / task.model_variant / task.arm / task.family
        / f"seed{task.training_seed}" / "fold0"
    )
    audit = fold / "shortcut_audit.json"
    payload = json.loads(audit.read_text())
    prediction = Path(payload["prediction_artifact"])
    manifest = json.loads(prediction.read_text().splitlines()[0])
    manifest["condition_prediction_records"] = 1
    prediction.write_text(json.dumps(manifest) + "\n", encoding="utf-8")
    payload["prediction_artifact_sha256"] = _sha(prediction)
    payload["condition_prediction_records"] = 1
    payload.pop("content_checksum_sha256")
    payload["content_checksum_sha256"] = canonical_sha256(payload)
    _write_json(audit, payload)
    with pytest.raises(ValueError, match="prediction count mismatch"):
        validate_console_failure_artifacts(
            suite_root=suite, protocol_path=protocol, task_ids=(21,)
        )


def _train_args(path: Path) -> list[str]:
    text = path.read_text(encoding="utf-8")
    match = re.search(r"TRAIN_ARGS=\(\n(.*?)\n\)", text, flags=re.DOTALL)
    assert match is not None
    return shlex.split(match.group(1), comments=True, posix=True)


def test_continuation_uses_exact_training_args_and_safe_dependencies() -> None:
    root = Path(__file__).resolve().parents[1]
    original = root / "scripts" / "submit_origin_shortcut_comparator_v2.sh"
    worker = root / "scripts" / "submit_origin_shortcut_pooled_continuation_v2.sh"
    launcher = root / "scripts" / "continue_origin_shortcut_pooled_v2.sh"
    assert _train_args(worker) == _train_args(original)
    for path in (worker, launcher):
        subprocess.run(["bash", "-n", str(path)], check=True)
    worker_text = worker.read_text(encoding="utf-8")
    launcher_text = launcher.read_text(encoding="utf-8")
    assert "git -C \"${REPO_ROOT}\" rev-parse HEAD" in worker_text
    assert "python scripts/audit_origin_shortcut.py" in worker_text
    assert "validate_origin_shortcut_comparator_recovery.py" in worker_text
    assert 'for audit_family in localized border diffuse' in worker_text
    assert "--resume" not in worker_text
    assert 'for task_id in $(seq 21 41)' in launcher_text
    assert 'POOLED_ORIGINAL_ELEMENTS+=("${ORIGINAL_WORKER_JOB}_${task_id}")' in launcher_text
    assert 'RUNNING_POOLED_DEPENDENCY="afterany:' in launcher_text
    assert '--dependency="${RUNNING_POOLED_DEPENDENCY}"' in launcher_text
    assert (
        '--dependency="afterok:${CONTINUATION_JOB},afterany:${ORIGINAL_WORKER_JOB}"'
        in launcher_text
    )
    assert '--array="21-41%' in launcher_text
    assert "git -C \"${SOURCE_ROOT}\" worktree add --detach" in launcher_text
    assert 'immutable_recovery_worktree' in launcher_text
    assert 'cd "${SNAPSHOT_ROOT}"' in launcher_text
