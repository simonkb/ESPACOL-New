from __future__ import annotations

import ast
import re
import shlex
import subprocess
from pathlib import Path

import pytest

from scripts.origin_fold9_ablation_common import (
    frozen_config_for_variant,
    validate_variant_config,
)


ROOT = Path(__file__).resolve().parents[1]
FRESH = ROOT / "scripts" / "submit_origin_fold9_ablation_train_array.sh"
RETRY = ROOT / "scripts" / "retry_origin_fold9_ablation_variants.sh"
RESUME = ROOT / "scripts" / "submit_origin_fold9_ablation_resume_array.sh"
FRESH_RESTART = ROOT / "scripts" / "restart_origin_fold9_ablation_variant_fresh.sh"


def _train_args(path: Path) -> list[str]:
    text = path.read_text(encoding="utf-8")
    match = re.search(r"TRAIN_ARGS=\(\n(.*?)\n\)", text, flags=re.DOTALL)
    assert match is not None
    return shlex.split(match.group(1), comments=True, posix=True)


def test_resume_worker_changes_only_invocation_policy() -> None:
    fresh = _train_args(FRESH)
    resumed = _train_args(RESUME)
    assert resumed.count("--resume") == 1
    resumed.remove("--resume")
    assert resumed == fresh
    assert "--skip_test" in resumed
    assert "--include_test" not in resumed


def test_retry_scripts_are_valid_and_fail_closed() -> None:
    for path in (RETRY, RESUME):
        subprocess.run(["bash", "-n", str(path)], check=True)

    launcher = RETRY.read_text(encoding="utf-8")
    worker = RESUME.read_text(encoding="utf-8")
    assert "verify_protocol" in launcher and "verify_protocol" in worker
    assert "validate_split_manifest" in launcher and "validate_split_manifest" in worker
    assert "validate_variant_config" in launcher and "validate_variant_config" in worker
    assert "origin_ablation_implementation_signature" in launcher
    assert "origin_ablation_implementation_signature" in worker
    assert "last.pth" in launcher and "last.pth" in worker
    assert "ABLATION_TRAIN_COMPLETE.json" in launcher
    assert ".release.staging.*" in launcher and ".release.staging.*" in worker
    assert "the retry list must equal all variants without completion markers" in launcher
    assert "retry_submission.lock" in launcher
    assert "RETRY_SUBMISSION_*.json" in launcher
    assert 'afterany:${RESUME_JOB}' in launcher
    assert 'afterany:${RELEASE_JOB}' in launcher
    assert "submit_origin_fold9_ablation_outer_release.sh" in launcher
    assert "submit_origin_fold9_ablation_aggregate.sh" in launcher
    assert "LOCKED_PROTOCOL.json" in launcher
    assert launcher.index('cd "${SNAPSHOT_ROOT}"') < launcher.index(
        "from scripts.origin_fold9_ablation_common import ("
    )
    assert "write_json_atomic(os.environ[\"ORIGIN_RETRY_PROTOCOL\"]" not in launcher
    assert "--resume" in worker


def test_incident_fresh_restart_is_one_time_and_uses_immutable_worker() -> None:
    subprocess.run(["bash", "-n", str(FRESH_RESTART)], check=True)

    launcher = FRESH_RESTART.read_text(encoding="utf-8")
    assert 'EXPECTED_VARIANT="origin_coarse_only"' in launcher
    assert "EXPECTED_INDEX=7" in launcher
    assert "EXPECTED_LAST_EPOCH=34" in launcher
    assert 'EXPECTED_REASON="repeated_forward_nan_after_exact_resume"' in launcher
    assert 'if [[ "${DRY_RUN}" == true ]]' in launcher
    assert "verify_protocol" in launcher
    assert "validate_split_manifest" in launcher
    assert "validate_variant_config" in launcher
    assert "origin_ablation_implementation_signature" in launcher
    assert "bad_tensors" in launcher
    assert "FRESH_RESTART_AUTHORIZATION_*.json" in launcher
    assert "FRESH_RESTART_SUBMISSION_*.json" in launcher
    assert "FRESH_RESTART_ONCE.json" in launcher
    assert "failed_attempts" in launcher
    assert '"failed_attempt_count": len(attempts)' in launcher
    assert '"fresh_restart_attempt_limit": 1' in launcher
    assert "post_hoc_operational_recovery_before_outer_release" in launcher
    assert "result_reporting_requirement" in launcher
    assert "ARCHIVE_MANIFEST.json" in launcher
    assert "atexit.register(rollback_uncommitted_archive)" in launcher
    assert "signal.signal(signal.SIGTERM, interrupt_archive)" in launcher
    assert "file_sha256" in launcher
    assert "submit_origin_fold9_ablation_train_array.sh" in launcher
    assert '--array="${EXPECTED_INDEX}-${EXPECTED_INDEX}%1"' in launcher
    assert "--hold" in launcher
    assert "SUBMISSION_TRANSACTION_ARMED=true" in launcher
    assert 'scancel "${submitted_job}"' in launcher
    assert 'scontrol release "${FRESH_JOB}"' in launcher
    assert 'afterany:${FRESH_JOB}' in launcher
    assert 'afterany:${RELEASE_JOB}' in launcher
    assert "train_origin_ablation.py" not in launcher
    assert "--resume" not in launcher
    assert "write_json_atomic(os.environ[\"ORIGIN_FRESH_PROTOCOL\"]" not in launcher
    assert "retry_count != 1" in launcher
    assert "squeue -h -j" not in launcher
    assert 'squeue -h -u "${CURRENT_USER}" -o \'%F|%j\'' in launcher
    assert launcher.index('if [[ "${DRY_RUN}" == true ]]') < launcher.index(
        'ORIGIN_FRESH_ARCHIVE="${ARCHIVE_ROOT}"'
    )
    assert launcher.index("SUBMISSION_TRANSACTION_ARMED=true") < launcher.index(
        "fresh_job_raw="
    )
    assert launcher.index("flock -u 7") < launcher.index(
        'scontrol release "${FRESH_JOB}"'
    )
    assert launcher.index('update_submission_record "chain_released"') < launcher.rindex(
        "SUBMISSION_TRANSACTION_ARMED=false"
    )
    python_blocks = re.findall(r"<<'PY'\n(.*?)\nPY(?:\n|$)", launcher, re.DOTALL)
    assert len(python_blocks) == 5
    for block in python_blocks:
        ast.parse(block)


def test_incident_fresh_restart_rejects_broader_scope_before_environment_setup() -> None:
    reason = "repeated_forward_nan_after_exact_resume"
    wrong_variant = subprocess.run(
        [str(FRESH_RESTART), "/tmp/not-used", "origin_full", reason],
        text=True,
        capture_output=True,
        check=False,
    )
    assert wrong_variant.returncode == 65
    assert "restricted to origin_coarse_only" in wrong_variant.stderr

    wrong_reason = subprocess.run(
        [str(FRESH_RESTART), "/tmp/not-used", "origin_coarse_only", "retry"],
        text=True,
        capture_output=True,
        check=False,
    )
    assert wrong_reason.returncode == 66
    assert "explicit one-time reason" in wrong_reason.stderr


def test_resume_is_the_only_allowed_runtime_config_difference() -> None:
    config = frozen_config_for_variant("origin_full")
    config["resume"] = True
    validate_variant_config(config, "origin_full")

    config["lr"] = 2e-4
    with pytest.raises(ValueError, match="locked origin_full configuration"):
        validate_variant_config(config, "origin_full")

    malformed = frozen_config_for_variant("origin_full")
    malformed["resume"] = "yes"
    with pytest.raises(ValueError, match="resume policy must be boolean"):
        validate_variant_config(malformed, "origin_full")
