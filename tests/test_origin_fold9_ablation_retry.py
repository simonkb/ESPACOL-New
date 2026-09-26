from __future__ import annotations

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
