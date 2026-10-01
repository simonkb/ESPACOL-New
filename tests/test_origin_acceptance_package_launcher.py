from __future__ import annotations

import hashlib
import json
import os
import shutil
import stat
import subprocess
from pathlib import Path


LAUNCHER = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "launch_origin_acceptance_package.sh"
)
LABELS = [
    "numerical",
    "decoder",
    "artifacts",
    "idrid",
    "oof",
    "shortcut",
    "baselines",
]
ARTIFACT_ENV = [
    "ORIGIN_NUMERICAL_AUDIT",
    "ORIGIN_DECODER_CONTRACT_AUDIT",
    "ORIGIN_ARTIFACT_MANIFEST",
    "ORIGIN_IDRID_STATISTICS_MANIFEST",
    "ORIGIN_OOF_STATISTICS_MANIFEST",
    "ORIGIN_SHORTCUT_AGGREGATE",
    "ORIGIN_BASELINE_AGGREGATE",
]
JOB_ENV = [
    "ORIGIN_NUMERICAL_AUDIT_JOB",
    "ORIGIN_DECODER_CONTRACT_JOB",
    "ORIGIN_ARTIFACT_MANIFEST_JOB",
    "ORIGIN_IDRID_STATISTICS_JOB",
    "ORIGIN_OOF_GATE_A_JOB",
    "ORIGIN_SHORTCUT_GATE_B_JOB",
    "ORIGIN_BASELINE_AGGREGATE_JOB",
]


def _write_executable(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR)


def _run(*args: str, cwd: Path, env: dict[str, str] | None = None) -> str:
    result = subprocess.run(
        args,
        cwd=cwd,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _fixture(tmp_path: Path) -> dict[str, object]:
    repo = tmp_path / "repo"
    scripts = repo / "scripts"
    scripts.mkdir(parents=True)
    shutil.copy2(LAUNCHER, scripts / LAUNCHER.name)
    _write_executable(
        scripts / "submit_origin_acceptance_package.sh",
        "#!/bin/bash\nset -euo pipefail\nexit 0\n",
    )
    _run("git", "init", cwd=repo)
    _run("git", "checkout", "-b", "origin-acceptance-revision", cwd=repo)
    _run("git", "add", "scripts", cwd=repo)
    commit_env = dict(os.environ)
    commit_env.update(
        {
            "GIT_AUTHOR_NAME": "test",
            "GIT_AUTHOR_EMAIL": "test@example.invalid",
            "GIT_COMMITTER_NAME": "test",
            "GIT_COMMITTER_EMAIL": "test@example.invalid",
        }
    )
    _run("git", "commit", "-m", "fixture", cwd=repo, env=commit_env)

    fake_bin = tmp_path / "bin"
    state_dir = tmp_path / "scheduler"
    fake_bin.mkdir()
    state_dir.mkdir()
    _write_executable(
        fake_bin / "squeue",
        """#!/bin/bash
set -euo pipefail
job=""
while [[ $# -gt 0 ]]; do
  if [[ "$1" == "-j" ]]; then job="$2"; shift 2; else shift; fi
done
if [[ -f "${FAKE_STATE_DIR}/${job}.queue_error" ]]; then
  cat "${FAKE_STATE_DIR}/${job}.queue_error" >&2
  exit 1
fi
[[ -f "${FAKE_STATE_DIR}/${job}.queue" ]] && cat "${FAKE_STATE_DIR}/${job}.queue"
exit 0
""",
    )
    _write_executable(
        fake_bin / "sacct",
        """#!/bin/bash
set -euo pipefail
job=""
while [[ $# -gt 0 ]]; do
  if [[ "$1" == "-j" ]]; then job="$2"; shift 2; else shift; fi
done
if [[ -f "${FAKE_STATE_DIR}/${job}.acct" ]]; then
  printf '%s|' "${job}"
  cat "${FAKE_STATE_DIR}/${job}.acct"
  printf '\n'
fi
""",
    )
    _write_executable(
        fake_bin / "sbatch",
        """#!/bin/bash
set -euo pipefail
printf '%s\n' "$*" >> "${FAKE_SBATCH_LOG}"
printf '9001;cluster\n'
""",
    )
    _write_executable(
        fake_bin / "scontrol",
        """#!/bin/bash
set -euo pipefail
printf '%s\n' "$*" >> "${FAKE_SCONTROL_LOG}"
[[ "${FAKE_SCONTROL_FAIL:-0}" != "1" ]]
""",
    )
    _write_executable(
        fake_bin / "scancel",
        """#!/bin/bash
set -euo pipefail
printf '%s\n' "$*" >> "${FAKE_SCANCEL_LOG}"
""",
    )

    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    output = tmp_path / "output" / "package"
    snapshot = tmp_path / "snapshot"
    env = dict(os.environ)
    env.update(
        {
            "PATH": f"{fake_bin}:{os.environ['PATH']}",
            "FAKE_STATE_DIR": str(state_dir),
            "FAKE_SBATCH_LOG": str(tmp_path / "sbatch.log"),
            "FAKE_SCONTROL_LOG": str(tmp_path / "scontrol.log"),
            "FAKE_SCANCEL_LOG": str(tmp_path / "scancel.log"),
            "ORIGIN_PACKAGE_OUTPUT": str(output),
            "ORIGIN_PACKAGE_WORKTREE": str(snapshot),
        }
    )
    artifact_paths: list[Path] = []
    for index, (artifact_name, job_name) in enumerate(zip(ARTIFACT_ENV, JOB_ENV)):
        artifact = artifacts / f"{LABELS[index]}.json"
        artifact.write_text(f'{{"label":"{LABELS[index]}"}}\n', encoding="utf-8")
        artifact_paths.append(artifact)
        env[artifact_name] = str(artifact)
        env[job_name] = str(101 + index)
    return {
        "repo": repo,
        "env": env,
        "state": state_dir,
        "artifacts": artifact_paths,
        "output": output,
        "snapshot": snapshot,
        "sbatch_log": tmp_path / "sbatch.log",
        "scontrol_log": tmp_path / "scontrol.log",
        "scancel_log": tmp_path / "scancel.log",
    }


def _launch(fixture: dict[str, object]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", "scripts/launch_origin_acceptance_package.sh"],
        cwd=fixture["repo"],
        env=fixture["env"],
        check=False,
        capture_output=True,
        text=True,
    )


def _set_accounting_states(state_dir: Path, states: dict[int, str]) -> None:
    for job, state in states.items():
        (state_dir / f"{job}.acct").write_text(state, encoding="utf-8")


def test_mixed_live_and_completed_jobs_build_minimal_dependency_and_record(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    state_dir = fixture["state"]
    assert isinstance(state_dir, Path)
    (state_dir / "101.queue").write_text("RUNNING\n", encoding="utf-8")
    _set_accounting_states(
        state_dir, {job: "COMPLETED" for job in range(102, 108)} | {777: "CANCELLED by 42"}
    )
    env = fixture["env"]
    assert isinstance(env, dict)
    env["ORIGIN_SUPERSEDED_PACKAGE_JOBS"] = "777"

    result = _launch(fixture)
    assert result.returncode == 0, result.stderr
    sbatch = Path(fixture["sbatch_log"]).read_text(encoding="utf-8")
    assert "--hold" in sbatch
    assert "--dependency=afterok:101" in sbatch
    for completed_job in range(102, 108):
        assert f"afterok:{completed_job}" not in sbatch
        assert f":{completed_job}" not in sbatch.split("--dependency=", 1)[1].split()[0]
    assert Path(fixture["scontrol_log"]).read_text(encoding="utf-8") == "release 9001\n"
    assert not Path(fixture["scancel_log"]).exists()

    output = fixture["output"]
    assert isinstance(output, Path)
    record_path = Path(f"{output}.SUBMISSION.json")
    record = json.loads(record_path.read_text(encoding="utf-8"))
    checksum = record.pop("content_checksum_sha256")
    encoded = json.dumps(
        record, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    assert checksum == hashlib.sha256(encoded).hexdigest()
    assert record["dependency_expression"] == "afterok:101"
    assert record["package_job_id"] == "9001"
    assert record["submission_control"] == "sbatch_hold_record_then_scontrol_release"
    assert [row["state_class"] for row in record["upstreams"]] == [
        "live",
        *(["completed"] * 6),
    ]
    assert [row["included_in_afterok"] for row in record["upstreams"]] == [
        True,
        *([False] * 6),
    ]
    assert record["superseded_package_jobs"] == [
        {
            "job_id": "777",
            "scheduler_state": "CANCELLED",
            "state_class": "failure",
        }
    ]
    for row in record["upstreams"][1:]:
        assert row["artifact"]["bytes_at_launch"] > 0
        assert len(row["artifact"]["sha256_at_launch"]) == 64
    assert Path(f"{output}.launch-reservation").is_dir()
    assert not output.exists()


def test_all_completed_jobs_are_not_reused_as_afterok_dependencies(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    state_dir = fixture["state"]
    assert isinstance(state_dir, Path)
    _set_accounting_states(
        state_dir, {job: "COMPLETED+" for job in range(101, 108)}
    )
    result = _launch(fixture)
    assert result.returncode == 0, result.stderr
    sbatch = Path(fixture["sbatch_log"]).read_text(encoding="utf-8")
    assert "--dependency=" not in sbatch
    output = fixture["output"]
    assert isinstance(output, Path)
    record = json.loads(Path(f"{output}.SUBMISSION.json").read_text(encoding="utf-8"))
    assert record["dependency_expression"] is None


def test_purged_squeue_job_falls_back_to_exact_completed_sacct_record(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    state_dir = fixture["state"]
    assert isinstance(state_dir, Path)
    (state_dir / "101.queue_error").write_text(
        "slurm_load_jobs error: Invalid job id specified\n", encoding="utf-8"
    )
    _set_accounting_states(
        state_dir, {job: "COMPLETED" for job in range(101, 108)}
    )

    result = _launch(fixture)
    assert result.returncode == 0, result.stderr
    assert "consulting sacct" in result.stderr
    assert "Invalid job id specified" in result.stderr
    sbatch = Path(fixture["sbatch_log"]).read_text(encoding="utf-8")
    assert "--dependency=" not in sbatch
    output = fixture["output"]
    assert isinstance(output, Path)
    record = json.loads(Path(f"{output}.SUBMISSION.json").read_text(encoding="utf-8"))
    assert record["upstreams"][0]["job_id"] == "101"
    assert record["upstreams"][0]["scheduler_state"] == "COMPLETED"
    assert record["upstreams"][0]["state_class"] == "completed"
    assert record["upstreams"][0]["included_in_afterok"] is False


def test_completed_job_without_nonempty_artifact_aborts_before_submission(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    state_dir = fixture["state"]
    assert isinstance(state_dir, Path)
    _set_accounting_states(state_dir, {job: "COMPLETED" for job in range(101, 108)})
    artifacts = fixture["artifacts"]
    assert isinstance(artifacts, list)
    artifacts[0].write_bytes(b"")

    result = _launch(fixture)
    assert result.returncode != 0
    assert "has no non-empty artifact" in result.stderr
    assert not Path(fixture["sbatch_log"]).exists()
    output = fixture["output"]
    assert isinstance(output, Path)
    assert not Path(f"{output}.launch-reservation").exists()


def test_failed_upstream_aborts_instead_of_creating_dependency(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    state_dir = fixture["state"]
    assert isinstance(state_dir, Path)
    _set_accounting_states(
        state_dir,
        {101: "FAILED"} | {job: "COMPLETED" for job in range(102, 108)},
    )
    result = _launch(fixture)
    assert result.returncode != 0
    assert "job 101 failed with state FAILED" in result.stderr
    assert not Path(fixture["sbatch_log"]).exists()


def test_superseded_package_job_must_be_terminal_cancelled(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    state_dir = fixture["state"]
    assert isinstance(state_dir, Path)
    _set_accounting_states(state_dir, {job: "COMPLETED" for job in range(101, 108)})
    (state_dir / "777.queue").write_text("PENDING\n", encoding="utf-8")
    env = fixture["env"]
    assert isinstance(env, dict)
    env["ORIGIN_SUPERSEDED_PACKAGE_JOBS"] = "777"
    result = _launch(fixture)
    assert result.returncode != 0
    assert "is not terminal CANCELLED" in result.stderr
    assert not Path(fixture["sbatch_log"]).exists()


def test_release_failure_cancels_held_job_before_it_can_write(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    state_dir = fixture["state"]
    assert isinstance(state_dir, Path)
    _set_accounting_states(state_dir, {job: "COMPLETED" for job in range(101, 108)})
    env = fixture["env"]
    assert isinstance(env, dict)
    env["FAKE_SCONTROL_FAIL"] = "1"

    result = _launch(fixture)
    assert result.returncode != 0
    assert Path(fixture["scancel_log"]).read_text(encoding="utf-8") == "9001\n"
    output = fixture["output"]
    assert isinstance(output, Path)
    assert not output.exists()
    assert not Path(f"{output}.launch-reservation").exists()
    # The checksummed receipt remains as a fail-closed audit trail and blocks
    # accidental resubmission to the same output location.
    assert Path(f"{output}.SUBMISSION.json").is_file()


def test_launcher_is_executable_and_shell_parses() -> None:
    assert os.access(LAUNCHER, os.X_OK)
    subprocess.run(["bash", "-n", str(LAUNCHER)], check=True)
