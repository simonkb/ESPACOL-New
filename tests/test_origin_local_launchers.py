from __future__ import annotations

import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _launcher(name: str) -> str:
    path = ROOT / "scripts" / name
    assert path.stat().st_mode & os.X_OK
    return path.read_text(encoding="utf-8")


def test_rf_sealed_training_launchers_lock_the_controlled_scale_change() -> None:
    cases = {
        "submit_origin_v4_local_aptos_f0.sh": (
            "runs/origin_aptos_f0_v4_local_s4s8",
            "submit_origin_v3_aptos_f0.sh",
        ),
        "submit_origin_v4_local_dr_f0.sh": (
            "runs/origin_dr_f0_v4_local_s4s8",
            "submit_origin_v3_dr_f0.sh",
        ),
    }
    for name, (run_dir, delegate) in cases.items():
        script = _launcher(name)
        assert script.count('export ORIGIN_SCALES="s4,s8"') == 1
        assert run_dir in script
        assert delegate in script
        assert "s16" not in script
        assert "s32" not in script
        assert 'export ORIGIN_IMAGE_SIZE="640"' in script
        assert 'export ORIGIN_REFERENCE_COUNT="4096"' in script
        assert 'export ORIGIN_ENCODER_LR="1e-4"' in script
        assert 'export ORIGIN_HEAD_LR="5e-4"' in script


def test_rf_sealed_audit_targets_only_its_own_checkpoint() -> None:
    script = _launcher("submit_origin_v4_local_dr_validation_audit.sh")
    assert "runs/origin_dr_f0_v4_local_s4s8" in script
    assert 'export ORIGIN_EXPECTED_SCALES="s4,s8"' in script
    assert 'export ORIGIN_AUDIT_TOP_KS="1,5,10"' in script
    assert 'export ORIGIN_AUDIT_CERTIFICATES_PER_GRADE="2"' in script
    assert 'export ORIGIN_CHECKPOINT="${RUN_DIR}/fold0/best.pth"' in script
    assert (
        'export ORIGIN_AUDIT_OUTPUT="${RUN_DIR}/fold0/audits/'
        'full_validation_audit_v2.json"'
    ) in script
    assert "submit_origin_v3_dr_validation_audit.sh" in script
    delegated_audit = _launcher("submit_origin_v3_dr_validation_audit.sh")
    assert "ORIGIN_EXPECTED_SCALES" in delegated_audit
    assert "audit checkpoint evidence scales" in delegated_audit
