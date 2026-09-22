from __future__ import annotations

import json
import math
from pathlib import Path

from scripts.origin_v3_cv_common import (
    canonical_sha256,
    metrics_from_confusion,
    sum_confusions,
    verify_checksummed_payload,
)


ROOT = Path(__file__).resolve().parents[1]


def test_confusion_metrics_are_exact_and_poolable() -> None:
    left = [[3, 1, 0, 0, 0], [0, 2, 0, 0, 0], [0, 0, 1, 0, 0], [0, 0, 0, 1, 0], [0, 0, 0, 0, 1]]
    right = [[1, 0, 0, 0, 0], [1, 1, 0, 0, 0], [0, 0, 2, 0, 0], [0, 0, 1, 1, 0], [0, 0, 0, 1, 1]]
    pooled = metrics_from_confusion(sum_confusions([left, right]))
    assert pooled["n"] == 18
    assert math.isclose(pooled["acc"], 100.0 * 14.0 / 18.0)
    assert math.isclose(pooled["mae"], 4.0 / 18.0)
    assert pooled["per_grade_support"] == [5, 4, 3, 3, 3]


def test_protocol_checksum_rejects_mutation() -> None:
    payload = {"schema": "example", "fold": 0}
    payload["content_checksum_sha256"] = canonical_sha256(payload)
    verify_checksummed_payload(payload, schema="example")
    mutated = json.loads(json.dumps(payload))
    mutated["fold"] = 1
    try:
        verify_checksummed_payload(mutated, schema="example")
    except ValueError as exc:
        assert "checksum mismatch" in str(exc)
    else:
        raise AssertionError("mutated protocol unexpectedly passed")


def test_cv_arrays_unlock_outer_test_and_isolate_worker_run_dirs() -> None:
    for name in (
        "submit_origin_v3_cv_aptos_array.sh",
        "submit_origin_v3_cv_dr_array.sh",
    ):
        source = (ROOT / "scripts" / name).read_text()
        assert "--include_test" in source
        assert "--skip_test" not in source
        assert 'workers/fold${FOLD}' in source
        assert "export_origin_v3_outer_predictions.py" in source
        assert "audit_origin_v3_cv_fold.py" in source
