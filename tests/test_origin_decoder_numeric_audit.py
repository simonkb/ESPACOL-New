from __future__ import annotations

import json

from tools.audit_origin_decoder_numeric import SCHEMA, audit_decoder


def test_independent_decoder_audit_passes_small_canary() -> None:
    payload = audit_decoder(
        samples=2,
        gradient_samples=2,
        seed=17,
        rate_cap=8.0,
        dps=70,
    )
    assert payload["schema"] == SCHEMA
    assert payload["summary"]["passed"]
    assert payload["n_forward_cases"] == 12
    assert payload["n_gradient_cases"] == 2
    unsigned = dict(payload)
    checksum = unsigned.pop("content_checksum_sha256")
    encoded = json.dumps(
        unsigned,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    import hashlib

    assert hashlib.sha256(encoded).hexdigest() == checksum

