"""Focused contract tests for the MOSAIC certificate exporter."""

from __future__ import annotations

import unittest

from export_mosaic_certificates import (
    _require_matching_implementation_signature,
    _verification_fields,
)


class MosaicCertificateExportTests(unittest.TestCase):
    def test_checkpoint_implementation_signature_must_match_active_code(self) -> None:
        signature = "a" * 64
        self.assertEqual(
            _require_matching_implementation_signature(
                {"implementation_signature": signature},
                active_signature=signature,
            ),
            signature,
        )
        with self.assertRaisesRegex(ValueError, "does not match"):
            _require_matching_implementation_signature(
                {"implementation_signature": "b" * 64},
                active_signature=signature,
            )
        with self.assertRaisesRegex(ValueError, "has no MOSAIC"):
            _require_matching_implementation_signature({}, active_signature=signature)

    def test_skipped_replay_is_not_reported_as_success(self) -> None:
        self.assertEqual(_verification_fields(None), (None, "not_run"))
        self.assertEqual(_verification_fields({"ok": True}), (True, "passed"))
        self.assertEqual(_verification_fields({"ok": False}), (False, "failed"))


if __name__ == "__main__":
    unittest.main()
