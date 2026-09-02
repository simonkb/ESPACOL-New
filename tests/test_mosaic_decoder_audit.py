"""Regression tests for checkpoint decision-rule provenance in the audit."""

import pytest

from tools.audit_mosaic_decoders import _checkpoint_decision_rule


def test_legacy_checkpoint_uses_historical_rounded_expected_rule() -> None:
    assert _checkpoint_decision_rule({}) == "rounded_expected"


def test_explicit_checkpoint_decision_rule_is_preserved() -> None:
    assert (
        _checkpoint_decision_rule({"decision_rule": "deweighted_class_map"})
        == "deweighted_class_map"
    )


def test_unknown_checkpoint_decision_rule_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown MOSAIC decision rule"):
        _checkpoint_decision_rule({"decision_rule": "validation_tuned"})
