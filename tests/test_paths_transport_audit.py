from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import torch

from models.paths import apply_signed_adjacent_transport

from audit_paths_transport import (
    _canonical_sha256,
    _load_bound_run,
    first_target_multiplier,
    parse_multiplier_grid,
    rescale_adjacent_transport,
    target_margin,
)


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_checkpoint_binding_accepts_only_json_roundtrip_type_changes(
    tmp_path: Path,
) -> None:
    config = {"dataset": "aptos", "evidence_scales": ("s4", "s8")}
    metrics = {"acc": 80.0, "confusion": ((1, 0), (0, 1))}
    provenance = {"source_checkpoint_sha256": "b" * 64}
    state = {
        "schema": "paths-checkpoint-v3-sapt",
        "paths_protocol_version": "paths-v3-sapt",
        "paths_variant": "signed_transport",
        "run_git_commit": "a" * 40,
        "split_signature": "split",
        "implementation_signature": "implementation",
        "architecture_signature": "architecture",
        "config_signature": _canonical_sha256(config),
        "critical_config": config,
        "metrics": metrics,
        "epoch": 3,
        "fold": 0,
        "checkpoint_role": "paths_selected_learned",
        "paths_warm_start_provenance": provenance,
    }
    checkpoint = tmp_path / "best.pth"
    torch.save(state, checkpoint)
    checkpoint_sha = _file_sha256(checkpoint)
    result = {
        "protocol": "paths-v3-sapt",
        "paths_variant": "signed_transport",
        "run_git_commit": "a" * 40,
        "split_signature": "split",
        "implementation_signature": "implementation",
        "architecture_signature": "architecture",
        "config_signature": _canonical_sha256(config),
        "critical_config": {"dataset": "aptos", "evidence_scales": ["s4", "s8"]},
        "best_learned_validation": {"acc": 80.0, "confusion": [[1, 0], [0, 1]]},
        "best_epoch": 3,
        "best_learned_epoch": 3,
        "fold": 0,
        "best_checkpoint_sha256": checkpoint_sha,
        "best_learned_checkpoint_sha256": checkpoint_sha,
        "best_learned_checkpoint_role": "paths_selected_learned",
        "best_learned_is_byte_exact_best_alias": True,
        "warm_start_provenance": provenance,
    }
    (tmp_path / "result.json").write_text(json.dumps(result))
    observed, _, _, observed_path = _load_bound_run(
        tmp_path, expected_variant="signed_transport"
    )
    assert observed["critical_config"] == config
    assert observed_path == checkpoint

    result["critical_config"]["dataset"] = "dr"
    (tmp_path / "result.json").write_text(json.dumps(result))
    with pytest.raises(ValueError, match="configuration mismatch"):
        _load_bound_run(tmp_path, expected_variant="signed_transport")


def test_multiplier_grid_is_registered_and_ordered() -> None:
    assert parse_multiplier_grid("0,0.5,1,2") == (0.0, 0.5, 1.0, 2.0)
    with pytest.raises(ValueError):
        parse_multiplier_grid((0.0, 2.0))
    with pytest.raises(ValueError):
        parse_multiplier_grid((0.0, 1.0, 1.0))
    with pytest.raises(ValueError):
        parse_multiplier_grid((0.0, 1.0, -2.0))


def test_rescaled_transport_has_exact_endpoints_and_conserves_mass() -> None:
    base = torch.tensor([[0.55, 0.30, 0.10, 0.04, 0.01]], dtype=torch.float64)
    upward = torch.tensor([[0.1, 0.2, 0.3, 0.4]], dtype=torch.float64)
    downward = torch.tensor([[0.4, 0.3, 0.2, 0.1]], dtype=torch.float64)
    zero, zero_matrix = rescale_adjacent_transport(base, upward, downward, 0.0)
    one, one_matrix = rescale_adjacent_transport(base, upward, downward, 1.0)
    large, large_matrix = rescale_adjacent_transport(base, upward, downward, 64.0)
    assert torch.equal(zero, base)
    assert torch.equal(zero_matrix, torch.eye(5, dtype=torch.float64).unsqueeze(0))
    assert not torch.equal(one, base)
    for posterior, matrix in ((one, one_matrix), (large, large_matrix)):
        assert torch.allclose(posterior.sum(-1), torch.ones(1, dtype=torch.float64))
        assert torch.allclose(matrix.sum(-1), torch.ones(1, 5, dtype=torch.float64))
        assert bool((posterior >= 0).all())
        assert bool((matrix >= 0).all())
        axis = torch.arange(5)
        assert torch.equal(
            matrix[..., (axis[:, None] - axis[None, :]).abs() > 1],
            torch.zeros(1, 12, dtype=torch.float64),
        )


def test_multiplier_one_exactly_replays_production_transport() -> None:
    base = torch.tensor([[0.55, 0.30, 0.10, 0.04, 0.01]], dtype=torch.float64)
    concentration = torch.tensor([[0.20, 0.40, 0.60, 0.80]], dtype=torch.float64)
    thresholds = torch.tensor([0.30, 0.50, 0.50, 0.70], dtype=torch.float64)
    slopes = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float64)
    gains = torch.tensor([0.10, 0.20, 0.30, 0.40], dtype=torch.float64)
    production = apply_signed_adjacent_transport(
        base,
        concentration,
        thresholds,
        slopes,
        gains,
        strength=1.0,
    )
    replay, matrix = rescale_adjacent_transport(
        base,
        production.upward_odds,
        production.downward_odds,
        1.0,
    )
    torch.testing.assert_close(replay, production.class_probs, rtol=0.0, atol=0.0)
    torch.testing.assert_close(
        matrix, production.transport_matrix, rtol=0.0, atol=0.0
    )


def test_first_target_multiplier_finds_grade_three_crossing() -> None:
    # Grade 4 initially wins. Downward transport across boundary 3 moves its
    # mass into grade 3, which becomes MAP at a finite multiplier.
    base = torch.tensor([[0.01, 0.02, 0.07, 0.30, 0.60]], dtype=torch.float64)
    upward = torch.zeros(1, 4, dtype=torch.float64)
    downward = torch.tensor([[0.0, 0.0, 0.0, 1.0]], dtype=torch.float64)
    grid = parse_multiplier_grid((0.0, 0.25, 0.5, 1.0, 2.0, 4.0))
    crossing = first_target_multiplier(
        base,
        upward,
        downward,
        target_grade=3,
        search_grid=grid,
    )
    assert crossing is not None
    # 0.30 + 0.60 m/(1+m) = 0.60/(1+m), so m = 1/3.
    assert (1.0 / 3.0) - 1e-10 < crossing < (1.0 / 3.0) + 1e-10
    before, _ = rescale_adjacent_transport(base, upward, downward, crossing * 0.999)
    after, _ = rescale_adjacent_transport(base, upward, downward, crossing * 1.001)
    assert float(target_margin(before, 3)) < 0.0
    assert float(target_margin(after, 3)) > 0.0


def test_first_target_multiplier_reports_no_observed_crossing() -> None:
    base = torch.tensor([[0.01, 0.02, 0.07, 0.30, 0.60]], dtype=torch.float64)
    # Upward-only flow cannot pull grade-4 mass back to grade 3.
    upward = torch.ones(1, 4, dtype=torch.float64)
    downward = torch.zeros(1, 4, dtype=torch.float64)
    assert first_target_multiplier(
        base,
        upward,
        downward,
        target_grade=3,
        search_grid=(0.0, 1.0, 4.0, 16.0),
    ) is None
