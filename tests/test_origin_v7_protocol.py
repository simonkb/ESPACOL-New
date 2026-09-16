"""Protocol boundaries for the ORIGIN-v7 sparse relation experiment."""

from __future__ import annotations

from dataclasses import replace

import pytest

from configs.origin_config import OriginConfig
from train_origin import build_parser
from training.origin_trainer import _relation_protocol_version


_V7_VARIANTS = (
    "identified_sparse_v1",
    "additive_endpoint_control_v1",
    "shuffled_pair_control_v1",
)


def test_relation_protocol_version_preserves_v3_and_v6() -> None:
    assert _relation_protocol_version(OriginConfig()) == "v3"
    assert _relation_protocol_version(OriginConfig(relation_enabled=True)) == "v6"


@pytest.mark.parametrize("variant", _V7_VARIANTS)
def test_every_sparse_target_or_control_has_v7_checkpoint_identity(variant: str) -> None:
    cfg = OriginConfig(relation_enabled=True, relation_variant=variant)
    assert _relation_protocol_version(cfg) == "v7"


@pytest.mark.parametrize("variant", _V7_VARIANTS)
def test_v7_variants_cannot_be_silently_disabled(variant: str) -> None:
    with pytest.raises(ValueError, match="requires relation_enabled=True"):
        OriginConfig(relation_enabled=False, relation_variant=variant)


def test_v7_edge_budget_is_bounded_by_the_directed_nonself_graph() -> None:
    valid = OriginConfig(
        relation_enabled=True,
        relation_variant="identified_sparse_v1",
        relation_grid_size=2,
        relation_edge_budget=11,
    )
    assert valid.relation_edge_budget == 11
    with pytest.raises(ValueError, match=r"\[1, R\(R-1\)\]"):
        replace(valid, relation_edge_budget=0)
    with pytest.raises(ValueError, match=r"\[1, R\(R-1\)\]"):
        replace(valid, relation_edge_budget=13)
    with pytest.raises(ValueError, match="strictly below"):
        replace(valid, relation_edge_budget=12)


def test_v7_permutation_seed_is_registered_even_for_the_target() -> None:
    cfg = OriginConfig(
        relation_enabled=True,
        relation_variant="identified_sparse_v1",
    )
    assert cfg.relation_permutation_seed == 617
    with pytest.raises(ValueError, match="must be non-negative"):
        replace(cfg, relation_permutation_seed=-1)


def test_cli_exposes_the_complete_registered_v7_identity() -> None:
    args = build_parser().parse_args(
        [
            "--relation_enabled",
            "--relation_variant",
            "identified_sparse_v1",
            "--relation_edge_budget",
            "8",
            "--relation_permutation_seed",
            "617",
        ]
    )
    assert args.relation_enabled is True
    assert args.relation_variant == "identified_sparse_v1"
    assert args.relation_edge_budget == 8
    assert args.relation_permutation_seed == 617
