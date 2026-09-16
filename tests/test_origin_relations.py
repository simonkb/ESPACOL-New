"""Structural invariants for ORIGIN's cumulative pair-interaction field.

These tests deliberately exercise the public replay surface.  A relation edge
is part of the prediction circuit only if deleting its stored message and
replaying the bounded rate generator exactly reproduces the counterfactual
law without re-encoding the image.
"""

from __future__ import annotations

import pytest
import torch

from models.origin import (
    ConservedOrdinalGenerator,
    aggregate_ordinal_pair_messages,
    bounded_rate_log_odds_merge,
    decode_pure_birth_rates,
    replay_without,
    replay_without_relations,
    reverse_cumulative_atoms,
    sum_local_rate_maps,
)
from models.origin_encoder import (
    OriginEncoderOutput,
    OriginEncoderScale,
    OriginScaleMetadata,
)


def _metadata(
    name: str,
    channels: int,
    height: int,
    width: int,
) -> OriginScaleMetadata:
    stride = {"s4": 4, "s8": 8}[name]
    return OriginScaleMetadata(
        name=name,
        feature_index=0,
        channels=channels,
        output_stride=stride,
        receptive_field={"s4": 7, "s8": 15}[name],
        center_offset=0.5 * stride,
        input_size=(height * stride, width * stride),
        lattice_size=(height, width),
    )


def _encoder_output(
    *,
    batch: int = 2,
    requires_grad: bool = False,
    invalidate_one_relation_region: bool = False,
) -> OriginEncoderOutput:
    torch.manual_seed(617)
    channels = {"s4": 3, "s8": 5}
    scales = {}
    for name, count in channels.items():
        height = width = 4
        features = torch.randn(
            batch,
            count,
            height,
            width,
            requires_grad=requires_grad,
        )
        valid = torch.ones(batch, height, width, dtype=torch.bool)
        if name == "s8" and invalidate_one_relation_region:
            # grid_size=2 makes this exactly the upper-left endpoint block.
            valid[0, :2, :2] = False
        scales[name] = OriginEncoderScale(
            features=features,
            valid_mask=valid,
            metadata=_metadata(name, count, height, width),
        )
    return OriginEncoderOutput(scales)


def _generator(*, relation_enabled: bool = True) -> ConservedOrdinalGenerator:
    return ConservedOrdinalGenerator(
        stage_channels={"s4": 3, "s8": 5},
        num_classes=5,
        hidden_channels=7,
        evidence_scales=("s4", "s8"),
        reference_count=8.0,
        atom_rate_init=0.02,
        prior_rate_init=0.001,
        relation_enabled=relation_enabled,
        relation_source_scale="s8",
        relation_grid_size=2,
        relation_dim=6,
        relation_head_dim=3,
        relation_delta_cap=2.0,
    ).eval()


def _nonzero_relation_output(*, batch: int = 2):
    generator = _generator(relation_enabled=True)
    assert generator.relation_field is not None
    with torch.no_grad():
        # Zero is the function-preserving initialization.  Moving only this
        # final pair-output parameter makes a non-trivial but still bounded
        # relation ledger for replay tests.
        generator.relation_field.trilinear_weight.fill_(0.35)
    return generator(_encoder_output(batch=batch), force_decoder_fp64=True)


def test_zero_initialized_relation_is_exact_v3_function_identity() -> None:
    generator = _generator(relation_enabled=True)
    relation_field = generator.relation_field
    assert relation_field is not None
    encoded = _encoder_output()

    # Evaluate the exact same v3 parameters once with the new module detached
    # from the graph, then restore its zero-initialized identity extension.
    generator.relation_field = None
    baseline = generator(encoded, force_decoder_fp64=True)
    generator.relation_field = relation_field
    related = generator(encoded, force_decoder_fp64=True)

    assert related.relation_evidence is not None
    assert torch.count_nonzero(related.relation_evidence.edge_messages) == 0
    assert torch.count_nonzero(related.relation_evidence.cumulative_log_odds) == 0
    assert torch.equal(related.base_total_rates, baseline.total_rates)
    assert torch.equal(related.total_rates, baseline.total_rates)
    assert torch.equal(related.class_probs, baseline.class_probs)
    assert torch.equal(related.cumulative_probs, baseline.cumulative_probs)


def test_zero_initialized_relation_has_a_live_first_order_gradient() -> None:
    generator = _generator(relation_enabled=True)
    relation_field = generator.relation_field
    assert relation_field is not None
    # Make the query/key vectors deterministic and non-zero.  At zero output
    # weight the forward remains exactly v3, but d(message)/d(weight) is live.
    with torch.no_grad():
        relation_field.query_projection.weight.zero_()
        relation_field.key_projection.weight.zero_()
        relation_field.query_projection.bias.fill_(0.4)
        relation_field.key_projection.bias.fill_(0.3)
        relation_field.relative_bias.zero_()
        relation_field.trilinear_weight.zero_()

    output = generator(_encoder_output(batch=1), force_decoder_fp64=True)
    assert torch.equal(output.total_rates, output.base_total_rates)
    output.total_rates.sum().backward()

    gradient = relation_field.trilinear_weight.grad
    assert gradient is not None
    assert torch.isfinite(gradient).all()
    assert torch.count_nonzero(gradient) == gradient.numel()


def test_relation_trace_shapes_masks_geometry_and_bounds() -> None:
    generator = _generator(relation_enabled=True)
    assert generator.relation_field is not None
    with torch.no_grad():
        generator.relation_field.trilinear_weight.fill_(0.5)
    output = generator(
        _encoder_output(invalidate_one_relation_region=True),
        force_decoder_fp64=True,
    )
    relation = output.relation_evidence
    assert relation is not None

    assert relation.grid_size == (2, 2)
    assert relation.block_size == (2, 2)
    assert relation.region_valid_mask.shape == (2, 4)
    assert relation.edge_valid_mask.shape == (2, 4, 4)
    assert relation.pair_gates.shape == (2, 4, 4, 4)
    assert relation.edge_messages.shape == (2, 4, 4, 4)
    assert relation.target_incremental_log_odds.shape == (2, 4, 4)
    assert relation.incremental_log_odds.shape == (2, 4)
    assert relation.cumulative_log_odds.shape == (2, 4)
    assert relation.region_centers_yx.shape == (4, 2)

    assert not relation.region_valid_mask[0, 0]
    assert relation.region_valid_mask[0, 1:].all()
    diagonal = torch.eye(4, dtype=torch.bool).expand(2, -1, -1)
    assert not relation.edge_valid_mask[diagonal].any()
    invalid_edges = ~relation.edge_valid_mask[:, None].expand_as(
        relation.edge_messages
    )
    assert torch.count_nonzero(relation.pair_gates[invalid_edges]) == 0
    assert torch.count_nonzero(relation.edge_messages[invalid_edges]) == 0

    assert torch.isfinite(relation.edge_messages).all()
    assert torch.all(relation.edge_messages.abs() <= 1.0)
    assert torch.all(
        relation.target_incremental_log_odds.abs() <= relation.delta_cap / 4.0
    )
    assert torch.all(relation.cumulative_log_odds.abs() <= relation.delta_cap)
    assert torch.all(output.total_rates > 0.0)
    assert torch.all(output.total_rates < output.total_rate_cap)


@pytest.mark.parametrize("bad_value", [float("nan"), float("inf")])
def test_relation_field_rejects_nonfinite_source_features(
    bad_value: float,
) -> None:
    generator = _generator(relation_enabled=True)
    encoded = _encoder_output(batch=1)
    encoded.scales["s8"].features[0, 0, 0, 0] = bad_value
    field = generator.relation_field
    assert field is not None

    with pytest.raises(FloatingPointError, match="relation source features"):
        field(encoded.scales["s8"])


@pytest.mark.parametrize(
    ("corrupt", "message"),
    [
        ("token", "relation region tokens"),
        ("query", "relation pair queries"),
        ("affinity", "relation relative-position biases"),
        ("trilinear", "raw relation trilinear messages"),
    ],
)
def test_relation_field_fails_closed_before_pair_nonlinearities(
    corrupt: str,
    message: str,
) -> None:
    generator = _generator(relation_enabled=True)
    field = generator.relation_field
    assert field is not None
    with torch.no_grad():
        if corrupt == "token":
            field.token_projection.bias[0] = torch.inf
        elif corrupt == "query":
            field.query_projection.bias[0] = torch.nan
        elif corrupt == "affinity":
            field.relative_bias[0, 0, 0] = torch.inf
        elif corrupt == "trilinear":
            field.query_projection.weight.zero_()
            field.key_projection.weight.zero_()
            field.query_projection.bias.fill_(1.0)
            field.key_projection.bias.fill_(1.0)
            field.trilinear_weight[0, 0] = torch.inf
        else:  # pragma: no cover - parametrization invariant
            raise AssertionError(corrupt)

    with pytest.raises(FloatingPointError, match=message):
        generator(_encoder_output(batch=1), force_decoder_fp64=True)


def test_signed_pair_atoms_use_the_declared_reverse_cumulative_mapping() -> None:
    messages = torch.zeros(1, 4, 3, 3)
    messages[0, :, 0, 1] = torch.tensor([0.25, -0.5, 0.75, -1.0])
    messages[0, :, 1, 2] = torch.tensor([-0.2, 0.4, -0.6, 0.8])
    region_valid = torch.tensor([[True, True, True]])
    edge_valid = ~torch.eye(3, dtype=torch.bool).unsqueeze(0)

    target_incremental, incremental, cumulative = (
        aggregate_ordinal_pair_messages(
            messages,
            edge_valid,
            region_valid,
            delta_cap=2.0,
        )
    )

    torch.testing.assert_close(
        cumulative,
        reverse_cumulative_atoms(incremental, dim=1),
        atol=0.0,
        rtol=0.0,
    )
    torch.testing.assert_close(
        cumulative[:, :-1] - cumulative[:, 1:],
        incremental[:, :-1],
        atol=2e-7,
        rtol=2e-7,
    )
    torch.testing.assert_close(
        cumulative[:, -1], incremental[:, -1], atol=0.0, rtol=0.0
    )
    assert torch.all(target_incremental.abs() <= 0.5)
    assert torch.all(cumulative.abs() <= 2.0)


def test_bounded_rate_merge_has_exact_zero_identity_and_open_interval_cap() -> None:
    cap = 64.0
    base = torch.tensor(
        [[1e-6, 0.25, 31.5, cap - 1e-6]],
        dtype=torch.float64,
    )
    identity = bounded_rate_log_odds_merge(
        base, torch.zeros_like(base), total_rate_cap=cap
    )
    assert torch.equal(identity, base)

    correction = torch.tensor(
        [[-2.0, 2.0, -2.0, 2.0]],
        dtype=torch.float64,
        requires_grad=True,
    )
    merged = bounded_rate_log_odds_merge(
        base, correction, total_rate_cap=cap
    )
    assert torch.isfinite(merged).all()
    assert torch.all(merged > 0.0)
    assert torch.all(merged < cap)
    merged.sum().backward()
    assert correction.grad is not None
    assert torch.isfinite(correction.grad).all()
    assert torch.all(correction.grad > 0.0)


def test_relation_edge_deletion_is_an_exact_stored_trace_replay() -> None:
    baseline = _nonzero_relation_output(batch=2)
    relation = baseline.relation_evidence
    assert relation is not None
    removal = torch.zeros_like(relation.edge_messages, dtype=torch.bool)
    removal[:, 1, 0, 1] = True
    removal[:, 3, 2, 3] = True

    intervention = replay_without_relations(
        baseline, removal, force_decoder_fp64=True
    )
    replayed = intervention.output.relation_evidence
    assert replayed is not None
    canonical = removal & relation.edge_valid_mask[:, None]
    expected_edges = torch.where(
        canonical,
        torch.zeros_like(relation.edge_messages),
        relation.edge_messages,
    )
    assert torch.equal(intervention.removal_mask, canonical)
    assert torch.equal(replayed.edge_messages, expected_edges)

    target, incremental, cumulative = aggregate_ordinal_pair_messages(
        expected_edges,
        relation.edge_valid_mask,
        relation.region_valid_mask,
        delta_cap=relation.delta_cap,
    )
    assert torch.equal(replayed.target_incremental_log_odds, target)
    assert torch.equal(replayed.incremental_log_odds, incremental)
    assert torch.equal(replayed.cumulative_log_odds, cumulative)
    expected_rates = bounded_rate_log_odds_merge(
        baseline.base_total_rates,
        cumulative,
        total_rate_cap=baseline.total_rate_cap,
    )
    assert torch.equal(intervention.output.total_rates, expected_rates)
    expected_law = decode_pure_birth_rates(expected_rates, force_fp64=True)
    assert torch.equal(intervention.output.class_probs, expected_law.class_probs)


def test_relation_removal_masks_require_explicit_boundary_axis() -> None:
    baseline = _nonzero_relation_output(batch=2)
    relation = baseline.relation_evidence
    assert relation is not None
    boundaries = relation.edge_messages.shape[1]
    regions = relation.num_regions

    # This formerly ambiguous shape could mean either B boundary masks or N
    # sample masks.  Because B != N here it must fail instead of guessing.
    ambiguous_boundary_mask = torch.zeros(
        boundaries, regions, regions, dtype=torch.bool
    )
    with pytest.raises(ValueError, match="explicitly four-dimensional"):
        replay_without_relations(baseline, ambiguous_boundary_mask)

    explicit_boundary_mask = torch.zeros(
        1, boundaries, regions, regions, dtype=torch.bool
    )
    explicit_boundary_mask[:, 2, 0, 1] = True
    intervention = replay_without_relations(baseline, explicit_boundary_mask)
    removed_per_boundary = intervention.removal_mask.sum(dim=(0, 2, 3))
    assert removed_per_boundary[2] == baseline.total_rates.shape[0]
    assert torch.count_nonzero(removed_per_boundary[:2]) == 0
    assert torch.count_nonzero(removed_per_boundary[3:]) == 0

    # Three dimensions now have one documented meaning: one endpoint mask per
    # sample, broadcast across all boundaries.
    sample_mask = torch.zeros(2, regions, regions, dtype=torch.bool)
    sample_mask[:, 0, 1] = True
    sample_intervention = replay_without_relations(baseline, sample_mask)
    assert sample_intervention.removal_mask[:, :, 0, 1].all()


def test_removing_all_relations_recovers_v3_and_removing_none_is_identity() -> None:
    baseline = _nonzero_relation_output(batch=1)
    relation = baseline.relation_evidence
    assert relation is not None

    no_removal = replay_without_relations(
        baseline,
        torch.zeros(relation.num_regions, relation.num_regions, dtype=torch.bool),
    )
    assert torch.equal(no_removal.output.total_rates, baseline.total_rates)
    assert torch.equal(no_removal.output.class_probs, baseline.class_probs)

    all_removal = replay_without_relations(
        baseline,
        torch.ones(relation.num_regions, relation.num_regions, dtype=torch.bool),
    )
    assert torch.count_nonzero(
        all_removal.output.relation_evidence.edge_messages
    ) == 0
    assert torch.equal(all_removal.output.total_rates, baseline.base_total_rates)
    v3_law = decode_pure_birth_rates(baseline.base_total_rates)
    assert torch.equal(all_removal.output.class_probs, v3_law.class_probs)


def test_local_deletion_replays_with_the_stored_relation_field_unchanged() -> None:
    baseline = _nonzero_relation_output(batch=1)
    relation = baseline.relation_evidence
    assert relation is not None
    removal = torch.zeros_like(baseline.valid_masks["s4"])
    removal[:, 0, 0] = True

    intervention = replay_without(baseline, {"s4": removal})
    assert intervention.output.relation_evidence is relation
    expected_base = sum_local_rate_maps(
        {
            name: evidence.local_rate_map
            for name, evidence in intervention.output.scale_evidence.items()
        },
        baseline.prior_rates,
    )
    expected_rates = bounded_rate_log_odds_merge(
        expected_base,
        relation.cumulative_log_odds,
        total_rate_cap=baseline.total_rate_cap,
    )
    assert torch.equal(intervention.output.base_total_rates, expected_base)
    assert torch.equal(intervention.output.total_rates, expected_rates)
