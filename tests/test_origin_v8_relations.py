"""Structural invariants for ORIGIN-v8 conserved sparse witnesses."""

from __future__ import annotations

import torch

from models.origin import (
    ConservedOrdinalGenerator,
    capped_entmax_witness_allocation,
    compile_conserved_witness_ledger,
    replay_without_relations,
    reverse_cumulative_atoms,
)
from models.origin_encoder import (
    OriginEncoderOutput,
    OriginEncoderScale,
    OriginScaleMetadata,
)


_V8_VARIANTS = (
    "identified_conserved_witness_v1",
    "additive_conserved_witness_control_v1",
    "shuffled_conserved_witness_control_v1",
)


def _encoded(*, batch: int = 2, constant: bool = False) -> OriginEncoderOutput:
    torch.manual_seed(808)
    features = torch.randn(batch, 5, 6, 6, requires_grad=True)
    if constant:
        features = torch.full_like(features, 2.5, requires_grad=True)
    valid = torch.ones(batch, 6, 6, dtype=torch.bool)
    return OriginEncoderOutput(
        {
            "s8": OriginEncoderScale(
                features=features,
                valid_mask=valid,
                metadata=OriginScaleMetadata(
                    name="s8",
                    feature_index=0,
                    channels=5,
                    output_stride=8,
                    receptive_field=15,
                    center_offset=4.0,
                    input_size=(48, 48),
                    lattice_size=(6, 6),
                ),
            )
        }
    )


def _generator(
    variant: str = "identified_conserved_witness_v1",
) -> ConservedOrdinalGenerator:
    return ConservedOrdinalGenerator(
        stage_channels={"s8": 5},
        num_classes=5,
        hidden_channels=7,
        evidence_scales=("s8",),
        reference_count=8.0,
        atom_rate_init=0.02,
        prior_rate_init=0.001,
        relation_enabled=True,
        relation_source_scale="s8",
        relation_grid_size=3,
        relation_dim=6,
        relation_head_dim=3,
        relation_delta_cap=2.0,
        relation_variant=variant,
        relation_edge_budget=5,
        relation_allocation_temperature=0.5,
        relation_permutation_seed=617,
    ).eval()


def test_capped_entmax_allocation_is_sparse_normalized_and_deterministic() -> None:
    scores = torch.zeros(1, 2, 4, 4, requires_grad=True)
    scores.data[0, 0, 0, 1] = 20.0
    valid = ~torch.eye(4, dtype=torch.bool)[None]
    first = capped_entmax_witness_allocation(
        scores,
        valid,
        max_witnesses=3,
        temperature=1.0,
    )
    second = capped_entmax_witness_allocation(
        scores,
        valid,
        max_witnesses=3,
        temperature=1.0,
    )
    shortlist, allocation, active = first
    assert all(torch.equal(left, right) for left, right in zip(first, second))
    assert torch.equal(shortlist.flatten(2).sum(-1), torch.full((1, 2), 3))
    assert torch.all(active.flatten(2).sum(-1).ge(1))
    assert torch.all(active.flatten(2).sum(-1).le(3))
    assert not bool((active & ~shortlist).any())
    assert not bool((shortlist & ~valid[:, None]).any())
    assert torch.count_nonzero(allocation.masked_select(~shortlist)) == 0
    torch.testing.assert_close(
        allocation.sum(dim=(-2, -1)),
        torch.ones(1, 2),
        atol=1e-6,
        rtol=1e-6,
    )
    # The dominant edge receives the complete boundary budget, demonstrating
    # that V8 is not constrained to V7's uniform one-third allocation.
    torch.testing.assert_close(allocation[0, 0, 0, 1], torch.tensor(1.0))
    (allocation * scores).sum().backward()
    assert scores.grad is not None and torch.isfinite(scores.grad).all()


def test_conserved_compiler_is_literal_and_respects_ordinal_capacity() -> None:
    torch.manual_seed(81)
    scores = torch.randn(2, 4, 5, 5)
    valid = ~torch.eye(5, dtype=torch.bool)[None].expand(2, -1, -1)
    _, allocation, active = capped_entmax_witness_allocation(
        scores, valid, max_witnesses=4, temperature=0.4
    )
    strength = torch.tensor([0.8, -0.5, 0.2, 1.0])
    contributions, target, incremental, cumulative = (
        compile_conserved_witness_ledger(
            scores,
            allocation,
            active,
            valid,
            strength,
            delta_cap=2.0,
        )
    )
    expected = (
        allocation.double()
        * scores.tanh().double()
        * strength.reshape(1, 4, 1, 1).double()
        * (2.0 / 4.0)
    )
    torch.testing.assert_close(contributions, expected, atol=0.0, rtol=0.0)
    torch.testing.assert_close(target, contributions.sum(-1), atol=0.0, rtol=0.0)
    torch.testing.assert_close(incremental, target.sum(-1), atol=0.0, rtol=0.0)
    torch.testing.assert_close(
        cumulative,
        reverse_cumulative_atoms(incremental, dim=1),
        atol=0.0,
        rtol=0.0,
    )
    l1 = contributions.abs().sum(dim=(-2, -1))
    assert torch.all(l1 <= strength.abs().double()[None] * 0.5 + 2e-12)
    assert torch.all(cumulative.abs() <= 2.0 + 2e-12)


def test_v8_zero_gate_is_exact_v3_and_opens_a_live_single_score_path() -> None:
    generator = _generator()
    field = generator.relation_field
    assert field is not None and field.raw_relation_strength is not None
    encoded = _encoded(batch=1)

    generator.relation_field = None
    baseline = generator(encoded, force_decoder_fp64=True)
    generator.relation_field = field
    related = generator(encoded, force_decoder_fp64=True)
    relation = related.relation_evidence
    assert relation is not None
    assert relation.aggregation_kind == "conserved_sparse_witness_allocation_v1"
    assert torch.equal(related.total_rates, baseline.total_rates)
    assert torch.equal(related.class_probs, baseline.class_probs)
    assert torch.count_nonzero(relation.edge_log_odds_contributions) == 0

    related.total_rates.sum().backward()
    gate_gradient = field.raw_relation_strength.grad
    assert gate_gradient is not None and torch.isfinite(gate_gradient).all()
    assert torch.count_nonzero(gate_gradient) > 0

    generator.zero_grad(set_to_none=True)
    with torch.no_grad():
        field.raw_relation_strength.fill_(0.4)
    opened = generator(_encoded(batch=1), force_decoder_fp64=True)
    opened.total_rates.sum().backward()
    for parameter in (
        field.query_projection.weight,
        field.key_projection.weight,
        field.trilinear_weight,
    ):
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
        assert torch.count_nonzero(parameter.grad) > 0


def test_v8_one_identified_score_drives_all_forward_decisions() -> None:
    generator = _generator()
    field = generator.relation_field
    assert field is not None and field.raw_relation_strength is not None
    with torch.no_grad():
        field.raw_relation_strength.fill_(0.5)
    output = generator(_encoded(batch=1), force_decoder_fp64=True)
    relation = output.relation_evidence
    assert relation is not None
    assert relation.identified_pair_scores is relation.interaction_residuals
    assert relation.selected_edge_mask is relation.active_witness_mask
    assert relation.shortlist_edge_mask is not None
    assert relation.witness_allocation is not None
    shortlist, allocation, active = capped_entmax_witness_allocation(
        relation.identified_pair_scores,
        relation.edge_valid_mask,
        max_witnesses=relation.edge_budget,
        temperature=relation.allocation_temperature,
    )
    assert torch.equal(shortlist, relation.shortlist_edge_mask)
    assert torch.equal(active, relation.active_witness_mask)
    torch.testing.assert_close(allocation, relation.witness_allocation)
    expected = (
        relation.witness_allocation.double()
        * relation.identified_pair_scores.tanh().double()
        * relation.relation_strength.reshape(1, 4, 1, 1).double()
        * (relation.delta_cap / 4.0)
    )
    torch.testing.assert_close(
        relation.edge_log_odds_contributions, expected, atol=0.0, rtol=0.0
    )


def test_v8_deletion_never_reselects_or_reallocates_and_all_recovers_v3() -> None:
    generator = _generator()
    field = generator.relation_field
    assert field is not None and field.raw_relation_strength is not None
    with torch.no_grad():
        field.raw_relation_strength.fill_(0.6)
    baseline = generator(_encoded(batch=1), force_decoder_fp64=True)
    relation = baseline.relation_evidence
    assert relation is not None and relation.active_witness_mask is not None
    active = relation.active_witness_mask
    one = torch.zeros_like(active)
    index = torch.nonzero(active, as_tuple=False)[0]
    one[tuple(index)] = True
    replayed = replay_without_relations(baseline, one).output.relation_evidence
    assert replayed is not None
    assert torch.equal(replayed.active_witness_mask, relation.active_witness_mask)
    assert torch.equal(replayed.shortlist_edge_mask, relation.shortlist_edge_mask)
    assert torch.equal(replayed.witness_allocation, relation.witness_allocation)
    expected = torch.where(
        one,
        torch.zeros_like(relation.edge_log_odds_contributions),
        relation.edge_log_odds_contributions,
    )
    assert torch.equal(replayed.edge_log_odds_contributions, expected)

    removed_all = replay_without_relations(baseline, active)
    assert torch.equal(removed_all.output.total_rates, baseline.base_total_rates)
    assert torch.equal(
        removed_all.output.relation_evidence.witness_allocation,
        relation.witness_allocation,
    )


def test_v8_constant_endpoint_field_emits_zero_correction() -> None:
    generator = _generator()
    field = generator.relation_field
    assert field is not None and field.raw_relation_strength is not None
    with torch.no_grad():
        field.raw_relation_strength.fill_(0.9)
        field.relative_bias.normal_()
    output = generator(_encoded(batch=1, constant=True), force_decoder_fp64=True)
    relation = output.relation_evidence
    assert relation is not None
    assert torch.count_nonzero(relation.identified_pair_scores) == 0
    assert torch.count_nonzero(relation.edge_log_odds_contributions) == 0
    assert torch.equal(output.total_rates, output.base_total_rates)


def test_v8_target_and_controls_are_parameter_matched() -> None:
    signatures = []
    for variant in _V8_VARIANTS:
        generator = _generator(variant)
        signatures.append(
            tuple(
                (name, tuple(parameter.shape), parameter.numel())
                for name, parameter in generator.named_parameters()
                if name.startswith("relation_field.")
            )
        )
    assert signatures[0] == signatures[1] == signatures[2]
