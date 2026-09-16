"""Structural tests for ORIGIN-v7's identified sparse relation proof."""

from __future__ import annotations

import torch

from models.origin import (
    ConservedOrdinalGenerator,
    aggregate_sparse_ordinal_pair_messages,
    masked_offdiagonal_pair_anova,
    replay_without_relations,
    reverse_cumulative_atoms,
)
from models.origin_encoder import (
    OriginEncoderOutput,
    OriginEncoderScale,
    OriginScaleMetadata,
)


def _encoded(*, batch: int = 2, invalidate_corner: bool = False) -> OriginEncoderOutput:
    torch.manual_seed(1701)
    scales = {}
    for name, channels, stride in (("s4", 3, 4), ("s8", 5, 8)):
        features = torch.randn(batch, channels, 6, 6, requires_grad=True)
        valid = torch.ones(batch, 6, 6, dtype=torch.bool)
        if name == "s8" and invalidate_corner:
            # With a 3x3 regional grid this invalidates exactly one 2x2 block.
            valid[0, :2, :2] = False
        scales[name] = OriginEncoderScale(
            features=features,
            valid_mask=valid,
            metadata=OriginScaleMetadata(
                name=name,
                feature_index=0,
                channels=channels,
                output_stride=stride,
                receptive_field=15 if name == "s8" else 7,
                center_offset=0.5 * stride,
                input_size=(6 * stride, 6 * stride),
                lattice_size=(6, 6),
            ),
        )
    return OriginEncoderOutput(scales)


def _generator(variant: str = "identified_sparse_v1") -> ConservedOrdinalGenerator:
    return ConservedOrdinalGenerator(
        stage_channels={"s4": 3, "s8": 5},
        num_classes=5,
        hidden_channels=7,
        evidence_scales=("s4", "s8"),
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
        relation_permutation_seed=617,
    ).eval()


def test_masked_pair_anova_is_an_exact_identifying_projection() -> None:
    torch.manual_seed(19)
    scores = torch.randn(2, 4, 6, 6, dtype=torch.float64)
    region_valid = torch.tensor(
        [[True, True, True, True, True, True],
         [True, True, True, True, False, False]]
    )
    diagonal = torch.eye(6, dtype=torch.bool)
    edge_valid = (
        region_valid[:, :, None]
        & region_valid[:, None, :]
        & ~diagonal[None]
    )
    residual, additive = masked_offdiagonal_pair_anova(
        scores, edge_valid, region_valid
    )
    mask = edge_valid[:, None]

    torch.testing.assert_close(
        residual + additive,
        torch.where(mask, scores, torch.zeros_like(scores)),
        atol=2e-14,
        rtol=2e-14,
    )
    assert float(residual.sum(-1).abs().max()) < 2e-14
    assert float(residual.sum(-2).abs().max()) < 2e-14

    target = torch.randn(2, 4, 6, dtype=torch.float64)
    source = torch.randn(2, 4, 6, dtype=torch.float64)
    intercept = torch.randn(2, 4, 1, 1, dtype=torch.float64)
    nuisance = intercept + target[..., :, None] + source[..., None, :]
    shifted, _ = masked_offdiagonal_pair_anova(
        scores + nuisance, edge_valid, region_valid
    )
    torch.testing.assert_close(shifted, residual, atol=3e-14, rtol=3e-14)
    projected_twice, _ = masked_offdiagonal_pair_anova(
        residual, edge_valid, region_valid
    )
    torch.testing.assert_close(projected_twice, residual, atol=3e-14, rtol=3e-14)


def test_v7_zero_start_is_exact_v3_and_has_a_live_relation_gradient() -> None:
    generator = _generator()
    relation_field = generator.relation_field
    assert relation_field is not None
    encoded = _encoded(batch=1)

    generator.relation_field = None
    baseline = generator(encoded, force_decoder_fp64=True)
    generator.relation_field = relation_field
    related = generator(encoded, force_decoder_fp64=True)

    relation = related.relation_evidence
    assert relation is not None
    assert relation.variant == "identified_sparse_v1"
    assert relation.candidate_edge_mask is not None
    assert torch.equal(relation.candidate_edge_mask, relation.edge_valid_mask)
    assert torch.count_nonzero(relation.edge_messages) == 0
    assert torch.equal(related.total_rates, baseline.total_rates)
    assert torch.equal(related.class_probs, baseline.class_probs)

    related.total_rates.sum().backward()
    gradient = relation_field.trilinear_weight.grad
    assert gradient is not None
    assert torch.isfinite(gradient).all()
    assert torch.count_nonzero(gradient) > 0


def test_v7_constant_endpoint_field_cannot_emit_a_geometry_only_correction() -> None:
    generator = _generator()
    relation_field = generator.relation_field
    assert relation_field is not None
    with torch.no_grad():
        relation_field.trilinear_weight.fill_(0.7)
        relation_field.relative_bias.normal_(mean=0.0, std=1.0)
    encoded = _encoded(batch=1)
    constant = torch.full_like(encoded.scales["s8"].features, 2.5)
    encoded.scales["s8"] = OriginEncoderScale(
        features=constant,
        valid_mask=encoded.scales["s8"].valid_mask,
        metadata=encoded.scales["s8"].metadata,
    )

    output = generator(encoded, force_decoder_fp64=True)
    relation = output.relation_evidence
    assert relation is not None
    assert torch.count_nonzero(relation.raw_proposal_scores) == 0
    assert torch.count_nonzero(relation.raw_interaction_scores) == 0
    assert torch.count_nonzero(relation.edge_messages) == 0
    assert torch.count_nonzero(relation.cumulative_log_odds) == 0
    assert torch.equal(output.total_rates, output.base_total_rates)


def test_v7_relation_is_invariant_to_global_query_key_offsets() -> None:
    generator = _generator()
    relation_field = generator.relation_field
    assert relation_field is not None
    with torch.no_grad():
        relation_field.trilinear_weight.fill_(0.4)
        relation_field.relative_bias.normal_(mean=0.0, std=0.5)
    encoded = _encoded(batch=1)
    before = generator(encoded, force_decoder_fp64=True).relation_evidence
    assert before is not None

    with torch.no_grad():
        relation_field.query_projection.bias.add_(0.37)
        relation_field.key_projection.bias.sub_(0.21)
    after = generator(encoded, force_decoder_fp64=True).relation_evidence
    assert after is not None
    torch.testing.assert_close(
        after.projected_proposal_scores,
        before.projected_proposal_scores,
        atol=2e-6,
        rtol=2e-6,
    )
    torch.testing.assert_close(
        after.interaction_residuals,
        before.interaction_residuals,
        atol=2e-6,
        rtol=2e-6,
    )
    assert torch.equal(after.selected_edge_mask, before.selected_edge_mask)
    torch.testing.assert_close(
        after.cumulative_log_odds,
        before.cumulative_log_odds,
        atol=2e-9,
        rtol=2e-6,
    )


def test_v7_additive_control_has_a_live_zero_start_gradient() -> None:
    generator = _generator("additive_endpoint_control_v1")
    relation_field = generator.relation_field
    assert relation_field is not None
    output = generator(_encoded(batch=1), force_decoder_fp64=True)
    assert torch.equal(output.total_rates, output.base_total_rates)
    output.total_rates.sum().backward()
    gradient = relation_field.trilinear_weight.grad
    assert gradient is not None
    assert torch.isfinite(gradient).all()
    assert torch.count_nonzero(gradient) > 0


def test_v7_fixed_support_is_sparse_ranked_and_conserved() -> None:
    generator = _generator()
    assert generator.relation_field is not None
    with torch.no_grad():
        generator.relation_field.trilinear_weight.fill_(0.35)
    output = generator(
        _encoded(invalidate_corner=True), force_decoder_fp64=True
    )
    relation = output.relation_evidence
    assert relation is not None
    assert relation.selected_edge_mask is not None
    assert relation.edge_log_odds_contributions is not None
    assert relation.projected_proposal_scores is not None
    assert relation.edge_budget == 5

    selected = relation.selected_edge_mask
    candidates = relation.edge_valid_mask[:, None].expand_as(selected)
    assert not bool((selected & ~candidates).any())
    assert torch.equal(selected.flatten(2).sum(-1), torch.full((2, 4), 5))
    assert torch.count_nonzero(relation.edge_messages.masked_select(~selected)) == 0
    assert torch.count_nonzero(
        relation.edge_log_odds_contributions.masked_select(~selected)
    ) == 0

    proposal = relation.projected_proposal_scores.abs()
    selected_floor = proposal.masked_fill(~selected, torch.inf).flatten(2).amin(-1)
    unselected_ceiling = proposal.masked_fill(
        ~(candidates & ~selected), -torch.inf
    ).flatten(2).amax(-1)
    assert torch.all(selected_floor >= unselected_ceiling)

    scale = relation.delta_cap / (4 * relation.edge_budget)
    torch.testing.assert_close(
        relation.edge_log_odds_contributions,
        relation.edge_messages.double() * scale,
        atol=0.0,
        rtol=0.0,
    )
    torch.testing.assert_close(
        relation.incremental_log_odds,
        relation.target_incremental_log_odds.sum(dim=-1),
        atol=0.0,
        rtol=0.0,
    )
    torch.testing.assert_close(
        relation.cumulative_log_odds,
        reverse_cumulative_atoms(relation.incremental_log_odds, dim=1),
        atol=0.0,
        rtol=0.0,
    )
    assert torch.all(relation.cumulative_log_odds.abs() <= relation.delta_cap)


def test_v7_deletion_never_reselects_or_renormalizes() -> None:
    generator = _generator()
    assert generator.relation_field is not None
    with torch.no_grad():
        generator.relation_field.trilinear_weight.fill_(0.25)
    baseline = generator(_encoded(batch=1), force_decoder_fp64=True)
    relation = baseline.relation_evidence
    assert relation is not None
    assert relation.selected_edge_mask is not None
    selected_flat = relation.selected_edge_mask.flatten()
    edge_index = int(torch.nonzero(selected_flat, as_tuple=False)[0])
    removal = torch.zeros_like(selected_flat)
    removal[edge_index] = True
    removal = removal.reshape_as(relation.selected_edge_mask)

    intervention = replay_without_relations(baseline, removal)
    replayed = intervention.output.relation_evidence
    assert replayed is not None
    assert torch.equal(replayed.selected_edge_mask, relation.selected_edge_mask)
    assert intervention.removed_edge_log_odds_contributions is not None
    expected_incremental = (
        relation.incremental_log_odds
        - intervention.removed_edge_log_odds_contributions.sum(dim=(-2, -1))
    )
    torch.testing.assert_close(
        replayed.incremental_log_odds, expected_incremental, atol=0.0, rtol=0.0
    )

    remove_all = replay_without_relations(
        baseline, relation.selected_edge_mask
    )
    assert torch.count_nonzero(
        remove_all.output.relation_evidence.edge_messages
    ) == 0
    assert torch.equal(remove_all.output.total_rates, baseline.base_total_rates)


def test_v7_target_and_controls_are_parameter_matched() -> None:
    variants = (
        "identified_sparse_v1",
        "additive_endpoint_control_v1",
        "shuffled_pair_control_v1",
    )
    signatures = []
    for variant in variants:
        field = _generator(variant).relation_field
        assert field is not None
        signatures.append(
            [(name, tuple(parameter.shape)) for name, parameter in field.named_parameters()]
        )
    assert signatures[0] == signatures[1] == signatures[2]


def test_sparse_compiler_rejects_selected_invalid_edges() -> None:
    messages = torch.zeros(1, 4, 3, 3)
    valid = ~torch.eye(3, dtype=torch.bool)[None]
    selected = torch.zeros_like(messages, dtype=torch.bool)
    selected[:, :, 0, 1] = True
    selected[:, :, 0, 0] = True
    region_valid = torch.ones(1, 3, dtype=torch.bool)
    with torch.no_grad():
        try:
            aggregate_sparse_ordinal_pair_messages(
                messages,
                selected,
                valid,
                region_valid,
                delta_cap=2.0,
                edge_budget=2,
            )
        except ValueError as error:
            assert "invalid edges" in str(error)
        else:  # pragma: no cover
            raise AssertionError("invalid selected support was accepted")
