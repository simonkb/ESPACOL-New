"""Exact ledger-removal tests for ORIGIN's intrinsic explanations."""

import torch

from models.origin import (
    ConservedOrdinalGenerator,
    decode_pure_birth_rates,
    replay_without,
    topk_rate_intervention,
)
from models.origin_encoder import (
    OriginEncoderOutput,
    OriginEncoderScale,
    OriginScaleMetadata,
)


def _metadata(name: str, channels: int, height: int, width: int) -> OriginScaleMetadata:
    return OriginScaleMetadata(
        name=name,
        feature_index=0,
        channels=channels,
        output_stride=4,
        receptive_field=7,
        center_offset=2.0,
        input_size=(height * 4, width * 4),
        lattice_size=(height, width),
    )


def _output(batch: int = 2):
    torch.manual_seed(19)
    scales = {}
    channels = {"s4": 3, "s8": 5}
    shapes = {"s4": (3, 4), "s8": (2, 2)}
    for name in ("s4", "s8"):
        height, width = shapes[name]
        feature = torch.randn(batch, channels[name], height, width)
        valid = torch.ones(batch, height, width, dtype=torch.bool)
        valid[0, -1, -1] = False
        scales[name] = OriginEncoderScale(
            feature, valid, _metadata(name, channels[name], height, width)
        )
    generator = ConservedOrdinalGenerator(
        stage_channels=channels,
        num_classes=5,
        hidden_channels=7,
        evidence_scales=("s4", "s8"),
        reference_count=8.0,
        atom_rate_init=0.02,
        prior_rate_init=0.001,
    ).eval()
    return generator(OriginEncoderOutput(scales), force_decoder_fp32=False)


def test_exact_deletion_matches_direct_rate_subtraction_and_decoder() -> None:
    baseline = _output()
    removals = {
        "s4": torch.zeros_like(baseline.valid_masks["s4"]),
        "s8": torch.zeros_like(baseline.valid_masks["s8"]),
    }
    removals["s4"][:, 0, 1] = True
    removals["s8"][:, 1, 0] = True
    intervention = replay_without(
        baseline, removals, force_decoder_fp32=False
    )

    expected_rates = baseline.total_rates - intervention.removed_rates
    torch.testing.assert_close(intervention.output.total_rates, expected_rates)
    direct = decode_pure_birth_rates(expected_rates, force_fp32=False)
    torch.testing.assert_close(intervention.output.class_probs, direct.class_probs)
    torch.testing.assert_close(
        intervention.output.cumulative_probs, direct.cumulative_probs
    )


def test_replay_never_renormalizes_original_geometry_weights() -> None:
    baseline = _output(batch=1)
    original_weights = {
        name: evidence.geometry_weight.clone()
        for name, evidence in baseline.scale_evidence.items()
    }
    remove = torch.zeros_like(baseline.valid_masks["s4"])
    remove[:, :, :2] = True
    intervention = replay_without(baseline, {"s4": remove})

    for name, evidence in intervention.output.scale_evidence.items():
        assert torch.equal(evidence.geometry_weight, original_weights[name])
    kept_map = intervention.output.scale_evidence["s4"].local_rate_map
    assert torch.equal(
        kept_map.permute(0, 2, 3, 1)[remove],
        torch.zeros_like(kept_map.permute(0, 2, 3, 1)[remove]),
    )


def test_invalid_cell_removal_is_exact_noop() -> None:
    baseline = _output(batch=1)
    invalid = ~baseline.valid_masks["s4"]
    intervention = replay_without(baseline, {"s4": invalid})
    assert torch.equal(intervention.removed_rates, torch.zeros_like(intervention.removed_rates))
    assert torch.equal(intervention.output.total_rates, baseline.total_rates)
    assert torch.equal(intervention.output.class_probs, baseline.class_probs)


def test_removing_every_valid_cell_leaves_only_visible_prior() -> None:
    baseline = _output()
    removals = {name: mask.clone() for name, mask in baseline.valid_masks.items()}
    intervention = replay_without(
        baseline, removals, force_decoder_fp32=False
    )
    torch.testing.assert_close(
        intervention.output.total_rates, baseline.prior_rates
    )
    prior_only = decode_pure_birth_rates(
        baseline.prior_rates, force_fp32=False
    )
    torch.testing.assert_close(
        intervention.output.class_probs, prior_only.class_probs
    )


def test_deleting_positive_evidence_cannot_increase_expected_severity() -> None:
    baseline = _output()
    removals = {}
    for name, valid in baseline.valid_masks.items():
        removals[name] = valid & (torch.rand_like(valid.float()) > 0.5)
    intervention = replay_without(baseline, removals)
    assert torch.all(
        intervention.output.expected_grade <= baseline.expected_grade + 2e-6
    )
    assert torch.all(
        intervention.output.posterior_median <= baseline.posterior_median
    )


def test_replaying_no_removals_reproduces_checkpoint_law() -> None:
    baseline = _output()
    intervention = replay_without(baseline, {})
    torch.testing.assert_close(intervention.output.total_rates, baseline.total_rates)
    torch.testing.assert_close(intervention.output.class_probs, baseline.class_probs)
    torch.testing.assert_close(
        intervention.output.cumulative_probs, baseline.cumulative_probs
    )


def test_topk_report_selects_valid_cells_and_matches_ordinary_replay() -> None:
    baseline = _output()
    report = topk_rate_intervention(
        baseline, boundary=2, k=3, force_decoder_fp32=False
    )
    assert report.selected_flat_indices.shape == (2, 3)
    assert report.selected_boundary_rates.shape == (2, 3)
    assert torch.all(
        report.selected_boundary_rates[:, :-1]
        >= report.selected_boundary_rates[:, 1:]
    )
    assert all(
        torch.all(mask <= baseline.valid_masks[name])
        for name, mask in report.intervention.removal_masks.items()
    )
    selected_count = sum(
        mask.sum(dim=(-2, -1))
        for mask in report.intervention.removal_masks.values()
    )
    assert torch.equal(selected_count, torch.full_like(selected_count, 3))
    replayed = replay_without(
        baseline,
        report.intervention.removal_masks,
        force_decoder_fp32=False,
    )
    torch.testing.assert_close(
        report.intervention.output.total_rates, replayed.output.total_rates
    )
    torch.testing.assert_close(
        report.intervention.output.class_probs, replayed.output.class_probs
    )


def test_repeated_replay_is_conservative_on_the_surviving_ledger() -> None:
    baseline = _output(batch=1)
    first_mask = torch.zeros_like(baseline.valid_masks["s4"])
    first_mask[:, 0, 0] = True
    first = replay_without(baseline, {"s4": first_mask}).output
    second_mask = torch.zeros_like(first.valid_masks["s8"])
    second_mask[:, 0, 0] = True
    second = replay_without(first, {"s8": second_mask})

    combined = replay_without(
        baseline, {"s4": first_mask, "s8": second_mask}
    )
    torch.testing.assert_close(second.output.total_rates, combined.output.total_rates)
    torch.testing.assert_close(second.output.class_probs, combined.output.class_probs)


def test_intervention_accepts_singleton_and_channel_masks() -> None:
    baseline = _output(batch=2)
    height, width = baseline.valid_masks["s8"].shape[-2:]
    singleton = torch.zeros(1, 1, height, width, dtype=torch.bool)
    singleton[:, :, 0, 0] = True
    intervention = replay_without(baseline, {"s8": singleton})
    assert intervention.removal_masks["s8"].shape == (2, height, width)
    assert torch.all(intervention.removal_masks["s8"][:, 0, 0])
