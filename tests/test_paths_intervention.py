"""Exact joint-ledger intervention tests for PATHS."""

from __future__ import annotations

import torch

from models.origin import ConservedOrdinalGenerator, decode_pure_birth_rates
from models.origin_encoder import (
    OriginEncoderOutput,
    OriginEncoderScale,
    OriginScaleMetadata,
)
from models.paths import (
    PathsContinuationRefiner,
    PathsOutput,
    continuation_logits_from_log_probs,
    decode_continuation_logits,
    replay_paths_without,
)


def _metadata(name: str, channels: int, height: int, width: int) -> OriginScaleMetadata:
    stride = {"s4": 4, "s8": 8}[name]
    return OriginScaleMetadata(
        name=name,
        feature_index=0,
        channels=channels,
        output_stride=stride,
        receptive_field=7,
        center_offset=2.0,
        input_size=(height * stride, width * stride),
        lattice_size=(height, width),
    )


def _output(batch: int = 2) -> PathsOutput:
    torch.manual_seed(211)
    scales = {}
    channels = {"s4": 3, "s8": 5}
    shapes = {"s4": (3, 4), "s8": (2, 2)}
    for name in ("s4", "s8"):
        height, width = shapes[name]
        features = torch.randn(batch, channels[name], height, width)
        valid = torch.ones(batch, height, width, dtype=torch.bool)
        valid[0, -1, -1] = False
        scales[name] = OriginEncoderScale(
            features=features,
            valid_mask=valid,
            metadata=_metadata(name, channels[name], height, width),
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
    base = generator(OriginEncoderOutput(scales), force_decoder_fp64=True)
    output = PathsContinuationRefiner(
        4,
        ("s4", "s8"),
        correction_cap=4.0,
        gain_init=0.05,
    )(base)
    assert isinstance(output, PathsOutput)
    return output


def test_joint_deletion_replays_rate_and_focality_ledgers_exactly() -> None:
    baseline = _output()
    removals = {
        "s4": torch.zeros_like(baseline.valid_masks["s4"]),
        "s8": torch.zeros_like(baseline.valid_masks["s8"]),
    }
    removals["s4"][:, 0, 1] = True
    removals["s8"][:, 1, 0] = True
    intervention = replay_paths_without(baseline, removals)

    torch.testing.assert_close(
        intervention.output.total_rates,
        baseline.total_rates - intervention.removed_rates,
    )
    torch.testing.assert_close(
        intervention.output.boundary_correction,
        baseline.boundary_correction - intervention.removed_boundary_correction,
    )
    removed_local = torch.zeros_like(baseline.boundary_correction)
    for name, mask in intervention.removal_masks.items():
        local = baseline.local_correction_maps[name]
        removed_local += torch.where(
            mask[..., None], local, torch.zeros_like(local)
        ).sum(dim=(1, 2))
    torch.testing.assert_close(
        intervention.removed_boundary_correction, removed_local
    )

    direct_base = decode_pure_birth_rates(intervention.output.total_rates)
    direct_logits = continuation_logits_from_log_probs(direct_base.log_class_probs)
    direct = decode_continuation_logits(
        direct_logits + intervention.output.boundary_correction
    )
    torch.testing.assert_close(intervention.output.class_probs, direct.class_probs)


def test_replay_freezes_baseline_mass_instead_of_renormalizing_survivors() -> None:
    baseline = _output(batch=1)
    evidence = baseline.spectrum_evidence["s4"]
    removal = torch.zeros_like(evidence.active_mask)
    removal[:, :, :2] = True
    replayed = replay_paths_without(baseline, {"s4": removal}).output.spectrum_evidence["s4"]

    torch.testing.assert_close(replayed.baseline_mass, evidence.baseline_mass)
    torch.testing.assert_close(
        replayed.concentration_spectrum,
        replayed.local_concentration_spectrum.sum(dim=(2, 3)),
    )
    surviving_phi = replayed.local_phi_spectrum.sum(dim=(2, 3))
    expected_frozen = torch.where(
        evidence.baseline_mass[:, :, None] > 0,
        surviving_phi / evidence.baseline_mass.clamp_min(1e-300)[:, :, None],
        torch.zeros_like(surviving_phi),
    )
    torch.testing.assert_close(replayed.concentration_spectrum, expected_frozen)

    renormalized = torch.where(
        replayed.surviving_mass[:, :, None] > 0,
        surviving_phi / replayed.surviving_mass.clamp_min(1e-300)[:, :, None],
        torch.zeros_like(surviving_phi),
    )
    assert not torch.allclose(replayed.concentration_spectrum, renormalized)


def test_invalid_cell_deletion_is_an_exact_noop() -> None:
    baseline = _output(batch=1)
    invalid = ~baseline.active_valid_masks["s4"]
    intervention = replay_paths_without(baseline, {"s4": invalid})

    assert torch.equal(intervention.removed_rates, torch.zeros_like(intervention.removed_rates))
    assert torch.equal(
        intervention.removed_boundary_correction,
        torch.zeros_like(intervention.removed_boundary_correction),
    )
    assert torch.equal(intervention.output.total_rates, baseline.total_rates)
    assert torch.equal(
        intervention.output.boundary_correction, baseline.boundary_correction
    )
    assert torch.equal(intervention.output.class_probs, baseline.class_probs)


def test_removing_all_cells_leaves_v3_prior_and_zero_paths_correction() -> None:
    baseline = _output()
    removals = {name: mask.clone() for name, mask in baseline.active_valid_masks.items()}
    intervention = replay_paths_without(baseline, removals)

    torch.testing.assert_close(intervention.output.total_rates, baseline.prior_rates)
    assert torch.equal(
        intervention.output.boundary_correction,
        torch.zeros_like(intervention.output.boundary_correction),
    )
    prior = decode_pure_birth_rates(baseline.prior_rates)
    torch.testing.assert_close(intervention.output.class_probs, prior.class_probs)


def test_repeated_same_deletion_is_an_exact_noop() -> None:
    baseline = _output(batch=1)
    mask = torch.zeros_like(baseline.active_valid_masks["s4"])
    mask[:, 0, 0] = True
    first = replay_paths_without(baseline, {"s4": mask}).output
    repeated = replay_paths_without(first, {"s4": mask})

    assert torch.equal(repeated.removed_rates, torch.zeros_like(repeated.removed_rates))
    assert torch.equal(
        repeated.removed_boundary_correction,
        torch.zeros_like(repeated.removed_boundary_correction),
    )
    assert torch.equal(repeated.output.total_rates, first.total_rates)
    assert torch.equal(repeated.output.boundary_correction, first.boundary_correction)
    assert torch.equal(repeated.output.class_probs, first.class_probs)


def test_repeated_distinct_deletions_equal_union_replay() -> None:
    baseline = _output(batch=1)
    first_mask = torch.zeros_like(baseline.active_valid_masks["s4"])
    first_mask[:, 0, 0] = True
    first = replay_paths_without(baseline, {"s4": first_mask}).output
    second_mask = torch.zeros_like(first.active_valid_masks["s8"])
    second_mask[:, 0, 0] = True
    repeated = replay_paths_without(first, {"s8": second_mask})
    combined = replay_paths_without(
        baseline, {"s4": first_mask, "s8": second_mask}
    )

    torch.testing.assert_close(repeated.output.total_rates, combined.output.total_rates)
    torch.testing.assert_close(
        repeated.output.boundary_correction,
        combined.output.boundary_correction,
    )
    torch.testing.assert_close(repeated.output.class_probs, combined.output.class_probs)


def test_intervention_accepts_singleton_channel_masks() -> None:
    baseline = _output(batch=2)
    height, width = baseline.active_valid_masks["s8"].shape[-2:]
    singleton = torch.zeros(1, 1, height, width, dtype=torch.bool)
    singleton[:, :, 0, 0] = True
    intervention = replay_paths_without(baseline, {"s8": singleton})

    assert intervention.removal_masks["s8"].shape == (2, height, width)
    assert torch.all(intervention.removal_masks["s8"][:, 0, 0])
