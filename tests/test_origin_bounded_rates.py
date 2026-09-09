"""Adversarial invariants for ORIGIN's bounded, replayable rate ledger.

These tests exercise the architectural bound rather than the decoder's
emergency range check.  The prediction-time ledger is required to remain a
literal additive measure: stored local entries plus the stored prior equal the
rates consumed by the pure-birth decoder, and interventions delete those
entries without renormalising the surviving evidence.
"""

import math
from typing import Sequence

import pytest
import torch

from models.origin import (
    ConservedOrdinalGenerator,
    PointwiseSeverityAtomHead,
    decode_pure_birth_rates,
    replay_without,
    sum_local_rate_maps,
)
from models.origin_encoder import (
    OriginEncoderOutput,
    OriginEncoderScale,
    OriginScaleMetadata,
)


_ATOM_MODES = ("cumulative", "independent", "hybrid")


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
        receptive_field=7,
        center_offset=float(stride) / 2.0,
        input_size=(height * stride, width * stride),
        lattice_size=(height, width),
    )


def _encoder_output(
    *,
    batch: int = 2,
    all_invalid_second_sample: bool = False,
    requires_grad: bool = False,
) -> OriginEncoderOutput:
    generator = torch.Generator().manual_seed(1701)
    specs = {
        "s4": (3, 3, 4),
        "s8": (5, 2, 3),
    }
    scales = {}
    for name, (channels, height, width) in specs.items():
        features = torch.randn(
            batch,
            channels,
            height,
            width,
            generator=generator,
            requires_grad=requires_grad,
        )
        valid = torch.ones(batch, height, width, dtype=torch.bool)
        valid[0, 0, 0] = False
        if all_invalid_second_sample and batch > 1:
            valid[1] = False
        scales[name] = OriginEncoderScale(
            features=features,
            valid_mask=valid,
            metadata=_metadata(name, channels, height, width),
        )
    return OriginEncoderOutput(scales)


def _generator(
    atom_mode: str,
    *,
    reference_count: float = 12.0,
    atom_rate_init: float = 1e-6,
    prior_rate_init: float = 1e-4,
    boundary_scale_init: float = 1.0,
    total_rate_cap: float = 64.0,
    prior_rate_cap: float = 1.0,
    boundary_scale_cap: float = 2.0,
    rate_roundoff_margin: float = 1.0,
) -> ConservedOrdinalGenerator:
    return ConservedOrdinalGenerator(
        stage_channels={"s4": 3, "s8": 5},
        num_classes=5,
        hidden_channels=7,
        evidence_scales=("s4", "s8"),
        atom_mode=atom_mode,
        hybrid_cumulative_init=0.8,
        reference_count=reference_count,
        atom_rate_init=atom_rate_init,
        prior_rate_init=prior_rate_init,
        boundary_scale_init=boundary_scale_init,
        total_rate_cap=total_rate_cap,
        prior_rate_cap=prior_rate_cap,
        boundary_scale_cap=boundary_scale_cap,
        rate_roundoff_margin=rate_roundoff_margin,
    )


def _set_constant_atom_logits(
    generator: ConservedOrdinalGenerator,
    values: Sequence[float],
) -> None:
    values_tensor = torch.tensor(values, dtype=torch.float32)
    assert values_tensor.numel() == generator.num_boundaries
    with torch.no_grad():
        for head in generator.heads.values():
            head.atom_logits.weight.zero_()
            head.atom_logits.bias.copy_(values_tensor)


def _assert_finite_distribution(output) -> None:
    tensors = (
        output.prior_rates,
        output.total_rates,
        output.generator,
        output.transition_matrix,
        output.class_probs,
        output.log_class_probs,
        output.cumulative_probs,
        output.expected_grade,
    )
    assert all(bool(torch.isfinite(tensor).all()) for tensor in tensors)
    assert bool((output.total_rates >= 0).all())
    assert bool((output.class_probs >= 0).all())
    torch.testing.assert_close(
        output.class_probs.sum(dim=-1),
        torch.ones(
            output.class_probs.shape[:-1],
            dtype=output.class_probs.dtype,
            device=output.class_probs.device,
        ),
        atol=5e-12,
        rtol=0.0,
    )


def test_null_augmented_head_bounds_any_finite_logits_and_allows_no_evidence() -> None:
    head = PointwiseSeverityAtomHead(
        in_channels=3,
        hidden_channels=5,
        num_boundaries=4,
        atom_rate_init=1e-6,
        atom_mass_cap=0.25,
    )
    features = torch.randn(2, 3, 2, 3)
    adversarial_logits = (
        (-1e20, -1e20, -1e20, -1e20),
        (1e20, -1e20, 0.0, 1e10),
        (1e4, 1e4, 1e4, 1e4),
    )
    for values in adversarial_logits:
        with torch.no_grad():
            head.atom_logits.weight.zero_()
            head.atom_logits.bias.copy_(torch.tensor(values))
        logits, atoms = head(features)
        assert bool(torch.isfinite(logits).all())
        assert bool(torch.isfinite(atoms).all())
        assert bool((atoms >= 0).all())
        assert float(atoms.detach().sum(dim=1).max()) <= head.atom_mass_cap + 1e-7

    # A null atom is semantically necessary: every severity atom may be absent.
    with torch.no_grad():
        head.atom_logits.bias.fill_(-1e4)
    _, absent_atoms = head(features)
    assert torch.equal(absent_atoms, torch.zeros_like(absent_atoms))


def test_generator_derives_a_proved_atom_mass_budget() -> None:
    generator = _generator(
        "cumulative",
        reference_count=10.0,
        total_rate_cap=80.0,
        prior_rate_cap=2.0,
        boundary_scale_cap=3.0,
        rate_roundoff_margin=0.5,
    )
    expected = (80.0 - 2.0 - 0.5) / (10.0 * 3.0)
    assert math.isclose(generator.atom_mass_cap, expected, rel_tol=0.0, abs_tol=1e-12)
    assert all(
        math.isclose(head.atom_mass_cap, expected, rel_tol=0.0, abs_tol=1e-12)
        for head in generator.heads.values()
    )


@pytest.mark.parametrize("atom_mode", _ATOM_MODES)
def test_adversarial_logits_cannot_escape_total_rate_cap(atom_mode: str) -> None:
    generator = _generator(atom_mode).eval()
    # Concentrate essentially all cell mass on the highest atom.  That atom
    # supports the final boundary in every atom mode, so saturating both other
    # bounded factors drives one total rate to the architectural cap minus
    # its explicit accumulation-roundoff reserve.
    _set_constant_atom_logits(generator, (-1e4, -1e4, -1e4, 1e4))
    with torch.no_grad():
        generator.raw_prior_rates.fill_(1e4)
        generator.raw_boundary_scales.fill_(1e4)
        generator.scale_simplex_logits[0].fill_(1e4)
        generator.scale_simplex_logits[1].fill_(-1e4)
        output = generator(_encoder_output())

    _assert_finite_distribution(output)
    maximum_rate = float(output.total_rates.detach().max())
    assert maximum_rate <= generator.total_rate_cap
    # This is not merely a loose emergency ceiling: the construction can
    # approach the proved usable budget while remaining a valid posterior.
    usable_cap = generator.total_rate_cap - generator.rate_roundoff_margin
    assert maximum_rate >= usable_cap - 5e-4
    assert maximum_rate <= usable_cap + 5e-4
    assert output.total_rate_cap == generator.total_rate_cap
    assert output.prior_rate_cap == generator.prior_rate_cap
    assert output.boundary_scale_cap == generator.boundary_scale_cap
    assert output.atom_mass_cap == generator.atom_mass_cap
    assert output.rate_roundoff_margin == generator.rate_roundoff_margin

    reconstructed = sum_local_rate_maps(
        output.local_rate_maps_channels_first,
        output.prior_rates,
    )
    torch.testing.assert_close(reconstructed, output.total_rates, atol=1e-6, rtol=2e-6)


@pytest.mark.parametrize("atom_mode", _ATOM_MODES)
def test_low_rate_initialization_is_preserved_and_has_finite_gradients(
    atom_mode: str,
) -> None:
    reference_count = 13.0
    atom_rate_init = 1e-6
    prior_rate_init = 1e-4
    boundary_scale_init = 1.0
    hybrid_mix = 0.8
    generator = _generator(
        atom_mode,
        reference_count=reference_count,
        atom_rate_init=atom_rate_init,
        prior_rate_init=prior_rate_init,
        boundary_scale_init=boundary_scale_init,
    )
    # Zeroing the final kernels makes the constructor's common bias observable
    # exactly at every cell, independent of its feature value.
    initial_bias = float(generator.heads["s4"].atom_logits.bias[0].detach())
    _set_constant_atom_logits(generator, [initial_bias] * generator.num_boundaries)

    output = generator(_encoder_output())
    for evidence in output.scale_evidence.values():
        torch.testing.assert_close(
            evidence.incremental_atoms,
            torch.full_like(evidence.incremental_atoms, atom_rate_init),
            atol=2e-11,
            rtol=2e-5,
        )

    boundaries = generator.num_boundaries
    if atom_mode == "cumulative":
        coefficients = torch.arange(boundaries, 0, -1, dtype=torch.float32)
    elif atom_mode == "independent":
        coefficients = torch.ones(boundaries)
    else:
        coefficients = 1.0 + hybrid_mix * torch.arange(
            boundaries - 1, -1, -1, dtype=torch.float32
        )
    expected = (
        prior_rate_init
        + reference_count
        * boundary_scale_init
        * atom_rate_init
        * coefficients
    ).to(dtype=output.total_rates.dtype)
    torch.testing.assert_close(
        output.total_rates,
        expected.unsqueeze(0).expand_as(output.total_rates),
        atol=2e-8,
        rtol=2e-5,
    )

    output.total_rates.sum().backward()
    for head in generator.heads.values():
        gradient = head.atom_logits.bias.grad
        assert gradient is not None
        assert bool(torch.isfinite(gradient).all())
        assert float(gradient.detach().abs().max()) > 0.0
    for parameter in (generator.raw_prior_rates, generator.raw_boundary_scales):
        assert parameter.grad is not None
        assert bool(torch.isfinite(parameter.grad).all())
        assert float(parameter.grad.detach().abs().max()) > 0.0


@pytest.mark.parametrize("atom_mode", _ATOM_MODES)
def test_high_but_unsaturated_bounded_path_backpropagates_finitely(
    atom_mode: str,
) -> None:
    encoded = _encoder_output(requires_grad=True)
    generator = _generator(atom_mode)
    _set_constant_atom_logits(generator, (-4.0, -2.0, 0.0, 2.0))
    # Re-introduce a small feature-dependent term after fixing the biases.
    with torch.no_grad():
        for head in generator.heads.values():
            head.atom_logits.weight.normal_(mean=0.0, std=0.01)
        generator.raw_prior_rates.zero_()
        generator.raw_boundary_scales.fill_(2.0)

    output = generator(encoded)
    labels = torch.tensor([0, 4])
    loss = -output.log_class_probs[torch.arange(2), labels].mean()
    loss.backward()

    _assert_finite_distribution(output)
    assert float(output.total_rates.detach().max()) <= generator.total_rate_cap
    for scale in encoded.scales.values():
        assert scale.features.grad is not None
        assert bool(torch.isfinite(scale.features.grad).all())
    gradients = [
        parameter.grad
        for parameter in generator.parameters()
        if parameter.requires_grad and parameter.grad is not None
    ]
    assert gradients
    assert all(bool(torch.isfinite(gradient).all()) for gradient in gradients)


@pytest.mark.parametrize("atom_mode", _ATOM_MODES)
def test_effective_ledger_and_exact_removal_replay_remain_additive(
    atom_mode: str,
) -> None:
    generator = _generator(atom_mode).eval()
    _set_constant_atom_logits(generator, (-5.0, -2.0, 1.0, 5.0))
    with torch.no_grad():
        generator.raw_prior_rates.fill_(4.0)
        generator.raw_boundary_scales.fill_(4.0)
        baseline = generator(_encoder_output())

    reconstructed = sum_local_rate_maps(
        baseline.local_rate_maps_channels_first,
        baseline.prior_rates,
    )
    torch.testing.assert_close(reconstructed, baseline.total_rates, atol=1e-6, rtol=2e-6)

    removals = {
        name: torch.zeros_like(mask)
        for name, mask in baseline.valid_masks.items()
    }
    removals["s4"][:, 0, 1] = True
    removals["s8"][:, 1, 0] = True
    intervention = replay_without(baseline, removals)
    expected_removed = torch.zeros_like(baseline.total_rates)
    for name, mask in intervention.removal_masks.items():
        rate_map = baseline.scale_evidence[name].local_rate_map
        expected_removed += torch.where(
            mask[:, None], rate_map, torch.zeros_like(rate_map)
        ).sum(dim=(-2, -1))
    torch.testing.assert_close(
        intervention.removed_rates,
        expected_removed,
        atol=1e-6,
        rtol=2e-6,
    )
    torch.testing.assert_close(
        intervention.output.total_rates,
        baseline.total_rates - expected_removed,
        atol=2e-5,
        rtol=2e-6,
    )
    replayed_ledger = sum_local_rate_maps(
        intervention.output.local_rate_maps_channels_first,
        intervention.output.prior_rates,
    )
    torch.testing.assert_close(
        replayed_ledger,
        intervention.output.total_rates,
        atol=1e-6,
        rtol=2e-6,
    )
    direct = decode_pure_birth_rates(intervention.output.total_rates)
    torch.testing.assert_close(intervention.output.class_probs, direct.class_probs)
    assert intervention.output.total_rate_cap == baseline.total_rate_cap
    assert intervention.output.atom_mass_cap == baseline.atom_mass_cap


@pytest.mark.parametrize("atom_mode", _ATOM_MODES)
def test_invalid_cells_are_zero_and_all_invalid_maps_reduce_to_bounded_prior(
    atom_mode: str,
) -> None:
    generator = _generator(atom_mode).eval()
    _set_constant_atom_logits(generator, (1e4, -1e4, 1e4, -1e4))
    with torch.no_grad():
        generator.raw_prior_rates.fill_(1e4)
        generator.raw_boundary_scales.fill_(1e4)
        output = generator(
            _encoder_output(all_invalid_second_sample=True)
        )

    for name, spatial_rates in output.local_rate_maps.items():
        invalid = ~output.valid_masks[name]
        assert torch.equal(
            spatial_rates[invalid],
            torch.zeros_like(spatial_rates[invalid]),
        )
        assert float(output.scale_evidence[name].geometry_weight[1]) == 0.0
    torch.testing.assert_close(output.total_rates[1], output.prior_rates[1])
    assert float(output.prior_rates.detach().max()) <= generator.prior_rate_cap + 1e-7
    assert float(output.total_rates.detach().max()) <= generator.total_rate_cap

    invalid_removals = {
        name: ~mask for name, mask in output.valid_masks.items()
    }
    replay = replay_without(output, invalid_removals)
    assert torch.equal(replay.removed_rates, torch.zeros_like(replay.removed_rates))
    assert torch.equal(replay.output.total_rates, output.total_rates)
    assert torch.equal(replay.output.class_probs, output.class_probs)


@pytest.mark.parametrize("num_classes", (2, 3, 5))
@pytest.mark.parametrize("atom_mode", _ATOM_MODES)
def test_one_valid_cell_respects_bound_for_supported_class_counts(
    num_classes: int,
    atom_mode: str,
) -> None:
    # One valid cell is the worst geometry-amplification case: it receives the
    # entire reference exposure. Exercise it at the binary edge case and at
    # the small and five-grade ordinal settings used by downstream datasets.
    height, width = 2, 3
    valid = torch.zeros(1, height, width, dtype=torch.bool)
    valid[:, 1, 2] = True
    encoded = OriginEncoderOutput(
        {
            "s4": OriginEncoderScale(
                features=torch.randn(1, 3, height, width),
                valid_mask=valid,
                metadata=_metadata("s4", 3, height, width),
            )
        }
    )
    generator = ConservedOrdinalGenerator(
        stage_channels={"s4": 3},
        num_classes=num_classes,
        hidden_channels=5,
        evidence_scales=("s4",),
        atom_mode=atom_mode,
        reference_count=4096.0,
        atom_rate_init=1e-6,
        total_rate_cap=64.0,
        prior_rate_cap=1.0,
        boundary_scale_cap=2.0,
        rate_roundoff_margin=1.0,
    ).eval()
    logits = [-1e4] * (num_classes - 1)
    logits[-1] = 1e4
    _set_constant_atom_logits(generator, logits)
    with torch.no_grad():
        generator.raw_prior_rates.fill_(1e4)
        generator.raw_boundary_scales.fill_(1e4)
        output = generator(encoded)

    maximum_rate = float(output.total_rates.max())
    usable_cap = generator.total_rate_cap - generator.rate_roundoff_margin
    assert usable_cap - 5e-4 <= maximum_rate <= usable_cap + 5e-4
    assert maximum_rate <= generator.total_rate_cap
    assert float(output.scale_evidence["s4"].geometry_weight[0]) == 4096.0
    spatial_rates = output.local_rate_maps["s4"]
    assert torch.equal(
        spatial_rates[~valid],
        torch.zeros_like(spatial_rates[~valid]),
    )
    reconstructed = sum_local_rate_maps(
        output.local_rate_maps_channels_first,
        output.prior_rates,
    )
    torch.testing.assert_close(reconstructed, output.total_rates)
    _assert_finite_distribution(output)


@pytest.mark.parametrize(
    "kwargs",
    (
        {"total_rate_cap": 0.0},
        {"total_rate_cap": float("nan")},
        {"total_rate_cap": 700.0001},
        {"prior_rate_cap": 0.0},
        {"prior_rate_cap": 64.0},
        {"boundary_scale_cap": 0.0},
        {"boundary_scale_cap": float("inf")},
        {"rate_roundoff_margin": 0.0},
        {"rate_roundoff_margin": 63.0},
    ),
)
def test_invalid_or_decoder_unsafe_caps_fail_at_construction(kwargs) -> None:
    with pytest.raises(ValueError):
        _generator("cumulative", **kwargs)
