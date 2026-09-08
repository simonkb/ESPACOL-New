"""Mathematical and architectural invariants for ORIGIN."""

import math

import pytest
import torch

from models.origin import (
    ConservedOrdinalGenerator,
    decode_pure_birth_rates,
    fit_pure_birth_rates,
    pure_birth_generator,
    reverse_cumulative_atoms,
    sum_local_rate_maps,
)
from models.origin_encoder import (
    ConvNeXtTinyPyramidEncoder,
    OriginEncoderOutput,
    OriginEncoderScale,
    OriginScaleMetadata,
    origin_convnext_stage_specs,
)


def _metadata(name: str, channels: int, height: int, width: int) -> OriginScaleMetadata:
    stride = {"s4": 4, "s8": 8, "s16": 16, "s32": 32}.get(name, 1)
    return OriginScaleMetadata(
        name=name,
        feature_index=0,
        channels=channels,
        output_stride=stride,
        receptive_field=1,
        center_offset=0.5,
        input_size=(height * stride, width * stride),
        lattice_size=(height, width),
    )


def _fake_encoder_output(
    *,
    batch: int = 2,
    channels: int = 4,
    height: int = 3,
    width: int = 4,
    requires_grad: bool = False,
) -> OriginEncoderOutput:
    torch.manual_seed(13)
    features = torch.randn(
        batch, channels, height, width, requires_grad=requires_grad
    )
    valid = torch.ones(batch, height, width, dtype=torch.bool)
    valid[0, 0, 0] = False
    return OriginEncoderOutput(
        {
            "s4": OriginEncoderScale(
                features=features,
                valid_mask=valid,
                metadata=_metadata("s4", channels, height, width),
            )
        }
    )


def test_generator_is_conservative_metzler_and_absorbing() -> None:
    rates = torch.tensor([[0.2, 1.1, 0.0, 2.3]], dtype=torch.float64)
    generator = pure_birth_generator(rates)

    assert generator.shape == (1, 5, 5)
    torch.testing.assert_close(
        generator.sum(dim=-1), torch.zeros(1, 5, dtype=torch.float64)
    )
    diagonal = generator.diagonal(dim1=-2, dim2=-1)
    assert torch.all(diagonal <= 0)
    off_diagonal = generator - torch.diag_embed(diagonal)
    assert torch.all(off_diagonal >= 0)
    torch.testing.assert_close(generator[0, -1], torch.zeros(5, dtype=torch.float64))
    torch.testing.assert_close(
        generator[0].diagonal(offset=1), rates[0]
    )


def test_binary_chain_matches_exact_analytic_noisy_or_law() -> None:
    rates = torch.tensor([[0.0], [0.2], [2.0]], dtype=torch.float64)
    decoded = decode_pure_birth_rates(rates, force_fp64=True)
    expected_zero = torch.exp(-rates[:, 0])
    expected = torch.stack((expected_zero, 1.0 - expected_zero), dim=-1)
    torch.testing.assert_close(decoded.class_probs, expected, atol=1e-12, rtol=1e-12)


def test_equal_rate_chain_matches_truncated_poisson_law() -> None:
    rate = 1.3
    rates = torch.full((1, 4), rate, dtype=torch.float64)
    decoded = decode_pure_birth_rates(rates, force_fp64=True)
    first = [math.exp(-rate) * rate**state / math.factorial(state) for state in range(4)]
    expected = torch.tensor(
        [first + [1.0 - sum(first)]], dtype=torch.float64
    )
    torch.testing.assert_close(decoded.class_probs, expected, atol=2e-12, rtol=2e-12)


def test_decoder_promotes_the_entire_probability_path_to_fp64() -> None:
    rates = torch.tensor([[0.2, 1.1, 2.3, 0.7]], dtype=torch.float32)
    decoded = decode_pure_birth_rates(rates)

    assert decoded.total_rates.dtype == torch.float32
    for tensor in (
        decoded.generator,
        decoded.transition_matrix,
        decoded.class_probs,
        decoded.log_class_probs,
        decoded.cumulative_probs,
        decoded.expected_grade,
    ):
        assert tensor.dtype == torch.float64
    with pytest.raises(ValueError, match="must run in FP64"):
        decode_pure_birth_rates(rates, force_fp64=False)


def test_stable_decoder_matches_matrix_exp_forward_at_moderate_rates() -> None:
    torch.manual_seed(41)
    rates = torch.rand(16, 4, dtype=torch.float64) * 12.0
    decoded = decode_pure_birth_rates(rates)
    reference = torch.matrix_exp(pure_birth_generator(rates))[..., 0, :]
    torch.testing.assert_close(
        decoded.class_probs,
        reference,
        atol=2e-13,
        rtol=2e-12,
    )


def test_high_unequal_rates_have_correct_finite_nll_gradients() -> None:
    # This is the rate regime observed immediately before the EyePACS AMP
    # failures. torch.matrix_exp has an accurate forward value here but its
    # generic backward has produced gradients around 1e61. For a sample in
    # grade zero, -log p_0 is exactly lambda_0.
    rates = torch.tensor(
        [[141.475647, 426.073364, 31.953876, 70.849700]],
        dtype=torch.float64,
        requires_grad=True,
    )
    decoded = decode_pure_birth_rates(rates)
    # Constants were generated independently with a 100-decimal mpmath
    # matrix-exponential oracle, not from this Taylor implementation.
    reference_probabilities = torch.tensor(
        [[
            3.613326317285435e-62,
            1.796211452251686e-62,
            1.852020049106147e-14,
            1.521480017974467e-14,
            0.9999999999999663,
        ]],
        dtype=torch.float64,
    )
    torch.testing.assert_close(
        decoded.class_probs,
        reference_probabilities,
        atol=0.0,
        rtol=2e-12,
    )
    loss = -decoded.log_class_probs[0, 0]
    gradient = torch.autograd.grad(loss, rates)[0]
    torch.testing.assert_close(
        gradient,
        torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float64),
        atol=2e-12,
        rtol=2e-12,
    )
    assert float(gradient.abs().max()) < 2.0


def test_high_rate_fp64_decoder_backpropagates_correctly_to_fp32_ledger() -> None:
    rates = torch.tensor(
        [[141.475647, 426.073364, 31.953876, 70.849700]],
        dtype=torch.float32,
        requires_grad=True,
    )
    loss = -decode_pure_birth_rates(rates).log_class_probs[0, 1]
    gradient = torch.autograd.grad(loss, rates)[0]
    expected = torch.tensor(
        [[0.9894179141, 0.0035137316, 0.0, 0.0]],
        dtype=torch.float32,
    )
    torch.testing.assert_close(gradient, expected, atol=2e-6, rtol=2e-5)


@pytest.mark.parametrize("target", range(5))
def test_high_rate_class_nll_gradient_matches_central_difference(target: int) -> None:
    base = torch.tensor(
        [141.475647, 426.073364, 31.953876, 70.849700],
        dtype=torch.float64,
    )
    rates = base.clone().requires_grad_(True)
    loss = -decode_pure_birth_rates(rates.unsqueeze(0)).log_class_probs[0, target]
    gradient = torch.autograd.grad(loss, rates)[0]

    step = 1e-4
    finite_difference = torch.empty_like(base)
    with torch.no_grad():
        for boundary in range(base.numel()):
            plus = base.clone()
            minus = base.clone()
            plus[boundary] += step
            minus[boundary] -= step
            plus_loss = -decode_pure_birth_rates(
                plus.unsqueeze(0)
            ).log_class_probs[0, target]
            minus_loss = -decode_pure_birth_rates(
                minus.unsqueeze(0)
            ).log_class_probs[0, target]
            finite_difference[boundary] = (plus_loss - minus_loss) / (2.0 * step)

    assert bool(torch.isfinite(gradient).all())
    torch.testing.assert_close(
        gradient,
        finite_difference,
        atol=2e-7,
        rtol=2e-5,
    )


def test_decoder_rejects_rates_beyond_its_audited_log_range() -> None:
    with pytest.raises(FloatingPointError, match="audited FP64 decoder range"):
        decode_pure_birth_rates(torch.tensor([[701.0]], dtype=torch.float64))


def test_zero_intermediate_rate_blocks_every_higher_state() -> None:
    decoded = decode_pure_birth_rates(
        torch.tensor([[2.0, 0.0, 50.0]], dtype=torch.float64),
        force_fp64=True,
    )
    torch.testing.assert_close(
        decoded.class_probs[:, 2:], torch.zeros(1, 2, dtype=torch.float64)
    )
    torch.testing.assert_close(
        decoded.class_probs.sum(dim=-1), torch.ones(1, dtype=torch.float64)
    )


def test_endpoint_probabilities_normalize_and_cumulatives_are_nested() -> None:
    torch.manual_seed(5)
    rates = torch.rand(12, 6, dtype=torch.float64) * 4.0
    decoded = decode_pure_birth_rates(rates, force_fp64=True)
    torch.testing.assert_close(
        decoded.class_probs.sum(dim=-1), torch.ones(12, dtype=torch.float64)
    )
    assert torch.all(decoded.class_probs >= 0)
    assert torch.all(decoded.cumulative_probs >= 0)
    assert torch.all(decoded.cumulative_probs <= 1)
    assert torch.all(
        decoded.cumulative_probs[:, :-1] >= decoded.cumulative_probs[:, 1:]
    )
    torch.testing.assert_close(
        decoded.expected_grade, decoded.cumulative_probs.sum(dim=-1)
    )


def test_rate_increase_cannot_decrease_expected_or_median_grade() -> None:
    base = torch.tensor([[0.8, 0.7, 0.6, 0.5]], dtype=torch.float64)
    baseline = decode_pure_birth_rates(base, force_fp64=True)
    for boundary in range(base.shape[-1]):
        increased = base.clone()
        increased[:, boundary] += 3.0
        changed = decode_pure_birth_rates(increased, force_fp64=True)
        assert bool((changed.expected_grade >= baseline.expected_grade - 1e-12).all())
        assert bool((changed.posterior_median >= baseline.posterior_median).all())


def test_map_is_diagnostic_not_a_monotone_decision_rule() -> None:
    # Increasing the second rate transfers probability from grade 1 to grade
    # 2. Grade 0 can consequently become the MAP state even though expected
    # severity cannot fall. MAP is prospectively used for accuracy, while all
    # monotonicity claims and monotone-grade reporting use posterior median.
    increased = torch.tensor([[0.979, 1.090]], dtype=torch.float64)
    base = torch.tensor([[0.979, 0.150]], dtype=torch.float64)
    before = decode_pure_birth_rates(base, force_fp64=True)
    after = decode_pure_birth_rates(increased, force_fp64=True)
    assert int(before.class_map) == 1
    assert int(after.class_map) == 0
    assert float(after.expected_grade) > float(before.expected_grade)
    assert torch.equal(before.predicted_grade, before.class_map)
    assert torch.equal(before.monotone_grade, before.posterior_median)


def test_reverse_cumulative_atoms_encode_prerequisite_support() -> None:
    atoms = torch.tensor([[[[1.0]], [[2.0]], [[3.0]], [[4.0]]]])
    compiled = reverse_cumulative_atoms(atoms, dim=1)
    torch.testing.assert_close(
        compiled.flatten(), torch.tensor([10.0, 9.0, 7.0, 4.0])
    )
    assert torch.all(compiled[:, :-1] >= compiled[:, 1:])


@pytest.mark.parametrize("atom_mode", ["cumulative", "independent", "hybrid"])
def test_local_generator_conserves_rates_and_zeros_invalid_cells(atom_mode: str) -> None:
    encoded = _fake_encoder_output()
    generator = ConservedOrdinalGenerator(
        stage_channels={"s4": 4},
        num_classes=5,
        hidden_channels=8,
        evidence_scales=("s4",),
        atom_mode=atom_mode,
        reference_count=10.0,
        atom_rate_init=0.01,
    )
    output = generator(encoded)

    assert output.class_probs.shape == (2, 5)
    assert output.local_rate_maps["s4"].shape == (2, 3, 4, 4)
    invalid = ~output.valid_masks["s4"]
    invalid_rates = output.local_rate_maps["s4"][invalid]
    assert torch.equal(invalid_rates, torch.zeros_like(invalid_rates))
    reconstructed = sum_local_rate_maps(
        output.local_rate_maps_channels_first, output.prior_rates
    )
    torch.testing.assert_close(reconstructed, output.total_rates)
    torch.testing.assert_close(
        output.scale_simplex.sum(dim=0), torch.ones(4)
    )
    if atom_mode == "cumulative":
        compiled = output.scale_evidence["s4"].compiled_atoms
        assert torch.all(compiled[:, :-1] >= compiled[:, 1:])


def test_original_geometry_is_scaled_to_reference_count() -> None:
    encoded = _fake_encoder_output(batch=2)
    generator = ConservedOrdinalGenerator(
        stage_channels={"s4": 4},
        num_classes=4,
        hidden_channels=5,
        evidence_scales=("s4",),
        reference_count=17.0,
        atom_rate_init=0.01,
    )
    output = generator(encoded)
    evidence = output.scale_evidence["s4"]
    valid_count = evidence.valid_mask.sum(dim=(-2, -1)).float()
    torch.testing.assert_close(
        evidence.geometry_weight * valid_count,
        torch.full_like(valid_count, 17.0),
    )


def test_spatial_permutation_preserves_total_rates_and_posterior() -> None:
    encoded = _fake_encoder_output(batch=1, height=2, width=4)
    generator = ConservedOrdinalGenerator(
        stage_channels={"s4": 4},
        num_classes=5,
        hidden_channels=7,
        evidence_scales=("s4",),
        atom_rate_init=0.01,
    ).eval()
    baseline = generator(encoded)

    scale = encoded.scales["s4"]
    permutation = torch.tensor([7, 1, 5, 0, 6, 2, 4, 3])
    features = scale.features.flatten(2)[:, :, permutation].reshape_as(scale.features)
    valid = scale.valid_mask.flatten(1)[:, permutation].reshape_as(scale.valid_mask)
    permuted = OriginEncoderOutput(
        {"s4": OriginEncoderScale(features, valid, scale.metadata)}
    )
    changed = generator(permuted)
    torch.testing.assert_close(changed.total_rates, baseline.total_rates)
    torch.testing.assert_close(changed.class_probs, baseline.class_probs)


def test_generator_path_has_finite_gradients() -> None:
    encoded = _fake_encoder_output(requires_grad=True)
    generator = ConservedOrdinalGenerator(
        stage_channels={"s4": 4},
        num_classes=5,
        hidden_channels=8,
        evidence_scales=("s4",),
    )
    output = generator(encoded)
    labels = torch.tensor([0, 3])
    loss = -output.log_class_probs[torch.arange(2), labels].mean()
    loss.backward()
    source = encoded.scales["s4"].features
    assert source.grad is not None
    assert torch.isfinite(source.grad).all()
    trainable_grads = [p.grad for p in generator.parameters() if p.requires_grad]
    assert trainable_grads and all(grad is not None for grad in trainable_grads)
    assert all(torch.isfinite(grad).all() for grad in trainable_grads)


def test_fit_pure_birth_rates_reconstructs_an_interior_distribution() -> None:
    target = torch.tensor([0.45, 0.22, 0.16, 0.10, 0.07], dtype=torch.float64)
    rates = fit_pure_birth_rates(target, tolerance=1e-12)
    decoded = decode_pure_birth_rates(rates, force_fp64=True)
    torch.testing.assert_close(decoded.class_probs, target, atol=2e-9, rtol=2e-9)


def test_convnext_stage_contract_and_fixed_receptive_fields() -> None:
    assert origin_convnext_stage_specs() == {
        "s4": (96, 4, 76),
        "s8": (192, 8, 224),
        "s16": (384, 16, 1096),
        "s32": (768, 32, 1688),
    }
    encoder = ConvNeXtTinyPyramidEncoder(pretrained=False).eval()
    image = torch.zeros(1, 3, 64, 64)
    valid = torch.ones(1, 1, 64, 64, dtype=torch.bool)
    with torch.no_grad():
        output = encoder(image, valid)
    assert tuple(output.scales) == ("s4", "s8", "s16", "s32")
    assert [tuple(scale.features.shape) for scale in output.scales.values()] == [
        (1, 96, 16, 16),
        (1, 192, 8, 8),
        (1, 384, 4, 4),
        (1, 768, 2, 2),
    ]
    assert all(not scale.metadata.globally_mixed for scale in output.scales.values())


@pytest.mark.parametrize(
    "bad_rates",
    [torch.tensor([[-1.0]]), torch.tensor([[float("nan")]])],
)
def test_generator_rejects_invalid_rates(bad_rates: torch.Tensor) -> None:
    with pytest.raises((ValueError, FloatingPointError)):
        pure_birth_generator(bad_rates)
