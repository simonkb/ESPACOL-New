"""Mathematical, gradient, and compatibility invariants for PATHS."""

from __future__ import annotations

from dataclasses import replace

import pytest
import torch
import torch.nn as nn

from models.origin import ConservedOrdinalGenerator, OriginModel, OriginOutput
from models.origin_encoder import (
    OriginEncoderOutput,
    OriginEncoderScale,
    OriginScaleMetadata,
)
from models.paths import (
    PathsContinuationRefiner,
    PathsModel,
    PathsOutput,
    continuation_logits_from_log_probs,
    decode_continuation_logits,
    tangent_removed_pgf_probe,
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


def _encoded(batch: int = 2, *, requires_grad: bool = False) -> OriginEncoderOutput:
    torch.manual_seed(117)
    scales = {}
    channels = {"s4": 3, "s8": 5}
    shapes = {"s4": (3, 4), "s8": (2, 2)}
    for name in ("s4", "s8"):
        height, width = shapes[name]
        features = torch.randn(
            batch,
            channels[name],
            height,
            width,
            requires_grad=requires_grad,
        )
        valid = torch.ones(batch, height, width, dtype=torch.bool)
        valid[0, -1, -1] = False
        scales[name] = OriginEncoderScale(
            features=features,
            valid_mask=valid,
            metadata=_metadata(name, channels[name], height, width),
        )
    return OriginEncoderOutput(scales)


def _base_output(*, requires_grad: bool = False) -> OriginOutput:
    generator = ConservedOrdinalGenerator(
        stage_channels={"s4": 3, "s8": 5},
        num_classes=5,
        hidden_channels=7,
        evidence_scales=("s4", "s8"),
        reference_count=8.0,
        atom_rate_init=0.02,
        prior_rate_init=0.001,
    ).eval()
    return generator(_encoded(requires_grad=requires_grad), force_decoder_fp64=True)


def _paths_output(
    *, requires_grad: bool = False, gain_init: float = 0.05
) -> tuple[PathsOutput, PathsContinuationRefiner]:
    base = _base_output(requires_grad=requires_grad)
    refiner = PathsContinuationRefiner(
        4,
        ("s4", "s8"),
        probe_z=(0.1, 0.25, 0.5, 0.75, 0.9),
        correction_cap=4.0,
        gain_init=gain_init,
    )
    output = refiner(base)
    assert isinstance(output, PathsOutput)
    return output, refiner


def test_tangent_removed_probe_endpoints_bounds_and_zero_tangent() -> None:
    atoms = torch.linspace(0.0, 1.0, 101, dtype=torch.float64, requires_grad=True)
    probes = torch.tensor([0.1, 0.37, 0.9], dtype=torch.float64)
    phi = tangent_removed_pgf_probe(atoms, probes)

    torch.testing.assert_close(phi[0], torch.zeros(3, dtype=torch.float64))
    torch.testing.assert_close(phi[-1], torch.ones(3, dtype=torch.float64))
    assert torch.all(phi >= 0.0)
    assert torch.all(phi <= atoms.detach()[:, None] + 4.0 * torch.finfo(phi.dtype).eps)

    tangent = torch.autograd.grad(phi[0].sum(), atoms, retain_graph=False)[0][0]
    torch.testing.assert_close(tangent, torch.tensor(0.0, dtype=torch.float64))
    with pytest.raises(ValueError, match="must lie in \\[0, 1\\]"):
        tangent_removed_pgf_probe(torch.tensor([1.01]), probes.float())


def test_mass_normalized_spectrum_is_larger_for_focal_equal_mass_evidence() -> None:
    probes = torch.tensor([0.2, 0.5, 0.8], dtype=torch.float64)
    focal = torch.tensor([1.0, 0.0], dtype=torch.float64)
    diffuse = torch.tensor([0.5, 0.5], dtype=torch.float64)
    focal_spectrum = tangent_removed_pgf_probe(focal, probes).sum(0) / focal.sum()
    diffuse_spectrum = tangent_removed_pgf_probe(diffuse, probes).sum(0) / diffuse.sum()

    assert torch.all(focal_spectrum > diffuse_spectrum)
    positive_gain = torch.tensor(0.4, dtype=torch.float64)
    assert torch.all(positive_gain * focal_spectrum > positive_gain * diffuse_spectrum)


def test_continuation_factorization_round_trips_an_ordinal_law_in_fp64() -> None:
    probabilities = torch.tensor(
        [[0.13, 0.21, 0.17, 0.29, 0.20], [0.55, 0.15, 0.12, 0.10, 0.08]],
        dtype=torch.float64,
    )
    logits = continuation_logits_from_log_probs(probabilities.log())
    decoded = decode_continuation_logits(logits)

    assert decoded.class_probs.dtype == torch.float64
    torch.testing.assert_close(decoded.class_probs, probabilities, atol=2e-15, rtol=2e-15)
    torch.testing.assert_close(
        decoded.class_probs.sum(dim=-1),
        torch.ones(2, dtype=torch.float64),
        atol=1e-15,
        rtol=0.0,
    )
    assert torch.all(decoded.cumulative_probs[:, 1:] <= decoded.cumulative_probs[:, :-1])


def test_strength_zero_returns_the_actual_v3_output_object() -> None:
    base = _base_output()
    refiner = PathsContinuationRefiner(4, ("s4", "s8"), strength=0.0)
    assert refiner(base) is base


def test_positive_focality_correction_is_bounded_and_exactly_additive() -> None:
    output, _ = _paths_output()

    assert torch.all(output.boundary_correction >= 0.0)
    assert torch.all(output.boundary_correction <= 4.0 + 1e-12)
    local_sum = sum(
        value.to(torch.float64).sum(dim=(1, 2))
        for value in output.local_correction_maps.values()
    )
    torch.testing.assert_close(local_sum, output.boundary_correction)
    torch.testing.assert_close(
        output.continuation_logits,
        output.base_continuation_logits + output.boundary_correction,
    )
    assert torch.all(output.continuation_probs >= torch.sigmoid(output.base_continuation_logits))
    torch.testing.assert_close(
        output.class_probs.sum(dim=-1),
        torch.ones(output.class_probs.shape[0], dtype=torch.float64),
    )

    for evidence in output.spectrum_evidence.values():
        invalid = ~evidence.original_valid_mask
        local_phi = evidence.local_phi_spectrum.permute(0, 2, 3, 1, 4)
        local_mass = evidence.local_mass_map.permute(0, 2, 3, 1)
        assert torch.equal(local_phi[invalid], torch.zeros_like(local_phi[invalid]))
        assert torch.equal(local_mass[invalid], torch.zeros_like(local_mass[invalid]))
        torch.testing.assert_close(
            evidence.concentration_spectrum,
            evidence.local_concentration_spectrum.sum(dim=(2, 3)),
        )
        assert torch.all(evidence.concentration_spectrum >= 0.0)
        assert torch.all(evidence.concentration_spectrum <= 1.0 + 1e-12)


def test_learned_path_backpropagates_to_probes_gain_and_base_atoms() -> None:
    output, refiner = _paths_output(requires_grad=True)
    for evidence in output.base_output.scale_evidence.values():
        evidence.compiled_atoms.retain_grad()
    target = torch.tensor([4, 2])
    loss = -output.log_class_probs.gather(1, target[:, None]).mean()
    loss.backward()

    assert refiner.probe_logits.grad is not None
    assert torch.isfinite(refiner.probe_logits.grad).all()
    assert float(refiner.probe_logits.grad.abs().sum()) > 0.0
    assert refiner.raw_gains.grad is not None
    assert torch.isfinite(refiner.raw_gains.grad).all()
    assert float(refiner.raw_gains.grad.abs().sum()) > 0.0
    atom_grad = sum(
        float(evidence.compiled_atoms.grad.abs().sum())
        for evidence in output.base_output.scale_evidence.values()
        if evidence.compiled_atoms.grad is not None
    )
    assert atom_grad > 0.0
    assert torch.all(refiner.gains >= 0.0)
    assert torch.all(refiner.gains <= refiner.correction_cap)


def test_zero_atom_mass_has_zero_finite_spectrum_and_correction() -> None:
    base = _base_output()
    zero_atoms = {
        name: torch.zeros_like(evidence.compiled_atoms, requires_grad=True)
        for name, evidence in base.scale_evidence.items()
    }
    zero_scales = {
        name: replace(evidence, compiled_atoms=zero_atoms[name])
        for name, evidence in base.scale_evidence.items()
    }
    zero_base = replace(base, scale_evidence=zero_scales)
    refiner = PathsContinuationRefiner(4, ("s4", "s8"))
    output = refiner(zero_base)
    assert isinstance(output, PathsOutput)
    assert torch.equal(output.boundary_correction, torch.zeros_like(output.boundary_correction))
    for evidence in output.spectrum_evidence.values():
        assert torch.equal(evidence.baseline_mass, torch.zeros_like(evidence.baseline_mass))
        assert torch.equal(
            evidence.concentration_spectrum,
            torch.zeros_like(evidence.concentration_spectrum),
        )
        assert torch.isfinite(evidence.concentration_spectrum).all()

    # The inactive branch must also be a mathematically safe extension in
    # backward.  Masking a division by float64 tiny only after it is evaluated
    # can leave a forward-finite graph whose denominator gradient is NaN.
    output.boundary_correction.sum().backward()
    for atoms in zero_atoms.values():
        assert atoms.grad is not None
        assert torch.isfinite(atoms.grad).all()
        assert torch.equal(atoms.grad, torch.zeros_like(atoms.grad))
    for parameter in refiner.parameters():
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()


def test_non_nested_cumulative_atoms_are_rejected() -> None:
    base = _base_output()
    first_name = next(iter(base.scale_evidence))
    scales = dict(base.scale_evidence)
    source = scales[first_name]
    broken = source.compiled_atoms.clone()
    broken[:, 1, 0, 0] = broken[:, 0, 0, 0] + 0.1 * base.atom_mass_cap
    scales[first_name] = replace(source, compiled_atoms=broken)
    with pytest.raises(ValueError, match="violates cumulative ordinal nesting"):
        PathsContinuationRefiner(4, ("s4", "s8"))(
            replace(base, scale_evidence=scales)
        )


class _TinyEncoder(nn.Module):
    STAGE_CHANNELS = {"s4": 3, "s8": 5}

    def forward(self, images, pixel_valid_mask=None):  # pragma: no cover - not used
        raise NotImplementedError


def test_v3_state_keys_load_strictly_without_renaming_the_base_model() -> None:
    origin = OriginModel(
        encoder=_TinyEncoder(),
        pretrained=False,
        evidence_scales=("s4", "s8"),
        projection_dim=7,
    )
    paths = PathsModel(
        encoder=_TinyEncoder(),
        pretrained=False,
        evidence_scales=("s4", "s8"),
        projection_dim=7,
    )
    source = origin.state_dict()
    paths.load_origin_v3_state_dict(source)

    paths_state = paths.state_dict()
    assert set(source).issubset(paths_state)
    assert all(torch.equal(paths_state[key], value) for key, value in source.items())
    assert any(key.startswith("paths_refiner.") for key in paths_state)

    damaged = dict(source)
    damaged.pop(next(iter(damaged)))
    with pytest.raises(ValueError, match="not an exact ORIGIN-v3 state dict"):
        paths.load_origin_v3_state_dict(damaged)
