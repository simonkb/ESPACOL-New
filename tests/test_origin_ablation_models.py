"""Structural tests for the frozen ORIGIN fold-9 ablation models."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from configs.origin_ablation_config import OriginAblationConfig
from models.origin import ConservedOrdinalGenerator, OriginModel
from models.origin_ablation import (
    LedgerSequentialHazardModel,
    MultiScaleMaskedPoolingModel,
    OriginVariantModel,
    build_origin_ablation_model,
    nominal_cumulative_origin_posterior,
)
from models.origin_encoder import OriginEncoderOutput, OriginEncoderScale, OriginScaleMetadata


class _TinyPyramid(nn.Module):
    STAGE_CHANNELS = {"s4": 4, "s8": 6, "s16": 8, "s32": 10}

    def __init__(self) -> None:
        super().__init__()
        self.projections = nn.ModuleDict(
            {name: nn.Conv2d(3, channels, 1) for name, channels in self.STAGE_CHANNELS.items()}
        )

    def forward(self, images: torch.Tensor, pixel_valid_mask=None) -> OriginEncoderOutput:
        if pixel_valid_mask is None:
            pixel_valid_mask = torch.ones(
                images.shape[0], images.shape[-2], images.shape[-1],
                dtype=torch.bool, device=images.device,
            )
        if pixel_valid_mask.ndim == 4:
            pixel_valid_mask = pixel_valid_mask[:, 0]
        sizes = {"s4": (8, 8), "s8": (4, 4), "s16": (2, 2), "s32": (1, 1)}
        result = {}
        for index, name in enumerate(("s4", "s8", "s16", "s32")):
            size = sizes[name]
            resized = F.adaptive_avg_pool2d(images, size)
            features = self.projections[name](resized)
            mask = F.adaptive_avg_pool2d(pixel_valid_mask[:, None].float(), size)[:, 0] >= 0.5
            metadata = OriginScaleMetadata(
                name=name,
                feature_index=index,
                channels=self.STAGE_CHANNELS[name],
                output_stride=4 * (2**index),
                receptive_field=7 + index * 8,
                center_offset=2.0 * (2**index),
                input_size=tuple(images.shape[-2:]),
                lattice_size=size,
            )
            result[name] = OriginEncoderScale(
                features.masked_fill(~mask[:, None], 0.0), mask, metadata
            )
        return OriginEncoderOutput(result)


def _assert_posterior(output, batch: int = 2, classes: int = 5) -> None:
    assert output.class_probs.shape == (batch, classes)
    assert output.cumulative_probs.shape == (batch, classes - 1)
    for name in (
        "class_probs", "log_class_probs", "cumulative_probs", "expected_grade"
    ):
        tensor = getattr(output, name)
        assert tensor.dtype == torch.float64
        assert torch.isfinite(tensor).all()
    torch.testing.assert_close(
        output.class_probs.sum(dim=-1),
        torch.ones(batch, dtype=torch.float64),
        atol=1e-12,
        rtol=1e-12,
    )
    assert torch.all(output.cumulative_probs[:, 1:] <= output.cumulative_probs[:, :-1])


@pytest.mark.parametrize(
    "variant",
    ["pooled_softmax", "pooled_cumulative_logit", "pooled_conditional"],
)
def test_pooled_controls_use_all_scales_and_return_fp64_posteriors(variant: str) -> None:
    model = MultiScaleMaskedPoolingModel(
        variant=variant,
        num_classes=5,
        projection_dim=7,
        pretrained=False,
        encoder=_TinyPyramid(),
    )
    image = torch.randn(2, 3, 32, 32, requires_grad=True)
    mask = torch.ones(2, 32, 32, dtype=torch.bool)
    mask[:, :4] = False
    output = model(image, mask)
    _assert_posterior(output)
    assert tuple(model.scale_projections) == ("s4", "s8", "s16", "s32")
    assert output.local_rate_maps == {}
    assert model.architecture_metadata()["supports_exact_replay"] is False
    (-output.log_class_probs[:, 2].mean()).backward()
    assert image.grad is not None and torch.isfinite(image.grad).all()


@pytest.mark.parametrize(
    "variant",
    ["pooled_softmax", "pooled_cumulative_logit", "pooled_conditional"],
)
def test_pooled_controls_match_nominal_origin_prior_at_initialization(
    variant: str,
) -> None:
    settings = {
        "reference_count": 64.0,
        "atom_rate_init": 2e-4,
        "prior_rate_init": 3e-4,
        "boundary_scale_init": 0.8,
    }
    model = MultiScaleMaskedPoolingModel(
        variant=variant,
        num_classes=5,
        projection_dim=7,
        pretrained=False,
        encoder=_TinyPyramid(),
        **settings,
    ).eval()
    expected_rates, expected_probs = nominal_cumulative_origin_posterior(
        num_classes=5, **settings
    )
    output = model(torch.randn(3, 3, 32, 32))
    torch.testing.assert_close(
        output.class_probs,
        expected_probs.expand(3, -1),
        atol=2e-8,
        rtol=2e-6,
    )
    torch.testing.assert_close(model.nominal_origin_total_rates, expected_rates)
    assert torch.count_nonzero(model.posterior_head.weight) == 0
    metadata = model.architecture_metadata()
    assert metadata["initialization"] == "matched_nominal_cumulative_origin_posterior_v1"
    assert metadata["nominal_origin_total_rates"] == expected_rates.tolist()


@pytest.mark.parametrize(
    "variant",
    ["pooled_softmax", "pooled_cumulative_logit", "pooled_conditional"],
)
def test_pooled_sensitive_head_stays_fp32_inside_outer_autocast(variant: str) -> None:
    model = MultiScaleMaskedPoolingModel(
        variant=variant,
        num_classes=5,
        projection_dim=7,
        pretrained=False,
        encoder=_TinyPyramid(),
    ).eval()
    observed: list[tuple[torch.dtype, torch.dtype]] = []

    def capture(_module, inputs, output):
        observed.append((inputs[0].dtype, output.dtype))

    handle = model.posterior_head.register_forward_hook(capture)
    try:
        with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
            output = model(torch.randn(2, 3, 32, 32))
    finally:
        handle.remove()
    assert observed == [(torch.float32, torch.float32)]
    _assert_posterior(output)


def test_simplex_direct_keeps_competition_but_removes_reverse_cumulative_compile() -> None:
    generator = ConservedOrdinalGenerator(
        stage_channels={"s4": 3},
        num_classes=5,
        hidden_channels=4,
        evidence_scales=("s4",),
        reference_count=8.0,
        atom_rate_init=0.01,
        atom_mode="simplex_direct",
    )
    atoms = torch.tensor([[[[0.1]], [[0.2]], [[0.3]], [[0.4]]]])
    torch.testing.assert_close(generator._compile_atoms(atoms), atoms)
    assert generator.heads["s4"].atom_mode == "simplex_direct"


@pytest.mark.parametrize(
    "variant,atom_mode",
    [
        ("origin_simplex_direct", "simplex_direct"),
        ("origin_independent_sigmoid", "independent"),
    ],
)
def test_direct_atom_controls_match_full_nominal_boundary_rates(
    variant: str, atom_mode: str
) -> None:
    cfg = OriginAblationConfig(
        ablation_variant=variant,
        atom_mode=atom_mode,
        pretrained=False,
        reference_count=64.0,
        atom_rate_init=2e-4,
        prior_rate_init=3e-4,
        boundary_scale_init=0.8,
    )
    model = build_origin_ablation_model(cfg, pretrained=False)
    generator = model.generator
    target_atoms = (
        torch.arange(generator.num_boundaries, 0, -1, dtype=torch.float32)
        * cfg.atom_rate_init
    )
    for name, head in generator.heads.items():
        head.atom_logits.weight.data.zero_()
        _, atoms = head(
            torch.zeros(2, model.encoder.STAGE_CHANNELS[name], 2, 2)
        )
        torch.testing.assert_close(
            atoms,
            target_atoms[None, :, None, None].expand_as(atoms),
            atol=2e-9,
            rtol=2e-5,
        )
    assert (
        model.architecture_metadata()["initialization"]
        == "matched_nominal_cumulative_origin_boundary_rates_v1"
    )


def test_sequential_hazard_uses_identical_ledger_and_replays_deletions() -> None:
    base = OriginVariantModel(
        ablation_variant="ledger_sequential_hazard",
        num_classes=5,
        encoder_name="convnext_tiny",
        pretrained=False,
        encoder=_TinyPyramid(),
        evidence_scales=("s4", "s8", "s16", "s32"),
        projection_dim=7,
        reference_count=32.0,
        atom_rate_init=1e-4,
        prior_rate_init=1e-4,
    )
    model = LedgerSequentialHazardModel(base).eval()
    output = model(torch.randn(2, 3, 32, 32))
    _assert_posterior(output)
    assert torch.equal(output.total_rates, output.ledger_output.total_rates)
    for name in base.evidence_scales:
        assert torch.equal(
            output.local_rate_maps[name], output.ledger_output.local_rate_maps[name]
        )

    removals = {
        name: torch.zeros_like(mask) for name, mask in output.valid_masks.items()
    }
    removals["s4"][:, 0, 0] = True
    replay = model.replay_without(output, removals)
    torch.testing.assert_close(
        replay.output.total_rates,
        output.total_rates - replay.removed_rates,
    )
    assert torch.all(replay.output.expected_grade <= output.expected_grade + 1e-12)
    assert model.architecture_metadata()["source_ledger_identical_to_origin_full"]


def test_origin_full_wrapper_is_behaviorally_identical_to_plain_v3() -> None:
    kwargs = {
        "num_classes": 5,
        "encoder_name": "convnext_tiny",
        "pretrained": False,
        "evidence_scales": ("s4", "s8", "s16", "s32"),
        "projection_dim": 7,
        "reference_count": 32.0,
        "atom_rate_init": 1e-4,
        "prior_rate_init": 1e-4,
    }
    torch.manual_seed(913)
    base = OriginModel(encoder=_TinyPyramid(), **kwargs).eval()
    torch.manual_seed(913)
    wrapped = OriginVariantModel(
        ablation_variant="origin_full", encoder=_TinyPyramid(), **kwargs
    ).eval()
    assert tuple(base.state_dict()) == tuple(wrapped.state_dict())
    for name, value in base.state_dict().items():
        torch.testing.assert_close(value, wrapped.state_dict()[name], rtol=0, atol=0)

    images = torch.randn(2, 3, 32, 32)
    base_output = base(images)
    wrapped_output = wrapped(images)
    torch.testing.assert_close(
        base_output.class_probs, wrapped_output.class_probs, rtol=0, atol=0
    )
    torch.testing.assert_close(
        base_output.total_rates, wrapped_output.total_rates, rtol=0, atol=0
    )
    for scale in base.evidence_scales:
        torch.testing.assert_close(
            base_output.local_rate_maps[scale],
            wrapped_output.local_rate_maps[scale],
            rtol=0,
            atol=0,
        )


def test_scale_subset_heads_match_corresponding_full_initialization() -> None:
    common = {
        "pretrained": False,
        "projection_dim": 8,
    }
    torch.manual_seed(117)
    full = build_origin_ablation_model(
        OriginAblationConfig(ablation_variant="origin_full", **common),
        pretrained=False,
    )
    for variant, scales in (
        ("origin_fine_only", ("s4", "s8")),
        ("origin_coarse_only", ("s16", "s32")),
    ):
        torch.manual_seed(117)
        subset = build_origin_ablation_model(
            OriginAblationConfig(
                ablation_variant=variant,
                evidence_scales=scales,
                **common,
            ),
            pretrained=False,
        )
        for scale in scales:
            full_state = full.generator.heads[scale].state_dict()
            subset_state = subset.generator.heads[scale].state_dict()
            assert tuple(full_state) == tuple(subset_state)
            for name, value in full_state.items():
                torch.testing.assert_close(
                    value, subset_state[name], rtol=0, atol=0
                )
        assert (
            subset.architecture_metadata()["initialization"]
            == "matched_corresponding_full_origin_scale_heads_v1"
        )


def test_ablation_config_rejects_unresolved_variant_overrides() -> None:
    with pytest.raises(ValueError, match="atom_mode='simplex_direct'"):
        OriginAblationConfig(ablation_variant="origin_simplex_direct")
    cfg = OriginAblationConfig(
        ablation_variant="origin_simplex_direct", atom_mode="simplex_direct"
    )
    assert cfg.atom_mode == "simplex_direct"
    with pytest.raises(ValueError, match="consume all native scales"):
        OriginAblationConfig(
            ablation_variant="pooled_softmax", evidence_scales=("s32",)
        )
