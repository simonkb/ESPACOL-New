"""Focused invariants for MOSAIC's spatially bounded local encoder."""

import pytest
import torch
import torch.nn as nn
from torchvision.ops.misc import SqueezeExcitation

from models.local_efficientnet import (
    LocalEfficientNetV2S,
    PointwiseResidualMLP,
    PointwiseSqueezeExcitation,
    _collapse_spatial_convolutions_to_pointwise,
    available_local_taps,
    centered_ellipse_mask,
    downsample_retinal_field_mask,
    efficientnet_v2_s_receptive_field,
    retinal_field_mask,
)


@pytest.mark.parametrize(
    "tap,feature_index,channels,stride,receptive_field",
    [
        ("rf_small", 2, 48, 4, 39),
        ("rf_medium", 3, 64, 8, 95),
        ("rf_large", 5, 160, 16, 559),
        ("deep_local_95", 7, 1280, 32, 95),
    ],
)
def test_exact_receptive_field_metadata(
    tap, feature_index, channels, stride, receptive_field
):
    metadata = efficientnet_v2_s_receptive_field(tap)
    assert metadata.feature_index == feature_index
    assert metadata.channels == channels
    assert metadata.output_stride == stride
    assert metadata.receptive_field == receptive_field
    assert metadata.center_offset == 0.5
    assert metadata.globally_mixed is False


def test_tap_aliases_and_public_channels():
    assert efficientnet_v2_s_receptive_field(4).tap == "rf_small"
    assert efficientnet_v2_s_receptive_field("s8").tap == "rf_medium"
    assert efficientnet_v2_s_receptive_field("stride16").tap == "rf_large"
    assert efficientnet_v2_s_receptive_field("dl95").tap == "deep_local_95"
    assert efficientnet_v2_s_receptive_field("stride32").tap == "deep_local_95"
    assert LocalEfficientNetV2S.TAP_CHANNELS == {
        "rf_small": 48,
        "rf_medium": 64,
        "rf_large": 160,
        "deep_local_95": 1280,
    }
    assert set(available_local_taps()) == {
        "rf_small",
        "rf_medium",
        "rf_large",
        "deep_local_95",
    }
    with pytest.raises(ValueError, match="Unknown local tap"):
        efficientnet_v2_s_receptive_field("stride64")


def test_pointwise_head_cannot_spread_a_spatial_perturbation():
    torch.manual_seed(1)
    head = PointwiseResidualMLP(4, 7, dropout=0.0).eval()
    base = torch.zeros(2, 5, 6, 4)
    changed = base.clone()
    changed[:, 2, 3] = 1.0

    delta = (head(changed) - head(base)).abs().sum(dim=-1)
    assert torch.count_nonzero(delta[:, 2, 3]) == 2
    outside = delta.clone()
    outside[:, 2, 3] = 0
    assert torch.count_nonzero(outside) == 0


def test_dl95_kernel_collapse_preserves_direction_caps_norm_and_is_fixed():
    conv = nn.Conv2d(
        4,
        4,
        kernel_size=3,
        stride=2,
        padding=1,
        groups=4,
        bias=True,
    )
    with torch.no_grad():
        conv.weight.copy_(torch.arange(36, dtype=torch.float32).reshape(4, 1, 3, 3))
        # A near-cancelling filter exercises the one-sided nature of the cap:
        # safe small sums must not be amplified to the source norm.
        conv.weight[3].zero_()
        conv.weight[3, 0, 0, 0] = 1.0
        conv.weight[3, 0, 0, 1] = -0.75
        conv.bias.copy_(torch.arange(4, dtype=torch.float32))
    source = conv.weight.detach().clone()
    spatial_sum = source.sum(dim=(-2, -1), keepdim=True)
    module = nn.Sequential(conv)

    _collapse_spatial_convolutions_to_pointwise(module)
    collapsed = module[0]

    assert collapsed.kernel_size == (1, 1)
    assert collapsed.stride == (2, 2)
    assert collapsed.groups == 4
    collapsed_norm = collapsed.weight.flatten(1).norm(dim=1)
    source_norm = source.flatten(1).norm(dim=1)
    assert torch.all(collapsed_norm <= source_norm + 1e-6)
    assert torch.all(collapsed.weight > 0)
    torch.testing.assert_close(collapsed_norm[:3], source_norm[:3])
    torch.testing.assert_close(collapsed.weight[3], spatial_sum[3])
    assert torch.equal(collapsed.bias, torch.arange(4, dtype=torch.float32))
    assert collapsed.weight.requires_grad is False
    assert collapsed.bias.requires_grad is False


def test_dl95_retains_spatial_kernels_only_through_feature_three():
    model = LocalEfficientNetV2S(
        tap="dl95", local_dim=8, pretrained=False, dropout=0.0
    )
    early_spatial = [
        module
        for module in model.trunk[:4].modules()
        if isinstance(module, nn.Conv2d) and module.kernel_size != (1, 1)
    ]
    late_spatial = [
        module
        for module in model.trunk[4:].modules()
        if isinstance(module, nn.Conv2d) and module.kernel_size != (1, 1)
    ]
    assert early_spatial
    assert late_spatial == []


def test_dl95_pointwise_squeeze_excitation_cannot_mix_sites():
    torch.manual_seed(17)
    gate = PointwiseSqueezeExcitation(SqueezeExcitation(4, 2)).eval()
    base = torch.zeros(1, 4, 3, 3)
    changed = base.clone()
    changed[:, :, 1, 1] = 1.0

    with torch.no_grad():
        delta = (gate(changed) - gate(base)).abs().sum(dim=1)

    assert delta[0, 1, 1] > 0
    delta[0, 1, 1] = 0
    assert torch.count_nonzero(delta) == 0


def test_pointwise_projection_is_fp32_under_encoder_autocast():
    """AMP may cover the trunk, but never the proof-path adapter."""

    model = LocalEfficientNetV2S(
        tap="rf_small", local_dim=8, pretrained=False, dropout=0.0
    ).eval()
    trunk_features = torch.full((1, model.tap_channels, 2, 2), 4.0e4)

    with torch.no_grad(), torch.autocast(
        device_type="cpu", dtype=torch.bfloat16
    ):
        projected = model.project_trunk_map(trunk_features)

    assert projected.dtype == torch.float32
    assert torch.isfinite(projected).all()


def test_dl95_trunk_is_fp32_under_outer_autocast():
    model = LocalEfficientNetV2S(
        tap="dl95",
        local_dim=8,
        pretrained=False,
        dropout=0.0,
        grad_checkpoint=True,
    ).train()
    image = torch.randn(1, 3, 64, 64, requires_grad=True)

    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        features = model.forward_trunk_map(image)
        loss = features.square().mean()
    loss.backward()

    assert features.dtype == torch.float32
    assert torch.isfinite(features).all()
    assert image.grad is not None
    assert image.grad.dtype == torch.float32
    assert torch.isfinite(image.grad).all()


def test_mask_downsampling_uses_coverage_and_preserves_empty_field():
    mask = torch.zeros(2, 8, 8)
    mask[0, :4, :4] = 1
    down = downsample_retinal_field_mask(mask, (2, 2), min_valid_fraction=0.5)
    assert down.dtype == torch.bool
    assert down.shape == (2, 2, 2)
    assert torch.equal(down[0], torch.tensor([[True, False], [False, False]]))
    assert not down[1].any()


def test_retinal_mask_is_identical_for_raw_and_imagenet_normalized_input():
    height = width = 64
    yy, xx = torch.meshgrid(torch.arange(height), torch.arange(width), indexing="ij")
    disk = ((yy - 31.5).square() + (xx - 31.5).square()) <= 27**2
    raw = torch.zeros(1, 3, height, width)
    raw[:, 0, disk] = 0.45
    raw[:, 1, disk] = 0.18
    raw[:, 2, disk] = 0.08
    mean = raw.new_tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1)
    std = raw.new_tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1)
    normalized = (raw - mean) / std

    raw_mask = retinal_field_mask(raw, normalized=False)
    normalized_mask = retinal_field_mask(normalized, normalized=True)
    assert torch.equal(raw_mask, normalized_mask)
    assert raw_mask[:, 32, 32].all()
    assert not raw_mask[:, 0, 0].any()


def test_empty_image_produces_empty_retinal_mask():
    assert not retinal_field_mask(torch.zeros(2, 3, 32, 32), normalized=False).any()


@pytest.mark.parametrize(
    "tap,expected_hw",
    [
        ("rf_small", 16),
        ("rf_medium", 8),
        ("rf_large", 4),
        ("dl95", 2),
    ],
)
def test_encoder_output_shape_masking_and_lattice_metadata(tap, expected_hw):
    torch.manual_seed(3)
    model = LocalEfficientNetV2S(
        tap=tap,
        local_dim=12,
        pretrained=False,
        dropout=0.0,
        image_is_normalized=False,
    ).eval()
    image = torch.rand(1, 3, 64, 64)
    pixel_mask = torch.ones(1, 64, 64)
    pixel_mask[:, :, 32:] = 0

    with torch.no_grad():
        output = model(image, pixel_mask, return_feature_map=True)

    assert output.tokens.shape == (1, expected_hw * expected_hw, 12)
    assert output.valid_mask.shape == (1, expected_hw * expected_hw)
    assert output.feature_map.shape == (1, 12, expected_hw, expected_hw)
    assert output.lattice.lattice_size == (expected_hw, expected_hw)
    assert output.lattice.num_cells == expected_hw * expected_hw
    assert output.lattice.output_stride in (4, 8, 16, 32)
    assert torch.count_nonzero(output.tokens[~output.valid_mask]) == 0
    assert torch.count_nonzero(output.feature_map[:, :, :, expected_hw // 2 :]) == 0

    centers = output.lattice.centers_yx()
    assert centers.shape == (expected_hw * expected_hw, 2)
    assert torch.equal(centers[0], torch.tensor([0.5, 0.5]))


@pytest.mark.parametrize("tap", ["rf_large", "dl95"])
def test_encoder_contains_no_global_or_training_time_spatial_normalization(tap):
    model = LocalEfficientNetV2S(
        tap=tap, local_dim=8, pretrained=False
    )
    forbidden = (nn.AdaptiveAvgPool2d, nn.BatchNorm2d, nn.MultiheadAttention)
    offenders = [module for module in model.trunk.modules() if isinstance(module, forbidden)]
    assert offenders == []


def test_real_small_encoder_does_not_leak_a_distant_pixel_to_center_cell():
    torch.manual_seed(5)
    model = LocalEfficientNetV2S(
        tap="rf_small", local_dim=8, pretrained=False, image_is_normalized=False
    ).eval()
    base = torch.zeros(1, 3, 64, 64)
    perturbed = base.clone()
    perturbed[:, :, 0, 0] = 1.0

    with torch.no_grad():
        base_map = model.forward_map(base)
        perturbed_map = model.forward_map(perturbed)

    # The stride-4 center cell has RF 39 and therefore cannot depend on (0,0).
    assert torch.equal(base_map[:, :, 8, 8], perturbed_map[:, :, 8, 8])


def test_real_dl95_encoder_keeps_the_stage_three_receptive_field():
    torch.manual_seed(23)
    model = LocalEfficientNetV2S(
        tap="dl95", local_dim=8, pretrained=False, image_is_normalized=False
    ).eval()
    base = torch.zeros(1, 3, 128, 128)
    perturbed = base.clone()
    perturbed[:, :, 0, 0] = 1.0

    with torch.no_grad():
        base_map = model.forward_map(base)
        perturbed_map = model.forward_map(perturbed)

    # Output (2,2) is centred at 64.5 px.  Its exact RF-95 support starts at
    # 17.5 px, so the corner perturbation cannot alter this evidence value.
    assert base_map.shape[-2:] == (4, 4)
    assert torch.equal(base_map[:, :, 2, 2], perturbed_map[:, :, 2, 2])


def test_encoder_fallback_mask_is_fixed_across_canonical_image_textures():
    torch.manual_seed(19)
    dark = torch.zeros(1, 3, 64, 64)
    textured = torch.randn(1, 3, 64, 64) * 3.0
    model = LocalEfficientNetV2S(
        tap="rf_medium", local_dim=8, pretrained=False, image_is_normalized=True
    ).eval()

    with torch.no_grad():
        dark_output = model(dark)
        textured_output = model(textured)

    assert torch.equal(dark_output.valid_mask, textured_output.valid_mask)
    expected = downsample_retinal_field_mask(
        centered_ellipse_mask(64, 64), (8, 8)
    ).flatten(1)
    assert torch.equal(dark_output.valid_mask, expected)
    assert dark_output.valid_mask.any()
    assert (~dark_output.valid_mask).any()
    assert torch.count_nonzero(dark_output.tokens[~dark_output.valid_mask]) == 0
    assert torch.count_nonzero(textured_output.tokens[~textured_output.valid_mask]) == 0


def test_pixel_mask_batch_must_match_image_batch():
    model = LocalEfficientNetV2S(tap="rf_small", local_dim=8, pretrained=False)
    with pytest.raises(ValueError, match="batch dimension"):
        model(torch.zeros(2, 3, 64, 64), torch.ones(1, 64, 64))
