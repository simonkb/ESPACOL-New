"""Structural tests for ORIGIN-v5's semantic receptive-field firewall.

The SRFF claim is about tensor dependency, not the apparent resolution of a
heatmap.  These tests therefore exercise the hard window boundary, immutable
source masking, declared input-space support, and the absence of any alternate
classification path.  A cheap deterministic ConvNeXt-shaped stub keeps the
tests suitable for every Slurm preflight while preserving the exact stage
shapes used by the real encoder.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from configs.origin_config import OriginConfig
from models.origin import OriginModel
import models.origin_encoder as origin_encoder_module
from models.origin_encoder import ConvNeXtTinySRFFEncoder
from train_origin import build_parser


class _MeanDownsample(nn.Module):
    """Cheap, local stand-in for a ConvNeXt stage/downsample operation."""

    def __init__(self, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.out_channels = int(out_channels)
        self.stride = int(stride)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        value = value.mean(dim=1, keepdim=True)
        if self.stride > 1:
            value = F.avg_pool2d(
                value,
                kernel_size=self.stride,
                stride=self.stride,
            )
        return value.expand(-1, self.out_channels, -1, -1)


class _InjectedS8(nn.Module):
    """Controllable raw s8 source used to audit isolation and masking."""

    def __init__(self, height: int, width: int) -> None:
        super().__init__()
        self.register_buffer("source", torch.zeros(1, 192, height, width))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        if tuple(value.shape[-2:]) != tuple(self.source.shape[-2:]):
            raise RuntimeError("injected s8 source shape does not match prefix")
        return self.source.expand(value.shape[0], -1, -1, -1)


def _fake_convnext() -> SimpleNamespace:
    # ConvNeXt feature indices 1, 3, 5, and 7 correspond to s4, s8, s16,
    # and s32.  SRFF runs 0--3 on the canvas and 4--7 on packed windows.
    return SimpleNamespace(
        features=nn.ModuleList(
            [
                _MeanDownsample(96, 4),
                nn.Identity(),
                _MeanDownsample(192, 2),
                nn.Identity(),
                _MeanDownsample(384, 2),
                nn.Identity(),
                _MeanDownsample(768, 2),
                nn.Identity(),
            ]
        )
    )


def _srff_encoder(monkeypatch: pytest.MonkeyPatch) -> ConvNeXtTinySRFFEncoder:
    monkeypatch.setattr(
        origin_encoder_module,
        "convnext_tiny",
        lambda **_: _fake_convnext(),
    )
    return ConvNeXtTinySRFFEncoder(pretrained=False).eval()


def test_srff_640_contract_shapes_and_receptive_field_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    encoder = _srff_encoder(monkeypatch)
    image = torch.zeros(1, 3, 640, 640)
    valid = torch.ones(1, 1, 640, 640, dtype=torch.bool)

    with torch.no_grad():
        output = encoder(image, valid)

    assert tuple(output.scales) == ("s4", "s8", "s128")
    assert [tuple(item.features.shape) for item in output.scales.values()] == [
        (1, 96, 160, 160),
        (1, 192, 80, 80),
        (1, 768, 5, 5),
    ]
    sealed = output.scales["s128"]
    assert sealed.metadata.output_stride == 128
    assert sealed.metadata.receptive_field == 344
    assert sealed.metadata.center_offset == 64.0
    assert sealed.metadata.lattice_size == (5, 5)
    assert sealed.metadata.input_size == (640, 640)
    assert not sealed.metadata.globally_mixed
    centres = sealed.metadata.centers_yx()
    torch.testing.assert_close(centres[0, 0], torch.tensor([64.0, 64.0]))
    torch.testing.assert_close(centres[-1, -1], torch.tensor([576.0, 576.0]))

    contract = encoder.spatial_contract()
    assert contract["source_scale"] == "s8"
    assert contract["source_stride"] == 8
    assert contract["source_receptive_field"] == 224
    assert contract["window_cells"] == 16
    assert contract["window_overlap_cells"] == 0
    assert contract["sealed_scale"] == "s128"
    assert contract["sealed_output_stride"] == 128
    assert contract["sealed_receptive_field"] == 344
    assert contract["sealed_center_offset"] == 64.0
    assert contract["source_masking_inside_trunk"] is True


def test_srff_deep_suffix_cannot_cross_window_boundaries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    encoder = _srff_encoder(monkeypatch)
    injected = _InjectedS8(32, 32)
    encoder.features[3] = injected
    image = torch.zeros(1, 3, 256, 256)
    valid = torch.ones(1, 1, 256, 256, dtype=torch.bool)

    with torch.no_grad():
        baseline = encoder(image, valid).scales["s128"].features.clone()
        injected.source[:, :, :16, :16] = 3.0
        changed = encoder(image, valid).scales["s128"].features

    delta = changed - baseline
    assert float(delta[:, :, 0, 0].abs().max()) > 0.0
    assert torch.equal(delta[:, :, 0, 1], torch.zeros_like(delta[:, :, 0, 1]))
    assert torch.equal(delta[:, :, 1, 0], torch.zeros_like(delta[:, :, 1, 0]))
    assert torch.equal(delta[:, :, 1, 1], torch.zeros_like(delta[:, :, 1, 1]))


def test_srff_invalid_s8_source_values_cannot_reach_sealed_ledger(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    encoder = _srff_encoder(monkeypatch)
    injected = _InjectedS8(32, 32)
    encoder.features[3] = injected
    image = torch.zeros(1, 3, 256, 256)
    # Only one 8x8 corner of the first 16x16 s8 window is invalid.  The
    # resulting s128 cell still has 75% valid pixel coverage and therefore
    # remains exposed.  This prevents final s128 output masking from hiding a
    # failure to mask the suffix's s8 source tensor.
    pixel_valid = torch.ones(1, 1, 256, 256, dtype=torch.bool)
    pixel_valid[:, :, :64, :64] = False
    with torch.no_grad():
        injected.source.fill_(0.25)
        baseline_output = encoder(image, pixel_valid)
        baseline = baseline_output.scales["s128"].features.clone()
        injected.source[:, :, :8, :8] = 1e6
        changed_output = encoder(image, pixel_valid)
        changed = changed_output.scales["s128"].features

    assert torch.equal(changed, baseline)
    assert bool(changed_output.scales["s128"].valid_mask[0, 0, 0])
    exposed_s8 = changed_output.scales["s8"]
    invalid_s8 = ~exposed_s8.valid_mask
    assert torch.equal(
        exposed_s8.features.masked_select(invalid_s8[:, None]),
        torch.zeros_like(exposed_s8.features.masked_select(invalid_s8[:, None])),
    )
    invalid_sealed = ~changed_output.scales["s128"].valid_mask
    assert torch.equal(
        changed.masked_select(invalid_sealed[:, None]),
        torch.zeros_like(changed.masked_select(invalid_sealed[:, None])),
    )


def test_srff_fails_closed_when_s8_cannot_be_partitioned(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    encoder = _srff_encoder(monkeypatch)
    image = torch.zeros(1, 3, 192, 192)  # s8 is 24x24, not divisible by 16.
    valid = torch.ones(1, 1, 192, 192, dtype=torch.bool)
    with pytest.raises(ValueError, match="s8 lattice dimensions.*divisible by 16"):
        encoder(image, valid)


@pytest.mark.parametrize(
    ("encoder", "scales"),
    [
        ("convnext_tiny_srff", ("s4", "s8", "s16")),
        ("convnext_tiny", ("s4", "s8", "s128")),
    ],
)
def test_srff_and_native_encoders_reject_incompatible_evidence_scales(
    encoder: str,
    scales: tuple[str, ...],
) -> None:
    with pytest.raises(ValueError, match="evidence_scales are incompatible"):
        OriginConfig(encoder=encoder, evidence_scales=scales)


def test_srff_config_rejects_nondivisible_canvas_and_cli_accepts_contract() -> None:
    with pytest.raises(ValueError, match="img_size divisible by 128"):
        OriginConfig(
            encoder="convnext_tiny_srff",
            evidence_scales=("s4", "s8", "s128"),
            img_size=600,
        )

    args = build_parser().parse_args(
        ["--encoder", "convnext_tiny_srff", "--scales", "s4,s8,s128"]
    )
    assert args.encoder == "convnext_tiny_srff"
    assert args.scales == ("s4", "s8", "s128")


def test_srff_full_generator_path_has_finite_backward(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    encoder = _srff_encoder(monkeypatch).train()
    model = OriginModel(
        encoder_name="convnext_tiny_srff",
        encoder=encoder,
        evidence_scales=("s4", "s8", "s128"),
        projection_dim=8,
        reference_count=32.0,
    ).train()
    image = torch.randn(2, 3, 256, 256, requires_grad=True)
    valid = torch.ones(2, 1, 256, 256, dtype=torch.bool)

    output = model(image, valid)
    labels = torch.tensor([0, 4])
    loss = -output.log_class_probs[torch.arange(2), labels].mean()
    loss.backward()

    assert image.grad is not None and bool(torch.isfinite(image.grad).all())
    gradients = [
        parameter.grad
        for parameter in model.parameters()
        if parameter.requires_grad
    ]
    assert gradients and all(gradient is not None for gradient in gradients)
    assert all(bool(torch.isfinite(gradient).all()) for gradient in gradients)
    assert bool(torch.isfinite(output.total_rates).all())
    assert float(output.total_rates.detach().max()) <= model.generator.total_rate_cap


def test_srff_architecture_metadata_declares_firewall_and_no_bypass(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    encoder = _srff_encoder(monkeypatch)
    model = OriginModel(
        encoder_name="convnext_tiny_srff",
        encoder=encoder,
        evidence_scales=("s4", "s8", "s128"),
        projection_dim=8,
    )
    metadata = model.architecture_metadata()

    assert metadata["encoder"] == "convnext_tiny_srff"
    assert metadata["evidence_scales"] == ["s4", "s8", "s128"]
    assert metadata["evidence_dependency_policy"] == (
        "spatial_receptive_field_firewall_v1"
    )
    assert metadata["srff_window_cells"] == 16
    assert metadata["sealed_source_masking"] is True
    assert metadata["max_evidence_receptive_field"] == 344
    assert metadata["no_classifier_bypass"] is True
    contract = metadata["encoder_spatial_contract"]
    assert contract["kind"] == "nonoverlapping_feature_window_firewall_v1"
    assert contract["source_masking_inside_trunk"] is True
    assert contract["suffix_feature_indices"] == [4, 5, 6, 7]
    assert not hasattr(encoder, "avgpool")
    assert not hasattr(encoder, "classifier")


def test_origin_legacy_defaults_remain_unchanged() -> None:
    cfg = OriginConfig()
    args = build_parser().parse_args([])

    assert cfg.encoder == "convnext_tiny"
    assert cfg.evidence_scales == ("s4", "s8", "s16", "s32")
    assert args.encoder == "convnext_tiny"
    assert args.scales == ("s4", "s8", "s16", "s32")
