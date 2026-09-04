"""Spatially bounded EfficientNetV2-S encoder for MOSAIC.

The existing backbone in :mod:`models.backbone` deliberately ends in global
average pooling.  MOSAIC has a different contract: every returned vector must
remain attached to one image location.  This module therefore

* consumes one full image canvas (not a bag of tiles),
* stops at a named EfficientNetV2-S spatial stage,
* removes or localizes squeeze--excitation gates that would otherwise
  introduce global image context,
* freezes BatchNorm statistics so training-time normalization cannot mix
  spatial locations, and
* uses only channel-wise (pointwise) operations after the tap.

The ``deep_local_95`` variant keeps the original stem and stages 1--3, then
continues through the complete EfficientNetV2-S semantic hierarchy without
enlarging the stage-3 receptive field.  Every spatial convolution in feature
indices 4--7 is replaced by a grouped 1x1 convolution whose weights are the
spatial sum of the pretrained kernel.  Strides and channel/group structure are
preserved, yielding a deep 1280-channel, stride-32 lattice with exact RF 95.

``output_stride`` is the spacing between lattice sites.  It is *not* the
receptive-field diameter.  Both values are reported explicitly in
:class:`LatticeMetadata`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint_sequential
from torchvision.models import EfficientNet_V2_S_Weights, efficientnet_v2_s
from torchvision.ops.misc import FrozenBatchNorm2d, SqueezeExcitation
from utils.spatial_mask import centered_ellipse_mask


_IMAGENET_MEAN: Tuple[float, float, float] = (0.485, 0.456, 0.406)
_IMAGENET_STD: Tuple[float, float, float] = (0.229, 0.224, 0.225)


@dataclass(frozen=True)
class _SpatialStage:
    """One torchvision EfficientNetV2-S feature stage.

    ``first_stride`` applies to the first block; subsequent blocks have
    stride one.  Each block has exactly one spatial convolution.  Expansion,
    projection, normalization, activation, and residual addition are
    pointwise and do not enlarge the theoretical receptive field.
    """

    feature_index: int
    name: str
    channels: int
    kernel_size: int
    first_stride: int
    blocks: int
    contains_squeeze_excitation: bool = False


# This mirrors torchvision.models.efficientnet._efficientnet_conf("efficientnet_v2_s").
# Feature index 0 is the 3x3/2 stem and index 7 is the final pointwise conv.
_V2S_STAGES: Tuple[_SpatialStage, ...] = (
    _SpatialStage(1, "stage1", 24, 3, 1, 2),
    _SpatialStage(2, "stage2", 48, 3, 2, 4),
    _SpatialStage(3, "stage3", 64, 3, 2, 4),
    _SpatialStage(4, "stage4", 128, 3, 2, 6, True),
    _SpatialStage(5, "stage5", 160, 3, 1, 9, True),
    _SpatialStage(6, "stage6", 256, 3, 2, 15, True),
)


@dataclass(frozen=True)
class ReceptiveFieldMetadata:
    """Exact theoretical geometry of an EfficientNetV2-S tap.

    Values use the standard recurrence

    ``rf' = rf + (kernel - 1) * jump`` and ``jump' = jump * stride``.

    The value is the maximum dependency support through a residual block.  It
    is exact for this encoder because squeeze--excitation and training-time
    BatchNorm statistics are removed from the dependency graph.
    """

    tap: str
    feature_index: int
    channels: int
    output_stride: int
    receptive_field: int
    center_offset: float
    squeeze_excitation_removed: bool
    globally_mixed: bool = False


@dataclass(frozen=True)
class LatticeMetadata:
    """Image-to-lattice coordinate metadata for one forward pass."""

    input_size: Tuple[int, int]
    lattice_size: Tuple[int, int]
    local_dim: int
    receptive_field: ReceptiveFieldMetadata

    @property
    def num_cells(self) -> int:
        return self.lattice_size[0] * self.lattice_size[1]

    @property
    def output_stride(self) -> int:
        return self.receptive_field.output_stride

    def centers_yx(
        self,
        *,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
        flatten: bool = True,
    ) -> torch.Tensor:
        """Return lattice-center coordinates in input-pixel coordinates.

        Pixel centers follow the conventional half-integer coordinate system.
        EfficientNetV2-S uses symmetric padding at every spatial convolution,
        so the first output center remains at ``0.5``.
        """

        height, width = self.lattice_size
        offset = self.receptive_field.center_offset
        stride = self.receptive_field.output_stride
        ys = offset + torch.arange(height, device=device, dtype=dtype) * stride
        xs = offset + torch.arange(width, device=device, dtype=dtype) * stride
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")
        centers = torch.stack((yy, xx), dim=-1)
        return centers.reshape(-1, 2) if flatten else centers


@dataclass
class LocalEncoderOutput:
    """Output of :class:`LocalEfficientNetV2S`.

    ``tokens`` has fixed shape ``(N, P, D_local)``.  Invalid padded/background
    cells are exactly zero and identified by ``valid_mask``.  Keeping a fixed
    ``P`` makes batching possible while allowing the downstream count circuit
    to exclude invalid cells exactly.
    """

    tokens: torch.Tensor
    valid_mask: torch.Tensor
    lattice: LatticeMetadata
    feature_map: Optional[torch.Tensor] = None


def _canonical_tap(tap: Union[str, int]) -> str:
    aliases: Dict[Union[str, int], str] = {
        4: "rf_small",
        "4": "rf_small",
        "s4": "rf_small",
        "stride4": "rf_small",
        "rf_small": "rf_small",
        "small": "rf_small",
        8: "rf_medium",
        "8": "rf_medium",
        "s8": "rf_medium",
        "stride8": "rf_medium",
        "rf_medium": "rf_medium",
        "medium": "rf_medium",
        16: "rf_large",
        "16": "rf_large",
        "s16": "rf_large",
        "stride16": "rf_large",
        "rf_large": "rf_large",
        "large": "rf_large",
        32: "deep_local_95",
        "32": "deep_local_95",
        "s32": "deep_local_95",
        "stride32": "deep_local_95",
        "dl95": "deep_local_95",
        "deep95": "deep_local_95",
        "rf95": "deep_local_95",
        "deep_local_95": "deep_local_95",
    }
    try:
        return aliases[tap]
    except (KeyError, TypeError) as exc:
        raise ValueError(
            f"Unknown local tap {tap!r}; choose rf_small/4, rf_medium/8, "
            "rf_large/16, or deep_local_95/dl95."
        ) from exc


_TAP_TO_FEATURE_INDEX = {
    "rf_small": 2,   # stage2: stride 4, 48 channels
    "rf_medium": 3,  # stage3: stride 8, 64 channels
    "rf_large": 5,   # stage5: stride 16, 160 channels (stage4+5 SE removed)
    # Full semantic hierarchy.  Spatial kernels in features 4--7 are
    # collapsed to 1x1, so downsampling continues while the RF remains 95.
    "deep_local_95": 7,
}


def efficientnet_v2_s_receptive_field(
    tap: Union[str, int] = "rf_medium",
) -> ReceptiveFieldMetadata:
    """Compute exact receptive-field metadata for a supported local tap."""

    canonical = _canonical_tap(tap)
    target_index = _TAP_TO_FEATURE_INDEX[canonical]

    # Input pixels: one-pixel RF, unit jump, first center at 0.5.
    receptive_field = 1
    jump = 1
    center = 0.5

    # torchvision feature[0]: Conv2d(3, 24, kernel=3, stride=2, padding=1).
    kernel, stride, padding = 3, 2, 1
    center += ((kernel - 1) / 2 - padding) * jump
    receptive_field += (kernel - 1) * jump
    jump *= stride

    target_stage: Optional[_SpatialStage] = None
    se_removed = False
    for stage in _V2S_STAGES:
        if stage.feature_index > target_index:
            break
        target_stage = stage
        se_removed = se_removed or stage.contains_squeeze_excitation
        for block_index in range(stage.blocks):
            stride = stage.first_stride if block_index == 0 else 1
            # deep_local_95 preserves stages 0--3 exactly and replaces every
            # later spatial kernel by 1x1.  The stage-4 and stage-6 first-block
            # strides remain, but those blocks no longer enlarge the RF.
            spatial_kernel = (
                1
                if canonical == "deep_local_95" and stage.feature_index >= 4
                else stage.kernel_size
            )
            padding = (spatial_kernel - 1) // 2
            center += ((spatial_kernel - 1) / 2 - padding) * jump
            receptive_field += (spatial_kernel - 1) * jump
            jump *= stride

    if target_stage is None:  # pragma: no cover - protected by the tap table
        raise RuntimeError(f"No EfficientNet stage found for feature index {target_index}")

    channels = 1280 if canonical == "deep_local_95" else target_stage.channels
    return ReceptiveFieldMetadata(
        tap=canonical,
        feature_index=target_index,
        channels=channels,
        output_stride=jump,
        receptive_field=receptive_field,
        center_offset=center,
        squeeze_excitation_removed=se_removed,
        globally_mixed=False,
    )


def available_local_taps() -> Dict[str, ReceptiveFieldMetadata]:
    """Return the supported locality/semantic trade-off points."""

    return {
        name: efficientnet_v2_s_receptive_field(name)
        for name in ("rf_small", "rf_medium", "rf_large", "deep_local_95")
    }


def _as_n1hw(mask: torch.Tensor) -> torch.Tensor:
    if mask.ndim == 2:
        mask = mask.unsqueeze(0).unsqueeze(0)
    elif mask.ndim == 3:
        mask = mask.unsqueeze(1)
    elif mask.ndim == 4 and mask.shape[1] == 1:
        pass
    else:
        raise ValueError(
            "retinal mask must have shape (H,W), (N,H,W), or (N,1,H,W); "
            f"got {tuple(mask.shape)}"
        )
    return mask


def downsample_retinal_field_mask(
    pixel_mask: torch.Tensor,
    output_size: Sequence[int],
    *,
    min_valid_fraction: float = 0.5,
) -> torch.Tensor:
    """Conservatively map a pixel-space retinal mask onto a feature lattice.

    A lattice cell is valid when at least ``min_valid_fraction`` of its
    adaptive pooling bin belongs to the retinal field.  The returned tensor is
    boolean with shape ``(N, H_lattice, W_lattice)``.
    """

    if len(output_size) != 2 or int(output_size[0]) < 1 or int(output_size[1]) < 1:
        raise ValueError(f"output_size must contain two positive values; got {output_size}")
    if not 0.0 <= min_valid_fraction <= 1.0:
        raise ValueError("min_valid_fraction must lie in [0, 1]")

    mask = _as_n1hw(pixel_mask).to(dtype=torch.float32)
    coverage = F.adaptive_avg_pool2d(mask, (int(output_size[0]), int(output_size[1])))
    return (coverage[:, 0] >= min_valid_fraction)


def retinal_field_mask(
    image: torch.Tensor,
    *,
    normalized: bool = True,
    mean: Sequence[float] = _IMAGENET_MEAN,
    std: Sequence[float] = _IMAGENET_STD,
    foreground_threshold: float = 0.04,
    min_axis_fraction: float = 0.01,
    ellipse_scale: float = 1.02,
) -> torch.Tensor:
    """Diagnostically derive a filled retinal-field mask from RGB pixels.

    The common black-padded fundus field is first detected by intensity.  Row
    and column support reject isolated bright border pixels; their bounding
    box defines a filled ellipse, which also fills dark internal structures
    such as vessels and the pupil-like macular region.  The method uses no
    labels and is deliberately non-learned.  This helper is retained for
    acquisition audits only.  The production encoder fallback deliberately
    does *not* call it, because pixel-derived support can leak camera format.

    Args:
        image: ``(N,3,H,W)`` RGB tensor.
        normalized: If true, undo channel normalization with ``mean/std``
            before thresholding.  Thus the helper is safe on the same
            ImageNet-normalized tensor consumed by EfficientNet.
    """

    if image.ndim != 4 or image.shape[1] != 3:
        raise ValueError(f"image must have shape (N,3,H,W); got {tuple(image.shape)}")
    if len(mean) != 3 or len(std) != 3:
        raise ValueError("mean and std must each contain three channel values")
    if not 0.0 <= foreground_threshold <= 1.0:
        raise ValueError("foreground_threshold must lie in [0, 1]")
    if not 0.0 <= min_axis_fraction <= 1.0:
        raise ValueError("min_axis_fraction must lie in [0, 1]")
    if ellipse_scale <= 0:
        raise ValueError("ellipse_scale must be positive")

    with torch.no_grad():
        rgb = image.detach()
        if normalized:
            mean_t = rgb.new_tensor(mean).view(1, 3, 1, 1)
            std_t = rgb.new_tensor(std).view(1, 3, 1, 1)
            rgb = rgb * std_t + mean_t
        rgb = rgb.clamp(0.0, 1.0)

        foreground = rgb.amax(dim=1) > foreground_threshold
        height, width = foreground.shape[-2:]

        row_support = foreground.float().mean(dim=-1) >= min_axis_fraction
        col_support = foreground.float().mean(dim=-2) >= min_axis_fraction

        ys = torch.arange(height, device=image.device).view(1, height)
        xs = torch.arange(width, device=image.device).view(1, width)
        y_min = torch.where(row_support, ys, height).amin(dim=1)
        y_max = torch.where(row_support, ys, -1).amax(dim=1)
        x_min = torch.where(col_support, xs, width).amin(dim=1)
        x_max = torch.where(col_support, xs, -1).amax(dim=1)

        found = (y_max >= y_min) & (x_max >= x_min)
        cy = (y_min + y_max).to(rgb.dtype) / 2.0
        cx = (x_min + x_max).to(rgb.dtype) / 2.0
        ry = ((y_max - y_min + 1).to(rgb.dtype) / 2.0).clamp_min(0.5)
        rx = ((x_max - x_min + 1).to(rgb.dtype) / 2.0).clamp_min(0.5)

        yy = torch.arange(height, device=image.device, dtype=rgb.dtype).view(1, height, 1)
        xx = torch.arange(width, device=image.device, dtype=rgb.dtype).view(1, 1, width)
        ellipse = (
            ((yy - cy[:, None, None]) / (ry[:, None, None] * ellipse_scale)).square()
            + ((xx - cx[:, None, None]) / (rx[:, None, None] * ellipse_scale)).square()
            <= 1.0
        )

        # Empty/all-black images remain empty; this is important for testing
        # and lets the downstream circuit handle the empty-field case honestly.
        return ellipse & found[:, None, None]


class PointwiseSqueezeExcitation(nn.Module):
    """Local replacement for EfficientNet's globally pooled SE gate.

    The pretrained squeeze/excitation channel transforms are retained, but
    they are evaluated independently at every lattice site instead of after
    adaptive global pooling.  This preserves the bounded receptive-field
    contract and, unlike replacing every SE block by identity, retains the
    multiplicative gates that stabilize the deep residual hierarchy.
    """

    def __init__(self, source: SqueezeExcitation) -> None:
        super().__init__()
        self.fc1 = source.fc1
        self.fc2 = source.fc2
        self.activation = source.activation
        self.scale_activation = source.scale_activation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = self.fc1(x)
        scale = self.activation(scale)
        scale = self.fc2(scale)
        scale = self.scale_activation(scale)
        return scale * x


def _replace_global_modules(
    module: nn.Module,
    *,
    pointwise_squeeze_excitation: bool = False,
) -> None:
    """Make an EfficientNet prefix spatially bounded in train and eval modes."""

    for name, child in list(module.named_children()):
        if isinstance(child, SqueezeExcitation):
            # Standard SE pools the complete spatial map.  Historical taps
            # retain their identity replacement for checkpoint compatibility;
            # DL95 keeps a pretrained, per-site channel gate so its much
            # deeper residual stack remains numerically well conditioned.
            replacement: nn.Module
            if pointwise_squeeze_excitation:
                replacement = PointwiseSqueezeExcitation(child)
            else:
                replacement = nn.Identity()
            setattr(module, name, replacement)
        elif isinstance(child, nn.BatchNorm2d):
            # BatchNorm in training mode mixes sites through batch/spatial
            # statistics.  FrozenBatchNorm is an exactly pointwise affine map.
            frozen = FrozenBatchNorm2d(child.num_features, eps=child.eps)
            with torch.no_grad():
                if child.affine:
                    frozen.weight.copy_(child.weight.detach())
                    frozen.bias.copy_(child.bias.detach())
                frozen.running_mean.copy_(child.running_mean.detach())
                frozen.running_var.copy_(child.running_var.detach())
            setattr(module, name, frozen)
        else:
            _replace_global_modules(
                child,
                pointwise_squeeze_excitation=pointwise_squeeze_excitation,
            )


def _collapse_spatial_convolutions_to_pointwise(module: nn.Module) -> None:
    """Replace spatial convolutions by pretrained grouped 1x1 operators.

    For a convolutional kernel ``W``, the replacement is initialized as
    ``W_point = sum_{u,v} W[..., u, v]``.  This is the exact response of the
    original convolution when every value in its spatial support is equal,
    and therefore provides a considerably less disruptive initialization
    than a new random pointwise layer.  Stride, grouping, channels and bias
    are retained; padding is zero because a 1x1 kernel needs none.

    The function is intentionally applied only to EfficientNet feature
    indices 4--7 by :class:`LocalEfficientNetV2S`.  Calling it on earlier
    stages would invalidate the RF-95 contract.
    """

    for name, child in list(module.named_children()):
        if isinstance(child, nn.Conv2d) and child.kernel_size != (1, 1):
            pointwise = nn.Conv2d(
                in_channels=child.in_channels,
                out_channels=child.out_channels,
                kernel_size=1,
                stride=child.stride,
                padding=0,
                dilation=1,
                groups=child.groups,
                bias=child.bias is not None,
                padding_mode="zeros",
                device=child.weight.device,
                dtype=child.weight.dtype,
            )
            with torch.no_grad():
                pointwise.weight.copy_(
                    child.weight.detach().sum(dim=(-2, -1), keepdim=True)
                )
                if child.bias is not None:
                    pointwise.bias.copy_(child.bias.detach())
            pointwise.weight.requires_grad_(child.weight.requires_grad)
            if pointwise.bias is not None and child.bias is not None:
                pointwise.bias.requires_grad_(child.bias.requires_grad)
            setattr(module, name, pointwise)
        else:
            _collapse_spatial_convolutions_to_pointwise(child)


class PointwiseResidualMLP(nn.Module):
    """Channel-only projection and residual MLP applied independently per cell."""

    def __init__(
        self,
        input_dim: int,
        local_dim: int,
        *,
        mlp_ratio: float = 2.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if input_dim < 1 or local_dim < 1:
            raise ValueError("input_dim and local_dim must be positive")
        hidden_dim = max(local_dim, int(round(local_dim * mlp_ratio)))
        self.projection = nn.Linear(input_dim, local_dim)
        self.norm = nn.LayerNorm(local_dim)
        self.fc1 = nn.Linear(local_dim, hidden_dim)
        self.activation = nn.SiLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, local_dim)
        self.output_norm = nn.LayerNorm(local_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x is NHWC. Linear and LayerNorm touch only its last (channel) axis.
        x = self.projection(x)
        residual = x
        x = self.fc1(self.norm(x))
        x = self.dropout(self.activation(x))
        x = self.dropout(self.fc2(x))
        return self.output_norm(residual + x)


class LocalEfficientNetV2S(nn.Module):
    """Full-canvas, spatially bounded EfficientNetV2-S encoder.

    Supported taps:

    * ``rf_small`` / ``4``: stage 2, stride 4, RF 39, 48 channels;
    * ``rf_medium`` / ``8``: stage 3, stride 8, RF 95, 64 channels;
    * ``rf_large`` / ``16``: stage 5, stride 16, RF 559, 160 channels.
    * ``deep_local_95`` / ``dl95``: final stage, stride 32, RF 95,
      1280 channels.

    The large variant removes squeeze--excitation from stages 4--5.  Otherwise
    every nominally local output would depend on the complete image through
    SE's adaptive global pooling and its RF claim would be false.  The deep
    RF-95 variant additionally replaces spatial kernels in features 4--7 by
    pretrained kernel-sum 1x1 operators, retaining the full channel hierarchy
    without increasing the receptive field beyond stage 3.
    """

    TAP_CHANNELS = {name: meta.channels for name, meta in available_local_taps().items()}

    def __init__(
        self,
        *,
        tap: Union[str, int] = "rf_medium",
        local_dim: int = 128,
        pretrained: bool = True,
        mlp_ratio: float = 2.0,
        dropout: float = 0.0,
        grad_checkpoint: bool = False,
        mask_valid_fraction: float = 0.5,
        image_is_normalized: bool = True,
        image_mean: Sequence[float] = _IMAGENET_MEAN,
        image_std: Sequence[float] = _IMAGENET_STD,
        progress: bool = True,
    ) -> None:
        super().__init__()
        if local_dim < 1:
            raise ValueError("local_dim must be positive")
        if not 0.0 <= mask_valid_fraction <= 1.0:
            raise ValueError("mask_valid_fraction must lie in [0, 1]")

        self.tap = _canonical_tap(tap)
        self.rf_metadata = efficientnet_v2_s_receptive_field(self.tap)
        self.local_dim = local_dim
        self.grad_checkpoint = grad_checkpoint
        self.mask_valid_fraction = mask_valid_fraction
        self.image_is_normalized = image_is_normalized
        self.image_mean = tuple(float(value) for value in image_mean)
        self.image_std = tuple(float(value) for value in image_std)

        # weights=None is fully offline and never attempts a download.
        weights = EfficientNet_V2_S_Weights.DEFAULT if pretrained else None
        base = efficientnet_v2_s(weights=weights, progress=progress)
        all_features = list(base.features.children())
        if self.tap == "deep_local_95":
            # Do not touch features 0--3: they define the exact RF-95 support.
            # Later feature stages retain their downsampling/channel hierarchy
            # but cannot acquire additional spatial context.
            for feature in all_features[4:8]:
                _collapse_spatial_convolutions_to_pointwise(feature)
        prefix = all_features[: self.rf_metadata.feature_index + 1]
        self.trunk = nn.Sequential(*prefix)
        _replace_global_modules(
            self.trunk,
            pointwise_squeeze_excitation=self.tap == "deep_local_95",
        )

        self.pointwise = PointwiseResidualMLP(
            self.rf_metadata.channels,
            local_dim,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
        )

    @property
    def output_stride(self) -> int:
        return self.rf_metadata.output_stride

    @property
    def receptive_field(self) -> int:
        return self.rf_metadata.receptive_field

    @property
    def tap_channels(self) -> int:
        return self.rf_metadata.channels

    def forward_trunk_map(self, image: torch.Tensor) -> torch.Tensor:
        """Return the pretrained tap map before the trainable pointwise head."""
        if image.ndim != 4 or image.shape[1] != 3:
            raise ValueError(f"image must have shape (N,3,H,W); got {tuple(image.shape)}")
        if self.grad_checkpoint and self.training and torch.is_grad_enabled():
            features = checkpoint_sequential(
                self.trunk,
                max(1, len(self.trunk)),
                image,
                use_reentrant=False,
            )
        else:
            features = self.trunk(image)
        return features

    def project_trunk_map(self, features: torch.Tensor) -> torch.Tensor:
        """Apply only cell-wise channel mixing to a cached trunk map."""
        if features.ndim != 4 or features.shape[1] != self.tap_channels:
            raise ValueError(
                f"features must have shape (N,{self.tap_channels},H,W); got "
                f"{tuple(features.shape)}"
            )
        # Keep the pretrained convolutional trunk under AMP, but do not run
        # the newly trained pointwise adapter in FP16.  A finite trunk value
        # can overflow in the adapter's Linear layers before the proof head's
        # later ``.float()`` conversion, permanently poisoning its local
        # logits and log-stop probabilities.  The adapter is small relative
        # to the trunk and is part of the numerically sensitive proof path.
        with torch.autocast(device_type=features.device.type, enabled=False):
            features = features.float().permute(0, 2, 3, 1)
            projected = self.pointwise(features)
        return projected.permute(0, 3, 1, 2).contiguous()

    def forward_map(self, image: torch.Tensor) -> torch.Tensor:
        """Return the pointwise-projected ``(N,D_local,H_l,W_l)`` map."""
        return self.project_trunk_map(self.forward_trunk_map(image))

    def forward(
        self,
        image: torch.Tensor,
        pixel_valid_mask: Optional[torch.Tensor] = None,
        *,
        return_feature_map: bool = False,
    ) -> LocalEncoderOutput:
        feature_map = self.forward_map(image)
        batch, _, lattice_height, lattice_width = feature_map.shape

        if pixel_valid_mask is None:
            # The data path normally supplies this same canonical geometry.
            # Keeping the fallback image-independent closes a bypass through
            # which acquisition framing or brightness could alter proof size.
            pixel_valid_mask = centered_ellipse_mask(
                int(image.shape[-2]),
                int(image.shape[-1]),
                batch_size=batch,
                device=feature_map.device,
            )
        pixel_valid_mask = _as_n1hw(pixel_valid_mask).to(device=feature_map.device)
        if pixel_valid_mask.shape[0] != batch:
            raise ValueError(
                "pixel_valid_mask batch dimension must match image: "
                f"{pixel_valid_mask.shape[0]} != {batch}"
            )

        valid_map = downsample_retinal_field_mask(
            pixel_valid_mask,
            (lattice_height, lattice_width),
            min_valid_fraction=self.mask_valid_fraction,
        )
        valid_flat = valid_map.reshape(batch, -1)
        tokens = feature_map.flatten(2).transpose(1, 2)
        tokens = tokens.masked_fill(~valid_flat.unsqueeze(-1), 0.0)

        masked_map: Optional[torch.Tensor] = None
        if return_feature_map:
            masked_map = feature_map.masked_fill(~valid_map.unsqueeze(1), 0.0)

        lattice = LatticeMetadata(
            input_size=(int(image.shape[-2]), int(image.shape[-1])),
            lattice_size=(lattice_height, lattice_width),
            local_dim=self.local_dim,
            receptive_field=self.rf_metadata,
        )
        return LocalEncoderOutput(
            tokens=tokens,
            valid_mask=valid_flat,
            lattice=lattice,
            feature_map=masked_map,
        )


__all__ = [
    "LatticeMetadata",
    "LocalEfficientNetV2S",
    "LocalEncoderOutput",
    "PointwiseResidualMLP",
    "PointwiseSqueezeExcitation",
    "ReceptiveFieldMetadata",
    "available_local_taps",
    "centered_ellipse_mask",
    "downsample_retinal_field_mask",
    "efficientnet_v2_s_receptive_field",
    "retinal_field_mask",
]
