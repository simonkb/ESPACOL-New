"""Multi-scale, spatially explicit ConvNeXt encoder for ORIGIN.

ORIGIN never globally pools an image.  Instead, this module exposes the four
native ConvNeXt-Tiny stage maps.  Every downstream operation is pointwise, so
the receptive field reported for a cell is the receptive field of its named
encoder tap rather than the complete image.

The validity mask is geometry, not evidence.  It is computed once from the
original pixel-space support supplied to :meth:`forward` (or from a fixed
centred ellipse) and travels with every stage.  Intervention code must reuse
these masks; it must never infer a new support or renormalize after deleting
evidence.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from torchvision.models import ConvNeXt_Tiny_Weights, convnext_tiny

from utils.spatial_mask import centered_ellipse_mask


@dataclass(frozen=True)
class OriginScaleMetadata:
    """Geometry and dependency support for one encoder stage."""

    name: str
    feature_index: int
    channels: int
    output_stride: int
    receptive_field: int
    center_offset: float
    input_size: Tuple[int, int]
    lattice_size: Tuple[int, int]
    globally_mixed: bool = False

    @property
    def num_cells(self) -> int:
        return self.lattice_size[0] * self.lattice_size[1]

    def centers_yx(
        self,
        *,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
        flatten: bool = False,
    ) -> torch.Tensor:
        """Return feature-cell centres in input-pixel coordinates."""

        height, width = self.lattice_size
        ys = self.center_offset + torch.arange(
            height, device=device, dtype=dtype
        ) * self.output_stride
        xs = self.center_offset + torch.arange(
            width, device=device, dtype=dtype
        ) * self.output_stride
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")
        result = torch.stack((yy, xx), dim=-1)
        return result.reshape(-1, 2) if flatten else result


@dataclass
class OriginEncoderScale:
    """One spatial feature map and its immutable original-support mask."""

    features: torch.Tensor
    valid_mask: torch.Tensor
    metadata: OriginScaleMetadata


@dataclass
class OriginEncoderOutput:
    """Ordered collection of ConvNeXt stage outputs."""

    scales: Dict[str, OriginEncoderScale]

    @property
    def feature_maps(self) -> Dict[str, torch.Tensor]:
        return {name: scale.features for name, scale in self.scales.items()}

    @property
    def valid_masks(self) -> Dict[str, torch.Tensor]:
        return {name: scale.valid_mask for name, scale in self.scales.items()}

    @property
    def metadata(self) -> Dict[str, OriginScaleMetadata]:
        return {name: scale.metadata for name, scale in self.scales.items()}


@dataclass(frozen=True)
class _StaticStage:
    name: str
    feature_index: int
    channels: int
    output_stride: int
    receptive_field: int
    center_offset: float


# torchvision ConvNeXt-Tiny has a 4x4/4 stem, depths (3, 3, 9, 3), and a
# 7x7 depthwise convolution in every residual block.  The values below follow
# the exact recurrence RF' = RF + (kernel-1)*jump; jump' = jump*stride.
_CONVNEXT_TINY_STAGES: Tuple[_StaticStage, ...] = (
    _StaticStage("s4", 1, 96, 4, 76, 2.0),
    _StaticStage("s8", 3, 192, 8, 224, 4.0),
    _StaticStage("s16", 5, 384, 16, 1096, 8.0),
    _StaticStage("s32", 7, 768, 32, 1688, 16.0),
)


# ORIGIN-v5 spatial receptive-field firewall (SRFF).  The ordinary ConvNeXt
# prefix is stopped at s8, whose cells have RF 224 and jump 8.  Independent
# 16x16 blocks of that lattice are then treated as separate batch elements for
# the complete pretrained deep suffix.  No suffix operation can cross a block
# boundary, so mean-pooling one block has the conservative input-space support
# 224 + (16 - 1) * 8 = 344 pixels.  Adjacent block centres are 16 * 8 = 128
# pixels apart and the first block is centred at input coordinate 64.
_SRFF_WINDOW_CELLS = 16
_SRFF_STAGE = _StaticStage("s128", 7, 768, 128, 344, 64.0)


def origin_convnext_stage_specs() -> Mapping[str, Tuple[int, int, int]]:
    """Return ``name -> (channels, stride, receptive_field)``."""

    return {
        stage.name: (
            stage.channels,
            stage.output_stride,
            stage.receptive_field,
        )
        for stage in _CONVNEXT_TINY_STAGES
    }


def _as_n1hw(mask: torch.Tensor) -> torch.Tensor:
    if mask.ndim == 2:
        mask = mask[None, None]
    elif mask.ndim == 3:
        mask = mask[:, None]
    elif mask.ndim == 4 and mask.shape[1] == 1:
        pass
    else:
        raise ValueError(
            "pixel_valid_mask must have shape (H,W), (N,H,W), or "
            f"(N,1,H,W); got {tuple(mask.shape)}"
        )
    return mask


def downsample_origin_validity(
    pixel_valid_mask: torch.Tensor,
    output_size: Sequence[int],
    *,
    min_valid_fraction: float = 0.5,
) -> torch.Tensor:
    """Map fixed pixel support to a boolean spatial lattice."""

    if len(output_size) != 2 or min(map(int, output_size)) < 1:
        raise ValueError("output_size must contain two positive integers")
    if not 0.0 <= min_valid_fraction <= 1.0:
        raise ValueError("min_valid_fraction must lie in [0, 1]")
    mask = _as_n1hw(pixel_valid_mask).to(dtype=torch.float32)
    coverage = F.adaptive_avg_pool2d(
        mask, (int(output_size[0]), int(output_size[1]))
    )
    return coverage[:, 0] >= min_valid_fraction


class ConvNeXtTinyPyramidEncoder(nn.Module):
    """ConvNeXt-Tiny without its global pooling or classifier.

    The returned maps are the raw outputs of feature indices 1, 3, 5, and 7.
    No cross-location operation is appended to a tap.  Invalid map cells are
    zeroed in the returned tensors, while their boolean masks are retained as
    first-class immutable metadata.
    """

    STAGE_CHANNELS: Dict[str, int] = {
        stage.name: stage.channels for stage in _CONVNEXT_TINY_STAGES
    }

    def __init__(
        self,
        *,
        pretrained: bool = True,
        weights: Optional[ConvNeXt_Tiny_Weights] = None,
        progress: bool = True,
        grad_checkpoint: bool = False,
        mask_valid_fraction: float = 0.5,
    ) -> None:
        super().__init__()
        if not 0.0 <= mask_valid_fraction <= 1.0:
            raise ValueError("mask_valid_fraction must lie in [0, 1]")
        if weights is not None and not isinstance(weights, ConvNeXt_Tiny_Weights):
            raise TypeError("weights must be a ConvNeXt_Tiny_Weights value or None")

        selected_weights = (
            weights
            if weights is not None
            else (ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None)
        )
        base = convnext_tiny(weights=selected_weights, progress=progress)
        # Keep only spatial features. avgpool/classifier are intentionally
        # unreachable from this module.
        self.features = base.features
        self.grad_checkpoint = bool(grad_checkpoint)
        self.mask_valid_fraction = float(mask_valid_fraction)

    @property
    def stage_names(self) -> Tuple[str, ...]:
        return tuple(stage.name for stage in _CONVNEXT_TINY_STAGES)

    def _run_feature(self, module: nn.Module, x: torch.Tensor) -> torch.Tensor:
        if self.grad_checkpoint and self.training and torch.is_grad_enabled():
            return checkpoint(module, x, use_reentrant=False)
        return module(x)

    def forward(
        self,
        image: torch.Tensor,
        pixel_valid_mask: Optional[torch.Tensor] = None,
    ) -> OriginEncoderOutput:
        if image.ndim != 4 or image.shape[1] != 3:
            raise ValueError(
                f"image must have shape (N,3,H,W); got {tuple(image.shape)}"
            )
        if not (image.is_floating_point() and torch.isfinite(image).all()):
            raise FloatingPointError("image must be a finite floating-point tensor")

        batch, _, input_height, input_width = image.shape
        if pixel_valid_mask is None:
            pixel_valid_mask = centered_ellipse_mask(
                input_height,
                input_width,
                batch_size=batch,
                device=image.device,
            )
        pixel_valid_mask = _as_n1hw(pixel_valid_mask).to(device=image.device)
        if pixel_valid_mask.shape[0] == 1 and batch > 1:
            pixel_valid_mask = pixel_valid_mask.expand(batch, -1, -1, -1)
        if pixel_valid_mask.shape[0] != batch:
            raise ValueError(
                "pixel_valid_mask batch dimension must match image: "
                f"{pixel_valid_mask.shape[0]} != {batch}"
            )
        if tuple(pixel_valid_mask.shape[-2:]) != (input_height, input_width):
            raise ValueError(
                "pixel_valid_mask spatial dimensions must match image: "
                f"{tuple(pixel_valid_mask.shape[-2:])} != "
                f"{(input_height, input_width)}"
            )

        by_index = {stage.feature_index: stage for stage in _CONVNEXT_TINY_STAGES}
        result: Dict[str, OriginEncoderScale] = {}
        x = image
        for feature_index, feature_module in enumerate(self.features):
            x = self._run_feature(feature_module, x)
            stage = by_index.get(feature_index)
            if stage is None:
                continue
            if x.shape[1] != stage.channels:
                raise RuntimeError(
                    f"ConvNeXt stage {stage.name} produced {x.shape[1]} channels; "
                    f"expected {stage.channels}"
                )
            valid_mask = downsample_origin_validity(
                pixel_valid_mask,
                x.shape[-2:],
                min_valid_fraction=self.mask_valid_fraction,
            )
            # This masks the exposed ledger only. It does not alter the trunk
            # computation or claim that border features have smaller RFs.
            masked = x.masked_fill(~valid_mask[:, None], 0.0)
            metadata = OriginScaleMetadata(
                name=stage.name,
                feature_index=stage.feature_index,
                channels=stage.channels,
                output_stride=stage.output_stride,
                receptive_field=stage.receptive_field,
                center_offset=stage.center_offset,
                input_size=(int(input_height), int(input_width)),
                lattice_size=(int(x.shape[-2]), int(x.shape[-1])),
            )
            result[stage.name] = OriginEncoderScale(masked, valid_mask, metadata)

        if tuple(result) != self.stage_names:
            raise RuntimeError(
                f"missing ConvNeXt stages: got {tuple(result)}, "
                f"expected {self.stage_names}"
            )
        return OriginEncoderOutput(result)


class ConvNeXtTinySRFFEncoder(ConvNeXtTinyPyramidEncoder):
    """ConvNeXt-Tiny with a hard spatial firewall around its deep suffix.

    The stem and first two ConvNeXt stages run once on the full canvas and
    expose the ordinary ``s4`` and ``s8`` maps.  The immutable-validity-masked
    ``s8`` map is then partitioned into non-overlapping 16x16 windows.  Every
    window runs through feature modules 4--7 as an independent batch item.
    Its final 4x4 map is spatially averaged into one ``s128`` feature.  Therefore
    the deep semantic tower is retained, but information can never propagate
    between windows inside that tower.

    Before window packing, invalid ``s8`` source sites are set to zero using
    the same immutable validity mask exposed with the ``s8`` ledger.  Thus an
    ``s128`` atom cannot acquire information through a source site that the
    model declares invalid.  The mask remains fixed input geometry; it is not
    inferred from learned evidence and is never recomputed by intervention.
    """

    STAGE_CHANNELS: Dict[str, int] = {
        "s4": 96,
        "s8": 192,
        "s128": 768,
    }
    WINDOW_CELLS = _SRFF_WINDOW_CELLS
    SOURCE_SCALE = "s8"

    @property
    def stage_names(self) -> Tuple[str, ...]:
        return ("s4", "s8", "s128")

    def spatial_contract(self) -> Dict[str, object]:
        """Return the fixed firewall contract recorded in checkpoints."""

        return {
            "kind": "nonoverlapping_feature_window_firewall_v1",
            "source_scale": self.SOURCE_SCALE,
            "source_stride": 8,
            "source_receptive_field": 224,
            "window_cells": self.WINDOW_CELLS,
            "window_overlap_cells": 0,
            "sealed_scale": _SRFF_STAGE.name,
            "sealed_channels": _SRFF_STAGE.channels,
            "sealed_output_stride": _SRFF_STAGE.output_stride,
            "sealed_receptive_field": _SRFF_STAGE.receptive_field,
            "sealed_center_offset": _SRFF_STAGE.center_offset,
            "sealed_pooling": "within_window_spatial_mean",
            "suffix_feature_indices": [4, 5, 6, 7],
            "source_masking_inside_trunk": True,
        }

    def forward(
        self,
        image: torch.Tensor,
        pixel_valid_mask: Optional[torch.Tensor] = None,
    ) -> OriginEncoderOutput:
        if image.ndim != 4 or image.shape[1] != 3:
            raise ValueError(
                f"image must have shape (N,3,H,W); got {tuple(image.shape)}"
            )
        if not (image.is_floating_point() and torch.isfinite(image).all()):
            raise FloatingPointError("image must be a finite floating-point tensor")

        batch, _, input_height, input_width = image.shape
        if pixel_valid_mask is None:
            pixel_valid_mask = centered_ellipse_mask(
                input_height,
                input_width,
                batch_size=batch,
                device=image.device,
            )
        pixel_valid_mask = _as_n1hw(pixel_valid_mask).to(device=image.device)
        if pixel_valid_mask.shape[0] == 1 and batch > 1:
            pixel_valid_mask = pixel_valid_mask.expand(batch, -1, -1, -1)
        if pixel_valid_mask.shape[0] != batch:
            raise ValueError(
                "pixel_valid_mask batch dimension must match image: "
                f"{pixel_valid_mask.shape[0]} != {batch}"
            )
        if tuple(pixel_valid_mask.shape[-2:]) != (input_height, input_width):
            raise ValueError(
                "pixel_valid_mask spatial dimensions must match image: "
                f"{tuple(pixel_valid_mask.shape[-2:])} != "
                f"{(input_height, input_width)}"
            )

        prefix_by_index = {
            stage.feature_index: stage
            for stage in _CONVNEXT_TINY_STAGES
            if stage.name in {"s4", "s8"}
        }
        result: Dict[str, OriginEncoderScale] = {}
        x = image
        for feature_index in range(4):
            x = self._run_feature(self.features[feature_index], x)
            stage = prefix_by_index.get(feature_index)
            if stage is None:
                continue
            if x.shape[1] != stage.channels:
                raise RuntimeError(
                    f"ConvNeXt stage {stage.name} produced {x.shape[1]} channels; "
                    f"expected {stage.channels}"
                )
            valid_mask = downsample_origin_validity(
                pixel_valid_mask,
                x.shape[-2:],
                min_valid_fraction=self.mask_valid_fraction,
            )
            masked = x.masked_fill(~valid_mask[:, None], 0.0)
            metadata = OriginScaleMetadata(
                name=stage.name,
                feature_index=stage.feature_index,
                channels=stage.channels,
                output_stride=stage.output_stride,
                receptive_field=stage.receptive_field,
                center_offset=stage.center_offset,
                input_size=(int(input_height), int(input_width)),
                lattice_size=(int(x.shape[-2]), int(x.shape[-1])),
            )
            result[stage.name] = OriginEncoderScale(masked, valid_mask, metadata)

        if tuple(result) != ("s4", "s8"):
            raise RuntimeError(
                f"missing ConvNeXt SRFF prefix stages: got {tuple(result)}"
            )
        source = result["s8"].features
        if source.shape[1] != 192:
            raise RuntimeError(
                "ConvNeXt SRFF source produced "
                f"{source.shape[1]} channels; expected 192"
            )

        window = self.WINDOW_CELLS
        source_height, source_width = map(int, source.shape[-2:])
        if source_height % window or source_width % window:
            raise ValueError(
                "ConvNeXt SRFF requires the s8 lattice dimensions to be "
                f"divisible by {window}; got {(source_height, source_width)}"
            )
        grid_height = source_height // window
        grid_width = source_width // window

        # The two unfold operations are non-overlapping.  Moving windows into
        # the batch dimension is the firewall: subsequent convolutions have no
        # tensor edge across which information from another window can flow.
        windows = source.unfold(2, window, window).unfold(3, window, window)
        windows = windows.permute(0, 2, 3, 1, 4, 5).contiguous()
        windows = windows.reshape(
            batch * grid_height * grid_width,
            source.shape[1],
            window,
            window,
        )
        for feature_index in range(4, 8):
            windows = self._run_feature(self.features[feature_index], windows)
        if windows.shape[1] != _SRFF_STAGE.channels or tuple(windows.shape[-2:]) != (4, 4):
            raise RuntimeError(
                "ConvNeXt SRFF suffix contract changed: expected "
                f"(*,{_SRFF_STAGE.channels},4,4), got {tuple(windows.shape)}"
            )

        sealed = windows.mean(dim=(-2, -1))
        sealed = sealed.reshape(
            batch,
            grid_height,
            grid_width,
            _SRFF_STAGE.channels,
        ).permute(0, 3, 1, 2).contiguous()
        sealed_valid_mask = downsample_origin_validity(
            pixel_valid_mask,
            (grid_height, grid_width),
            min_valid_fraction=self.mask_valid_fraction,
        )
        sealed_masked = sealed.masked_fill(~sealed_valid_mask[:, None], 0.0)
        sealed_metadata = OriginScaleMetadata(
            name=_SRFF_STAGE.name,
            feature_index=_SRFF_STAGE.feature_index,
            channels=_SRFF_STAGE.channels,
            output_stride=_SRFF_STAGE.output_stride,
            receptive_field=_SRFF_STAGE.receptive_field,
            center_offset=_SRFF_STAGE.center_offset,
            input_size=(int(input_height), int(input_width)),
            lattice_size=(grid_height, grid_width),
            globally_mixed=False,
        )
        result[_SRFF_STAGE.name] = OriginEncoderScale(
            sealed_masked,
            sealed_valid_mask,
            sealed_metadata,
        )

        if tuple(result) != self.stage_names:
            raise RuntimeError(
                f"missing ConvNeXt SRFF stages: got {tuple(result)}, "
                f"expected {self.stage_names}"
            )
        return OriginEncoderOutput(result)


__all__ = [
    "ConvNeXtTinyPyramidEncoder",
    "ConvNeXtTinySRFFEncoder",
    "OriginEncoderOutput",
    "OriginEncoderScale",
    "OriginScaleMetadata",
    "downsample_origin_validity",
    "origin_convnext_stage_specs",
]
