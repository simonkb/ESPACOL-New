"""Shared, image-independent spatial validity geometry for MOSAIC."""

from __future__ import annotations

from typing import Optional

import torch


def centered_ellipse_mask(
    height: int,
    width: int,
    *,
    batch_size: int = 1,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Return a fixed centred ellipse with shape ``(N,1,H,W)``.

    Geometry is defined at pixel centres in normalized canonical-canvas
    coordinates.  It depends only on ``H`` and ``W``; image content,
    acquisition format, labels, and normalization never enter the result.
    The expanded batch shares storage because callers treat validity masks as
    immutable metadata.
    """

    height, width, batch_size = int(height), int(width), int(batch_size)
    if height <= 0 or width <= 0:
        raise ValueError(
            f"ellipse dimensions must be positive, got {(height, width)}"
        )
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")

    # Integer arithmetic makes boundary membership bit-exact on CPU/GPU and
    # avoids small device-dependent changes in proof size.  This is the
    # pixel-centre ellipse equation with denominators cleared:
    #   ((2y+1-H)/H)^2 + ((2x+1-W)/W)^2 <= 1.
    y = 2 * torch.arange(height, device=device, dtype=torch.int64) + 1 - height
    x = 2 * torch.arange(width, device=device, dtype=torch.int64) + 1 - width
    ellipse = (
        y[:, None].square() * (width * width)
        + x[None, :].square() * (height * height)
        <= (height * width) ** 2
    )
    return ellipse.view(1, 1, height, width).expand(batch_size, -1, -1, -1)


__all__ = ["centered_ellipse_mask"]
