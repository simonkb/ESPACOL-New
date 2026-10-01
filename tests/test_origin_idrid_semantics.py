from __future__ import annotations

import math

import torch

from tools.audit_origin_idrid_semantics import (
    binary_average_precision,
    count_matched_nonlesion_mask,
    lesion_cell_mask,
)


def test_binary_average_precision_perfect_and_reversed() -> None:
    labels = torch.tensor([True, False, True, False])
    assert binary_average_precision(torch.tensor([4.0, 1.0, 3.0, 0.0]), labels) == 1.0
    reversed_ap = binary_average_precision(
        torch.tensor([0.0, 4.0, 1.0, 3.0]), labels
    )
    assert math.isclose(reversed_ap, (1 / 3 + 2 / 4) / 2)


def test_lesion_cell_mask_uses_any_pixel_occupancy() -> None:
    mask = torch.zeros(1, 8, 8, dtype=torch.bool)
    mask[0, 1, 1] = True
    mask[0, 6, 6] = True
    cells = lesion_cell_mask(mask, (2, 2))
    assert cells.shape == (1, 2, 2)
    assert torch.equal(cells, torch.tensor([[[True, False], [False, True]]]))


def test_count_matched_control_is_valid_nonlesion_and_deterministic() -> None:
    lesion = torch.zeros(2, 4, 4, dtype=torch.bool)
    lesion[:, 0, :2] = True
    valid = torch.ones_like(lesion)
    valid[:, 3, 3] = False
    first = count_matched_nonlesion_mask(
        lesion, valid, generator=torch.Generator().manual_seed(7)
    )
    second = count_matched_nonlesion_mask(
        lesion, valid, generator=torch.Generator().manual_seed(7)
    )
    assert torch.equal(first, second)
    assert torch.equal(first.sum((1, 2)), lesion.sum((1, 2)))
    assert not bool((first & lesion).any())
    assert not bool((first & ~valid).any())

