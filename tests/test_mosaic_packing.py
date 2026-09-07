"""Focused invariants for MOSAIC Ordinal Receptive-Field Packing (ORFP)."""

import torch

from models.mosaic import (
    nested_witness_probabilities,
    receptive_field_packed_ordinal_evidence,
    receptive_field_packing_mask,
)


def _maximum_pairwise_overlap_fraction(
    mask: torch.Tensor,
    lattice_size: tuple[int, int],
    *,
    output_stride: int,
    receptive_field: float,
) -> float:
    """Independently evaluate equal, theoretical, unclipped RF squares."""

    _, width = lattice_size
    indices = torch.nonzero(mask, as_tuple=False).flatten()
    if indices.numel() < 2:
        return 0.0
    rows = torch.div(indices, width, rounding_mode="floor").float()
    columns = torch.remainder(indices, width).float()
    dy = (rows[:, None] - rows[None, :]).abs() * float(output_stride)
    dx = (columns[:, None] - columns[None, :]).abs() * float(output_stride)
    overlap = (
        (float(receptive_field) - dy).clamp_min(0.0)
        * (float(receptive_field) - dx).clamp_min(0.0)
        / float(receptive_field) ** 2
    )
    overlap.fill_diagonal_(0.0)
    return float(overlap.max())


def test_rf_packing_retains_exact_overlap_boundary() -> None:
    # Adjacent RF-2 boxes at stride 1 intersect over exactly half of either
    # square (IoU=1/3). The declared <= rho rule must retain equality.
    priority = torch.zeros(1, 3)
    valid = torch.ones(1, 3, dtype=torch.bool)
    packed = receptive_field_packing_mask(
        priority,
        valid,
        (1, 3),
        output_stride=1,
        receptive_field=2.0,
        max_overlap=0.5,
    )
    assert torch.equal(packed, valid)


def test_rf_packing_ties_prefer_lower_row_major_index() -> None:
    # Every pair exceeds rho, so a tied field can retain only one source.
    # Stable descending order must choose flat row-major index zero.
    priority = torch.zeros(1, 4)
    valid = torch.ones(1, 4, dtype=torch.bool)
    packed = receptive_field_packing_mask(
        priority,
        valid,
        (2, 2),
        output_stride=1,
        receptive_field=10.0,
        max_overlap=0.0,
    )
    assert torch.equal(
        packed, torch.tensor([[True, False, False, False]])
    )


def test_rf_packing_uses_expected_ordinal_state_and_coherent_normal_exclusion() -> None:
    logits = torch.tensor(
        [[
            [2.0, 0.0, 0.0, 0.0],
            [1.5, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 2.0],
            [0.0, 0.0, 2.0, 0.0],
        ]],
        requires_grad=True,
    )
    valid = torch.ones(1, 4, dtype=torch.bool)
    source = nested_witness_probabilities(logits, valid)
    packed = receptive_field_packed_ordinal_evidence(
        source,
        valid,
        (1, 4),
        output_stride=1,
        receptive_field=10.0,
        max_overlap=0.0,
    )

    # The grade-3-like third site has the greatest sum of cumulative witness
    # probabilities and must win the one-event packing.
    assert torch.equal(
        packed.packing_mask, torch.tensor([[False, False, True, False]])
    )
    assert packed.source_evidence is source
    excluded = ~packed.packing_mask
    expected_normal = torch.zeros_like(
        packed.evidence.state_probabilities[excluded]
    )
    expected_normal[..., 0] = 1.0
    assert torch.equal(
        packed.evidence.state_probabilities[excluded], expected_normal
    )
    assert torch.equal(
        packed.evidence.witness_probabilities[excluded],
        torch.zeros_like(packed.evidence.witness_probabilities[excluded]),
    )
    assert torch.isneginf(
        packed.evidence.log_witness_probabilities[excluded]
    ).all()
    assert torch.equal(
        packed.evidence.log_nonwitness_probabilities[excluded],
        torch.zeros_like(
            packed.evidence.log_nonwitness_probabilities[excluded]
        ),
    )
    assert torch.all(
        packed.evidence.witness_probabilities[..., :-1]
        >= packed.evidence.witness_probabilities[..., 1:]
    )

    packed.evidence.witness_probabilities.sum().backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()
    assert logits.grad[0, 2].abs().sum() > 0
    assert torch.equal(logits.grad[0, [0, 1, 3]], torch.zeros(3, 4))


def test_rf_packing_is_deterministic_and_respects_pairwise_rf_bound() -> None:
    generator = torch.Generator().manual_seed(20260907)
    priority = torch.rand(3, 63, generator=generator)
    # Include both invalid holes and one exact tie in each sample.
    priority[:, 7] = priority[:, 3]
    valid = torch.rand(3, 63, generator=generator) > 0.15
    first = receptive_field_packing_mask(
        priority,
        valid,
        (7, 9),
        output_stride=4,
        receptive_field=15.0,
        max_overlap=0.5,
    )
    second = receptive_field_packing_mask(
        priority,
        valid,
        (7, 9),
        output_stride=4,
        receptive_field=15.0,
        max_overlap=0.5,
    )
    assert torch.equal(first, second)
    assert not bool((first & ~valid).any())
    assert torch.all(first.sum(dim=1) > 0)
    for sample in first:
        observed = _maximum_pairwise_overlap_fraction(
            sample,
            (7, 9),
            output_stride=4,
            receptive_field=15.0,
        )
        assert observed <= 0.5 + 1e-6
