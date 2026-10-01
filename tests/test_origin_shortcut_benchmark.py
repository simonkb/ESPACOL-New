"""Contract tests for the preregistered controlled ordinal shortcuts."""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

from benchmarks.ordinal_shortcut import (
    FactorialShortcutDataset,
    OrdinalShortcutDataset,
    OrdinalShortcutProtocol,
    make_shortcut_loaders,
)
from benchmarks.shortcut_metrics import (
    GateBThresholds,
    assess_diffuse_nonfocal_gate,
    assess_gate_b,
    binary_average_precision,
    bootstrap_spearman_image_cluster_interval,
    boundary_response_matrix,
    boundary_response_selectivity,
    familywise_localization_permutation_test,
    internal_pixel_effect_audit,
    localization_metrics,
    rasterize_native_ledger,
    spearman_correlation,
)
from models.origin import OriginModel
from models.origin_ablation import MultiScaleMaskedPoolingModel
from models.origin_acceptance_baselines import FixedCountLocalEvidenceModel
from models.origin_encoder import (
    OriginEncoderOutput,
    OriginEncoderScale,
    OriginScaleMetadata,
)
from scripts.audit_origin_shortcut import _main_audit, _transform_parameter_hash
from scripts.origin_shortcut_comparator_common import (
    COMPARATOR_VARIANTS,
    PROTOCOL_CORE_SHA256,
    TASK_COUNT,
    all_tasks,
    canonical_sha256,
    protocol_core,
    task_at,
)


class _CleanDataset(Dataset):
    def __init__(self, count: int = 100, size: int = 48, stochastic: bool = False):
        self.labels = [index % 5 for index in range(count)]
        self.targets = self.labels
        self.size = size
        self.stochastic = stochastic

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int):
        yy, xx = torch.meshgrid(
            torch.arange(self.size), torch.arange(self.size), indexing="ij"
        )
        radius = (self.size - 2) / 2.0
        valid = (
            (yy - (self.size - 1) / 2.0).square()
            + (xx - (self.size - 1) / 2.0).square()
            <= radius**2
        )[None]
        jitter = random.random() if self.stochastic else 0.0
        image = torch.full((3, self.size, self.size), index / 1000.0 + jitter)
        return image, valid, torch.tensor(self.labels[index]), index


class _TinyPyramid(nn.Module):
    """Small four-scale encoder for a full render/model/audit smoke test."""

    STAGE_CHANNELS = {"s4": 4, "s8": 5, "s16": 6, "s32": 7}

    def __init__(self) -> None:
        super().__init__()
        self.projections = nn.ModuleDict(
            {
                name: nn.Conv2d(3, channels, kernel_size=1)
                for name, channels in self.STAGE_CHANNELS.items()
            }
        )

    def forward(self, images: torch.Tensor, pixel_valid_mask=None) -> OriginEncoderOutput:
        if pixel_valid_mask is None:
            pixel_valid_mask = torch.ones(
                images.shape[0], *images.shape[-2:], dtype=torch.bool,
                device=images.device,
            )
        if pixel_valid_mask.ndim == 4:
            pixel_valid_mask = pixel_valid_mask[:, 0]
        sizes = {"s4": (8, 8), "s8": (4, 4), "s16": (2, 2), "s32": (1, 1)}
        scales = {}
        for index, name in enumerate(("s4", "s8", "s16", "s32")):
            size = sizes[name]
            features = self.projections[name](F.adaptive_avg_pool2d(images, size))
            valid = F.adaptive_avg_pool2d(
                pixel_valid_mask[:, None].float(), size
            )[:, 0] >= 0.5
            metadata = OriginScaleMetadata(
                name=name,
                feature_index=index,
                channels=self.STAGE_CHANNELS[name],
                output_stride=4 * 2**index,
                receptive_field=7 + 8 * index,
                center_offset=float(2 * 2**index),
                input_size=tuple(images.shape[-2:]),
                lattice_size=size,
            )
            scales[name] = OriginEncoderScale(
                features.masked_fill(~valid[:, None], 0.0), valid, metadata
            )
        return OriginEncoderOutput(scales)


def test_render_is_deterministic_and_does_not_advance_global_rngs() -> None:
    dataset = OrdinalShortcutDataset(
        _CleanDataset(stochastic=True), family="localized", condition="aligned"
    )
    random.seed(81)
    np.random.seed(81)
    torch.manual_seed(81)
    expected_python = random.random()
    expected_numpy = float(np.random.random())
    expected_torch = float(torch.rand(()))

    random.seed(81)
    np.random.seed(81)
    torch.manual_seed(81)
    first = dataset.render(7)
    observed_python = random.random()
    observed_numpy = float(np.random.random())
    observed_torch = float(torch.rand(()))
    second = dataset.render(7)

    assert torch.equal(first.image, second.image)
    assert first.metadata.to_record() == second.metadata.to_record()
    assert observed_python == expected_python
    assert observed_numpy == expected_numpy
    assert observed_torch == expected_torch


def test_aligned_bits_have_independent_boundary_noise_at_declared_rates() -> None:
    dataset = OrdinalShortcutDataset(
        _CleanDataset(count=800, size=32), family="localized", condition="aligned"
    )
    targets = []
    markers = []
    for index in range(len(dataset)):
        metadata = dataset.render(index).metadata
        targets.append(metadata.target_bits)
        markers.append(metadata.marker_bits)
    target = np.asarray(targets, dtype=bool)
    marker = np.asarray(markers, dtype=bool)
    for boundary in range(4):
        true_positive = marker[target[:, boundary], boundary].mean()
        false_positive = marker[~target[:, boundary], boundary].mean()
        assert true_positive == pytest.approx(0.90, abs=0.06)
        assert false_positive == pytest.approx(0.10, abs=0.06)
    # Independent noise creates many non-nested patterns; columns cannot be
    # aliases for one perfectly nested grade code.
    assert len({tuple(row) for row in marker.tolist()}) >= 12
    assert all(not np.array_equal(marker[:, left], marker[:, right]) for left in range(4) for right in range(left + 1, 4))


@pytest.mark.parametrize("family", ["localized", "border"])
def test_active_and_inactive_codewords_match_area_and_first_order_statistics(
    family: str,
) -> None:
    dataset = OrdinalShortcutDataset(_CleanDataset(), family=family)
    for boundary in range(4):
        present = [False] * 4
        present[boundary] = True
        inactive = dataset.render(11, marker_bits=(0, 0, 0, 0), marker_present=present)
        active_bits = [False] * 4
        active_bits[boundary] = True
        active = dataset.render(11, marker_bits=active_bits, marker_present=present)
        mask = active.metadata.support_masks[boundary]
        inactive_delta = inactive.image - inactive.clean_image
        active_delta = active.image - active.clean_image
        assert int(mask.sum()) > 0
        for channel in range(3):
            assert float(inactive_delta[channel, mask].mean()) == pytest.approx(0.0, abs=1e-6)
            assert float(active_delta[channel, mask].mean()) == pytest.approx(0.0, abs=1e-6)
            torch.testing.assert_close(
                inactive_delta[channel, mask].sort().values,
                active_delta[channel, mask].sort().values,
            )
    masks = dataset.render(11).metadata.support_masks
    if family == "localized":
        assert int((masks.sum(dim=0) > 1).sum()) == 0


def test_unseen_position_and_appearance_domains_are_disjoint_constructions() -> None:
    base = _CleanDataset()
    seen = OrdinalShortcutDataset(
        base,
        family="localized",
        position_domain="seen",
        appearance_domain="seen",
    ).render(9)
    unseen = OrdinalShortcutDataset(
        base,
        family="localized",
        position_domain="unseen",
        appearance_domain="unseen",
    ).render(9)
    assert seen.metadata.centers_yx != unseen.metadata.centers_yx
    assert not torch.equal(seen.metadata.support_masks, unseen.metadata.support_masks)
    assert not torch.equal(seen.image, unseen.image)


def test_factorial_pairs_change_only_the_requested_marker_pixels() -> None:
    base = OrdinalShortcutDataset(_CleanDataset(), family="localized", condition="neutral")
    factorial = FactorialShortcutDataset(base)
    all_inactive = factorial.render(0)
    only_boundary_zero_active = factorial.render(1)
    difference = (only_boundary_zero_active.image - all_inactive.image).abs().sum(dim=0) > 0
    target_support = all_inactive.metadata.support_masks[0]
    assert bool(difference.any())
    assert not bool((difference & ~target_support).any())
    assert torch.equal(all_inactive.clean_image, only_boundary_zero_active.clean_image)
    assert len(factorial) == 16 * len(base)


def test_missing_swapped_conflicting_clean_and_diffuse_metadata() -> None:
    base = _CleanDataset()
    missing = OrdinalShortcutDataset(base, family="localized", condition="missing").render(4)
    assert not all(missing.metadata.marker_present)
    swapped = OrdinalShortcutDataset(
        base, family="localized", condition="boundary_swapped"
    ).render(4)
    assert swapped.metadata.source_boundaries == (1, 0, 3, 2)
    conflict = OrdinalShortcutDataset(
        base, family="localized", condition="conflicting"
    ).render(4)
    assert conflict.metadata.marker_bits in {
        (False, True, False, True),
        (True, False, True, False),
    }
    clean = OrdinalShortcutDataset(base, family="localized", condition="clean").render(4)
    assert torch.equal(clean.image, clean.clean_image)
    assert not bool(clean.metadata.rendered_masks.any())
    diffuse = OrdinalShortcutDataset(base, family="diffuse", condition="aligned").render(4)
    assert not diffuse.metadata.localization_applicable
    assert diffuse.metadata.support_kind == "global_diffuse"
    assert torch.equal(
        diffuse.metadata.support_masks[0], diffuse.metadata.support_masks[3]
    )


def test_cue_only_permutes_images_but_perfectly_encodes_target_label() -> None:
    base = _CleanDataset(count=25)
    cue_only = OrdinalShortcutDataset(base, family="localized", condition="cue_only")
    sample = cue_only.render(8)
    assert sample.metadata.source_index != 8
    assert sample.metadata.marker_bits == tuple(8 % 5 > boundary for boundary in range(4))
    assert int(sample.label) == 8 % 5
    source = base[sample.metadata.source_index][0]
    assert torch.equal(sample.clean_image, source)


def test_loader_preserves_four_field_training_contract() -> None:
    datasets = [
        OrdinalShortcutDataset(_CleanDataset(count=20), family="localized")
        for _ in range(3)
    ]
    loaders = make_shortcut_loaders(
        *datasets, batch_size=4, num_workers=0, pin_memory=False, seed=13
    )
    batch = next(iter(loaders[0]))
    assert len(batch) == 4
    assert batch[0].shape == (4, 3, 48, 48)
    assert batch[1].shape == (4, 1, 48, 48)
    assert batch[2].dtype == torch.long


def test_render_model_factorial_and_intervention_audit_smoke() -> None:
    """Exercise the complete in-process benchmark path on a tiny model."""

    torch.manual_seed(29)
    protocol = OrdinalShortcutProtocol(seed=101, marker_radius_fraction=0.06)
    dataset = OrdinalShortcutDataset(
        _CleanDataset(count=3, size=48),
        family="localized",
        condition="aligned",
        position_domain="unseen",
        appearance_domain="unseen",
        protocol=protocol,
    )
    model = OriginModel(
        encoder=_TinyPyramid(),
        pretrained=False,
        projection_dim=8,
        reference_count=64.0,
        atom_rate_init=1e-4,
        prior_rate_init=1e-4,
    ).eval()
    transform_hash = _transform_parameter_hash(
        protocol=protocol,
        image_size=48,
        family="localized",
        condition="aligned",
        position_domain="unseen",
        appearance_domain="unseen",
    )
    result = _main_audit(
        model,
        dataset,
        [0, 1, 2],
        device=torch.device("cpu"),
        batch_size=2,
        audit_grid=13,
        audit_seed=17,
        permutations=7,
        bootstrap_replicates=7,
        fold=0,
        factorial_transform_sha256=transform_hash,
        main_transform_sha256=transform_hash,
        decision_rule="class_map",
        model_variant="origin_ctmc",
    )
    assert result["factorial_states_per_sample"] == 16
    assert len(result["factorial_prediction_records"]) == 3 * 16
    assert len(result["internal_effect_records"]) == 3 * 4
    assert np.asarray(result["boundary_response_matrix"]).shape == (4, 4)
    assert result["localization_permutation"]["permutations"] == 7


@pytest.mark.parametrize("variant", ["ordinal_additive_mil", "sparse_bagnet"])
def test_matched_local_comparator_runs_same_shortcut_audit(variant: str) -> None:
    dataset = OrdinalShortcutDataset(
        _CleanDataset(count=5, size=32), family="localized", condition="aligned"
    )
    model = FixedCountLocalEvidenceModel(
        variant=variant,
        num_classes=5,
        projection_dim=4,
        pretrained=False,
        encoder=_TinyPyramid(),
    ).eval()
    transform_hash = _transform_parameter_hash(
        protocol=dataset.protocol,
        image_size=32,
        family="localized",
        condition="aligned",
        position_domain="seen",
        appearance_domain="seen",
    )
    result = _main_audit(
        model,
        dataset,
        [0, 1, 2],
        device=torch.device("cpu"),
        batch_size=2,
        audit_grid=13,
        audit_seed=17,
        permutations=3,
        bootstrap_replicates=5,
        fold=0,
        factorial_transform_sha256=transform_hash,
        main_transform_sha256=transform_hash,
        decision_rule="class_map",
        model_variant=variant,
    )
    assert result["local_ledger_applicable"]
    assert len(result["internal_effect_records"]) == 12
    assert result["internal_pixel_effects"]["n_image_clusters"] == 3


def test_pooled_comparator_reports_behavior_without_fabricated_local_ledger() -> None:
    dataset = OrdinalShortcutDataset(
        _CleanDataset(count=5, size=32), family="localized", condition="aligned"
    )
    model = MultiScaleMaskedPoolingModel(
        variant="pooled_conditional",
        num_classes=5,
        projection_dim=4,
        pretrained=False,
        encoder=_TinyPyramid(),
    ).eval()
    transform_hash = _transform_parameter_hash(
        protocol=dataset.protocol,
        image_size=32,
        family="localized",
        condition="aligned",
        position_domain="seen",
        appearance_domain="seen",
    )
    result = _main_audit(
        model,
        dataset,
        [0, 1, 2],
        device=torch.device("cpu"),
        batch_size=2,
        audit_grid=13,
        audit_seed=17,
        permutations=3,
        bootstrap_replicates=5,
        fold=0,
        factorial_transform_sha256=transform_hash,
        main_transform_sha256=transform_hash,
        decision_rule="class_map",
        model_variant="pooled_conditional",
    )
    assert not result["local_ledger_applicable"]
    assert not result["localization"]["applicable"]
    assert not result["internal_pixel_effects"]["applicable"]
    assert len(result["internal_effect_records"]) == 0
    assert np.asarray(result["boundary_response_matrix"]).shape == (4, 4)


def _perfect_localization_fixture(samples: int = 12):
    masks = np.zeros((samples, 4, 12, 12), dtype=bool)
    for sample in range(samples):
        for boundary in range(4):
            y = 1 + ((sample + boundary * 2) % 9)
            x = 1 + ((sample * 2 + boundary * 3) % 9)
            masks[sample, boundary, y : y + 2, x : x + 2] = True
    evidence = masks.astype(np.float64) * 10.0 + 0.01
    valid = np.ones((samples, 1, 12, 12), dtype=bool)
    return evidence, masks, valid


def test_localization_response_and_internal_pixel_metrics() -> None:
    evidence, masks, valid = _perfect_localization_fixture()
    localization = localization_metrics(evidence, masks, valid)
    np.testing.assert_allclose(localization["boundary_auprc"], np.ones(4))
    np.testing.assert_allclose(localization["boundary_pointing_accuracy"], np.ones(4))
    np.testing.assert_allclose(
        localization["boundary_identification_accuracy"], np.ones(4)
    )
    permutation = familywise_localization_permutation_test(
        evidence, masks, valid, permutations=199, seed=17
    )
    assert np.all(np.asarray(permutation["fwer_p"]) <= 0.05)

    paired = np.zeros((20, 4, 2, 4), dtype=np.float64)
    for cue in range(4):
        paired[:, cue, 1, cue] = 0.8
        paired[:, cue, 1, :] += 0.05
    response = boundary_response_matrix(paired)
    selectivity = boundary_response_selectivity(response)
    assert selectivity["dominant_diagonal_fraction"] == 1.0
    assert selectivity["diagonal_to_off_diagonal_ratio"] > 10.0

    internal = np.linspace(0.1, 1.0, 30)
    random_effect = internal * 0.2
    pixel = internal**2
    audit = internal_pixel_effect_audit(
        internal, random_effect, pixel, bootstrap_replicates=100, seed=3
    )
    assert audit["median_internal_to_random_ratio"] == pytest.approx(5.0)
    assert audit["internal_pixel_spearman"] == pytest.approx(1.0)
    assert audit["n_image_clusters"] == 30
    assert audit["observations_per_image"] == 1
    assert audit["bootstrap_unit"] == (
        "image_cluster_all_boundary_observations_retained"
    )
    assert spearman_correlation(internal, pixel) == pytest.approx(1.0)


def test_spearman_bootstrap_resamples_images_and_keeps_boundary_clusters() -> None:
    internal = np.asarray(
        [[0.1, 0.2, 0.4, 0.8], [0.9, 0.3, 0.2, 0.1], [0.3, 0.7, 0.5, 0.2],
         [0.8, 0.9, 0.1, 0.4], [0.6, 0.2, 0.9, 0.7]],
        dtype=np.float64,
    )
    pixel = internal * np.asarray([1.0, -0.4, 0.7, 0.2])[None]
    observed = bootstrap_spearman_image_cluster_interval(
        internal, pixel, replicates=37, seed=118
    )

    # Independent reference implementation: one integer draw per image row,
    # then flatten all four retained boundary observations.
    generator = np.random.default_rng(118)
    estimates = []
    for _ in range(37):
        rows = generator.integers(0, len(internal), size=len(internal))
        estimates.append(
            spearman_correlation(internal[rows].reshape(-1), pixel[rows].reshape(-1))
        )
    expected = tuple(np.quantile(estimates, [0.025, 0.975]))
    assert observed == pytest.approx(expected)
    assert observed == bootstrap_spearman_image_cluster_interval(
        internal, pixel, replicates=37, seed=118
    )


def test_rasterization_preserves_ledger_mass_for_divisible_lattices() -> None:
    maps = {
        "s4": torch.arange(2 * 4 * 2 * 2, dtype=torch.float32).reshape(2, 4, 2, 2),
        "s8": torch.ones(2, 4, 1, 1),
    }
    raster = rasterize_native_ledger(maps, output_size=(8, 8))
    torch.testing.assert_close(raster.sum(dim=(-2, -1)), sum(value.sum(dim=(-2, -1)) for value in maps.values()))


@pytest.mark.parametrize("source_size", [20, 40, 80, 160])
def test_rasterization_cannot_drop_a_native_unit_witness(source_size: int) -> None:
    maps = torch.zeros(1, 4, source_size, source_size, dtype=torch.float64)
    # Deliberately choose a coordinate that nearest-neighbour downsampling from
    # 160 to 64 used to skip completely.
    maps[0, 0, 1, 1] = 1.0
    maps[0, 1, source_size // 2, source_size - 2] = 2.0
    raster = rasterize_native_ledger({"scale": maps}, output_size=(64, 64))
    torch.testing.assert_close(
        raster.sum(dim=(-2, -1)),
        maps.sum(dim=(-2, -1)),
        atol=1e-12,
        rtol=1e-12,
    )
    assert float(raster[0, 0].sum()) == pytest.approx(1.0, abs=1e-12)


@pytest.mark.parametrize(
    ("source_shape", "target_shape"),
    [((3, 5), (7, 11)), ((7, 11), (3, 5)), ((13, 17), (19, 23))],
)
def test_rasterization_conserves_random_odd_lattices(
    source_shape: tuple[int, int], target_shape: tuple[int, int]
) -> None:
    generator = torch.Generator().manual_seed(713)
    values = torch.rand(
        2, 4, *source_shape, generator=generator, dtype=torch.float64
    )
    raster = rasterize_native_ledger({"odd": values}, output_size=target_shape)
    assert raster.shape == (2, 4, *target_shape)
    torch.testing.assert_close(
        raster.sum(dim=(-2, -1)),
        values.sum(dim=(-2, -1)),
        atol=2e-12,
        rtol=2e-12,
    )


def _passing_seed_report(clean: bool = False):
    return {
        "condition_performance": {
            "aligned_minus_neutral_accuracy": 0.20,
            "aligned_minus_inverted_accuracy": 0.30,
        },
        "localization_permutation": {"fwer_p": [0.01, 0.02, 0.03, 0.20]},
        "localization": {
            "macro_auprc_lift": 0.0 if clean else 0.8,
            "macro_effective_support_fraction": 0.20,
        },
        "boundary_response_selectivity": {
            "dominant_diagonal_fraction": 1.0,
            "diagonal_to_off_diagonal_ratio": 4.0,
            "diagonal_minus_off_diagonal": 0.3,
        },
        "internal_pixel_effects": {
            "median_internal_minus_random": 0.30,
            "median_internal_to_random_ratio": 3.0,
            "internal_pixel_spearman": 0.70,
            "internal_pixel_spearman_ci95": [0.40, 0.80],
        },
    }


def _passing_cue_only_report(accuracy: float = 0.95):
    return {"condition_performance": {"cue_only": {"accuracy": accuracy}}}


def test_gate_b_requires_multiseed_localization_and_clean_negative_control() -> None:
    shortcut = [_passing_seed_report(), _passing_seed_report(), _passing_seed_report()]
    clean = [_passing_seed_report(clean=True) for _ in range(3)]
    cue_only = [_passing_cue_only_report() for _ in range(3)]
    result = assess_gate_b(
        shortcut, clean, cue_only, thresholds=GateBThresholds()
    )
    assert result["passed"]
    assert result["checks"]["cue_only_positive_control"]
    assert not result["positive_control_is_sufficient_for_promotion"]
    shortcut[0]["localization_permutation"]["fwer_p"] = [0.5] * 4
    shortcut[1]["localization_permutation"]["fwer_p"] = [0.5] * 4
    assert not assess_gate_b(shortcut, clean, cue_only)["passed"]


def test_gate_b_rejects_failed_positive_control_even_when_other_checks_pass() -> None:
    shortcut = [_passing_seed_report(), _passing_seed_report(), _passing_seed_report()]
    clean = [_passing_seed_report(clean=True) for _ in range(3)]
    cue_only = [_passing_cue_only_report(0.79) for _ in range(3)]
    result = assess_gate_b(shortcut, clean, cue_only)
    assert not result["checks"]["cue_only_positive_control"]
    assert not result["passed"]


def test_gate_b_rejects_nonselective_boundary_response() -> None:
    shortcut = [_passing_seed_report(), _passing_seed_report(), _passing_seed_report()]
    clean = [_passing_seed_report(clean=True) for _ in range(3)]
    cue_only = [_passing_cue_only_report() for _ in range(3)]
    for report in shortcut[:2]:
        report["boundary_response_selectivity"] = {
            "dominant_diagonal_fraction": 0.25,
            "diagonal_to_off_diagonal_ratio": 1.0,
            "diagonal_minus_off_diagonal": 0.0,
        }
    result = assess_gate_b(shortcut, clean, cue_only)
    assert not result["checks"]["boundary_response_selectivity"]
    assert not result["passed"]


def test_diffuse_nonfocal_gate_requires_broad_effective_support() -> None:
    reports = [_passing_seed_report() for _ in range(3)]
    reports[0]["localization"]["macro_effective_support_fraction"] = 0.14
    reports[1]["localization"]["macro_effective_support_fraction"] = 0.11
    reports[2]["localization"]["macro_effective_support_fraction"] = 0.03
    cue_only = [_passing_cue_only_report() for _ in range(3)]
    passing = assess_diffuse_nonfocal_gate(reports, cue_only)
    assert passing["passed"]
    assert passing["must_not_be_reported_as_focal_lesion"]

    reports[1]["localization"]["macro_effective_support_fraction"] = 0.09
    failing = assess_diffuse_nonfocal_gate(reports, cue_only)
    assert not failing["passed"]


def test_diffuse_nonfocal_gate_fails_closed_on_missing_support() -> None:
    reports = [_passing_seed_report() for _ in range(3)]
    for report in reports:
        report["localization"].pop("macro_effective_support_fraction")
    result = assess_diffuse_nonfocal_gate(
        reports, [_passing_cue_only_report() for _ in range(3)]
    )
    assert result["broad_support_values"] == [None, None, None]
    assert not result["passed"]


def test_diffuse_nonfocal_gate_still_requires_positive_control() -> None:
    reports = [_passing_seed_report() for _ in range(3)]
    cue_only = [_passing_cue_only_report(0.79) for _ in range(3)]
    result = assess_diffuse_nonfocal_gate(reports, cue_only)
    assert not result["checks"]["cue_only_positive_control"]
    assert not result["passed"]
    assert not result["positive_control_is_sufficient_for_promotion"]


def test_shortcut_comparator_v2_exactly_enumerates_84_additional_workers() -> None:
    tasks = all_tasks()
    assert TASK_COUNT == 84
    assert len(tasks) == len({task.key for task in tasks}) == 84
    assert {task.model_variant for task in tasks} == set(COMPARATOR_VARIANTS)
    for variant in COMPARATOR_VARIANTS:
        variant_tasks = [task for task in tasks if task.model_variant == variant]
        assert len(variant_tasks) == 21
        assert sum(task.arm == "shortcut" for task in variant_tasks) == 9
        assert sum(task.arm == "cue_only" for task in variant_tasks) == 9
        assert sum(task.arm == "clean" for task in variant_tasks) == 3
    assert task_at(0).model_variant == "ledger_sequential_hazard"
    assert task_at(83).model_variant == "sparse_bagnet"
    with pytest.raises(IndexError):
        task_at(84)
    assert canonical_sha256(protocol_core()) == PROTOCOL_CORE_SHA256
    assert protocol_core()["implementation_scope"]["official_author_implementations"] is False
    assert protocol_core()["execution_topology"] == {
        "cpu_preflight_jobs": 1,
        "gpu_training_and_blind_audit_workers": 84,
        "cpu_expanded_aggregate_jobs": 1,
        "sealed_origin_v1_workers_reused": 21,
    }
    frozen_thresholds = protocol_core()["gate_b_thresholds"]
    assert frozen_thresholds["boundary_dominant_diagonal_fraction_min"] == 0.75
    assert frozen_thresholds["boundary_diagonal_to_off_ratio_min"] == 2.0
    assert frozen_thresholds["diffuse_effective_support_fraction_min"] == 0.10


def test_binary_average_precision_is_tie_order_independent() -> None:
    scores = [1.0, 1.0, 0.0, 0.0]
    assert binary_average_precision(scores, [1, 0, 1, 0]) == pytest.approx(0.5)
    assert binary_average_precision(scores, [0, 1, 0, 1]) == pytest.approx(0.5)
