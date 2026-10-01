"""Contracts for the isolated acceptance-revision matched baselines."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from configs.origin_acceptance_baseline_config import (
    OriginAcceptanceBaselineConfig,
)
from losses.origin_acceptance_baselines import OriginAcceptanceBaselineLoss
from models.origin_acceptance_baselines import (
    FixedCountLocalEvidenceModel,
    build_origin_acceptance_baseline,
)
from models.origin_encoder import (
    OriginEncoderOutput,
    OriginEncoderScale,
    OriginScaleMetadata,
)
from scripts.origin_acceptance_baseline_common import (
    BASELINE_ORDER,
    PROTOCOL_SHA256,
    canary_tasks,
    full_tasks,
    task_at,
)
from training.origin_acceptance_baseline_trainer import (
    OriginAcceptanceBaselineTrainer,
)


class _TinyPyramid(nn.Module):
    STAGE_CHANNELS = {"s4": 4, "s8": 5, "s16": 6, "s32": 7}

    def __init__(self) -> None:
        super().__init__()
        self.projections = nn.ModuleDict(
            {
                name: nn.Conv2d(3, channels, 1)
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
        result = {}
        for index, name in enumerate(("s4", "s8", "s16", "s32")):
            size = sizes[name]
            features = self.projections[name](F.adaptive_avg_pool2d(images, size))
            mask = F.adaptive_avg_pool2d(
                pixel_valid_mask[:, None].float(), size
            )[:, 0] >= 0.5
            metadata = OriginScaleMetadata(
                name=name,
                feature_index=index,
                channels=self.STAGE_CHANNELS[name],
                output_stride=4 * 2**index,
                receptive_field=9 + 8 * index,
                center_offset=float(2 * 2**index),
                input_size=tuple(images.shape[-2:]),
                lattice_size=size,
            )
            result[name] = OriginEncoderScale(
                features.masked_fill(~mask[:, None], 0.0), mask, metadata
            )
        return OriginEncoderOutput(result)


def _model(variant: str) -> FixedCountLocalEvidenceModel:
    return FixedCountLocalEvidenceModel(
        variant=variant,
        num_classes=5,
        projection_dim=4,
        pretrained=False,
        encoder=_TinyPyramid(),
        reference_count=32.0,
        atom_rate_init=1e-4,
        prior_rate_init=2e-4,
    )


def _assert_posterior(output, batch: int = 2) -> None:
    assert output.class_probs.shape == (batch, 5)
    assert output.cumulative_probs.shape == (batch, 4)
    assert output.total_rates.shape == (batch, 4)
    for field in ("class_probs", "log_class_probs", "cumulative_probs", "expected_grade"):
        value = getattr(output, field)
        assert value.dtype == torch.float64
        assert torch.isfinite(value).all()
    torch.testing.assert_close(
        output.class_probs.sum(-1),
        torch.ones(batch, dtype=torch.float64),
        atol=1e-12,
        rtol=1e-12,
    )
    assert torch.all(output.cumulative_probs[:, 1:] <= output.cumulative_probs[:, :-1])


@pytest.mark.parametrize("variant", ["ordinal_additive_mil", "sparse_bagnet"])
def test_local_heads_match_nominal_floor_and_have_no_bypass(variant: str) -> None:
    model = _model(variant).eval()
    output = model(torch.randn(2, 3, 32, 32))
    _assert_posterior(output)
    torch.testing.assert_close(
        output.class_probs,
        model.nominal_origin_class_probs.expand(2, -1),
        atol=2e-8,
        rtol=2e-6,
    )
    reconstructed = model.intercept.double()[None].expand_as(output.aggregated_logits)
    for local_map in output.effective_local_maps.values():
        reconstructed = reconstructed + local_map.double().sum((1, 2))
    torch.testing.assert_close(reconstructed, output.aggregated_logits)
    metadata = model.architecture_metadata()
    assert metadata["no_classifier_bypass"] is True
    assert metadata["deletion_denominator"] == "frozen_original_valid_count"
    assert metadata["rate_telemetry_applicable"] is False


def test_additive_mil_signed_fixed_mean_and_exact_replay() -> None:
    model = _model("ordinal_additive_mil").eval()
    with torch.no_grad():
        for index, head in enumerate(model.heads.values()):
            head.local_logits.weight.zero_()
            head.local_logits.bias.copy_(
                torch.tensor([0.4, -0.3, 0.2, -0.1]) * (index + 1)
            )
    images = torch.randn(2, 3, 32, 32)
    mask = torch.ones(2, 32, 32, dtype=torch.bool)
    mask[:, :8, :8] = False
    output = model(images, mask)
    _assert_posterior(output)
    assert any(bool((value < 0).any()) for value in output.local_contribution_maps.values())
    for name, local_map in output.effective_local_maps.items():
        assert torch.count_nonzero(local_map[~output.valid_masks[name]]) == 0

    removals = {name: torch.zeros_like(valid) for name, valid in output.valid_masks.items()}
    location = output.valid_masks["s4"][0].nonzero()[0]
    y, x = (int(location[0]), int(location[1]))
    removals["s4"][0, y, x] = True
    replay = model.replay_without(output, removals)
    expected_removed = output.effective_local_maps["s4"][0, y, x].double()
    torch.testing.assert_close(replay.removed_contributions[0], expected_removed)
    torch.testing.assert_close(
        replay.output.aggregated_logits,
        output.aggregated_logits - replay.removed_contributions,
        atol=2e-7,
        rtol=0,
    )
    # All untouched cells are bitwise identical: there is no survivor
    # renormalization and no re-encoding.
    survivor = ~removals["s4"].unsqueeze(-1).expand_as(output.effective_local_maps["s4"])
    assert torch.equal(
        replay.output.effective_local_maps["s4"][survivor],
        output.effective_local_maps["s4"][survivor],
    )
    partition = replay.output.effective_local_maps["s4"].clone()
    partition[0, y, x] += output.effective_local_maps["s4"][0, y, x]
    assert torch.equal(partition, output.effective_local_maps["s4"])


def test_sparse_bagnet_masked_mean_l1_and_exact_replay() -> None:
    model = _model("sparse_bagnet").eval()
    with torch.no_grad():
        for index, head in enumerate(model.heads.values()):
            head.local_logits.weight.zero_()
            head.local_logits.bias.fill_(-2.0 + 0.2 * index)
    mask = torch.ones(2, 32, 32, dtype=torch.bool)
    mask[:, :8, :] = False
    output = model(torch.randn(2, 3, 32, 32), mask)
    _assert_posterior(output)
    assert all(bool((value >= 0).all()) for value in output.local_evidence_maps.values())
    manual_terms = []
    for name, raw in output.raw_local_maps.items():
        valid = output.valid_masks[name]
        per_sample = (raw * valid.unsqueeze(-1)).sum((1, 2, 3)) / (
            valid.sum((1, 2)) * model.num_classes
        )
        manual_terms.append(per_sample.mean())
        assert torch.count_nonzero(output.effective_local_maps[name][~valid]) == 0
    torch.testing.assert_close(output.sparse_activation_l1, torch.stack(manual_terms).mean())

    removals = {name: torch.zeros_like(valid) for name, valid in output.valid_masks.items()}
    removals["s8"][:, 1, 1] = output.valid_masks["s8"][:, 1, 1]
    replay = model.replay_without(output, removals)
    torch.testing.assert_close(
        replay.output.aggregated_logits,
        output.aggregated_logits - replay.removed_contributions,
        atol=2e-7,
        rtol=0,
    )


def test_sparse_loss_uses_unpooled_activation_penalty_and_backpropagates() -> None:
    model = _model("sparse_bagnet")
    output = model(torch.randn(2, 3, 32, 32))
    labels = torch.tensor([0, 4])
    criterion = OriginAcceptanceBaselineLoss(
        5,
        baseline_variant="sparse_bagnet",
        sparse_l1_weight=0.3,
        sparse_l1_delay_epochs=1,
    )
    warm_loss, warm = criterion(output, labels, epoch=1)
    active_loss, active = criterion(output, labels, epoch=2)
    assert warm["budget_active"] is False
    assert active["budget_active"] is True
    torch.testing.assert_close(
        active_loss - warm_loss, 0.3 * output.sparse_activation_l1.double()
    )
    active_loss.backward()
    assert model.heads["s4"].local_logits.bias.grad is not None
    with pytest.raises(ValueError, match="only sparse_bagnet"):
        OriginAcceptanceBaselineLoss(
            5, baseline_variant="ordinal_additive_mil", sparse_l1_weight=0.1
        )


def test_builder_delegates_the_two_existing_controls(monkeypatch) -> None:
    sentinel = nn.Identity()
    captured = []

    def fake_builder(cfg, *, pretrained=None):
        captured.append((cfg.ablation_variant, pretrained))
        return sentinel

    monkeypatch.setattr(
        "models.origin_acceptance_baselines.build_origin_ablation_model",
        fake_builder,
    )
    for variant in ("ledger_sequential_hazard", "pooled_conditional"):
        cfg = OriginAcceptanceBaselineConfig(
            baseline_variant=variant,
            ablation_variant=variant,
            pretrained=False,
            sparse_l1_weight=0.0,
        )
        assert build_origin_acceptance_baseline(cfg, pretrained=False) is sentinel
    assert captured == [
        ("ledger_sequential_hazard", False),
        ("pooled_conditional", False),
    ]


def test_config_rejects_unmatched_or_unfair_controls() -> None:
    with pytest.raises(ValueError, match="must match"):
        OriginAcceptanceBaselineConfig(
            baseline_variant="pooled_conditional",
            ablation_variant="ledger_sequential_hazard",
        )
    with pytest.raises(ValueError, match="all native encoder scales"):
        OriginAcceptanceBaselineConfig(
            baseline_variant="ordinal_additive_mil",
            ablation_variant="ordinal_additive_mil",
            evidence_scales=("s32",),
        )
    with pytest.raises(ValueError, match="zero outside"):
        OriginAcceptanceBaselineConfig(
            baseline_variant="ordinal_additive_mil",
            ablation_variant="ordinal_additive_mil",
            sparse_l1_weight=1e-4,
        )
    sparse = OriginAcceptanceBaselineConfig(
        baseline_variant="sparse_bagnet",
        ablation_variant="sparse_bagnet",
        sparse_l1_weight=1e-4,
    )
    assert sparse.seed == 42 and sparse.training_seed == 42


def test_protocol_task_map_is_stable_and_canary_first() -> None:
    assert tuple(task.baseline_variant for task in canary_tasks()) == BASELINE_ORDER
    assert all(task.dataset == "aptos" and task.fold == 0 for task in canary_tasks())
    assert len(full_tasks()) == 180
    assert task_at("full", 0).key == "aptos__fold0__ledger_sequential_hazard__seed42"
    assert task_at("full", 179).key == "dr__fold9__sparse_bagnet__seed27182"
    assert len(PROTOCOL_SHA256) == 64


class _ToyDataset(Dataset):
    def __init__(self) -> None:
        self.images = torch.randn(4, 3, 32, 32)
        self.labels = [0, 1, 3, 4]

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int):
        return (
            self.images[index],
            torch.ones(1, 32, 32, dtype=torch.bool),
            torch.tensor(self.labels[index]),
            index,
        )


def test_trainer_keeps_warm_floor_out_of_learned_selection(tmp_path: Path) -> None:
    cfg = OriginAcceptanceBaselineConfig(
        dataset="aptos",
        n_classes=5,
        n_folds=2,
        img_size=32,
        pretrained=False,
        projection_dim=4,
        reference_count=32.0,
        atom_rate_init=1e-4,
        prior_rate_init=2e-4,
        baseline_variant="ordinal_additive_mil",
        ablation_variant="ordinal_additive_mil",
        sparse_l1_weight=0.0,
        epochs=1,
        batch_size=2,
        num_workers=0,
        encoder_freeze_epochs=0,
        amp=False,
        certificate_samples=1,
    )
    loader = DataLoader(_ToyDataset(), batch_size=2, shuffle=False)
    model = FixedCountLocalEvidenceModel(
        variant="ordinal_additive_mil",
        num_classes=5,
        projection_dim=4,
        pretrained=False,
        encoder=_TinyPyramid(),
        reference_count=32.0,
        atom_rate_init=1e-4,
        prior_rate_init=2e-4,
    )
    trainer = OriginAcceptanceBaselineTrainer(
        model,
        loader,
        loader,
        None,
        cfg,
        tmp_path,
        fold=0,
        split_signature="toy-split",
        device="cpu",
    )
    result = trainer.fit(evaluate_test=False)
    assert result["best_epoch"] == 1
    assert result["test_evaluated"] is False
    assert result["warm_start_eligible_for_selection"] is False
    assert (tmp_path / "warm_start_floor.json").is_file()
    assert (tmp_path / "best_learned.pth").is_file()
    assert not (tmp_path / "best.pth").exists()
    with (tmp_path / "warm_start_floor.json").open() as stream:
        warm = json.load(stream)
    assert warm["epoch"] == 0 and warm["eligible_for_model_selection"] is False
    checkpoint = torch.load(
        tmp_path / "best_learned.pth", map_location="cpu", weights_only=False
    )
    assert checkpoint["epoch"] == 1
    assert checkpoint["checkpoint_role"] == "learned_epoch_only"


def test_slurm_pipeline_is_fail_closed_and_test_locked() -> None:
    root = Path(__file__).resolve().parents[1]
    train_text = (root / "scripts/submit_origin_acceptance_full_array.sh").read_text()
    release_text = (root / "scripts/submit_origin_acceptance_outer_release.sh").read_text()
    launch_text = (root / "scripts/launch_origin_acceptance_baselines.sh").read_text()
    assert "CANARY_PASSED.json" in train_text
    assert "--skip_test" in train_text
    assert "--include_test" not in train_text
    assert "TRAINING_FROZEN" in (
        root / "release_origin_acceptance_baselines.py"
    ).read_text()
    assert "--array=0-179" in release_text
    assert "afterok:${CANARY_AUDIT_JOB}" in launch_text
    assert "afterok:${FULL_AUDIT_JOB}" in launch_text
