from __future__ import annotations

import hashlib
import json
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, TensorDataset

from training.origin_trainer import (
    OriginTrainer,
    build_class_weights,
    evaluate_origin_predictions,
    validation_selection_key,
)


class _TinyOrigin(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = nn.Sequential(nn.Flatten(), nn.Linear(4, 6), nn.Tanh())
        self.rate_head = nn.Linear(6, 2)

    @staticmethod
    def _decode(total_rates: torch.Tensor, local_rates: torch.Tensor):
        total_rates = total_rates.float()
        # The trainer test needs a differentiable, normalized, ordinal
        # posterior; mathematical generator tests live in the model suite.
        scores = torch.stack(
            (
                -total_rates[:, 0],
                total_rates[:, 0] - total_rates[:, 1],
                total_rates[:, 1],
            ),
            dim=1,
        )
        probs = scores.softmax(dim=1)
        cumulative = torch.stack((probs[:, 1:].sum(1), probs[:, 2]), dim=1)
        expected = (probs * torch.arange(3, device=probs.device)).sum(1)
        median = (probs.cumsum(1) < 0.5).sum(1)
        return SimpleNamespace(
            class_probs=probs,
            log_class_probs=probs.log(),
            cumulative_probs=cumulative,
            expected_grade=expected,
            posterior_median=median,
            class_map=probs.argmax(1),
            total_rates=total_rates,
            local_rate_maps={"s8": local_rates.float()},
            valid_masks={
                "s8": torch.ones(
                    local_rates.shape[:-1],
                    dtype=torch.bool,
                    device=local_rates.device,
                )
            },
        )

    def forward(
        self,
        images: torch.Tensor,
        *,
        pixel_valid_mask: torch.Tensor | None = None,
        force_decoder_fp32: bool = True,
    ):
        del pixel_valid_mask, force_decoder_fp32
        features = self.encoder(images)
        rates = F.softplus(self.rate_head(features)).float()
        local = rates[:, None, None, :]
        return self._decode(rates, local)

    def replay_without(
        self,
        baseline,
        removal_masks,
        *,
        force_decoder_fp32: bool = True,
    ):
        del force_decoder_fp32
        mask = removal_masks["s8"].unsqueeze(-1)
        removed = (baseline.local_rate_maps["s8"] * mask).sum(dim=(1, 2))
        surviving = baseline.local_rate_maps["s8"].masked_fill(mask, 0.0)
        output = self._decode(baseline.total_rates - removed, surviving)
        return SimpleNamespace(
            output=output,
            baseline=baseline,
            removed_rates=removed,
            removal_masks=removal_masks,
        )

    def architecture_metadata(self):
        return {"num_classes": 3, "test_model": True}


class _ExplodingDataset(Dataset):
    def __len__(self) -> int:
        return 2

    def __getitem__(self, index: int):
        raise AssertionError(f"outer test dataset was accessed at {index}")


def _dataset() -> TensorDataset:
    images = torch.tensor(
        [
            [[[0.0, 0.0], [0.0, 0.0]]],
            [[[1.0, 0.0], [0.0, 0.0]]],
            [[[0.0, 1.0], [1.0, 0.0]]],
            [[[1.0, 1.0], [1.0, 1.0]]],
            [[[0.2, 0.2], [0.2, 0.2]]],
            [[[0.8, 0.8], [0.8, 0.8]]],
        ],
        dtype=torch.float32,
    )
    masks = torch.ones(len(images), 1, 2, 2, dtype=torch.bool)
    labels = torch.tensor([0, 1, 1, 2, 0, 2])
    indices = torch.arange(len(images))
    return TensorDataset(images, masks, labels, indices)


def _config(**changes):
    values = {
        "n_classes": 3,
        "epochs": 2,
        "lr": 1e-3,
        "head_lr": 2e-3,
        "weight_decay": 0.0,
        "lr_factor": 0.5,
        "lr_patience": 2,
        "lr_min": 1e-7,
        "grad_clip_norm": 5.0,
        "amp": False,
        "early_stopping_patience": 5,
        "rps_weight": 0.2,
        "evidence_budget_weight": 0.0,
        "evidence_budget_delay_epochs": 1,
        "checkpoint_selection": "acc_then_qwk",
        "selection_qwk_weight": 0.1,
        "decision_rule": "posterior_median",
        "resume": False,
        "class_weighting": "none",
        "effective_num_beta": 0.999,
        "class_weight_cap": 10.0,
        "allow_weighted_likelihood": False,
        "encoder_freeze_epochs": 0,
        "certificate_samples": 2,
        "certificate_replay_tolerance": 1e-6,
    }
    values.update(changes)
    return SimpleNamespace(**values)


def test_selection_is_prospectively_accuracy_first_then_qwk() -> None:
    higher_accuracy = {"acc": 84.0, "qwk": 0.70, "loss": 0.5}
    lower_accuracy = {"acc": 83.9, "qwk": 0.99, "loss": 0.1}
    assert validation_selection_key(higher_accuracy) > validation_selection_key(lower_accuracy)
    tied_accuracy_better_qwk = {"acc": 84.0, "qwk": 0.80, "loss": 0.8}
    assert validation_selection_key(tied_accuracy_better_qwk) > validation_selection_key(
        higher_accuracy
    )


def test_metrics_include_imbalance_and_calibration_diagnostics() -> None:
    probabilities = torch.tensor(
        [[0.8, 0.1, 0.1], [0.1, 0.7, 0.2], [0.1, 0.6, 0.3]]
    )
    labels = torch.tensor([0, 1, 2])
    predicted = torch.tensor([0, 1, 1])
    metrics = evaluate_origin_predictions(probabilities, predicted, labels)
    assert metrics["acc"] == pytest.approx(200.0 / 3.0)
    assert metrics["balanced_acc"] == pytest.approx(200.0 / 3.0)
    assert len(metrics["confusion"]) == 3
    assert len(metrics["per_grade_recall"]) == 3
    assert 0.0 <= metrics["ece"] <= 1.0


def test_class_weights_require_explicit_non_population_interpretation() -> None:
    weights = build_class_weights(
        [0, 0, 0, 1, 2],
        3,
        method="effective_num",
        beta=0.9,
        cap=5.0,
    )
    assert weights is not None
    assert weights.shape == (3,)
    assert weights[2] > weights[0]


def test_stratified_sampling_does_not_claim_population_proper_objective(tmp_path) -> None:
    loader = DataLoader(_dataset(), batch_size=3, shuffle=False)
    trainer = OriginTrainer(
        _TinyOrigin(),
        loader,
        loader,
        None,
        _config(stratified_batches=True),
        tmp_path,
        device="cpu",
    )
    assert trainer.criterion.likelihood_component_unweighted
    assert not trainer.population_objective_proper


def test_fit_defaults_to_validation_only_and_writes_exact_certificates(tmp_path) -> None:
    torch.manual_seed(7)
    train = DataLoader(_dataset(), batch_size=3, shuffle=False)
    validation = DataLoader(_dataset(), batch_size=3, shuffle=False)
    outer_test = DataLoader(_ExplodingDataset(), batch_size=2)
    trainer = OriginTrainer(
        _TinyOrigin(),
        train,
        validation,
        outer_test,
        _config(),
        tmp_path,
        fold=0,
        split_signature="synthetic-split",
        device="cpu",
    )
    result = trainer.fit()
    assert result["test_evaluated"] is False
    assert (tmp_path / "best.pth").is_file()
    assert (tmp_path / "last.pth").is_file()
    assert (tmp_path / "history.csv").read_text().count("\n") == 3

    certificate_path = tmp_path / "validation_certificates.json"
    payload = json.loads(certificate_path.read_text())
    checksum = payload.pop("content_checksum_sha256")
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    assert checksum == hashlib.sha256(encoded).hexdigest()
    assert payload["scope"] == "inner_validation_only"
    assert len(payload["certificates"]) == 2
    assert payload["exact_replay_max_abs_total_rate_error"] <= 1e-6


def test_resume_guard_rejects_numerically_changed_configuration(tmp_path) -> None:
    loader = DataLoader(_dataset(), batch_size=3)
    original = OriginTrainer(
        _TinyOrigin(),
        loader,
        loader,
        None,
        _config(),
        tmp_path / "original",
        split_signature="same-split",
        device="cpu",
    )
    state = original._checkpoint_payload(
        epoch=1,
        metrics={"acc": 1.0, "qwk": 0.0, "loss": 1.0},
        best_key=(1.0, 0.0, -1.0),
        best_epoch=1,
        bad_epochs=0,
    )
    changed = OriginTrainer(
        _TinyOrigin(),
        loader,
        loader,
        None,
        _config(rps_weight=0.9),
        tmp_path / "changed",
        split_signature="same-split",
        device="cpu",
    )
    with pytest.raises(ValueError, match="config signature mismatch"):
        changed._validate_resume(state)


def test_extending_only_epoch_count_is_resume_compatible(tmp_path) -> None:
    loader = DataLoader(_dataset(), batch_size=3)
    short = OriginTrainer(
        _TinyOrigin(), loader, loader, None, _config(epochs=2), tmp_path / "short", device="cpu"
    )
    state = short._checkpoint_payload(
        epoch=1,
        metrics={"acc": 1.0, "qwk": 0.0, "loss": 1.0},
        best_key=(1.0, 0.0, -1.0),
        best_epoch=1,
        bad_epochs=0,
    )
    extended = OriginTrainer(
        _TinyOrigin(), loader, loader, None, _config(epochs=5), tmp_path / "extended", device="cpu"
    )
    extended._validate_resume(state)
