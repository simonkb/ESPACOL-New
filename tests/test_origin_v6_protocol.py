"""Protocol guarantees for the prospective ORIGIN-v6 relation pilot.

These tests are intentionally stricter than ordinary model-shape tests.  The
first experiment is meaningful only if its v3 source has verified content
identity, the
new relation field is the sole trainable component, epoch zero is an eligible
no-regression checkpoint, and the reported pair explanation is replayed by
the same ordinal circuit.
"""

from __future__ import annotations

import copy
import csv
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from configs.origin_config import OriginConfig
from models.origin import OriginModel
from models.origin_encoder import (
    OriginEncoderOutput,
    OriginEncoderScale,
    OriginScaleMetadata,
)
import training.origin_trainer as trainer_module
from training.origin_trainer import OriginTrainer, validation_selection_key


class _TinyPyramidEncoder(nn.Module):
    """Cheap four-scale encoder with the same state interface as v3."""

    STAGE_CHANNELS = {"s4": 4, "s8": 4, "s16": 4, "s32": 4}

    def __init__(self) -> None:
        super().__init__()
        self.projection = nn.Conv2d(3, 4, kernel_size=1)
        self.seen_training_modes: list[bool] = []

    def forward(
        self,
        images: torch.Tensor,
        pixel_valid_mask: torch.Tensor | None = None,
    ) -> OriginEncoderOutput:
        del pixel_valid_mask
        self.seen_training_modes.append(self.training)
        base = self.projection(images)
        sizes = {"s4": 8, "s8": 4, "s16": 2, "s32": 1}
        strides = {"s4": 4, "s8": 8, "s16": 16, "s32": 32}
        receptive_fields = {"s4": 7, "s8": 15, "s16": 31, "s32": 63}
        scales: dict[str, OriginEncoderScale] = {}
        for feature_index, name in enumerate(self.STAGE_CHANNELS):
            size = sizes[name]
            features = F.adaptive_avg_pool2d(base, (size, size))
            valid = torch.ones(
                images.shape[0], size, size, dtype=torch.bool, device=images.device
            )
            scales[name] = OriginEncoderScale(
                features=features,
                valid_mask=valid,
                metadata=OriginScaleMetadata(
                    name=name,
                    feature_index=feature_index,
                    channels=4,
                    output_stride=strides[name],
                    receptive_field=receptive_fields[name],
                    center_offset=0.5 * strides[name],
                    input_size=(32, 32),
                    lattice_size=(size, size),
                ),
            )
        return OriginEncoderOutput(scales)


def _loader() -> DataLoader:
    generator = torch.Generator().manual_seed(606)
    images = torch.randn(4, 3, 32, 32, generator=generator)
    masks = torch.ones(4, 1, 32, 32, dtype=torch.bool)
    labels = torch.tensor([0, 1, 2, 4])
    indices = torch.arange(4)
    return DataLoader(
        TensorDataset(images, masks, labels, indices),
        batch_size=2,
        shuffle=False,
    )


def _config(tmp_path: Path, *, relation_enabled: bool) -> OriginConfig:
    return OriginConfig(
        dataset="generic",
        n_classes=5,
        n_folds=2,
        val_fraction=0.2,
        run_dir=str(tmp_path),
        img_size=32,
        encoder="convnext_tiny",
        pretrained=False,
        evidence_scales=("s4", "s8", "s16", "s32"),
        projection_dim=8,
        reference_count=16.0,
        atom_rate_init=1e-4,
        prior_rate_init=1e-3,
        relation_enabled=relation_enabled,
        relation_source_scale="s8",
        relation_grid_size=2,
        relation_dim=6,
        relation_head_dim=3,
        relation_delta_cap=2.0,
        epochs=1,
        batch_size=2,
        num_workers=0,
        lr=1e-3,
        head_lr=2e-3,
        weight_decay=0.0,
        encoder_freeze_epochs=0,
        early_stopping_patience=1,
        amp=False,
    )


def _model(*, relation_enabled: bool) -> OriginModel:
    return OriginModel(
        num_classes=5,
        encoder_name="convnext_tiny",
        pretrained=False,
        encoder=_TinyPyramidEncoder(),
        evidence_scales=("s4", "s8", "s16", "s32"),
        projection_dim=8,
        reference_count=16.0,
        atom_mode="cumulative",
        atom_rate_init=1e-4,
        prior_rate_init=1e-3,
        relation_enabled=relation_enabled,
        relation_source_scale="s8",
        relation_grid_size=2,
        relation_dim=6,
        relation_head_dim=3,
        relation_delta_cap=2.0,
    )


def _write_v3_checkpoint(
    tmp_path: Path,
) -> tuple[Path, str, OriginConfig, dict[str, Any]]:
    """Write a self-consistent synthetic v3 checkpoint and return its hash."""

    source_cfg = _config(tmp_path / "source_run", relation_enabled=False)
    loader = _loader()
    source = OriginTrainer(
        _model(relation_enabled=False),
        loader,
        loader,
        None,
        source_cfg,
        tmp_path / "source_run",
        fold=0,
        split_signature="split-v1",
        device="cpu",
    )
    metrics = source._run_epoch(loader, train=False, epoch=0)
    key = validation_selection_key(
        metrics,
        policy=source.selection_policy,
        qwk_weight=source.selection_qwk_weight,
    )
    payload = source._checkpoint_payload(7, metrics, key, 7, 0)
    assert payload["schema"] == "origin-checkpoint-v3"
    checkpoint = tmp_path / "source_v3.pth"
    torch.save(payload, checkpoint)
    return checkpoint, trainer_module._file_sha256(checkpoint), source_cfg, payload


def _target_config(
    source_cfg: OriginConfig,
    checkpoint: Path,
    checksum: str,
    run_dir: Path,
) -> OriginConfig:
    return replace(
        source_cfg,
        run_dir=str(run_dir),
        relation_enabled=True,
        warm_start_checkpoint=str(checkpoint),
        warm_start_sha256=checksum,
        relation_only_epochs=source_cfg.epochs,
        early_stopping_patience=source_cfg.epochs,
    )


def test_v3_to_v6_migration_accepts_only_the_exact_relation_subset(tmp_path) -> None:
    checkpoint, checksum, source_cfg, payload = _write_v3_checkpoint(tmp_path)
    target = _model(relation_enabled=True)
    cfg = _target_config(
        source_cfg, checkpoint, checksum, tmp_path / "accepted_target"
    )
    provenance = trainer_module.load_origin_v3_relation_warm_start(
        target,
        cfg,
        fold=0,
        split_signature="split-v1",
    )
    assert provenance is not None
    expected_missing = sorted(
        name
        for name in target.state_dict()
        if name.startswith("generator.relation_field.")
    )
    assert provenance["initialized_relation_keys"] == expected_missing
    for name, value in payload["model_state"].items():
        assert torch.equal(target.state_dict()[name], value)


def test_v3_to_v6_migration_rejects_bad_hash_fold_and_split(tmp_path) -> None:
    checkpoint, checksum, source_cfg, _ = _write_v3_checkpoint(tmp_path)
    cfg = _target_config(source_cfg, checkpoint, checksum, tmp_path / "target")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        trainer_module.load_origin_v3_relation_warm_start(
            _model(relation_enabled=True),
            replace(cfg, warm_start_sha256="0" * 64),
            fold=0,
            split_signature="split-v1",
        )
    with pytest.raises(ValueError, match="fold mismatch"):
        trainer_module.load_origin_v3_relation_warm_start(
            _model(relation_enabled=True), cfg, fold=1, split_signature="split-v1"
        )
    with pytest.raises(ValueError, match="split signature mismatch"):
        trainer_module.load_origin_v3_relation_warm_start(
            _model(relation_enabled=True), cfg, fold=0, split_signature="different"
        )


def test_v3_to_v6_migration_rejects_a_hash_bound_extra_state_key(tmp_path) -> None:
    _, _, source_cfg, source_payload = _write_v3_checkpoint(tmp_path)
    payload = copy.deepcopy(source_payload)
    payload["model_state"]["unexpected.weight"] = torch.ones(1)
    payload["architecture"]["state_shapes"]["unexpected.weight"] = {
        "shape": [1],
        "dtype": "torch.float32",
    }
    payload["architecture_signature"] = trainer_module._canonical_sha256(
        payload["architecture"]
    )
    checkpoint = tmp_path / "source_v3_with_extra_key.pth"
    torch.save(payload, checkpoint)
    checksum = trainer_module._file_sha256(checkpoint)
    cfg = _target_config(source_cfg, checkpoint, checksum, tmp_path / "target")
    with pytest.raises(ValueError, match="unexpected model keys"):
        trainer_module.load_origin_v3_relation_warm_start(
            _model(relation_enabled=True), cfg, fold=0, split_signature="split-v1"
        )


def test_relation_only_training_freezes_and_evaluates_the_v3_base(tmp_path) -> None:
    cfg = _config(tmp_path / "relation_only", relation_enabled=True)
    cfg.relation_only_epochs = cfg.epochs
    loader = _loader()
    model = _model(relation_enabled=True)
    trainer = OriginTrainer(
        model,
        loader,
        loader,
        None,
        cfg,
        tmp_path / "relation_only",
        device="cpu",
    )
    relation_modes: list[bool] = []
    assert model.generator.relation_field is not None
    relation_output_before = (
        model.generator.relation_field.trilinear_weight.detach().clone()
    )
    frozen_before = {
        name: value.detach().clone()
        for name, value in model.state_dict().items()
        if not name.startswith("generator.relation_field.")
    }
    hook = model.generator.relation_field.register_forward_pre_hook(
        lambda module, inputs: relation_modes.append(module.training)
    )
    trainer._set_training_phase(1)
    try:
        trainer._run_epoch(loader, train=True, epoch=1)
    finally:
        hook.remove()

    for name, parameter in model.named_parameters():
        assert parameter.requires_grad == name.startswith("generator.relation_field.")
    encoder = model.encoder
    assert isinstance(encoder, _TinyPyramidEncoder)
    assert encoder.seen_training_modes
    assert not any(encoder.seen_training_modes)
    assert relation_modes and all(relation_modes)
    for name, expected in frozen_before.items():
        assert torch.equal(model.state_dict()[name], expected), name
    assert not torch.equal(
        model.generator.relation_field.trilinear_weight,
        relation_output_before,
    )


def test_higher_accuracy_candidate_with_worse_qwk_cannot_replace_epoch_zero_floor(
    tmp_path,
) -> None:
    checkpoint, checksum, source_cfg, source_payload = _write_v3_checkpoint(tmp_path)
    cfg = _target_config(
        source_cfg, checkpoint, checksum, tmp_path / "epoch0_target"
    )
    loader = _loader()
    trainer = OriginTrainer(
        _model(relation_enabled=True),
        loader,
        loader,
        None,
        cfg,
        tmp_path / "epoch0_target",
        fold=0,
        split_signature="split-v1",
        device="cpu",
    )
    source_metrics = copy.deepcopy(source_payload["metrics"])
    worse = copy.deepcopy(source_metrics)
    # Accuracy-first selection alone would accept this candidate.  The v6
    # safety floor must reject it because it damages ordinal agreement.
    worse["acc"] = float(source_metrics["acc"]) + 1.0
    worse["qwk"] = float(source_metrics["qwk"]) - 0.1
    worse["loss"] = float(source_metrics["loss"]) + 1.0
    for name in (
        "mean_abs_relation_log_odds",
        "max_abs_relation_log_odds",
        "mean_abs_relation_rate_delta",
        "max_abs_relation_rate_delta",
        "mean_abs_relation_edge_message",
        "max_abs_relation_edge_message",
    ):
        worse[name] = 0.0

    def fake_run_epoch(loader_arg, *, train: bool, epoch: int):
        del loader_arg
        if not train and epoch == 0:
            return copy.deepcopy(source_metrics)
        return copy.deepcopy(worse)

    trainer._run_epoch = fake_run_epoch  # type: ignore[method-assign]
    trainer._write_validation_certificates = (  # type: ignore[method-assign]
        lambda checkpoint_epoch: {
            "content_checksum_sha256": "f" * 64,
            "checkpoint_epoch": checkpoint_epoch,
        }
    )
    result = trainer.fit()
    saved = torch.load(trainer.best_path, map_location="cpu", weights_only=False)
    last = torch.load(trainer.last_path, map_location="cpu", weights_only=False)
    assert saved["epoch"] == 0
    assert result["best_epoch"] == 0
    assert result["epoch0_warm_start_eligible"] is True
    assert result["best_checkpoint_is_hash_bound_v3_floor"] is True
    assert result["selected_checkpoint_training_phase"] == "hash_bound_v3_floor"
    assert saved["training_phase"] == "hash_bound_v3_floor"
    assert result["warm_start_provenance"]["source_checkpoint_sha256"] == checksum
    candidate = last["candidate_metric_safety_floor_evaluation"]
    assert candidate["passes_normal_checkpoint_selector"] is True
    assert candidate["checks"]["accuracy_strict_improvement"] is True
    assert candidate["checks"]["qwk_non_regression"] is False
    assert candidate["passes_metric_floor"] is False
    assert candidate["checkpoint_eligible"] is False
    with trainer.history_path.open(newline="") as stream:
        history = list(csv.DictReader(stream))
    assert len(history) == 1
    assert history[0]["warm_start_floor_selector_improved"] == "True"
    assert history[0]["warm_start_floor_qwk_non_regression"] == "False"
    assert history[0]["warm_start_floor_checkpoint_eligible"] == "False"


def test_v6_certificate_emits_exact_relation_edge_replay(tmp_path) -> None:
    cfg = _config(tmp_path / "certificates", relation_enabled=True)
    cfg.relation_only_epochs = cfg.epochs
    loader = _loader()
    model = _model(relation_enabled=True)
    assert model.generator.relation_field is not None
    with torch.no_grad():
        model.generator.relation_field.trilinear_weight.fill_(0.3)
    trainer = OriginTrainer(
        model,
        loader,
        loader,
        None,
        cfg,
        tmp_path / "certificates",
        fold=0,
        split_signature="split-v1",
        device="cpu",
    )
    # Certificates bind to the selected checkpoint bytes; their contents are
    # irrelevant to this focused replay test.
    trainer.best_path.write_bytes(b"synthetic-selected-checkpoint")
    payload = trainer._write_validation_certificates(checkpoint_epoch=1)

    assert payload["schema"] == "origin-exact-validation-certificates-v6"
    assert payload["relation_certificates"]
    certificate = payload["relation_certificates"][0]
    assert certificate["selection"] == "largest_absolute_stored_pair_message"
    for field in (
        "boundary",
        "target_region",
        "source_region",
        "target_center_yx",
        "source_center_yx",
        "signed_pair_message",
        "baseline_class_probs",
        "replayed_class_probs",
        "class_probability_delta",
        "prediction_delta",
        "expected_grade_delta",
        "exact_base_ledger_invariance_error",
        "exact_edge_partition_error",
        "exact_reaggregation_error",
        "exact_final_merge_error",
    ):
        assert field in certificate
    assert certificate["exact_base_ledger_invariance_error"] == 0.0
    assert certificate["exact_edge_partition_error"] == 0.0
    assert certificate["exact_reaggregation_error"] <= 2e-10
    assert certificate["exact_final_merge_error"] <= 2e-10
    assert payload["exact_relation_base_ledger_invariance_error"] == 0.0
    assert payload["exact_relation_edge_partition_error"] == 0.0
    assert payload["exact_relation_reaggregation_error"] <= 2e-10
    assert payload["exact_relation_final_merge_error"] <= 2e-10
