"""Image-to-proof and data-path integration tests for MOSAIC."""

from pathlib import Path
import csv
import random
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.nn as nn
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader, TensorDataset

from configs.config import MOSAICConfig
from Datasets.mosaic_data import (
    MOSAIC_PREPROCESSING_VERSION,
    MosaicFundusTransform,
    MosaicImageDataset,
    _field_mask,
    _tight_field_crop,
    aptos_fold,
    eyepacs_fold,
    make_mosaic_loaders,
)
from models.local_efficientnet import downsample_retinal_field_mask
from models.mosaic_decoder import proof_only_decisions
from models.mosaic_model import MOSAICModel
from train_mosaic import (
    default_run_dir,
    parse_folds,
    split_signature,
    summary_filename,
)
from training.mosaic_trainer import MosaicTrainer


def test_mosaic_output_validation_allows_only_negative_infinite_log_stops() -> None:
    valid = SimpleNamespace(
        transitions=torch.tensor([[0.1, 0.2]]),
        dense_transitions=torch.tensor([[0.3, 0.4]]),
        log_stop_probabilities=torch.tensor([[-torch.inf, -2.0]]),
        dense_log_stop_probabilities=torch.tensor([[-3.0, -torch.inf]]),
    )
    assert MosaicTrainer._nonfinite_output_summary(valid) == ""

    invalid = SimpleNamespace(
        transitions=torch.tensor([[torch.nan, 0.2]]),
        dense_transitions=torch.tensor([[0.3, torch.inf]]),
        log_stop_probabilities=torch.tensor([[torch.inf, -2.0]]),
        dense_log_stop_probabilities=torch.tensor([[torch.nan, -torch.inf]]),
    )
    summary = MosaicTrainer._nonfinite_output_summary(invalid)
    assert "transitions[nonfinite=1]" in summary
    assert "dense_transitions[nonfinite=1]" in summary
    assert "log_stop_probabilities[nonfinite=1]" in summary
    assert "dense_log_stop_probabilities[nonfinite=1]" in summary


def _tiny_model(**overrides) -> MOSAICModel:
    kwargs = {
        "num_classes": 5,
        "image_size": 64,
        "local_stage": "rf_medium",
        "local_dim": 8,
        "pretrained": False,
        "max_count": 4,
        "sufficiency_tolerance": 0.02,
        "complement_suppression": 0.5,
        "count_implementation": "block_tree",
        "count_block_size": 8,
    }
    kwargs.update(overrides)
    return MOSAICModel(**kwargs).eval()


def _tiny_trainer(
    tmp_path,
    *,
    cfg_overrides=None,
    fold=0,
    signature="split-a",
    train_labels=None,
):
    cfg_values = {
        "img_size": 64,
        "batch_size": 2,
        "epochs": 1,
        "num_workers": 0,
        "pretrained": False,
        "local_stage": "rf_medium",
        "evidence_dim": 8,
        "max_count": 4,
        "count_block_size": 8,
        "amp": False,
    }
    cfg_values.update(cfg_overrides or {})
    cfg = MOSAICConfig(**cfg_values)
    trainer = MosaicTrainer(
        _tiny_model(),
        [],
        [],
        [],
        cfg,
        str(tmp_path),
        [0, 1, 2, 3, 4] if train_labels is None else train_labels,
        fold=fold,
        split_signature=signature,
        device=torch.device("cpu"),
    )
    return trainer


def test_end_to_end_offline_forward_at_64_pixels() -> None:
    torch.manual_seed(20)
    model = _tiny_model()
    image = torch.randn(2, 3, 64, 64)
    pixel_mask = torch.ones(2, 1, 64, 64, dtype=torch.bool)
    pixel_mask[1, :, :, 32:] = False

    with torch.no_grad():
        output = model(
            image,
            pixel_mask,
            return_pivotality=True,
            return_local_features=True,
        )

    # RF-medium is stride 8: a 64x64 canvas gives 8x8 local witnesses.
    assert output.local_features is not None
    assert output.local_features.shape == (2, 64, 8)
    assert output.valid_mask.shape == (2, 64)
    assert output.lattice.input_size == (64, 64)
    assert output.lattice.lattice_size == (8, 8)
    assert output.transitions.shape == (2, 4)
    assert output.class_probabilities.shape == (2, 5)
    assert output.predicted_grade.shape == (2,)
    assert output.proof.selected_mask.shape == (2, 64, 4)
    assert output.evidence.pivotality is not None
    assert output.evidence.pivotality.shape == (2, 64, 4)
    assert torch.isfinite(output.class_probabilities).all()
    assert torch.all(output.class_probabilities >= 0)
    torch.testing.assert_close(
        output.class_probabilities.sum(dim=-1),
        torch.ones(2),
        atol=2e-6,
        rtol=0,
    )
    assert not output.proof.selected_mask[1][~output.valid_mask[1]].any()


def test_configured_direct_forward_exposes_selected_proof_only_decision() -> None:
    torch.manual_seed(201)
    model = _tiny_model()
    weights = torch.tensor(
        [[0.4, 2.0], [1.8, 0.7], [0.6, 3.0], [2.4, 0.5]]
    )
    model.configure_proof_decoder("deweighted_class_map", weights)
    with torch.no_grad():
        output = model(torch.randn(2, 3, 64, 64))
    expected = proof_only_decisions(
        output.transitions,
        output.log_stop_probabilities,
        weights,
    )

    assert output.decision_rule == "deweighted_class_map"
    torch.testing.assert_close(
        output.class_probabilities,
        expected.deweighted_class_probabilities,
    )
    torch.testing.assert_close(
        output.cumulative_probabilities,
        expected.deweighted_cumulative_probabilities,
    )
    torch.testing.assert_close(
        output.expected_grade,
        expected.deweighted_expected_grade,
    )
    assert torch.equal(output.predicted_grade, expected.deweighted_argmax)
    assert torch.equal(output.argmax_grade, expected.deweighted_argmax)
    # The original core law remains explicit for audits and old consumers.
    assert output.raw_class_probabilities is output.evidence.class_probabilities
    assert output.raw_predicted_grade is output.evidence.predicted_grade


def test_runtime_decoder_configuration_does_not_change_checkpoint_state() -> None:
    model = _tiny_model()
    original_keys = tuple(model.state_dict().keys())
    state = {name: value.clone() for name, value in model.state_dict().items()}
    model.configure_proof_decoder(
        "deweighted_posterior_median",
        torch.tensor(
            [[0.5, 2.0], [1.5, 0.8], [0.7, 1.9], [2.1, 0.6]]
        ),
    )
    assert tuple(model.state_dict().keys()) == original_keys
    assert "_decision_transition_weights" not in model.state_dict()

    legacy_default = _tiny_model()
    legacy_default.load_state_dict(state, strict=True)
    assert legacy_default.decision_rule == "rounded_expected"


@pytest.mark.parametrize("decision_rule", ["rounded_expected", "deweighted_class_map"])
def test_model_configuration_rejects_incomplete_boundary_support(
    decision_rule: str,
) -> None:
    model = _tiny_model()
    weights = torch.ones(4, 2)
    weights[-1, 1] = 0.0
    with pytest.raises(ValueError, match="fold incomplete"):
        model.configure_proof_decoder(decision_rule, weights)


def test_model_initial_count_prior_matches_canonical_fallback_support() -> None:
    model = _tiny_model()
    with torch.no_grad():
        output = model(torch.zeros(1, 3, 64, 64))
    assert model.expected_valid_cells == int(output.valid_mask.sum())
    assert (
        model.proof_head.local_state_head.expected_num_cells
        == model.expected_valid_cells
    )


def test_cached_local_features_reproduce_image_forward_exactly() -> None:
    torch.manual_seed(21)
    model = _tiny_model()
    image = torch.randn(1, 3, 64, 64)
    pixel_mask = torch.ones(1, 1, 64, 64, dtype=torch.bool)

    with torch.no_grad():
        image_output = model(
            image,
            pixel_mask,
            return_pivotality=True,
            return_local_features=True,
        )
        cached_output = model.forward_from_features(
            image_output.local_features,
            image_output.valid_mask,
            project=True,
            return_pivotality=True,
        )

    torch.testing.assert_close(
        cached_output.transitions, image_output.transitions, atol=0, rtol=0
    )
    torch.testing.assert_close(
        cached_output.class_probabilities,
        image_output.class_probabilities,
        atol=0,
        rtol=0,
    )
    torch.testing.assert_close(
        cached_output.witness_probabilities,
        image_output.evidence.witness_probabilities,
        atol=0,
        rtol=0,
    )
    torch.testing.assert_close(
        cached_output.pivotality,
        image_output.evidence.pivotality,
        atol=0,
        rtol=0,
    )
    assert torch.equal(
        cached_output.proof.selected_mask, image_output.proof.selected_mask
    )
    assert torch.equal(cached_output.proof.proof_size, image_output.proof.proof_size)


def test_optimizer_uses_head_lr_for_random_pointwise_adapter(tmp_path: Path) -> None:
    trainer = _tiny_trainer(tmp_path)
    groups = {group["name"]: group for group in trainer.optimizer.param_groups}
    trunk_ids = {id(parameter) for parameter in trainer.model.encoder.trunk.parameters()}
    pointwise_ids = {
        id(parameter) for parameter in trainer.model.encoder.pointwise.parameters()
    }
    proof_ids = {id(parameter) for parameter in trainer.model.proof_head.parameters()}

    assert {id(parameter) for parameter in groups["local_encoder"]["params"]} == trunk_ids
    head_ids = {id(parameter) for parameter in groups["proof_head"]["params"]}
    assert pointwise_ids <= head_ids
    assert proof_ids <= head_ids
    assert groups["local_encoder"]["lr"] == trainer.cfg.lr
    assert groups["proof_head"]["lr"] == trainer.cfg.head_lr


def test_proof_tolerance_setter_changes_only_the_projector_threshold() -> None:
    model = _tiny_model(sufficiency_tolerance=0.0)
    before = {
        name: parameter.detach().clone()
        for name, parameter in model.named_parameters()
    }
    model.set_proof_tolerance(0.125)

    assert model.proof_tolerance == pytest.approx(0.125)
    for name, parameter in model.named_parameters():
        torch.testing.assert_close(parameter, before[name], atol=0, rtol=0)
    with pytest.raises(ValueError, match="non-negative"):
        model.set_proof_tolerance(-1e-6)


def test_resume_validation_allows_epoch_extension_but_rejects_split_changes(
    tmp_path: Path,
) -> None:
    original = _tiny_trainer(tmp_path / "original", cfg_overrides={"epochs": 1})
    state = original._checkpoint_payload(epoch=1, metrics={}, best_accuracy=80.0)

    extended = _tiny_trainer(
        tmp_path / "extended", cfg_overrides={"epochs": 5}, signature="split-a"
    )
    extended._validate_resume_checkpoint(state)

    wrong_fold = _tiny_trainer(tmp_path / "fold", fold=1, signature="split-a")
    with pytest.raises(ValueError, match="current fold"):
        wrong_fold._validate_resume_checkpoint(state)

    wrong_split = _tiny_trainer(tmp_path / "split", signature="split-b")
    with pytest.raises(ValueError, match="split signature"):
        wrong_split._validate_resume_checkpoint(state)

    wrong_seed = _tiny_trainer(
        tmp_path / "seed", cfg_overrides={"seed": original.cfg.seed + 1}
    )
    with pytest.raises(ValueError, match="seed"):
        wrong_seed._validate_resume_checkpoint(state)

    wrong_workers = _tiny_trainer(
        tmp_path / "workers", cfg_overrides={"num_workers": 1}
    )
    with pytest.raises(ValueError, match="num_workers"):
        wrong_workers._validate_resume_checkpoint(state)

    wrong_implementation = dict(state)
    wrong_implementation["implementation_signature"] = "0" * 64
    with pytest.raises(ValueError, match="different MOSAIC implementation"):
        extended._validate_resume_checkpoint(wrong_implementation)

    missing_implementation = dict(state)
    missing_implementation.pop("implementation_signature")
    with pytest.raises(ValueError, match="no MOSAIC implementation signature"):
        extended._validate_resume_checkpoint(missing_implementation)


def test_checkpoint_rng_state_restores_python_numpy_and_torch(tmp_path: Path) -> None:
    trainer = _tiny_trainer(tmp_path)
    random.seed(71)
    np.random.seed(71)
    torch.manual_seed(71)
    state = trainer._checkpoint_payload(epoch=1, metrics={}, best_accuracy=0.0)
    expected = (random.random(), float(np.random.rand()), float(torch.rand(())))

    random.seed(999)
    np.random.seed(999)
    torch.manual_seed(999)
    trainer._restore_rng_state(state)
    actual = (random.random(), float(np.random.rand()), float(torch.rand(())))
    assert actual == pytest.approx(expected)


def test_checkpoint_write_is_atomic_and_history_reconciles_to_resume_epoch(
    tmp_path: Path,
) -> None:
    trainer = _tiny_trainer(tmp_path)
    trainer._append_history({"epoch": 1, "val_qwk": 0.1})
    trainer._append_history({"epoch": 2, "val_qwk": 0.2})
    trainer._append_history({"epoch": 2, "val_qwk": 0.3})
    trainer._reconcile_history(completed_epoch=2)
    with trainer.history_path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert [int(row["epoch"]) for row in rows] == [1, 2]
    assert float(rows[-1]["val_qwk"]) == pytest.approx(0.3)

    trainer._save(
        epoch=2,
        metrics={"qwk": 0.3},
        best_accuracy=80.0,
        best=False,
    )
    state = torch.load(
        trainer.last_checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    assert int(state["epoch"]) == 2
    assert not list(tmp_path.glob(".*.tmp"))

    # A preemption after CSV append but before checkpoint commit leaves a
    # future row; resume must discard it before the epoch is re-run.
    trainer._append_history({"epoch": 3, "val_qwk": 0.9})
    trainer._reconcile_history(completed_epoch=2)
    with trainer.history_path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert [int(row["epoch"]) for row in rows] == [1, 2]


def test_validation_only_fit_uses_accuracy_selection_and_loss_scheduler(
    tmp_path: Path,
) -> None:
    trainer = _tiny_trainer(tmp_path, cfg_overrides={"epochs": 2})
    trainer.train_loader = "train-loader"
    trainer.val_loader = "validation-loader"
    trainer.test_loader = "outer-test-loader"
    calls = []

    def fake_epoch(loader, *, train: bool, epoch: int):
        calls.append(loader)
        # Accuracy improves while QWK declines. The second epoch must still be
        # selected, and the plateau scheduler must consume the continuous loss.
        validation = {
            1: {"loss": 1.0, "acc": 42.0, "qwk": 0.50, "mae": 0.8},
            2: {"loss": 0.8, "acc": 43.0, "qwk": 0.31, "mae": 0.7},
        }
        metrics = validation[epoch] if not train else {
            "loss": 1.2,
            "acc": 40.0,
            "qwk": 0.2,
            "mae": 0.9,
        }
        return {
            **metrics,
            "proof_size_median": 3.0,
            "proof_fraction_mean": 0.2,
            "proof_tolerance": 0.0,
        }

    trainer._run_epoch = fake_epoch
    result = trainer.fit(evaluate_test=False)
    assert calls == [
        "train-loader",
        "validation-loader",
        "train-loader",
        "validation-loader",
    ]
    assert result["test_evaluated"] is False
    assert result["best_epoch"] == 2
    assert result["best_validation_metrics"]["acc"] == pytest.approx(43.0)
    assert result["best_validation_metrics"]["qwk"] == pytest.approx(0.31)
    assert trainer.scheduler.mode == "min"
    assert trainer.scheduler.best == pytest.approx(0.8)


def test_epoch_diagnostics_aggregate_boundary_risk_sets_exactly(tmp_path: Path) -> None:
    trainer = _tiny_trainer(tmp_path)
    labels = torch.tensor([0, 1, 2, 4])
    loader = DataLoader(
        TensorDataset(
            torch.randn(4, 3, 64, 64),
            torch.ones(4, 1, 64, 64, dtype=torch.bool),
            labels,
            torch.arange(4),
        ),
        batch_size=2,
        shuffle=False,
    )
    metrics = trainer._run_epoch(loader, train=False, epoch=1)
    assert metrics["decision_rule"] == "posterior_median"
    assert metrics["acc"] == pytest.approx(
        metrics["decoder_posterior_median_acc"]
    )
    for rule in (
        "rounded_expected",
        "class_map",
        "posterior_median",
        "deweighted_mean_round",
        "deweighted_class_map",
        "deweighted_posterior_median",
    ):
        for metric in ("acc", "mae", "qwk", "ece"):
            assert f"decoder_{rule}_{metric}" in metrics
    expected_risk = (4.0, 3.0, 2.0, 1.0)
    expected_advance = (3 / 4, 2 / 3, 1 / 2, 1.0)
    for boundary in range(4):
        assert metrics[f"at_risk_boundary_{boundary}"] == expected_risk[boundary]
        assert metrics[f"advance_rate_boundary_{boundary}"] == pytest.approx(
            expected_advance[boundary]
        )
        assert 0.0 <= metrics[f"zero_proof_rate_boundary_{boundary}"] <= 1.0
        assert (
            0.0
            <= metrics[f"zero_proof_advance_rate_boundary_{boundary}"]
            <= 1.0
        )
        assert (
            0.0
            <= metrics[f"zero_transition_advance_rate_boundary_{boundary}"]
            <= 1.0
        )


def test_invalid_proof_decision_rule_fails_before_training(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="unknown MOSAIC decision rule"):
        _tiny_trainer(
            tmp_path,
            cfg_overrides={"decision_rule": "validation_tuned_threshold"},
        )


@pytest.mark.parametrize("decision_rule", ["rounded_expected", "deweighted_class_map"])
def test_incomplete_boundary_support_fails_at_trainer_initialization(
    tmp_path: Path,
    decision_rule: str,
) -> None:
    # With no grade-4 example, advance at the final boundary is unobserved and
    # receives zero criterion weight.  That split cannot identify the complete
    # declared five-grade model, regardless of the selected point rule.
    labels_without_grade_four = [0, 0, 1, 1, 2, 2, 3, 3]
    with pytest.raises(ValueError, match="invalid MOSAIC training fold"):
        _tiny_trainer(
            tmp_path / decision_rule,
            cfg_overrides={"decision_rule": decision_rule},
            train_labels=labels_without_grade_four,
        )


def test_decision_rule_is_resume_critical_and_serialized(tmp_path: Path) -> None:
    original = _tiny_trainer(
        tmp_path / "original",
        cfg_overrides={"decision_rule": "deweighted_class_map"},
    )
    state = original._checkpoint_payload(epoch=1, metrics={}, best_accuracy=0.0)
    assert state["architecture"]["decision_rule"] == "deweighted_class_map"
    assert state["architecture"]["no_global_bypass"] is True
    assert state["architecture"]["decision_inputs"] == (
        "selected_proof_transitions",
        "selected_proof_log_stop_probabilities",
        "training_fold_boundary_outcome_weights",
    )
    assert state["config"]["transition_reduction"] == "boundary_mean"
    assert state["architecture"]["transition_reduction"] == "boundary_mean"
    assert state["architecture"]["training_fold_at_risk_counts"] == [
        5,
        4,
        3,
        2,
    ]
    assert torch.equal(
        state["criterion_state"]["at_risk_counts"],
        torch.tensor([5, 4, 3, 2]),
    )

    changed = _tiny_trainer(
        tmp_path / "changed",
        cfg_overrides={"decision_rule": "rounded_expected"},
    )
    with pytest.raises(ValueError, match="decision_rule"):
        changed._validate_resume_checkpoint(state)


def test_transition_reduction_is_resume_critical_with_legacy_identity(
    tmp_path: Path,
) -> None:
    boundary_mean = _tiny_trainer(tmp_path / "boundary-mean")
    state = boundary_mean._checkpoint_payload(
        epoch=1, metrics={}, best_accuracy=0.0
    )
    sample_mean = _tiny_trainer(
        tmp_path / "sample-mean",
        cfg_overrides={"transition_reduction": "sample_mean"},
    )
    with pytest.raises(ValueError, match="transition_reduction"):
        sample_mean._validate_resume_checkpoint(state)

    # Missing metadata denotes the historical sample-mean objective; it must
    # never silently inherit the prospective boundary-mean default.
    legacy_state = dict(state)
    legacy_state["config"] = dict(state["config"])
    legacy_state["config"].pop("transition_reduction")
    sample_mean._validate_resume_checkpoint(legacy_state)
    with pytest.raises(ValueError, match="transition_reduction"):
        boundary_mean._validate_resume_checkpoint(legacy_state)


def test_boundary_mean_rejects_nonuniform_training_sampler(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="uniformly sampled"):
        _tiny_trainer(
            tmp_path,
            cfg_overrides={"stratified": True},
        )


def test_cpu_training_step_remains_finite_and_updates_parameters(tmp_path: Path) -> None:
    trainer = _tiny_trainer(tmp_path)
    labels = torch.tensor([0, 1, 3, 4])
    loader = DataLoader(
        TensorDataset(
            torch.randn(4, 3, 64, 64),
            torch.ones(4, 1, 64, 64, dtype=torch.bool),
            labels,
            torch.arange(4),
        ),
        batch_size=2,
        shuffle=False,
    )
    before = trainer.model.proof_head.local_state_head.linear.bias.detach().clone()
    metrics = trainer._run_epoch(loader, train=True, epoch=1)
    after = trainer.model.proof_head.local_state_head.linear.bias.detach()
    assert metrics["amp_skipped_steps"] == 0.0
    assert metrics["amp_loss_scale"] == 1.0
    assert torch.isfinite(after).all()
    assert not torch.equal(before, after)


def test_non_amp_nonfinite_gradient_is_immediately_fatal(tmp_path: Path) -> None:
    trainer = _tiny_trainer(tmp_path)
    parameter = trainer.model.proof_head.local_state_head.linear.bias
    hook = parameter.register_hook(lambda gradient: torch.full_like(gradient, torch.inf))
    loader = DataLoader(
        TensorDataset(
            torch.randn(2, 3, 64, 64),
            torch.ones(2, 1, 64, 64, dtype=torch.bool),
            torch.tensor([0, 4]),
            torch.arange(2),
        ),
        batch_size=2,
    )
    try:
        with pytest.raises(FloatingPointError, match="without AMP"):
            trainer._run_epoch(loader, train=True, epoch=1)
    finally:
        hook.remove()


def test_no_global_classifier_bypass_exists_statically_or_functionally() -> None:
    torch.manual_seed(22)
    model = _tiny_model()
    forbidden_module_types = (nn.AdaptiveAvgPool2d, nn.MultiheadAttention)
    assert not any(
        isinstance(module, forbidden_module_types) for module in model.modules()
    )
    forbidden_names = ("classifier", "regression_head", "ordinal_head", "ctot", "gpa")
    assert not any(
        any(forbidden in name.lower() for forbidden in forbidden_names)
        for name, _ in model.named_modules()
    )

    # With no valid regional witnesses, two completely different images must
    # both reduce to the normal-state proof and grade-0 distribution.  A
    # pooled/global residual branch would violate this invariant.
    images = torch.stack((torch.zeros(3, 64, 64), torch.randn(3, 64, 64)))
    empty_mask = torch.zeros(2, 1, 64, 64, dtype=torch.bool)
    with torch.no_grad():
        output = model(images, empty_mask)
    torch.testing.assert_close(output.transitions, torch.zeros(2, 4), atol=0, rtol=0)
    expected = torch.zeros(2, 5)
    expected[:, 0] = 1.0
    torch.testing.assert_close(output.class_probabilities, expected, atol=0, rtol=0)
    assert not output.proof.selected_mask.any()


def _synthetic_fundus(width: int = 80, height: int = 40) -> Image.Image:
    image = Image.new("RGB", (width, height), color=(0, 0, 0))
    draw = ImageDraw.Draw(image)
    draw.ellipse((5, 3, width - 6, height - 4), fill=(155, 55, 24))
    draw.ellipse((width // 2 - 2, height // 2 - 2, width // 2 + 2, height // 2 + 2), fill=(20, 8, 3))
    return image


def test_full_canvas_transform_returns_aligned_image_and_boolean_mask() -> None:
    transform = MosaicFundusTransform(size=64, augment=False)
    image, valid_mask = transform(_synthetic_fundus())

    assert image.shape == (3, 64, 64)
    assert image.dtype == torch.float32
    assert valid_mask.shape == (1, 64, 64)
    assert valid_mask.dtype == torch.bool
    assert torch.isfinite(image).all()
    assert valid_mask.any()
    assert (~valid_mask).any()
    # A fixed centred ellipse excludes canonical-canvas corners without
    # exposing acquisition padding or source aspect ratio.
    assert not valid_mask[0, 0, 0]
    assert not valid_mask[0, 0, -1]
    assert not valid_mask[0, -1, 0]
    assert not valid_mask[0, -1, -1]
    assert transform.preprocessing_version == MOSAIC_PREPROCESSING_VERSION


def test_canonical_proof_mask_is_identical_across_source_aspect_ratios() -> None:
    transform = MosaicFundusTransform(size=64, augment=False)
    sources = (
        _synthetic_fundus(width=120, height=40),
        _synthetic_fundus(width=40, height=120),
        _synthetic_fundus(width=80, height=80),
    )

    transformed = [transform(source) for source in sources]
    reference = transformed[0][1]
    assert reference.any()
    for image, mask in transformed:
        assert image.shape == (3, 64, 64)
        assert torch.equal(mask, reference)

    lattice_masks = [
        downsample_retinal_field_mask(mask.unsqueeze(0), (8, 8)).flatten(1)
        for _, mask in transformed
    ]
    lattice_counts = [int(mask.sum()) for mask in lattice_masks]
    assert len(set(lattice_counts)) == 1
    assert lattice_counts[0] > 0


def test_canonical_proof_mask_never_becomes_empty_for_dark_or_tiny_sources() -> None:
    for size in (16, 31, 64):
        transform = MosaicFundusTransform(size=size, augment=False)
        sources = (
            Image.new("RGB", (1, 1), color=(0, 0, 0)),
            Image.new("RGB", (200, 20), color=(0, 0, 0)),
            _synthetic_fundus(width=20, height=200),
        )
        masks = [transform(source)[1] for source in sources]
        assert all(mask.any() for mask in masks)
        assert all(torch.equal(mask, masks[0]) for mask in masks[1:])


def test_dominant_field_rejects_disconnected_bright_camera_artifact() -> None:
    image = Image.new("RGB", (120, 80), color=(0, 0, 0))
    draw = ImageDraw.Draw(image)
    draw.ellipse((30, 10, 90, 70), fill=(155, 55, 24))
    draw.rectangle((0, 0, 12, 5), fill=(255, 255, 255))

    cropped = _tight_field_crop(image)
    # The isolated label must not enlarge the retinal crop to the left border.
    assert cropped.width < 80
    assert cropped.height > 50

    # The proof mask is fixed and cannot re-introduce the rejected artifact.
    mask = torch.from_numpy(np.array(_field_mask(image), copy=True)).bool()
    assert mask[40, 60]
    assert not mask[2, 5]
    assert not mask[0, 119]


def test_impossible_stratified_batch_size_fails_loudly() -> None:
    items = [(f"unused-{grade}.png", grade) for grade in range(5)]
    with pytest.raises(ValueError, match="batch_size >= number"):
        make_mosaic_loaders(
            items,
            items,
            items,
            image_size=64,
            batch_size=4,
            num_workers=0,
            pin_memory=False,
            seed=1,
            stratified=True,
        )


def test_loader_workers_are_recreated_for_exact_resume_rng(tmp_path: Path) -> None:
    path = tmp_path / "fundus.png"
    _synthetic_fundus().save(path)
    items = [(str(path), grade) for grade in range(5)]
    train_loader, val_loader, test_loader = make_mosaic_loaders(
        items,
        items,
        items,
        image_size=64,
        batch_size=5,
        num_workers=1,
        pin_memory=False,
        seed=1,
    )
    # Persistent workers retain an uncheckpointed private augmentation RNG.
    # Recreating them each epoch makes the restored torch RNG authoritative.
    assert train_loader.persistent_workers is False
    assert val_loader.persistent_workers is False
    assert test_loader.persistent_workers is False


def test_stratified_sampler_sequence_continues_exactly_after_resume(
    tmp_path: Path,
) -> None:
    items = [
        (f"unused-grade-{grade}-sample-{sample}.png", grade)
        for grade in range(5)
        for sample in range(4)
    ]

    def build_trainer(directory: Path):
        loaders = make_mosaic_loaders(
            items,
            items,
            items,
            image_size=64,
            batch_size=5,
            num_workers=0,
            pin_memory=False,
            seed=91,
            stratified=True,
        )
        cfg = MOSAICConfig(
            img_size=64,
            batch_size=5,
            epochs=3,
            num_workers=0,
            pretrained=False,
            local_stage="rf_medium",
            evidence_dim=8,
            max_count=4,
            count_block_size=8,
            amp=False,
            stratified=True,
            transition_reduction="sample_mean",
            seed=91,
        )
        return MosaicTrainer(
            _tiny_model(),
            *loaders,
            cfg,
            str(directory),
            [label for _, label in items],
            fold=0,
            split_signature="stratified-split",
            device=torch.device("cpu"),
        )

    original = build_trainer(tmp_path / "original-stratified")
    first_epoch = list(iter(original.train_loader.batch_sampler))
    assert first_epoch
    state = original._checkpoint_payload(epoch=1, metrics={}, best_accuracy=0.0)
    expected_second_epoch = list(iter(original.train_loader.batch_sampler))

    resumed = build_trainer(tmp_path / "resumed-stratified")
    resumed._validate_resume_checkpoint(state)
    resumed._restore_training_sampler_state(state)
    actual_second_epoch = list(iter(resumed.train_loader.batch_sampler))
    assert actual_second_epoch == expected_second_epoch


def test_dataset_default_run_dirs_and_fold_summaries_do_not_collide() -> None:
    assert default_run_dir("aptos") == "runs/mosaic_aptos"
    assert default_run_dir("dr") == "runs/mosaic_dr"
    assert summary_filename([0], 5) == "final_results_folds_0.csv"
    assert summary_filename([1], 5) == "final_results_folds_1.csv"
    assert summary_filename(list(range(5)), 5) == "final_results.csv"
    assert parse_folds("0,2", 5) == [0, 2]
    with pytest.raises(ValueError, match="at least one"):
        parse_folds("", 5)
    with pytest.raises(ValueError, match="duplicate"):
        parse_folds("0,0", 5)


def test_dataset_item_preserves_label_and_stable_sample_index(tmp_path: Path) -> None:
    path = tmp_path / "synthetic.png"
    _synthetic_fundus().save(path)
    dataset = MosaicImageDataset(
        [(str(path), 3)], MosaicFundusTransform(size=64, augment=False)
    )

    image, valid_mask, label, sample_index = dataset[0]
    assert image.shape == (3, 64, 64)
    assert valid_mask.shape == (1, 64, 64)
    assert label.dtype == torch.long
    assert int(label) == 3
    assert sample_index == 0


def test_aptos_fold_is_disjoint_complete_stratified_and_deterministic() -> None:
    items = [
        (f"grade-{grade}-sample-{sample}.png", grade)
        for grade in range(5)
        for sample in range(20)
    ]
    train, validation, test = aptos_fold(
        items, fold=2, n_folds=5, val_fraction=0.2, seed=314
    )
    repeated = aptos_fold(items, fold=2, n_folds=5, val_fraction=0.2, seed=314)

    train_set, validation_set, test_set = map(set, (train, validation, test))
    assert train_set.isdisjoint(validation_set)
    assert train_set.isdisjoint(test_set)
    assert validation_set.isdisjoint(test_set)
    assert train_set | validation_set | test_set == set(items)
    assert repeated == (train, validation, test)

    for grade in range(5):
        assert sum(label == grade for _, label in test) == 4
        assert sum(label == grade for _, label in validation) == 3
        assert sum(label == grade for _, label in train) == 13


def test_split_signature_is_root_independent_and_partition_sensitive() -> None:
    train_a = [("/first/root/a.png", 0), ("/first/root/b.png", 1)]
    train_b = [("/moved/root/b.png", 1), ("/moved/root/a.png", 0)]
    validation = [("/first/root/c.png", 2)]
    test = [("/first/root/d.png", 3)]
    signature_a = split_signature(
        ("train", train_a), ("validation", validation), ("test", test)
    )
    signature_b = split_signature(
        ("train", train_b), ("validation", validation), ("test", test)
    )
    assert signature_a == signature_b
    assert signature_a != split_signature(
        ("train", train_a + validation), ("validation", []), ("test", test)
    )


def test_eyepacs_fold_keeps_both_eyes_patient_disjoint() -> None:
    items = []
    for patient in range(100):
        grade = patient % 5
        items.extend(
            [
                (f"/images/{patient}_left.jpeg", grade),
                (f"/images/{patient}_right.jpeg", grade),
            ]
        )
    train, validation, test = eyepacs_fold(
        items, fold=3, n_folds=10, val_fraction=0.1, seed=9
    )

    def patients(split):
        return {Path(path).stem.rsplit("_", 1)[0] for path, _ in split}

    train_patients = patients(train)
    validation_patients = patients(validation)
    test_patients = patients(test)
    assert train_patients.isdisjoint(validation_patients)
    assert train_patients.isdisjoint(test_patients)
    assert validation_patients.isdisjoint(test_patients)
    assert train_patients | validation_patients | test_patients == {
        str(patient) for patient in range(100)
    }
    for split in (train, validation, test):
        counts = {}
        for path, _ in split:
            patient = Path(path).stem.rsplit("_", 1)[0]
            counts[patient] = counts.get(patient, 0) + 1
        assert set(counts.values()) == {2}
