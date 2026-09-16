"""Integration/provenance invariants for the PATHS experiment."""

from __future__ import annotations

import hashlib
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from configs.origin_config import OriginConfig
from configs.paths_config import PathsConfig
from models.origin import ConservedOrdinalGenerator, OriginModel, decode_pure_birth_rates
from models.origin_encoder import (
    OriginEncoderOutput,
    OriginEncoderScale,
    OriginScaleMetadata,
)
from models.paths import (
    PathsContinuationRefiner,
    PathsModel,
    continuation_logits_from_log_probs,
    decode_continuation_logits,
    replay_paths_without,
)
from training.origin_trainer import _architecture_record, _canonical_sha256, _critical_config
from training.paths_trainer import (
    _PATHS_IMPLEMENTATION_FILES,
    _fixed_grade_decision_margin,
    audit_paths_joint_replay,
    load_audited_origin_v3_warm_start,
    paths_implementation_signature,
    PathsTrainer,
    rank_paths_singleton_witnesses,
    target_log_probability_boundary_contributions,
)
from losses.paths import risk_set_boundary_weights


class _TinyEncoder(nn.Module):
    STAGE_CHANNELS = {"s4": 2, "s8": 3, "s16": 4, "s32": 5}

    def __init__(self):
        super().__init__()
        self.audit_parameter = nn.Parameter(torch.ones(()))

    def forward(self, images, pixel_valid_mask=None):  # pragma: no cover
        raise NotImplementedError


def _model_kwargs() -> dict:
    return {
        "num_classes": 5,
        "encoder_name": "convnext_tiny",
        "encoder": _TinyEncoder(),
        "pretrained": False,
        "evidence_scales": ("s4", "s8", "s16", "s32"),
        "projection_dim": 8,
        "reference_count": 4096.0,
        "atom_mode": "cumulative",
        "atom_rate_init": 1e-6,
        "prior_rate_init": 1e-4,
        "boundary_scale_init": 1.0,
        "total_rate_cap": 64.0,
        "prior_rate_cap": 1.0,
        "boundary_scale_cap": 2.0,
        "rate_roundoff_margin": 1.0,
    }


def _source_checkpoint(path: Path, *, split: str = "split-a") -> str:
    model = OriginModel(**_model_kwargs())
    architecture = _architecture_record(model)
    cfg = OriginConfig()
    critical = _critical_config(cfg)
    state = {
        "schema": "origin-checkpoint-v3",
        "fold": 0,
        "split_signature": split,
        "implementation_signature": "a" * 64,
        "architecture": architecture,
        "architecture_signature": _canonical_sha256(architecture),
        "config": vars(cfg),
        "critical_config": critical,
        "config_signature": _canonical_sha256(critical),
        "model_state": model.state_dict(),
        "metrics": {
            "acc": 86.0,
            "qwk": 0.92,
            "mae": 0.18,
            "balanced_acc": 60.0,
            "macro_f1": 0.60,
        },
        "likelihood_component_unweighted": True,
        "population_objective_proper": True,
    }
    torch.save(state, path)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_paths_config_rejects_protocol_drift() -> None:
    cfg = PathsConfig()
    assert cfg.evidence_scales == ("s4", "s8", "s16", "s32")
    assert cfg.class_weighting == "none"
    assert cfg.stratified_batches is False
    with pytest.raises(ValueError, match="four-scale"):
        PathsConfig(evidence_scales=("s8", "s16"))
    with pytest.raises(ValueError, match="supplied together"):
        PathsConfig(warm_start_checkpoint="best.pth")
    with pytest.raises(ValueError, match="outcome weighting"):
        PathsConfig(class_weighting="inverse_frequency", allow_weighted_likelihood=True)
    with pytest.raises(ValueError, match="correction-only"):
        PathsConfig(correction_only_epochs=4, encoder_freeze_epochs=3)
    with pytest.raises(ValueError, match="encoder <= base <= refiner"):
        PathsConfig(paths_encoder_lr=1e-4, paths_base_lr=1e-5)


def test_paths_implementation_signature_covers_every_declared_file() -> None:
    root = Path(__file__).resolve().parents[1]
    digest = hashlib.sha256()
    for relative in _PATHS_IMPLEMENTATION_FILES:
        path = root / relative
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes() if path.is_file() else b"<missing>")
        digest.update(b"\0")
    assert paths_implementation_signature() == digest.hexdigest()
    assert "models/paths.py" in _PATHS_IMPLEMENTATION_FILES
    assert "losses/paths.py" in _PATHS_IMPLEMENTATION_FILES


def test_v3_warm_start_is_hash_split_architecture_and_state_strict(tmp_path) -> None:
    checkpoint = tmp_path / "best.pth"
    checksum = _source_checkpoint(checkpoint)
    cfg = PathsConfig(
        warm_start_checkpoint=str(checkpoint), warm_start_sha256=checksum
    )
    model = PathsModel(
        **_model_kwargs(),
        paths_probe_z=cfg.pgf_probes,
        paths_correction_cap=cfg.correction_cap,
        paths_gain_init=cfg.correction_gain_init,
        paths_strength=cfg.correction_strength,
    )
    provenance = load_audited_origin_v3_warm_start(
        model, cfg, fold=0, split_signature="split-a"
    )
    assert provenance["source_checkpoint_sha256"] == checksum
    assert provenance["source_metrics"]["acc"] == 86.0

    bad_hash = PathsConfig(
        warm_start_checkpoint=str(checkpoint), warm_start_sha256="0" * 64
    )
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        load_audited_origin_v3_warm_start(
            model, bad_hash, fold=0, split_signature="split-a"
        )
    with pytest.raises(ValueError, match="split signature"):
        load_audited_origin_v3_warm_start(
            model, cfg, fold=0, split_signature="split-b"
        )


class _LabelDataset(Dataset):
    def __init__(self, labels):
        self.targets = list(labels)

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        return (
            torch.zeros(3, 8, 8),
            torch.ones(1, 8, 8, dtype=torch.bool),
            torch.tensor(self.targets[index]),
            torch.tensor(index),
        )


def test_trainer_fixes_risk_weights_from_training_fold_labels_only(tmp_path) -> None:
    checkpoint = tmp_path / "source.pth"
    checksum = _source_checkpoint(checkpoint)
    cfg = PathsConfig(
        warm_start_checkpoint=str(checkpoint),
        warm_start_sha256=checksum,
        risk_set_alpha=0.5,
        amp=False,
    )
    train_labels = [0, 0, 1, 2, 2, 2, 3, 4]
    train = DataLoader(_LabelDataset(train_labels), batch_size=2)
    # Deliberately different validation labels: they must not influence the
    # fixed boundary coefficients.
    validation = DataLoader(_LabelDataset([0, 4, 4, 4, 4]), batch_size=2)
    model = PathsModel(
        **_model_kwargs(),
        paths_probe_z=cfg.pgf_probes,
        paths_correction_cap=cfg.correction_cap,
        paths_gain_init=cfg.correction_gain_init,
        paths_strength=cfg.correction_strength,
    )
    trainer = PathsTrainer(
        model,
        train,
        validation,
        None,
        cfg,
        tmp_path / "run",
        fold=0,
        split_signature="split-a",
        device="cpu",
    )
    assert cfg.warm_start_checkpoint == str(checkpoint)
    assert cfg.warm_start_sha256 == checksum
    assert trainer.cfg is cfg
    assert trainer.warm_start_provenance is None
    assert trainer.paths_warm_start_provenance["source_checkpoint_sha256"] == checksum
    expected_counts = torch.bincount(torch.tensor(train_labels), minlength=5)
    assert trainer.training_label_counts == expected_counts.tolist()
    torch.testing.assert_close(
        trainer.criterion.boundary_weights,
        risk_set_boundary_weights(expected_counts, power=0.5),
    )
    assert [group["name"] for group in trainer.optimizer.param_groups] == [
        "encoder",
        "origin_v3_generator",
        "paths_refiner",
    ]
    assert [group["lr"] for group in trainer.optimizer.param_groups] == [
        cfg.paths_encoder_lr,
        cfg.paths_base_lr,
        cfg.paths_refiner_lr,
    ]
    grouped = [
        parameter
        for group in trainer.optimizer.param_groups
        for parameter in group["params"]
    ]
    assert len({id(parameter) for parameter in grouped}) == len(grouped)
    assert {id(parameter) for parameter in grouped} == {
        id(parameter) for parameter in trainer.model.parameters()
    }

    trainer._set_encoder_trainable(1)
    assert all(parameter.requires_grad for parameter in trainer.model.paths_refiner.parameters())
    assert not any(parameter.requires_grad for parameter in trainer.model.generator.parameters())
    assert not any(parameter.requires_grad for parameter in trainer.model.encoder.parameters())
    trainer._set_encoder_trainable(4)
    assert all(parameter.requires_grad for parameter in trainer.model.parameters())


def _joint_output():
    torch.manual_seed(73)
    metadata = OriginScaleMetadata(
        name="s4",
        feature_index=0,
        channels=3,
        output_stride=4,
        receptive_field=7,
        center_offset=2.0,
        input_size=(12, 16),
        lattice_size=(3, 4),
    )
    encoded = OriginEncoderOutput(
        {
            "s4": OriginEncoderScale(
                features=torch.randn(2, 3, 3, 4),
                valid_mask=torch.ones(2, 3, 4, dtype=torch.bool),
                metadata=metadata,
            )
        }
    )
    generator = ConservedOrdinalGenerator(
        stage_channels={"s4": 3},
        num_classes=5,
        hidden_channels=7,
        evidence_scales=("s4",),
        reference_count=8.0,
        atom_rate_init=0.02,
        prior_rate_init=0.001,
    ).eval()
    base = generator(encoded, force_decoder_fp64=True)
    return PathsContinuationRefiner(
        4, ("s4",), probe_z=(0.1, 0.5, 0.9), correction_cap=3.0
    )(base)


def test_joint_audit_certifies_final_not_base_only_replay() -> None:
    baseline = _joint_output()
    removal = torch.zeros_like(baseline.valid_masks["s4"])
    removal[:, 0, 0] = True
    intervention = replay_paths_without(baseline, {"s4": removal})
    audit = audit_paths_joint_replay(baseline, intervention)
    for name in (
        "rate_replay_error",
        "correction_replay_error",
        "baseline_joint_logit_error",
        "replayed_joint_logit_error",
        "baseline_posterior_replay_error",
        "replayed_posterior_replay_error",
    ):
        assert audit[name] < 1e-10
    assert audit["final_correction_max_abs"] > 0.0
    assert audit["removed_correction_max_abs"] > 0.0


def _synthetic_rank_output(
    *,
    cell_rate: float = 2.0,
    correction: float = 0.0,
) -> SimpleNamespace:
    """Small exact ledger with two tied valid cells and one invalid distractor."""

    boundaries = 4
    rates = torch.tensor(
        [[[[cell_rate] * boundaries, [cell_rate] * boundaries, [99.0] * boundaries]]],
        dtype=torch.float64,
    )
    corrections = torch.tensor(
        [[[[correction] * boundaries, [correction] * boundaries, [99.0] * boundaries]]],
        dtype=torch.float64,
    )
    valid = torch.tensor([[[True, True, False]]])
    valid_rates = rates * valid.unsqueeze(-1)
    valid_corrections = corrections * valid.unsqueeze(-1)
    total_rates = valid_rates.sum(dim=(1, 2)) + 1e-4
    boundary_correction = valid_corrections.sum(dim=(1, 2))
    base = decode_pure_birth_rates(total_rates, force_fp64=True)
    base_logits = continuation_logits_from_log_probs(base.log_class_probs)
    final = decode_continuation_logits(base_logits + boundary_correction)
    return SimpleNamespace(
        local_rate_maps={"s4": valid_rates},
        local_correction_maps={"s4": valid_corrections},
        valid_masks={"s4": valid},
        total_rates=total_rates,
        boundary_correction=boundary_correction,
        base_continuation_logits=base_logits,
        continuation_logits=final.continuation_logits,
        class_probs=final.class_probs,
        log_class_probs=final.log_class_probs,
        cumulative_probs=final.cumulative_probs,
        expected_grade=final.expected_grade,
        posterior_median=final.posterior_median,
        class_map=final.class_map,
    )


@pytest.mark.parametrize(
    "decision_rule", ["class_map", "posterior_median", "rounded_expected"]
)
def test_singleton_ranker_matches_brute_replay_and_ties_are_deterministic(
    decision_rule: str,
) -> None:
    output = _synthetic_rank_output(correction=0.0)
    ranked = rank_paths_singleton_witnesses(
        output, decision_rule=decision_rule, chunk_size=1
    )[0]
    assert ranked["supporter_found"]
    # Both valid cells are identical. The invalid high-valued distractor must
    # never be considered, and the documented tie rule picks flat index zero.
    assert ranked["selected"]["flat_index"] == 0
    assert ranked["selected"]["spatial_index"] == [0, 0]
    assert ranked["valid_cell_count"] == 2
    # This is intentionally a rate-only witness: a correction-map ranking
    # would see an all-zero tie, while exact final-predictor replay identifies
    # that deleting the rate ledger weakens the predicted grade.
    assert ranked["selected"]["configured_decision_margin_drop"] > 0.0
    assert ranked["selected"]["predicted_grade_log_probability_drop"] > 0.0

    target = int(ranked["target_grade"])
    candidate_effects = []
    candidate_margins = []
    baseline_margin = _fixed_grade_decision_margin(
        output.class_probs,
        output.cumulative_probs,
        output.expected_grade,
        target_grade=target,
        decision_rule=decision_rule,
        log_class_probs=output.log_class_probs,
    )[0]
    for flat in (0, 1):
        removed_rate = output.local_rate_maps["s4"].reshape(-1, 4)[flat]
        removed_correction = output.local_correction_maps["s4"].reshape(-1, 4)[flat]
        replay_base = decode_pure_birth_rates(
            output.total_rates - removed_rate.unsqueeze(0), force_fp64=True
        )
        replay_logits = (
            continuation_logits_from_log_probs(replay_base.log_class_probs)
            + output.boundary_correction
            - removed_correction.unsqueeze(0)
        )
        replay = decode_continuation_logits(replay_logits)
        base_log = output.log_class_probs[0, target]
        log_drop = float(base_log - replay.log_class_probs[0, target])
        candidate_effects.append(log_drop)
        replay_margin = _fixed_grade_decision_margin(
            replay.class_probs,
            replay.cumulative_probs,
            replay.expected_grade,
            target_grade=target,
            decision_rule=decision_rule,
            log_class_probs=replay.log_class_probs,
        )[0]
        candidate_margins.append(float(baseline_margin - replay_margin))
    assert ranked["selected"]["predicted_grade_log_probability_drop"] == pytest.approx(
        max(candidate_effects), abs=1e-12
    )
    assert ranked["selected"]["configured_decision_margin_drop"] == pytest.approx(
        max(candidate_margins), abs=1e-12
    )


def test_singleton_ranker_reports_when_no_positive_supporter_exists() -> None:
    output = _synthetic_rank_output(cell_rate=0.005, correction=0.0)
    ranked = rank_paths_singleton_witnesses(
        output, decision_rule="class_map", chunk_size=2
    )[0]
    assert ranked["target_grade"] == 0
    assert ranked["supporter_found"] is False
    assert ranked["selected"] is None
    assert ranked["strongest_candidate_regardless_of_sign"] is not None


@pytest.mark.parametrize("grade", [0, 1, 2, 3, 4])
def test_target_logprob_boundary_contributions_are_exact(grade: int) -> None:
    baseline = torch.tensor([1.2, 0.7, -0.4, 0.2], dtype=torch.float64)
    replayed = torch.tensor([1.0, 0.9, -0.8, -0.1], dtype=torch.float64)
    base = decode_continuation_logits(baseline.unsqueeze(0))
    new = decode_continuation_logits(replayed.unsqueeze(0))
    contributions = target_log_probability_boundary_contributions(
        baseline, replayed, target_grade=grade
    )
    exact = base.log_class_probs[0, grade] - new.log_class_probs[0, grade]
    torch.testing.assert_close(contributions.sum(), exact, atol=1e-14, rtol=0.0)


def test_certificate_v2_serializes_exact_final_predictor_and_geometry(tmp_path) -> None:
    baseline = _joint_output()

    class _ReplayModel(nn.Module):
        def replay_without(self, output, removal_masks, *, force_decoder_fp64=True):
            return replay_paths_without(
                output, removal_masks, force_decoder_fp64=force_decoder_fp64
            )

    trainer = PathsTrainer.__new__(PathsTrainer)
    trainer.model = _ReplayModel()
    trainer.val_loader = [
        (
            torch.zeros(2, 3, 12, 16),
            torch.ones(2, 1, 12, 16, dtype=torch.bool),
            torch.tensor([1, 1]),
            torch.tensor([17, 19]),
        )
    ]
    trainer.cfg = SimpleNamespace(
        certificate_samples=2, certificate_replay_tolerance=2e-5
    )
    trainer.use_amp = False
    trainer.decision_rule = "rounded_expected"
    trainer.num_classes = 5
    trainer.fold = 0
    trainer.split_signature = "split-a"
    trainer.implementation_signature = "a" * 64
    trainer.architecture_signature = "b" * 64
    trainer.paths_warm_start_provenance = {
        "source_checkpoint_sha256": "c" * 64
    }
    trainer.certificate_path = tmp_path / "certificates.json"
    trainer.best_path = tmp_path / "best.pth"
    trainer.best_path.write_bytes(b"checkpoint")
    trainer._unpack_batch = lambda batch: batch
    trainer._forward = lambda images, pixel_mask: baseline

    payload = trainer._write_validation_certificates(checkpoint_epoch=7)
    assert payload["schema"] == "paths-exact-joint-validation-certificates-v2"
    assert payload["certifies_final_paths_prediction"] is True
    assert payload["singleton_ranking_canonical_replay_max_abs_error"] < 2e-5
    assert payload["faithfulness_audit"]["paired_sample_count"] == 2
    certificate = payload["certificates"][0]
    selected = certificate["selected_positive_supporter"]
    assert selected is not None
    assert selected["clipped_receptive_field_box_yxyx"] is not None
    assert isinstance(selected["touches_padding"], bool)
    assert isinstance(selected["covers_entire_input"], bool)
    assert isinstance(selected["global_support"], bool)
    assert selected["target_log_probability_contribution_error"] < 2e-5
    assert certificate["deterministic_scale_matched_random"] is not None
    assert trainer.certificate_path.is_file()
