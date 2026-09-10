from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from audit_origin_validation import audit_origin_validation
from audit_origin_validation import _verify_provenance
from configs.origin_config import OriginConfig
from Datasets.origin_data import class_histogram
from models.origin import decode_pure_birth_rates
from train_origin import split_signature
from training.origin_trainer import (
    _architecture_record,
    _canonical_sha256,
    _critical_config,
    origin_implementation_signature,
)


class _AuditModel(nn.Module):
    """Tiny replayable ledger; no classifier bypass is used by the audit."""

    def __init__(self) -> None:
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))

    @staticmethod
    def _output(local_rate_maps, valid_masks):
        # Match production: conserve the channels-first source ledger in FP64,
        # while the audit is free to aggregate its exposed NHWK view for
        # descriptive scale summaries in FP32.
        total = sum(
            value.permute(0, 3, 1, 2).double().sum(dim=(-2, -1))
            for value in local_rate_maps.values()
        )
        decoded = decode_pure_birth_rates(total.double(), force_fp64=True)
        metadata = {
            "s4": SimpleNamespace(
                output_stride=4, receptive_field=9, center_offset=2.0,
                input_size=(64, 64),
            ),
            "s32": SimpleNamespace(
                output_stride=32, receptive_field=128, center_offset=16.0,
                input_size=(64, 64),
            ),
        }
        prior_rates = torch.zeros_like(total)
        return SimpleNamespace(
            local_rate_maps=local_rate_maps,
            valid_masks=valid_masks,
            total_rates=decoded.total_rates,
            prior_rates=prior_rates,
            class_probs=decoded.class_probs,
            log_class_probs=decoded.log_class_probs,
            cumulative_probs=decoded.cumulative_probs,
            expected_grade=decoded.expected_grade,
            posterior_median=decoded.posterior_median,
            class_map=decoded.class_map,
            metadata=metadata,
        )

    def forward(self, images, pixel_valid_mask=None, force_decoder_fp64=True):
        del pixel_valid_mask, force_decoder_fp64
        identity = images[:, 0, 0, 0].float()
        batch = len(images)
        # s4 has two focal entries; s32 is deliberately strongest for some,
        # but not all, samples so scale histograms are non-degenerate.
        s4 = torch.zeros(batch, 1, 2, 2, device=images.device)
        s32 = torch.zeros(batch, 1, 1, 2, device=images.device)
        s4[:, 0, 0, 0] = 0.15 + identity * 0.10
        s4[:, 0, 1, 1] = 0.05 + identity * 0.05
        s32[:, 0, 0, 0] = 0.35 - identity * 0.04
        s32[:, 0, 0, 1] = 0.08 + identity * 0.12
        valid = {
            "s4": torch.ones(batch, 1, 2, dtype=torch.bool, device=images.device),
            "s32": torch.ones(batch, 1, 1, dtype=torch.bool, device=images.device),
        }
        return self._output({"s4": s4, "s32": s32}, valid)

    def replay_without(self, baseline, removals, force_decoder_fp64=True):
        del force_decoder_fp64
        kept = {}
        removed = torch.zeros_like(baseline.total_rates)
        canonical = {}
        for scale, rates in baseline.local_rate_maps.items():
            mask = removals.get(scale, torch.zeros_like(baseline.valid_masks[scale])).bool()
            canonical[scale] = mask
            removed += (
                (rates * mask.unsqueeze(-1))
                .permute(0, 3, 1, 2)
                .double()
                .sum(dim=(-2, -1))
            )
            kept[scale] = rates.masked_fill(mask.unsqueeze(-1), 0.0)
        output = self._output(kept, baseline.valid_masks)
        return SimpleNamespace(
            baseline=baseline,
            output=output,
            removed_rates=removed,
            removal_masks=canonical,
        )


def test_full_validation_audit_is_stratified_and_exact() -> None:
    images = torch.arange(6, dtype=torch.float32).reshape(6, 1, 1, 1)
    masks = torch.ones(6, 1, 1, 1, dtype=torch.bool)
    labels = torch.tensor([0, 0, 1, 1, 2, 2])
    indices = torch.arange(6)
    loader = DataLoader(TensorDataset(images, masks, labels, indices), batch_size=2)

    model = _AuditModel()
    state_before = {name: value.detach().clone() for name, value in model.state_dict().items()}
    audit = audit_origin_validation(
        model,
        loader,
        validation_items=[(f"image_{index}.png", int(labels[index])) for index in range(6)],
        top_ks=(1, 2),
        certificates_per_grade=1,
        split_signature="unit-test-split",
        device="cpu",
        amp=False,
    )

    assert audit["schema"] == "origin-full-validation-audit-v2"
    assert audit["scope"] == "inner_validation_only"
    assert audit["n"] == 6
    assert audit["validation_sample_ids"] == list(range(6))
    assert set(audit["grade_stratified_metrics"]) == {"0", "1", "2"}
    assert [item["label"] for item in audit["grade_stratified_certificates"]] == [0, 1, 2]
    # The synthetic ledger is FP32, so independently reduced survivors can
    # differ from baseline-minus-removed by a few ledger ulps.
    assert audit["max_exact_total_rate_replay_error"] < 1e-6

    for boundary in ("0", "1"):
        total_winners = audit["winning_scale_histograms"][
            "largest_raw_rate_single_cell_by_boundary"
        ][boundary]
        assert sum(total_winners.values()) == 6
        assert set(audit["topk_deletion_effects"][boundary]) == {"1", "2"}
        for k in ("1", "2"):
            assert sum(
                audit["topk_deletion_effects"][boundary][k][
                    "selected_scale_histogram"
                ].values()
            ) == 6 * int(k)

    shares = audit["per_scale_raw_rate_contributions"]
    for boundary in range(2):
        assert abs(sum(item["share_by_boundary"][boundary] for item in shares.values()) - 1.0) < 1e-9
    assert sum(item["overall_share"] for item in shares.values()) == pytest.approx(1.0)
    grade0 = audit["grade0_boundary0_evidence_diagnostics"]
    assert grade0["true_grade0"]["support"] == 2
    assert grade0["max_prior_only_identity_error"] < 1e-12
    assert grade0["max_log_space_local_factorization_identity_error"] < 1e-12
    assert set(audit["grade_and_scale_stratified_certificates"]) == {"s4", "s32"}
    assert any(
        item["global_input_support"]
        for item in audit["grade_stratified_certificates"]
        if item["removed_scale"] == "s32"
    )
    for name, value in model.state_dict().items():
        assert torch.equal(value, state_before[name])


class _DenseReductionAuditModel(_AuditModel):
    """Production-shaped ledger exposing the FP32 reduction-order trap."""

    def forward(self, images, pixel_valid_mask=None, force_decoder_fp64=True):
        del pixel_valid_mask, force_decoder_fp64
        batch = len(images)
        # Construct NCHW first, as the real pointwise atom head does. The
        # public audit interface exposes a zero-copy NHWK permutation.
        channels_first = torch.full(
            (batch, 4, 160, 160),
            1e-5,
            dtype=torch.float32,
            device=images.device,
        )
        exposed = channels_first.permute(0, 2, 3, 1)
        valid = torch.ones(
            batch, 160, 160, dtype=torch.bool, device=images.device
        )
        return self._output({"s4": exposed}, {"s4": valid})


def test_grade0_factorization_uses_the_conserved_fp64_ledger() -> None:
    loader = DataLoader(
        TensorDataset(
            torch.zeros(1, 1, 1, 1),
            torch.ones(1, 1, 1, 1, dtype=torch.bool),
            torch.zeros(1, dtype=torch.long),
            torch.zeros(1, dtype=torch.long),
        )
    )
    audit = audit_origin_validation(
        _DenseReductionAuditModel(),
        loader,
        validation_items=[("dense.png", 0)],
        top_ks=(1,),
        certificates_per_grade=1,
        split_signature="dense-reduction-regression",
        device="cpu",
        amp=False,
    )
    diagnostics = audit["grade0_boundary0_evidence_diagnostics"]
    assert diagnostics["max_log_space_local_factorization_identity_error"] < 2e-10
    assert diagnostics["max_fp64_ledger_reconstruction_error"] < 2e-10
    # A direct FP32 reduction of 25,600 cells is measurably different. This
    # diagnostic proves the regression exercises the exact cluster failure.
    assert diagnostics["max_alternative_fp32_reduction_difference"] > 1e-9


def test_audit_rejects_invalid_topk() -> None:
    loader = DataLoader(
        TensorDataset(
            torch.zeros(1, 1, 1, 1),
            torch.ones(1, 1, 1, 1, dtype=torch.bool),
            torch.zeros(1, dtype=torch.long),
            torch.zeros(1, dtype=torch.long),
        )
    )
    with pytest.raises(ValueError, match="top_ks"):
        audit_origin_validation(_AuditModel(), loader, top_ks=(0,), amp=False)


class _CorruptReplayModel(_AuditModel):
    def replay_without(self, baseline, removals, force_decoder_fp64=True):
        result = super().replay_without(
            baseline, removals, force_decoder_fp64=force_decoder_fp64
        )
        result.output.total_rates = result.output.total_rates + 0.1
        return result


def test_audit_rejects_corrupt_exact_replay() -> None:
    loader = DataLoader(
        TensorDataset(
            torch.zeros(1, 1, 1, 1),
            torch.ones(1, 1, 1, 1, dtype=torch.bool),
            torch.zeros(1, dtype=torch.long),
            torch.zeros(1, dtype=torch.long),
        )
    )
    with pytest.raises(AssertionError, match="exact total-rate replay"):
        audit_origin_validation(_CorruptReplayModel(), loader, top_ks=(1,), amp=False)


def test_audit_rejects_duplicate_validation_coverage() -> None:
    loader = DataLoader(
        TensorDataset(
            torch.zeros(2, 1, 1, 1),
            torch.ones(2, 1, 1, 1, dtype=torch.bool),
            torch.tensor([0, 1]),
            torch.tensor([0, 0]),
        )
    )
    with pytest.raises(AssertionError, match="sample IDs"):
        audit_origin_validation(
            _AuditModel(), loader,
            validation_items=[("a.png", 0), ("b.png", 1)],
            top_ks=(1,), amp=False,
        )


def test_provenance_helper_recomputes_all_signatures(tmp_path) -> None:
    paths = []
    for index in range(3):
        path = tmp_path / f"sample_{index}.png"
        path.write_bytes(f"content-{index}".encode())
        paths.append(str(path))
    train = [(paths[0], 0)]
    validation = [(paths[1], 1)]
    locked = [(paths[2], 2)]
    cfg = OriginConfig()
    model = _AuditModel()
    architecture = _architecture_record(model)
    critical = _critical_config(cfg)
    signature = split_signature(
        ("train", train), ("validation", validation), ("locked_test", locked)
    )
    state = {
        "schema": "origin-checkpoint-v3",
        "fold": 0,
        "implementation_signature": origin_implementation_signature(),
        "architecture": architecture,
        "architecture_signature": _canonical_sha256(architecture),
        "critical_config": critical,
        "config_signature": _canonical_sha256(critical),
        "split_signature": signature,
    }
    manifest = {
        "schema": "origin-split-v2",
        "evaluation_scope": "inner_validation_only",
        "dataset": cfg.dataset,
        "fold": 0,
        "signature": signature,
        "counts": {"train": 1, "validation": 1, "locked_test": 1},
        "histograms": {
            "train": class_histogram(train, cfg.n_classes),
            "validation": class_histogram(validation, cfg.n_classes),
            "locked_test": class_histogram(locked, cfg.n_classes),
        },
    }
    assert _verify_provenance(
        state, cfg, model, manifest, train, validation, locked
    ) == signature
    corrupt = dict(state)
    corrupt["config_signature"] = "corrupt"
    with pytest.raises(ValueError, match="configuration signature"):
        _verify_provenance(
            corrupt, cfg, model, manifest, train, validation, locked
        )
