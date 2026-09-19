"""Forward-ledger and intervention invariants for PATHS-OSWT."""

from __future__ import annotations

from dataclasses import replace
import hashlib
from pathlib import Path

import pytest
import torch

from configs.paths_oswt_config import PathsOSWTConfig
from models.origin import ConservedOrdinalGenerator, OriginOutput
from models.origin_encoder import (
    OriginEncoderOutput,
    OriginEncoderScale,
    OriginScaleMetadata,
)
from models.paths_oswt import (
    PathsOSWTOutput,
    PathsOSWTRefiner,
    replay_paths_oswt_without,
)
from training.paths_oswt_trainer import (
    _OSWT_IMPLEMENTATION_FILES,
    audit_paths_oswt_output,
    paths_oswt_implementation_signature,
)
from train_paths_oswt import require_one_sha_bound_fold


def test_oswt_config_registers_exact_treatment_controls_and_rejects_drift() -> None:
    assert PathsOSWTConfig().oswt_variant == "shell_warranted"
    assert PathsOSWTConfig(oswt_variant="shell_warranted_ungated").oswt_strength == 1.0
    assert PathsOSWTConfig(oswt_variant="identity_v3", oswt_strength=0.0).oswt_strength == 0.0
    with pytest.raises(ValueError, match="identity_v3"):
        PathsOSWTConfig(oswt_variant="identity_v3", oswt_strength=1.0)
    with pytest.raises(ValueError, match="four-scale"):
        PathsOSWTConfig(evidence_scales=("s8", "s16"))
    with pytest.raises(ValueError, match="outcome weighting"):
        PathsOSWTConfig(
            class_weighting="inverse_frequency", allow_weighted_likelihood=True
        )
    with pytest.raises(ValueError, match="weight_decay=0"):
        PathsOSWTConfig(weight_decay=1e-5)
    with pytest.raises(ValueError, match="tau parameters"):
        PathsOSWTConfig(oswt_tau_floor=0.1)


def test_oswt_cli_requires_one_fold_specific_hash_bound_source() -> None:
    assert require_one_sha_bound_fold([0]) == 0
    with pytest.raises(ValueError, match="exactly one fold"):
        require_one_sha_bound_fold([0, 1])


def test_oswt_implementation_signature_covers_registered_launch_graph() -> None:
    root = Path(__file__).resolve().parents[1]
    digest = hashlib.sha256()
    for relative in _OSWT_IMPLEMENTATION_FILES:
        path = root / relative
        digest.update(relative.encode())
        digest.update(b"\0")
        digest.update(path.read_bytes() if path.is_file() else b"<missing>")
        digest.update(b"\0")
    assert paths_oswt_implementation_signature() == digest.hexdigest()
    required = {
        "models/paths_oswt.py",
        "models/paths.py",
        "configs/paths_config.py",
        "training/paths_oswt_trainer.py",
        "train_paths_oswt.py",
        "train_origin.py",
        "scripts/submit_paths_oswt_preflight.sh",
        "scripts/submit_paths_oswt_promotion_gate.sh",
        "scripts/launch_paths_oswt_v8_aptos_f0.sh",
    }
    assert required.issubset(set(_OSWT_IMPLEMENTATION_FILES))


def _metadata(name: str, channels: int, height: int, width: int) -> OriginScaleMetadata:
    stride = {"s4": 4, "s8": 8}[name]
    return OriginScaleMetadata(
        name=name,
        feature_index=0,
        channels=channels,
        output_stride=stride,
        receptive_field=7,
        center_offset=2.0,
        input_size=(height * stride, width * stride),
        lattice_size=(height, width),
    )


def _encoded(batch: int = 2) -> OriginEncoderOutput:
    torch.manual_seed(802)
    scales = {}
    for name, channels, shape in (("s4", 3, (3, 4)), ("s8", 5, (2, 2))):
        height, width = shape
        valid = torch.ones(batch, height, width, dtype=torch.bool)
        valid[0, -1, -1] = False
        scales[name] = OriginEncoderScale(
            features=torch.randn(batch, channels, height, width),
            valid_mask=valid,
            metadata=_metadata(name, channels, height, width),
        )
    return OriginEncoderOutput(scales)


def _base_output() -> OriginOutput:
    generator = ConservedOrdinalGenerator(
        stage_channels={"s4": 3, "s8": 5},
        num_classes=5,
        hidden_channels=7,
        evidence_scales=("s4", "s8"),
        reference_count=8.0,
        atom_rate_init=0.02,
        prior_rate_init=0.001,
    ).eval()
    return generator(_encoded(), force_decoder_fp64=True)


def _output() -> PathsOSWTOutput:
    refiner = PathsOSWTRefiner(
        5,
        ("s4", "s8"),
        probe_z=(0.1, 0.3, 0.6, 0.9),
        beta_init=1.2,
        tau_init=0.05,
    )
    result = refiner(_base_output())
    assert isinstance(result, PathsOSWTOutput)
    return result


def _empty_masks(output: PathsOSWTOutput) -> dict[str, torch.Tensor]:
    return {
        name: torch.zeros_like(evidence.active_mask)
        for name, evidence in output.shell_evidence.items()
    }


def test_forward_ledger_is_local_additive_normalized_and_has_no_bypass() -> None:
    output = _output()
    local_left_sum = sum(
        item.local_left_boundary_warrant_map.sum(dim=(-2, -1))
        for item in output.shell_evidence.values()
    )
    local_right_sum = sum(
        item.local_right_boundary_warrant_map.sum(dim=(-2, -1))
        for item in output.shell_evidence.values()
    )
    torch.testing.assert_close(local_left_sum, output.left_boundary_warrant)
    torch.testing.assert_close(local_right_sum, output.right_boundary_warrant)
    assert torch.all((output.left_boundary_warrant >= 0.0) & (output.left_boundary_warrant <= 1.0))
    assert torch.all((output.right_boundary_warrant >= 0.0) & (output.right_boundary_warrant <= 1.0))
    torch.testing.assert_close(
        output.class_probs.sum(dim=-1),
        torch.ones(output.class_probs.shape[0], dtype=torch.float64),
        atol=2e-15,
        rtol=0.0,
    )
    torch.testing.assert_close(
        output.refined_adjacent_log_odds,
        output.base_adjacent_log_odds + output.log_odds_increment,
    )
    for evidence in output.shell_evidence.values():
        invalid = ~evidence.original_valid_mask
        spatial_shells = evidence.ordinal_shells.permute(0, 2, 3, 1)
        assert torch.equal(spatial_shells[invalid], torch.zeros_like(spatial_shells[invalid]))
        expected_partition = evidence.original_valid_mask[:, None].to(
            torch.float64
        ).expand_as(evidence.local_left_partition_mass_map)
        torch.testing.assert_close(
            evidence.local_left_partition_mass_map
            + evidence.local_right_partition_mass_map,
            expected_partition,
        )
        for local_map in (
            evidence.local_left_boundary_warrant_map,
            evidence.local_right_boundary_warrant_map,
        ):
            spatial = local_map.permute(0, 2, 3, 1)
            assert torch.equal(spatial[invalid], torch.zeros_like(spatial[invalid]))
        for local_warrant, local_phi, baseline_mass in (
            (
                evidence.local_left_warrant_spectrum,
                evidence.local_left_phi_spectrum,
                evidence.baseline_left_mass,
            ),
            (
                evidence.local_right_warrant_spectrum,
                evidence.local_right_phi_spectrum,
                evidence.baseline_right_mass,
            ),
        ):
            torch.testing.assert_close(
                local_warrant,
                local_phi * baseline_mass[:, :, None, None, None],
            )
    torch.testing.assert_close(
        output.boundary_scale_weights.sum(dim=0),
        torch.ones(output.boundary_scale_weights.shape[1], dtype=torch.float64),
    )


def test_noop_single_union_and_all_evidence_deletions_replay_stored_ledgers() -> None:
    output = _output()

    no_op = replay_paths_oswt_without(output, _empty_masks(output))
    torch.testing.assert_close(no_op.output.class_probs, output.class_probs)
    for removed, baseline in (
        (no_op.removed_left_boundary_warrant, output.left_boundary_warrant),
        (no_op.removed_right_boundary_warrant, output.right_boundary_warrant),
    ):
        torch.testing.assert_close(
            removed, torch.zeros_like(baseline), atol=0.0, rtol=0.0
        )

    first = _empty_masks(output)
    second = _empty_masks(output)
    first["s4"][:, 0, 0] = True
    second["s4"][:, 0, 1] = True
    union = {name: first[name] | second[name] for name in first}
    removed_first = replay_paths_oswt_without(output, first)
    removed_second = replay_paths_oswt_without(output, second)
    removed_union = replay_paths_oswt_without(output, union)

    for side in ("left", "right"):
        removed_name = f"removed_{side}_boundary_warrant"
        output_name = f"{side}_boundary_warrant"
        union_removed = getattr(removed_union, removed_name)
        torch.testing.assert_close(
            union_removed,
            getattr(removed_first, removed_name) + getattr(removed_second, removed_name),
        )
        torch.testing.assert_close(
            getattr(removed_union.output, output_name),
            getattr(output, output_name) - union_removed,
        )
    assert torch.isfinite(removed_union.output.class_probs).all()
    torch.testing.assert_close(
        removed_union.output.class_probs.sum(dim=-1),
        torch.ones(output.class_probs.shape[0], dtype=torch.float64),
    )

    all_cells = {
        name: evidence.active_mask.clone()
        for name, evidence in output.shell_evidence.items()
    }
    removed_all = replay_paths_oswt_without(output, all_cells)
    for side in ("left", "right"):
        removed_name = f"removed_{side}_boundary_warrant"
        output_name = f"{side}_boundary_warrant"
        torch.testing.assert_close(
            getattr(removed_all, removed_name), getattr(output, output_name)
        )
        assert torch.equal(
            getattr(removed_all.output, output_name),
            torch.zeros_like(getattr(removed_all.output, output_name)),
        )
    # Crucial intervention invariant: deleting a stored cell cannot turn its
    # missing cumulative atom into synthetic v0 (grade-0) evidence.
    for evidence in removed_all.output.shell_evidence.values():
        assert torch.equal(evidence.ordinal_shells, torch.zeros_like(evidence.ordinal_shells))
        for local_map in (
            evidence.local_left_boundary_warrant_map,
            evidence.local_right_boundary_warrant_map,
        ):
            assert torch.equal(local_map, torch.zeros_like(local_map))
    torch.testing.assert_close(
        removed_all.output.class_probs,
        removed_all.base_intervention.output.class_probs,
    )


def test_intervention_reports_exact_signed_flow_and_wasserstein_replay() -> None:
    output = _output()
    masks = _empty_masks(output)
    masks["s4"][:, 0, 0] = True
    intervention = replay_paths_oswt_without(output, masks)

    torch.testing.assert_close(
        intervention.removed_signed_boundary_flow,
        output.signed_boundary_flow - intervention.output.signed_boundary_flow,
    )
    for candidate in (output, intervention.output):
        expected_flow = torch.cumsum(
            candidate.base_output.class_probs - candidate.class_probs, dim=-1
        )[..., :-1]
        torch.testing.assert_close(candidate.signed_boundary_flow, expected_flow)
        torch.testing.assert_close(
            candidate.wasserstein1,
            expected_flow.abs().sum(dim=-1),
        )

    audit = audit_paths_oswt_output(output, intervention)
    failures = {
        key: value
        for key, value in audit.items()
        if key.endswith("_error") and value > 2e-12
    }
    assert failures == {}


def test_complete_shell_warrant_path_has_finite_nonzero_refiner_gradients() -> None:
    """The trainable parameters must receive signal through the full ledger."""

    refiner = PathsOSWTRefiner(
        5,
        ("s4", "s8"),
        probe_z=(0.1, 0.3, 0.6, 0.9),
        beta_init=0.7,
        tau_init=0.08,
    )
    output = refiner(_base_output())
    assert isinstance(output, PathsOSWTOutput)
    targets = torch.tensor([1, 4], dtype=torch.long)
    loss = -output.log_class_probs.gather(1, targets[:, None]).mean()
    loss.backward()

    for name in ("probe_logits", "raw_beta", "raw_tau"):
        parameter = getattr(refiner, name)
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name
        assert float(parameter.grad.abs().sum()) > 0.0, name


def test_structural_audit_binds_shells_and_baseline_mass_to_source_atoms() -> None:
    """Simplex/complement checks alone must not certify a forged ledger."""

    output = _output()
    name = next(iter(output.shell_evidence))
    evidence = output.shell_evidence[name]
    forged = replace(
        evidence,
        # Rolling preserves non-negativity and unit mass while breaking the
        # exact telescoping transform from the stored cumulative atoms.
        ordinal_shells=torch.roll(evidence.ordinal_shells, shifts=1, dims=1),
        baseline_left_mass=evidence.baseline_left_mass + 0.125,
    )
    corrupted = replace(
        output,
        shell_evidence={**output.shell_evidence, name: forged},
    )
    audit = audit_paths_oswt_output(corrupted)
    assert audit["ordinal_shell_simplex_error"] < 2e-12
    assert audit["ordinal_shell_reconstruction_error"] > 1e-6
    assert audit["left_baseline_mass_reconstruction_error"] > 1e-6
