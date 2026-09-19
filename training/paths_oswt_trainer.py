"""Frozen-base training and exact certificates for PATHS-OSWT.

The registered OSWT experiment is deliberately narrower than historical
PATHS training: an audited ORIGIN-v3 checkpoint is loaded by content hash,
the complete encoder/generator is frozen (including stochastic behaviour),
and only ``paths_refiner.*`` is optimized with the proper risk-set score plus
RPS.  This prevents a co-adapting base from hiding whether shell-warranted
transport itself contributes.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.amp import autocast

from configs.paths_oswt_config import (
    PATHS_OSWT_CHECKPOINT_SCHEMA,
    PATHS_OSWT_PROTOCOL_VERSION,
)
from models.paths import tangent_removed_pgf_probe
from models.paths_oswt import (
    ordinal_shells_from_cumulative_atoms,
    replay_paths_oswt_without,
)
from training.origin_trainer import (
    OriginTrainer,
    _architecture_record,
    _canonical_sha256,
    _critical_config,
    _tensor_field,
    evaluate_origin_predictions,
    validation_selection_key,
)
from training.paths_trainer import PathsTrainer


_OSWT_IMPLEMENTATION_FILES = (
    "configs/paths_oswt_config.py",
    "configs/paths_config.py",
    "configs/origin_config.py",
    "Datasets/origin_data.py",
    "Datasets/mosaic_data.py",
    "Datasets/dataloaders.py",
    "models/origin_encoder.py",
    "models/origin.py",
    "models/paths.py",
    "models/paths_oswt.py",
    "losses/origin.py",
    "losses/paths.py",
    "training/origin_trainer.py",
    "training/paths_trainer.py",
    "training/paths_oswt_trainer.py",
    "train_paths_oswt.py",
    "train_origin.py",
    "scripts/submit_paths_oswt_preflight.sh",
    "scripts/submit_paths_oswt_aptos_f0.sh",
    "scripts/submit_paths_oswt_aptos_ungated_f0.sh",
    "scripts/submit_paths_oswt_promotion_gate.sh",
    "scripts/submit_paths_oswt_dr_f0.sh",
    "scripts/launch_paths_oswt_v8_aptos_f0.sh",
    "utils/spatial_mask.py",
)

_OSWT_CERTIFICATE_CANDIDATE_LIMIT = 8


def _field(output: object, name: str) -> Any:
    if isinstance(output, Mapping):
        if name not in output:
            raise KeyError(f"OSWT output has no {name!r} field")
        return output[name]
    if not hasattr(output, name):
        raise AttributeError(f"OSWT output has no {name!r} attribute")
    return getattr(output, name)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _v3_base_state_sha256(state: Mapping[str, Any]) -> str:
    """Hash sorted non-OSWT state tensors by identity and exact CPU bytes."""

    digest = hashlib.sha256()
    keys = sorted(
        key for key in state if not str(key).startswith("paths_refiner.")
    )
    if not keys:
        raise ValueError("model state has no V3 base tensors")
    for key in keys:
        tensor = state[key]
        if not torch.is_tensor(tensor):
            raise TypeError(f"V3 base state entry {key!r} is not a tensor")
        contiguous = tensor.detach().cpu().contiguous().reshape(-1)
        raw = contiguous.view(torch.uint8).numpy().tobytes()
        digest.update(str(key).encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(b"\0")
        digest.update(
            json.dumps(list(tensor.shape), separators=(",", ":")).encode("ascii")
        )
        digest.update(b"\0")
        digest.update(raw)
        digest.update(b"\0")
    return digest.hexdigest()


def _normalized_warm_start_provenance(value: Any) -> Any:
    """Compare warm starts by content, not by the invocation path spelling."""

    if not isinstance(value, Mapping):
        return value
    normalized = dict(value)
    normalized.pop("source_checkpoint", None)
    return normalized


def paths_oswt_implementation_signature() -> str:
    """Hash every file that can change OSWT data, prediction, or training."""

    root = Path(__file__).resolve().parents[1]
    digest = hashlib.sha256()
    for relative in _OSWT_IMPLEMENTATION_FILES:
        path = root / relative
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes() if path.is_file() else b"<missing>")
        digest.update(b"\0")
    return digest.hexdigest()


def _max_abs(value: torch.Tensor) -> float:
    return float(value.detach().abs().max().cpu()) if value.numel() else 0.0


def _flow_reconstruction(
    source: torch.Tensor,
    flow: torch.Tensor,
) -> torch.Tensor:
    """Reconstruct a grade posterior from its signed adjacent boundary flow."""

    delta = torch.cat(
        (
            -flow[..., :1],
            flow[..., :-1] - flow[..., 1:],
            flow[..., -1:],
        ),
        dim=-1,
    )
    return source.to(flow.dtype) + delta


def audit_paths_oswt_output(
    output: object,
    intervention: object | None = None,
) -> dict[str, float]:
    """Audit the complementary cut ledger and its exact posterior replay."""

    base = _field(output, "base_output")
    base_probs = _tensor_field(base, "class_probs").to(torch.float64)
    final_probs = _tensor_field(output, "class_probs").to(torch.float64)
    base_odds = _tensor_field(output, "base_adjacent_log_odds").to(torch.float64)
    refined_odds = _tensor_field(output, "refined_adjacent_log_odds").to(torch.float64)
    increment = _tensor_field(output, "log_odds_increment").to(torch.float64)
    correction = _tensor_field(output, "grade_log_correction").to(torch.float64)
    flow = _tensor_field(output, "signed_boundary_flow").to(torch.float64)
    wasserstein = _tensor_field(output, "wasserstein1").to(torch.float64)
    left_warrant = _tensor_field(output, "left_boundary_warrant").to(torch.float64)
    right_warrant = _tensor_field(output, "right_boundary_warrant").to(torch.float64)
    beta = _tensor_field(output, "beta").to(torch.float64)
    tau = _tensor_field(output, "tau").to(torch.float64)

    total_warrant = left_warrant + right_warrant
    expected_coverage = total_warrant / (total_warrant + tau)
    expected_contrast = (right_warrant - left_warrant) / (total_warrant + tau)
    # q_k is the mass to the right of ordinal cut k, not the mass in the two
    # classes adjacent to that cut.
    expected_boundary_probability = torch.flip(
        torch.cumsum(torch.flip(base_probs[..., 1:], dims=(-1,)), dim=-1),
        dims=(-1,),
    )
    uncertainty_gate = (
        4.0
        * expected_boundary_probability
        * (1.0 - expected_boundary_probability)
    )
    expected_gate = (
        uncertainty_gate
        if bool(_field(output, "risk_gate_enabled"))
        else torch.ones_like(uncertainty_gate)
    )
    expected_increment = (
        float(_field(output, "strength"))
        * beta
        * expected_gate
        * expected_contrast
    )
    expected_correction = torch.cat(
        (
            torch.zeros_like(expected_increment[..., :1]),
            torch.cumsum(expected_increment, dim=-1),
        ),
        dim=-1,
    )

    normalized_from_correction = torch.softmax(
        _tensor_field(base, "log_class_probs").to(torch.float64) + correction,
        dim=-1,
    )
    flow_reconstruction = _flow_reconstruction(base_probs, flow)
    audit: dict[str, float] = {
        "adjacent_log_odds_replay_error": _max_abs(
            refined_odds - base_odds - increment
        ),
        "boundary_coverage_formula_error": _max_abs(
            _tensor_field(output, "boundary_coverage").to(torch.float64)
            - expected_coverage
        ),
        "shell_contrast_formula_error": _max_abs(
            _tensor_field(output, "shell_contrast").to(torch.float64)
            - expected_contrast
        ),
        "base_boundary_probability_formula_error": _max_abs(
            _tensor_field(output, "base_boundary_probability").to(torch.float64)
            - expected_boundary_probability
        ),
        "boundary_gate_formula_error": _max_abs(
            _tensor_field(output, "boundary_gate").to(torch.float64)
            - expected_gate
        ),
        "log_odds_increment_formula_error": _max_abs(
            increment - expected_increment
        ),
        "grade_log_correction_reconstruction_error": _max_abs(
            correction - expected_correction
        ),
        "boundary_correction_reconstruction_error": _max_abs(
            _tensor_field(output, "boundary_correction").to(torch.float64)
            - (
                _tensor_field(output, "continuation_logits").to(torch.float64)
                - _tensor_field(output, "base_continuation_logits").to(torch.float64)
            )
        ),
        "posterior_normalization_error": _max_abs(final_probs.sum(dim=-1) - 1.0),
        "posterior_from_grade_correction_error": _max_abs(
            final_probs - normalized_from_correction
        ),
        "signed_flow_reconstruction_error": _max_abs(
            final_probs - flow_reconstruction
        ),
        "wasserstein1_reconstruction_error": _max_abs(
            wasserstein - flow.abs().sum(dim=-1)
        ),
        "posterior_minimum": float(final_probs.min().detach().cpu()),
        "transport_effect_max_abs": _max_abs(final_probs - base_probs),
        "beta_minimum": float(beta.min().detach().cpu()),
        "beta_maximum": float(beta.max().detach().cpu()),
        "tau_minimum": float(tau.min().detach().cpu()),
        "tau_maximum": float(tau.max().detach().cpu()),
    }

    shell_evidence = _field(output, "shell_evidence")
    if not isinstance(shell_evidence, Mapping) or not shell_evidence:
        raise TypeError("OSWT shell_evidence must be a non-empty scale mapping")
    probe_z = _tensor_field(output, "probe_z").to(torch.float64)
    probe_weights = _tensor_field(output, "probe_weights").to(torch.float64)
    boundary_scale_weights = _tensor_field(
        output, "boundary_scale_weights"
    ).to(torch.float64)
    num_boundaries = final_probs.shape[-1] - 1
    if tuple(boundary_scale_weights.shape) != (
        len(shell_evidence),
        num_boundaries,
    ):
        raise ValueError("OSWT boundary-scale weights have an invalid layout")

    def evidence_tensor(item: object, name: str) -> torch.Tensor:
        value = getattr(item, name, None)
        if not torch.is_tensor(value):
            raise ValueError(f"OSWT shell evidence omits {name}")
        return value.to(torch.float64)

    shell_simplex_errors: list[float] = []
    shell_reconstruction_errors: list[float] = []
    complementary_partition_errors: list[float] = []
    left_partition_reconstruction_errors: list[float] = []
    right_partition_reconstruction_errors: list[float] = []
    left_baseline_mass_errors: list[float] = []
    right_baseline_mass_errors: list[float] = []
    side_errors: dict[str, dict[str, list[float]]] = {
        side: {
            "surviving_mass": [],
            "surviving_mass_bound": [],
            "share": [],
            "share_mass": [],
            "phi": [],
            "local_warrant": [],
            "warrant_spectrum": [],
            "mixed_warrant": [],
            "local_boundary_warrant": [],
        }
        for side in ("left", "right")
    }
    reconstructed_boundary_warrants = {
        "left": torch.zeros_like(left_warrant),
        "right": torch.zeros_like(right_warrant),
    }
    shell_minimum = math.inf
    partition_mass_minimum = math.inf
    boundary_warrant_minimum = math.inf
    untouched_forward_ledger = all(
        torch.is_tensor(getattr(item, "active_mask", None))
        and torch.is_tensor(getattr(item, "original_valid_mask", None))
        and torch.equal(item.active_mask, item.original_valid_mask)
        for item in shell_evidence.values()
    )

    for scale_index, item in enumerate(shell_evidence.values()):
        shells = evidence_tensor(item, "ordinal_shells")
        normalized_atoms = evidence_tensor(item, "normalized_cumulative_atoms")
        active = getattr(item, "active_mask", None)
        if (
            shells.ndim != 4
            or shells.shape[1] != final_probs.shape[-1]
            or not torch.is_tensor(active)
            or tuple(active.shape)
            != (shells.shape[0], shells.shape[2], shells.shape[3])
        ):
            raise ValueError("OSWT shell evidence has an invalid grade/mask layout")
        active = active.bool()
        expected_cell_mass = active.to(dtype=shells.dtype)
        shell_simplex_errors.append(
            _max_abs(shells.sum(dim=1) - expected_cell_mass)
        )
        reconstructed_shells = ordinal_shells_from_cumulative_atoms(
            normalized_atoms, active
        )
        shell_reconstruction_errors.append(
            _max_abs(shells - reconstructed_shells)
        )

        left_mass = evidence_tensor(item, "local_left_partition_mass_map")
        right_mass = evidence_tensor(item, "local_right_partition_mass_map")
        expected_mass_shape = (
            shells.shape[0],
            num_boundaries,
            shells.shape[2],
            shells.shape[3],
        )
        if tuple(left_mass.shape) != expected_mass_shape or tuple(
            right_mass.shape
        ) != expected_mass_shape:
            raise ValueError("OSWT complementary partition maps have an invalid layout")
        complementary_partition_errors.append(
            _max_abs(left_mass + right_mass - expected_cell_mass[:, None])
        )
        expected_left_mass = torch.cumsum(shells, dim=1)[:, :-1]
        reverse_shell_mass = torch.flip(
            torch.cumsum(torch.flip(shells, dims=(1,)), dim=1), dims=(1,)
        )
        expected_right_mass = reverse_shell_mass[:, 1:]
        left_partition_reconstruction_errors.append(
            _max_abs(left_mass - expected_left_mass)
        )
        right_partition_reconstruction_errors.append(
            _max_abs(right_mass - expected_right_mass)
        )

        for side, mass_map in (("left", left_mass), ("right", right_mass)):
            baseline_mass = evidence_tensor(item, f"baseline_{side}_mass")
            surviving_mass = evidence_tensor(item, f"surviving_{side}_mass")
            share_map = evidence_tensor(item, f"local_{side}_share_map")
            local_phi = evidence_tensor(item, f"local_{side}_phi_spectrum")
            local_warrant = evidence_tensor(
                item, f"local_{side}_warrant_spectrum"
            )
            warrant_spectrum = evidence_tensor(item, f"{side}_warrant_spectrum")
            mixed_warrant = evidence_tensor(item, f"mixed_{side}_warrant")
            local_boundary_warrant = evidence_tensor(
                item, f"local_{side}_boundary_warrant_map"
            )
            if tuple(baseline_mass.shape) != left_warrant.shape or tuple(
                surviving_mass.shape
            ) != left_warrant.shape:
                raise ValueError("OSWT partition masses have an invalid layout")
            if untouched_forward_ledger:
                target = (
                    left_baseline_mass_errors
                    if side == "left"
                    else right_baseline_mass_errors
                )
                target.append(
                    _max_abs(baseline_mass - mass_map.sum(dim=(-2, -1)))
                )
            if tuple(share_map.shape) != expected_mass_shape:
                raise ValueError("OSWT partition-share map has an invalid layout")
            expected_spectrum_shape = expected_mass_shape + (probe_z.numel(),)
            if tuple(local_phi.shape) != expected_spectrum_shape or tuple(
                local_warrant.shape
            ) != expected_spectrum_shape:
                raise ValueError("OSWT local spectrum has an invalid layout")
            if tuple(warrant_spectrum.shape) != (
                shells.shape[0],
                num_boundaries,
                probe_z.numel(),
            ):
                raise ValueError("OSWT reduced warrant spectrum has an invalid layout")
            if tuple(mixed_warrant.shape) != left_warrant.shape or tuple(
                local_boundary_warrant.shape
            ) != expected_mass_shape:
                raise ValueError("OSWT mixed boundary warrant has an invalid layout")

            reconstructed_surviving = mass_map.sum(dim=(-2, -1))
            positive = baseline_mass > 0.0
            safe_mass = torch.where(
                positive, baseline_mass, torch.ones_like(baseline_mass)
            )
            expected_share = mass_map / safe_mass[:, :, None, None]
            expected_share = torch.where(
                positive[:, :, None, None],
                expected_share,
                torch.zeros_like(expected_share),
            )
            expected_phi = tangent_removed_pgf_probe(expected_share, probe_z)
            expected_phi = torch.where(
                active[:, None, :, :, None],
                expected_phi,
                torch.zeros_like(expected_phi),
            )
            expected_local_warrant = (
                expected_phi * baseline_mass[:, :, None, None, None]
            )
            expected_warrant_spectrum = expected_local_warrant.sum(dim=(2, 3))
            expected_mixed = torch.einsum(
                "nbl,l->nb", expected_warrant_spectrum, probe_weights
            )
            expected_local_boundary = torch.einsum(
                "nbhwl,l->nbhw", expected_local_warrant, probe_weights
            ) * boundary_scale_weights[scale_index][None, :, None, None]

            errors = side_errors[side]
            errors["surviving_mass"].append(
                _max_abs(surviving_mass - reconstructed_surviving)
            )
            errors["surviving_mass_bound"].append(
                _max_abs((surviving_mass - baseline_mass).clamp_min(0.0))
            )
            errors["share"].append(_max_abs(share_map - expected_share))
            errors["share_mass"].append(
                _max_abs(
                    share_map.sum(dim=(-2, -1))
                    - torch.where(
                        positive,
                        surviving_mass / safe_mass,
                        torch.zeros_like(surviving_mass),
                    )
                )
            )
            errors["phi"].append(_max_abs(local_phi - expected_phi))
            errors["local_warrant"].append(
                _max_abs(local_warrant - expected_local_warrant)
            )
            errors["warrant_spectrum"].append(
                _max_abs(warrant_spectrum - expected_warrant_spectrum)
            )
            errors["mixed_warrant"].append(
                _max_abs(mixed_warrant - expected_mixed)
            )
            errors["local_boundary_warrant"].append(
                _max_abs(local_boundary_warrant - expected_local_boundary)
            )
            reconstructed_boundary_warrants[side] += local_boundary_warrant.sum(
                dim=(-2, -1)
            )
            partition_mass_minimum = min(
                partition_mass_minimum,
                float(mass_map.min().detach().cpu()),
            )
            boundary_warrant_minimum = min(
                boundary_warrant_minimum,
                float(local_boundary_warrant.min().detach().cpu()),
            )
        shell_minimum = min(shell_minimum, float(shells.min().detach().cpu()))

    audit.update(
        {
            "ordinal_shell_simplex_error": max(shell_simplex_errors),
            "ordinal_shell_reconstruction_error": max(
                shell_reconstruction_errors
            ),
            "complementary_partition_mass_error": max(
                complementary_partition_errors
            ),
            "left_partition_map_reconstruction_error": max(
                left_partition_reconstruction_errors
            ),
            "right_partition_map_reconstruction_error": max(
                right_partition_reconstruction_errors
            ),
            "left_baseline_mass_reconstruction_error": max(
                left_baseline_mass_errors, default=0.0
            ),
            "right_baseline_mass_reconstruction_error": max(
                right_baseline_mass_errors, default=0.0
            ),
            "probe_weight_simplex_error": _max_abs(
                probe_weights.sum(dim=-1) - 1.0
            ),
            # Scale is the first axis and ordinal boundary is the second.
            "boundary_scale_weight_simplex_error": _max_abs(
                boundary_scale_weights.sum(dim=0) - 1.0
            ),
            "left_boundary_warrant_ledger_error": _max_abs(
                left_warrant - reconstructed_boundary_warrants["left"]
            ),
            "right_boundary_warrant_ledger_error": _max_abs(
                right_warrant - reconstructed_boundary_warrants["right"]
            ),
            "ordinal_shell_minimum": shell_minimum,
            "partition_mass_minimum": partition_mass_minimum,
            "local_boundary_warrant_minimum": boundary_warrant_minimum,
        }
    )
    for side in ("left", "right"):
        errors = side_errors[side]
        audit.update(
            {
                f"{side}_surviving_mass_reconstruction_error": max(
                    errors["surviving_mass"]
                ),
                f"{side}_surviving_mass_bound_error": max(
                    errors["surviving_mass_bound"]
                ),
                f"{side}_partition_share_reconstruction_error": max(
                    errors["share"]
                ),
                f"{side}_partition_share_mass_error": max(
                    errors["share_mass"]
                ),
                f"{side}_phi_spectrum_reconstruction_error": max(
                    errors["phi"]
                ),
                f"{side}_local_warrant_spectrum_formula_error": max(
                    errors["local_warrant"]
                ),
                f"{side}_warrant_spectrum_reduction_error": max(
                    errors["warrant_spectrum"]
                ),
                f"{side}_mixed_warrant_reconstruction_error": max(
                    errors["mixed_warrant"]
                ),
                f"{side}_local_boundary_warrant_reconstruction_error": max(
                    errors["local_boundary_warrant"]
                ),
            }
        )

    if intervention is None:
        return audit

    replayed = _field(intervention, "output")
    replay_audit = audit_paths_oswt_output(replayed)
    audit.update({f"replayed_{key}": value for key, value in replay_audit.items()})
    replay_base = _field(replayed, "base_output")
    replay_rates = _tensor_field(replayed, "total_rates")
    audit.update(
        {
            "origin_rate_deletion_replay_error": _max_abs(
                _tensor_field(output, "total_rates")
                - _field(intervention, "removed_rates")
                - replay_rates
            ),
            "left_boundary_warrant_deletion_replay_error": _max_abs(
                _tensor_field(output, "left_boundary_warrant")
                - _field(intervention, "removed_left_boundary_warrant")
                - _tensor_field(replayed, "left_boundary_warrant")
            ),
            "right_boundary_warrant_deletion_replay_error": _max_abs(
                _tensor_field(output, "right_boundary_warrant")
                - _field(intervention, "removed_right_boundary_warrant")
                - _tensor_field(replayed, "right_boundary_warrant")
            ),
            "log_odds_increment_deletion_replay_error": _max_abs(
                _tensor_field(output, "log_odds_increment")
                - _field(intervention, "removed_log_odds_increment")
                - _tensor_field(replayed, "log_odds_increment")
            ),
            "signed_flow_deletion_replay_error": _max_abs(
                _tensor_field(output, "signed_boundary_flow")
                - _field(intervention, "removed_signed_boundary_flow")
                - _tensor_field(replayed, "signed_boundary_flow")
            ),
            "continuation_correction_deletion_replay_error": _max_abs(
                _tensor_field(output, "boundary_correction")
                - _field(intervention, "removed_boundary_correction")
                - _tensor_field(replayed, "boundary_correction")
            ),
            "base_rate_intervention_consistency_error": _max_abs(
                _tensor_field(replay_base, "total_rates") - replay_rates
            ),
        }
    )
    return audit


def _shortlist_intervention_cells(
    output: object,
    sample: int,
    *,
    limit: int,
) -> list[dict[str, Any]]:
    """Rank predicted-grade support without using any replay outcome."""

    shell_evidence = _field(output, "shell_evidence")
    if not isinstance(shell_evidence, Mapping) or not shell_evidence:
        raise TypeError("OSWT shell_evidence must be a non-empty scale mapping")
    local_warrants = _field(output, "local_warrant_maps")
    if not isinstance(local_warrants, Mapping) or set(local_warrants) != set(
        shell_evidence
    ):
        raise TypeError("OSWT predicted-grade warrant maps do not match its scales")
    predicted_grade = int(
        _tensor_field(output, "class_map")[sample].detach().cpu()
    )
    candidates: list[dict[str, Any]] = []
    for scale_rank, (scale, item) in enumerate(shell_evidence.items()):
        valid = getattr(item, "active_mask", None)
        warrant = local_warrants[scale]
        if not torch.is_tensor(valid) or not torch.is_tensor(warrant):
            raise ValueError("OSWT evidence omits a predicted-grade warrant map")
        valid_sample = valid[sample].bool()
        if tuple(warrant.shape[1:3]) != tuple(valid_sample.shape):
            raise ValueError("OSWT local grade warrants do not match the valid mask")
        score = warrant[sample, :, :, predicted_grade]
        flat_valid = torch.nonzero(
            valid_sample.reshape(-1), as_tuple=False
        ).flatten()
        for flat_tensor in flat_valid:
            flat_index = int(flat_tensor.detach().cpu())
            spatial = tuple(
                int(value)
                for value in np.unravel_index(flat_index, valid_sample.shape)
            )
            candidates.append(
                {
                    "scale": str(scale),
                    "scale_rank": scale_rank,
                    "spatial_index": spatial,
                    "flat_index": flat_index,
                    "local_intervention_score": float(
                        score.reshape(-1)[flat_index].detach().cpu()
                    ),
                }
            )
    if not candidates:
        raise ValueError("OSWT certificate sample has no valid evidence cell")
    candidates.sort(
        key=lambda item: (
            -float(item["local_intervention_score"]),
            int(item["scale_rank"]),
            int(item["flat_index"]),
        )
    )
    return candidates[: min(int(limit), len(candidates))]


def _predicted_grade_margin(
    probabilities: torch.Tensor,
    predicted_grade: int,
) -> torch.Tensor:
    target = probabilities[predicted_grade]
    competitors = torch.cat(
        (probabilities[:predicted_grade], probabilities[predicted_grade + 1 :])
    )
    return target - competitors.max()


def _merge_audit_maxima(
    aggregate: dict[str, float],
    current: Mapping[str, float],
) -> None:
    for key, value in current.items():
        numeric = float(value)
        if key.endswith("_minimum"):
            aggregate[key] = min(aggregate.get(key, math.inf), numeric)
        else:
            aggregate[key] = max(aggregate.get(key, -math.inf), numeric)


class PathsOSWTTrainer(PathsTrainer):
    """Train only the OSWT refiner on top of an immutable audited V3 model."""

    def __init__(
        self,
        model: nn.Module,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        test_loader: torch.utils.data.DataLoader | None,
        cfg: object,
        run_dir: str | Path,
        *,
        fold: int = 0,
        split_signature: str | None = None,
        device: torch.device | str | None = None,
    ) -> None:
        # Reuse only PATHS's strict V3 loader and proper risk-set loss.  Its
        # public variant validation is SAPT-specific, so present a temporary
        # internal value during construction and replace every protocol field
        # immediately afterwards.
        had_paths_variant = hasattr(cfg, "paths_variant")
        old_paths_variant = getattr(cfg, "paths_variant", None)
        internal_variant = (
            "risk_objective_v3"
            if str(getattr(cfg, "oswt_variant")) == "identity_v3"
            else "signed_transport"
        )
        setattr(cfg, "paths_variant", internal_variant)
        try:
            super().__init__(
                model,
                train_loader,
                val_loader,
                test_loader,
                cfg,
                run_dir,
                fold=fold,
                split_signature=split_signature,
                device=device,
            )
        finally:
            if had_paths_variant:
                setattr(cfg, "paths_variant", old_paths_variant)
            else:
                delattr(cfg, "paths_variant")

        self.oswt_variant = str(getattr(cfg, "oswt_variant"))
        self.paths_variant = self.oswt_variant
        self._install_oswt_optimizer(cfg)
        self._set_training_phase(0)
        self.source_v3_base_state_sha256 = _v3_base_state_sha256(
            self.model.state_dict()
        )
        self.implementation_signature = paths_oswt_implementation_signature()
        self.architecture = _architecture_record(self.model)
        declared_architecture = self.architecture.get("declared")
        if not isinstance(declared_architecture, Mapping) or not isinstance(
            declared_architecture.get("posterior_path"), str
        ):
            raise ValueError("OSWT architecture metadata omits its posterior path")
        self.architecture["posterior_path"] = declared_architecture[
            "posterior_path"
        ]
        self.architecture_signature = _canonical_sha256(self.architecture)
        self.critical_config = _critical_config(cfg)
        self.config_signature = _canonical_sha256(self.critical_config)
        self.run_git_commit = os.environ.get(
            "PATHS_OSWT_RUN_GIT_COMMIT",
            os.environ.get("PATHS_RUN_GIT_COMMIT"),
        )
        if self.run_git_commit is not None:
            commit = self.run_git_commit.lower()
            if len(commit) not in {40, 64} or any(
                character not in "0123456789abcdef" for character in commit
            ):
                raise ValueError("PATHS_OSWT_RUN_GIT_COMMIT must be hexadecimal")
            self.run_git_commit = commit
        self.checkpoint_schema = PATHS_OSWT_CHECKPOINT_SCHEMA
        self.base_control_path = self.run_dir / "hash_bound_v3_identity_control.json"
        self.structural_audit_path = self.run_dir / "validation_oswt_audit.json"

    def _install_oswt_optimizer(self, cfg: object) -> None:
        refiner = getattr(self.model, "paths_refiner", None)
        if not isinstance(refiner, nn.Module):
            raise TypeError("OSWT model must expose paths_refiner")
        parameters = list(refiner.parameters())
        if not parameters:
            raise ValueError("OSWT refiner has no parameters")
        refiner_ids = {id(parameter) for parameter in parameters}
        for parameter in self.model.parameters():
            parameter.requires_grad_(id(parameter) in refiner_ids)
        self.optimizer_group_contract = [
            {
                "name": "paths_oswt_refiner",
                "initial_lr": float(getattr(cfg, "paths_refiner_lr", 5e-4)),
                "weight_decay": 0.0,
                "parameter_count": sum(parameter.numel() for parameter in parameters),
            }
        ]
        self.optimizer = torch.optim.AdamW(
            [
                {
                    "params": parameters,
                    "lr": float(getattr(cfg, "paths_refiner_lr", 5e-4)),
                    "weight_decay": 0.0,
                    "name": "paths_oswt_refiner",
                }
            ],
            weight_decay=0.0,
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            factor=float(getattr(cfg, "lr_factor", 0.2)),
            patience=int(getattr(cfg, "lr_patience", 5)),
            min_lr=float(getattr(cfg, "lr_min", 1e-6)),
        )

    def _set_training_phase(self, epoch: int) -> None:
        del epoch
        learned = self.oswt_variant != "identity_v3"
        for name, parameter in self.model.named_parameters():
            parameter.requires_grad_(learned and name.startswith("paths_refiner."))

    def _set_encoder_trainable(self, epoch: int) -> None:
        self._set_training_phase(epoch)

    def _reset_amp_scaler_at_encoder_unfreeze(self, epoch: int) -> bool:
        del epoch
        return False

    def _forward(self, images: torch.Tensor, pixel_mask: torch.Tensor | None) -> object:
        # ``model.train()`` is called by the generic loop.  Restore behavioral
        # immutability of the V3 trunk on every batch so ConvNeXt stochastic
        # depth and any generator dropout cannot change the hash-bound base.
        training = bool(self.model.training)
        for name in ("encoder", "generator"):
            module = getattr(self.model, name, None)
            if isinstance(module, nn.Module):
                module.eval()
        refiner = getattr(self.model, "paths_refiner", None)
        if isinstance(refiner, nn.Module):
            refiner.train(training and self.oswt_variant != "identity_v3")
        return OriginTrainer._forward(self, images, pixel_mask)

    def _checkpoint_payload(
        self,
        epoch: int,
        metrics: Mapping[str, Any],
        best_key: tuple[float, float, float],
        best_epoch: int,
        bad_epochs: int,
        candidate_floor_evaluation: Mapping[str, Any] | None = None,
        best_learned_key: tuple[float, float, float] | None = None,
        best_learned_epoch: int | None = None,
        learned_bad_epochs: int = 0,
        checkpoint_role: str | None = None,
        deployable_best_checkpoint_sha256: str | None = None,
        best_learned_checkpoint_sha256: str | None = None,
        checkpoint_transaction: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        if candidate_floor_evaluation is not None:
            raise ValueError("OSWT has no inherited metric-floor selector")
        if best_learned_key is not None or best_learned_epoch is not None:
            raise ValueError("OSWT uses one learned checkpoint track")
        payload = OriginTrainer._checkpoint_payload(
            self,
            epoch,
            metrics,
            best_key,
            best_epoch,
            bad_epochs,
            candidate_floor_evaluation=candidate_floor_evaluation,
            best_learned_key=best_learned_key,
            best_learned_epoch=best_learned_epoch,
            learned_bad_epochs=learned_bad_epochs,
            checkpoint_role=checkpoint_role,
            deployable_best_checkpoint_sha256=deployable_best_checkpoint_sha256,
            best_learned_checkpoint_sha256=best_learned_checkpoint_sha256,
            checkpoint_transaction=checkpoint_transaction,
        )
        role = payload.get("checkpoint_role")
        if role == "deployable_selected":
            role = "oswt_selected_learned"
        checkpoint_base_hash = _v3_base_state_sha256(self.model.state_dict())
        base_matches_source = (
            checkpoint_base_hash == self.source_v3_base_state_sha256
        )
        if not base_matches_source:
            raise AssertionError("OSWT checkpoint V3 base differs from its loaded source")
        payload.update(
            {
                "schema": PATHS_OSWT_CHECKPOINT_SCHEMA,
                "paths_oswt_protocol_version": PATHS_OSWT_PROTOCOL_VERSION,
                "oswt_variant": self.oswt_variant,
                "run_git_commit": self.run_git_commit,
                "checkpoint_role": role,
                "warm_start_provenance": None,
                "paths_oswt_warm_start_provenance": self.paths_warm_start_provenance,
                "training_label_counts": list(self.training_label_counts),
                "risk_set_boundary_weights": self.criterion.boundary_weights.detach()
                .cpu()
                .tolist(),
                "risk_set_weights_from_training_labels_only": True,
                "optimizer_group_contract": self.optimizer_group_contract,
                "source_v3_base_state_sha256": self.source_v3_base_state_sha256,
                "checkpoint_v3_base_state_sha256": checkpoint_base_hash,
                "checkpoint_v3_base_matches_source": base_matches_source,
                "training_phase": (
                    "identity_v3" if self.oswt_variant == "identity_v3" else "oswt_refiner_only"
                ),
            }
        )
        return payload

    def _validate_resume(
        self,
        state: Mapping[str, Any],
        *,
        recover_transaction: bool = False,
    ) -> None:
        if state.get("schema") != PATHS_OSWT_CHECKPOINT_SCHEMA:
            raise ValueError(
                f"OSWT resume requires a {PATHS_OSWT_CHECKPOINT_SCHEMA} checkpoint"
            )
        if state.get("paths_oswt_protocol_version") != PATHS_OSWT_PROTOCOL_VERSION:
            raise ValueError("OSWT resume protocol mismatch")
        if state.get("oswt_variant") != self.oswt_variant:
            raise ValueError("OSWT resume variant mismatch")
        if state.get("run_git_commit") != self.run_git_commit:
            raise ValueError("OSWT resume git-commit identity mismatch")
        if _normalized_warm_start_provenance(
            state.get("paths_oswt_warm_start_provenance")
        ) != _normalized_warm_start_provenance(self.paths_warm_start_provenance):
            raise ValueError("OSWT resume warm-start provenance mismatch")
        if state.get("training_label_counts") != self.training_label_counts:
            raise ValueError("OSWT resume training-label counts mismatch")
        if state.get("optimizer_group_contract") != self.optimizer_group_contract:
            raise ValueError("OSWT resume optimizer-group contract mismatch")
        if state.get("source_v3_base_state_sha256") != (
            self.source_v3_base_state_sha256
        ):
            raise ValueError("OSWT resume source V3 base hash mismatch")
        saved_model_state = state.get("model_state")
        if not isinstance(saved_model_state, Mapping):
            raise ValueError("OSWT resume checkpoint omits model_state")
        saved_base_hash = _v3_base_state_sha256(saved_model_state)
        if state.get("checkpoint_v3_base_state_sha256") != saved_base_hash:
            raise ValueError("OSWT resume checkpoint V3 base hash is invalid")
        if (
            saved_base_hash != self.source_v3_base_state_sha256
            or state.get("checkpoint_v3_base_matches_source") is not True
        ):
            raise ValueError("OSWT resume checkpoint changed the frozen V3 base")
        OriginTrainer._validate_resume(
            self,
            state,
            recover_transaction=recover_transaction,
        )

    def _evaluate_v3_control(self) -> dict[str, Any]:
        self.model.eval()
        probabilities: list[torch.Tensor] = []
        predictions: list[torch.Tensor] = []
        labels_all: list[torch.Tensor] = []
        ordered_ids: list[str] = []
        running_index = 0
        with torch.no_grad():
            for batch in self.val_loader:
                images, pixel_mask, labels, indices = self._unpack_batch(batch)
                with autocast(device_type="cuda", enabled=self.use_amp):
                    output = self._forward(images, pixel_mask)
                base = _field(output, "base_output") if hasattr(output, "base_output") else output
                probabilities.append(_tensor_field(base, "class_probs").float().cpu())
                if self.decision_rule == "rounded_expected":
                    predicted = _tensor_field(base, "expected_grade").round()
                else:
                    predicted = _tensor_field(base, self.decision_rule)
                predictions.append(predicted.long().cpu())
                labels_all.append(labels.cpu())
                if indices is None:
                    ordered_ids.extend(
                        str(value)
                        for value in range(running_index, running_index + int(labels.numel()))
                    )
                elif torch.is_tensor(indices):
                    ordered_ids.extend(str(value) for value in indices.detach().cpu().tolist())
                else:
                    ordered_ids.extend(str(value) for value in indices)
                running_index += int(labels.numel())
        metrics = evaluate_origin_predictions(
            torch.cat(probabilities), torch.cat(predictions), torch.cat(labels_all)
        )
        source = self.paths_warm_start_provenance["source_metrics"]
        tolerance = float(getattr(self.cfg, "warm_start_metric_tolerance", 1e-6))
        errors = {
            name: abs(float(metrics[name]) - float(source[name]))
            for name in ("acc", "qwk", "mae", "balanced_acc", "macro_f1")
        }
        if any(value > tolerance for value in errors.values()):
            raise AssertionError(
                "hash-bound V3 warm start did not reproduce source metrics: "
                f"errors={errors}, tolerance={tolerance}"
            )
        confusion = [[int(value) for value in row] for row in metrics["confusion"]]
        if confusion != source["confusion"]:
            raise AssertionError("hash-bound V3 warm start did not reproduce confusion")
        current_base_hash = _v3_base_state_sha256(self.model.state_dict())
        if current_base_hash != self.source_v3_base_state_sha256:
            raise AssertionError("V3 identity control observed base-state drift")
        payload = {
            "schema": "paths-oswt-v3-identity-control-v1",
            "scope": "inner_validation_only",
            "fold": self.fold,
            "split_signature": self.split_signature,
            "source_checkpoint_sha256": self.paths_warm_start_provenance[
                "source_checkpoint_sha256"
            ],
            "paths_oswt_protocol_version": PATHS_OSWT_PROTOCOL_VERSION,
            "oswt_variant": self.oswt_variant,
            "implementation_signature": self.implementation_signature,
            "architecture_signature": self.architecture_signature,
            "metrics": metrics,
            "ordered_prediction_checksum_sha256": _canonical_sha256(
                {
                    "ordered_ids": ordered_ids,
                    "labels": torch.cat(labels_all).tolist(),
                    "predictions": torch.cat(predictions).tolist(),
                }
            ),
            "source_metric_absolute_errors": errors,
            "reproduction_tolerance": tolerance,
            "reproduced": True,
            "source_v3_base_state_sha256": self.source_v3_base_state_sha256,
            "control_v3_base_state_sha256": current_base_hash,
            "control_v3_base_matches_source": True,
        }
        payload["content_checksum_sha256"] = _canonical_sha256(payload)
        temporary = self.base_control_path.with_name(
            f".{self.base_control_path.name}.{os.getpid()}.tmp"
        )
        try:
            with temporary.open("w", encoding="utf-8") as stream:
                json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
                stream.write("\n")
            os.replace(temporary, self.base_control_path)
        finally:
            if temporary.exists():
                temporary.unlink()
        return payload

    def _validation_certificate_samples(self) -> list[dict[str, Any]]:
        """Choose deterministic grade-stratified samples from all validation rows."""

        records: list[dict[str, Any]] = []
        position = 0
        for batch in self.val_loader:
            images, pixel_mask, labels, indices = self._unpack_batch(batch)
            del images, pixel_mask
            for sample in range(int(labels.numel())):
                sample_id = (
                    position
                    if indices is None
                    else self._index_value(indices, sample)
                )
                records.append(
                    {
                        "validation_position": position,
                        "sample_id": sample_id,
                        "label": int(labels[sample].detach().cpu()),
                    }
                )
                position += 1
        if not records:
            raise ValueError("OSWT certificate requires a non-empty validation set")

        requested = min(
            int(getattr(self.cfg, "certificate_samples", 8)), len(records)
        )
        by_grade: dict[int, list[dict[str, Any]]] = {}
        for record in records:
            by_grade.setdefault(int(record["label"]), []).append(record)
        if requested < len(by_grade):
            raise ValueError(
                "certificate_samples must cover every validation grade; "
                f"requested={requested}, represented_grades={len(by_grade)}"
            )
        for bucket in by_grade.values():
            bucket.sort(
                key=lambda record: (
                    _canonical_sha256({"sample_id": record["sample_id"]}),
                    int(record["validation_position"]),
                )
            )

        chosen: list[dict[str, Any]] = []
        depth = 0
        grades = sorted(by_grade)
        while len(chosen) < requested:
            progressed = False
            for grade in grades:
                bucket = by_grade[grade]
                if depth < len(bucket):
                    chosen.append(dict(bucket[depth]))
                    progressed = True
                    if len(chosen) == requested:
                        break
            if not progressed:
                break
            depth += 1
        if len(chosen) != requested:
            raise AssertionError("failed to select the requested certificate samples")
        for rank, record in enumerate(chosen):
            record["certificate_rank"] = rank
        return chosen

    def _exact_best_shortlist_witness(
        self,
        baseline: object,
        sample: int,
    ) -> tuple[dict[str, Any], object]:
        """Replay every prespecified candidate and return the exact best witness."""

        candidates = _shortlist_intervention_cells(
            baseline,
            sample,
            limit=int(
                getattr(
                    self.cfg,
                    "certificate_shortlist_size",
                    _OSWT_CERTIFICATE_CANDIDATE_LIMIT,
                )
            ),
        )
        baseline_probs = _tensor_field(baseline, "class_probs")[sample].to(
            torch.float64
        )
        predicted_grade = int(self._predictions(baseline)[sample].detach().cpu())
        baseline_margin = _predicted_grade_margin(
            baseline_probs, predicted_grade
        )
        shell_evidence = _field(baseline, "shell_evidence")
        if not isinstance(shell_evidence, Mapping):
            raise TypeError("OSWT shell_evidence must be a scale mapping")
        replay = getattr(self.model, "replay_without", None)

        best_key: tuple[float, float, int, int] | None = None
        best: dict[str, Any] | None = None
        best_intervention: object | None = None
        candidate_replays: list[dict[str, Any]] = []
        for shortlist_rank, candidate in enumerate(candidates):
            masks = {
                name: torch.zeros_like(item.active_mask, dtype=torch.bool)
                for name, item in shell_evidence.items()
            }
            spatial = tuple(int(value) for value in candidate["spatial_index"])
            masks[str(candidate["scale"])][(sample, *spatial)] = True
            intervention = (
                replay(baseline, masks, force_decoder_fp64=True)
                if callable(replay)
                else replay_paths_oswt_without(
                    baseline, masks, force_decoder_fp64=True
                )
            )
            replayed = _field(intervention, "output")
            replayed_probs = _tensor_field(replayed, "class_probs")[sample].to(
                torch.float64
            )
            replayed_margin = _predicted_grade_margin(
                replayed_probs, predicted_grade
            )
            margin_drop = baseline_margin - replayed_margin
            target_log_probability_drop = (
                baseline_probs[predicted_grade].log()
                - replayed_probs[predicted_grade].log()
            )
            replay_record = {
                "shortlist_rank": shortlist_rank,
                "scale": str(candidate["scale"]),
                "spatial_index": list(spatial),
                "flat_index": int(candidate["flat_index"]),
                "local_intervention_score": float(
                    candidate["local_intervention_score"]
                ),
                "predicted_grade_margin_drop": float(
                    margin_drop.detach().cpu()
                ),
                "target_log_probability_drop": float(
                    target_log_probability_drop.detach().cpu()
                ),
            }
            candidate_replays.append(replay_record)
            key = (
                replay_record["predicted_grade_margin_drop"],
                replay_record["target_log_probability_drop"],
                -int(candidate["scale_rank"]),
                -int(candidate["flat_index"]),
            )
            if best_key is None or key > best_key:
                best_key = key
                best = {
                    **replay_record,
                    "scale_rank": int(candidate["scale_rank"]),
                    "baseline_predicted_grade": predicted_grade,
                    "baseline_predicted_grade_margin": float(
                        baseline_margin.detach().cpu()
                    ),
                    "replayed_predicted_grade_margin": float(
                        replayed_margin.detach().cpu()
                    ),
                }
                best_intervention = intervention

        if best is None or best_intervention is None:
            raise AssertionError("OSWT exact witness selection produced no candidate")
        positive_margin_drop = float(best["predicted_grade_margin_drop"]) > 0.0
        best.update(
            {
                "selection_scope": "exact_best_within_prespecified_shortlist",
                "selection_rule": (
                    "max_predicted_grade_margin_drop_then_target_log_probability_"
                    "drop_then_scale_then_flat_index"
                ),
                "shortlist_ranking_rule": (
                    "descending_predicted_grade_local_warrant_then_scale_then_"
                    "flat_index"
                ),
                "positive_margin_drop": positive_margin_drop,
                "positive_margin_drop_required_for_promotion": True,
                "candidate_shortlist_limit": int(
                    getattr(
                        self.cfg,
                        "certificate_shortlist_size",
                        _OSWT_CERTIFICATE_CANDIDATE_LIMIT,
                    )
                ),
                "candidate_shortlist_size": len(candidates),
                "candidate_replays": candidate_replays,
            }
        )
        return best, best_intervention

    def _write_validation_certificates(self, checkpoint_epoch: int) -> dict[str, Any]:
        self.model.eval()
        selected_records = self._validation_certificate_samples()
        selected_by_position = {
            int(record["validation_position"]): record
            for record in selected_records
        }
        certificate_rows_by_rank: dict[int, dict[str, Any]] = {}
        audit: dict[str, float] = {}
        validation_position = 0
        with torch.no_grad():
            for batch in self.val_loader:
                images, pixel_mask, labels, indices = self._unpack_batch(batch)
                batch_size = int(labels.numel())
                selected_in_batch = [
                    (sample, selected_by_position[validation_position + sample])
                    for sample in range(batch_size)
                    if validation_position + sample in selected_by_position
                ]
                if selected_in_batch:
                    with autocast(device_type="cuda", enabled=self.use_amp):
                        baseline = self._forward(images, pixel_mask)
                    baseline_probs = _tensor_field(
                        baseline, "class_probs"
                    ).detach().cpu()
                    baseline_predictions = self._predictions(baseline).detach().cpu()
                    baseline_expected = _tensor_field(
                        baseline, "expected_grade"
                    ).detach().cpu()

                    if self.oswt_variant == "identity_v3":
                        current = {
                            "posterior_normalization_error": _max_abs(
                                _tensor_field(baseline, "class_probs").sum(dim=-1)
                                - 1.0
                            )
                        }
                        _merge_audit_maxima(audit, current)

                    for sample, record in selected_in_batch:
                        sample_id = (
                            validation_position + sample
                            if indices is None
                            else self._index_value(indices, sample)
                        )
                        label = int(labels[sample].detach().cpu())
                        if sample_id != record["sample_id"] or label != int(
                            record["label"]
                        ):
                            raise RuntimeError(
                                "validation order changed during certificate replay"
                            )
                        rank = int(record["certificate_rank"])
                        if self.oswt_variant == "identity_v3":
                            certificate_rows_by_rank[rank] = {
                                **record,
                                "prediction": int(baseline_predictions[sample]),
                                "class_probs": baseline_probs[sample].tolist(),
                            }
                            continue

                        witness, intervention = self._exact_best_shortlist_witness(
                            baseline, sample
                        )
                        replayed = _field(intervention, "output")
                        _merge_audit_maxima(
                            audit,
                            audit_paths_oswt_output(baseline, intervention),
                        )
                        replay_probs = _tensor_field(
                            replayed, "class_probs"
                        ).detach().cpu()
                        replay_predictions = self._predictions(replayed).detach().cpu()
                        replay_expected = _tensor_field(
                            replayed, "expected_grade"
                        ).detach().cpu()
                        removed_left = _field(
                            intervention, "removed_left_boundary_warrant"
                        ).detach().cpu()
                        removed_right = _field(
                            intervention, "removed_right_boundary_warrant"
                        ).detach().cpu()
                        removed_flow = _field(
                            intervention, "removed_signed_boundary_flow"
                        ).detach().cpu()
                        removed_increment = _field(
                            intervention, "removed_log_odds_increment"
                        ).detach().cpu()
                        certificate_rows_by_rank[rank] = {
                            **record,
                            **witness,
                            "baseline_prediction": int(
                                baseline_predictions[sample]
                            ),
                            "replayed_prediction": int(
                                replay_predictions[sample]
                            ),
                            "baseline_expected_grade": float(
                                baseline_expected[sample]
                            ),
                            "replayed_expected_grade": float(
                                replay_expected[sample]
                            ),
                            "expected_grade_drop": float(
                                baseline_expected[sample]
                                - replay_expected[sample]
                            ),
                            "baseline_class_probs": baseline_probs[sample].tolist(),
                            "replayed_class_probs": replay_probs[sample].tolist(),
                            "posterior_l1_change": float(
                                (
                                    baseline_probs[sample]
                                    - replay_probs[sample]
                                )
                                .abs()
                                .sum()
                            ),
                            "removed_left_boundary_warrant": removed_left[
                                sample
                            ].tolist(),
                            "removed_right_boundary_warrant": removed_right[
                                sample
                            ].tolist(),
                            "removed_log_odds_increment": removed_increment[
                                sample
                            ].tolist(),
                            "removed_signed_boundary_flow": removed_flow[
                                sample
                            ].tolist(),
                        }
                validation_position += batch_size

        if len(certificate_rows_by_rank) != len(selected_records):
            raise RuntimeError("not every selected validation certificate was replayed")
        certificate_rows = [
            certificate_rows_by_rank[index]
            for index in range(len(selected_records))
        ]
        positive_witness_count = (
            None
            if self.oswt_variant == "identity_v3"
            else sum(
                bool(row["positive_margin_drop"])
                for row in certificate_rows
            )
        )
        positive_witness_rate = (
            None
            if positive_witness_count is None
            else positive_witness_count / len(certificate_rows)
        )
        all_witnesses_positive = (
            None
            if positive_witness_count is None
            else positive_witness_count == len(certificate_rows)
        )
        if self.oswt_variant == "identity_v3":
            audit["identity_control"] = True
        else:
            tolerance = float(
                getattr(self.cfg, "certificate_replay_tolerance", 2e-5)
            )
            failures = {
                key: value
                for key, value in audit.items()
                if key.endswith("_error") and value > tolerance
            }
            if failures:
                raise AssertionError(
                    "OSWT certificate identities failed: "
                    f"{failures}, tolerance={tolerance}"
                )
            if min(
                audit["posterior_minimum"],
                audit["replayed_posterior_minimum"],
            ) < -tolerance:
                raise AssertionError("OSWT certificate contains negative posterior mass")
            if audit["transport_effect_max_abs"] <= 1e-12:
                raise AssertionError("learned OSWT checkpoint has zero transport effect")

        selected_checkpoint_sha256 = _file_sha256(self.best_path)
        selected_state = torch.load(
            self.best_path, map_location="cpu", weights_only=False
        )
        if int(selected_state.get("epoch", -1)) != int(checkpoint_epoch):
            raise ValueError("certificate checkpoint epoch mismatch")
        selected_model_state = selected_state.get("model_state")
        if not isinstance(selected_model_state, Mapping):
            raise ValueError("selected OSWT checkpoint omits model_state")
        selected_base_hash = _v3_base_state_sha256(selected_model_state)
        final_base_hash = _v3_base_state_sha256(self.model.state_dict())
        if not (
            selected_base_hash
            == final_base_hash
            == self.source_v3_base_state_sha256
        ):
            raise AssertionError("OSWT certificate observed V3 base-state drift")

        payload: dict[str, Any] = {
            "schema": "paths-oswt-exact-validation-certificates-v2",
            "scope": "inner_validation_only",
            "fold": self.fold,
            "split_signature": self.split_signature,
            "checkpoint_epoch": int(checkpoint_epoch),
            "selected_checkpoint_sha256": selected_checkpoint_sha256,
            "paths_oswt_protocol_version": PATHS_OSWT_PROTOCOL_VERSION,
            "oswt_variant": self.oswt_variant,
            "implementation_signature": self.implementation_signature,
            "architecture_signature": self.architecture_signature,
            "source_checkpoint_sha256": self.paths_warm_start_provenance[
                "source_checkpoint_sha256"
            ],
            "source_v3_base_state_sha256": self.source_v3_base_state_sha256,
            "final_v3_base_state_sha256": final_base_hash,
            "selected_v3_base_state_sha256": selected_base_hash,
            "final_v3_base_matches_source": True,
            "selected_v3_base_matches_source": True,
            "prediction_is_transported_posterior": self.oswt_variant != "identity_v3",
            "sample_selection": "deterministic_grade_round_robin_full_validation",
            "candidate_shortlist_ranking_rule": (
                "descending_predicted_grade_local_warrant_then_scale_then_flat_index"
            ),
            "certificate_shortlist_size": int(
                getattr(
                    self.cfg,
                    "certificate_shortlist_size",
                    _OSWT_CERTIFICATE_CANDIDATE_LIMIT,
                )
            ),
            "witness_selection_scope": (
                None
                if self.oswt_variant == "identity_v3"
                else "exact_best_within_prespecified_shortlist"
            ),
            "positive_witness_count": positive_witness_count,
            "positive_witness_rate": positive_witness_rate,
            "all_witnesses_positive": all_witnesses_positive,
            "audit": audit,
            "certificates": certificate_rows,
            "certificate_count": len(certificate_rows),
            "deletion_replay_audited": self.oswt_variant != "identity_v3",
            "deletion_replay_error_keys": sorted(
                key
                for key in audit
                if key.endswith("_deletion_replay_error")
                or key == "base_rate_intervention_consistency_error"
            ),
        }
        checksum = _canonical_sha256(payload)
        payload["content_checksum_sha256"] = checksum
        temporary = self.certificate_path.with_name(
            f".{self.certificate_path.name}.{os.getpid()}.tmp"
        )
        try:
            with temporary.open("w", encoding="utf-8") as stream:
                json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
                stream.write("\n")
            os.replace(temporary, self.certificate_path)
        finally:
            if temporary.exists():
                temporary.unlink()
        return payload

    def _evaluate_validation_structure(
        self, checkpoint_epoch: int
    ) -> dict[str, Any]:
        self.model.eval()
        aggregate: dict[str, float] = {}
        samples = 0
        tolerance = float(getattr(self.cfg, "certificate_replay_tolerance", 2e-5))
        with torch.no_grad():
            for batch in self.val_loader:
                images, pixel_mask, labels, _ = self._unpack_batch(batch)
                with autocast(device_type="cuda", enabled=self.use_amp):
                    output = self._forward(images, pixel_mask)
                samples += int(labels.numel())
                if self.oswt_variant == "identity_v3":
                    current = {
                        "posterior_normalization_error": _max_abs(
                            _tensor_field(output, "class_probs").sum(dim=-1) - 1.0
                        )
                    }
                else:
                    current = audit_paths_oswt_output(output)
                    failures = {
                        key: value
                        for key, value in current.items()
                        if key.endswith("_error") and value > tolerance
                    }
                    if failures:
                        raise AssertionError(
                            "OSWT full-validation structural identities failed: "
                            f"{failures}, tolerance={tolerance}"
                        )
                    if current["posterior_minimum"] < -tolerance:
                        raise AssertionError(
                            "OSWT full-validation posterior contains negative mass"
                        )
                _merge_audit_maxima(aggregate, current)

        maxima = {
            key: value
            for key, value in aggregate.items()
            if key.endswith("_error") or key.endswith("_max_abs")
        }
        selected_checkpoint_sha256 = _file_sha256(self.best_path)
        selected_state = torch.load(
            self.best_path, map_location="cpu", weights_only=False
        )
        if int(selected_state.get("epoch", -1)) != int(checkpoint_epoch):
            raise ValueError("structural audit checkpoint epoch mismatch")
        selected_model_state = selected_state.get("model_state")
        if not isinstance(selected_model_state, Mapping):
            raise ValueError("selected OSWT checkpoint omits model_state")
        selected_base_hash = _v3_base_state_sha256(selected_model_state)
        final_base_hash = _v3_base_state_sha256(self.model.state_dict())
        if not (
            selected_base_hash
            == final_base_hash
            == self.source_v3_base_state_sha256
        ):
            raise AssertionError("OSWT structural audit observed V3 base-state drift")
        payload: dict[str, Any] = {
            "schema": "paths-oswt-full-validation-structural-audit-v2",
            "scope": "inner_validation_only",
            "fold": self.fold,
            "split_signature": self.split_signature,
            "sample_count": samples,
            "checkpoint_epoch": int(checkpoint_epoch),
            "selected_checkpoint_sha256": selected_checkpoint_sha256,
            "paths_oswt_protocol_version": PATHS_OSWT_PROTOCOL_VERSION,
            "oswt_variant": self.oswt_variant,
            "implementation_signature": self.implementation_signature,
            "architecture_signature": self.architecture_signature,
            "source_v3_base_state_sha256": self.source_v3_base_state_sha256,
            "final_v3_base_state_sha256": final_base_hash,
            "selected_v3_base_state_sha256": selected_base_hash,
            "final_v3_base_matches_source": True,
            "selected_v3_base_matches_source": True,
            "maximum_identity_errors": maxima,
            "parameter_extrema": {
                key: aggregate[key]
                for key in (
                    "beta_minimum",
                    "beta_maximum",
                    "tau_minimum",
                    "tau_maximum",
                )
                if key in aggregate
            },
            "forward_identity_error_keys": sorted(
                key for key in maxima if key.endswith("_error")
            ),
            "all_forward_posterior_identities_audited": True,
            "deletion_replay_scope": "separate_exact_certificate_subset",
        }
        payload["content_checksum_sha256"] = _canonical_sha256(payload)
        temporary = self.structural_audit_path.with_name(
            f".{self.structural_audit_path.name}.{os.getpid()}.tmp"
        )
        try:
            with temporary.open("w", encoding="utf-8") as stream:
                json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
                stream.write("\n")
            os.replace(temporary, self.structural_audit_path)
        finally:
            if temporary.exists():
                temporary.unlink()
        return payload

    def _materialize_best_learned_checkpoint(self) -> Mapping[str, Any]:
        if not self.best_path.is_file():
            raise FileNotFoundError(f"selected OSWT checkpoint does not exist: {self.best_path}")
        state = torch.load(self.best_path, map_location="cpu", weights_only=False)
        if state.get("schema") != PATHS_OSWT_CHECKPOINT_SCHEMA:
            raise ValueError("selected OSWT checkpoint schema mismatch")
        if state.get("checkpoint_role") not in {
            "oswt_selected_learned",
            "oswt_identity_v3",
        }:
            raise ValueError("selected OSWT checkpoint has an invalid role")
        selected_model_state = state.get("model_state")
        if not isinstance(selected_model_state, Mapping):
            raise ValueError("selected OSWT checkpoint omits model_state")
        selected_base_hash = _v3_base_state_sha256(selected_model_state)
        if (
            state.get("source_v3_base_state_sha256")
            != self.source_v3_base_state_sha256
            or state.get("checkpoint_v3_base_state_sha256")
            != selected_base_hash
            or state.get("checkpoint_v3_base_matches_source") is not True
            or selected_base_hash != self.source_v3_base_state_sha256
        ):
            raise ValueError("selected OSWT checkpoint changed its V3 base")
        temporary = self.best_learned_path.with_name(
            f".{self.best_learned_path.name}.{os.getpid()}.tmp"
        )
        try:
            shutil.copyfile(self.best_path, temporary)
            os.replace(temporary, self.best_learned_path)
        finally:
            if temporary.exists():
                temporary.unlink()
        if _file_sha256(self.best_learned_path) != _file_sha256(self.best_path):
            raise AssertionError("best_learned.pth is not a byte-exact best.pth alias")
        return state

    def _fit_identity(self, *, evaluate_test: bool) -> dict[str, Any]:
        if bool(getattr(self.cfg, "resume", False)):
            raise ValueError("identity_v3 is an evaluation control and cannot resume")
        existing = [
            path
            for path in (self.best_path, self.best_learned_path, self.last_path)
            if path.exists()
        ]
        if existing:
            raise FileExistsError(
                "fresh identity_v3 run would overwrite: "
                + ", ".join(str(path) for path in existing)
            )
        control = self._evaluate_v3_control()
        metrics = self._run_epoch(self.val_loader, train=False, epoch=0)
        key = validation_selection_key(
            metrics,
            policy=self.selection_policy,
            qwk_weight=self.selection_qwk_weight,
        )
        prepared = self._save_checkpoint(
            self.best_path,
            epoch=0,
            metrics=metrics,
            best_key=key,
            best_epoch=0,
            bad_epochs=0,
            checkpoint_role="deployable_selected",
            defer_commit=True,
        )
        # Change only the role in the serialized payload before committing the
        # evaluation-only control.
        state = torch.load(prepared, map_location="cpu", weights_only=False)
        state["checkpoint_role"] = "oswt_identity_v3"
        torch.save(state, prepared)
        os.replace(prepared, self.best_path)
        self._save_checkpoint(
            self.last_path,
            epoch=0,
            metrics=metrics,
            best_key=key,
            best_epoch=0,
            bad_epochs=0,
            checkpoint_role="resume_state",
            deployable_best_checkpoint_sha256=_file_sha256(self.best_path),
        )
        certificates = self._write_validation_certificates(0)
        structural = self._evaluate_validation_structure(0)
        selected = self._materialize_best_learned_checkpoint()
        selected_model_state = selected.get("model_state")
        if not isinstance(selected_model_state, Mapping):
            raise ValueError("selected identity checkpoint omits model_state")
        selected_base_hash = _v3_base_state_sha256(selected_model_state)
        final_base_hash = _v3_base_state_sha256(self.model.state_dict())
        if not (
            selected_base_hash
            == final_base_hash
            == self.source_v3_base_state_sha256
        ):
            raise AssertionError("identity control changed the V3 base state")
        result: dict[str, Any] = {
            "best_epoch": 0,
            "best_validation": dict(metrics),
            "best_validation_metrics": dict(metrics),
            "test_evaluated": False,
            "run_dir": str(self.run_dir),
            "validation_certificate_path": str(self.certificate_path),
            "validation_certificate_checksum": certificates[
                "content_checksum_sha256"
            ],
            "checkpoint_schema": PATHS_OSWT_CHECKPOINT_SCHEMA,
            "protocol": PATHS_OSWT_PROTOCOL_VERSION,
            "oswt_variant": self.oswt_variant,
            "v3_identity_control": control,
            "validation_structural_audit": structural,
            "best_checkpoint_sha256": _file_sha256(self.best_path),
            "best_learned_checkpoint": str(self.best_learned_path),
            "best_learned_checkpoint_sha256": _file_sha256(self.best_learned_path),
            "best_learned_is_byte_exact_best_alias": True,
            "best_learned_checkpoint_role": selected["checkpoint_role"],
            "source_v3_base_state_sha256": self.source_v3_base_state_sha256,
            "final_v3_base_state_sha256": final_base_hash,
            "selected_v3_base_state_sha256": selected_base_hash,
            "final_v3_base_matches_source": True,
            "selected_v3_base_matches_source": True,
        }
        if evaluate_test:
            if self.test_loader is None:
                raise ValueError("evaluate_test=True requires a test loader")
            test_metrics = self._run_epoch(self.test_loader, train=False, epoch=0)
            result["test_evaluated"] = True
            result["test"] = test_metrics
        return result

    def fit(self, *, evaluate_test: bool = False) -> dict[str, Any]:
        if self.oswt_variant == "identity_v3":
            return self._fit_identity(evaluate_test=evaluate_test)
        if not bool(getattr(self.cfg, "resume", False)):
            if self.best_learned_path.exists():
                raise FileExistsError(
                    f"fresh OSWT run would overwrite {self.best_learned_path}"
                )
        # Reproduce the hash-bound V3 source on every invocation, including a
        # resume, before any resumed optimizer state is trusted.
        control = self._evaluate_v3_control()
        result = OriginTrainer.fit(self, evaluate_test=evaluate_test)
        structural = self._evaluate_validation_structure(int(result["best_epoch"]))
        best_learned = self._materialize_best_learned_checkpoint()
        selected_model_state = best_learned.get("model_state")
        if not isinstance(selected_model_state, Mapping):
            raise ValueError("selected OSWT checkpoint omits model_state")
        selected_base_hash = _v3_base_state_sha256(selected_model_state)
        final_base_hash = _v3_base_state_sha256(self.model.state_dict())
        final_matches_source = final_base_hash == self.source_v3_base_state_sha256
        selected_matches_source = (
            selected_base_hash == self.source_v3_base_state_sha256
        )
        if not final_matches_source or not selected_matches_source:
            raise AssertionError("OSWT training changed the hash-bound V3 base")
        result.update(
            {
                "protocol": PATHS_OSWT_PROTOCOL_VERSION,
                "oswt_variant": self.oswt_variant,
                "implementation_signature": self.implementation_signature,
                "architecture_signature": self.architecture_signature,
                "config_signature": self.config_signature,
                "critical_config": self.critical_config,
                "run_git_commit": self.run_git_commit,
                "split_signature": self.split_signature,
                "warm_start_provenance": self.paths_warm_start_provenance,
                "v3_identity_control_path": str(self.base_control_path),
                "v3_identity_control": control,
                "source_v3_base_state_sha256": self.source_v3_base_state_sha256,
                "final_v3_base_state_sha256": final_base_hash,
                "selected_v3_base_state_sha256": selected_base_hash,
                "final_v3_base_matches_source": final_matches_source,
                "selected_v3_base_matches_source": selected_matches_source,
                "training_label_counts": list(self.training_label_counts),
                "risk_set_boundary_weights": self.criterion.boundary_weights.detach()
                .cpu()
                .tolist(),
                "risk_set_weights_from_training_labels_only": True,
                "optimizer_group_contract": self.optimizer_group_contract,
                "best_learned_checkpoint": str(self.best_learned_path),
                "best_learned_checkpoint_sha256": _file_sha256(self.best_learned_path),
                "best_learned_epoch": int(best_learned["epoch"]),
                "best_learned_validation": dict(best_learned["metrics"]),
                "best_learned_checkpoint_role": best_learned["checkpoint_role"],
                "best_learned_is_byte_exact_best_alias": True,
                "validation_structural_audit_path": str(self.structural_audit_path),
                "validation_structural_audit": structural,
            }
        )
        return result


__all__ = [
    "PathsOSWTTrainer",
    "audit_paths_oswt_output",
    "paths_oswt_implementation_signature",
]
