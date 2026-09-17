"""Validation-first training and exact joint certificates for PATHS.

The implementation deliberately reuses the battle-tested ORIGIN AMP,
checkpointing, scheduler, and metric loop.  This module changes four things:

* a hash-bound, architecture-audited ORIGIN-v3 warm start is mandatory;
* the objective is the proper training-risk-set score in ``losses.paths``;
* three differential-learning-rate groups protect the audited warm start; and
* certificates replay both the V3 rate ledger and PATHS concentration ledger,
  including the signed adjacent-grade transport compiled from that ledger.

A base-only replay is never labelled a PATHS certificate.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast

from losses.paths import PathsLoss
from configs.paths_config import PATHS_PROTOCOL_VERSION
from models.origin import decode_pure_birth_rates
from models.paths import (
    apply_signed_adjacent_transport,
    continuation_logits_from_log_probs,
    decode_continuation_logits,
)
from training.origin_trainer import (
    OriginTrainer,
    _architecture_record,
    _canonical_sha256,
    _config_dict,
    _critical_config,
    _labels_from_dataset,
    _mapping_field,
    _tensor_field,
    evaluate_origin_predictions,
)


_PATHS_IMPLEMENTATION_FILES = (
    "configs/paths_config.py",
    "configs/origin_config.py",
    "Datasets/origin_data.py",
    "Datasets/mosaic_data.py",
    "Datasets/dataloaders.py",
    "models/origin_encoder.py",
    "models/origin.py",
    "models/paths.py",
    "losses/origin.py",
    "losses/paths.py",
    "training/origin_trainer.py",
    "training/paths_trainer.py",
    "train_paths.py",
    "scripts/submit_paths_preflight.sh",
    "scripts/submit_paths_aptos_f0.sh",
    "scripts/submit_paths_aptos_control_f0.sh",
    "scripts/submit_paths_aptos_gate.sh",
    "scripts/submit_paths_dr_f0.sh",
    "utils/spatial_mask.py",
)

_V3_ARCHITECTURE_IDENTITY = {
    "name": "ORIGIN",
    "encoder": "convnext_tiny",
    "evidence_scales": ["s4", "s8", "s16", "s32"],
    "atom_mode": "cumulative",
    "rate_parameterization": "bounded_null_simplex_v1",
    "decoder": "fp64_taylor24_scaled_squared_pure_birth_exponential",
    "local_rate_layout": "NHW(K-1)",
    "intervention": "stored_local_rate_subtraction_without_renormalization",
    "no_classifier_bypass": True,
}

_V3_CONFIG_IDENTITY_FIELDS = (
    "dataset",
    "n_classes",
    "n_folds",
    "val_fraction",
    "preprocessing_version",
    "seed",
    "img_size",
    "encoder",
    "pretrained",
    "evidence_scales",
    "projection_dim",
    "grad_checkpoint",
    "mask_valid_fraction",
    "reference_count",
    "atom_rate_init",
    "prior_rate_init",
    "boundary_scale_init",
    "total_rate_cap",
    "prior_rate_cap",
    "boundary_scale_cap",
    "rate_roundoff_margin",
    "atom_mode",
    "hybrid_cumulative_init",
    "evidence_dropout",
    "force_decoder_fp64",
)


def _field(output: object, name: str) -> Any:
    if isinstance(output, Mapping):
        if name not in output:
            raise KeyError(f"PATHS output has no {name!r} field")
        return output[name]
    if not hasattr(output, name):
        raise AttributeError(f"PATHS output has no {name!r} attribute")
    return getattr(output, name)


def _normalized(value: Any) -> Any:
    if isinstance(value, (list, tuple)):
        return tuple(_normalized(item) for item in value)
    if isinstance(value, np.generic):
        return value.item()
    return value


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def paths_implementation_signature() -> str:
    """Hash the complete PATHS prediction/training implementation."""

    root = Path(__file__).resolve().parents[1]
    digest = hashlib.sha256()
    for relative in _PATHS_IMPLEMENTATION_FILES:
        path = root / relative
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes() if path.is_file() else b"<missing>")
        digest.update(b"\0")
    return digest.hexdigest()


def _validate_source_config(source: Mapping[str, Any], cfg: object) -> None:
    critical = source.get("critical_config")
    if not isinstance(critical, Mapping):
        raise ValueError("ORIGIN-v3 warm start has no critical_config record")
    if source.get("config_signature") != _canonical_sha256(critical):
        raise ValueError("ORIGIN-v3 warm-start config signature is invalid")
    target = _config_dict(cfg)
    mismatches: list[str] = []
    for name in _V3_CONFIG_IDENTITY_FIELDS:
        if name not in critical:
            mismatches.append(f"{name}=<missing>")
            continue
        if _normalized(critical[name]) != _normalized(target.get(name)):
            mismatches.append(
                f"{name}: source={critical[name]!r}, target={target.get(name)!r}"
            )
    if mismatches:
        raise ValueError(
            "PATHS warm-start base configuration mismatch: " + "; ".join(mismatches)
        )


def load_audited_origin_v3_warm_start(
    model: nn.Module,
    cfg: object,
    *,
    fold: int,
    split_signature: str | None,
) -> dict[str, Any]:
    """Strict-load one immutable V3 checkpoint into the PATHS V3 subset.

    The checksum, split, architecture record, configuration record, posterior
    contract, and every source tensor key/shape are checked before mutation.
    ``PathsModel.load_origin_v3_state_dict`` is responsible for permitting
    only the newly introduced ``paths_refiner.*`` target keys to be absent.
    """

    checkpoint_value = getattr(cfg, "warm_start_checkpoint", None)
    expected_sha = str(getattr(cfg, "warm_start_sha256", "") or "").lower()
    if checkpoint_value is None:
        raise ValueError("fresh PATHS training requires an audited V3 checkpoint")
    if len(expected_sha) != 64 or any(c not in "0123456789abcdef" for c in expected_sha):
        raise ValueError("PATHS warm start requires an explicit hexadecimal SHA-256")
    checkpoint = Path(str(checkpoint_value)).expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"ORIGIN-v3 checkpoint not found: {checkpoint}")
    observed_sha = _file_sha256(checkpoint)
    if observed_sha != expected_sha:
        raise ValueError(
            "ORIGIN-v3 warm-start SHA-256 mismatch: "
            f"expected {expected_sha}, observed {observed_sha}"
        )

    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if not isinstance(state, Mapping) or state.get("schema") != "origin-checkpoint-v3":
        raise ValueError("PATHS requires an origin-checkpoint-v3 warm start")
    if int(state.get("fold", -1)) != int(fold):
        raise ValueError("ORIGIN-v3 warm-start fold mismatch")
    if split_signature is not None and state.get("split_signature") != split_signature:
        raise ValueError("ORIGIN-v3 warm-start split signature mismatch")
    implementation_signature = str(state.get("implementation_signature", "")).lower()
    if len(implementation_signature) != 64 or any(
        character not in "0123456789abcdef" for character in implementation_signature
    ):
        raise ValueError("ORIGIN-v3 source implementation signature is malformed")

    architecture = state.get("architecture")
    if not isinstance(architecture, Mapping):
        raise ValueError("ORIGIN-v3 warm start has no architecture record")
    if state.get("architecture_signature") != _canonical_sha256(architecture):
        raise ValueError("ORIGIN-v3 warm-start architecture signature is invalid")
    if architecture.get("no_classifier_bypass") is not True:
        raise ValueError("ORIGIN-v3 source permits a classifier bypass")
    if architecture.get("posterior_path") != (
        "conserved_local_rates_to_pure_birth_matrix_exponential"
    ):
        raise ValueError("ORIGIN-v3 source posterior path is incompatible")
    declared = architecture.get("declared")
    if not isinstance(declared, Mapping):
        raise ValueError("ORIGIN-v3 architecture declaration is missing")
    for name, expected in _V3_ARCHITECTURE_IDENTITY.items():
        if _normalized(declared.get(name)) != _normalized(expected):
            raise ValueError(
                f"warm-start source is not audited V3: {name}="
                f"{declared.get(name)!r}, expected {expected!r}"
            )
    _validate_source_config(state, cfg)
    if state.get("likelihood_component_unweighted") is not True:
        raise ValueError("ORIGIN-v3 warm start was not trained with unweighted likelihood")
    if state.get("population_objective_proper") is not True:
        raise ValueError("ORIGIN-v3 warm start does not declare a proper population objective")
    source_state = state.get("model_state")
    if not isinstance(source_state, Mapping):
        raise ValueError("ORIGIN-v3 checkpoint has no model_state")
    loader = getattr(model, "load_origin_v3_state_dict", None)
    if not callable(loader):
        raise TypeError("PathsModel must expose strict load_origin_v3_state_dict")
    load_record = loader(source_state)

    metrics = state.get("metrics")
    if not isinstance(metrics, Mapping):
        raise ValueError("ORIGIN-v3 checkpoint has no validation metrics")
    required_metrics = ("acc", "qwk", "mae", "balanced_acc", "macro_f1")
    source_metrics: dict[str, Any] = {}
    for name in required_metrics:
        value = float(metrics.get(name, math.nan))
        if not math.isfinite(value):
            raise ValueError(f"ORIGIN-v3 source metric {name!r} is missing/non-finite")
        source_metrics[name] = value
    confusion = metrics.get("confusion")
    if (
        not isinstance(confusion, Sequence)
        or len(confusion) != int(getattr(cfg, "n_classes"))
    ):
        raise ValueError("ORIGIN-v3 source confusion matrix is missing/malformed")
    source_metrics["confusion"] = [
        [int(value) for value in row] for row in confusion
    ]
    return {
        "schema": "paths-audited-origin-v3-warm-start-v1",
        "source_checkpoint": str(checkpoint),
        "source_checkpoint_sha256": observed_sha,
        "source_implementation_signature": implementation_signature,
        "source_architecture_signature": state["architecture_signature"],
        "source_config_signature": state["config_signature"],
        "source_epoch": int(state.get("epoch", -1)),
        "source_metrics": source_metrics,
        "load_record": load_record,
    }


def continuation_logits_from_class_probs(class_probs: torch.Tensor) -> torch.Tensor:
    """Convert a normalized ordinal posterior to continuation log-odds."""

    if class_probs.ndim != 2 or class_probs.shape[1] < 2:
        raise ValueError("class_probs must have shape (N,K), K>=2")
    if not bool(torch.isfinite(class_probs).all()) or bool((class_probs <= 0).any()):
        raise ValueError("class_probs must be finite and strictly positive")
    tail = class_probs[:, 1:].flip((1,)).cumsum(dim=1).flip((1,))
    return torch.log(tail) - torch.log(class_probs[:, :-1])


def continuation_posterior_from_logits(logits: torch.Tensor) -> torch.Tensor:
    """Replay the normalized continuation law from its conditional logits."""

    if logits.ndim != 2 or logits.shape[1] < 1:
        raise ValueError("continuation logits must have shape (N,K-1)")
    if not bool(torch.isfinite(logits).all()):
        raise ValueError("continuation logits must be finite")
    hazards = torch.sigmoid(logits)
    survival = torch.ones(
        logits.shape[0], dtype=logits.dtype, device=logits.device
    )
    probabilities: list[torch.Tensor] = []
    for boundary in range(logits.shape[1]):
        probabilities.append(survival * (1.0 - hazards[:, boundary]))
        survival = survival * hazards[:, boundary]
    probabilities.append(survival)
    return torch.stack(probabilities, dim=1)


def _decision_grade(output: object, decision_rule: str) -> torch.Tensor:
    """Return the configured deterministic grade decision."""

    if decision_rule == "class_map":
        value = _tensor_field(output, "class_map")
    elif decision_rule == "posterior_median":
        value = _tensor_field(output, "posterior_median")
    elif decision_rule == "rounded_expected":
        value = _tensor_field(output, "expected_grade").round()
    else:
        raise ValueError(f"unsupported PATHS decision rule {decision_rule!r}")
    return value.long()


def _fixed_grade_decision_margin(
    class_probs: torch.Tensor,
    cumulative_probs: torch.Tensor,
    expected_grade: torch.Tensor,
    *,
    target_grade: int,
    decision_rule: str,
    log_class_probs: torch.Tensor | None = None,
) -> torch.Tensor:
    """Signed distance from changing a *fixed* configured grade decision.

    Positive values mean that ``target_grade`` remains inside the decision
    region.  The target is deliberately held fixed under intervention: the
    certificate asks how much a cell supports the original prediction, not
    which class wins after deleting it.
    """

    if class_probs.ndim != 2 or class_probs.shape[1] < 2:
        raise ValueError("class_probs must have shape (N,K), K>=2")
    if cumulative_probs.shape != class_probs.shape[:-1] + (
        class_probs.shape[-1] - 1,
    ):
        raise ValueError("cumulative_probs shape is incompatible")
    if expected_grade.shape != class_probs.shape[:-1]:
        raise ValueError("expected_grade shape is incompatible")
    grade = int(target_grade)
    classes = int(class_probs.shape[-1])
    if not 0 <= grade < classes:
        raise ValueError("target_grade is outside the class range")
    if decision_rule == "class_map":
        if log_class_probs is None:
            tiny = torch.finfo(class_probs.dtype).tiny
            logs = class_probs.clamp_min(tiny).log()
        else:
            if log_class_probs.shape != class_probs.shape:
                raise ValueError("log_class_probs shape is incompatible")
            logs = log_class_probs
        target = logs[:, grade]
        other = logs.clone()
        other[:, grade] = -torch.inf
        return target - other.max(dim=1).values
    if decision_rule == "posterior_median":
        candidates: list[torch.Tensor] = []
        if grade > 0:
            candidates.append(cumulative_probs[:, grade - 1] - 0.5)
        if grade < classes - 1:
            candidates.append(0.5 - cumulative_probs[:, grade])
        if not candidates:  # pragma: no cover - K>=2 makes this impossible
            raise RuntimeError("median decision has no defining boundary")
        return torch.stack(candidates, dim=1).min(dim=1).values
    if decision_rule == "rounded_expected":
        # PyTorch's half-to-even convention only affects the zero-margin tie.
        return 0.5 - (expected_grade - float(grade)).abs()
    raise ValueError(f"unsupported PATHS decision rule {decision_rule!r}")


def _candidate_metrics(
    *,
    base_total_rates: torch.Tensor,
    removed_rates: torch.Tensor,
    base_boundary_concentration: torch.Tensor,
    removed_concentration: torch.Tensor,
    transport_thresholds: torch.Tensor,
    transport_slopes: torch.Tensor,
    transport_gains: torch.Tensor,
    transport_strength: float,
    target_grade: int,
    decision_rule: str,
    baseline_margin: torch.Tensor,
    baseline_target_log_prob: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Vectorized exact-ledger singleton intervention effects."""

    candidate_rates = (
        base_total_rates.unsqueeze(0).to(torch.float64)
        - removed_rates.to(torch.float64)
    )
    # Rates are sums of non-negative ledgers. A negative value indicates that
    # caller tensors are not from the same replayable computation.
    if bool((candidate_rates < 0.0).any()):
        minimum = float(candidate_rates.min().detach().cpu())
        raise FloatingPointError(
            f"singleton removal produced a negative base rate ({minimum:g})"
        )
    base_replay = decode_pure_birth_rates(candidate_rates, force_fp64=True)
    candidate_concentration = (
        base_boundary_concentration.unsqueeze(0).to(torch.float64)
        - removed_concentration.to(torch.float64)
    )
    tolerance = 128.0 * torch.finfo(candidate_concentration.dtype).eps
    if bool((candidate_concentration < -tolerance).any()) or bool(
        (candidate_concentration > 1.0 + tolerance).any()
    ):
        raise FloatingPointError(
            "singleton removal produced an invalid transport concentration"
        )
    candidate_concentration = candidate_concentration.clamp(0.0, 1.0)
    transported = apply_signed_adjacent_transport(
        base_replay.class_probs,
        candidate_concentration,
        transport_thresholds,
        transport_slopes,
        transport_gains,
        strength=float(transport_strength),
    )
    final_logits = continuation_logits_from_log_probs(
        transported.class_probs.log()
    )
    final_replay = decode_continuation_logits(final_logits)
    margin = _fixed_grade_decision_margin(
        final_replay.class_probs,
        final_replay.cumulative_probs,
        final_replay.expected_grade,
        target_grade=target_grade,
        decision_rule=decision_rule,
        log_class_probs=final_replay.log_class_probs,
    )
    target_log_prob = final_replay.log_class_probs[:, int(target_grade)]
    return baseline_margin - margin, baseline_target_log_prob - target_log_prob


def rank_paths_singleton_witnesses(
    output: object,
    *,
    decision_rule: str,
    sample_indices: Sequence[int] | None = None,
    chunk_size: int = 2048,
    positive_tolerance: float = 1e-12,
) -> list[dict[str, Any]]:
    """Rank every valid cell by exact intervention on the final predictor.

    The enumeration is vectorized in bounded chunks.  Each candidate deletes
    both its V3 transition-rate vector and its PATHS concentration vector,
    decodes the V3 CTMC, then decodes the refined continuation law.  Ranking
    is lexicographic: configured-decision margin drop, predicted-grade log
    probability drop, canonical scale order, and flat spatial index.  A cell
    is called a positive supporter only when both effects are positive.
    """

    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    if not math.isfinite(positive_tolerance) or positive_tolerance < 0.0:
        raise ValueError("positive_tolerance must be finite and non-negative")
    local_rates = _mapping_field(output, "local_rate_maps")
    local_transport = _mapping_field(output, "local_transport_maps")
    valid_masks = _mapping_field(output, "valid_masks")
    if set(local_rates) != set(local_transport) or set(local_rates) != set(
        valid_masks
    ):
        raise ValueError("PATHS rate, transport, and validity scales differ")
    # ORIGIN emits scales in the audited increasing-stride order. Preserve
    # that order; lexicographic sorting would incorrectly place s16 before s4.
    scale_order = tuple(str(name) for name in local_rates)
    class_probs = _tensor_field(output, "class_probs")
    log_class_probs = _tensor_field(output, "log_class_probs")
    cumulative = _tensor_field(output, "cumulative_probs")
    expected = _tensor_field(output, "expected_grade")
    predictions = _decision_grade(output, decision_rule)
    total_rates = _tensor_field(output, "total_rates")
    concentration = _tensor_field(output, "boundary_concentration")
    thresholds = _tensor_field(output, "transport_thresholds")
    slopes = _tensor_field(output, "transport_slopes")
    gains = _tensor_field(output, "transport_gains")
    strength = float(_field(output, "strength"))
    batch_size = int(class_probs.shape[0])
    requested = (
        tuple(range(batch_size))
        if sample_indices is None
        else tuple(int(value) for value in sample_indices)
    )
    if any(value < 0 or value >= batch_size for value in requested):
        raise IndexError("sample_indices contains an out-of-range value")

    results: list[dict[str, Any]] = []
    for sample in requested:
        target = int(predictions[sample].detach().cpu())
        base_margin = _fixed_grade_decision_margin(
            class_probs[sample : sample + 1],
            cumulative[sample : sample + 1],
            expected[sample : sample + 1],
            target_grade=target,
            decision_rule=decision_rule,
            log_class_probs=log_class_probs[sample : sample + 1],
        )[0]
        base_log_prob = log_class_probs[sample, target]
        best_positive: tuple[float, float, int, int, str, tuple[int, ...]] | None = None
        best_any: tuple[float, float, int, int, str, tuple[int, ...]] | None = None
        valid_count = 0
        for scale_rank, scale in enumerate(scale_order):
            rates = local_rates[scale][sample]
            transport = local_transport[scale][sample]
            valid = valid_masks[scale][sample].bool()
            if rates.shape[:-1] != valid.shape or transport.shape != rates.shape:
                raise ValueError(f"PATHS singleton ledger shape mismatch at {scale!r}")
            flat_valid = torch.nonzero(valid.reshape(-1), as_tuple=False).flatten()
            valid_count += int(flat_valid.numel())
            flat_rates = rates.reshape(-1, rates.shape[-1])
            flat_transport = transport.reshape(-1, transport.shape[-1])
            for start in range(0, int(flat_valid.numel()), chunk_size):
                indices = flat_valid[start : start + chunk_size]
                margin_drop, log_prob_drop = _candidate_metrics(
                    base_total_rates=total_rates[sample],
                    removed_rates=flat_rates.index_select(0, indices),
                    base_boundary_concentration=concentration[sample],
                    removed_concentration=flat_transport.index_select(0, indices),
                    transport_thresholds=thresholds,
                    transport_slopes=slopes,
                    transport_gains=gains,
                    transport_strength=strength,
                    target_grade=target,
                    decision_rule=decision_rule,
                    baseline_margin=base_margin,
                    baseline_target_log_prob=base_log_prob,
                )
                def chunk_best(mask: torch.Tensor) -> tuple[
                    float, float, int, int, str, tuple[int, ...]
                ] | None:
                    if not bool(mask.any()):
                        return None
                    masked_margin = margin_drop.masked_fill(~mask, -torch.inf)
                    maximum_margin = masked_margin.max()
                    margin_ties = mask & (margin_drop == maximum_margin)
                    masked_log = log_prob_drop.masked_fill(
                        ~margin_ties, -torch.inf
                    )
                    maximum_log = masked_log.max()
                    final_ties = margin_ties & (log_prob_drop == maximum_log)
                    # nonzero is ordered, so this implements the documented
                    # lowest-flat-index tie rule without one CPU sync/cell.
                    local_index = int(
                        torch.nonzero(final_ties, as_tuple=False)[0, 0]
                        .detach()
                        .cpu()
                    )
                    flat_index = int(indices[local_index].detach().cpu())
                    spatial = tuple(
                        int(value)
                        for value in np.unravel_index(flat_index, valid.shape)
                    )
                    margin_value = float(margin_drop[local_index].detach().cpu())
                    log_value = float(log_prob_drop[local_index].detach().cpu())
                    # Negated order terms make Python's max prefer the first
                    # canonical scale and lowest flat index on exact ties.
                    candidate = (
                        margin_value,
                        log_value,
                        -scale_rank,
                        -flat_index,
                        scale,
                        spatial,
                    )
                    return candidate

                candidate_any = chunk_best(torch.ones_like(margin_drop, dtype=torch.bool))
                if candidate_any is not None and (
                    best_any is None or candidate_any[:4] > best_any[:4]
                ):
                    best_any = candidate_any
                candidate_positive = chunk_best(
                    (margin_drop > positive_tolerance)
                    & (log_prob_drop > positive_tolerance)
                )
                if candidate_positive is not None and (
                    best_positive is None
                    or candidate_positive[:4] > best_positive[:4]
                ):
                    best_positive = candidate_positive
        if valid_count == 0 or best_any is None:
            raise ValueError("PATHS certificate sample has no valid evidence cell")

        def encode(
            candidate: tuple[float, float, int, int, str, tuple[int, ...]] | None,
        ) -> dict[str, Any] | None:
            if candidate is None:
                return None
            margin_value, log_value, _, negative_flat, scale, spatial = candidate
            return {
                "scale": scale,
                "spatial_index": list(spatial),
                "flat_index": -negative_flat,
                "configured_decision_margin_drop": margin_value,
                "predicted_grade_log_probability_drop": log_value,
            }

        results.append(
            {
                "sample_index": sample,
                "target_grade": target,
                "valid_cell_count": valid_count,
                "supporter_found": best_positive is not None,
                "selected": encode(best_positive),
                "strongest_candidate_regardless_of_sign": encode(best_any),
            }
        )
    return results


def target_log_probability_boundary_contributions(
    baseline_logits: torch.Tensor,
    replayed_logits: torch.Tensor,
    *,
    target_grade: int,
) -> torch.Tensor:
    """Decompose ``log p_g - log p'_g`` into continuation boundaries."""

    if baseline_logits.ndim != 1 or replayed_logits.shape != baseline_logits.shape:
        raise ValueError("boundary logits must be matching one-dimensional tensors")
    boundaries = int(baseline_logits.numel())
    grade = int(target_grade)
    if not 0 <= grade <= boundaries:
        raise ValueError("target_grade is outside the class range")
    base = baseline_logits.to(torch.float64)
    replayed = replayed_logits.to(torch.float64)
    contributions = torch.zeros_like(base)
    if grade > 0:
        contributions[:grade] = (
            F.logsigmoid(base[:grade]) - F.logsigmoid(replayed[:grade])
        )
    if grade < boundaries:
        contributions[grade] = (
            F.logsigmoid(-base[grade]) - F.logsigmoid(-replayed[grade])
        )
    return contributions


def _deterministic_matched_random_flat_index(
    valid_mask: torch.Tensor,
    *,
    excluded_flat_index: int,
    token: str,
) -> int | None:
    """Choose a reproducible valid control cell at the selected scale."""

    valid = torch.nonzero(valid_mask.reshape(-1).bool(), as_tuple=False).flatten()
    valid = valid[valid != int(excluded_flat_index)]
    if valid.numel() == 0:
        return None
    digest = hashlib.sha256(token.encode("utf-8")).digest()
    offset = int.from_bytes(digest[:8], byteorder="big") % int(valid.numel())
    return int(valid[offset].detach().cpu())


def _receptive_field_geometry(metadata: object, spatial: Sequence[int]) -> dict[str, Any]:
    """Report the theoretical dependency box, clipped to the input canvas."""

    if len(spatial) != 2:
        return {
            "input_center_yx": None,
            "theoretical_receptive_field_box_yxyx": None,
            "clipped_receptive_field_box_yxyx": None,
            "touches_padding": None,
            "covers_entire_input": None,
            "global_support": bool(getattr(metadata, "globally_mixed", False)),
        }
    stride = getattr(metadata, "output_stride", None)
    receptive_field = getattr(metadata, "receptive_field", None)
    center_offset = getattr(metadata, "center_offset", None)
    input_size = getattr(metadata, "input_size", None)
    if (
        stride is None
        or receptive_field is None
        or center_offset is None
        or input_size is None
        or len(input_size) != 2
    ):
        return {
            "input_center_yx": None,
            "theoretical_receptive_field_box_yxyx": None,
            "clipped_receptive_field_box_yxyx": None,
            "touches_padding": None,
            "covers_entire_input": None,
            "global_support": bool(getattr(metadata, "globally_mixed", False)),
        }
    center_y = float(center_offset + int(spatial[0]) * int(stride))
    center_x = float(center_offset + int(spatial[1]) * int(stride))
    half = float(receptive_field) / 2.0
    raw = [center_y - half, center_x - half, center_y + half, center_x + half]
    height, width = (float(input_size[0]), float(input_size[1]))
    clipped = [
        max(0.0, raw[0]),
        max(0.0, raw[1]),
        min(height, raw[2]),
        min(width, raw[3]),
    ]
    touches_padding = raw != clipped
    covers_entire = (
        clipped[0] <= 0.0
        and clipped[1] <= 0.0
        and clipped[2] >= height
        and clipped[3] >= width
    )
    globally_mixed = bool(getattr(metadata, "globally_mixed", False))
    return {
        "input_center_yx": [center_y, center_x],
        "theoretical_receptive_field_box_yxyx": raw,
        "clipped_receptive_field_box_yxyx": clipped,
        "touches_padding": touches_padding,
        "covers_entire_input": covers_entire,
        "global_support": globally_mixed or covers_entire,
    }


def audit_paths_joint_replay(
    baseline: object,
    intervention: object,
) -> dict[str, float]:
    """Audit V3 rate, focality ledger, transport, and posterior identities."""

    replayed = _field(intervention, "output")
    removed_rates = _field(intervention, "removed_rates")
    removed_concentration = _field(
        intervention, "removed_boundary_concentration"
    )
    removed_flow = _field(intervention, "removed_net_boundary_flow")
    removed_correction = _field(intervention, "removed_boundary_correction")
    base_rates = _tensor_field(baseline, "total_rates")
    replay_rates = _tensor_field(replayed, "total_rates")
    base_correction = _tensor_field(baseline, "boundary_correction")
    replay_correction = _tensor_field(replayed, "boundary_correction")
    base_concentration = _tensor_field(baseline, "boundary_concentration")
    replay_concentration = _tensor_field(replayed, "boundary_concentration")
    base_transport = _tensor_field(baseline, "transport_matrix")
    replay_transport = _tensor_field(replayed, "transport_matrix")
    base_flow = _tensor_field(baseline, "net_boundary_flow")
    replay_flow = _tensor_field(replayed, "net_boundary_flow")
    # Use the stored FP64 base logits rather than reconstructing them from
    # probability-space tails, which can underflow for rare grades.
    base_logits = _tensor_field(baseline, "base_continuation_logits")
    replay_base_logits = _tensor_field(replayed, "base_continuation_logits")
    final_logits = _tensor_field(baseline, "continuation_logits")
    replay_logits = _tensor_field(replayed, "continuation_logits")

    expected_final = base_logits + base_correction.to(base_logits.dtype)
    expected_replay_final = replay_base_logits + replay_correction.to(
        replay_base_logits.dtype
    )
    posterior = continuation_posterior_from_logits(final_logits)
    replay_posterior = continuation_posterior_from_logits(replay_logits)
    base_v3_probs = _tensor_field(_field(baseline, "base_output"), "class_probs")
    replay_v3_probs = _tensor_field(_field(replayed, "base_output"), "class_probs")
    transported = torch.matmul(
        base_v3_probs.to(torch.float64).unsqueeze(-2), base_transport
    ).squeeze(-2)
    replay_transported = torch.matmul(
        replay_v3_probs.to(torch.float64).unsqueeze(-2), replay_transport
    ).squeeze(-2)
    row_sum_error = max(
        float((base_transport.sum(dim=-1) - 1.0).abs().max().detach().cpu()),
        float((replay_transport.sum(dim=-1) - 1.0).abs().max().detach().cpu()),
    )
    grade_axis = torch.arange(
        base_transport.shape[-1], device=base_transport.device
    )
    non_adjacent = (grade_axis[:, None] - grade_axis[None, :]).abs() > 1
    if bool(non_adjacent.any()):
        non_adjacent_max = max(
            float(base_transport[..., non_adjacent].abs().max().detach().cpu()),
            float(replay_transport[..., non_adjacent].abs().max().detach().cpu()),
        )
    else:
        non_adjacent_max = 0.0

    def _flow_replay_error(
        source: torch.Tensor,
        transported_probs: torch.Tensor,
        flow: torch.Tensor,
    ) -> float:
        delta = torch.cat(
            (
                -flow[..., :1],
                flow[..., :-1] - flow[..., 1:],
                flow[..., -1:],
            ),
            dim=-1,
        )
        return float(
            (transported_probs - source.to(torch.float64) - delta)
            .abs()
            .max()
            .detach()
            .cpu()
        )
    return {
        "rate_replay_error": float(
            (base_rates - removed_rates - replay_rates).abs().max().detach().cpu()
        ),
        "correction_replay_error": float(
            (base_correction - removed_correction - replay_correction)
            .abs()
            .max()
            .detach()
            .cpu()
        ),
        "concentration_replay_error": float(
            (
                base_concentration
                - removed_concentration
                - replay_concentration
            )
            .abs()
            .max()
            .detach()
            .cpu()
        ),
        "baseline_joint_logit_error": float(
            (final_logits - expected_final).abs().max().detach().cpu()
        ),
        "replayed_joint_logit_error": float(
            (replay_logits - expected_replay_final).abs().max().detach().cpu()
        ),
        "baseline_posterior_replay_error": float(
            (posterior - _tensor_field(baseline, "class_probs"))
            .abs()
            .max()
            .detach()
            .cpu()
        ),
        "replayed_posterior_replay_error": float(
            (replay_posterior - _tensor_field(replayed, "class_probs"))
            .abs()
            .max()
            .detach()
            .cpu()
        ),
        "transport_matrix_row_sum_error": row_sum_error,
        "transport_matrix_minimum": float(
            torch.minimum(base_transport.min(), replay_transport.min())
            .detach()
            .cpu()
        ),
        "transport_non_adjacent_max_abs": non_adjacent_max,
        "baseline_transport_posterior_error": float(
            (transported - _tensor_field(baseline, "class_probs"))
            .abs()
            .max()
            .detach()
            .cpu()
        ),
        "replayed_transport_posterior_error": float(
            (replay_transported - _tensor_field(replayed, "class_probs"))
            .abs()
            .max()
            .detach()
            .cpu()
        ),
        "transport_effect_max_abs": float(
            (
                _tensor_field(baseline, "class_probs")
                - base_v3_probs
            )
            .abs()
            .max()
            .detach()
            .cpu()
        ),
        "net_boundary_flow_min": float(base_flow.min().detach().cpu()),
        "net_boundary_flow_max": float(base_flow.max().detach().cpu()),
        "baseline_flow_reconstruction_error": _flow_replay_error(
            base_v3_probs,
            _tensor_field(baseline, "class_probs"),
            base_flow,
        ),
        "replayed_flow_reconstruction_error": _flow_replay_error(
            replay_v3_probs,
            _tensor_field(replayed, "class_probs"),
            replay_flow,
        ),
        "net_boundary_flow_replay_error": float(
            (base_flow - removed_flow - replay_flow)
            .abs()
            .max()
            .detach()
            .cpu()
        ),
        "removed_net_boundary_flow_max_abs": float(
            removed_flow.abs().max().detach().cpu()
        ),
        "final_correction_max_abs": float(base_correction.abs().max().detach().cpu()),
        "removed_correction_max_abs": float(
            removed_correction.abs().max().detach().cpu()
        ),
        "joint_probability_effect_max_abs": float(
            (
                _tensor_field(baseline, "class_probs")
                - _tensor_field(replayed, "class_probs")
            )
            .abs()
            .max()
            .detach()
            .cpu()
        ),
    }


class PathsTrainer(OriginTrainer):
    """ORIGIN training mechanics with a PATHS objective and certificate."""

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
        paths_warm_start_provenance = load_audited_origin_v3_warm_start(
            model,
            cfg,
            fold=fold,
            split_signature=split_signature,
        )
        # PATHS owns and has already completed its strict V3 migration above.
        # Later ORIGIN branches also inspect these generically named fields to
        # trigger a relation-only migration from ``OriginTrainer.__init__``.
        # Hide them only for the parent constructor so an inherited extension
        # cannot load the same checkpoint a second time under the wrong state
        # contract.  ``self.cfg`` retains this same object, so restoring in the
        # finally block also restores the complete PATHS experiment identity.
        warm_start_checkpoint = getattr(cfg, "warm_start_checkpoint", None)
        warm_start_sha256 = getattr(cfg, "warm_start_sha256", None)
        setattr(cfg, "warm_start_checkpoint", None)
        setattr(cfg, "warm_start_sha256", None)
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
            setattr(cfg, "warm_start_checkpoint", warm_start_checkpoint)
            setattr(cfg, "warm_start_sha256", warm_start_sha256)
        # Keep the inherited extension hook disabled for its entire lifetime:
        # newer OriginTrainer implementations use this same-named field to
        # activate relation-specific checkpoint selection. PATHS owns a
        # distinct provenance field and a distinct checkpoint contract.
        self.warm_start_provenance = None
        self.paths_warm_start_provenance = paths_warm_start_provenance
        self._install_paths_optimizer(cfg)
        labels = _labels_from_dataset(train_loader.dataset)
        if labels is None or len(labels) != len(train_loader.dataset):
            raise ValueError(
                "PATHS requires all training-fold labels to fix risk-set weights"
            )
        counts = torch.bincount(
            torch.as_tensor(labels, dtype=torch.long), minlength=self.num_classes
        )
        if counts.numel() != self.num_classes or bool((counts <= 0).any()):
            raise ValueError("every grade must be represented in the PATHS training fold")
        self.training_label_counts = counts.tolist()
        self.criterion = PathsLoss(
            self.num_classes,
            label_counts=self.training_label_counts,
            risk_set_power=float(getattr(cfg, "risk_set_alpha", 0.5)),
            rps_weight=float(getattr(cfg, "rps_weight", 0.25)),
        ).to(self.device)
        self.population_objective_proper = (
            self.criterion.configured_objective_is_proper
            and not self.stratified_batches
        )
        self.correction_only_epochs = int(
            getattr(cfg, "correction_only_epochs", 0)
        )
        self.paths_variant = str(getattr(cfg, "paths_variant", "signed_transport"))
        if self.paths_variant not in {"signed_transport", "risk_objective_v3"}:
            raise ValueError(f"unsupported PATHS variant {self.paths_variant!r}")
        self.implementation_signature = paths_implementation_signature()
        self.architecture = _architecture_record(self.model)
        self.architecture_signature = _canonical_sha256(self.architecture)
        self.critical_config = _critical_config(cfg)
        self.config_signature = _canonical_sha256(self.critical_config)
        self.run_git_commit = os.environ.get("PATHS_RUN_GIT_COMMIT")
        if self.run_git_commit is not None:
            commit = self.run_git_commit.lower()
            if len(commit) not in {40, 64} or any(
                character not in "0123456789abcdef" for character in commit
            ):
                raise ValueError("PATHS_RUN_GIT_COMMIT must be a hexadecimal commit id")
            self.run_git_commit = commit
        self.checkpoint_schema = "paths-checkpoint-v3-sapt"
        self.base_control_path = self.run_dir / "v3_strength_zero_control.json"
        self.best_learned_path = self.run_dir / "best_learned.pth"

    def _install_paths_optimizer(self, cfg: object) -> None:
        """Separate representation, V3 generator, and PATHS adaptation rates.

        The parent trainer's two-group optimizer is intentionally replaced
        before the first update.  This prevents a 5e-4 refiner learning rate
        from also being applied to every warm-started V3 generator tensor.
        """

        encoder = getattr(self.model, "encoder", None)
        generator = getattr(self.model, "generator", None)
        refiner = getattr(self.model, "paths_refiner", None)
        if not all(isinstance(module, nn.Module) for module in (encoder, generator, refiner)):
            raise TypeError("PATHS requires encoder, generator, and paths_refiner modules")
        encoder_parameters = list(encoder.parameters())
        base_parameters = list(generator.parameters())
        refiner_parameters = list(refiner.parameters())
        encoder_ids = {id(parameter) for parameter in encoder_parameters}
        base_ids = {id(parameter) for parameter in base_parameters}
        refiner_ids = {id(parameter) for parameter in refiner_parameters}
        if encoder_ids & base_ids or encoder_ids & refiner_ids or base_ids & refiner_ids:
            raise RuntimeError("PATHS optimizer groups overlap")
        groups = (
            (
                "encoder",
                encoder_parameters,
                float(getattr(cfg, "paths_encoder_lr", 1e-5)),
            ),
            (
                "origin_v3_generator",
                base_parameters,
                float(getattr(cfg, "paths_base_lr", 5e-5)),
            ),
            (
                "paths_refiner",
                refiner_parameters,
                float(getattr(cfg, "paths_refiner_lr", 5e-4)),
            ),
        )
        if any(not parameters for _, parameters, _ in groups):
            raise ValueError("every PATHS optimizer group must be non-empty")
        grouped_ids = {
            id(parameter)
            for _, parameters, _ in groups
            for parameter in parameters
        }
        model_ids = {id(parameter) for parameter in self.model.parameters()}
        if grouped_ids != model_ids:
            raise RuntimeError("PATHS optimizer groups do not partition the model")
        self.optimizer_group_contract = [
            {
                "name": name,
                "initial_lr": learning_rate,
                "parameter_count": sum(p.numel() for p in parameters),
            }
            for name, parameters, learning_rate in groups
        ]
        self.optimizer = torch.optim.AdamW(
            [
                {"params": parameters, "lr": learning_rate, "name": name}
                for name, parameters, learning_rate in groups
            ],
            weight_decay=float(getattr(cfg, "weight_decay", 1e-5)),
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            factor=float(getattr(cfg, "lr_factor", 0.2)),
            patience=int(getattr(cfg, "lr_patience", 5)),
            min_lr=float(getattr(cfg, "lr_min", 1e-6)),
        )

    def _set_encoder_trainable(self, epoch: int) -> None:
        correction_only = epoch <= self.correction_only_epochs
        for name, parameter in self.model.named_parameters():
            if name.startswith("paths_refiner."):
                parameter.requires_grad_(self.paths_variant == "signed_transport")
            else:
                parameter.requires_grad_(not correction_only)
        encoder = getattr(self.model, "encoder", None)
        if isinstance(encoder, nn.Module):
            encoder_trainable = (
                not correction_only and epoch > self.encoder_freeze_epochs
            )
            for parameter in encoder.parameters():
                parameter.requires_grad_(encoder_trainable)

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
            raise ValueError("PATHS does not accept an inherited candidate metric floor")
        if (
            best_learned_key is not None
            or best_learned_epoch is not None
            or learned_bad_epochs != 0
            or best_learned_checkpoint_sha256 is not None
        ):
            raise ValueError("PATHS uses one learned checkpoint-selection track")
        payload = super()._checkpoint_payload(
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
        selected_role = payload.get("checkpoint_role")
        if selected_role == "deployable_selected":
            selected_role = "paths_selected_learned"
        payload.update(
            {
                "schema": self.checkpoint_schema,
                "paths_protocol_version": PATHS_PROTOCOL_VERSION,
                "paths_variant": self.paths_variant,
                "run_git_commit": self.run_git_commit,
                "checkpoint_role": selected_role,
                "warm_start_provenance": None,
                "paths_warm_start_provenance": self.paths_warm_start_provenance,
                "training_label_counts": list(self.training_label_counts),
                "risk_set_boundary_weights": (
                    self.criterion.boundary_weights.detach().cpu().tolist()
                ),
                "risk_set_weights_from_training_labels_only": True,
                "correction_only_epochs": self.correction_only_epochs,
                "optimizer_group_contract": self.optimizer_group_contract,
                "training_phase": (
                    "paths_refiner_only"
                    if 1 <= int(epoch) <= self.correction_only_epochs
                    else "paths_joint"
                ),
            }
        )
        return payload

    def _materialize_best_learned_checkpoint(self) -> Mapping[str, Any]:
        """Atomically mirror the committed selected PATHS checkpoint.

        This is deliberately performed only after the parent's transactional
        training loop returns. During ``_save_checkpoint`` the selected
        sidecar may still be an uncommitted temporary file.
        """

        if not self.best_path.is_file():
            raise FileNotFoundError(
                f"selected PATHS checkpoint does not exist: {self.best_path}"
            )
        state = torch.load(self.best_path, map_location="cpu", weights_only=False)
        if state.get("schema") != self.checkpoint_schema:
            raise ValueError("selected PATHS checkpoint schema mismatch")
        if state.get("checkpoint_role") != "paths_selected_learned":
            raise ValueError(
                "selected PATHS checkpoint does not declare its learned role"
            )
        temporary = self.best_learned_path.with_name(
            f".{self.best_learned_path.name}.{os.getpid()}.tmp"
        )
        try:
            # PATHS has one learned selection track.  Keep a separately named
            # artifact for paired-control tooling, but make it a byte-exact
            # alias rather than mutating and reserializing checkpoint metadata.
            shutil.copyfile(self.best_path, temporary)
            os.replace(temporary, self.best_learned_path)
        finally:
            if temporary.exists():
                temporary.unlink()
        if _file_sha256(self.best_learned_path) != _file_sha256(self.best_path):
            raise AssertionError("best_learned.pth is not a byte-exact best.pth alias")
        return state

    def _validate_resume(
        self,
        state: Mapping[str, Any],
        *,
        recover_transaction: bool = False,
    ) -> None:
        # Authenticate PATHS-owned fields before allowing the parent to act on
        # a recorded filesystem transaction.
        if state.get("schema") != self.checkpoint_schema:
            raise ValueError("PATHS resume requires a paths-checkpoint-v3-sapt checkpoint")
        if state.get("paths_protocol_version") != PATHS_PROTOCOL_VERSION:
            raise ValueError("PATHS resume protocol mismatch")
        if state.get("paths_variant") != self.paths_variant:
            raise ValueError("PATHS resume variant mismatch")
        if state.get("run_git_commit") != self.run_git_commit:
            raise ValueError("PATHS resume git-commit identity mismatch")
        if state.get("warm_start_provenance") is not None:
            raise ValueError("PATHS resume unexpectedly activates the ORIGIN relation hook")
        if state.get("paths_warm_start_provenance") != self.paths_warm_start_provenance:
            raise ValueError("PATHS resume warm-start provenance mismatch")
        if state.get("training_label_counts") != self.training_label_counts:
            raise ValueError("PATHS resume training-label counts mismatch")
        if state.get("optimizer_group_contract") != self.optimizer_group_contract:
            raise ValueError("PATHS resume optimizer-group contract mismatch")
        if state.get("candidate_metric_safety_floor_evaluation") is not None:
            raise ValueError("PATHS resume contains an inherited candidate metric floor")
        if state.get("best_learned_selection_key") is not None or state.get(
            "best_learned_epoch"
        ) is not None:
            raise ValueError("PATHS resume contains an inherited learned checkpoint track")

        # The parent owns generic identity/role checks, transaction recovery,
        # and hash-binding of the committed selected checkpoint.
        super()._validate_resume(
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
                base = (
                    _field(output, "base_output")
                    if hasattr(output, "base_output")
                    or (isinstance(output, Mapping) and "base_output" in output)
                    else output
                )
                probabilities.append(_tensor_field(base, "class_probs").float().cpu())
                if self.decision_rule == "rounded_expected":
                    base_prediction = _tensor_field(base, "expected_grade").round()
                else:
                    base_prediction = _tensor_field(base, self.decision_rule)
                predictions.append(base_prediction.long().cpu())
                labels_all.append(labels.cpu())
                if indices is None:
                    ordered_ids.extend(
                        str(value)
                        for value in range(
                            running_index,
                            running_index + int(labels.numel()),
                        )
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
        scalar_names = ("acc", "qwk", "mae", "balanced_acc", "macro_f1")
        errors = {
            name: abs(float(metrics[name]) - float(source[name]))
            for name in scalar_names
        }
        if any(value > tolerance for value in errors.values()):
            raise AssertionError(
                "hash-bound V3 warm start did not reproduce source validation metrics: "
                f"errors={errors}, tolerance={tolerance}"
            )
        observed_confusion = [
            [int(value) for value in row] for row in metrics["confusion"]
        ]
        if observed_confusion != source["confusion"]:
            raise AssertionError(
                "hash-bound V3 warm start did not reproduce source confusion matrix"
            )
        identity_payload = {
            "ordered_ids": ordered_ids,
            "labels": torch.cat(labels_all).tolist(),
            "predictions": torch.cat(predictions).tolist(),
        }
        payload = {
            "schema": "paths-v3-strength-zero-control-v2",
            "scope": "inner_validation_only",
            "fold": self.fold,
            "split_signature": self.split_signature,
            "source_checkpoint_sha256": self.paths_warm_start_provenance[
                "source_checkpoint_sha256"
            ],
            "paths_protocol_version": PATHS_PROTOCOL_VERSION,
            "paths_variant": self.paths_variant,
            "implementation_signature": self.implementation_signature,
            "architecture_signature": self.architecture_signature,
            "metrics": metrics,
            "ordered_prediction_checksum_sha256": _canonical_sha256(identity_payload),
            "source_metric_absolute_errors": errors,
            "reproduction_tolerance": tolerance,
            "reproduced": True,
        }
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

    def _write_validation_certificates(self, checkpoint_epoch: int) -> dict[str, Any]:
        if getattr(self, "paths_variant", "signed_transport") == "risk_objective_v3":
            # The matched control intentionally has no PATHS transport.  Use
            # the inherited exact V3 rate-ledger certificate and label it as
            # such instead of fabricating a nonzero transport witness.
            return OriginTrainer._write_validation_certificates(
                self, checkpoint_epoch
            )
        replay = getattr(self.model, "replay_without", None)
        if not callable(replay):
            raise TypeError("PATHS model must expose replay_without")
        self.model.eval()
        batch = next(iter(self.val_loader))
        images, pixel_mask, labels, indices = self._unpack_batch(batch)
        requested = min(
            int(getattr(self.cfg, "certificate_samples", 8)), int(labels.numel())
        )
        with torch.no_grad():
            with autocast(device_type="cuda", enabled=self.use_amp):
                baseline = self._forward(images, pixel_mask)
            valid_masks = _mapping_field(baseline, "valid_masks")
            rankings = rank_paths_singleton_witnesses(
                baseline,
                decision_rule=self.decision_rule,
                sample_indices=range(requested),
            )
            removal_masks = {
                scale: torch.zeros_like(mask, dtype=torch.bool)
                for scale, mask in valid_masks.items()
            }
            random_masks = {
                scale: torch.zeros_like(mask, dtype=torch.bool)
                for scale, mask in valid_masks.items()
            }
            random_locations: dict[int, tuple[str, tuple[int, ...], int]] = {}
            for ranking in rankings:
                sample = int(ranking["sample_index"])
                selected = ranking["selected"]
                if selected is None:
                    continue
                scale = str(selected["scale"])
                spatial = tuple(int(value) for value in selected["spatial_index"])
                flat = int(selected["flat_index"])
                removal_masks[scale][(sample, *spatial)] = True
                sample_id = self._index_value(indices, sample)
                random_flat = _deterministic_matched_random_flat_index(
                    valid_masks[scale][sample],
                    excluded_flat_index=flat,
                    token=f"paths-v3-sapt|{sample_id!r}|{scale}",
                )
                if random_flat is not None:
                    random_spatial = tuple(
                        int(value)
                        for value in np.unravel_index(
                            random_flat, valid_masks[scale][sample].shape
                        )
                    )
                    random_masks[scale][(sample, *random_spatial)] = True
                    random_locations[sample] = (
                        scale,
                        random_spatial,
                        random_flat,
                    )

            intervention = replay(baseline, removal_masks, force_decoder_fp64=True)
            replayed = _field(intervention, "output")
            audit = audit_paths_joint_replay(baseline, intervention)
            random_intervention = replay(
                baseline, random_masks, force_decoder_fp64=True
            )
            random_replayed = _field(random_intervention, "output")
            random_audit = audit_paths_joint_replay(baseline, random_intervention)
            configured = float(
                getattr(self.cfg, "certificate_replay_tolerance", 2e-5)
            )
            scale = max(
                1.0,
                float(_tensor_field(baseline, "total_rates").abs().max().cpu()),
                float(_tensor_field(baseline, "boundary_correction").abs().max().cpu()),
            )
            tolerance = max(
                configured,
                32.0 * torch.finfo(torch.float32).eps * scale,
            )
            identity_keys = (
                "rate_replay_error",
                "concentration_replay_error",
                "correction_replay_error",
                "baseline_joint_logit_error",
                "replayed_joint_logit_error",
                "baseline_posterior_replay_error",
                "replayed_posterior_replay_error",
                "transport_matrix_row_sum_error",
                "transport_non_adjacent_max_abs",
                "baseline_transport_posterior_error",
                "replayed_transport_posterior_error",
                "baseline_flow_reconstruction_error",
                "replayed_flow_reconstruction_error",
                "net_boundary_flow_replay_error",
            )
            failures = {
                key: audit[key] for key in identity_keys if audit[key] > tolerance
            }
            if failures:
                raise AssertionError(
                    f"PATHS joint replay identity failed: {failures}, tolerance={tolerance}"
                )
            random_failures = {
                key: random_audit[key]
                for key in identity_keys
                if random_audit[key] > tolerance
            }
            if random_failures:
                raise AssertionError(
                    "PATHS matched-random replay identity failed: "
                    f"{random_failures}, tolerance={tolerance}"
                )
            if min(
                audit["transport_matrix_minimum"],
                random_audit["transport_matrix_minimum"],
            ) < -tolerance:
                raise AssertionError("PATHS transport matrix contains negative mass")
            if audit["transport_effect_max_abs"] <= 1e-10:
                raise AssertionError(
                    "selected checkpoint has zero signed transport effect; "
                    "refusing a base-only certificate"
                )

            base_probs = _tensor_field(baseline, "class_probs").detach().cpu()
            replay_probs = _tensor_field(replayed, "class_probs").detach().cpu()
            random_probs = _tensor_field(
                random_replayed, "class_probs"
            ).detach().cpu()
            base_log_probs = _tensor_field(
                baseline, "log_class_probs"
            ).detach().cpu()
            replay_log_probs = _tensor_field(
                replayed, "log_class_probs"
            ).detach().cpu()
            random_log_probs = _tensor_field(
                random_replayed, "log_class_probs"
            ).detach().cpu()
            base_cumulative = _tensor_field(
                baseline, "cumulative_probs"
            ).detach().cpu()
            replay_cumulative = _tensor_field(
                replayed, "cumulative_probs"
            ).detach().cpu()
            random_cumulative = _tensor_field(
                random_replayed, "cumulative_probs"
            ).detach().cpu()
            base_expected = _tensor_field(baseline, "expected_grade").detach().cpu()
            replay_expected = _tensor_field(replayed, "expected_grade").detach().cpu()
            random_expected = _tensor_field(
                random_replayed, "expected_grade"
            ).detach().cpu()
            base_predictions = self._predictions(baseline).detach().cpu()
            replay_predictions = self._predictions(replayed).detach().cpu()
            random_predictions = self._predictions(random_replayed).detach().cpu()
            removed_rates = _field(intervention, "removed_rates").detach().cpu()
            removed_concentration = _field(
                intervention, "removed_boundary_concentration"
            ).detach().cpu()
            removed_flow = _field(
                intervention, "removed_net_boundary_flow"
            ).detach().cpu()
            removed_correction = _field(
                intervention, "removed_boundary_correction"
            ).detach().cpu()
            base_concentration = _tensor_field(
                baseline, "boundary_concentration"
            ).detach().cpu()
            replay_concentration = _tensor_field(
                replayed, "boundary_concentration"
            ).detach().cpu()
            base_direction = _tensor_field(
                baseline, "transport_direction"
            ).detach().cpu()
            replay_direction = _tensor_field(
                replayed, "transport_direction"
            ).detach().cpu()
            base_transport = _tensor_field(
                baseline, "transport_matrix"
            ).detach().cpu()
            replay_transport = _tensor_field(
                replayed, "transport_matrix"
            ).detach().cpu()
            base_flow = _tensor_field(
                baseline, "net_boundary_flow"
            ).detach().cpu()
            replay_flow = _tensor_field(
                replayed, "net_boundary_flow"
            ).detach().cpu()
            base_base_logits = _tensor_field(
                baseline, "base_continuation_logits"
            ).detach().cpu()
            replay_base_logits = _tensor_field(
                replayed, "base_continuation_logits"
            ).detach().cpu()
            base_correction = _tensor_field(
                baseline, "boundary_correction"
            ).detach().cpu()
            replay_correction = _tensor_field(
                replayed, "boundary_correction"
            ).detach().cpu()
            base_final_logits = _tensor_field(
                baseline, "continuation_logits"
            ).detach().cpu()
            replay_final_logits = _tensor_field(
                replayed, "continuation_logits"
            ).detach().cpu()
            metadata = _field(baseline, "metadata")
            certificates: list[dict[str, Any]] = []
            ranking_replay_error = 0.0
            top_margin_effects: list[float] = []
            random_margin_effects: list[float] = []
            top_log_effects: list[float] = []
            random_log_effects: list[float] = []
            paired_top_margin_effects: list[float] = []
            paired_top_log_effects: list[float] = []
            for ranking in rankings:
                sample = int(ranking["sample_index"])
                target = int(ranking["target_grade"])
                base_margin = float(
                    _fixed_grade_decision_margin(
                        base_probs[sample : sample + 1],
                        base_cumulative[sample : sample + 1],
                        base_expected[sample : sample + 1],
                        target_grade=target,
                        decision_rule=self.decision_rule,
                        log_class_probs=base_log_probs[sample : sample + 1],
                    )[0]
                )
                replay_margin = float(
                    _fixed_grade_decision_margin(
                        replay_probs[sample : sample + 1],
                        replay_cumulative[sample : sample + 1],
                        replay_expected[sample : sample + 1],
                        target_grade=target,
                        decision_rule=self.decision_rule,
                        log_class_probs=replay_log_probs[sample : sample + 1],
                    )[0]
                )
                top_margin_drop = base_margin - replay_margin
                top_log_drop = float(
                    base_log_probs[sample, target]
                    - replay_log_probs[sample, target]
                )
                selected = ranking["selected"]
                selected_payload: dict[str, Any] | None = None
                if selected is not None:
                    ranking_replay_error = max(
                        ranking_replay_error,
                        abs(
                            float(selected["configured_decision_margin_drop"])
                            - top_margin_drop
                        ),
                        abs(
                            float(selected["predicted_grade_log_probability_drop"])
                            - top_log_drop
                        ),
                    )
                    scale_name = str(selected["scale"])
                    spatial = tuple(
                        int(value) for value in selected["spatial_index"]
                    )
                    scale_metadata = metadata.get(scale_name)
                    geometry = _receptive_field_geometry(scale_metadata, spatial)
                    base_logit_delta = (
                        base_base_logits[sample] - replay_base_logits[sample]
                    )
                    correction_delta = (
                        base_correction[sample] - replay_correction[sample]
                    )
                    final_logit_delta = (
                        base_final_logits[sample] - replay_final_logits[sample]
                    )
                    contributions = target_log_probability_boundary_contributions(
                        base_final_logits[sample],
                        replay_final_logits[sample],
                        target_grade=target,
                    )
                    contribution_error = abs(
                        float(contributions.sum()) - top_log_drop
                    )
                    if contribution_error > tolerance:
                        raise AssertionError(
                            "PATHS target-log-probability boundary decomposition "
                            f"failed by {contribution_error:g}"
                        )
                    selected_payload = {
                        "scale": scale_name,
                        "spatial_index": list(spatial),
                        "flat_index": int(selected["flat_index"]),
                        **geometry,
                        "output_stride_pixels": (
                            None
                            if scale_metadata is None
                            else int(scale_metadata.output_stride)
                        ),
                        "receptive_field_pixels": (
                            None
                            if scale_metadata is None
                            else int(scale_metadata.receptive_field)
                        ),
                        "removed_base_boundary_rates": removed_rates[
                            sample
                        ].tolist(),
                        "removed_boundary_concentration": removed_concentration[
                            sample
                        ].tolist(),
                        "removed_net_boundary_flow": removed_flow[sample].tolist(),
                        "removed_boundary_correction": removed_correction[
                            sample
                        ].tolist(),
                        "baseline_boundary_concentration": base_concentration[
                            sample
                        ].tolist(),
                        "replayed_boundary_concentration": replay_concentration[
                            sample
                        ].tolist(),
                        "baseline_transport_direction": base_direction[
                            sample
                        ].tolist(),
                        "replayed_transport_direction": replay_direction[
                            sample
                        ].tolist(),
                        "baseline_transport_matrix": base_transport[sample].tolist(),
                        "replayed_transport_matrix": replay_transport[sample].tolist(),
                        "baseline_net_boundary_flow": base_flow[sample].tolist(),
                        "replayed_net_boundary_flow": replay_flow[sample].tolist(),
                        "base_boundary_logit_delta": base_logit_delta.tolist(),
                        "correction_boundary_logit_delta": correction_delta.tolist(),
                        "final_boundary_logit_delta": final_logit_delta.tolist(),
                        "target_log_probability_boundary_contributions": (
                            contributions.tolist()
                        ),
                        "target_log_probability_contribution_sum": float(
                            contributions.sum()
                        ),
                        "target_log_probability_contribution_error": (
                            contribution_error
                        ),
                        "configured_decision_margin_drop": top_margin_drop,
                        "predicted_grade_log_probability_drop": top_log_drop,
                    }
                    top_margin_effects.append(top_margin_drop)
                    top_log_effects.append(top_log_drop)

                random_payload: dict[str, Any] | None = None
                if sample in random_locations:
                    random_scale, random_spatial, random_flat = random_locations[sample]
                    random_margin = float(
                        _fixed_grade_decision_margin(
                            random_probs[sample : sample + 1],
                            random_cumulative[sample : sample + 1],
                            random_expected[sample : sample + 1],
                            target_grade=target,
                            decision_rule=self.decision_rule,
                            log_class_probs=random_log_probs[
                                sample : sample + 1
                            ],
                        )[0]
                    )
                    random_margin_drop = base_margin - random_margin
                    random_log_drop = float(
                        base_log_probs[sample, target]
                        - random_log_probs[sample, target]
                    )
                    random_margin_effects.append(random_margin_drop)
                    random_log_effects.append(random_log_drop)
                    paired_top_margin_effects.append(top_margin_drop)
                    paired_top_log_effects.append(top_log_drop)
                    random_payload = {
                        "scale": random_scale,
                        "spatial_index": list(random_spatial),
                        "flat_index": random_flat,
                        "selection": "sha256_deterministic_same_scale_valid_cell",
                        "configured_decision_margin_drop": random_margin_drop,
                        "predicted_grade_log_probability_drop": random_log_drop,
                        "replayed_prediction": int(random_predictions[sample]),
                    }

                certificates.append(
                    {
                        "sample_id": self._index_value(indices, sample),
                        "label": int(labels[sample].cpu()),
                        "target_predicted_grade": target,
                        "supporter_found": bool(ranking["supporter_found"]),
                        "no_positive_supporter_reason": (
                            None
                            if ranking["supporter_found"]
                            else (
                                "no valid singleton jointly decreased the configured-"
                                "decision margin and predicted-grade log probability"
                            )
                        ),
                        "valid_cell_count": int(ranking["valid_cell_count"]),
                        "strongest_candidate_regardless_of_sign": ranking[
                            "strongest_candidate_regardless_of_sign"
                        ],
                        "selected_positive_supporter": selected_payload,
                        "deterministic_scale_matched_random": random_payload,
                        "baseline_prediction": int(base_predictions[sample]),
                        "replayed_prediction": int(replay_predictions[sample]),
                        "baseline_class_probs": base_probs[sample].tolist(),
                        "replayed_class_probs": replay_probs[sample].tolist(),
                        "expected_grade_delta": float(
                            base_expected[sample] - replay_expected[sample]
                        ),
                    }
                )

            if ranking_replay_error > tolerance:
                raise AssertionError(
                    "PATHS vectorized singleton ranking disagrees with canonical "
                    f"replay by {ranking_replay_error:g} > {tolerance:g}"
                )

            paired_count = len(random_margin_effects)
            faithfulness = {
                "scope": (
                    "selected_validation_batch_only; deterministic same-scale "
                    "singleton baseline; not a dataset-level localization claim"
                ),
                "paired_sample_count": paired_count,
                "selected_positive_supporter_count": len(top_margin_effects),
                "mean_top_configured_decision_margin_drop": (
                    None
                    if not paired_top_margin_effects
                    else float(np.mean(paired_top_margin_effects))
                ),
                "mean_random_configured_decision_margin_drop": (
                    None
                    if not random_margin_effects
                    else float(np.mean(random_margin_effects))
                ),
                "mean_top_predicted_grade_log_probability_drop": (
                    None
                    if not paired_top_log_effects
                    else float(np.mean(paired_top_log_effects))
                ),
                "mean_random_predicted_grade_log_probability_drop": (
                    None
                    if not random_log_effects
                    else float(np.mean(random_log_effects))
                ),
                "top_exceeds_random_margin_fraction": (
                    None
                    if paired_count == 0
                    else float(
                        np.mean(
                            np.asarray(paired_top_margin_effects)
                            > np.asarray(random_margin_effects)
                        )
                    )
                ),
                "top_exceeds_random_logprob_fraction": (
                    None
                    if paired_count == 0
                    else float(
                        np.mean(
                            np.asarray(paired_top_log_effects)
                            > np.asarray(random_log_effects)
                        )
                    )
                ),
            }

        payload: dict[str, Any] = {
            "schema": "paths-exact-joint-validation-certificates-v3-sapt",
            "scope": "inner_validation_only",
            "protocol": PATHS_PROTOCOL_VERSION,
            "paths_variant": "signed_transport",
            "fold": self.fold,
            "split_signature": self.split_signature,
            "checkpoint_epoch": int(checkpoint_epoch),
            "checkpoint_sha256": _file_sha256(self.best_path),
            "implementation_signature": self.implementation_signature,
            "architecture_signature": self.architecture_signature,
            "warm_start_checkpoint_sha256": self.paths_warm_start_provenance[
                "source_checkpoint_sha256"
            ],
            "decision_rule": self.decision_rule,
            "selection_rule": (
                "exact_joint_singleton_replay_lexicographic_configured_decision_"
                "margin_drop_then_predicted_grade_log_probability_drop"
            ),
            "positive_supporter_definition": (
                "both configured-decision margin drop and predicted-grade log-"
                "probability drop are strictly positive"
            ),
            "interpretation_scope": (
                "exact_joint_stored_base_rate_and_pgf_concentration_intervention;"
                "exact_recompilation_of_mass_conserving_signed_adjacent_transport;"
                "theoretical_encoder_receptive_field_support;"
                "not_input_pixel_causality_or_named_lesion_semantics"
            ),
            "certifies_final_paths_prediction": True,
            "base_only_certificate": False,
            "replay_tolerance": tolerance,
            "singleton_ranking_canonical_replay_max_abs_error": (
                ranking_replay_error
            ),
            "joint_replay_audit": audit,
            "matched_random_joint_replay_audit": random_audit,
            "faithfulness_audit": faithfulness,
            "certificates": certificates,
        }
        payload["content_checksum_sha256"] = _canonical_sha256(payload)
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

    def _evaluate_validation_transport(self) -> dict[str, Any]:
        """Audit the learned transport over the complete validation split.

        The small certificate batch proves exact cell-deletion replay.  This
        split-wide audit answers a different question: whether the selected
        checkpoint actually uses both directions of the signed transport and
        whether every emitted kernel remains a valid adjacent Markov kernel.
        """

        if self.paths_variant == "risk_objective_v3":
            return {
                "schema": "paths-v3-sapt-validation-transport-audit-v1",
                "scope": "full_inner_validation",
                "variant": self.paths_variant,
                "transport_active": False,
                "reason": "matched_strength_zero_control",
            }

        self.model.eval()
        flows: list[torch.Tensor] = []
        concentrations: list[torch.Tensor] = []
        directions: list[torch.Tensor] = []
        sample_count = 0
        effect_abs_sum = 0.0
        effect_count = 0
        effect_max = 0.0
        row_sum_error = 0.0
        non_adjacent_max = 0.0
        matrix_minimum = math.inf
        posterior_mass_error = 0.0
        with torch.no_grad():
            for batch in self.val_loader:
                images, pixel_mask, labels, _ = self._unpack_batch(batch)
                with autocast(device_type="cuda", enabled=self.use_amp):
                    output = self._forward(images, pixel_mask)
                if not hasattr(output, "transport_matrix"):
                    raise TypeError(
                        "signed-transport validation produced no transport matrix"
                    )
                flow = _tensor_field(output, "net_boundary_flow").double()
                concentration = _tensor_field(
                    output, "boundary_concentration"
                ).double()
                direction = _tensor_field(output, "transport_direction").double()
                matrix = _tensor_field(output, "transport_matrix").double()
                probabilities = _tensor_field(output, "class_probs").double()
                base_probabilities = _tensor_field(
                    _field(output, "base_output"), "class_probs"
                ).double()
                flows.append(flow.detach().cpu())
                concentrations.append(concentration.detach().cpu())
                directions.append(direction.detach().cpu())
                sample_count += int(labels.numel())
                effect = (probabilities - base_probabilities).abs()
                effect_abs_sum += float(effect.sum().detach().cpu())
                effect_count += int(effect.numel())
                effect_max = max(effect_max, float(effect.max().detach().cpu()))
                row_sum_error = max(
                    row_sum_error,
                    float((matrix.sum(dim=-1) - 1.0).abs().max().detach().cpu()),
                )
                matrix_minimum = min(
                    matrix_minimum, float(matrix.min().detach().cpu())
                )
                posterior_mass_error = max(
                    posterior_mass_error,
                    float(
                        (probabilities.sum(dim=-1) - 1.0)
                        .abs()
                        .max()
                        .detach()
                        .cpu()
                    ),
                )
                grade_axis = torch.arange(matrix.shape[-1], device=matrix.device)
                non_adjacent = (
                    grade_axis[:, None] - grade_axis[None, :]
                ).abs() > 1
                if bool(non_adjacent.any()):
                    non_adjacent_max = max(
                        non_adjacent_max,
                        float(
                            matrix[..., non_adjacent]
                            .abs()
                            .max()
                            .detach()
                            .cpu()
                        ),
                    )

        if not flows or sample_count == 0:
            raise RuntimeError("PATHS validation transport audit saw no samples")
        flow = torch.cat(flows, dim=0)
        concentration = torch.cat(concentrations, dim=0)
        direction = torch.cat(directions, dim=0)
        sign_tolerance = 1e-12

        def _quantiles(values: torch.Tensor) -> dict[str, float]:
            flattened = values.flatten().double()
            return {
                name: float(torch.quantile(flattened, quantile))
                for name, quantile in (
                    ("q0", 0.0),
                    ("q25", 0.25),
                    ("q50", 0.50),
                    ("q75", 0.75),
                    ("q95", 0.95),
                    ("q100", 1.0),
                )
            }

        positive = int((flow > sign_tolerance).sum())
        negative = int((flow < -sign_tolerance).sum())
        total = int(flow.numel())
        return {
            "schema": "paths-v3-sapt-validation-transport-audit-v1",
            "scope": "full_inner_validation",
            "variant": self.paths_variant,
            "transport_active": effect_max > sign_tolerance,
            "sample_count": sample_count,
            "boundary_value_count": total,
            "positive_upward_flow_count": positive,
            "negative_downward_flow_count": negative,
            "near_zero_flow_count": total - positive - negative,
            "bidirectional_flow_observed": positive > 0 and negative > 0,
            "net_boundary_flow_min": float(flow.min()),
            "net_boundary_flow_max": float(flow.max()),
            "mean_absolute_net_boundary_flow": float(flow.abs().mean()),
            "transport_effect_mean_abs": (
                effect_abs_sum / float(max(1, effect_count))
            ),
            "transport_effect_max_abs": effect_max,
            "transport_matrix_row_sum_error": row_sum_error,
            "transport_matrix_minimum": matrix_minimum,
            "transport_non_adjacent_max_abs": non_adjacent_max,
            "posterior_mass_error": posterior_mass_error,
            "boundary_concentration_quantiles": _quantiles(concentration),
            "transport_direction_quantiles": _quantiles(direction),
        }

    def fit(self, *, evaluate_test: bool = False) -> dict[str, Any]:
        if not bool(getattr(self.cfg, "resume", False)):
            if self.best_learned_path.exists():
                raise FileExistsError(
                    f"fresh PATHS run would overwrite {self.best_learned_path}"
                )
            self._evaluate_v3_control()
        result = super().fit(evaluate_test=evaluate_test)
        validation_transport = self._evaluate_validation_transport()
        best_learned = self._materialize_best_learned_checkpoint()
        if not self.best_learned_path.is_file():
            raise RuntimeError("PATHS training produced no best_learned.pth")
        result.update(
            {
                "protocol": PATHS_PROTOCOL_VERSION,
                "paths_variant": self.paths_variant,
                "implementation_signature": self.implementation_signature,
                "architecture_signature": self.architecture_signature,
                "config_signature": self.config_signature,
                "critical_config": self.critical_config,
                "run_git_commit": self.run_git_commit,
                "split_signature": self.split_signature,
                "warm_start_provenance": self.paths_warm_start_provenance,
                "v3_strength_zero_control_path": str(self.base_control_path),
                "training_label_counts": list(self.training_label_counts),
                "risk_set_boundary_weights": (
                    self.criterion.boundary_weights.detach().cpu().tolist()
                ),
                "risk_set_weights_from_training_labels_only": True,
                "optimizer_group_contract": self.optimizer_group_contract,
                "best_learned_checkpoint": str(self.best_learned_path),
                "best_learned_checkpoint_sha256": _file_sha256(
                    self.best_learned_path
                ),
                "best_learned_epoch": int(best_learned["epoch"]),
                "best_learned_validation": dict(best_learned["metrics"]),
                "best_learned_checkpoint_role": best_learned["checkpoint_role"],
                "best_learned_is_byte_exact_best_alias": True,
                "v3_control_promoted_as_paths": False,
                "validation_transport_summary": validation_transport,
            }
        )
        return result


__all__ = [
    "PathsTrainer",
    "audit_paths_joint_replay",
    "continuation_logits_from_class_probs",
    "continuation_posterior_from_logits",
    "load_audited_origin_v3_warm_start",
    "paths_implementation_signature",
    "rank_paths_singleton_witnesses",
    "target_log_probability_boundary_contributions",
]
