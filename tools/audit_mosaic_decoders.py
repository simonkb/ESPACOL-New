#!/usr/bin/env python3
"""Validation-only audit of MOSAIC's proof-to-grade decision rule.

This utility deliberately cannot evaluate the outer test split.  It replays a
fixed best checkpoint on its inner validation fold and compares only
pre-specified, parameter-free decisions derived from the selected MOSAIC proof:

* the historical rounded posterior mean;
* raw class MAP;
* raw ordinal posterior median;
* the same rules after analytically undoing the boundary outcome weights.

No threshold, temperature, or calibration parameter is fitted to validation
data.  In addition to the historical decoder audit, this utility compares the
checkpoint's selected-proof path with the *dense pre-projection regional path*
on exactly the same images.  This is a diagnostic counterfactual, not an
alternative model selection rule: it isolates information removed by proof
selection from information already lost by the regional evidence compiler.

For regional checkpoints the audit also replays the exact source-cell to
regional LogMeanExp operator and measures its attenuation relative to the
strongest source cell.  These intermediate measurements are read-only; no
feature, global logit, or fitted calibration path is introduced.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

import torch
from torch.amp import autocast
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from configs.config import MOSAICConfig
from Datasets.mosaic_data import (
    MOSAIC_PREPROCESSING_VERSION,
    MosaicFundusTransform,
    MosaicImageDataset,
    aptos_fold,
    eyepacs_fold,
    load_aptos_items,
    load_eyepacs_items,
)
from models.mosaic_decoder import (
    DeweightedContinuation,
    PROOF_DECISION_RULES,
    ProofOnlyDecisionBundle,
    decision_rule_outputs,
    proof_only_decisions,
)
from models.mosaic import (
    nested_witness_probabilities,
    regional_logmeanexp_ordinal_evidence,
)
from models.mosaic_model import build_mosaic_model
from training.mosaic_trainer import mosaic_implementation_signature
from utils.metrics import evaluate_predictions


SCHEMA = "mosaic-validation-decoder-audit-v3"
WITNESS_TOP_K = (1, 4, 16, 32, 64)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _checkpoint_decision_rule(stored_config: dict[str, Any]) -> str:
    """Return the rule that produced the checkpoint's serialized metrics.

    Checkpoints written before decision rules became explicit used rounded
    expected grade.  They must not silently inherit the new configuration
    default when reconstructed with the current dataclass.
    """

    rule = stored_config.get("decision_rule", "rounded_expected")
    if rule not in PROOF_DECISION_RULES:
        raise ValueError(
            f"checkpoint has unknown MOSAIC decision rule {rule!r}; "
            f"expected one of {PROOF_DECISION_RULES}"
        )
    return rule


def _split_signature(*named_splits) -> str:
    """Match the fold identity serialized by :mod:`train_mosaic`."""

    digest = hashlib.sha256()
    for split_name, items in named_splits:
        digest.update(f"[{split_name}]\n".encode("utf-8"))
        canonical = sorted((Path(path).name, int(label)) for path, label in items)
        for image_name, label in canonical:
            digest.update(f"{image_name}\t{label}\n".encode("utf-8"))
    return digest.hexdigest()


def _json_safe(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu()
        return value.item() if value.ndim == 0 else value.tolist()
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "item"):
        return value.item()
    return value


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w") as stream:
            json.dump(_json_safe(payload), stream, indent=2, allow_nan=False)
            stream.write("\n")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _classification_metrics(
    prediction: torch.Tensor,
    labels: torch.Tensor,
    classes: int,
    cumulative: torch.Tensor,
) -> dict[str, Any]:
    prediction = prediction.long().cpu()
    labels = labels.long().cpu()
    metrics: dict[str, Any] = evaluate_predictions(
        prediction.float(),
        labels,
        classes,
        ordinal_probs=cumulative.float().cpu(),
    )
    confusion = torch.zeros(classes, classes, dtype=torch.long)
    for truth, predicted in zip(labels, prediction):
        confusion[int(truth), int(predicted)] += 1
    recall = confusion.diag().float() / confusion.sum(dim=1).clamp_min(1)
    precision = confusion.diag().float() / confusion.sum(dim=0).clamp_min(1)
    f1 = 2.0 * precision * recall / (precision + recall).clamp_min(1e-12)
    histogram = torch.bincount(prediction, minlength=classes)
    metrics.update(
        {
            "balanced_acc": 100.0 * float(recall.mean()),
            "macro_f1": float(f1.mean()),
            "per_grade_recall": recall.tolist(),
            "prediction_histogram": histogram.tolist(),
            "confusion": confusion.tolist(),
        }
    )
    return metrics


def _comparison(
    current: torch.Tensor,
    candidate: torch.Tensor,
    labels: torch.Tensor,
) -> dict[str, int]:
    current = current.long()
    candidate = candidate.long()
    labels = labels.long()
    changed = current != candidate
    current_correct = current == labels
    candidate_correct = candidate == labels
    return {
        "changed": int(changed.sum()),
        "current_wrong_alternative_correct": int(
            (changed & ~current_correct & candidate_correct).sum()
        ),
        "current_correct_alternative_wrong": int(
            (changed & current_correct & ~candidate_correct).sum()
        ),
        "both_wrong": int((changed & ~current_correct & ~candidate_correct).sum()),
    }


def _declared_decision_replay_mismatches(
    model_predictions: torch.Tensor,
    decoder_specs: dict[str, tuple[torch.Tensor, torch.Tensor]],
    decision_rule: str,
) -> int:
    """Compare model output with the checkpoint's declared decision rule."""

    if decision_rule not in decoder_specs:
        raise ValueError(f"decision rule {decision_rule!r} is absent from decoder specs")
    expected = decoder_specs[decision_rule][0].detach().long().cpu()
    actual = model_predictions.detach().long().cpu()
    if actual.shape != expected.shape:
        raise ValueError("model and replayed decisions must have matching shapes")
    return int((actual != expected).sum())


def _rate(numerator: int, denominator: int) -> float | None:
    return None if denominator == 0 else numerator / denominator


def _binary_auroc(
    scores: torch.Tensor,
    targets: torch.Tensor,
) -> float | None:
    """Return an exact, dependency-free binary AUROC with average tie ranks.

    ``None`` is the explicit result when a validation boundary contains only
    one outcome.  No threshold is selected and no validation statistic is fed
    back into prediction.
    """

    scores = scores.detach().double().flatten().cpu()
    targets = targets.detach().bool().flatten().cpu()
    if scores.shape != targets.shape:
        raise ValueError("AUROC scores and targets must have matching shapes")
    if scores.numel() == 0:
        return None
    if bool(torch.isnan(scores).any()):
        raise ValueError("AUROC scores must not contain NaN")
    positive_count = int(targets.sum())
    negative_count = int(targets.numel() - positive_count)
    if positive_count == 0 or negative_count == 0:
        return None

    order = torch.argsort(scores, stable=True)
    sorted_scores = scores[order]
    sorted_targets = targets[order]
    _, counts = torch.unique_consecutive(sorted_scores, return_counts=True)
    starts = counts.cumsum(dim=0) - counts
    # Ranks are one-based.  Every member of a tied group receives the group's
    # mean rank, which gives the standard half-credit tie convention.
    average_ranks = starts.double() + (counts.double() + 1.0) / 2.0
    ranks = torch.repeat_interleave(average_ranks, counts)
    positive_rank_sum = ranks[sorted_targets].sum()
    auc = (
        positive_rank_sum
        - positive_count * (positive_count + 1) / 2.0
    ) / (positive_count * negative_count)
    return float(auc)


def _signed_delta_summary(values: torch.Tensor) -> dict[str, float | int | None]:
    """Summarize a signed diagnostic difference without hiding direction."""

    values = values.detach().double().flatten().cpu()
    if values.numel() == 0:
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "minimum": None,
            "maximum": None,
            "mean_absolute": None,
            "p90_absolute": None,
            "maximum_absolute": None,
        }
    if not bool(torch.isfinite(values).all()):
        raise ValueError("cannot summarize non-finite diagnostic differences")
    absolute = values.abs()
    return {
        "count": int(values.numel()),
        "mean": float(values.mean()),
        "median": float(torch.quantile(values, 0.5)),
        "minimum": float(values.min()),
        "maximum": float(values.max()),
        "mean_absolute": float(absolute.mean()),
        "p90_absolute": float(torch.quantile(absolute, 0.9)),
        "maximum_absolute": float(absolute.max()),
    }


def _log_probability_invariants(
    probabilities: torch.Tensor,
    log_probabilities: torch.Tensor,
) -> dict[str, float | int | bool]:
    """Audit exact log endpoints and probability-space underflow."""

    probability = probabilities.detach().float().cpu()
    log_probability = log_probabilities.detach().float().cpu()
    if probability.shape != log_probability.shape:
        raise ValueError("probability and log-probability tensors must match")
    if not bool(torch.isfinite(probability).all()) or bool(
        ((probability < 0.0) | (probability > 1.0)).any()
    ):
        raise ValueError("probabilities must be finite and lie in [0, 1]")
    if bool(torch.isnan(log_probability).any()) or bool(
        torch.isposinf(log_probability).any()
    ):
        raise ValueError("log probabilities must not contain NaN or +inf")
    finite = torch.isfinite(log_probability)
    positive = probability > 0.0
    replay_error = (log_probability[finite].exp() - probability[finite]).abs()
    return {
        "all_valid": bool(
            not bool((positive & ~finite).any())
            and not bool((finite & (log_probability > 1e-7)).any())
        ),
        "probability_zero_finite_log_count_underflow_preserved": int(
            ((probability == 0.0) & finite).sum()
        ),
        "probability_zero_negative_infinite_log_count_exact_endpoint": int(
            ((probability == 0.0) & torch.isneginf(log_probability)).sum()
        ),
        "probability_positive_negative_infinite_log_count_violation": int(
            (positive & torch.isneginf(log_probability)).sum()
        ),
        "finite_log_positive_count_violation": int(
            (finite & (log_probability > 1e-7)).sum()
        ),
        "exp_log_probability_max_absolute_error": (
            0.0 if replay_error.numel() == 0 else float(replay_error.max())
        ),
    }


def _normalised_boundary_log_pairs(
    log_advance_probabilities: torch.Tensor,
    log_stop_probabilities: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Normalize paired log scores without a probability-space round trip."""

    log_advance = log_advance_probabilities.detach().double().cpu()
    log_stop = log_stop_probabilities.detach().double().cpu()
    if log_advance.ndim < 1 or log_advance.shape[-1] < 1:
        raise ValueError("at least one continuation boundary is required")
    if log_stop.shape != log_advance.shape:
        raise ValueError("paired log-advance and log-stop tensors must match")
    for name, tensor in (("log advance", log_advance), ("log stop", log_stop)):
        if bool(torch.isnan(tensor).any()) or bool(torch.isposinf(tensor).any()):
            raise ValueError(f"{name} must not contain NaN or +inf")
    log_total = torch.logaddexp(log_advance, log_stop)
    if bool(torch.isneginf(log_total).any()):
        raise ValueError("each boundary needs finite advance or stop log mass")
    return log_advance - log_total, log_stop - log_total


def _cascade_from_normalised_log_pairs(
    log_advance: torch.Tensor,
    log_stop: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert normalized continuation log pairs into an ordinal class law."""

    log_cumulative = torch.cumsum(log_advance, dim=-1)
    if log_advance.shape[-1] > 1:
        log_classes = torch.cat(
            (
                log_stop[..., :1],
                log_cumulative[..., :-1] + log_stop[..., 1:],
                log_cumulative[..., -1:],
            ),
            dim=-1,
        )
    else:
        log_classes = torch.cat(
            (log_stop[..., :1], log_cumulative[..., -1:]), dim=-1
        )
    log_classes = log_classes - torch.logsumexp(
        log_classes, dim=-1, keepdim=True
    )
    return log_cumulative.exp(), log_classes.exp()


def _deweight_normalised_log_pairs(
    log_advance: torch.Tensor,
    log_stop: torch.Tensor,
    outcome_weights: torch.Tensor,
) -> DeweightedContinuation:
    boundaries = log_advance.shape[-1]
    weights = outcome_weights.detach().double().cpu()
    if weights.shape != (boundaries, 2):
        raise ValueError(
            "outcome weights must have shape "
            f"({boundaries}, 2) ordered as [stop, advance]"
        )
    if not bool(torch.isfinite(weights).all()) or bool((weights <= 0.0).any()):
        raise ValueError("outcome weights must be finite and strictly positive")
    log_advance_numerator = log_advance + weights[:, 0].log()
    log_stop_numerator = log_stop + weights[:, 1].log()
    log_total = torch.logaddexp(log_advance_numerator, log_stop_numerator)
    corrected_log_advance = log_advance_numerator - log_total
    corrected_log_stop = log_stop_numerator - log_total
    return DeweightedContinuation(
        transitions=corrected_log_advance.exp(),
        stop_probabilities=corrected_log_stop.exp(),
        log_transitions=corrected_log_advance,
        log_stop_probabilities=corrected_log_stop,
    )


def _proof_only_decisions_from_log_pairs(
    log_advance_probabilities: torch.Tensor,
    log_stop_probabilities: torch.Tensor,
    outcome_weights: torch.Tensor,
) -> ProofOnlyDecisionBundle:
    """Decode exact paired log masses, including probability underflow cases.

    This is audit-local deliberately: adding it to the training forward files
    would change the implementation signature of the already-trained
    checkpoint being diagnosed.
    """

    log_advance, log_stop = _normalised_boundary_log_pairs(
        log_advance_probabilities,
        log_stop_probabilities,
    )
    raw_cumulative, raw_classes = _cascade_from_normalised_log_pairs(
        log_advance, log_stop
    )
    corrected = _deweight_normalised_log_pairs(
        log_advance,
        log_stop,
        outcome_weights,
    )
    corrected_cumulative, corrected_classes = _cascade_from_normalised_log_pairs(
        corrected.log_transitions,
        corrected.log_stop_probabilities,
    )
    axis = torch.arange(raw_classes.shape[-1], dtype=raw_classes.dtype)
    raw_expected = (raw_classes * axis).sum(dim=-1)
    corrected_expected = (corrected_classes * axis).sum(dim=-1)
    max_grade = raw_classes.shape[-1] - 1
    return ProofOnlyDecisionBundle(
        raw_cumulative_probabilities=raw_cumulative,
        raw_class_probabilities=raw_classes,
        deweighted_transitions=corrected.transitions,
        deweighted_stop_probabilities=corrected.stop_probabilities,
        deweighted_log_stop_probabilities=corrected.log_stop_probabilities,
        deweighted_cumulative_probabilities=corrected_cumulative,
        deweighted_class_probabilities=corrected_classes,
        raw_expected_grade=raw_expected,
        deweighted_expected_grade=corrected_expected,
        raw_mean_round=raw_expected.round().long().clamp(0, max_grade),
        raw_argmax=raw_classes.argmax(dim=-1),
        raw_posterior_median=(raw_cumulative >= 0.5).sum(dim=-1),
        deweighted_mean_round=corrected_expected.round()
        .long()
        .clamp(0, max_grade),
        deweighted_argmax=corrected_classes.argmax(dim=-1),
        deweighted_posterior_median=(corrected_cumulative >= 0.5).sum(dim=-1),
    )


def _transition_path_diagnostics(
    projected_transitions: torch.Tensor,
    dense_transitions: torch.Tensor,
    labels: torch.Tensor,
    *,
    projected_log_transitions: torch.Tensor | None = None,
    projected_log_stops: torch.Tensor | None = None,
    dense_log_transitions: torch.Tensor | None = None,
    dense_log_stops: torch.Tensor | None = None,
) -> dict[str, Any]:
    """Compare selected-proof and dense pre-projection boundary outputs."""

    projected = projected_transitions.detach().float().cpu()
    dense = dense_transitions.detach().float().cpu()
    labels = labels.detach().long().flatten().cpu()
    if projected.ndim != 2 or dense.shape != projected.shape:
        raise ValueError("projected and dense transitions must share shape (N, K-1)")
    if projected.shape[0] != labels.numel():
        raise ValueError("transition batch size must match labels")
    if not bool(torch.isfinite(projected).all()) or not bool(
        torch.isfinite(dense).all()
    ):
        raise ValueError("transition diagnostics require finite probabilities")
    if bool(((projected < 0.0) | (projected > 1.0)).any()) or bool(
        ((dense < 0.0) | (dense > 1.0)).any()
    ):
        raise ValueError("transition diagnostics require probabilities in [0, 1]")

    log_tensors: dict[str, torch.Tensor] = {}
    for name, tensor in (
        ("projected_transition", projected_log_transitions),
        ("projected_stop", projected_log_stops),
        ("dense_pre_projection_transition", dense_log_transitions),
        ("dense_pre_projection_stop", dense_log_stops),
    ):
        if tensor is None:
            continue
        value = tensor.detach().float().cpu()
        if value.shape != projected.shape:
            raise ValueError(f"{name} log transitions must match transitions")
        if bool(torch.isnan(value).any()) or bool(torch.isposinf(value).any()):
            raise ValueError(f"{name} log transitions must not contain NaN or +inf")
        log_tensors[name] = value

    rows: list[dict[str, Any]] = []
    for boundary in range(projected.shape[1]):
        # Boundary k is trained only for samples still at risk of crossing it:
        # stop labels are Y=k and advance labels are Y>k.  Labels Y<k never
        # reach this continuation decision, so including them in the primary
        # AUROC would measure an unsupervised output and can be misleading.
        at_risk = labels >= boundary
        target = labels[at_risk] > boundary
        delta = dense[:, boundary] - projected[:, boundary]
        advance = labels > boundary
        stop = labels == boundary
        exact_log_pairs_available = all(
            name in log_tensors
            for name in (
                "projected_transition",
                "projected_stop",
                "dense_pre_projection_transition",
                "dense_pre_projection_stop",
            )
        )
        if exact_log_pairs_available:
            projected_auc_score = (
                log_tensors["projected_transition"][:, boundary]
                - log_tensors["projected_stop"][:, boundary]
            )
            dense_auc_score = (
                log_tensors["dense_pre_projection_transition"][:, boundary]
                - log_tensors["dense_pre_projection_stop"][:, boundary]
            )
            auc_score_definition = "exact_log_advance_minus_log_stop"
        else:
            projected_auc_score = projected[:, boundary]
            dense_auc_score = dense[:, boundary]
            auc_score_definition = "transition_probability_fallback"
        projected_auc = _binary_auroc(projected_auc_score[at_risk], target)
        dense_auc = _binary_auroc(dense_auc_score[at_risk], target)
        log_endpoint_counts = {}
        for name, value in log_tensors.items():
            log_endpoint_counts[name] = {
                "negative_infinite_count": int(
                    torch.isneginf(value[:, boundary]).sum()
                ),
                "finite_count": int(torch.isfinite(value[:, boundary]).sum()),
            }
        rows.append(
            {
                "boundary": boundary,
                "risk_set": f"Y>={boundary}",
                "target_within_risk_set": f"1[Y>{boundary}]",
                "at_risk_count": int(at_risk.sum()),
                "advance_count": int(advance.sum()),
                "stop_count": int(stop.sum()),
                "below_risk_count": int((labels < boundary).sum()),
                "projected_auroc": projected_auc,
                "dense_pre_projection_auroc": dense_auc,
                "auroc_score_definition": auc_score_definition,
                "dense_minus_projected_auroc": (
                    None
                    if projected_auc is None or dense_auc is None
                    else dense_auc - projected_auc
                ),
                "projected_transition_mean": float(
                    projected[:, boundary].mean()
                ),
                "dense_pre_projection_transition_mean": float(
                    dense[:, boundary].mean()
                ),
                "dense_minus_projected": {
                    "overall": _signed_delta_summary(delta),
                    "at_risk": _signed_delta_summary(delta[at_risk]),
                    "advance_targets": _signed_delta_summary(delta[advance]),
                    "stop_targets": _signed_delta_summary(delta[stop]),
                    "below_risk_diagnostic_only": _signed_delta_summary(
                        delta[labels < boundary]
                    ),
                },
                "dense_greater_count": int((delta > 1e-7).sum()),
                "projected_greater_count": int((delta < -1e-7).sum()),
                "equal_within_1e-7_count": int((delta.abs() <= 1e-7).sum()),
                "exact_log_transition_endpoints": log_endpoint_counts,
            }
        )
    return {
        "delta_definition": "dense_pre_projection_transition - projected_transition",
        "primary_auroc_population": "continuation risk set Y >= boundary",
        "auroc_target_within_risk_set": "1[Y > boundary]",
        "auroc_note": (
            "AUROC is threshold-free, excludes below-risk labels Y<boundary, "
            "and is null when stop or advance is absent from the reconstructed "
            "inner-validation risk set. With paired logs it ranks exact boundary "
            "log-odds (log advance - log stop), preserving distinctions after "
            "probability underflow. Overall probability transition deltas are "
            "retained only as a separate implementation diagnostic."
        ),
        "boundaries": rows,
    }


def _fixed_region_valid_counts(
    source_valid_mask: torch.Tensor,
    source_lattice_size: tuple[int, int],
    regional_block_size: tuple[int, int],
) -> torch.Tensor:
    """Count valid source cells in each fixed, disjoint regional block."""

    source_valid = source_valid_mask.detach().bool()
    if source_valid.ndim != 2:
        raise ValueError("source valid mask must have shape (N, P)")
    height, width = map(int, source_lattice_size)
    block_h, block_w = map(int, regional_block_size)
    if height < 1 or width < 1 or height * width != source_valid.shape[1]:
        raise ValueError("source lattice size does not match source valid mask")
    if block_h < 1 or block_w < 1 or height % block_h or width % block_w:
        raise ValueError("regional block size must divide the source lattice")
    grid_h, grid_w = height // block_h, width // block_w
    return (
        source_valid.reshape(-1, height, width)
        .reshape(-1, grid_h, block_h, grid_w, block_w)
        .permute(0, 1, 3, 2, 4)
        .reshape(-1, grid_h * grid_w, block_h * block_w)
        .sum(dim=-1)
    )


def _batch_regional_lme_attenuation(
    *,
    source_log_witness_probabilities: torch.Tensor,
    source_log_nonwitness_probabilities: torch.Tensor,
    regional_log_witness_probabilities: torch.Tensor,
    regional_log_nonwitness_probabilities: torch.Tensor,
    peak_source_indices: torch.Tensor,
    regional_valid_mask: torch.Tensor,
    valid_source_counts: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Collect exact source-maximum versus regional-LME attenuation values."""

    source_log_witness = source_log_witness_probabilities.detach().float()
    source_log_nonwitness = source_log_nonwitness_probabilities.detach().float()
    regional_log_witness = regional_log_witness_probabilities.detach().float()
    regional_log_nonwitness = regional_log_nonwitness_probabilities.detach().float()
    peak_indices = peak_source_indices.detach().long()
    valid_mask = regional_valid_mask.detach().bool()
    valid_counts = valid_source_counts.detach().long()

    if source_log_witness.ndim != 3:
        raise ValueError("source log witnesses must have shape (N, P, K-1)")
    if source_log_nonwitness.shape != source_log_witness.shape:
        raise ValueError("source witness log-probability tensors must match")
    if regional_log_witness.ndim != 3:
        raise ValueError("regional log witnesses must have shape (N, R, K-1)")
    if regional_log_nonwitness.shape != regional_log_witness.shape:
        raise ValueError("regional witness log-probability tensors must match")
    if peak_indices.shape != regional_log_witness.shape:
        raise ValueError("regional peak source indices must match regional witnesses")
    if valid_mask.shape != regional_log_witness.shape[:2]:
        raise ValueError("regional valid mask must have shape (N, R)")
    if valid_counts.shape != valid_mask.shape:
        raise ValueError("valid source counts must have shape (N, R)")

    n, source_cells, boundaries = source_log_witness.shape
    if regional_log_witness.shape[0] != n or regional_log_witness.shape[2] != boundaries:
        raise ValueError("source and regional witness dimensions disagree")
    expanded_valid = valid_mask[..., None].expand_as(regional_log_witness)
    if bool((peak_indices[expanded_valid] < 0).any()) or bool(
        (peak_indices[expanded_valid] >= source_cells).any()
    ):
        raise ValueError("valid regional events contain an invalid source index")
    if bool((valid_counts[valid_mask] <= 0).any()):
        raise ValueError("valid regional events need at least one valid source cell")

    safe_indices = peak_indices.clamp(0, source_cells - 1)
    source_logits = source_log_witness - source_log_nonwitness
    source_peak_logits = torch.gather(source_logits, 1, safe_indices)
    source_peak_probabilities = torch.gather(
        source_log_witness.exp(), 1, safe_indices
    )
    regional_logits = regional_log_witness - regional_log_nonwitness
    regional_probabilities = regional_log_witness.exp()
    for name, tensor in (
        ("source peak logits", source_peak_logits),
        ("regional LME logits", regional_logits),
        ("source peak probabilities", source_peak_probabilities),
        ("regional LME probabilities", regional_probabilities),
    ):
        if not bool(torch.isfinite(tensor[expanded_valid]).all()):
            raise ValueError(f"{name} contain non-finite valid entries")

    return {
        "valid_mask": expanded_valid.cpu(),
        "valid_source_counts": valid_counts.cpu(),
        "source_peak_logit": source_peak_logits.cpu(),
        "regional_lme_logit": regional_logits.cpu(),
        "logit_attenuation": (source_peak_logits - regional_logits).cpu(),
        "source_peak_probability": source_peak_probabilities.cpu(),
        "regional_lme_probability": regional_probabilities.cpu(),
        "probability_attenuation": (
            source_peak_probabilities - regional_probabilities
        ).cpu(),
    }


def _merge_regional_lme_attenuation_batches(
    batches: list[dict[str, torch.Tensor]],
) -> dict[str, torch.Tensor]:
    if not batches:
        raise ValueError("cannot merge an empty regional attenuation audit")
    keys = tuple(batches[0])
    if any(tuple(batch) != keys for batch in batches[1:]):
        raise ValueError("regional attenuation batches have inconsistent fields")
    return {
        name: torch.cat([batch[name] for batch in batches], dim=0)
        for name in keys
    }


def _regional_lme_attenuation_summary(
    attenuation: dict[str, torch.Tensor],
    *,
    temperature: float,
    source_lattice_size: tuple[int, int],
    regional_block_size: tuple[int, int],
    replay_max_probability_error: float,
    replay_peak_index_mismatches: int,
) -> dict[str, Any]:
    valid = attenuation["valid_mask"].bool()
    if valid.ndim != 3:
        raise ValueError("regional attenuation valid mask must have shape (N,R,K-1)")
    rows: list[dict[str, Any]] = []
    for boundary in range(valid.shape[2]):
        mask = valid[..., boundary]
        logit_attenuation = attenuation["logit_attenuation"][..., boundary][mask]
        probability_attenuation = attenuation["probability_attenuation"][
            ..., boundary
        ][mask]
        rows.append(
            {
                "boundary": boundary,
                "valid_region_events": int(mask.sum()),
                "valid_source_cells_per_region": _distribution_summary(
                    attenuation["valid_source_counts"][mask]
                ),
                "source_peak_logit": _distribution_summary(
                    attenuation["source_peak_logit"][..., boundary][mask]
                ),
                "regional_lme_logit": _distribution_summary(
                    attenuation["regional_lme_logit"][..., boundary][mask]
                ),
                "source_peak_minus_regional_lme_logit": (
                    _signed_delta_summary(logit_attenuation)
                ),
                "source_peak_probability": _distribution_summary(
                    attenuation["source_peak_probability"][..., boundary][mask]
                ),
                "regional_lme_probability": _distribution_summary(
                    attenuation["regional_lme_probability"][..., boundary][mask]
                ),
                "source_peak_minus_regional_lme_probability": (
                    _signed_delta_summary(probability_attenuation)
                ),
                "lme_above_source_max_count_tolerance_1e-6": int(
                    (logit_attenuation < -1e-6).sum()
                ),
            }
        )
    return {
        "available": True,
        "operator": "normalized_logmeanexp_in_source_logit_space",
        "temperature": float(temperature),
        "source_lattice_size": list(source_lattice_size),
        "regional_block_size": list(regional_block_size),
        "regional_event_count": int(valid.shape[1]),
        "definition": (
            "attenuation = strongest valid source-cell value - exact regional "
            "LogMeanExp value, measured before the cardinality circuit"
        ),
        "architecture_replay": {
            "regional_probability_max_absolute_error": float(
                replay_max_probability_error
            ),
            "peak_source_index_mismatches": int(replay_peak_index_mismatches),
        },
        "boundaries": rows,
    }


def _batch_witness_concentration(
    witness_probabilities: torch.Tensor,
) -> dict[str, Any]:
    """Reduce dense witnesses to per-image concentration scalars.

    Only the largest 64 witnesses and ``(N, K-1)`` reductions are materialized.
    The caller can therefore retain exact per-image values for conditional
    quantiles without copying the much larger ``(N, P, K-1)`` ledger to CPU.
    A zero-mass witness vector has effective support and top-k fractions zero.
    """

    if witness_probabilities.ndim != 3:
        raise ValueError(
            "witness_probabilities must have shape (N, P, K-1); got "
            f"{tuple(witness_probabilities.shape)}"
        )
    spatial_cells = int(witness_probabilities.shape[1])
    if spatial_cells < 1:
        raise ValueError("witness_probabilities must contain at least one cell")

    probabilities = witness_probabilities.detach().float()
    if not bool(torch.isfinite(probabilities).all()):
        raise ValueError("witness_probabilities contain non-finite values")
    if bool((probabilities < 0.0).any()) or bool((probabilities > 1.0).any()):
        raise ValueError("witness_probabilities must lie in [0, 1]")

    witness_count = probabilities.sum(dim=1)
    # einsum performs the reduction without retaining a full-sized squared
    # witness tensor alongside the already-live model output.
    squared_sum = torch.einsum("npb,npb->nb", probabilities, probabilities)
    effective_support = torch.where(
        squared_sum > 0.0,
        witness_count.square() / squared_sum.clamp_min(
            torch.finfo(probabilities.dtype).tiny
        ),
        torch.zeros_like(witness_count),
    ).clamp(0.0, float(spatial_cells))

    largest_k = min(max(WITNESS_TOP_K), spatial_cells)
    largest = torch.topk(
        probabilities,
        k=largest_k,
        dim=1,
        largest=True,
        sorted=True,
    ).values
    cumulative_largest = largest.cumsum(dim=1)
    positive_mass = witness_count > 0.0
    top_k_mass_fractions: dict[int, torch.Tensor] = {}
    clamped_top_k: dict[int, int] = {}
    for requested_k in WITNESS_TOP_K:
        used_k = min(requested_k, spatial_cells)
        clamped_top_k[requested_k] = used_k
        top_mass = cumulative_largest[:, used_k - 1, :]
        fraction = torch.where(
            positive_mass,
            top_mass / witness_count.clamp_min(
                torch.finfo(probabilities.dtype).tiny
            ),
            torch.zeros_like(witness_count),
        )
        top_k_mass_fractions[requested_k] = fraction.clamp(0.0, 1.0).cpu()

    return {
        "spatial_cell_count": spatial_cells,
        "clamped_top_k": clamped_top_k,
        "witness_count": witness_count.cpu(),
        "effective_support": effective_support.cpu(),
        "max_witness_probability": largest[:, 0, :].cpu(),
        "top_k_witness_mass_fraction": top_k_mass_fractions,
    }


def _merge_witness_concentration_batches(
    batches: list[dict[str, Any]],
) -> dict[str, Any]:
    """Concatenate only reduced per-image concentration statistics."""

    if not batches:
        raise ValueError("cannot merge an empty witness-concentration audit")
    spatial_cell_count = int(batches[0]["spatial_cell_count"])
    clamped_top_k = batches[0]["clamped_top_k"]
    for batch in batches[1:]:
        if int(batch["spatial_cell_count"]) != spatial_cell_count:
            raise ValueError("witness spatial cell count changed across batches")
        if batch["clamped_top_k"] != clamped_top_k:
            raise ValueError("clamped witness top-k values changed across batches")

    return {
        "spatial_cell_count": spatial_cell_count,
        "clamped_top_k": dict(clamped_top_k),
        "witness_count": torch.cat(
            [batch["witness_count"] for batch in batches], dim=0
        ),
        "effective_support": torch.cat(
            [batch["effective_support"] for batch in batches], dim=0
        ),
        "max_witness_probability": torch.cat(
            [batch["max_witness_probability"] for batch in batches], dim=0
        ),
        "top_k_witness_mass_fraction": {
            requested_k: torch.cat(
                [
                    batch["top_k_witness_mass_fraction"][requested_k]
                    for batch in batches
                ],
                dim=0,
            )
            for requested_k in WITNESS_TOP_K
        },
    }


def _distribution_summary(values: torch.Tensor) -> dict[str, float | None]:
    """Return exact validation-set summaries with explicit empty semantics."""

    values = values.detach().float().flatten()
    if values.numel() == 0:
        return {"mean": None, "median": None, "p90": None}
    if not bool(torch.isfinite(values).all()):
        raise ValueError("cannot summarize non-finite diagnostic values")
    return {
        "mean": float(values.mean()),
        "median": float(torch.quantile(values, 0.5)),
        "p90": float(torch.quantile(values, 0.9)),
    }


def _alpha_diagnostics(alpha: torch.Tensor) -> dict[str, Any]:
    """Describe a boundary's learned distribution over count thresholds."""

    weights = alpha.detach().float().flatten()
    if weights.numel() == 0:
        raise ValueError("alpha must contain at least one count threshold")
    if not bool(torch.isfinite(weights).all()) or bool((weights < 0.0).any()):
        raise ValueError("alpha must contain finite non-negative weights")
    total = weights.sum()
    if not bool(total > 0.0):
        raise ValueError("alpha weights must have positive total mass")
    weights = weights / total
    thresholds = torch.arange(
        1,
        weights.numel() + 1,
        device=weights.device,
        dtype=weights.dtype,
    )
    positive = weights > 0.0
    entropy = -(weights[positive] * weights[positive].log()).sum()
    return {
        "expected_threshold": float((weights * thresholds).sum()),
        "mode_threshold": int(weights.argmax()) + 1,
        "entropy_nats": float(entropy),
        "weights": weights.cpu().tolist(),
    }


def _conditional_concentration_summary(
    *,
    mask: torch.Tensor,
    boundary: int,
    proof_sizes: torch.Tensor,
    retained_overflow: torch.Tensor,
    witness_concentration: dict[str, Any],
) -> dict[str, Any]:
    """Summarize proof and dense-witness concentration for one label group."""

    mask = mask.bool().cpu()
    witness_count = witness_concentration["witness_count"][:, boundary]
    selected_proof_sizes = proof_sizes[mask, boundary]
    proof_size_summary = _distribution_summary(selected_proof_sizes)
    proof_size_summary["zero_rate"] = (
        None
        if selected_proof_sizes.numel() == 0
        else float((selected_proof_sizes == 0).float().mean())
    )
    sample_count = int(mask.sum())
    zero_witness_mass_count = int(((witness_count == 0.0) & mask).sum())
    group = {
        "sample_count": sample_count,
        "zero_witness_mass_count": zero_witness_mass_count,
        "zero_witness_mass_rate": _rate(zero_witness_mass_count, sample_count),
        "proof_size": proof_size_summary,
        "witness_count": _distribution_summary(witness_count[mask]),
        "effective_support": _distribution_summary(
            witness_concentration["effective_support"][mask, boundary]
        ),
        "max_witness_probability": _distribution_summary(
            witness_concentration["max_witness_probability"][mask, boundary]
        ),
        "retained_overflow_mass": _distribution_summary(
            retained_overflow[mask, boundary]
        ),
        "top_k_witness_mass_fraction": {},
    }
    for requested_k in WITNESS_TOP_K:
        group["top_k_witness_mass_fraction"][str(requested_k)] = {
            "clamped_k": int(witness_concentration["clamped_top_k"][requested_k]),
            **_distribution_summary(
                witness_concentration["top_k_witness_mass_fraction"][requested_k][
                    mask, boundary
                ]
            ),
        }
    return group


def _proof_diagnostics(
    proof_sizes: torch.Tensor,
    transitions: torch.Tensor,
    labels: torch.Tensor,
    witness_concentration: dict[str, Any],
    retained_overflow: torch.Tensor,
    alpha: torch.Tensor,
) -> dict[str, Any]:
    """Expose boundary-wise hard-proof and probability-dust diagnostics."""

    proof_sizes = proof_sizes.long().cpu()
    transitions = transitions.float().cpu()
    labels = labels.long().cpu()
    retained_overflow = retained_overflow.float().cpu()
    alpha = alpha.float().cpu()
    boundaries = transitions.shape[1]
    expected_shape = (labels.numel(), boundaries)
    for name, tensor in (
        ("proof_sizes", proof_sizes),
        ("transitions", transitions),
        ("retained_overflow", retained_overflow),
        ("witness_count", witness_concentration["witness_count"]),
        ("effective_support", witness_concentration["effective_support"]),
        (
            "max_witness_probability",
            witness_concentration["max_witness_probability"],
        ),
    ):
        if tuple(tensor.shape) != expected_shape:
            raise ValueError(
                f"{name} must have shape {expected_shape}; got {tuple(tensor.shape)}"
            )
    if tuple(alpha.shape[:1]) != (boundaries,) or alpha.ndim != 2:
        raise ValueError(
            "alpha must have shape (K-1, max_count); got "
            f"{tuple(alpha.shape)}"
        )

    rows: list[dict[str, Any]] = []
    for boundary in range(boundaries):
        overall = torch.ones_like(labels, dtype=torch.bool)
        at_risk = labels >= boundary
        advance = labels > boundary
        stop = labels == boundary
        below_risk = labels < boundary
        empty = proof_sizes[:, boundary] == 0
        exact_zero = transitions[:, boundary] == 0
        advance_count = int(advance.sum())
        stop_count = int(stop.sum())
        empty_advance = int((empty & advance).sum())
        exact_zero_advance = int((exact_zero & advance).sum())
        rows.append(
            {
                "boundary": boundary,
                "overall_count": int(labels.numel()),
                "at_risk_count": int(at_risk.sum()),
                "advance_count": advance_count,
                "stop_count": stop_count,
                "below_risk_count": int(below_risk.sum()),
                "empty_proof_count": int(empty.sum()),
                "empty_proof_rate": float(empty.float().mean()),
                "empty_proof_advance_count": empty_advance,
                "empty_proof_advance_rate": _rate(empty_advance, advance_count),
                "empty_proof_stop_rate": _rate(int((empty & stop).sum()), stop_count),
                "exact_zero_transition_advance_count": exact_zero_advance,
                "exact_zero_transition_advance_rate": _rate(
                    exact_zero_advance, advance_count
                ),
                "proof_size_mean": float(proof_sizes[:, boundary].float().mean()),
                "witness_count_mean": float(
                    witness_concentration["witness_count"][:, boundary].mean()
                ),
                "retained_overflow_mass_mean": float(
                    retained_overflow[:, boundary].mean()
                ),
                "alpha_at_max_count": float(alpha[boundary, -1]),
                "alpha": _alpha_diagnostics(alpha[boundary]),
                "conditional_concentration": {
                    name: _conditional_concentration_summary(
                        mask=mask,
                        boundary=boundary,
                        proof_sizes=proof_sizes,
                        retained_overflow=retained_overflow,
                        witness_concentration=witness_concentration,
                    )
                    for name, mask in (
                        ("overall", overall),
                        ("at_risk", at_risk),
                        ("advance", advance),
                        ("stop", stop),
                        ("below_risk", below_risk),
                    )
                },
            }
        )
    return {
        "note": (
            "An empty proof with an advance target yields an exact-zero projected "
            "transition; current training replaces that boundary's primary term "
            "with the exact dense log-likelihood recovery term."
        ),
        "concentration_note": (
            "Witness concentration is computed from the dense regional witness "
            "ledger before proof selection. Witness count is sum(p), effective "
            "support is "
            "sum(p)^2/sum(p^2); top-k mass is divided by sum(p), with zero "
            "reported for a zero-mass ledger. Requested k is clamped to the "
            "spatial cell count."
        ),
        "condition_note": (
            "At boundary k: at_risk means Y>=k, advance means Y>k, stop means "
            "Y=k, and below_risk means Y<k."
        ),
        "witness_spatial_cell_count": int(
            witness_concentration["spatial_cell_count"]
        ),
        "boundaries": rows,
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Audit MOSAIC proof-only decoders on inner validation only",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data_root", default=None)
    parser.add_argument("--labels_csv", default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument(
        "--output_dir",
        default=None,
        help="Defaults to <checkpoint directory>/decoder_audit.",
    )
    parser.add_argument(
        "--require_implementation_match",
        action="store_true",
        help=(
            "Fail if the aggregate training-source signature changed. Strict model "
            "loading plus checkpoint-metric reproduction are always required."
        ),
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    checkpoint_path = Path(args.checkpoint).resolve()
    checkpoint_sha256 = _file_sha256(checkpoint_path)
    output_dir = (
        Path(args.output_dir).resolve()
        if args.output_dir
        else checkpoint_path.parent / "decoder_audit"
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    stored = checkpoint.get("config")
    if not isinstance(stored, dict):
        raise ValueError("checkpoint has no usable MOSAIC configuration")
    checkpoint_decision_rule = _checkpoint_decision_rule(stored)
    cfg = MOSAICConfig(
        **{
            key: value
            for key, value in stored.items()
            if key in MOSAICConfig.__dataclass_fields__
        }
    )
    cfg.decision_rule = checkpoint_decision_rule
    dataset_name = cfg.dataset.lower()
    if dataset_name not in {"aptos", "dr"}:
        raise ValueError(f"unsupported checkpoint dataset {cfg.dataset!r}")
    fold_value = checkpoint.get("fold")
    if fold_value is None:
        raise ValueError("checkpoint has no fold identity")
    fold = int(fold_value)
    if cfg.preprocessing_version != MOSAIC_PREPROCESSING_VERSION:
        raise ValueError(
            "checkpoint preprocessing does not match this data path: "
            f"{cfg.preprocessing_version!r} != {MOSAIC_PREPROCESSING_VERSION!r}"
        )

    saved_signature = checkpoint.get("implementation_signature")
    current_signature = mosaic_implementation_signature()
    implementation_match = bool(saved_signature == current_signature)
    if args.require_implementation_match and not implementation_match:
        raise ValueError(
            "checkpoint implementation signature differs from current source"
        )

    root = args.data_root or (
        "Datasets/aptos2019-blindness-detection"
        if dataset_name == "aptos"
        else "Datasets/DR"
    )
    if dataset_name == "aptos":
        all_items = load_aptos_items(root, args.labels_csv)
        train_items, validation_items, test_items = aptos_fold(
            all_items,
            fold,
            n_folds=cfg.n_folds,
            val_fraction=cfg.val_fraction,
            seed=cfg.seed,
        )
    else:
        all_items = load_eyepacs_items(root, args.labels_csv)
        train_items, validation_items, test_items = eyepacs_fold(
            all_items,
            fold,
            n_folds=cfg.n_folds,
            val_fraction=cfg.val_fraction,
            seed=cfg.seed,
        )
    current_split_signature = _split_signature(
        ("train", train_items),
        ("validation", validation_items),
        ("test", test_items),
    )
    if checkpoint.get("split_signature") != current_split_signature:
        raise ValueError(
            "checkpoint split signature does not match the reconstructed fold"
        )

    batch_size = args.batch_size or cfg.batch_size
    num_workers = cfg.num_workers if args.num_workers is None else args.num_workers
    validation_dataset = MosaicImageDataset(
        validation_items,
        MosaicFundusTransform(cfg.img_size, augment=False),
    )
    loader = DataLoader(
        validation_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )
    model = build_mosaic_model(
        num_classes=cfg.n_classes,
        image_size=cfg.img_size,
        local_stage=cfg.local_stage,
        local_dim=cfg.evidence_dim,
        pretrained=False,
        grad_checkpoint=False,
        initial_abnormal_count=cfg.normal_expected_count,
        max_count=cfg.max_count,
        sufficiency_tolerance=cfg.proof_epsilon,
        complement_suppression=cfg.necessity_fraction,
        count_implementation=cfg.count_implementation,
        count_block_size=cfg.count_block_size,
        region_grid_size=cfg.region_grid_size,
        region_pool_temperature=cfg.region_pool_temperature,
    ).to(device)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.eval()

    criterion_state = checkpoint.get("criterion_state")
    if not isinstance(criterion_state, dict) or "transition_weights" not in criterion_state:
        raise ValueError("checkpoint has no serialized boundary transition weights")
    transition_weights = criterion_state["transition_weights"].detach().float().cpu()
    if tuple(transition_weights.shape) != (cfg.n_classes - 1, 2):
        raise ValueError(
            "checkpoint transition weights do not match the configured ordinal "
            "boundaries"
        )
    if not bool(torch.isfinite(transition_weights).all()) or bool(
        (transition_weights <= 0.0).any()
    ):
        raise ValueError(
            "decoder audit requires positive stop and advance training support "
            "at every configured boundary"
        )
    model.configure_proof_decoder(
        checkpoint_decision_rule,
        transition_weights.to(device),
    )

    labels_batches: list[torch.Tensor] = []
    indices_batches: list[torch.Tensor] = []
    model_predictions: list[torch.Tensor] = []
    transitions_batches: list[torch.Tensor] = []
    log_transitions_batches: list[torch.Tensor] = []
    stops_batches: list[torch.Tensor] = []
    log_stops_batches: list[torch.Tensor] = []
    dense_transitions_batches: list[torch.Tensor] = []
    dense_log_transitions_batches: list[torch.Tensor] = []
    dense_stops_batches: list[torch.Tensor] = []
    dense_log_stops_batches: list[torch.Tensor] = []
    model_classes_batches: list[torch.Tensor] = []
    proof_sizes_batches: list[torch.Tensor] = []
    witness_concentration_batches: list[dict[str, Any]] = []
    overflow_batches: list[torch.Tensor] = []
    regional_attenuation_batches: list[dict[str, torch.Tensor]] = []
    regional_replay_max_errors: list[float] = []
    regional_replay_peak_mismatches = 0
    regional_source_lattice_size: tuple[int, int] | None = None
    regional_block_size: tuple[int, int] | None = None
    proof_dense_transition_replay_max_error = 0.0
    proof_sufficiency_gap_replay_max_error = 0.0
    alpha_reference: torch.Tensor | None = None

    use_amp = bool(cfg.amp and device.type == "cuda")
    with torch.no_grad():
        for images, masks, labels, indices in loader:
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            try:
                with autocast(device_type="cuda", enabled=use_amp):
                    output = model(
                        images,
                        masks,
                        project=True,
                        return_local_features=bool(cfg.region_grid_size),
                    )
            except FloatingPointError:
                if not use_amp:
                    raise
                with autocast(device_type="cuda", enabled=False):
                    output = model(
                        images,
                        masks,
                        project=True,
                        return_local_features=bool(cfg.region_grid_size),
                    )

            labels_batches.append(labels.long().cpu())
            indices_batches.append(indices.long().cpu())
            model_predictions.append(output.predicted_grade.long().cpu())
            transitions_batches.append(output.transitions.float().cpu())
            log_transitions_batches.append(
                output.log_transition_probabilities.float().cpu()
            )
            stops_batches.append(output.stop_probabilities.float().cpu())
            log_stops_batches.append(output.log_stop_probabilities.float().cpu())
            dense_transitions_batches.append(output.dense_transitions.float().cpu())
            dense_log_transitions_batches.append(
                output.dense_log_transition_probabilities.float().cpu()
            )
            dense_stops_batches.append(
                output.dense_stop_probabilities.float().cpu()
            )
            dense_log_stops_batches.append(
                output.dense_log_stop_probabilities.float().cpu()
            )
            # Reproduce the raw circuit distribution independently of whether
            # the checkpoint's point decision uses a deweighted rule.
            model_classes_batches.append(
                output.evidence.class_probabilities.float().cpu()
            )
            proof_sizes_batches.append(output.proof.proof_size.long().cpu())
            witness_concentration_batches.append(
                _batch_witness_concentration(
                    output.evidence.witness_probabilities
                )
            )
            overflow_batches.append(
                output.proof.retained_distribution[..., -1].float().cpu()
            )
            proof_dense_transition_replay_max_error = max(
                proof_dense_transition_replay_max_error,
                float(
                    (
                        output.dense_transitions
                        - output.proof.dense_transition
                    ).abs().max()
                ),
            )
            proof_sufficiency_gap_replay_max_error = max(
                proof_sufficiency_gap_replay_max_error,
                float(
                    (
                        output.proof.sufficiency_gap
                        - (
                            output.dense_transitions
                            - output.transitions
                        )
                    ).abs().max()
                ),
            )
            current_alpha = output.evidence.alpha.detach().float().cpu()
            if current_alpha.ndim == 3:
                current_alpha = current_alpha[0]
            if alpha_reference is None:
                alpha_reference = current_alpha
            elif not torch.allclose(alpha_reference, current_alpha, atol=0.0, rtol=0.0):
                raise RuntimeError("MOSAIC alpha unexpectedly changed across batches")

            if cfg.region_grid_size:
                if (
                    output.local_features is None
                    or output.source_valid_mask is None
                    or output.source_lattice is None
                    or output.evidence.regional_source_indices is None
                    or output.evidence.regional_block_size is None
                    or output.evidence.regional_pool_temperature is None
                ):
                    raise RuntimeError(
                        "regional checkpoint omitted source-to-region audit trace"
                    )
                current_source_lattice_size = tuple(
                    map(int, output.source_lattice.lattice_size)
                )
                current_block_size = tuple(
                    map(int, output.evidence.regional_block_size)
                )
                if regional_source_lattice_size is None:
                    regional_source_lattice_size = current_source_lattice_size
                    regional_block_size = current_block_size
                elif (
                    regional_source_lattice_size != current_source_lattice_size
                    or regional_block_size != current_block_size
                ):
                    raise RuntimeError(
                        "regional source geometry unexpectedly changed across batches"
                    )

                # Replay the exact pointwise state head and exact regional
                # compiler used by the forward pass.  This reads an internal
                # trace only; neither result is used as a classifier input.
                with autocast(device_type="cuda", enabled=False):
                    source_logits = model.proof_head.local_state_head.logits(
                        output.local_features.float()
                    )
                    source_evidence = nested_witness_probabilities(
                        source_logits,
                        output.source_valid_mask,
                    )
                    replayed_regional = regional_logmeanexp_ordinal_evidence(
                        source_evidence,
                        output.source_valid_mask,
                        current_source_lattice_size,
                        (cfg.region_grid_size, cfg.region_grid_size),
                        temperature=output.evidence.regional_pool_temperature,
                    )
                if not torch.equal(
                    replayed_regional.valid_mask,
                    output.evidence.evidence_valid_mask,
                ):
                    raise RuntimeError(
                        "regional audit replay produced a different valid mask"
                    )
                replay_error = float(
                    (
                        replayed_regional.evidence.witness_probabilities
                        - output.evidence.witness_probabilities
                    )
                    .abs()
                    .max()
                )
                regional_replay_max_errors.append(replay_error)
                regional_replay_peak_mismatches += int(
                    (
                        replayed_regional.source_indices
                        != output.evidence.regional_source_indices
                    ).sum()
                )
                valid_source_counts = _fixed_region_valid_counts(
                    output.source_valid_mask,
                    current_source_lattice_size,
                    current_block_size,
                )
                regional_attenuation_batches.append(
                    _batch_regional_lme_attenuation(
                        source_log_witness_probabilities=(
                            source_evidence.log_witness_probabilities
                        ),
                        source_log_nonwitness_probabilities=(
                            source_evidence.log_nonwitness_probabilities
                        ),
                        regional_log_witness_probabilities=(
                            replayed_regional.evidence.log_witness_probabilities
                        ),
                        regional_log_nonwitness_probabilities=(
                            replayed_regional.evidence.log_nonwitness_probabilities
                        ),
                        peak_source_indices=replayed_regional.source_indices,
                        regional_valid_mask=replayed_regional.valid_mask,
                        valid_source_counts=valid_source_counts,
                    )
                )

    labels = torch.cat(labels_batches)
    indices = torch.cat(indices_batches)
    checkpoint_predictions = torch.cat(model_predictions)
    transitions = torch.cat(transitions_batches)
    log_transitions = torch.cat(log_transitions_batches)
    stops = torch.cat(stops_batches)
    log_stops = torch.cat(log_stops_batches)
    dense_transitions = torch.cat(dense_transitions_batches)
    dense_log_transitions = torch.cat(dense_log_transitions_batches)
    dense_stops = torch.cat(dense_stops_batches)
    dense_log_stops = torch.cat(dense_log_stops_batches)
    model_classes = torch.cat(model_classes_batches)
    proof_sizes = torch.cat(proof_sizes_batches)
    witness_concentration = _merge_witness_concentration_batches(
        witness_concentration_batches
    )
    retained_overflow = torch.cat(overflow_batches)
    assert alpha_reference is not None

    # Historical decoding reconstructs log(advance) from the FP32 probability
    # tensor and is retained solely to reproduce the checkpoint. Exact audit
    # decoding consumes the paired log masses emitted by the same circuit, so
    # finite subnormal evidence cannot be confused with a structural zero.
    decision = proof_only_decisions(transitions, log_stops, transition_weights)
    dense_decision = proof_only_decisions(
        dense_transitions,
        dense_log_stops,
        transition_weights,
    )
    exact_decision = _proof_only_decisions_from_log_pairs(
        log_transitions,
        log_stops,
        transition_weights,
    )
    exact_dense_decision = _proof_only_decisions_from_log_pairs(
        dense_log_transitions,
        dense_log_stops,
        transition_weights,
    )
    decoder_specs = decision_rule_outputs(decision)
    dense_decoder_specs = decision_rule_outputs(dense_decision)
    exact_decoder_specs = decision_rule_outputs(exact_decision)
    exact_dense_decoder_specs = decision_rule_outputs(exact_dense_decision)
    decoder_metrics = {
        name: _classification_metrics(prediction, labels, cfg.n_classes, cumulative)
        for name, (prediction, cumulative) in decoder_specs.items()
    }
    dense_decoder_metrics = {
        name: _classification_metrics(prediction, labels, cfg.n_classes, cumulative)
        for name, (prediction, cumulative) in dense_decoder_specs.items()
    }
    exact_decoder_metrics = {
        name: _classification_metrics(prediction, labels, cfg.n_classes, cumulative)
        for name, (prediction, cumulative) in exact_decoder_specs.items()
    }
    exact_dense_decoder_metrics = {
        name: _classification_metrics(prediction, labels, cfg.n_classes, cumulative)
        for name, (prediction, cumulative) in exact_dense_decoder_specs.items()
    }
    checkpoint_prediction = decoder_specs[checkpoint_decision_rule][0]
    comparisons = {
        name: _comparison(checkpoint_prediction, prediction, labels)
        for name, (prediction, _cumulative) in decoder_specs.items()
        if name != checkpoint_decision_rule
    }
    dense_vs_projected_comparisons = {
        name: _comparison(
            exact_decoder_specs[name][0],
            exact_dense_decoder_specs[name][0],
            labels,
        )
        for name in exact_decoder_specs
    }
    projected_exact_vs_historical = {
        name: _comparison(
            decoder_specs[name][0],
            exact_decoder_specs[name][0],
            labels,
        )
        for name in decoder_specs
    }
    dense_exact_vs_historical = {
        name: _comparison(
            dense_decoder_specs[name][0],
            exact_dense_decoder_specs[name][0],
            labels,
        )
        for name in dense_decoder_specs
    }
    boundary_path_diagnostics = _transition_path_diagnostics(
        transitions,
        dense_transitions,
        labels,
        projected_log_transitions=log_transitions,
        projected_log_stops=log_stops,
        dense_log_transitions=dense_log_transitions,
        dense_log_stops=dense_log_stops,
    )
    proof_gap = dense_transitions - transitions
    sufficiency_limit = float(cfg.proof_epsilon)
    sufficiency_violation_count = int(
        (proof_gap > sufficiency_limit + 1e-6).sum()
    )
    projected_above_dense_count = int((proof_gap < -1e-6).sum())
    if proof_dense_transition_replay_max_error > 1e-7:
        raise RuntimeError(
            "proof.dense_transition does not replay the dense circuit output"
        )
    if proof_sufficiency_gap_replay_max_error > 1e-7:
        raise RuntimeError("serialized proof sufficiency gap is inconsistent")

    if regional_attenuation_batches:
        if regional_source_lattice_size is None or regional_block_size is None:
            raise RuntimeError("regional attenuation audit omitted geometry")
        regional_attenuation = _merge_regional_lme_attenuation_batches(
            regional_attenuation_batches
        )
        regional_attenuation_summary: dict[str, Any] = (
            _regional_lme_attenuation_summary(
                regional_attenuation,
                temperature=cfg.region_pool_temperature,
                source_lattice_size=regional_source_lattice_size,
                regional_block_size=regional_block_size,
                replay_max_probability_error=max(regional_replay_max_errors),
                replay_peak_index_mismatches=regional_replay_peak_mismatches,
            )
        )
    else:
        regional_attenuation_summary = {
            "available": False,
            "reason": (
                "checkpoint has no regional evidence compiler "
                "(region_grid_size=0)"
            ),
        }

    class_sum_error = max(
        float((decision.raw_class_probabilities.sum(dim=1) - 1.0).abs().max()),
        float(
            (decision.deweighted_class_probabilities.sum(dim=1) - 1.0)
            .abs()
            .max()
        ),
    )
    cumulative_violations = int(
        (
            decision.raw_cumulative_probabilities[:, 1:]
            > decision.raw_cumulative_probabilities[:, :-1] + 1e-7
        ).sum()
        + (
            decision.deweighted_cumulative_probabilities[:, 1:]
            > decision.deweighted_cumulative_probabilities[:, :-1] + 1e-7
        ).sum()
    )
    dense_class_sum_error = max(
        float(
            (
                dense_decision.raw_class_probabilities.sum(dim=1) - 1.0
            ).abs().max()
        ),
        float(
            (
                dense_decision.deweighted_class_probabilities.sum(dim=1) - 1.0
            ).abs().max()
        ),
    )
    dense_cumulative_violations = int(
        (
            dense_decision.raw_cumulative_probabilities[:, 1:]
            > dense_decision.raw_cumulative_probabilities[:, :-1] + 1e-7
        ).sum()
        + (
            dense_decision.deweighted_cumulative_probabilities[:, 1:]
            > dense_decision.deweighted_cumulative_probabilities[:, :-1] + 1e-7
        ).sum()
    )
    exact_class_sum_error = max(
        float(
            (exact_decision.raw_class_probabilities.sum(dim=1) - 1.0)
            .abs()
            .max()
        ),
        float(
            (exact_decision.deweighted_class_probabilities.sum(dim=1) - 1.0)
            .abs()
            .max()
        ),
        float(
            (exact_dense_decision.raw_class_probabilities.sum(dim=1) - 1.0)
            .abs()
            .max()
        ),
        float(
            (
                exact_dense_decision.deweighted_class_probabilities.sum(dim=1)
                - 1.0
            )
            .abs()
            .max()
        ),
    )
    exact_cumulative_violations = sum(
        int((cumulative[:, 1:] > cumulative[:, :-1] + 1e-7).sum())
        for cumulative in (
            exact_decision.raw_cumulative_probabilities,
            exact_decision.deweighted_cumulative_probabilities,
            exact_dense_decision.raw_cumulative_probabilities,
            exact_dense_decision.deweighted_cumulative_probabilities,
        )
    )
    raw_class_replay_error = float(
        (model_classes - decision.raw_class_probabilities).abs().max()
    )
    model_decision_replay_mismatches = _declared_decision_replay_mismatches(
        checkpoint_predictions,
        decoder_specs,
        checkpoint_decision_rule,
    )

    stored_metrics = checkpoint.get("metrics", {})
    reproduced = decoder_metrics[checkpoint_decision_rule]
    reproduction_differences: dict[str, float] = {}
    for key in ("acc", "qwk", "mae"):
        if key not in stored_metrics:
            raise ValueError(f"checkpoint validation metrics omit {key!r}")
        reproduction_differences[key] = abs(
            float(stored_metrics[key]) - float(reproduced[key])
        )
    metric_reproduction = (
        model_decision_replay_mismatches == 0
        and reproduction_differences["acc"] <= 1e-7
        and reproduction_differences["qwk"] <= 1e-6
        and reproduction_differences["mae"] <= 1e-7
    )
    if not metric_reproduction:
        raise RuntimeError(
            "checkpoint validation metrics were not reproduced with saved "
            f"decision rule {checkpoint_decision_rule!r}; absolute differences="
            f"{reproduction_differences}, declared-rule replay mismatches="
            f"{model_decision_replay_mismatches}. No audit artifacts were "
            "published."
        )

    label_counts = torch.bincount(labels, minlength=cfg.n_classes)
    summary: dict[str, Any] = {
        "schema": SCHEMA,
        "scope": "inner_validation_only",
        "outer_test_images_decoded": 0,
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_sha256,
        "checkpoint_epoch": int(checkpoint.get("epoch", -1)),
        "dataset": dataset_name,
        "fold": fold,
        "samples": int(labels.numel()),
        "true_class_counts": label_counts.tolist(),
        "checkpoint_decision_rule": checkpoint_decision_rule,
        "checkpoint_metric_reproduction": metric_reproduction,
        "checkpoint_metric_absolute_differences": reproduction_differences,
        "implementation_signature_match": implementation_match,
        "proof_exclusivity": {
            "no_feature_or_global_input_to_decoder": True,
            "decoder_inputs": [
                "selected_proof_transitions",
                "selected_proof_log_stop_probabilities",
                "training-fold boundary outcome weights",
            ],
            "validation_fitted_parameters": 0,
            "dense_path_is_diagnostic_only": True,
            "dense_path_location": (
                "same regional events and cardinality circuit immediately before "
                "deterministic proof selection"
            ),
        },
        "outcome_weight_correction": {
            "weight_order": ["stop", "advance"],
            "weights": transition_weights.tolist(),
            "boundary_logit_offsets_log_w_stop_over_w_advance": (
                transition_weights[:, 0].log()
                - transition_weights[:, 1].log()
            ).tolist(),
        },
        "probability_checks": {
            "all_finite": bool(
                torch.isfinite(decision.raw_class_probabilities).all()
                and torch.isfinite(decision.deweighted_class_probabilities).all()
            ),
            "class_sum_max_error": class_sum_error,
            "cumulative_monotonicity_violations": cumulative_violations,
            "raw_class_replay_max_error": raw_class_replay_error,
            "declared_model_decision_replay_mismatches": (
                model_decision_replay_mismatches
            ),
            # Deprecated v2 alias retained for downstream readers.  In v3 it
            # correctly replays the checkpoint's declared rule, not always
            # rounded_expected.
            "raw_model_decision_replay_mismatches": (
                model_decision_replay_mismatches
            ),
            "dense_all_finite": bool(
                torch.isfinite(dense_decision.raw_class_probabilities).all()
                and torch.isfinite(
                    dense_decision.deweighted_class_probabilities
                ).all()
            ),
            "dense_class_sum_max_error": dense_class_sum_error,
            "dense_cumulative_monotonicity_violations": (
                dense_cumulative_violations
            ),
            "exact_paired_log_class_sum_max_error": exact_class_sum_error,
            "exact_paired_log_cumulative_monotonicity_violations": (
                exact_cumulative_violations
            ),
            "log_probability_invariants": {
                "projected_transition": _log_probability_invariants(
                    transitions, log_transitions
                ),
                "projected_stop": _log_probability_invariants(
                    stops, log_stops
                ),
                "dense_pre_projection_transition": (
                    _log_probability_invariants(
                        dense_transitions, dense_log_transitions
                    )
                ),
                "dense_pre_projection_stop": _log_probability_invariants(
                    dense_stops, dense_log_stops
                ),
            },
            "projection_invariants": {
                "proof_dense_transition_replay_max_absolute_error": (
                    proof_dense_transition_replay_max_error
                ),
                "proof_sufficiency_gap_replay_max_absolute_error": (
                    proof_sufficiency_gap_replay_max_error
                ),
                "declared_sufficiency_tolerance": sufficiency_limit,
                "dense_minus_projected_above_tolerance_count": (
                    sufficiency_violation_count
                ),
                "projected_above_dense_beyond_1e-6_count": (
                    projected_above_dense_count
                ),
                "all_hold": bool(
                    proof_dense_transition_replay_max_error <= 1e-7
                    and proof_sufficiency_gap_replay_max_error <= 1e-7
                    and sufficiency_violation_count == 0
                    and projected_above_dense_count == 0
                ),
            },
        },
        # Retain the v2 top-level field as the projected checkpoint path for
        # existing analysis scripts.
        "decoders": decoder_metrics,
        "comparisons_to_checkpoint_decision": comparisons,
        "historical_checkpoint_probability_semantics": {
            "definition": (
                "log advance is reconstructed from the serialized/runtime FP32 "
                "transition probability; this exactly reproduces training-time "
                "checkpoint decisions but can discard a finite direct log mass "
                "after probability underflow"
            ),
            "projected_selected_proof_decoders": decoder_metrics,
            "dense_pre_projection_decoders": dense_decoder_metrics,
            "projected_exact_vs_historical_by_decoder": (
                projected_exact_vs_historical
            ),
            "dense_exact_vs_historical_by_decoder": (
                dense_exact_vs_historical
            ),
        },
        "prediction_path_comparison": {
            "decoder_semantics": (
                "exact paired log-advance/log-stop normalization with no "
                "probability-space round trip"
            ),
            "projected_selected_proof": {
                "is_checkpoint_structural_path": True,
                "uses_historical_checkpoint_rounding_semantics": False,
                "decoders": exact_decoder_metrics,
            },
            "dense_pre_projection_regional": {
                "is_checkpoint_structural_path": False,
                "uses_historical_checkpoint_rounding_semantics": False,
                "decoders": exact_dense_decoder_metrics,
            },
            "dense_vs_projected_by_decoder": dense_vs_projected_comparisons,
            "interpretation": (
                "If dense materially outperforms projected, proof selection is "
                "discarding useful regional evidence. If both are similarly weak, "
                "the bottleneck is upstream of proof selection."
            ),
        },
        "boundary_transition_comparison": boundary_path_diagnostics,
        "source_to_regional_lme_attenuation": regional_attenuation_summary,
        "hard_proof_diagnostics": _proof_diagnostics(
            proof_sizes,
            transitions,
            labels,
            witness_concentration,
            retained_overflow,
            alpha_reference,
        ),
        "audit_decision_rule": (
            f"The checkpoint decision is {checkpoint_decision_rule}. All other "
            "rows are fixed diagnostic alternatives; do not choose the best row "
            "post hoc on this validation audit."
        ),
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json_atomic(output_dir / "summary.json", summary)

    ordered = indices.argsort()
    predictions_path = output_dir / "predictions.csv"
    probability_names = [f"p{grade}" for grade in range(cfg.n_classes)]
    deweighted_probability_names = [
        f"p_deweighted{grade}" for grade in range(cfg.n_classes)
    ]
    cumulative_names = [f"q{boundary}" for boundary in range(cfg.n_classes - 1)]
    deweighted_cumulative_names = [
        f"q_deweighted{boundary}" for boundary in range(cfg.n_classes - 1)
    ]
    decision_names = list(decoder_specs)
    dense_probability_names = [
        f"dense_p{grade}" for grade in range(cfg.n_classes)
    ]
    dense_deweighted_probability_names = [
        f"dense_p_deweighted{grade}" for grade in range(cfg.n_classes)
    ]
    dense_cumulative_names = [
        f"dense_q{boundary}" for boundary in range(cfg.n_classes - 1)
    ]
    dense_deweighted_cumulative_names = [
        f"dense_q_deweighted{boundary}"
        for boundary in range(cfg.n_classes - 1)
    ]
    dense_decision_names = [f"dense_{name}" for name in dense_decoder_specs]
    exact_decision_names = [f"exact_{name}" for name in exact_decoder_specs]
    exact_dense_decision_names = [
        f"exact_dense_{name}" for name in exact_dense_decoder_specs
    ]
    fieldnames = [
        "sample_id",
        "true_grade",
        "raw_expected_grade",
        "deweighted_expected_grade",
        "dense_raw_expected_grade",
        "dense_deweighted_expected_grade",
        *probability_names,
        *deweighted_probability_names,
        *cumulative_names,
        *deweighted_cumulative_names,
        *decision_names,
        *dense_probability_names,
        *dense_deweighted_probability_names,
        *dense_cumulative_names,
        *dense_deweighted_cumulative_names,
        *dense_decision_names,
        *exact_decision_names,
        *exact_dense_decision_names,
    ]
    with predictions_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for position in ordered.tolist():
            item_index = int(indices[position])
            image_path, _ = validation_items[item_index]
            row: dict[str, Any] = {
                "sample_id": Path(image_path).stem,
                "true_grade": int(labels[position]),
                "raw_expected_grade": float(decision.raw_expected_grade[position]),
                "deweighted_expected_grade": float(
                    decision.deweighted_expected_grade[position]
                ),
                "dense_raw_expected_grade": float(
                    dense_decision.raw_expected_grade[position]
                ),
                "dense_deweighted_expected_grade": float(
                    dense_decision.deweighted_expected_grade[position]
                ),
            }
            row.update(
                {
                    name: float(decision.raw_class_probabilities[position, grade])
                    for grade, name in enumerate(probability_names)
                }
            )
            row.update(
                {
                    name: float(
                        decision.deweighted_class_probabilities[position, grade]
                    )
                    for grade, name in enumerate(deweighted_probability_names)
                }
            )
            row.update(
                {
                    name: float(
                        decision.raw_cumulative_probabilities[position, boundary]
                    )
                    for boundary, name in enumerate(cumulative_names)
                }
            )
            row.update(
                {
                    name: float(
                        decision.deweighted_cumulative_probabilities[
                            position, boundary
                        ]
                    )
                    for boundary, name in enumerate(deweighted_cumulative_names)
                }
            )
            row.update(
                {
                    name: int(prediction[position])
                    for name, (prediction, _cumulative) in decoder_specs.items()
                }
            )
            row.update(
                {
                    name: float(
                        dense_decision.raw_class_probabilities[position, grade]
                    )
                    for grade, name in enumerate(dense_probability_names)
                }
            )
            row.update(
                {
                    name: float(
                        dense_decision.deweighted_class_probabilities[
                            position, grade
                        ]
                    )
                    for grade, name in enumerate(
                        dense_deweighted_probability_names
                    )
                }
            )
            row.update(
                {
                    name: float(
                        dense_decision.raw_cumulative_probabilities[
                            position, boundary
                        ]
                    )
                    for boundary, name in enumerate(dense_cumulative_names)
                }
            )
            row.update(
                {
                    name: float(
                        dense_decision.deweighted_cumulative_probabilities[
                            position, boundary
                        ]
                    )
                    for boundary, name in enumerate(
                        dense_deweighted_cumulative_names
                    )
                }
            )
            row.update(
                {
                    f"dense_{name}": int(prediction[position])
                    for name, (prediction, _cumulative) in (
                        dense_decoder_specs.items()
                    )
                }
            )
            row.update(
                {
                    f"exact_{name}": int(prediction[position])
                    for name, (prediction, _cumulative) in (
                        exact_decoder_specs.items()
                    )
                }
            )
            row.update(
                {
                    f"exact_dense_{name}": int(prediction[position])
                    for name, (prediction, _cumulative) in (
                        exact_dense_decoder_specs.items()
                    )
                }
            )
            writer.writerow(row)

    print(
        f"MOSAIC decoder audit: dataset={dataset_name} fold={fold} "
        f"validation_n={len(validation_items)} "
        f"checkpoint_epoch={checkpoint.get('epoch')} "
        f"checkpoint_decision={checkpoint_decision_rule}"
    )
    print("\nHistorical projected path (checkpoint reproduction only)")
    print(
        "Decoder                         Acc  BalAcc  MacroF1     QWK     MAE"
    )
    for name, metrics in decoder_metrics.items():
        print(
            f"{name:30s} {metrics['acc']:6.2f} {metrics['balanced_acc']:7.2f} "
            f"{metrics['macro_f1']:8.4f} {metrics['qwk']:7.4f} "
            f"{metrics['mae']:7.4f}"
        )

    print("\nExact paired-log projected selected-proof path")
    print(
        "Decoder                         Acc  BalAcc  MacroF1     QWK     MAE  "
        "Changed-vs-hist"
    )
    for name, metrics in exact_decoder_metrics.items():
        comparison = projected_exact_vs_historical[name]
        print(
            f"{name:30s} {metrics['acc']:6.2f} {metrics['balanced_acc']:7.2f} "
            f"{metrics['macro_f1']:8.4f} {metrics['qwk']:7.4f} "
            f"{metrics['mae']:7.4f} {comparison['changed']:15d}"
        )

    print("\nExact paired-log dense pre-projection regional path")
    print(
        "Decoder                         Acc  BalAcc  MacroF1     QWK     MAE  "
        "Changed  Help  Harm"
    )
    for name, metrics in exact_dense_decoder_metrics.items():
        comparison = dense_vs_projected_comparisons[name]
        print(
            f"{name:30s} {metrics['acc']:6.2f} {metrics['balanced_acc']:7.2f} "
            f"{metrics['macro_f1']:8.4f} {metrics['qwk']:7.4f} "
            f"{metrics['mae']:7.4f} {comparison['changed']:8d} "
            f"{comparison['current_wrong_alternative_correct']:5d} "
            f"{comparison['current_correct_alternative_wrong']:5d}"
        )

    print("\nBoundary discrimination and projection delta")
    print(
        "Boundary  Risk-N  AUROC-proj  AUROC-dense  Delta-AUC  "
        "Risk-mean|delta|  Risk-max|delta|"
    )
    for row in boundary_path_diagnostics["boundaries"]:
        projected_auc = row["projected_auroc"]
        dense_auc = row["dense_pre_projection_auroc"]
        auc_delta = row["dense_minus_projected_auroc"]
        delta_summary = row["dense_minus_projected"]["at_risk"]
        print(
            f"Y>{row['boundary']:<5d} "
            f"{row['at_risk_count']:6d} "
            f"{projected_auc if projected_auc is not None else float('nan'):11.4f} "
            f"{dense_auc if dense_auc is not None else float('nan'):12.4f} "
            f"{auc_delta if auc_delta is not None else float('nan'):10.4f} "
            f"{delta_summary['mean_absolute']:17.6f} "
            f"{delta_summary['maximum_absolute']:11.6f}"
        )

    if regional_attenuation_summary["available"]:
        print("\nSource maximum to normalized regional LogMeanExp attenuation")
        print("Boundary  Peak-prob  LME-prob  Mean-delta-prob  Mean-delta-logit")
        for row in regional_attenuation_summary["boundaries"]:
            print(
                f"Y>{row['boundary']:<5d} "
                f"{row['source_peak_probability']['mean']:10.4f} "
                f"{row['regional_lme_probability']['mean']:9.4f} "
                f"{row['source_peak_minus_regional_lme_probability']['mean']:16.4f} "
                f"{row['source_peak_minus_regional_lme_logit']['mean']:16.4f}"
            )
    else:
        print("\nSource-to-regional attenuation unavailable: "
              f"{regional_attenuation_summary['reason']}")
    print(f"Checkpoint metrics reproduced: {metric_reproduction}")
    print(f"Summary: {output_dir / 'summary.json'}")
    print(f"Predictions: {predictions_path}")


if __name__ == "__main__":
    main()
