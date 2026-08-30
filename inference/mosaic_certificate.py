"""Replayable JSON certificates for MOSAIC predictions.

The certificate records the dense regional witness ledger which determines
the proof, as well as the selected proof that is the exclusive numerical
prediction path.  :func:`verify_mosaic_certificate` reconstructs the count
distributions, transitions, continuation cascade, minimum-prefix conditions,
and fixed-proof pivotalities from the serialized values.

The utility deliberately serializes receptive-field support metadata rather
than drawing an interpolated heatmap.  A lattice centre is not a pixel-level
lesion boundary.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import fields, is_dataclass
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

import torch

from models.mosaic import (
    OrdinalCardinalityCircuit,
    ProofProjectionResult,
    TruncatedPoissonBinomial,
    continuation_probabilities,
    fixed_proof_pivotality,
)


SCHEMA_VERSION = "mosaic-certificate-v2"
DEFAULT_REPLAY_ATOL = 2e-5
DEFAULT_REPLAY_RTOL = 2e-5


class CertificateReplayError(ValueError):
    """Raised when a serialized certificate cannot reproduce its prediction."""


def _payload_sha256(certificate: Mapping[str, Any]) -> str:
    """Hash the complete canonical payload except its own integrity record."""

    payload = {key: value for key, value in certificate.items() if key != "integrity"}
    encoded = json.dumps(
        _json_safe(payload),
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _field(container: Any, name: str) -> Any:
    if isinstance(container, Mapping):
        if name not in container:
            raise KeyError(f"missing required MOSAIC output field {name!r}")
        return container[name]
    if hasattr(container, name):
        return getattr(container, name)
    raise KeyError(f"missing required MOSAIC output field {name!r}")


def _optional_field(container: Any, name: str, default: Any = None) -> Any:
    if isinstance(container, Mapping):
        return container.get(name, default)
    return getattr(container, name, default)


def _tensor(value: Any, name: str) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value
    try:
        return torch.as_tensor(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be tensor-like") from exc


def _sample(
    value: Any,
    name: str,
    *,
    sample_index: int,
    batched_ndim: int,
) -> torch.Tensor:
    tensor = _tensor(value, name)
    if tensor.ndim == batched_ndim:
        if not 0 <= sample_index < tensor.shape[0]:
            raise IndexError(
                f"sample_index {sample_index} is outside {name} batch of size "
                f"{tensor.shape[0]}"
            )
        return tensor[sample_index]
    if tensor.ndim == batched_ndim - 1 and sample_index == 0:
        return tensor
    raise ValueError(
        f"{name} must have {batched_ndim} batched or {batched_ndim - 1} "
        f"unbatched dimensions; got {tuple(tensor.shape)}"
    )


def _json_safe(value: Any) -> Any:
    """Recursively convert tensors/dataclasses/scalars into strict JSON data."""

    if isinstance(value, torch.Tensor):
        value = value.detach().cpu()
        return value.item() if value.ndim == 0 else value.tolist()
    if is_dataclass(value) and not isinstance(value, type):
        return {field.name: _json_safe(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return value
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("certificate values must be finite")
        return float(value)
    # NumPy scalar/array support without making NumPy a required import.
    if hasattr(value, "tolist"):
        return _json_safe(value.tolist())
    if hasattr(value, "item"):
        return _json_safe(value.item())
    raise TypeError(f"value of type {type(value).__name__} is not JSON serializable")


def _score_ledger(
    witness_ledger: torch.Tensor,
    alpha: torch.Tensor,
    selected_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Score all boundaries from a ``(P,B)`` witness ledger."""

    if witness_ledger.ndim != 2:
        raise ValueError("witness ledger must have shape (P, B)")
    p, boundaries = witness_ledger.shape
    if alpha.ndim != 2 or alpha.shape[0] != boundaries:
        raise ValueError("alpha must have shape (B, R)")
    probabilities = witness_ledger.float()
    if selected_mask is not None:
        if selected_mask.shape != witness_ledger.shape:
            raise ValueError("selected mask and witness ledger shapes differ")
        probabilities = probabilities * selected_mask.to(probabilities.dtype)
    count = TruncatedPoissonBinomial(
        max_count=alpha.shape[1], implementation="block_tree", block_size=64
    )
    distributions = count(probabilities.transpose(0, 1).contiguous())
    scores, _stops, _tails = OrdinalCardinalityCircuit.score_distributions(
        distributions, alpha.float()
    )
    return scores, distributions


def _log_stop_ledger(
    witness_ledger: torch.Tensor,
    alpha: torch.Tensor,
    log_alpha: torch.Tensor,
    selected_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Replay stable log stops and their scaled lower-tail state."""

    probabilities = witness_ledger.float()
    if selected_mask is not None:
        probabilities = probabilities * selected_mask.to(probabilities.dtype)
    count = TruncatedPoissonBinomial(
        max_count=alpha.shape[1], implementation="block_tree", block_size=64
    )
    log_conditional, log_survival = count.scaled_log_lower_tail(
        probabilities.transpose(0, 1).contiguous()
    )
    log_stop = OrdinalCardinalityCircuit.score_log_stops(
        log_conditional, log_survival, alpha.float(), log_alpha.float()
    )
    return log_stop, log_conditional, log_survival


def _lattice_geometry(metadata: Mapping[str, Any], num_cells: int) -> dict[str, Any]:
    """Validate and extract mandatory certificate geometry."""

    try:
        height, width = (int(x) for x in metadata["lattice_size"])
        input_height, input_width = (int(x) for x in metadata["input_size"])
        receptive = metadata["receptive_field"]
        stride = int(receptive["output_stride"])
        rf = float(receptive["receptive_field"])
        offset = float(receptive["center_offset"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("malformed receptive-field lattice metadata") from exc
    if min(height, width, input_height, input_width, stride) <= 0:
        raise ValueError("lattice, input, and stride dimensions must be positive")
    if not math.isfinite(rf) or not math.isfinite(offset) or rf <= 0.0:
        raise ValueError("receptive field and center offset must be finite")
    if height * width != num_cells:
        raise ValueError(
            f"lattice metadata describes {height * width} cells but ledger has {num_cells}"
        )
    if math.ceil(input_height / stride) != height or math.ceil(input_width / stride) != width:
        raise ValueError("lattice dimensions are inconsistent with input size and stride")
    return {
        "height": height,
        "width": width,
        "input_height": input_height,
        "input_width": input_width,
        "stride": stride,
        "receptive_field": rf,
        "offset": offset,
    }


def _cell_record(
    index: int,
    boundary: int,
    witness: torch.Tensor,
    pivotality: torch.Tensor,
    geometry: dict[str, Any],
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "index": int(index),
        "witness_probability": float(witness[index, boundary]),
        "fixed_proof_pivotality": float(pivotality[index, boundary]),
    }
    row, column = divmod(index, geometry["width"])
    center_y = geometry["offset"] + row * geometry["stride"]
    center_x = geometry["offset"] + column * geometry["stride"]
    half_extent = geometry["receptive_field"] / 2.0
    record.update(
        {
            "row": int(row),
            "column": int(column),
            "center_yx": [float(center_y), float(center_x)],
            # Half-open support box, clipped to the actual canvas.  It is
            # receptive-field support, not a lesion segmentation box.
            "receptive_field_box_yxyx": [
                float(max(0.0, center_y - half_extent)),
                float(max(0.0, center_x - half_extent)),
                float(min(geometry["input_height"], center_y + half_extent)),
                float(min(geometry["input_width"], center_x + half_extent)),
            ],
        }
    )
    return record


def build_mosaic_certificate(
    output: Any,
    *,
    lattice_metadata: Any = None,
    valid_mask: torch.Tensor | Sequence[bool] | None = None,
    sample_index: int = 0,
    sample_id: str | int | None = None,
    sufficiency_tolerance: float = 0.02,
    complement_suppression: float = 0.5,
    comparison_atol: float = 1e-7,
    replay_atol: float = DEFAULT_REPLAY_ATOL,
    replay_rtol: float = DEFAULT_REPLAY_RTOL,
    provenance: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build one strict JSON-safe certificate from a MOSAIC output.

    ``output`` may be :class:`models.mosaic.MOSAICOutput`, the image-level
    ``MOSAICModelOutput`` wrapper, or a mapping with the same fields.  The
    wrapper supplies its lattice and valid mask automatically.  Receptive-field
    metadata remains mandatory so a consumer cannot mistake lattice spacing
    for the true image support of a witness.
    """

    wrapped_evidence = _optional_field(output, "evidence")
    if wrapped_evidence is not None:
        if lattice_metadata is None:
            lattice_metadata = _optional_field(output, "lattice")
        if valid_mask is None:
            valid_mask = _optional_field(output, "valid_mask")
        output = wrapped_evidence
    if lattice_metadata is None:
        raise ValueError("lattice_metadata is required for a faithful certificate")
    if sufficiency_tolerance < 0.0:
        raise ValueError("sufficiency_tolerance must be non-negative")
    if not 0.0 <= complement_suppression <= 1.0:
        raise ValueError("complement_suppression must lie in [0, 1]")
    if comparison_atol < 0.0:
        raise ValueError("comparison_atol must be non-negative")
    if replay_atol < 0.0 or replay_rtol < 0.0:
        raise ValueError("replay tolerances must be non-negative")
    if provenance is not None and not isinstance(provenance, Mapping):
        raise TypeError("provenance must be a mapping")

    proof = _field(output, "proof")
    witnesses = _sample(
        _field(output, "witness_probabilities"),
        "witness_probabilities",
        sample_index=sample_index,
        batched_ndim=3,
    ).detach().float().cpu()
    local_states = _sample(
        _field(output, "local_state_probabilities"),
        "local_state_probabilities",
        sample_index=sample_index,
        batched_ndim=3,
    ).detach().float().cpu()
    selected_mask = _sample(
        _field(proof, "selected_mask"),
        "proof.selected_mask",
        sample_index=sample_index,
        batched_ndim=3,
    ).detach().bool().cpu()
    sorted_indices = _sample(
        _field(proof, "sorted_indices"),
        "proof.sorted_indices",
        sample_index=sample_index,
        batched_ndim=3,
    ).detach().long().cpu()
    proof_size = _sample(
        _field(proof, "proof_size"),
        "proof.proof_size",
        sample_index=sample_index,
        batched_ndim=2,
    ).detach().long().cpu()
    dense = _sample(
        _field(output, "dense_transitions"),
        "dense_transitions",
        sample_index=sample_index,
        batched_ndim=2,
    ).detach().float().cpu()
    transitions = _sample(
        _field(output, "transitions"),
        "transitions",
        sample_index=sample_index,
        batched_ndim=2,
    ).detach().float().cpu()
    dense_stop = _sample(
        _field(output, "dense_stop_probabilities"),
        "dense_stop_probabilities",
        sample_index=sample_index,
        batched_ndim=2,
    ).detach().float().cpu()
    stop = _sample(
        _field(output, "stop_probabilities"),
        "stop_probabilities",
        sample_index=sample_index,
        batched_ndim=2,
    ).detach().float().cpu()
    dense_log_stop = _sample(
        _field(output, "dense_log_stop_probabilities"),
        "dense_log_stop_probabilities",
        sample_index=sample_index,
        batched_ndim=2,
    ).detach().float().cpu()
    log_stop = _sample(
        _field(output, "log_stop_probabilities"),
        "log_stop_probabilities",
        sample_index=sample_index,
        batched_ndim=2,
    ).detach().float().cpu()
    cumulative = _sample(
        _field(output, "cumulative_probabilities"),
        "cumulative_probabilities",
        sample_index=sample_index,
        batched_ndim=2,
    ).detach().float().cpu()
    classes = _sample(
        _field(output, "class_probabilities"),
        "class_probabilities",
        sample_index=sample_index,
        batched_ndim=2,
    ).detach().float().cpu()
    expected = _sample(
        _field(output, "expected_grade"),
        "expected_grade",
        sample_index=sample_index,
        batched_ndim=1,
    ).detach().float().cpu()
    predicted = _sample(
        _field(output, "predicted_grade"),
        "predicted_grade",
        sample_index=sample_index,
        batched_ndim=1,
    ).detach().long().cpu()
    argmax_value = _optional_field(output, "argmax_grade")
    if argmax_value is None:
        argmax_grade = classes.argmax().long()
    else:
        argmax_grade = _sample(
            argmax_value,
            "argmax_grade",
            sample_index=sample_index,
            batched_ndim=1,
        ).detach().long().cpu()

    alpha_value = _tensor(_field(output, "alpha"), "alpha").detach().float().cpu()
    log_alpha_value = _tensor(
        _field(output, "log_alpha"), "log_alpha"
    ).detach().float().cpu()
    if alpha_value.ndim == 3:
        alpha = alpha_value[sample_index]
        log_alpha = log_alpha_value[sample_index]
    elif alpha_value.ndim == 2:
        alpha = alpha_value
        log_alpha = log_alpha_value
    else:
        raise ValueError("alpha must have shape (B,R) or (N,B,R)")
    if log_alpha.shape != alpha.shape:
        raise ValueError("log_alpha must have the same shape as alpha")

    retained_distribution = _sample(
        _field(proof, "retained_distribution"),
        "proof.retained_distribution",
        sample_index=sample_index,
        batched_ndim=3,
    ).detach().float().cpu()
    complement_distribution = _sample(
        _field(proof, "complement_distribution"),
        "proof.complement_distribution",
        sample_index=sample_index,
        batched_ndim=3,
    ).detach().float().cpu()
    complement_score = _sample(
        _field(proof, "complement_transition"),
        "proof.complement_transition",
        sample_index=sample_index,
        batched_ndim=2,
    ).detach().float().cpu()
    sufficiency_gap = _sample(
        _field(proof, "sufficiency_gap"),
        "proof.sufficiency_gap",
        sample_index=sample_index,
        batched_ndim=2,
    ).detach().float().cpu()
    complement_drop = _sample(
        _field(proof, "complement_drop"),
        "proof.complement_drop",
        sample_index=sample_index,
        batched_ndim=2,
    ).detach().float().cpu()

    if witnesses.ndim != 2 or witnesses.shape[1] != transitions.numel():
        raise ValueError("sample witness ledger must have shape (P, K-1)")
    p, boundaries = witnesses.shape
    if local_states.shape != (p, boundaries + 1):
        raise ValueError("local state probabilities and witness ledger disagree")
    if selected_mask.shape != witnesses.shape:
        raise ValueError("selected mask and witness ledger disagree")
    if alpha.shape[0] != boundaries:
        raise ValueError("alpha and witness ledger boundary counts disagree")

    if valid_mask is None:
        valid = torch.ones(p, dtype=torch.bool)
        valid_source = "assumed_all_cells_valid"
    else:
        valid_tensor = _tensor(valid_mask, "valid_mask")
        if valid_tensor.ndim == 2:
            valid_tensor = valid_tensor[sample_index]
        if valid_tensor.ndim != 1 or valid_tensor.numel() != p:
            raise ValueError(f"valid_mask must resolve to shape ({p},)")
        valid = valid_tensor.detach().bool().cpu()
        valid_source = "provided"
    if (selected_mask & ~valid[:, None]).any():
        raise ValueError("proof selects a cell outside the valid retinal field")

    pivotality_value = _optional_field(output, "pivotality")
    if pivotality_value is None:
        # The public function is exact and uses the same fixed certificate; no
        # image re-evaluation or gradient approximation is introduced.  Build
        # a one-sample proof object so this path also supports a plain mapping
        # (including a JSON-like mapping produced from the output dataclass).
        sample_proof = ProofProjectionResult(
            selected_mask=selected_mask.unsqueeze(0),
            sorted_indices=sorted_indices.unsqueeze(0),
            proof_size=proof_size.unsqueeze(0),
            dense_transition=dense.unsqueeze(0),
            projected_transition=transitions.unsqueeze(0),
            complement_transition=complement_score.unsqueeze(0),
            retained_distribution=retained_distribution.unsqueeze(0),
            complement_distribution=complement_distribution.unsqueeze(0),
            sufficiency_gap=sufficiency_gap.unsqueeze(0),
            complement_drop=complement_drop.unsqueeze(0),
        )
        all_pivotality = fixed_proof_pivotality(
            witnesses.unsqueeze(0),
            sample_proof,
            alpha,
            alpha.shape[1],
        )
        pivotality = all_pivotality[0].detach().float().cpu()
    else:
        pivotality = _sample(
            pivotality_value,
            "pivotality",
            sample_index=sample_index,
            batched_ndim=3,
        ).detach().float().cpu()

    metadata = _json_safe(lattice_metadata)
    if not isinstance(metadata, Mapping):
        raise TypeError("lattice_metadata must serialize to an object")
    geometry = _lattice_geometry(metadata, p)
    _replay_dense_log_stop, dense_log_conditional, dense_log_survival = (
        _log_stop_ledger(witnesses, alpha, log_alpha)
    )
    _replay_retained_log_stop, retained_log_conditional, retained_log_survival = (
        _log_stop_ledger(witnesses, alpha, log_alpha, selected_mask)
    )

    selected_indices: list[list[int]] = []
    selected_cells: list[list[dict[str, Any]]] = []
    diagnostics: list[dict[str, Any]] = []
    for boundary in range(boundaries):
        ids = torch.nonzero(selected_mask[:, boundary], as_tuple=False).flatten().tolist()
        # Report in descending witness order, matching the canonical prefix.
        rank = {int(index): position for position, index in enumerate(sorted_indices[boundary].tolist())}
        ids.sort(key=lambda index: rank[index])
        selected_indices.append([int(index) for index in ids])
        selected_cells.append(
            [
                _cell_record(index, boundary, witnesses, pivotality, geometry)
                for index in ids
            ]
        )
        target = max(float(dense[boundary]) - sufficiency_tolerance, 0.0)
        required_drop = complement_suppression * target
        diagnostics.append(
            {
                "boundary": boundary,
                "dense_score": float(dense[boundary]),
                "retained_score": float(transitions[boundary]),
                "complement_score": float(complement_score[boundary]),
                "sufficiency_gap": float(sufficiency_gap[boundary]),
                "sufficiency_target": target,
                "sufficiency_satisfied": bool(
                    float(transitions[boundary]) + comparison_atol >= target
                ),
                "complement_drop": float(complement_drop[boundary]),
                "required_complement_drop": required_drop,
                "complement_suppression_satisfied": bool(
                    float(complement_drop[boundary]) + comparison_atol >= required_drop
                ),
            }
        )

    certificate = {
        "schema_version": SCHEMA_VERSION,
        "sample_id": sample_id,
        "sample_index": int(sample_index),
        "provenance": {} if provenance is None else _json_safe(provenance),
        "prediction": {
            "predicted_grade": int(predicted),
            "decision_rule": "rounded_expected_grade",
            "class_argmax_grade": int(argmax_grade),
            "expected_grade": float(expected),
            "class_probabilities": classes,
            "cumulative_probabilities": cumulative,
            "projected_transitions": transitions,
            "projected_stop_probabilities": stop,
            "dense_transitions": dense,
            "dense_stop_probabilities": dense_stop,
            "projected_log_stop_probabilities": log_stop,
            "dense_log_stop_probabilities": dense_log_stop,
        },
        "cardinality": {
            "alpha": alpha,
            "log_alpha": log_alpha,
            "max_count": int(alpha.shape[1]),
            "dense_log_conditional_low_distribution": dense_log_conditional,
            "dense_log_low_survival": dense_log_survival,
            "retained_log_conditional_low_distribution": retained_log_conditional,
            "retained_log_low_survival": retained_log_survival,
        },
        "proof_rule": {
            "sufficiency_tolerance": float(sufficiency_tolerance),
            "complement_suppression": float(complement_suppression),
            "comparison_atol": float(comparison_atol),
            "selection": "stable_descending_top_prefix",
        },
        "numerical_contract": {
            # CPU/GPU FP32 reduction orders are not bit-identical at a
            # 12,544-cell lattice.  These bounds are twice the observed
            # meaningful-regime serial/tree envelope and are distinct from
            # the projector's comparison_atol above.
            "replay_atol": float(replay_atol),
            "replay_rtol": float(replay_rtol),
            "arithmetic": "fp32_scaled_log_lower_tail_poisson_binomial",
        },
        "dense_ledger": {
            "witness_probabilities": witnesses,
            "local_state_probabilities": local_states,
            "valid_mask": valid,
            "valid_mask_source": valid_source,
        },
        "proof": {
            "proof_sizes": proof_size,
            "selected_indices": selected_indices,
            "selected_cells": selected_cells,
            "retained_scores": transitions,
            "complement_scores": complement_score,
            "retained_count_distributions": retained_distribution,
            "complement_count_distributions": complement_distribution,
            "sufficiency_gaps": sufficiency_gap,
            "complement_drops": complement_drop,
            "fixed_proof_pivotality": pivotality,
            "diagnostics": diagnostics,
        },
        "receptive_field_metadata": metadata,
        "interpretation_scope": (
            "Fine-grid computational evidence with the serialized receptive-field "
            "support; not pixel segmentation or a named-lesion annotation."
        ),
    }
    # A round-trip with allow_nan=False is the final strict JSON-safety check.
    safe = _json_safe(certificate)
    safe["integrity"] = {
        "algorithm": "sha256",
        "payload_sha256": _payload_sha256(safe),
    }
    json.dumps(safe, allow_nan=False)
    return safe


def certificate_to_json(certificate: Mapping[str, Any], *, indent: int = 2) -> str:
    """Serialize a certificate while rejecting NaN and infinity."""

    return json.dumps(_json_safe(certificate), indent=indent, allow_nan=False, sort_keys=True)


def save_mosaic_certificate(
    certificate: Mapping[str, Any], path: str | Path, *, indent: int = 2
) -> None:
    """Write a strict JSON certificate to ``path``."""

    destination = Path(path)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(certificate_to_json(certificate, indent=indent) + "\n")
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def load_mosaic_certificate(path: str | Path) -> dict[str, Any]:
    """Load a JSON certificate without performing replay verification."""

    value = json.loads(Path(path).read_text())
    if not isinstance(value, dict):
        raise ValueError("a MOSAIC certificate must be a JSON object")
    return value


def _close_check(
    checks: dict[str, bool],
    errors: dict[str, float],
    name: str,
    actual: torch.Tensor,
    expected: torch.Tensor,
    atol: float,
    rtol: float,
) -> None:
    if actual.shape != expected.shape:
        checks[name] = False
        errors[name] = math.inf
        return
    difference = (actual.float() - expected.float()).abs()
    errors[name] = float(difference.max()) if difference.numel() else 0.0
    checks[name] = bool(torch.allclose(actual.float(), expected.float(), atol=atol, rtol=rtol))


def verify_mosaic_certificate(
    certificate: Mapping[str, Any],
    *,
    atol: float | None = None,
    rtol: float | None = None,
    raise_on_error: bool = False,
) -> dict[str, Any]:
    """Replay and verify a serialized MOSAIC proof certificate.

    The verifier establishes score replay and minimum top-prefix feasibility
    for the *serialized witness ledger*.  It does not claim that those
    witnesses correspond to clinically named lesions.
    """

    if certificate.get("schema_version") != SCHEMA_VERSION:
        if certificate.get("schema_version") == "mosaic-certificate-v1":
            raise ValueError(
                "mosaic-certificate-v1 has no stable log-stop trace; "
                "regenerate it with the v2 exporter"
            )
        raise ValueError(
            f"unsupported certificate schema {certificate.get('schema_version')!r}"
        )
    numerical = certificate.get("numerical_contract", {})
    if not isinstance(numerical, Mapping):
        raise ValueError("certificate numerical_contract must be an object")
    if atol is None:
        atol = float(numerical.get("replay_atol", DEFAULT_REPLAY_ATOL))
    if rtol is None:
        rtol = float(numerical.get("replay_rtol", DEFAULT_REPLAY_RTOL))
    if atol < 0.0 or rtol < 0.0 or not math.isfinite(atol) or not math.isfinite(rtol):
        raise ValueError("replay tolerances must be non-negative")

    integrity = certificate.get("integrity", {})
    integrity_valid = bool(
        isinstance(integrity, Mapping)
        and integrity.get("algorithm") == "sha256"
        and integrity.get("payload_sha256") == _payload_sha256(certificate)
    )

    prediction = certificate["prediction"]
    cardinality = certificate["cardinality"]
    ledger = certificate["dense_ledger"]
    proof = certificate["proof"]
    rule = certificate["proof_rule"]

    witnesses = torch.tensor(ledger["witness_probabilities"], dtype=torch.float32)
    local_states = torch.tensor(
        ledger["local_state_probabilities"], dtype=torch.float32
    )
    valid = torch.tensor(ledger["valid_mask"], dtype=torch.bool)
    alpha = torch.tensor(cardinality["alpha"], dtype=torch.float32)
    log_alpha = torch.tensor(cardinality["log_alpha"], dtype=torch.float32)
    selected_indices = proof["selected_indices"]
    p, boundaries = witnesses.shape
    if (
        valid.shape != (p,)
        or alpha.shape[0] != boundaries
        or log_alpha.shape != alpha.shape
        or local_states.shape != (p, boundaries + 1)
    ):
        raise ValueError("certificate ledger, valid mask, and alpha shapes disagree")
    raw_witnesses = witnesses
    witnesses = raw_witnesses * valid[:, None].float()

    geometry = None
    try:
        metadata_value = certificate["receptive_field_metadata"]
        if not isinstance(metadata_value, Mapping):
            raise ValueError("receptive-field metadata must be an object")
        geometry = _lattice_geometry(metadata_value, p)
        geometry_valid = True
    except (KeyError, TypeError, ValueError):
        geometry_valid = False

    selected_mask = torch.zeros_like(witnesses, dtype=torch.bool)
    duplicate_or_invalid = False
    for boundary, ids_value in enumerate(selected_indices):
        ids = [int(index) for index in ids_value]
        if len(ids) != len(set(ids)):
            duplicate_or_invalid = True
        if any(index < 0 or index >= p or not bool(valid[index]) for index in ids):
            duplicate_or_invalid = True
            continue
        selected_mask[ids, boundary] = True

    dense_score, dense_distribution = _score_ledger(witnesses, alpha)
    retained_score, retained_distribution = _score_ledger(witnesses, alpha, selected_mask)
    complement_score, complement_distribution = _score_ledger(
        witnesses, alpha, (~selected_mask) & valid[:, None]
    )
    dense_log_stop, dense_log_conditional, dense_log_survival = _log_stop_ledger(
        witnesses, alpha, log_alpha
    )
    retained_log_stop, retained_log_conditional, retained_log_survival = (
        _log_stop_ledger(witnesses, alpha, log_alpha, selected_mask)
    )
    _dense_continue, dense_stop, _dense_tails = (
        OrdinalCardinalityCircuit.score_distributions(dense_distribution, alpha)
    )
    _retained_continue, retained_stop, _retained_tails = (
        OrdinalCardinalityCircuit.score_distributions(retained_distribution, alpha)
    )
    cumulative, classes = continuation_probabilities(retained_score, retained_stop)
    class_axis = torch.arange(classes.numel(), dtype=classes.dtype)
    expected_grade = (classes * class_axis).sum()

    checks: dict[str, bool] = {
        "integrity_sha256": integrity_valid,
        "numerical_contract": bool(
            numerical.get("arithmetic")
            == "fp32_scaled_log_lower_tail_poisson_binomial"
            and "replay_atol" in numerical
            and "replay_rtol" in numerical
        ),
        "receptive_field_metadata": geometry_valid,
        "selected_indices_valid": not duplicate_or_invalid,
    }
    errors: dict[str, float] = {}
    checks["invalid_cells_are_normal"] = bool(
        torch.all(raw_witnesses[~valid].abs() <= atol)
        and torch.all(local_states[~valid, 0] >= 1.0 - atol)
        and torch.all(local_states[~valid, 1:].abs() <= atol)
    )
    checks["alpha_distribution"] = bool(
        (alpha >= -atol).all()
        and torch.allclose(
            alpha.sum(dim=-1), torch.ones(boundaries), atol=atol, rtol=rtol
        )
        and int(cardinality["max_count"]) == alpha.shape[1]
        and torch.allclose(
            torch.softmax(log_alpha, dim=-1), alpha, atol=atol, rtol=rtol
        )
    )
    checks["local_state_distributions"] = bool(
        (local_states >= -atol).all()
        and torch.allclose(
            local_states.sum(dim=-1), torch.ones(p), atol=atol, rtol=rtol
        )
    )
    replay_witnesses = torch.flip(
        torch.cumsum(
            torch.flip(local_states[:, 1:], dims=(-1,)), dim=-1
        ),
        dims=(-1,),
    )
    replay_witnesses = replay_witnesses * valid[:, None].float()
    _close_check(
        checks,
        errors,
        "nested_witness_ledger",
        witnesses,
        replay_witnesses,
        atol,
        rtol,
    )
    _close_check(
        checks,
        errors,
        "dense_log_stop_probabilities",
        dense_log_stop,
        torch.tensor(prediction["dense_log_stop_probabilities"]),
        atol,
        rtol,
    )
    _close_check(
        checks,
        errors,
        "projected_log_stop_probabilities",
        retained_log_stop,
        torch.tensor(prediction["projected_log_stop_probabilities"]),
        atol,
        rtol,
    )
    _close_check(
        checks,
        errors,
        "dense_log_conditional_low_distribution",
        dense_log_conditional,
        torch.tensor(cardinality["dense_log_conditional_low_distribution"]),
        atol,
        rtol,
    )
    _close_check(
        checks,
        errors,
        "dense_log_low_survival",
        dense_log_survival,
        torch.tensor(cardinality["dense_log_low_survival"]),
        atol,
        rtol,
    )
    _close_check(
        checks,
        errors,
        "retained_log_conditional_low_distribution",
        retained_log_conditional,
        torch.tensor(cardinality["retained_log_conditional_low_distribution"]),
        atol,
        rtol,
    )
    _close_check(
        checks,
        errors,
        "retained_log_low_survival",
        retained_log_survival,
        torch.tensor(cardinality["retained_log_low_survival"]),
        atol,
        rtol,
    )
    _close_check(
        checks,
        errors,
        "dense_transitions",
        dense_score,
        torch.tensor(prediction["dense_transitions"]),
        atol,
        rtol,
    )
    _close_check(
        checks,
        errors,
        "projected_transitions",
        retained_score,
        torch.tensor(prediction["projected_transitions"]),
        atol,
        rtol,
    )
    _close_check(
        checks,
        errors,
        "dense_stop_probabilities",
        dense_stop,
        torch.tensor(prediction["dense_stop_probabilities"]),
        atol,
        rtol,
    )
    _close_check(
        checks,
        errors,
        "projected_stop_probabilities",
        retained_stop,
        torch.tensor(prediction["projected_stop_probabilities"]),
        atol,
        rtol,
    )
    _close_check(
        checks,
        errors,
        "proof_retained_scores",
        retained_score,
        torch.tensor(proof["retained_scores"]),
        atol,
        rtol,
    )
    _close_check(
        checks,
        errors,
        "complement_scores",
        complement_score,
        torch.tensor(proof["complement_scores"]),
        atol,
        rtol,
    )
    _close_check(
        checks,
        errors,
        "retained_distributions",
        retained_distribution,
        torch.tensor(proof["retained_count_distributions"]),
        atol,
        rtol,
    )
    _close_check(
        checks,
        errors,
        "complement_distributions",
        complement_distribution,
        torch.tensor(proof["complement_count_distributions"]),
        atol,
        rtol,
    )
    _close_check(
        checks,
        errors,
        "cumulative_probabilities",
        cumulative,
        torch.tensor(prediction["cumulative_probabilities"]),
        atol,
        rtol,
    )
    _close_check(
        checks,
        errors,
        "class_probabilities",
        classes,
        torch.tensor(prediction["class_probabilities"]),
        atol,
        rtol,
    )
    _close_check(
        checks,
        errors,
        "expected_grade",
        expected_grade.reshape(1),
        torch.tensor([prediction["expected_grade"]]),
        atol,
        rtol,
    )
    replayed_grade = int(expected_grade.round().clamp(0, classes.numel() - 1))
    checks["decision_rule"] = prediction.get("decision_rule") == "rounded_expected_grade"
    checks["predicted_grade"] = replayed_grade == int(prediction["predicted_grade"])
    checks["class_argmax_grade"] = int(classes.argmax()) == int(
        prediction["class_argmax_grade"]
    )
    checks["class_probabilities_normalized"] = bool(
        (classes >= -atol).all() and abs(float(classes.sum()) - 1.0) <= atol + rtol
    )

    proof_sizes = torch.tensor(proof["proof_sizes"], dtype=torch.long)
    checks["proof_sizes"] = bool(
        torch.equal(proof_sizes, selected_mask.sum(dim=0).to(torch.long))
    )
    eps_s = float(rule["sufficiency_tolerance"])
    rho_n = float(rule["complement_suppression"])
    rule_atol = float(rule.get("comparison_atol", 1e-7))
    if rule_atol < 0.0:
        raise ValueError("certificate comparison_atol must be non-negative")
    target = (dense_score - eps_s).clamp_min(0.0)
    required_drop = rho_n * target
    complement_drop = dense_score - complement_score
    sufficient = retained_score + rule_atol >= target
    suppressed = complement_drop + rule_atol >= required_drop
    feasible = sufficient & suppressed
    checks["dual_proof_feasible"] = bool(feasible.all())
    _close_check(
        checks,
        errors,
        "sufficiency_gaps",
        dense_score - retained_score,
        torch.tensor(proof["sufficiency_gaps"]),
        atol,
        rtol,
    )

    diagnostics_valid = len(proof["diagnostics"]) == boundaries
    if diagnostics_valid:
        for boundary, diagnostic in enumerate(proof["diagnostics"]):
            expected_values = {
                "dense_score": float(dense_score[boundary]),
                "retained_score": float(retained_score[boundary]),
                "complement_score": float(complement_score[boundary]),
                "sufficiency_gap": float(dense_score[boundary] - retained_score[boundary]),
                "sufficiency_target": float(target[boundary]),
                "complement_drop": float(complement_drop[boundary]),
                "required_complement_drop": float(required_drop[boundary]),
            }
            if int(diagnostic.get("boundary", -1)) != boundary:
                diagnostics_valid = False
            for name, expected_value in expected_values.items():
                value = float(diagnostic.get(name, math.inf))
                if not math.isclose(value, expected_value, abs_tol=atol, rel_tol=rtol):
                    diagnostics_valid = False
            if bool(diagnostic.get("sufficiency_satisfied")) != bool(sufficient[boundary]):
                diagnostics_valid = False
            if bool(diagnostic.get("complement_suppression_satisfied")) != bool(
                suppressed[boundary]
            ):
                diagnostics_valid = False
    checks["proof_diagnostics"] = diagnostics_valid
    _close_check(
        checks,
        errors,
        "complement_drops",
        complement_drop,
        torch.tensor(proof["complement_drops"]),
        atol,
        rtol,
    )

    # Canonical-prefix and minimality replay.  Feasibility is monotone with m,
    # so it is sufficient to show the selected prefix is feasible and its
    # immediate predecessor is not.
    canonical = True
    minimal = True
    for boundary in range(boundaries):
        valid_ids = torch.nonzero(valid, as_tuple=False).flatten().tolist()
        canonical_order = sorted(
            valid_ids, key=lambda index: (-float(witnesses[index, boundary]), index)
        )
        m = int(proof_sizes[boundary])
        if [int(x) for x in selected_indices[boundary]] != canonical_order[:m]:
            canonical = False
        if m > 0:
            predecessor = torch.zeros_like(selected_mask)
            predecessor[canonical_order[: m - 1], boundary] = True
            predecessor_selected = predecessor[:, boundary : boundary + 1]
            boundary_witness = witnesses[:, boundary : boundary + 1]
            boundary_alpha = alpha[boundary : boundary + 1]
            prev_retained, _ = _score_ledger(
                boundary_witness, boundary_alpha, predecessor_selected
            )
            prev_complement, _ = _score_ledger(
                boundary_witness,
                boundary_alpha,
                (~predecessor_selected) & valid[:, None],
            )
            prev_drop = dense_score[boundary] - prev_complement[0]
            prev_feasible = bool(
                prev_retained[0] + rule_atol >= target[boundary]
                and prev_drop + rule_atol >= required_drop[boundary]
            )
            if prev_feasible:
                minimal = False
    checks["canonical_top_prefix"] = canonical
    checks["minimum_prefix"] = minimal

    # Recompute fixed-proof effects with the same closed-form prefix/suffix
    # identity used by the model.  Also retain a literal score-difference
    # replay as a second numerical consistency check.
    replay_pivotality_direct = torch.zeros_like(witnesses)
    count = TruncatedPoissonBinomial(
        max_count=alpha.shape[1], implementation="serial"
    )
    for boundary in range(boundaries):
        # Use the already validated canonical mask here.  Raw serialized lists
        # can contain duplicates or out-of-range indices in a tampered file;
        # those are reported by ``selected_indices_valid`` without crashing
        # the remainder of the audit.
        ids = torch.nonzero(selected_mask[:, boundary], as_tuple=False).flatten().tolist()
        values = witnesses[ids, boundary]
        empty = count.empty_distribution((), witnesses.device)
        prefixes = [empty]
        for value in values:
            prefixes.append(count.update(prefixes[-1], value))
        suffixes = [empty for _ in range(len(ids) + 1)]
        suffixes[-1] = empty
        for position in range(len(ids) - 1, -1, -1):
            suffixes[position] = count.update(suffixes[position + 1], values[position])
        for position, index in enumerate(ids):
            without = count.merge(prefixes[position], suffixes[position + 1])
            without_score = (count.tails(without) * alpha[boundary]).sum()
            replay_pivotality_direct[index, boundary] = (
                retained_score[boundary] - without_score
            )

    try:
        sorted_indices = torch.argsort(
            witnesses.transpose(0, 1).masked_fill(~valid[None, :], -1.0),
            dim=-1,
            descending=True,
            stable=True,
        ).unsqueeze(0)
    except TypeError:  # pragma: no cover
        sorted_indices = torch.argsort(
            witnesses.transpose(0, 1).masked_fill(~valid[None, :], -1.0),
            dim=-1,
            descending=True,
        ).unsqueeze(0)
    replay_proof = ProofProjectionResult(
        selected_mask=selected_mask.unsqueeze(0),
        sorted_indices=sorted_indices,
        proof_size=proof_sizes.unsqueeze(0),
        dense_transition=dense_score.unsqueeze(0),
        projected_transition=retained_score.unsqueeze(0),
        complement_transition=complement_score.unsqueeze(0),
        retained_distribution=retained_distribution.unsqueeze(0),
        complement_distribution=complement_distribution.unsqueeze(0),
        sufficiency_gap=(dense_score - retained_score).unsqueeze(0),
        complement_drop=(dense_score - complement_score).unsqueeze(0),
    )
    replay_pivotality = fixed_proof_pivotality(
        witnesses.unsqueeze(0), replay_proof, alpha, alpha.shape[1]
    )[0]
    _close_check(
        checks,
        errors,
        "fixed_proof_pivotality",
        replay_pivotality,
        torch.tensor(proof["fixed_proof_pivotality"]),
        atol,
        rtol,
    )
    checks["pivotality_direct_intervention"] = bool(
        torch.allclose(
            replay_pivotality,
            replay_pivotality_direct,
            atol=max(atol, 1e-5),
            rtol=max(rtol, 1e-5),
        )
    )

    # The human-facing cell table must be a deterministic rendering of the
    # ledger, proof indices, pivotality, and receptive-field metadata.  It is
    # verified explicitly in addition to the whole-payload integrity hash.
    selected_cells_valid = geometry is not None
    cells_value = proof.get("selected_cells")
    if not isinstance(cells_value, list) or len(cells_value) != boundaries:
        selected_cells_valid = False
    elif geometry is not None:
        for boundary in range(boundaries):
            actual_cells = cells_value[boundary]
            ids = [int(index) for index in selected_indices[boundary]]
            expected_cells = [
                _cell_record(index, boundary, witnesses, replay_pivotality, geometry)
                for index in ids
            ]
            if not isinstance(actual_cells, list) or len(actual_cells) != len(expected_cells):
                selected_cells_valid = False
                continue
            for actual, expected_cell in zip(actual_cells, expected_cells):
                if not isinstance(actual, Mapping):
                    selected_cells_valid = False
                    continue
                for name in ("index", "row", "column"):
                    if int(actual.get(name, -1)) != int(expected_cell[name]):
                        selected_cells_valid = False
                for name in ("witness_probability", "fixed_proof_pivotality"):
                    if not math.isclose(
                        float(actual.get(name, math.inf)),
                        float(expected_cell[name]),
                        abs_tol=max(atol, 1e-5),
                        rel_tol=max(rtol, 1e-5),
                    ):
                        selected_cells_valid = False
                for name in ("center_yx", "receptive_field_box_yxyx"):
                    try:
                        actual_values = [float(value) for value in actual[name]]
                    except (KeyError, TypeError, ValueError):
                        selected_cells_valid = False
                        continue
                    expected_values = expected_cell[name]
                    if len(actual_values) != len(expected_values) or any(
                        not math.isclose(a, e, abs_tol=atol, rel_tol=rtol)
                        for a, e in zip(actual_values, expected_values)
                    ):
                        selected_cells_valid = False
    checks["selected_cell_records"] = selected_cells_valid

    ok = all(checks.values())
    result = {
        "ok": ok,
        "checks": checks,
        "max_abs_errors": errors,
        "replayed_predicted_grade": replayed_grade,
        "replayed_class_argmax_grade": int(classes.argmax()),
        "replayed_expected_grade": float(expected_grade),
        "replay_atol": float(atol),
        "replay_rtol": float(rtol),
    }
    if raise_on_error and not ok:
        failed = [name for name, passed in checks.items() if not passed]
        raise CertificateReplayError(
            "certificate replay failed: " + ", ".join(failed)
        )
    return result


__all__ = [
    "CertificateReplayError",
    "SCHEMA_VERSION",
    "DEFAULT_REPLAY_ATOL",
    "DEFAULT_REPLAY_RTOL",
    "build_mosaic_certificate",
    "certificate_to_json",
    "load_mosaic_certificate",
    "save_mosaic_certificate",
    "verify_mosaic_certificate",
]
