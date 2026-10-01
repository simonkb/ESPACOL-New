#!/usr/bin/env python3
"""Blind checkpoint audit for the controlled ordinal-shortcut benchmark."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch
from torch.nn import functional as F

from benchmarks.ordinal_shortcut import (
    SHORTCUT_FAMILIES,
    OrdinalShortcutDataset,
    OrdinalShortcutProtocol,
)
from benchmarks.shortcut_metrics import (
    boundary_response_matrix,
    boundary_response_selectivity,
    condition_performance,
    familywise_localization_permutation_test,
    internal_pixel_effect_audit,
    json_ready,
    localization_metrics,
    rasterize_native_ledger,
)
from Datasets.origin_data import (
    ORIGIN_PREPROCESSING_VERSION,
    OriginFundusTransform,
    OriginImageDataset,
    load_aptos_items,
    split_origin_items,
    split_paths_are_disjoint,
)
from models.origin import build_origin_model
from models.origin_acceptance_baselines import (
    AcceptanceLocalOutput,
    build_origin_acceptance_baseline,
)
from train_origin import split_signature


def _chunks(values: Sequence[int], size: int):
    for start in range(0, len(values), size):
        yield values[start : start + size]


def _stable_int(*values: object) -> int:
    digest = hashlib.blake2b(digest_size=8, person=b"orig-audit-v1")
    for value in values:
        digest.update(str(value).encode("utf-8"))
        digest.update(b"\0")
    return int.from_bytes(digest.digest(), "little")


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as stream:
            json.dump(json_ready(payload), stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _write_jsonl_atomic(
    path: Path, manifest: Mapping[str, Any], records: Sequence[Mapping[str, Any]]
) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    digest = hashlib.sha256()
    try:
        with temporary.open("wb") as stream:
            for payload in (
                {"record_type": "manifest", **dict(manifest)},
                *records,
            ):
                encoded = (
                    json.dumps(
                        json_ready(payload),
                        sort_keys=True,
                        separators=(",", ":"),
                        allow_nan=False,
                    )
                    + "\n"
                ).encode("utf-8")
                stream.write(encoded)
                digest.update(encoded)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return digest.hexdigest()


def _transform_parameter_hash(
    *,
    protocol: OrdinalShortcutProtocol,
    image_size: int,
    family: str,
    condition: str,
    position_domain: str,
    appearance_domain: str,
    canonical_augmentation: bool = False,
) -> str:
    payload = {
        "canonical_preprocessing_version": ORIGIN_PREPROCESSING_VERSION,
        "image_size": int(image_size),
        "canonical_augmentation": bool(canonical_augmentation),
        "canonical_geometry": {
            "dominant_field_threshold": 10,
            "canvas": "tight-field direct square resize",
            "interpolation": "PIL bilinear",
            "valid_support": "fixed centered filled ellipse",
        },
        "normalization": {
            "mean": [0.485, 0.456, 0.406],
            "standard_deviation": [0.229, 0.224, 0.225],
        },
        "training_augmentation": (
            {
                "horizontal_flip_probability": 0.5,
                "vertical_flip_probability": 0.5,
                "rotation_degrees": [-30.0, 30.0],
                "brightness": 0.20,
                "contrast": 0.20,
                "saturation": 0.10,
                "hue": 0.02,
            }
            if canonical_augmentation
            else None
        ),
        "shortcut_protocol": protocol.as_dict(),
        "shortcut_protocol_signature": protocol.signature,
        "family": family,
        "condition": condition,
        "position_domain": position_domain,
        "appearance_domain": appearance_domain,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode(
            "utf-8"
        )
    ).hexdigest()


def _model_from_config(config: Mapping[str, Any]):
    variant = str(config.get("model_variant", config.get("baseline_variant", "origin_ctmc")))
    if variant != "origin_ctmc":
        return build_origin_acceptance_baseline(config, pretrained=False)
    return build_origin_model(
        num_classes=int(config["n_classes"]),
        encoder_name=str(config["encoder"]),
        pretrained=False,
        evidence_scales=tuple(config["evidence_scales"]),
        projection_dim=int(config["projection_dim"]),
        reference_count=float(config["reference_count"]),
        atom_rate_init=float(config["atom_rate_init"]),
        prior_rate_init=float(config["prior_rate_init"]),
        boundary_scale_init=float(config["boundary_scale_init"]),
        total_rate_cap=float(config["total_rate_cap"]),
        prior_rate_cap=float(config["prior_rate_cap"]),
        boundary_scale_cap=float(config["boundary_scale_cap"]),
        rate_roundoff_margin=float(config["rate_roundoff_margin"]),
        atom_mode=str(config["atom_mode"]),
        hybrid_cumulative_init=float(config["hybrid_cumulative_init"]),
        evidence_dropout=float(config["evidence_dropout"]),
        mask_valid_fraction=float(config["mask_valid_fraction"]),
        grad_checkpoint=bool(config["grad_checkpoint"]),
    )


@dataclass(frozen=True)
class _LocalLedgerView:
    boundary_maps_cf: Mapping[str, torch.Tensor]
    valid_masks: Mapping[str, torch.Tensor]
    semantics: str


@dataclass(frozen=True)
class _LocalIntervention:
    output: object
    removal_masks: Mapping[str, torch.Tensor]


def _local_ledger_view(output: object, model_variant: str) -> _LocalLedgerView | None:
    """Expose comparable boundary-support maps without changing predictions.

    Sparse-BagNet stores class activations, so its boundary-k support is the
    sum of the stored class channels above k.  Additive-MIL stores signed
    boundary logit contributions directly.  These are explicitly custom
    matched analogues, not author-code implementations or CTMC rates.
    """

    source = getattr(output, "ledger_output", output)
    scale_evidence = getattr(source, "scale_evidence", None)
    if isinstance(scale_evidence, Mapping) and scale_evidence:
        return _LocalLedgerView(
            boundary_maps_cf={
                name: item.local_rate_map for name, item in scale_evidence.items()
            },
            valid_masks={name: item.valid_mask for name, item in scale_evidence.items()},
            semantics="nonnegative_boundary_rate_ledger",
        )
    if isinstance(output, AcceptanceLocalOutput):
        if model_variant == "ordinal_additive_mil":
            maps = {
                name: value.permute(0, 3, 1, 2).contiguous()
                for name, value in output.effective_local_maps.items()
            }
            semantics = "signed_boundary_logit_contribution"
        elif model_variant == "sparse_bagnet":
            maps = {}
            for name, value in output.effective_local_maps.items():
                # (N,H,W,K) -> (N,K-1,H,W), channel k supports Y>k.
                severity = torch.stack(
                    [value[..., boundary + 1 :].sum(dim=-1) for boundary in range(4)],
                    dim=1,
                )
                maps[name] = severity
            semantics = "nonnegative_sum_of_class_activations_above_boundary"
        else:
            raise ValueError(f"unsupported local comparator {model_variant!r}")
        return _LocalLedgerView(
            boundary_maps_cf=maps,
            valid_masks=output.original_valid_masks,
            semantics=semantics,
        )
    return None


def _topk_local_intervention(
    model: object,
    output: object,
    view: _LocalLedgerView,
    *,
    boundary: int,
    k: int,
) -> _LocalIntervention:
    score_parts: list[torch.Tensor] = []
    valid_parts: list[torch.Tensor] = []
    offsets: dict[str, tuple[int, int]] = {}
    cursor = 0
    for name, scores_cf in view.boundary_maps_cf.items():
        scores = scores_cf[:, boundary].flatten(1)
        valid = view.valid_masks[name].flatten(1).bool()
        score_parts.append(scores)
        valid_parts.append(valid)
        offsets[name] = (cursor, cursor + scores.shape[1])
        cursor += scores.shape[1]
    scores = torch.cat(score_parts, dim=1)
    valid = torch.cat(valid_parts, dim=1)
    if bool((valid.sum(dim=1) < k).any()):
        raise ValueError("k exceeds the valid local-cell count")
    selected = scores.masked_fill(~valid, -torch.inf).topk(k, dim=1).indices
    removals: dict[str, torch.Tensor] = {}
    for name, valid_mask in view.valid_masks.items():
        start, stop = offsets[name]
        counts = torch.zeros_like(valid_mask.flatten(1), dtype=torch.int64)
        within = (selected >= start) & (selected < stop)
        local = (selected - start).clamp(0, stop - start - 1)
        counts.scatter_add_(1, local, within.to(torch.int64))
        removals[name] = (counts > 0).reshape_as(valid_mask)
    replayed = model.replay_without(output, removals, force_decoder_fp64=True)
    return _LocalIntervention(output=replayed.output, removal_masks=removals)


def _prediction(output: object, rule: str) -> torch.Tensor:
    if rule == "class_map":
        return output.class_map
    if rule == "posterior_median":
        return output.posterior_median
    if rule == "rounded_expected":
        return output.expected_grade.round().clamp(0, output.class_probs.shape[-1] - 1).long()
    raise ValueError(f"unknown decision rule {rule!r}")


def _select_indices(labels: Sequence[int], maximum: int, seed: int) -> list[int]:
    if maximum <= 0 or maximum >= len(labels):
        return list(range(len(labels)))
    groups: dict[int, list[int]] = {}
    for index, label in enumerate(labels):
        groups.setdefault(int(label), []).append(index)
    generator = np.random.default_rng(int(seed))
    selected: list[int] = []
    allocations = {label: maximum // len(groups) for label in groups}
    for label in sorted(groups)[: maximum % len(groups)]:
        allocations[label] += 1
    for label, members in sorted(groups.items()):
        count = min(len(members), allocations[label])
        selected.extend(int(value) for value in generator.choice(members, count, replace=False))
    if len(selected) < maximum:
        remaining = sorted(set(range(len(labels))) - set(selected))
        selected.extend(remaining[: maximum - len(selected)])
    return sorted(selected)


def _dataset(
    items,
    *,
    image_size: int,
    family: str,
    condition: str,
    protocol: OrdinalShortcutProtocol,
    held_out: bool,
) -> OrdinalShortcutDataset:
    clean = OriginImageDataset(
        items,
        OriginFundusTransform(size=image_size, augment=False),
    )
    return OrdinalShortcutDataset(
        clean,
        family=family,
        condition=condition,
        position_domain="unseen" if held_out else "seen",
        appearance_domain="unseen" if held_out else "seen",
        protocol=protocol,
    )


def _render_batch(dataset: OrdinalShortcutDataset, indices: Sequence[int]):
    samples = [dataset.render(int(index)) for index in indices]
    images = torch.stack([sample.image for sample in samples])
    masks = torch.stack([sample.valid_mask for sample in samples])
    labels = torch.stack([sample.label for sample in samples])
    return samples, images, masks, labels


def _run_condition(
    model,
    dataset: OrdinalShortcutDataset,
    indices: Sequence[int],
    *,
    device: torch.device,
    batch_size: int,
    decision_rule: str,
    fold: int,
    transform_parameter_sha256: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict[str, Any]]]:
    predictions: list[torch.Tensor] = []
    labels: list[torch.Tensor] = []
    groups: list[int] = []
    records: list[dict[str, Any]] = []
    with torch.inference_mode():
        for batch_indices in _chunks(indices, batch_size):
            samples, images, valid, target = _render_batch(dataset, batch_indices)
            output = model(images.to(device), valid.to(device), force_decoder_fp64=True)
            predicted = _prediction(output, decision_rule).detach().cpu()
            class_probabilities = output.class_probs.detach().cpu()
            threshold_probabilities = output.cumulative_probs.detach().cpu()
            predictions.append(predicted)
            labels.append(target)
            for row, sample in enumerate(samples):
                bits = sample.metadata.marker_bits
                present = sample.metadata.marker_present
                bit_code = sum(int(value) << boundary for boundary, value in enumerate(bits))
                present_code = sum(
                    int(value) << boundary for boundary, value in enumerate(present)
                )
                groups.append(bit_code + 16 * present_code)
                target_id = f"aptos:fold{fold}:outer:{sample.metadata.sample_index}"
                clean_source_id = (
                    f"aptos:fold{fold}:clean-source:{sample.metadata.source_index}"
                )
                render_id = (
                    f"{target_id}:source{sample.metadata.source_index}:"
                    f"{dataset.family}:{dataset.condition}:"
                    f"position-{dataset.position_domain}:appearance-{dataset.appearance_domain}:"
                    f"bits{bit_code}:present{present_code}"
                )
                source_path = None
                target_path = None
                if dataset.items is not None:
                    source_path = Path(dataset.items[sample.metadata.source_index][0]).name
                    target_path = Path(dataset.items[sample.metadata.sample_index][0]).name
                records.append(
                    {
                        "record_type": "condition_prediction",
                        "sample_id": target_id,
                        "target_image_id": target_path,
                        "clean_source_id": clean_source_id,
                        "clean_source_image_id": source_path,
                        "render_id": render_id,
                        "cue_only_assignment_id": (
                            f"{target_id}:permuted-to:{clean_source_id}"
                            if dataset.condition == "cue_only"
                            else None
                        ),
                        "sample_index": sample.metadata.sample_index,
                        "source_index": sample.metadata.source_index,
                        "label": int(sample.label),
                        "predicted_grade": int(predicted[row]),
                        "class_probabilities": class_probabilities[row].tolist(),
                        "threshold_probabilities": threshold_probabilities[row].tolist(),
                        "family": dataset.family,
                        "condition": dataset.condition,
                        "position_domain": dataset.position_domain,
                        "appearance_domain": dataset.appearance_domain,
                        "protocol_seed": dataset.protocol.seed,
                        "sample_seed": sample.metadata.sample_seed,
                        "marker_bits": list(bits),
                        "marker_present": list(present),
                        "source_boundaries": list(sample.metadata.source_boundaries),
                        "transform_parameter_sha256": transform_parameter_sha256,
                    }
                )
    return (
        torch.cat(predictions).numpy(),
        torch.cat(labels).numpy(),
        np.asarray(groups, dtype=np.int64),
        records,
    )


def _scale_matched_random_masks(
    valid_masks: Mapping[str, torch.Tensor],
    selected_masks: Mapping[str, torch.Tensor],
    *,
    sample_indices: Sequence[int],
    boundary: int,
    audit_seed: int,
) -> dict[str, torch.Tensor]:
    random_masks: dict[str, torch.Tensor] = {}
    for scale_index, (name, valid) in enumerate(valid_masks.items()):
        requested = selected_masks[name].flatten(1).sum(dim=1)
        flat_valid = valid.flatten(1)
        chosen = torch.zeros_like(flat_valid)
        coordinates = torch.arange(
            flat_valid.shape[1], device=flat_valid.device, dtype=torch.int64
        )
        for row, requested_count in enumerate(requested.tolist()):
            count = int(requested_count)
            if count == 0:
                continue
            seed = _stable_int(
                audit_seed, sample_indices[row], boundary, scale_index
            ) % (2**31 - 1)
            # A deterministic integer permutation avoids device-specific RNG
            # streams while remaining independent of pixels and labels.
            scores = ((coordinates + seed) * 6364136223846793005 + 1442695040888963407)
            scores = torch.remainder(scores, 2**31 - 1)
            scores = scores.masked_fill(~flat_valid[row], -1)
            selected = scores.topk(count).indices
            chosen[row, selected] = True
        random_masks[name] = chosen.reshape_as(valid)
    return random_masks


def _downsample_masks(masks: torch.Tensor, size: int) -> torch.Tensor:
    batch, channels = masks.shape[:2]
    pooled = F.adaptive_max_pool2d(
        masks.float().reshape(batch * channels, 1, *masks.shape[-2:]),
        (size, size),
    )
    return pooled.reshape(batch, channels, size, size).bool()


def _main_audit(
    model,
    dataset: OrdinalShortcutDataset,
    indices: Sequence[int],
    *,
    device: torch.device,
    batch_size: int,
    audit_grid: int,
    audit_seed: int,
    permutations: int,
    bootstrap_replicates: int,
    fold: int,
    factorial_transform_sha256: str,
    main_transform_sha256: str,
    decision_rule: str,
    model_variant: str,
) -> dict[str, Any]:
    evidence_batches: list[torch.Tensor] = []
    marker_batches: list[torch.Tensor] = []
    valid_batches: list[torch.Tensor] = []
    paired_tail_batches: list[torch.Tensor] = []
    internal_effects: list[torch.Tensor] = []
    random_effects: list[torch.Tensor] = []
    pixel_effects: list[torch.Tensor] = []
    metadata_records: list[dict[str, Any]] = []
    factorial_prediction_records: list[dict[str, Any]] = []
    internal_effect_records: list[dict[str, Any]] = []
    mask_digest = hashlib.sha256()
    local_semantics: str | None = None

    with torch.inference_mode():
        for batch_indices in _chunks(indices, batch_size):
            samples, images, valid, _ = _render_batch(dataset, batch_indices)
            images_device = images.to(device)
            valid_device = valid.to(device)
            output = model(images_device, valid_device, force_decoder_fp64=True)
            support = torch.stack([sample.metadata.support_masks for sample in samples])
            for sample in samples:
                mask_digest.update(sample.metadata.support_masks.numpy().tobytes())
                if len(metadata_records) < 16:
                    metadata_records.append(sample.metadata.to_record())
            full_tails = output.cumulative_probs
            local_view = _local_ledger_view(output, model_variant)
            if local_view is not None:
                if local_semantics is None:
                    local_semantics = local_view.semantics
                elif local_semantics != local_view.semantics:
                    raise AssertionError("local ledger semantics changed between batches")
                raster = rasterize_native_ledger(
                    local_view.boundary_maps_cf,
                    output_size=(audit_grid, audit_grid),
                )
                evidence_batches.append(raster.detach().float().cpu())
                marker_batches.append(_downsample_masks(support, audit_grid).cpu())
                valid_batches.append(
                    F.interpolate(
                        valid.float(), size=(audit_grid, audit_grid), mode="nearest"
                    ).bool()
                )
                all_cells = {
                    name: value.clone() for name, value in local_view.valid_masks.items()
                }
                prior = model.replay_without(
                    output, all_cells, force_decoder_fp64=True
                ).output
                denominator = (full_tails - prior.cumulative_probs).abs().clamp_min(1e-12)
            else:
                prior = None
                denominator = None

            batch_paired = torch.empty(
                len(batch_indices),
                dataset.protocol.num_boundaries,
                2,
                dataset.protocol.num_boundaries,
                dtype=torch.float64,
            )
            batch_internal = (
                torch.empty(
                    len(batch_indices), dataset.protocol.num_boundaries, dtype=torch.float64
                )
                if local_view is not None
                else None
            )
            batch_random = (
                torch.empty_like(batch_internal) if batch_internal is not None else None
            )
            batch_pixel = torch.empty(
                len(batch_indices), dataset.protocol.num_boundaries, dtype=torch.float64
            )
            pixel_fraction = torch.stack(
                [
                    sample.metadata.support_masks.float().mean(dim=(-2, -1))
                    for sample in samples
                ]
            )
            budget_fraction = pixel_fraction.median(dim=0).values.clamp(max=0.10)
            minimum_valid = (
                min(
                    int(
                        sum(
                            mask[row].sum()
                            for mask in local_view.valid_masks.values()
                        )
                    )
                    for row in range(len(batch_indices))
                )
                if local_view is not None
                else 0
            )

            # Evaluate the complete 2^4 factorial on exactly the same clean
            # pixels. Marginal cue effects average over all eight settings of
            # the other three cues, rather than privileging one reference bit
            # pattern.
            factorial_samples = []
            for index in batch_indices:
                for state_code in range(2 ** dataset.protocol.num_boundaries):
                    bits = tuple(
                        bool((state_code >> boundary) & 1)
                        for boundary in range(dataset.protocol.num_boundaries)
                    )
                    factorial_samples.append(
                        dataset.render(
                            index,
                            marker_bits=bits,
                            marker_present=(True,) * dataset.protocol.num_boundaries,
                        )
                    )
            factorial_probabilities: list[torch.Tensor] = []
            factorial_tails: list[torch.Tensor] = []
            factorial_predictions: list[torch.Tensor] = []
            for render_indices in _chunks(list(range(len(factorial_samples))), batch_size):
                rendered = [factorial_samples[index] for index in render_indices]
                rendered_output = model(
                    torch.stack([sample.image for sample in rendered]).to(device),
                    torch.stack([sample.valid_mask for sample in rendered]).to(device),
                    force_decoder_fp64=True,
                )
                factorial_probabilities.append(rendered_output.class_probs.detach().cpu())
                factorial_tails.append(rendered_output.cumulative_probs.detach().cpu())
                factorial_predictions.append(
                    _prediction(rendered_output, decision_rule).detach().cpu()
                )
            state_count = 2 ** dataset.protocol.num_boundaries
            factorial_probs_tensor = torch.cat(factorial_probabilities).reshape(
                len(batch_indices), state_count, -1
            )
            factorial_tails_tensor = torch.cat(factorial_tails).reshape(
                len(batch_indices), state_count, dataset.protocol.num_boundaries
            )
            factorial_predictions_tensor = torch.cat(factorial_predictions).reshape(
                len(batch_indices), state_count
            )
            state_axis = torch.arange(state_count)
            for boundary in range(dataset.protocol.num_boundaries):
                inactive_states = ((state_axis >> boundary) & 1) == 0
                active_states = ~inactive_states
                batch_paired[:, boundary, 0] = factorial_tails_tensor[
                    :, inactive_states
                ].mean(dim=1)
                batch_paired[:, boundary, 1] = factorial_tails_tensor[
                    :, active_states
                ].mean(dim=1)
                batch_pixel[:, boundary] = (
                    batch_paired[:, boundary, 1, boundary]
                    - batch_paired[:, boundary, 0, boundary]
                )

            for sample_row, sample_index in enumerate(batch_indices):
                for state_code in range(state_count):
                    sample = factorial_samples[sample_row * state_count + state_code]
                    target_id = f"aptos:fold{fold}:outer:{sample.metadata.sample_index}"
                    clean_source_id = (
                        f"aptos:fold{fold}:clean-source:{sample.metadata.source_index}"
                    )
                    bits = tuple(
                        bool((state_code >> boundary) & 1)
                        for boundary in range(dataset.protocol.num_boundaries)
                    )
                    context_pairs = {
                        str(boundary): (
                            f"{target_id}:source{sample.metadata.source_index}:"
                            f"factorial-toggle-b{boundary}-context"
                            f"{state_code & ~(1 << boundary)}"
                        )
                        for boundary in range(dataset.protocol.num_boundaries)
                    }
                    source_path = None
                    target_path = None
                    if dataset.items is not None:
                        source_path = Path(
                            dataset.items[sample.metadata.source_index][0]
                        ).name
                        target_path = Path(
                            dataset.items[sample.metadata.sample_index][0]
                        ).name
                    factorial_prediction_records.append(
                        {
                            "record_type": "factorial_prediction",
                            "sample_id": target_id,
                            "target_image_id": target_path,
                            "clean_source_id": clean_source_id,
                            "clean_source_image_id": source_path,
                            "cue_only_assignment_id": (
                                f"{target_id}:permuted-to:{clean_source_id}"
                                if dataset.condition == "cue_only"
                                else None
                            ),
                            "factorial_render_id": (
                                f"{target_id}:source{sample.metadata.source_index}:"
                                f"{dataset.family}:factorial-state{state_code:02d}"
                            ),
                            "toggle_pair_ids": context_pairs,
                            "sample_index": sample.metadata.sample_index,
                            "source_index": sample.metadata.source_index,
                            "factorial_state": state_code,
                            "marker_bits": list(bits),
                            "marker_present": [True] * dataset.protocol.num_boundaries,
                            "label": int(sample.label),
                            "predicted_grade": int(
                                factorial_predictions_tensor[sample_row, state_code]
                            ),
                            "class_probabilities": factorial_probs_tensor[
                                sample_row, state_code
                            ].tolist(),
                            "threshold_probabilities": factorial_tails_tensor[
                                sample_row, state_code
                            ].tolist(),
                            "family": dataset.family,
                            "condition": "factorial",
                            "base_condition": dataset.condition,
                            "position_domain": dataset.position_domain,
                            "appearance_domain": dataset.appearance_domain,
                            "protocol_seed": dataset.protocol.seed,
                            "sample_seed": sample.metadata.sample_seed,
                            "transform_parameter_sha256": factorial_transform_sha256,
                        }
                    )

            if local_view is not None:
                assert prior is not None and denominator is not None
                assert batch_internal is not None and batch_random is not None
                for boundary in range(dataset.protocol.num_boundaries):
                    k = max(
                        1,
                        int(
                            round(
                                float(budget_fraction[boundary]) * minimum_valid
                            )
                        ),
                    )
                    top = _topk_local_intervention(
                        model, output, local_view, boundary=boundary, k=k
                    )
                    random_masks = _scale_matched_random_masks(
                        local_view.valid_masks,
                        top.removal_masks,
                        sample_indices=batch_indices,
                        boundary=boundary,
                        audit_seed=audit_seed,
                    )
                    random_replay = model.replay_without(
                        output, random_masks, force_decoder_fp64=True
                    ).output
                    batch_internal[:, boundary] = (
                        (
                            full_tails[:, boundary]
                            - top.output.cumulative_probs[:, boundary]
                        )
                        / denominator[:, boundary]
                    ).detach().cpu()
                    batch_random[:, boundary] = (
                        (
                            full_tails[:, boundary]
                            - random_replay.cumulative_probs[:, boundary]
                        )
                        / denominator[:, boundary]
                    ).detach().cpu()
                    for row, sample in enumerate(samples):
                        target_id = (
                            f"aptos:fold{fold}:outer:{sample.metadata.sample_index}"
                        )
                        internal_effect_records.append(
                            {
                                "record_type": "internal_pixel_effect",
                                "sample_id": target_id,
                                "clean_source_id": (
                                    f"aptos:fold{fold}:clean-source:"
                                    f"{sample.metadata.source_index}"
                                ),
                                "sample_index": sample.metadata.sample_index,
                                "source_index": sample.metadata.source_index,
                                "label": int(sample.label),
                                "family": dataset.family,
                                "condition": dataset.condition,
                                "boundary": boundary,
                                "protocol_seed": dataset.protocol.seed,
                                "sample_seed": sample.metadata.sample_seed,
                                "native_selected_cells_by_scale": {
                                    name: int(mask[row].sum())
                                    for name, mask in top.removal_masks.items()
                                },
                                "full_class_probabilities": output.class_probs[row]
                                .detach()
                                .cpu()
                                .tolist(),
                                "prior_class_probabilities": prior.class_probs[row]
                                .detach()
                                .cpu()
                                .tolist(),
                                "top_deleted_class_probabilities": (
                                    top.output.class_probs[row]
                                    .detach()
                                    .cpu()
                                    .tolist()
                                ),
                                "random_deleted_class_probabilities": (
                                    random_replay.class_probs[row]
                                    .detach()
                                    .cpu()
                                    .tolist()
                                ),
                                "full_threshold_probabilities": full_tails[row]
                                .detach()
                                .cpu()
                                .tolist(),
                                "prior_threshold_probabilities": (
                                    prior.cumulative_probs[row]
                                    .detach()
                                    .cpu()
                                    .tolist()
                                ),
                                "top_deleted_threshold_probabilities": (
                                    top.output.cumulative_probs[row]
                                    .detach()
                                    .cpu()
                                    .tolist()
                                ),
                                "random_deleted_threshold_probabilities": (
                                    random_replay.cumulative_probs[row]
                                    .detach()
                                    .cpu()
                                    .tolist()
                                ),
                                "normalized_internal_effect": float(
                                    batch_internal[row, boundary]
                                ),
                                "normalized_random_effect": float(
                                    batch_random[row, boundary]
                                ),
                                "factorial_pixel_marginal_effect": float(
                                    batch_pixel[row, boundary]
                                ),
                                "transform_parameter_sha256": main_transform_sha256,
                                "factorial_transform_parameter_sha256": (
                                    factorial_transform_sha256
                                ),
                            }
                        )

            paired_tail_batches.append(batch_paired)
            if batch_internal is not None and batch_random is not None:
                internal_effects.append(batch_internal)
                random_effects.append(batch_random)
            pixel_effects.append(batch_pixel)

    paired_tails = torch.cat(paired_tail_batches).numpy()
    pixel = torch.cat(pixel_effects).numpy()
    response = boundary_response_matrix(paired_tails)
    if evidence_batches:
        evidence = torch.cat(evidence_batches).numpy()
        marker_masks = torch.cat(marker_batches).numpy()
        valid_masks = torch.cat(valid_batches).numpy()
        internal = torch.cat(internal_effects).numpy()
        random_effect = torch.cat(random_effects).numpy()
        localization = localization_metrics(
            evidence,
            marker_masks,
            valid_masks,
            localization_applicable=dataset.family != "diffuse",
        )
        permutation = familywise_localization_permutation_test(
            evidence,
            marker_masks,
            valid_masks,
            permutations=permutations,
            seed=audit_seed,
        )
        effects = internal_pixel_effect_audit(
            internal,
            random_effect,
            pixel,
            bootstrap_replicates=bootstrap_replicates,
            seed=audit_seed + 1,
        )
        effects["per_boundary"] = [
            internal_pixel_effect_audit(
                internal[:, boundary],
                random_effect[:, boundary],
                pixel[:, boundary],
                bootstrap_replicates=bootstrap_replicates,
                seed=audit_seed + 11 + boundary,
            )
            for boundary in range(dataset.protocol.num_boundaries)
        ]
    else:
        localization = {
            "applicable": False,
            "reason": "pooled comparator exposes no spatial prediction ledger",
        }
        permutation = {
            "applicable": False,
            "reason": "pooled comparator exposes no spatial prediction ledger",
        }
        effects = {
            "applicable": False,
            "reason": "pooled comparator has no exact stored-ledger intervention",
            "pixel_factorial_effects_remain_reported": True,
        }
    return {
        "localization": localization,
        "localization_permutation": permutation,
        "boundary_response_matrix": response,
        "boundary_response_selectivity": boundary_response_selectivity(response),
        "internal_pixel_effects": effects,
        "procedural_mask_sha256": mask_digest.hexdigest(),
        "procedural_metadata_examples": metadata_records,
        "audit_grid": audit_grid,
        "native_selection": bool(evidence_batches),
        "local_ledger_applicable": bool(evidence_batches),
        "local_ledger_semantics": local_semantics,
        "selection_rule": (
            "top native target-boundary ledger entries"
            if evidence_batches
            else "not_applicable"
        ),
        "random_control": (
            "same per-scale native-cell count"
            if evidence_batches
            else "not_applicable"
        ),
        "factorial_prediction_records": factorial_prediction_records,
        "internal_effect_records": internal_effect_records,
        "factorial_states_per_sample": 16,
        "pixel_effect": "balanced marginal contrast over all eight paired contexts",
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Blind ORIGIN controlled-shortcut checkpoint audit",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--family", choices=SHORTCUT_FAMILIES, default=None)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--audit-grid", type=int, default=64)
    parser.add_argument("--audit-seed", type=int, default=7193)
    parser.add_argument("--permutations", type=int, default=999)
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument(
        "--max-samples",
        type=int,
        default=0,
        help="zero audits the complete outer fold; positive values are smoke tests only",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    prediction_artifact = args.output.with_name(
        f"{args.output.stem}_predictions.jsonl"
    )
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite audit: {args.output}")
    if prediction_artifact.exists():
        raise FileExistsError(
            f"refusing to overwrite prediction artifact: {prediction_artifact}"
        )
    if args.batch_size < 1 or args.audit_grid < 8:
        raise ValueError("batch-size must be positive and audit-grid at least 8")
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if state.get("schema") != "origin-checkpoint-v3":
        raise ValueError("shortcut audit requires an ORIGIN-v3 checkpoint")
    config = dict(state.get("config", {}))
    required = {
        "shortcut_arm",
        "shortcut_family",
        "shortcut_protocol",
        "shortcut_protocol_signature",
        "split_seed",
    }
    if not required.issubset(config):
        raise ValueError("checkpoint is not a controlled-shortcut run")
    arm = str(config["shortcut_arm"])
    model_variant = str(
        config.get("model_variant", config.get("baseline_variant", "origin_ctmc"))
    )
    family = args.family or str(config["shortcut_family"])
    if family not in SHORTCUT_FAMILIES:
        raise ValueError(f"invalid family {family!r}")
    protocol = OrdinalShortcutProtocol(**dict(config["shortcut_protocol"]))
    if protocol.signature != config["shortcut_protocol_signature"]:
        raise ValueError("checkpoint shortcut protocol signature mismatch")

    items = load_aptos_items(str(args.data_root), config.get("labels_csv"))
    fold = int(state["fold"])
    train_items, validation_items, test_items = split_origin_items(
        "aptos",
        items,
        fold,
        n_folds=int(config["n_folds"]),
        val_fraction=float(config["val_fraction"]),
        seed=int(config["split_seed"]),
    )
    if not split_paths_are_disjoint(train_items, validation_items, test_items):
        raise RuntimeError("reconstructed shortcut split overlaps")
    observed_signature = split_signature(
        ("train", train_items),
        ("validation", validation_items),
        ("locked_test", test_items),
    )
    if observed_signature != state.get("split_signature"):
        raise ValueError("checkpoint split signature does not match reconstructed APTOS fold")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _model_from_config(config).to(device).eval()
    model.load_state_dict(state["model_state"], strict=True)
    labels = [int(label) for _, label in test_items]
    indices = _select_indices(labels, args.max_samples, args.audit_seed)
    if len(indices) < 3:
        raise ValueError("audit requires at least three outer-fold samples")

    if arm == "shortcut":
        conditions = (
            "aligned",
            "neutral",
            "inverted",
            "missing",
            "boundary_swapped",
            "conflicting",
            "clean",
        )
        main_condition = "aligned"
    elif arm == "cue_only":
        conditions = ("cue_only", "neutral", "inverted", "clean")
        main_condition = "cue_only"
    elif arm == "clean":
        conditions = ("clean",)
        main_condition = "clean"
    else:
        raise ValueError(f"unknown shortcut arm {arm!r}")

    predictions: dict[str, np.ndarray] = {}
    groups: dict[str, np.ndarray] = {}
    condition_prediction_records: list[dict[str, Any]] = []
    transform_hashes: dict[str, str] = {}
    common_labels: np.ndarray | None = None
    for condition in conditions:
        transform_hashes[condition] = _transform_parameter_hash(
            protocol=protocol,
            image_size=int(config["img_size"]),
            family=family,
            condition=condition,
            position_domain="unseen",
            appearance_domain="unseen",
        )
        dataset = _dataset(
            test_items,
            image_size=int(config["img_size"]),
            family=family,
            condition=condition,
            protocol=protocol,
            held_out=True,
        )
        predicted, observed_labels, observed_groups, condition_records = _run_condition(
            model,
            dataset,
            indices,
            device=device,
            batch_size=args.batch_size,
            decision_rule=str(config["decision_rule"]),
            fold=fold,
            transform_parameter_sha256=transform_hashes[condition],
        )
        predictions[condition] = predicted
        groups[condition] = observed_groups
        condition_prediction_records.extend(condition_records)
        if common_labels is None:
            common_labels = observed_labels
        elif not np.array_equal(common_labels, observed_labels):
            raise RuntimeError("condition arms changed target labels")

    # Explicit seen-versus-unseen generalization disclosure for the correlated arm.
    if arm == "shortcut":
        transform_hashes["aligned_seen"] = _transform_parameter_hash(
            protocol=protocol,
            image_size=int(config["img_size"]),
            family=family,
            condition="aligned",
            position_domain="seen",
            appearance_domain="seen",
        )
        seen_dataset = _dataset(
            test_items,
            image_size=int(config["img_size"]),
            family=family,
            condition="aligned",
            protocol=protocol,
            held_out=False,
        )
        seen_result = _run_condition(
            model,
            seen_dataset,
            indices,
            device=device,
            batch_size=args.batch_size,
            decision_rule=str(config["decision_rule"]),
            fold=fold,
            transform_parameter_sha256=transform_hashes["aligned_seen"],
        )
        predictions["aligned_seen"] = seen_result[0]
        condition_prediction_records.extend(seen_result[3])

    assert common_labels is not None
    performance = condition_performance(common_labels, predictions, groups)
    main_dataset = _dataset(
        test_items,
        image_size=int(config["img_size"]),
        family=family,
        condition=main_condition,
        protocol=protocol,
        held_out=True,
    )
    transform_hashes["factorial"] = _transform_parameter_hash(
        protocol=protocol,
        image_size=int(config["img_size"]),
        family=family,
        condition="factorial",
        position_domain="unseen",
        appearance_domain="unseen",
    )
    training_condition = {
        "shortcut": "aligned",
        "cue_only": "cue_only",
        "clean": "clean",
    }[arm]
    transform_hashes["training"] = _transform_parameter_hash(
        protocol=protocol,
        image_size=int(config["img_size"]),
        family=str(config["shortcut_family"]),
        condition=training_condition,
        position_domain="seen",
        appearance_domain="seen",
        canonical_augmentation=True,
    )
    audit = _main_audit(
        model,
        main_dataset,
        indices,
        device=device,
        batch_size=args.batch_size,
        audit_grid=args.audit_grid,
        audit_seed=args.audit_seed,
        permutations=args.permutations,
        bootstrap_replicates=args.bootstrap_replicates,
        fold=fold,
        factorial_transform_sha256=transform_hashes["factorial"],
        main_transform_sha256=transform_hashes[main_condition],
        decision_rule=str(config["decision_rule"]),
        model_variant=model_variant,
    )
    factorial_prediction_records = audit.pop("factorial_prediction_records")
    internal_effect_records = audit.pop("internal_effect_records")
    checkpoint_sha256 = hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()
    all_prediction_records = [
        *condition_prediction_records,
        *factorial_prediction_records,
        *internal_effect_records,
    ]
    for record in all_prediction_records:
        record["training_seed"] = int(config.get("training_seed", config["seed"]))
        record["split_seed"] = int(config["split_seed"])
        record["checkpoint_sha256"] = checkpoint_sha256
    prediction_manifest = {
        "schema": "origin-ordinal-shortcut-predictions-v1",
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": checkpoint_sha256,
        "dataset": "aptos",
        "fold": fold,
        "shortcut_arm": arm,
        "shortcut_family": family,
        "training_seed": int(config.get("training_seed", config["seed"])),
        "split_seed": int(config["split_seed"]),
        "model_variant": model_variant,
        "comparator_implementation_origin": config.get(
            "comparator_implementation_origin", "in_repo_proposed_method"
        ),
        "official_author_implementation": False,
        "protocol_seed": protocol.seed,
        "shortcut_protocol_signature": protocol.signature,
        "transform_parameter_sha256": transform_hashes,
        "class_probability_order": [0, 1, 2, 3, 4],
        "threshold_probability_order": ["P(Y>0)", "P(Y>1)", "P(Y>2)", "P(Y>3)"],
        "factorial_state_encoding": "bit k is marker k; little-endian integer 0..15",
        "factorial_toggle_pair_rule": "same sample and other three bits, bit k changed",
        "condition_prediction_records": len(condition_prediction_records),
        "factorial_prediction_records": len(factorial_prediction_records),
        "internal_effect_records": len(internal_effect_records),
    }
    prediction_artifact_sha256 = _write_jsonl_atomic(
        prediction_artifact, prediction_manifest, all_prediction_records
    )
    report = {
        "schema": "origin-ordinal-shortcut-audit-v1",
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": checkpoint_sha256,
        "dataset": "aptos",
        "fold": fold,
        "shortcut_arm": arm,
        "shortcut_family": family,
        "training_seed": int(config.get("training_seed", config["seed"])),
        "split_seed": int(config["split_seed"]),
        "model_variant": model_variant,
        "comparator_implementation_origin": config.get(
            "comparator_implementation_origin", "in_repo_proposed_method"
        ),
        "official_author_implementation": False,
        "shortcut_protocol": protocol.as_dict(),
        "shortcut_protocol_signature": protocol.signature,
        "transform_parameter_sha256": transform_hashes,
        "prediction_artifact": str(prediction_artifact.resolve()),
        "prediction_artifact_sha256": prediction_artifact_sha256,
        "prediction_artifact_schema": "origin-ordinal-shortcut-predictions-v1",
        "condition_prediction_records": len(condition_prediction_records),
        "factorial_prediction_records": len(factorial_prediction_records),
        "internal_effect_records": len(internal_effect_records),
        "sample_count": len(indices),
        "complete_outer_fold": len(indices) == len(test_items),
        "condition_performance": performance,
        **audit,
    }
    unsigned = json_ready(report)
    report["content_checksum_sha256"] = hashlib.sha256(
        json.dumps(unsigned, sort_keys=True, separators=(",", ":"), allow_nan=False).encode(
            "utf-8"
        )
    ).hexdigest()
    _write_json_atomic(args.output, report)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "arm": arm,
                "family": family,
                "samples": len(indices),
                "macro_auprc": json_ready(report["localization"]["macro_auprc"]),
                "internal_pixel_spearman": json_ready(
                    report["internal_pixel_effects"]["internal_pixel_spearman"]
                ),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
