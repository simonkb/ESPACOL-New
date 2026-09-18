#!/usr/bin/env python3
"""Audit the amplitude and direction of a trained PATHS-v3 SAPT checkpoint.

This program performs no optimization and never touches the locked outer test
split.  It replays the complete inner-validation predictor, exports its base
and transported posteriors, and then rescales only the already learned
adjacent-transport odds.  The counterfactual therefore diagnoses whether a
failed SAPT run has the right direction but insufficient amplitude; it is not
a newly selected model or a reportable validation result.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import statistics
import subprocess
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from torch.amp import autocast

from Datasets.origin_data import (
    ORIGIN_PREPROCESSING_VERSION,
    load_origin_items,
    make_origin_loaders,
    split_origin_items,
    split_paths_are_disjoint,
)
from models.paths import PathsOutput, build_paths_model
from train_origin import set_seed, split_signature, write_json_atomic
from training.origin_trainer import evaluate_origin_predictions
from training.paths_trainer import paths_implementation_signature


AUDIT_SCHEMA = "paths-v3-sapt-margin-flow-audit-v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_commit(root: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _canonical_json(payload: Any) -> str:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def _canonical_sha256(payload: Any) -> str:
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def parse_multiplier_grid(value: str | Sequence[float]) -> tuple[float, ...]:
    """Parse a unique, increasing, non-negative transport multiplier grid."""

    if isinstance(value, str):
        try:
            values = tuple(float(item.strip()) for item in value.split(",") if item.strip())
        except ValueError as exc:
            raise argparse.ArgumentTypeError(
                "transport multipliers must be comma-separated floats"
            ) from exc
    else:
        values = tuple(float(item) for item in value)
    if not values:
        raise ValueError("at least one transport multiplier is required")
    if any(not math.isfinite(item) or item < 0.0 for item in values):
        raise ValueError("transport multipliers must be finite and non-negative")
    if tuple(sorted(set(values))) != values:
        raise ValueError("transport multipliers must be unique and increasing")
    if 0.0 not in values or 1.0 not in values:
        raise ValueError("transport multiplier grid must contain both 0 and 1")
    return values


def rescale_adjacent_transport(
    base_class_probs: torch.Tensor,
    upward_odds: torch.Tensor,
    downward_odds: torch.Tensor,
    multiplier: float | torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Recompile the learned tridiagonal kernel after odds rescaling.

    ``multiplier=0`` is the trained checkpoint's V3 base, and ``multiplier=1``
    exactly replays its selected SAPT posterior.  Values above one are
    diagnostic counterfactuals; row normalization keeps every kernel
    non-negative and probability conserving.
    """

    base = torch.as_tensor(base_class_probs, dtype=torch.float64)
    up = torch.as_tensor(upward_odds, device=base.device, dtype=torch.float64)
    down = torch.as_tensor(downward_odds, device=base.device, dtype=torch.float64)
    if base.ndim < 2 or base.shape[-1] < 2:
        raise ValueError("base_class_probs must have shape (..., K), K>=2")
    expected = base.shape[:-1] + (base.shape[-1] - 1,)
    if up.shape != expected or down.shape != expected:
        raise ValueError("transport odds do not match the base posterior shape")
    scale = torch.as_tensor(multiplier, device=base.device, dtype=torch.float64)
    if scale.numel() != 1 or not bool(torch.isfinite(scale)) or float(scale) < 0.0:
        raise ValueError("multiplier must be one finite non-negative scalar")
    if not all(
        bool(torch.isfinite(value).all()) for value in (base, up, down)
    ):
        raise ValueError("base probabilities and transport odds must be finite")
    if bool((base < 0.0).any()) or bool((up < 0.0).any()) or bool((down < 0.0).any()):
        raise ValueError("probabilities and transport odds must be non-negative")
    if not torch.allclose(
        base.sum(dim=-1), torch.ones_like(base[..., 0]), atol=5e-12, rtol=0.0
    ):
        raise ValueError("base probabilities must sum to one")

    up = up * scale
    down = down * scale
    zero = torch.zeros_like(up[..., :1])
    row_up = torch.cat((up, zero), dim=-1)
    row_down = torch.cat((zero, down), dim=-1)
    normalizer = 1.0 + row_up + row_down
    diagonal = 1.0 / normalizer
    matrix = (
        torch.diag_embed(diagonal)
        + torch.diag_embed(row_up[..., :-1] / normalizer[..., :-1], offset=1)
        + torch.diag_embed(row_down[..., 1:] / normalizer[..., 1:], offset=-1)
    )
    transported = torch.matmul(base.unsqueeze(-2), matrix).squeeze(-2)
    if not torch.allclose(
        matrix.sum(dim=-1), torch.ones_like(matrix[..., 0]), atol=5e-12, rtol=0.0
    ):
        raise FloatingPointError("rescaled transport matrix is not row stochastic")
    if not torch.allclose(
        transported.sum(dim=-1),
        torch.ones_like(transported[..., 0]),
        atol=5e-12,
        rtol=0.0,
    ):
        raise FloatingPointError("rescaled transport did not conserve probability")
    return transported, matrix


def target_margin(probabilities: torch.Tensor, target_grade: int) -> torch.Tensor:
    """Return target probability minus the strongest competing grade."""

    probabilities = torch.as_tensor(probabilities)
    if probabilities.ndim < 2:
        raise ValueError("probabilities must have shape (..., K)")
    if not 0 <= int(target_grade) < probabilities.shape[-1]:
        raise ValueError("target grade is outside the posterior")
    target = probabilities[..., int(target_grade)]
    competitors = probabilities.clone()
    competitors[..., int(target_grade)] = -torch.inf
    return target - competitors.max(dim=-1).values


def first_target_multiplier(
    base_class_probs: torch.Tensor,
    upward_odds: torch.Tensor,
    downward_odds: torch.Tensor,
    *,
    target_grade: int,
    search_grid: Sequence[float],
    bisection_steps: int = 60,
) -> float | None:
    """Find the first bracketed odds multiplier making ``target_grade`` MAP.

    The result is exact within the supplied bracket and bisection tolerance.
    A ``None`` result means no crossing was observed on the registered grid;
    it does not claim mathematical impossibility beyond the grid maximum.
    """

    grid = parse_multiplier_grid(search_grid)
    if base_class_probs.ndim != 2 or base_class_probs.shape[0] != 1:
        raise ValueError("first_target_multiplier expects one posterior")

    def margin(multiplier: float) -> float:
        posterior, _ = rescale_adjacent_transport(
            base_class_probs, upward_odds, downward_odds, multiplier
        )
        return float(target_margin(posterior, target_grade).item())

    previous = float(grid[0])
    previous_margin = margin(previous)
    if previous_margin >= 0.0:
        return previous
    for current in grid[1:]:
        current = float(current)
        current_margin = margin(current)
        if current_margin >= 0.0:
            low, high = previous, current
            for _ in range(int(bisection_steps)):
                middle = 0.5 * (low + high)
                if margin(middle) >= 0.0:
                    high = middle
                else:
                    low = middle
            return high
        previous, previous_margin = current, current_margin
    return None


def _unpack_batch(batch: object, device: torch.device):
    if not isinstance(batch, (tuple, list)) or len(batch) != 4:
        raise TypeError("PATHS audit requires the four-field fundus batch contract")
    images, pixel_mask, labels, indices = batch
    return (
        images.to(device, non_blocking=True),
        pixel_mask.to(device, non_blocking=True),
        labels.to(device=device, dtype=torch.long, non_blocking=True),
        indices,
    )


def _build_model(config: Mapping[str, Any]):
    return build_paths_model(
        num_classes=int(config["n_classes"]),
        encoder_name=str(config["encoder"]),
        pretrained=False,
        evidence_scales=tuple(config["evidence_scales"]),
        projection_dim=int(config["projection_dim"]),
        reference_count=float(config["reference_count"]),
        atom_mode=str(config["atom_mode"]),
        hybrid_cumulative_init=float(config["hybrid_cumulative_init"]),
        atom_rate_init=float(config["atom_rate_init"]),
        prior_rate_init=float(config["prior_rate_init"]),
        boundary_scale_init=float(config["boundary_scale_init"]),
        total_rate_cap=float(config["total_rate_cap"]),
        prior_rate_cap=float(config["prior_rate_cap"]),
        boundary_scale_cap=float(config["boundary_scale_cap"]),
        rate_roundoff_margin=float(config["rate_roundoff_margin"]),
        evidence_dropout=float(config["evidence_dropout"]),
        mask_valid_fraction=float(config["mask_valid_fraction"]),
        grad_checkpoint=bool(config["grad_checkpoint"]),
        paths_probe_z=tuple(config["pgf_probes"]),
        paths_transport_gain_cap=float(config["transport_gain_cap"]),
        paths_transport_gain_init=float(config["transport_gain_init"]),
        paths_transport_threshold_init=float(config["transport_threshold_init"]),
        paths_transport_slope_init=float(config["transport_slope_init"]),
        paths_transport_slope_cap=float(config["transport_slope_cap"]),
        paths_strength=float(config["transport_strength"]),
    )


def _assert_metric_reproduction(
    observed: Mapping[str, Any], expected: Mapping[str, Any], tolerance: float
) -> None:
    for name in ("acc", "qwk", "mae", "balanced_acc", "macro_f1", "ece"):
        if abs(float(observed[name]) - float(expected[name])) > tolerance:
            raise AssertionError(
                f"selected validation {name} did not reproduce: "
                f"{observed[name]} vs {expected[name]}"
            )
    if observed["confusion"] != expected["confusion"]:
        raise AssertionError("selected validation confusion matrix did not reproduce")


def _relative_identity(path: str, data_root: Path) -> str:
    resolved = Path(path).expanduser().resolve(strict=False)
    try:
        return resolved.relative_to(data_root.resolve(strict=False)).as_posix()
    except ValueError:
        return resolved.name


def _scalar_metrics(metrics: Mapping[str, Any]) -> dict[str, Any]:
    return {
        name: metrics[name]
        for name in (
            "acc",
            "qwk",
            "mae",
            "balanced_acc",
            "macro_f1",
            "ece",
            "confusion",
            "per_grade_recall",
            "per_grade_support",
        )
    }


def _matched_paths_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Remove only the two prospectively registered arm differences."""

    matched = dict(config)
    matched.pop("paths_variant", None)
    matched.pop("transport_strength", None)
    return matched


def _load_bound_run(
    fold_dir: Path,
    *,
    expected_variant: str,
) -> tuple[dict[str, Any], dict[str, Any], Mapping[str, Any], Path]:
    checkpoint_path = fold_dir / "best.pth"
    result_path = fold_dir / "result.json"
    for path in (checkpoint_path, result_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    result = json.loads(result_path.read_text())
    checkpoint_sha = _sha256(checkpoint_path)
    if checkpoint_sha != result.get("best_checkpoint_sha256"):
        raise ValueError(f"{expected_variant} checkpoint SHA-256 does not match result.json")
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if state.get("schema") != "paths-checkpoint-v3-sapt":
        raise ValueError(f"{expected_variant} has the wrong checkpoint schema")
    if state.get("paths_variant") != expected_variant:
        raise ValueError(
            f"expected {expected_variant!r}, observed {state.get('paths_variant')!r}"
        )
    if result.get("paths_variant") != expected_variant:
        raise ValueError(f"{expected_variant} result declares the wrong variant")
    if int(state.get("epoch", -1)) != int(result.get("best_epoch", -2)):
        raise ValueError(f"{expected_variant} checkpoint epoch does not match result.json")
    if int(state.get("epoch", -1)) != int(result.get("best_learned_epoch", -2)):
        raise ValueError(
            f"{expected_variant} checkpoint epoch does not match best_learned_epoch"
        )
    if int(state.get("fold", -1)) != int(result.get("fold", -2)):
        raise ValueError(f"{expected_variant} checkpoint/result fold mismatch")
    if int(state.get("fold", -1)) != 0:
        raise ValueError("the registered first SAPT audit is fold 0 only")
    bindings = {
        "protocol": (state.get("paths_protocol_version"), result.get("protocol")),
        "run commit": (state.get("run_git_commit"), result.get("run_git_commit")),
        "split": (state.get("split_signature"), result.get("split_signature")),
        "implementation": (
            state.get("implementation_signature"),
            result.get("implementation_signature"),
        ),
        "architecture": (
            state.get("architecture_signature"),
            result.get("architecture_signature"),
        ),
    }
    mismatches = [name for name, pair in bindings.items() if pair[0] != pair[1]]
    if mismatches:
        raise ValueError(
            f"{expected_variant} checkpoint/result binding mismatch: {mismatches}"
        )
    if state.get("checkpoint_role") != result.get("best_learned_checkpoint_role"):
        raise ValueError(f"{expected_variant} checkpoint role is not result-bound")
    if result.get("best_learned_is_byte_exact_best_alias") is not True:
        raise ValueError(f"{expected_variant} best_learned alias is not declared byte-exact")
    if checkpoint_sha != result.get("best_learned_checkpoint_sha256"):
        raise ValueError(f"{expected_variant} best_learned checkpoint hash mismatch")
    if _canonical_json(state.get("paths_warm_start_provenance")) != _canonical_json(
        result.get("warm_start_provenance")
    ):
        raise ValueError(f"{expected_variant} warm-start provenance mismatch")
    config = state.get("critical_config")
    if not isinstance(config, Mapping):
        raise ValueError(f"{expected_variant} checkpoint has no critical configuration")
    if _canonical_json(config) != _canonical_json(result.get("critical_config")):
        raise ValueError(f"{expected_variant} checkpoint/result configuration mismatch")
    if _canonical_sha256(config) != state.get("config_signature"):
        raise ValueError(f"{expected_variant} checkpoint configuration hash mismatch")
    if state.get("config_signature") != result.get("config_signature"):
        raise ValueError(f"{expected_variant} result configuration hash mismatch")
    if _canonical_json(state.get("metrics")) != _canonical_json(
        result.get("best_learned_validation")
    ):
        raise ValueError(f"{expected_variant} checkpoint/result metric mismatch")
    return state, result, config, checkpoint_path


def _validate_static_v3_record(
    record: Mapping[str, Any],
    *,
    result: Mapping[str, Any],
    expected_variant: str,
    source_sha256: str,
) -> None:
    required = {
        "schema": "paths-v3-strength-zero-control-v2",
        "scope": "inner_validation_only",
        "fold": 0,
        "split_signature": result.get("split_signature"),
        "source_checkpoint_sha256": source_sha256,
        "paths_protocol_version": result.get("protocol"),
        "paths_variant": expected_variant,
        "implementation_signature": result.get("implementation_signature"),
        "architecture_signature": result.get("architecture_signature"),
        "reproduced": True,
    }
    mismatches = [
        name for name, expected in required.items() if record.get(name) != expected
    ]
    checksum = record.get("ordered_prediction_checksum_sha256")
    if not isinstance(checksum, str) or len(checksum) != 64 or any(
        character not in "0123456789abcdef" for character in checksum.lower()
    ):
        mismatches.append("ordered_prediction_checksum_sha256")
    if not isinstance(record.get("metrics"), Mapping):
        mismatches.append("metrics")
    if mismatches:
        raise ValueError(
            f"{expected_variant} static-V3 provenance mismatch: {sorted(set(mismatches))}"
        )


def audit(args: argparse.Namespace) -> dict[str, Any]:
    for name in ("metric_tolerance", "replay_tolerance"):
        value = float(getattr(args, name))
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(f"{name} must be finite and non-negative")
    treatment_dir = Path(args.treatment_fold_dir).expanduser().resolve()
    control_dir = Path(args.control_fold_dir).expanduser().resolve()
    state, result, config, checkpoint_path = _load_bound_run(
        treatment_dir, expected_variant="signed_transport"
    )
    control_state, control_result, control_config, control_checkpoint_path = (
        _load_bound_run(control_dir, expected_variant="risk_objective_v3")
    )
    static_path = treatment_dir / "v3_strength_zero_control.json"
    control_static_path = control_dir / "v3_strength_zero_control.json"
    for path in (static_path, control_static_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    static_v3 = json.loads(static_path.read_text())
    control_static_v3 = json.loads(control_static_path.read_text())
    checkpoint_sha = _sha256(checkpoint_path)
    control_checkpoint_sha = _sha256(control_checkpoint_path)

    if result.get("split_signature") != control_result.get("split_signature"):
        raise ValueError("treatment and matched control use different splits")
    if result.get("run_git_commit") != control_result.get("run_git_commit"):
        raise ValueError("treatment and matched control use different source commits")
    if result.get("implementation_signature") != control_result.get(
        "implementation_signature"
    ):
        raise ValueError("treatment and control have different implementation signatures")
    active_implementation = paths_implementation_signature()
    if active_implementation != result.get("implementation_signature"):
        raise ValueError(
            "active training implementation no longer matches the audited runs"
        )
    if _matched_paths_config(config) != _matched_paths_config(control_config):
        raise ValueError("treatment and matched control configurations are not matched")
    treatment_source = result.get("warm_start_provenance", {}).get(
        "source_checkpoint_sha256"
    )
    control_source = control_result.get("warm_start_provenance", {}).get(
        "source_checkpoint_sha256"
    )
    if not treatment_source or treatment_source != control_source:
        raise ValueError("treatment and control are not bound to the same V3 source")
    _validate_static_v3_record(
        static_v3,
        result=result,
        expected_variant="signed_transport",
        source_sha256=treatment_source,
    )
    _validate_static_v3_record(
        control_static_v3,
        result=control_result,
        expected_variant="risk_objective_v3",
        source_sha256=treatment_source,
    )
    if (
        static_v3.get("ordered_prediction_checksum_sha256")
        != control_static_v3.get("ordered_prediction_checksum_sha256")
    ):
        raise ValueError("the two recorded static-V3 controls are not identical")
    if str(config.get("dataset")) != "aptos":
        raise ValueError("the registered first audit is APTOS only")
    if str(control_config.get("dataset")) != "aptos":
        raise ValueError("the matched control is not an APTOS run")
    if (
        str(config.get("decision_rule")) != "class_map"
        or str(control_config.get("decision_rule")) != "class_map"
    ):
        raise ValueError("this audit's argmax metrics require the registered class_map rule")

    data_root = Path(args.data_root).expanduser().resolve()
    items = load_origin_items(
        "aptos",
        str(data_root),
        labels_csv=config.get("labels_csv"),
    )
    fold = int(state["fold"])
    if int(control_state["fold"]) != fold:
        raise ValueError("treatment and matched control use different folds")
    train_items, validation_items, test_items = split_origin_items(
        "aptos",
        items,
        fold,
        n_folds=int(config["n_folds"]),
        val_fraction=float(config["val_fraction"]),
        seed=int(config["seed"]),
    )
    if not split_paths_are_disjoint(train_items, validation_items, test_items):
        raise RuntimeError("audit split contains overlapping paths")
    observed_split = split_signature(
        ("train", train_items),
        ("validation", validation_items),
        ("locked_test", test_items),
    )
    if (
        observed_split != state.get("split_signature")
        or observed_split != result.get("split_signature")
    ):
        raise ValueError("audit split signature does not match the trained checkpoint")

    device = torch.device(
        args.device
        if args.device
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    set_seed(int(config["seed"]) + fold)
    _, validation_loader, _ = make_origin_loaders(
        train_items,
        validation_items,
        test_items,
        image_size=int(config["img_size"]),
        batch_size=int(args.batch_size or config["batch_size"]),
        num_workers=int(args.num_workers),
        pin_memory=device.type == "cuda",
        seed=int(config["seed"]) + fold,
        fundus=str(config.get("preprocessing_version")) == ORIGIN_PREPROCESSING_VERSION,
        stratified=False,
    )
    model = _build_model(config)
    model.load_state_dict(state["model_state"], strict=True)
    model.to(device).eval()
    control_model = _build_model(control_config)
    control_model.load_state_dict(control_state["model_state"], strict=True)
    control_model.to(device).eval()
    use_amp = bool(config.get("amp", True)) and device.type == "cuda"

    collected: dict[str, list[torch.Tensor]] = {
        name: []
        for name in (
            "labels",
            "indices",
            "base_probs",
            "final_probs",
            "concentration",
            "direction",
            "upward_odds",
            "downward_odds",
            "flow",
            "control_probs",
        )
    }
    with torch.no_grad():
        for batch in validation_loader:
            images, pixel_mask, labels, indices = _unpack_batch(batch, device)
            with autocast(device_type=device.type, enabled=use_amp):
                output = model(
                    images,
                    pixel_valid_mask=pixel_mask,
                    force_decoder_fp64=True,
                )
            if not isinstance(output, PathsOutput):
                raise TypeError("signed treatment bypassed the PATHS transport")
            collected["labels"].append(labels.cpu())
            collected["indices"].append(torch.as_tensor(indices).long().cpu())
            collected["base_probs"].append(output.base_output.class_probs.double().cpu())
            collected["final_probs"].append(output.class_probs.double().cpu())
            collected["concentration"].append(output.boundary_concentration.double().cpu())
            collected["direction"].append(output.transport_direction.double().cpu())
            collected["upward_odds"].append(output.upward_odds.double().cpu())
            collected["downward_odds"].append(output.downward_odds.double().cpu())
            collected["flow"].append(output.net_boundary_flow.double().cpu())
            del output
            with autocast(device_type=device.type, enabled=use_amp):
                control_output = control_model(
                    images,
                    pixel_valid_mask=pixel_mask,
                    force_decoder_fp64=True,
                )
            collected["control_probs"].append(
                control_output.class_probs.double().cpu()
            )
            del control_output
    tensors = {name: torch.cat(values, dim=0) for name, values in collected.items()}
    labels = tensors["labels"].long()
    base_probs = tensors["base_probs"]
    final_probs = tensors["final_probs"]
    control_probs = tensors["control_probs"]
    base_predictions = base_probs.argmax(dim=-1)
    final_predictions = final_probs.argmax(dim=-1)
    control_predictions = control_probs.argmax(dim=-1)
    # Training stores validation posteriors as FP32 before metric evaluation.
    # Preserve that exact convention because ECE bin membership is discontinuous.
    final_metrics = evaluate_origin_predictions(
        final_probs.float(), final_predictions, labels
    )
    base_metrics = evaluate_origin_predictions(
        base_probs.float(), base_predictions, labels
    )
    control_metrics = evaluate_origin_predictions(
        control_probs.float(), control_predictions, labels
    )
    tolerance = float(args.metric_tolerance)
    _assert_metric_reproduction(final_metrics, result["best_validation"], tolerance)
    _assert_metric_reproduction(
        control_metrics, control_result["best_validation"], tolerance
    )
    _assert_metric_reproduction(
        static_v3["metrics"], control_static_v3["metrics"], tolerance
    )
    replayed, _ = rescale_adjacent_transport(
        base_probs,
        tensors["upward_odds"],
        tensors["downward_odds"],
        1.0,
    )
    replay_error = float((replayed - final_probs).abs().max())
    if replay_error > float(args.replay_tolerance):
        raise AssertionError(
            f"multiplier-one transport replay failed by {replay_error:g}"
        )

    registered_grid = parse_multiplier_grid(args.multiplier_grid)
    # A denser registered search prevents a broad summary grid from hiding a
    # narrow grade-3 crossing.  It contains every reported sweep point.
    dense = torch.logspace(-3.0, 3.0, 1201, dtype=torch.float64).tolist()
    search_grid = tuple(sorted(set((0.0, 1.0, *registered_grid, *dense))))
    sweep: list[dict[str, Any]] = []
    base_correct = base_predictions.eq(labels)
    control_correct = control_predictions.eq(labels)
    for multiplier in registered_grid:
        probabilities, matrix = rescale_adjacent_transport(
            base_probs,
            tensors["upward_odds"],
            tensors["downward_odds"],
            multiplier,
        )
        predicted = probabilities.argmax(dim=-1)
        metrics = evaluate_origin_predictions(
            probabilities.float(), predicted, labels
        )
        correct = predicted.eq(labels)
        sweep.append(
            {
                "multiplier": multiplier,
                "metrics": _scalar_metrics(metrics),
                "prediction_change_count_vs_base": int(predicted.ne(base_predictions).sum()),
                "help_count_vs_base": int((correct & ~base_correct).sum()),
                "harm_count_vs_base": int((~correct & base_correct).sum()),
                "prediction_change_count_vs_matched_control": int(
                    predicted.ne(control_predictions).sum()
                ),
                "help_count_vs_matched_control": int(
                    (correct & ~control_correct).sum()
                ),
                "harm_count_vs_matched_control": int(
                    (~correct & control_correct).sum()
                ),
                "transport_matrix_row_sum_error": float(
                    (matrix.sum(dim=-1) - 1.0).abs().max()
                ),
            }
        )

    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else treatment_dir / "audits" / "transport_margin_v1"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "per_sample.csv"
    summary_path = output_dir / "summary.json"
    if (csv_path.exists() or summary_path.exists()) and not args.overwrite:
        raise FileExistsError(
            f"audit output exists in {output_dir}; use a new directory"
        )

    grade3_cases: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    indices = tensors["indices"].tolist()
    for sample in range(labels.numel()):
        stable_index = int(indices[sample])
        if not 0 <= stable_index < len(validation_items):
            raise IndexError("validation stable index is outside the registered split")
        label = int(labels[sample])
        base_margin = float(target_margin(base_probs[sample : sample + 1], 3))
        final_margin = float(target_margin(final_probs[sample : sample + 1], 3))
        minimum: float | None = None
        if label == 3:
            minimum = first_target_multiplier(
                base_probs[sample : sample + 1],
                tensors["upward_odds"][sample : sample + 1],
                tensors["downward_odds"][sample : sample + 1],
                target_grade=3,
                search_grid=search_grid,
            )
        row: dict[str, Any] = {
            "validation_position": sample,
            "stable_index": stable_index,
            "image_id": _relative_identity(validation_items[stable_index][0], data_root),
            "true_grade": label,
            "matched_control_prediction": int(control_predictions[sample]),
            "base_prediction": int(base_predictions[sample]),
            "final_prediction": int(final_predictions[sample]),
            "base_grade3_margin": base_margin,
            "final_grade3_margin": final_margin,
            "delta_grade3_margin": final_margin - base_margin,
            "grade3_first_bracketed_multiplier": minimum,
        }
        for grade in range(base_probs.shape[-1]):
            row[f"matched_control_p{grade}"] = float(control_probs[sample, grade])
            row[f"base_p{grade}"] = float(base_probs[sample, grade])
            row[f"final_p{grade}"] = float(final_probs[sample, grade])
        for boundary in range(tensors["concentration"].shape[-1]):
            row[f"q{boundary}"] = float(tensors["concentration"][sample, boundary])
            row[f"direction{boundary}"] = float(tensors["direction"][sample, boundary])
            row[f"upward_odds{boundary}"] = float(tensors["upward_odds"][sample, boundary])
            row[f"downward_odds{boundary}"] = float(tensors["downward_odds"][sample, boundary])
            row[f"net_flow{boundary}"] = float(tensors["flow"][sample, boundary])
        rows.append(row)
        if label == 3:
            grade3_cases.append(dict(row))

    fieldnames = list(rows[0])
    temporary_csv = csv_path.with_name(f".{csv_path.name}.{os.getpid()}.tmp")
    try:
        with temporary_csv.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temporary_csv, csv_path)
    finally:
        if temporary_csv.exists():
            temporary_csv.unlink()

    finite_minimums = [
        float(case["grade3_first_bracketed_multiplier"])
        for case in grade3_cases
        if case["grade3_first_bracketed_multiplier"] is not None
    ]
    new_crossings = [value for value in finite_minimums if value > 0.0]
    already_grade3_at_zero = sum(value == 0.0 for value in finite_minimums)
    generator_delta = base_probs - control_probs
    transport_delta = final_probs - base_probs
    net_delta = final_probs - control_probs
    generator_norm = generator_delta.square().sum(dim=-1).sqrt()
    transport_norm = transport_delta.square().sum(dim=-1).sqrt()
    dot = (generator_delta * transport_delta).sum(dim=-1)
    defined = (generator_norm > 0.0) & (transport_norm > 0.0)
    opposition_cosine = torch.zeros_like(dot)
    opposition_cosine[defined] = -dot[defined] / (
        generator_norm[defined] * transport_norm[defined]
    )
    path_length = generator_delta.abs().sum(dim=-1) + transport_delta.abs().sum(dim=-1)
    cancellation_fraction = torch.zeros_like(path_length)
    nonzero_path = path_length > 0.0
    cancellation_fraction[nonzero_path] = 1.0 - (
        net_delta.abs().sum(dim=-1)[nonzero_path] / path_length[nonzero_path]
    )
    opposed_coordinates = (generator_delta * transport_delta) < 0.0
    transport_nonzero = transport_delta != 0.0

    root = Path(__file__).resolve().parent
    refiner = model.paths_refiner
    payload: dict[str, Any] = {
        "schema": AUDIT_SCHEMA,
        "scope": "aptos_fold0_inner_validation_only_no_retraining",
        "diagnostic_only_not_model_selection": True,
        "canonical_checkpoint_result_metadata_binding_verified": True,
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_sha,
        "matched_control_checkpoint": str(control_checkpoint_path),
        "matched_control_checkpoint_sha256": control_checkpoint_sha,
        "checkpoint_epoch": int(state["epoch"]),
        "matched_control_checkpoint_epoch": int(control_state["epoch"]),
        "source_run_git_commit": result.get("run_git_commit"),
        "source_implementation_signature": result.get("implementation_signature"),
        "active_implementation_signature": active_implementation,
        "source_v3_checkpoint_sha256": treatment_source,
        "audit_git_commit": _git_commit(root),
        "split_signature": observed_split,
        "sample_count": int(labels.numel()),
        "multiplier_one_replay_max_abs_error": replay_error,
        "base_metrics": _scalar_metrics(base_metrics),
        "selected_metrics": _scalar_metrics(final_metrics),
        "matched_control_metrics": _scalar_metrics(control_metrics),
        "static_v3_metrics": _scalar_metrics(static_v3["metrics"]),
        "selected_checkpoint_generator_transport_opposition": {
            "definition": (
                "generator_delta=treatment_base-matched_control; "
                "transport_delta=treatment_final-treatment_base"
            ),
            "interpretation_limit": (
                "diagnostic association, not a causal decomposition, because the "
                "two arms selected checkpoints independently"
            ),
            "treatment_epoch": int(state["epoch"]),
            "matched_control_epoch": int(control_state["epoch"]),
            "mean_abs_generator_delta": float(generator_delta.abs().mean()),
            "mean_abs_transport_delta": float(transport_delta.abs().mean()),
            "mean_abs_net_delta": float(net_delta.abs().mean()),
            "mean_opposition_cosine_defined_samples": (
                float(opposition_cosine[defined].mean()) if bool(defined.any()) else None
            ),
            "median_opposition_cosine_defined_samples": (
                float(opposition_cosine[defined].median()) if bool(defined.any()) else None
            ),
            "mean_cancellation_fraction": float(cancellation_fraction.mean()),
            "median_cancellation_fraction": float(cancellation_fraction.median()),
            "opposed_nonzero_coordinate_fraction": (
                float(opposed_coordinates[transport_nonzero].double().mean())
                if bool(transport_nonzero.any())
                else None
            ),
            "treatment_base_prediction_change_count_vs_control": int(
                base_predictions.ne(control_predictions).sum()
            ),
            "selected_prediction_change_count_vs_control": int(
                final_predictions.ne(control_predictions).sum()
            ),
        },
        "learned_transport_parameters": {
            "probe_weights": refiner.probe_weights.detach().cpu().tolist(),
            "transport_gains": refiner.transport_gains.detach().cpu().tolist(),
            "transport_thresholds": refiner.transport_thresholds.detach().cpu().tolist(),
            "transport_slopes": refiner.transport_slopes.detach().cpu().tolist(),
        },
        "registered_multiplier_sweep": sweep,
        "grade3": {
            "support": len(grade3_cases),
            "base_prediction_histogram": {
                str(grade): sum(case["base_prediction"] == grade for case in grade3_cases)
                for grade in range(base_probs.shape[-1])
            },
            "selected_prediction_histogram": {
                str(grade): sum(case["final_prediction"] == grade for case in grade3_cases)
                for grade in range(base_probs.shape[-1])
            },
            "matched_control_prediction_histogram": {
                str(grade): sum(
                    case["matched_control_prediction"] == grade
                    for case in grade3_cases
                )
                for grade in range(base_probs.shape[-1])
            },
            "positive_margin_delta_count": sum(
                float(case["delta_grade3_margin"]) > 0.0 for case in grade3_cases
            ),
            "nonpositive_margin_delta_count": sum(
                float(case["delta_grade3_margin"]) <= 0.0 for case in grade3_cases
            ),
            "counterfactual_search_max_multiplier": float(max(search_grid)),
            "already_grade3_map_at_zero_count": already_grade3_at_zero,
            "new_bracketed_crossing_count": len(new_crossings),
            "no_crossing_observed_count": (
                len(grade3_cases) - already_grade3_at_zero - len(new_crossings)
            ),
            "first_new_crossing_multiplier_min": (
                min(new_crossings) if new_crossings else None
            ),
            "first_new_crossing_multiplier_median": (
                float(statistics.median(new_crossings)) if new_crossings else None
            ),
            "first_new_crossing_multiplier_max": (
                max(new_crossings) if new_crossings else None
            ),
            "cases": grade3_cases,
        },
        "per_sample_csv": str(csv_path),
        "per_sample_csv_sha256": _sha256(csv_path),
    }
    write_json_atomic(summary_path, payload)
    payload["summary_path"] = str(summary_path)
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="No-retraining PATHS SAPT margin/flow audit",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--treatment_fold_dir",
        default="runs/paths_aptos_f0_v3_sapt_9a860c8/fold0",
    )
    parser.add_argument(
        "--control_fold_dir",
        default="runs/paths_aptos_f0_v3_risk_control_9a860c8/fold0",
    )
    parser.add_argument(
        "--data_root",
        default="Datasets/aptos2019-blindness-detection",
    )
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--multiplier_grid",
        type=parse_multiplier_grid,
        default=parse_multiplier_grid("0,0.25,0.5,1,2,4,8,16,32,64"),
    )
    parser.add_argument("--metric_tolerance", type=float, default=1e-5)
    parser.add_argument("--replay_tolerance", type=float, default=2e-5)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    payload = audit(args)
    print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
