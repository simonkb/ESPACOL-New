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
data.  The audit also exposes empty-proof advance cases, the only known region
where the hard projected likelihood has no primary recovery gradient.
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
    PROOF_DECISION_RULES,
    decision_rule_outputs,
    proof_only_decisions,
)
from models.mosaic_model import build_mosaic_model
from training.mosaic_trainer import mosaic_implementation_signature
from utils.metrics import evaluate_predictions


SCHEMA = "mosaic-validation-decoder-audit-v1"


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


def _rate(numerator: int, denominator: int) -> float | None:
    return None if denominator == 0 else numerator / denominator


def _proof_diagnostics(
    proof_sizes: torch.Tensor,
    transitions: torch.Tensor,
    labels: torch.Tensor,
    witness_counts: torch.Tensor,
    retained_overflow: torch.Tensor,
    alpha: torch.Tensor,
) -> dict[str, Any]:
    """Expose boundary-wise hard-proof optimization failure modes."""

    proof_sizes = proof_sizes.long()
    transitions = transitions.float()
    labels = labels.long()
    boundaries = transitions.shape[1]
    rows: list[dict[str, Any]] = []
    for boundary in range(boundaries):
        at_risk = labels >= boundary
        advance = labels > boundary
        stop = labels == boundary
        empty = proof_sizes[:, boundary] == 0
        exact_zero = transitions[:, boundary] == 0
        advance_count = int(advance.sum())
        stop_count = int(stop.sum())
        empty_advance = int((empty & advance).sum())
        exact_zero_advance = int((exact_zero & advance).sum())
        rows.append(
            {
                "boundary": boundary,
                "at_risk_count": int(at_risk.sum()),
                "advance_count": advance_count,
                "stop_count": stop_count,
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
                "witness_count_mean": float(witness_counts[:, boundary].mean()),
                "retained_overflow_mass_mean": float(
                    retained_overflow[:, boundary].mean()
                ),
                "alpha_at_max_count": float(alpha[boundary, -1]),
            }
        )
    return {
        "note": (
            "An empty proof with an advance target yields an exact-zero projected "
            "transition; the primary clamped projected NLL has zero recovery "
            "gradient there, leaving only the dense auxiliary path."
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

    labels_batches: list[torch.Tensor] = []
    indices_batches: list[torch.Tensor] = []
    model_predictions: list[torch.Tensor] = []
    transitions_batches: list[torch.Tensor] = []
    log_stops_batches: list[torch.Tensor] = []
    model_classes_batches: list[torch.Tensor] = []
    proof_sizes_batches: list[torch.Tensor] = []
    witness_counts_batches: list[torch.Tensor] = []
    overflow_batches: list[torch.Tensor] = []
    alpha_reference: torch.Tensor | None = None

    use_amp = bool(cfg.amp and device.type == "cuda")
    with torch.no_grad():
        for images, masks, labels, indices in loader:
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            try:
                with autocast(device_type="cuda", enabled=use_amp):
                    output = model(images, masks, project=True)
            except FloatingPointError:
                if not use_amp:
                    raise
                with autocast(device_type="cuda", enabled=False):
                    output = model(images, masks, project=True)

            labels_batches.append(labels.long().cpu())
            indices_batches.append(indices.long().cpu())
            model_predictions.append(output.predicted_grade.long().cpu())
            transitions_batches.append(output.transitions.float().cpu())
            log_stops_batches.append(output.log_stop_probabilities.float().cpu())
            model_classes_batches.append(output.class_probabilities.float().cpu())
            proof_sizes_batches.append(output.proof.proof_size.long().cpu())
            witness_counts_batches.append(
                output.evidence.witness_probabilities.float().sum(dim=1).cpu()
            )
            overflow_batches.append(
                output.proof.retained_distribution[..., -1].float().cpu()
            )
            current_alpha = output.evidence.alpha.detach().float().cpu()
            if current_alpha.ndim == 3:
                current_alpha = current_alpha[0]
            if alpha_reference is None:
                alpha_reference = current_alpha
            elif not torch.allclose(alpha_reference, current_alpha, atol=0.0, rtol=0.0):
                raise RuntimeError("MOSAIC alpha unexpectedly changed across batches")

    labels = torch.cat(labels_batches)
    indices = torch.cat(indices_batches)
    checkpoint_predictions = torch.cat(model_predictions)
    transitions = torch.cat(transitions_batches)
    log_stops = torch.cat(log_stops_batches)
    model_classes = torch.cat(model_classes_batches)
    proof_sizes = torch.cat(proof_sizes_batches)
    witness_counts = torch.cat(witness_counts_batches)
    retained_overflow = torch.cat(overflow_batches)
    assert alpha_reference is not None

    decision = proof_only_decisions(transitions, log_stops, transition_weights)
    decoder_specs = decision_rule_outputs(decision)
    decoder_metrics = {
        name: _classification_metrics(prediction, labels, cfg.n_classes, cumulative)
        for name, (prediction, cumulative) in decoder_specs.items()
    }
    checkpoint_prediction = decoder_specs[checkpoint_decision_rule][0]
    comparisons = {
        name: _comparison(checkpoint_prediction, prediction, labels)
        for name, (prediction, _cumulative) in decoder_specs.items()
        if name != checkpoint_decision_rule
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
    raw_class_replay_error = float(
        (model_classes - decision.raw_class_probabilities).abs().max()
    )
    raw_model_decision_replay_mismatches = int(
        (checkpoint_predictions != decision.raw_mean_round).sum()
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
        raw_model_decision_replay_mismatches == 0
        and reproduction_differences["acc"] <= 1e-7
        and reproduction_differences["qwk"] <= 1e-6
        and reproduction_differences["mae"] <= 1e-7
    )
    if not metric_reproduction:
        raise RuntimeError(
            "checkpoint validation metrics were not reproduced with saved "
            f"decision rule {checkpoint_decision_rule!r}; absolute differences="
            f"{reproduction_differences}, raw-model replay mismatches="
            f"{raw_model_decision_replay_mismatches}. No audit artifacts were "
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
            "raw_model_decision_replay_mismatches": (
                raw_model_decision_replay_mismatches
            ),
        },
        "decoders": decoder_metrics,
        "comparisons_to_checkpoint_decision": comparisons,
        "hard_proof_diagnostics": _proof_diagnostics(
            proof_sizes,
            transitions,
            labels,
            witness_counts,
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
    fieldnames = [
        "sample_id",
        "true_grade",
        "raw_expected_grade",
        "deweighted_expected_grade",
        *probability_names,
        *deweighted_probability_names,
        *cumulative_names,
        *deweighted_cumulative_names,
        *decision_names,
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
            writer.writerow(row)

    print(
        f"MOSAIC decoder audit: dataset={dataset_name} fold={fold} "
        f"validation_n={len(validation_items)} "
        f"checkpoint_epoch={checkpoint.get('epoch')} "
        f"checkpoint_decision={checkpoint_decision_rule}"
    )
    print(
        "Decoder                         Acc  BalAcc  MacroF1     QWK     MAE  "
        "Changed  Help  Harm"
    )
    for name in decoder_specs:
        metrics = decoder_metrics[name]
        comparison = comparisons.get(
            name,
            {
                "changed": 0,
                "current_wrong_alternative_correct": 0,
                "current_correct_alternative_wrong": 0,
            },
        )
        print(
            f"{name:30s} {metrics['acc']:6.2f} {metrics['balanced_acc']:7.2f} "
            f"{metrics['macro_f1']:8.4f} {metrics['qwk']:7.4f} "
            f"{metrics['mae']:7.4f} {comparison['changed']:8d} "
            f"{comparison['current_wrong_alternative_correct']:5d} "
            f"{comparison['current_correct_alternative_wrong']:5d}"
        )
    print(f"Checkpoint metrics reproduced: {metric_reproduction}")
    print(f"Summary: {output_dir / 'summary.json'}")
    print(f"Predictions: {predictions_path}")


if __name__ == "__main__":
    main()
