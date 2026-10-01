#!/usr/bin/env python3
"""Post-freeze outer-test release worker for one registered baseline task."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.amp import autocast
from torch.utils.data import DataLoader

from configs.origin_acceptance_baseline_config import (
    OriginAcceptanceBaselineConfig,
)
from Datasets.origin_data import (
    OriginFundusTransform,
    OriginImageDataset,
    load_origin_items,
    split_origin_items,
)
from losses.origin import categorical_nll, ranked_probability_score
from models.origin_acceptance_baselines import build_origin_acceptance_baseline
from scripts.origin_acceptance_baseline_common import (
    PROTOCOL_ID,
    PROTOCOL_SHA256,
    canonical_sha256,
    dataset_scoped_identifier,
    file_sha256,
    release_identifier_policy,
    task_at,
    worker_run_dir,
)
from train_origin import split_signature, write_json_atomic
from training.origin_trainer import evaluate_origin_predictions


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object at {path}")
    return value


def _verify_gate(path: Path) -> dict[str, Any]:
    gate = _load_json(path)
    checksum = gate.get("content_checksum_sha256")
    unsigned = dict(gate)
    unsigned.pop("content_checksum_sha256", None)
    if checksum != canonical_sha256(unsigned):
        raise ValueError("full-training gate checksum mismatch")
    if gate.get("status") != "passed" or gate.get("scope") != "full":
        raise ValueError("outer release requires a passed full-training audit")
    if gate.get("protocol_sha256") != PROTOCOL_SHA256:
        raise ValueError("outer release gate belongs to another protocol")
    if gate.get("outer_test_released") is not False:
        raise ValueError("training gate unexpectedly claims an outer release")
    return gate


def _decision(output: object, rule: str, n_classes: int) -> torch.Tensor:
    if rule == "posterior_median":
        value = output.posterior_median
    elif rule == "class_map":
        value = output.class_map
    elif rule == "rounded_expected":
        value = output.expected_grade.round()
    else:
        raise ValueError(f"unknown decision rule {rule!r}")
    return value.long().clamp(0, n_classes - 1)


def multiclass_brier_score(
    class_probs: torch.Tensor, labels: torch.Tensor
) -> float:
    """Mean unnormalized multiclass Brier score, sum over all classes."""

    if class_probs.ndim != 2 or labels.shape != class_probs.shape[:1]:
        raise ValueError("Brier inputs must have shapes (N,K) and (N,)")
    target = torch.nn.functional.one_hot(
        labels.long(), num_classes=class_probs.shape[1]
    ).to(class_probs.dtype)
    return float((class_probs - target).square().sum(dim=1).mean())


def threshold_reliability(
    cumulative_probs: torch.Tensor,
    labels: torch.Tensor,
    *,
    bins: int = 15,
) -> dict[str, Any]:
    """Fixed-bin calibration for every ordinal event ``Y > k``."""

    if cumulative_probs.ndim != 2 or labels.shape != cumulative_probs.shape[:1]:
        raise ValueError("threshold calibration inputs have incompatible shapes")
    if bins < 2:
        raise ValueError("threshold calibration requires at least two bins")
    n, boundaries = cumulative_probs.shape
    if n < 1:
        raise ValueError("threshold calibration requires at least one sample")
    table: list[dict[str, Any]] = []
    eces: list[float] = []
    briers: list[float] = []
    edges = torch.linspace(0.0, 1.0, bins + 1, dtype=torch.float64)
    for boundary in range(boundaries):
        predicted = cumulative_probs[:, boundary].double().clamp(0.0, 1.0)
        observed = (labels.long() > boundary).double()
        binary_brier = float((predicted - observed).square().mean())
        assignments = torch.clamp((predicted * bins).long(), max=bins - 1)
        reliability_bins: list[dict[str, Any]] = []
        ece = 0.0
        for index in range(bins):
            selected = assignments == index
            count = int(selected.sum())
            if count:
                mean_predicted = float(predicted[selected].mean())
                empirical = float(observed[selected].mean())
                gap = abs(mean_predicted - empirical)
                ece += count / n * gap
            else:
                mean_predicted = None
                empirical = None
                gap = None
            reliability_bins.append(
                {
                    "bin": index,
                    "lower": float(edges[index]),
                    "upper": float(edges[index + 1]),
                    "right_edge_inclusive": index == bins - 1,
                    "count": count,
                    "mean_predicted": mean_predicted,
                    "empirical_frequency": empirical,
                    "absolute_gap": gap,
                }
            )
        eces.append(ece)
        briers.append(binary_brier)
        table.append(
            {
                "boundary": boundary,
                "event": f"Y>{boundary}",
                "ece": ece,
                "binary_brier": binary_brier,
                "bins": reliability_bins,
            }
        )
    return {
        "bin_count": bins,
        "binning": "equal_width_[0,1]",
        "aggregation": "unweighted_mean_across_ordinal_boundaries",
        "mean_binary_brier_identity": (
            "equals_ranked_probability_score_for_this_boundary_mean_definition"
        ),
        "threshold_ece": float(sum(eces) / len(eces)),
        "threshold_ece_by_boundary": eces,
        "threshold_binary_brier": float(sum(briers) / len(briers)),
        "threshold_binary_brier_by_boundary": briers,
        "boundaries": table,
    }


def classwise_reliability(
    class_probs: torch.Tensor,
    labels: torch.Tensor,
    *,
    bins: int = 15,
) -> dict[str, Any]:
    """Fixed-bin one-vs-rest reliability for every mutually exclusive class.

    This is deliberately distinct from top-label ECE: every class contributes
    one binary calibration problem, including all negative examples, and the
    reported scalar is the unweighted mean of the per-class ECEs.
    """

    if class_probs.ndim != 2 or labels.shape != class_probs.shape[:1]:
        raise ValueError("classwise calibration inputs have incompatible shapes")
    if bins < 2:
        raise ValueError("classwise calibration requires at least two bins")
    n, classes = class_probs.shape
    if n < 1 or classes < 2:
        raise ValueError("classwise calibration requires samples and two classes")
    edges = torch.linspace(0.0, 1.0, bins + 1, dtype=torch.float64)
    table: list[dict[str, Any]] = []
    eces: list[float] = []
    for class_index in range(classes):
        predicted = class_probs[:, class_index].double().clamp(0.0, 1.0)
        observed = (labels.long() == class_index).double()
        assignments = torch.clamp((predicted * bins).long(), max=bins - 1)
        reliability_bins: list[dict[str, Any]] = []
        ece = 0.0
        for index in range(bins):
            selected = assignments == index
            count = int(selected.sum())
            if count:
                mean_predicted = float(predicted[selected].mean())
                empirical = float(observed[selected].mean())
                gap = abs(mean_predicted - empirical)
                ece += count / n * gap
            else:
                mean_predicted = None
                empirical = None
                gap = None
            reliability_bins.append(
                {
                    "bin": index,
                    "lower": float(edges[index]),
                    "upper": float(edges[index + 1]),
                    "right_edge_inclusive": index == bins - 1,
                    "count": count,
                    "mean_predicted": mean_predicted,
                    "empirical_frequency": empirical,
                    "absolute_gap": gap,
                }
            )
        eces.append(ece)
        table.append(
            {
                "class": class_index,
                "event": f"Y={class_index}",
                "prevalence": float(observed.mean()),
                "ece": ece,
                "bins": reliability_bins,
            }
        )
    return {
        "bin_count": bins,
        "binning": "equal_width_[0,1]",
        "definition": "one_vs_rest_per_class",
        "aggregation": "unweighted_mean_across_classes",
        "classwise_ece": float(sum(eces) / len(eces)),
        "classwise_ece_by_class": eces,
        "classes": table,
    }


def privacy_safe_release_identifiers(
    dataset: str,
    ordered_paths: list[str] | np.ndarray,
    *,
    data_root: str | Path,
) -> tuple[np.ndarray, np.ndarray]:
    """Hash dataset-relative image and cluster keys for public artifacts."""

    root = Path(data_root).expanduser().resolve()
    relative_keys: list[str] = []
    raw_cluster_keys: list[str] = []
    for value in ordered_paths:
        path = Path(str(value)).expanduser().resolve()
        try:
            relative = path.relative_to(root).as_posix()
        except ValueError as exc:
            raise ValueError(
                f"release image is outside its declared dataset root: {path}"
            ) from exc
        relative_keys.append(relative)
        raw_cluster_keys.append(
            path.stem.rsplit("_", 1)[0] if dataset == "dr" else relative
        )
    image_ids = np.asarray(
        [dataset_scoped_identifier(dataset, "image", key) for key in relative_keys],
        dtype=str,
    )
    cluster_namespace = "patient_cluster" if dataset == "dr" else "image_cluster"
    cluster_ids = np.asarray(
        [
            dataset_scoped_identifier(dataset, cluster_namespace, key)
            for key in raw_cluster_keys
        ],
        dtype=str,
    )
    if len(set(image_ids.tolist())) != len(image_ids):
        raise ValueError("dataset-relative image identifiers are not unique")
    return image_ids, cluster_ids


def write_privacy_safe_prediction_archive(
    path: str | Path,
    *,
    sample_index: np.ndarray,
    image_ids: np.ndarray,
    cluster_ids: np.ndarray,
    labels: np.ndarray,
    predictions: np.ndarray,
    expected_grade: np.ndarray,
    class_probs: np.ndarray,
    cumulative_probs: np.ndarray,
) -> None:
    """Write the fixed public schema without paths or raw patient identifiers."""

    hexadecimal = set("0123456789abcdef")
    for name, values in (("image_id", image_ids), ("cluster_id", cluster_ids)):
        strings = np.asarray(values).astype(str)
        if strings.ndim != 1 or any(
            len(value) != 64 or not set(value) <= hexadecimal for value in strings
        ):
            raise ValueError(f"{name} must contain dataset-scoped SHA-256 identifiers")
    np.savez_compressed(
        path,
        sample_index=sample_index,
        image_id=image_ids,
        cluster_id=cluster_ids,
        label=labels,
        prediction=predictions,
        expected_grade=expected_grade,
        class_probs=class_probs,
        cumulative_probs=cumulative_probs,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment_root", required=True)
    parser.add_argument("--task_index", type=int, required=True)
    parser.add_argument("--aptos_root", default="Datasets/aptos2019-blindness-detection")
    parser.add_argument("--dr_root", default="Datasets/DR")
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    root = Path(args.experiment_root)
    gate_path = root / "full" / "TRAINING_FROZEN.json"
    gate = _verify_gate(gate_path)
    task = task_at("full", args.task_index)
    output_dir = root / "full" / "outer_release" / task.key
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"release worker refuses to reopen the outer test or overwrite {output_dir}"
        )
    fold_dir = worker_run_dir(root, "full", task) / f"fold{task.fold}"
    completion = _load_json(fold_dir / "TRAINING_COMPLETE.json")
    if completion.get("protocol_sha256") != PROTOCOL_SHA256:
        raise ValueError("worker completion belongs to another protocol")
    result = _load_json(fold_dir / "result.json")
    if result.get("test_evaluated") is not False:
        raise ValueError("worker result is not outer-test locked")

    checkpoint_path = fold_dir / "best_learned.pth"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint.get("acceptance_schema") != "origin-acceptance-learned-checkpoint-v1":
        raise ValueError("best learned checkpoint has the wrong schema")
    if int(checkpoint.get("epoch", 0)) < 1:
        raise ValueError("outer release refuses an epoch-zero checkpoint")
    cfg = OriginAcceptanceBaselineConfig(**dict(checkpoint["config"]))
    if cfg.baseline_variant != task.baseline_variant:
        raise ValueError("checkpoint/task baseline mismatch")
    if cfg.training_seed != task.training_seed or checkpoint.get("fold") != task.fold:
        raise ValueError("checkpoint/task seed or fold mismatch")

    data_root = args.aptos_root if task.dataset == "aptos" else args.dr_root
    items = load_origin_items(
        task.dataset,
        data_root,
        labels_csv=cfg.labels_csv,
        image_column=cfg.image_column,
        label_column=cfg.label_column,
        image_dir=cfg.image_dir,
    )
    train_items, val_items, test_items = split_origin_items(
        task.dataset,
        items,
        task.fold,
        n_folds=cfg.n_folds,
        val_fraction=cfg.val_fraction,
        seed=cfg.seed,
    )
    signature = split_signature(
        ("train", train_items),
        ("validation", val_items),
        ("locked_test", test_items),
    )
    if signature != checkpoint.get("split_signature") or signature != result.get(
        "split_signature"
    ):
        raise ValueError("reconstructed release split does not match training")

    transform = OriginFundusTransform(cfg.img_size, augment=False)
    dataset = OriginImageDataset(test_items, transform)
    device = torch.device(
        args.device
        if args.device is not None
        else "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    workers = cfg.num_workers if args.num_workers is None else args.num_workers
    loader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=0 if device.type == "mps" else workers,
        pin_memory=device.type == "cuda",
        persistent_workers=False,
    )
    model = build_origin_acceptance_baseline(cfg, pretrained=False)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.to(device).eval()

    probabilities: list[torch.Tensor] = []
    log_probabilities: list[torch.Tensor] = []
    cumulatives: list[torch.Tensor] = []
    expected_values: list[torch.Tensor] = []
    predictions: list[torch.Tensor] = []
    labels_all: list[torch.Tensor] = []
    indices_all: list[torch.Tensor] = []
    use_amp = bool(cfg.amp and device.type == "cuda")
    with torch.no_grad():
        for images, pixel_mask, labels, indices in loader:
            images = images.to(device, non_blocking=True)
            pixel_mask = pixel_mask.to(device, non_blocking=True)
            with autocast(device_type="cuda", enabled=use_amp):
                output = model(
                    images,
                    pixel_valid_mask=pixel_mask,
                    force_decoder_fp64=True,
                )
            probabilities.append(output.class_probs.cpu())
            log_probabilities.append(output.log_class_probs.cpu())
            cumulatives.append(output.cumulative_probs.cpu())
            expected_values.append(output.expected_grade.cpu())
            predictions.append(_decision(output, cfg.decision_rule, cfg.n_classes).cpu())
            labels_all.append(labels.long().cpu())
            indices_all.append(indices.long().cpu())

    probs = torch.cat(probabilities)
    log_probs = torch.cat(log_probabilities)
    cumulative = torch.cat(cumulatives)
    expected = torch.cat(expected_values)
    predicted = torch.cat(predictions)
    labels = torch.cat(labels_all)
    indices = torch.cat(indices_all)
    if len(labels) != len(test_items):
        raise RuntimeError("outer loader did not emit every locked test item")
    metrics = evaluate_origin_predictions(probs.float(), predicted, labels)
    metrics.update(
        {
            "nll": float(categorical_nll(log_probs, labels)),
            "rps": float(ranked_probability_score(cumulative, labels)),
            "expected_grade_mae": float((expected - labels).abs().mean()),
            "mean_expected_grade": float(expected.mean()),
            "multiclass_brier": multiclass_brier_score(probs, labels),
            "n": int(len(labels)),
        }
    )
    threshold_calibration = threshold_reliability(cumulative, labels, bins=15)
    classwise_calibration = classwise_reliability(probs, labels, bins=15)
    metrics["threshold_ece"] = threshold_calibration["threshold_ece"]
    metrics["threshold_ece_by_boundary"] = threshold_calibration[
        "threshold_ece_by_boundary"
    ]
    metrics["threshold_binary_brier"] = threshold_calibration[
        "threshold_binary_brier"
    ]
    metrics["threshold_binary_brier_by_boundary"] = threshold_calibration[
        "threshold_binary_brier_by_boundary"
    ]
    metrics["classwise_ece"] = classwise_calibration["classwise_ece"]
    metrics["classwise_ece_by_class"] = classwise_calibration[
        "classwise_ece_by_class"
    ]

    output_dir.mkdir(parents=True, exist_ok=True)
    prediction_path = output_dir / "outer_predictions.npz"
    ordered_paths = [str(test_items[int(index)][0]) for index in indices.tolist()]
    image_ids, cluster_ids = privacy_safe_release_identifiers(
        task.dataset,
        ordered_paths,
        data_root=data_root,
    )
    write_privacy_safe_prediction_archive(
        prediction_path,
        sample_index=indices.numpy(),
        image_ids=image_ids,
        cluster_ids=cluster_ids,
        labels=labels.numpy(),
        predictions=predicted.numpy(),
        expected_grade=expected.numpy(),
        class_probs=probs.numpy(),
        cumulative_probs=cumulative.numpy(),
    )
    payload: dict[str, Any] = {
        "schema": "origin-acceptance-outer-result-v2",
        "protocol_id": PROTOCOL_ID,
        "protocol_sha256": PROTOCOL_SHA256,
        "task": asdict(task),
        "split_signature": signature,
        "training_gate_checksum": gate["content_checksum_sha256"],
        "best_learned_checkpoint_sha256": file_sha256(checkpoint_path),
        "best_epoch": int(checkpoint["epoch"]),
        "decision_rule": cfg.decision_rule,
        "metrics": metrics,
        "posterior_quality": {
            "multiclass_brier_definition": "mean_sum_k_(p_k-onehot_k)^2",
            "threshold_binary_brier_definition": "mean_(P(Y>k)-1[Y>k])^2",
            "threshold_binary_brier_identity": (
                "unweighted_boundary_mean_equals_reported_ranked_probability_score"
            ),
            "threshold_reliability": threshold_calibration,
            "classwise_ece_definition": (
                "15_equal_width_bins_one_vs_rest_per_class_then_unweighted_class_mean"
            ),
            "classwise_reliability": classwise_calibration,
            "exact_per_sample_probabilities": "outer_predictions.npz:class_probs",
        },
        "identifier_privacy": release_identifier_policy(task.dataset),
        "test_evaluated": True,
        "selection_reopened": False,
        "predictions_sha256": file_sha256(prediction_path),
    }
    payload["content_checksum_sha256"] = canonical_sha256(payload)
    write_json_atomic(output_dir / "outer_metrics.json", payload)
    complete = {
        "schema": "origin-acceptance-outer-complete-v2",
        "protocol_sha256": PROTOCOL_SHA256,
        "task": asdict(task),
        "outer_metrics_sha256": file_sha256(output_dir / "outer_metrics.json"),
        "predictions_sha256": payload["predictions_sha256"],
    }
    write_json_atomic(output_dir / "OUTER_COMPLETE.json", complete)
    print(json.dumps({"task": task.key, "status": "released", "metrics": metrics}))


if __name__ == "__main__":
    main()
