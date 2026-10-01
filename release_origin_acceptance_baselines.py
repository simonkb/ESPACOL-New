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
    file_sha256,
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
    edges = torch.linspace(0.0, 1.0, bins + 1, dtype=torch.float64)
    for boundary in range(boundaries):
        predicted = cumulative_probs[:, boundary].double().clamp(0.0, 1.0)
        observed = (labels.long() > boundary).double()
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
                "boundary": boundary,
                "event": f"Y>{boundary}",
                "ece": ece,
                "bins": reliability_bins,
            }
        )
    return {
        "bin_count": bins,
        "binning": "equal_width_[0,1]",
        "aggregation": "unweighted_mean_across_ordinal_boundaries",
        "threshold_ece": float(sum(eces) / len(eces)),
        "threshold_ece_by_boundary": eces,
        "boundaries": table,
    }


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
    reliability = threshold_reliability(cumulative, labels, bins=15)
    metrics["threshold_ece"] = reliability["threshold_ece"]
    metrics["threshold_ece_by_boundary"] = reliability[
        "threshold_ece_by_boundary"
    ]

    output_dir.mkdir(parents=True, exist_ok=True)
    prediction_path = output_dir / "outer_predictions.npz"
    ordered_paths = np.asarray(
        [str(test_items[int(index)][0]) for index in indices.tolist()], dtype=str
    )
    patient_ids = np.asarray(
        [
            Path(path).stem.rsplit("_", 1)[0] if task.dataset == "dr" else ""
            for path in ordered_paths.tolist()
        ],
        dtype=str,
    )
    cluster_ids = np.asarray(
        [patient if patient else path for patient, path in zip(patient_ids, ordered_paths)],
        dtype=str,
    )
    np.savez_compressed(
        prediction_path,
        sample_index=indices.numpy(),
        image_path=ordered_paths,
        patient_id=patient_ids,
        cluster_id=cluster_ids,
        label=labels.numpy(),
        prediction=predicted.numpy(),
        expected_grade=expected.numpy(),
        class_probs=probs.numpy(),
        cumulative_probs=cumulative.numpy(),
    )
    payload: dict[str, Any] = {
        "schema": "origin-acceptance-outer-result-v1",
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
            "threshold_reliability": reliability,
            "exact_per_sample_probabilities": "outer_predictions.npz:class_probs",
        },
        "test_evaluated": True,
        "selection_reopened": False,
        "predictions_sha256": file_sha256(prediction_path),
    }
    payload["content_checksum_sha256"] = canonical_sha256(payload)
    write_json_atomic(output_dir / "outer_metrics.json", payload)
    complete = {
        "schema": "origin-acceptance-outer-complete-v1",
        "protocol_sha256": PROTOCOL_SHA256,
        "task": asdict(task),
        "outer_metrics_sha256": file_sha256(output_dir / "outer_metrics.json"),
        "predictions_sha256": payload["predictions_sha256"],
    }
    write_json_atomic(output_dir / "OUTER_COMPLETE.json", complete)
    print(json.dumps({"task": task.key, "status": "released", "metrics": metrics}))


if __name__ == "__main__":
    main()
