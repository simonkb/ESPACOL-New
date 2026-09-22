#!/usr/bin/env python3
"""Pure-Python integrity and metric helpers for frozen ORIGIN-v3 full CV."""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping, Sequence


BASE_COMMIT = "b0f4609b8847d862554c84b29dec59bc562d256f"
IMPLEMENTATION_SIGNATURE = (
    "0d9734495c4aeccb2a0038f2b9672f88c445a13b276d7d1073cae1cc0c86909b"
)
ARCHITECTURE_SIGNATURE = (
    "02d2e5455d89b9dbacd1522a22fa7da4a25e91ebcaf0ce0c60de6aea214b44e6"
)
DATASET_SPECS: dict[str, dict[str, Any]] = {
    "aptos": {
        "n_folds": 5,
        "n_images": 3662,
        "epochs": 35,
        "early_stopping_patience": 12,
        "config_signature": (
            "b4409a1bc0cb7158494294c67e86e37b8710d2bf54f445e9f72f715ef65e62d7"
        ),
    },
    "dr": {
        "n_folds": 10,
        "n_images": 35126,
        "epochs": 75,
        "early_stopping_patience": 15,
        "config_signature": (
            "9ddf05ca3fd8660f7ee563799b73422f3d9f05bf446ffd40fd03c38c7ba5f914"
        ),
    },
}

PINNED_CONFIG: dict[str, Any] = {
    "n_classes": 5,
    "val_fraction": 0.1,
    "preprocessing_version": "canonical-square-fixed-ellipse-v1",
    "seed": 42,
    "img_size": 640,
    "encoder": "convnext_tiny",
    "pretrained": True,
    "evidence_scales": ["s4", "s8", "s16", "s32"],
    "projection_dim": 128,
    "grad_checkpoint": False,
    "mask_valid_fraction": 0.5,
    "reference_count": 4096.0,
    "atom_rate_init": 1e-6,
    "prior_rate_init": 1e-4,
    "boundary_scale_init": 1.0,
    "total_rate_cap": 64.0,
    "prior_rate_cap": 1.0,
    "boundary_scale_cap": 2.0,
    "rate_roundoff_margin": 1.0,
    "atom_mode": "cumulative",
    "hybrid_cumulative_init": 0.9,
    "evidence_dropout": 0.0,
    "force_decoder_fp64": True,
    "decision_rule": "class_map",
    "rps_weight": 0.25,
    "evidence_budget_weight": 0.0,
    "evidence_budget_delay_epochs": 5,
    "class_weighting": "none",
    "effective_num_beta": 0.999,
    "class_weight_cap": 10.0,
    "allow_weighted_likelihood": False,
    "batch_size": 8,
    "num_workers": 8,
    "pin_memory": True,
    "lr": 1e-4,
    "head_lr": 5e-4,
    "weight_decay": 1e-5,
    "grad_clip_norm": 5.0,
    "encoder_freeze_epochs": 2,
    "scheduler": "plateau",
    "lr_factor": 0.2,
    "lr_patience": 5,
    "lr_min": 1e-6,
    "checkpoint_selection": "acc_then_qwk",
    "selection_qwk_weight": 0.1,
    "amp": True,
    "amp_init_scale": 4096.0,
    "amp_unfreeze_scale": 256.0,
    "amp_growth_interval": 2000,
    "amp_max_consecutive_skips": 8,
    "resume": False,
    "stratified_batches": False,
    "labels_csv": None,
    "image_column": None,
    "label_column": None,
    "image_dir": None,
}


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: str | Path) -> Any:
    with Path(path).open(encoding="utf-8") as stream:
        return json.load(stream)


def write_json_atomic(path: str | Path, value: Any) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def verify_checksummed_payload(
    payload: Mapping[str, Any], *, schema: str | None = None
) -> None:
    if schema is not None and payload.get("schema") != schema:
        raise ValueError(
            f"unexpected schema {payload.get('schema')!r}; expected {schema!r}"
        )
    recorded = payload.get("content_checksum_sha256")
    if not isinstance(recorded, str):
        raise ValueError("checksummed payload has no content_checksum_sha256")
    unsigned = dict(payload)
    del unsigned["content_checksum_sha256"]
    observed = canonical_sha256(unsigned)
    if observed != recorded:
        raise ValueError(
            f"payload checksum mismatch: recorded={recorded}, observed={observed}"
        )


def values_close(left: Any, right: Any, *, atol: float = 2e-5) -> bool:
    if isinstance(left, bool) or isinstance(right, bool):
        return left is right
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return math.isfinite(float(left)) and math.isfinite(float(right)) and math.isclose(
            float(left), float(right), rel_tol=1e-7, abs_tol=atol
        )
    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        return len(left) == len(right) and all(
            values_close(a, b, atol=atol) for a, b in zip(left, right)
        )
    return left == right


def validate_pinned_config(config: Mapping[str, Any], dataset: str) -> None:
    spec = DATASET_SPECS[dataset]
    expected = dict(PINNED_CONFIG)
    expected.update(
        {
            "dataset": dataset,
            "n_folds": spec["n_folds"],
            "epochs": spec["epochs"],
            "early_stopping_patience": spec["early_stopping_patience"],
        }
    )
    mismatches = {
        key: {"expected": value, "observed": config.get(key)}
        for key, value in expected.items()
        if not values_close(config.get(key), value)
    }
    if mismatches:
        raise ValueError(f"checkpoint violates the frozen V3 config: {mismatches}")


def validate_confusion(confusion: Sequence[Sequence[Any]], classes: int = 5) -> list[list[int]]:
    if len(confusion) != classes or any(len(row) != classes for row in confusion):
        raise ValueError(f"confusion matrix must have shape {classes}x{classes}")
    result: list[list[int]] = []
    for row in confusion:
        converted: list[int] = []
        for value in row:
            number = float(value)
            if not math.isfinite(number) or number < 0 or not number.is_integer():
                raise ValueError(f"invalid confusion count: {value!r}")
            converted.append(int(number))
        result.append(converted)
    return result


def metrics_from_confusion(confusion: Sequence[Sequence[Any]]) -> dict[str, Any]:
    matrix = validate_confusion(confusion)
    classes = len(matrix)
    support = [sum(row) for row in matrix]
    predicted_support = [sum(matrix[i][j] for i in range(classes)) for j in range(classes)]
    total = sum(support)
    if total <= 0:
        raise ValueError("confusion matrix is empty")
    correct = sum(matrix[i][i] for i in range(classes))
    absolute_error = sum(
        abs(i - j) * matrix[i][j]
        for i in range(classes)
        for j in range(classes)
    )
    recalls = [matrix[i][i] / max(support[i], 1) for i in range(classes)]
    precisions = [
        matrix[i][i] / max(predicted_support[i], 1) for i in range(classes)
    ]
    f1 = [
        (2.0 * recalls[i] * precisions[i] / (recalls[i] + precisions[i]))
        if recalls[i] + precisions[i] > 0.0
        else 0.0
        for i in range(classes)
    ]
    present = [i for i, count in enumerate(support) if count > 0]
    observed_weighted = 0.0
    expected_weighted = 0.0
    denominator_scale = float(max(1, (classes - 1) ** 2))
    for i in range(classes):
        for j in range(classes):
            weight = ((i - j) ** 2) / denominator_scale
            observed_weighted += weight * matrix[i][j]
            expected_weighted += weight * support[i] * predicted_support[j] / total
    qwk = 0.0 if expected_weighted == 0.0 else 1.0 - observed_weighted / expected_weighted
    return {
        "n": total,
        "acc": 100.0 * correct / total,
        "mae": absolute_error / total,
        "qwk": qwk,
        "balanced_acc": 100.0 * sum(recalls[i] for i in present) / len(present),
        "macro_f1": sum(f1[i] for i in present) / len(present),
        "confusion": matrix,
        "per_grade_support": support,
        "per_grade_recall": recalls,
        "per_grade_f1": f1,
    }


def sum_confusions(confusions: Sequence[Sequence[Sequence[Any]]]) -> list[list[int]]:
    if not confusions:
        raise ValueError("at least one confusion matrix is required")
    converted = [validate_confusion(value) for value in confusions]
    classes = len(converted[0])
    return [
        [sum(matrix[i][j] for matrix in converted) for j in range(classes)]
        for i in range(classes)
    ]


__all__ = [
    "ARCHITECTURE_SIGNATURE",
    "BASE_COMMIT",
    "DATASET_SPECS",
    "IMPLEMENTATION_SIGNATURE",
    "PINNED_CONFIG",
    "canonical_sha256",
    "file_sha256",
    "metrics_from_confusion",
    "read_json",
    "sum_confusions",
    "validate_confusion",
    "validate_pinned_config",
    "values_close",
    "verify_checksummed_payload",
    "write_json_atomic",
]
