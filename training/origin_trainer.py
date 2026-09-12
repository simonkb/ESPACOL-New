"""Numerically guarded training and evaluation for ORIGIN.

This trainer deliberately depends on a small structural interface instead of a
particular encoder implementation.  An ORIGIN model must return an object with
``class_probs``, ``log_class_probs``, ``cumulative_probs``,
``expected_grade``, ``posterior_median``, ``class_map``, ``total_rates``,
``local_rate_maps``, and ``valid_masks``.  Its forward pass must keep the
generator/matrix-exponential decoder in FP64, even when the visual encoder runs
under AMP.  Exact certificates use ``model.replay_without(output, masks)``.

Validation drives every training decision.  The outer test loader is untouched
unless ``fit(evaluate_test=True)`` is requested explicitly.
"""

from __future__ import annotations

import csv
import hashlib
import inspect
import json
import logging
import math
import os
import random
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Subset

from losses.origin import OriginLoss
from models.origin import aggregate_ordinal_pair_messages, bounded_rate_log_odds_merge


logger = logging.getLogger(__name__)


_IMPLEMENTATION_FILES = (
    "configs/origin_config.py",
    "Datasets/origin_data.py",
    "Datasets/mosaic_data.py",
    "Datasets/dataloaders.py",
    "models/origin_encoder.py",
    "models/origin.py",
    "losses/origin.py",
    "training/origin_trainer.py",
    "train_origin.py",
    "utils/spatial_mask.py",
)

_RATE_CAP_WARNING_FRACTION = 0.80

_RUNTIME_CONFIG_FIELDS = {
    "epochs",             # may be extended after a preemption
    "resume",             # invocation policy, not numerical identity
    "evaluate_test",      # evaluation authorization only
    "run_dir",
    "output_dir",
    "warm_start_checkpoint",  # path is transport; SHA-256 is the content identity
}

_RELATION_PARAMETER_PREFIX = "generator.relation_field."
_V3_WARMSTART_MODEL_FIELDS = (
    "dataset",
    "n_classes",
    "n_folds",
    "val_fraction",
    "preprocessing_version",
    "seed",
    "img_size",
    "encoder",
    "evidence_scales",
    "projection_dim",
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
    "decision_rule",
)

_DECISION_RULES = ("posterior_median", "class_map", "rounded_expected")
_SELECTION_POLICIES = ("acc_then_qwk", "acc_qwk_score")


def _cfg_get(cfg: object, name: str, default: Any = None) -> Any:
    if isinstance(cfg, Mapping):
        return cfg.get(name, default)
    return getattr(cfg, name, default)


def _config_dict(cfg: object) -> dict[str, Any]:
    if is_dataclass(cfg):
        return dict(asdict(cfg))
    if isinstance(cfg, Mapping):
        return dict(cfg)
    try:
        return dict(vars(cfg))
    except TypeError as exc:
        raise TypeError("ORIGIN config must be a dataclass, mapping, or namespace") from exc


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.device):
        return str(value)
    if isinstance(value, torch.dtype):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    return repr(value)


def _canonical_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
        default=_json_default,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def origin_implementation_signature() -> str:
    """Hash every available source file defining the ORIGIN experiment.

    During isolated unit tests some integration files may not yet exist, so the
    manifest itself is hashed and missing optional paths are recorded.  In a
    real run all listed files exist; adding one then necessarily changes the
    signature and prevents an incompatible resume.
    """

    root = Path(__file__).resolve().parents[1]
    digest = hashlib.sha256()
    for relative in _IMPLEMENTATION_FILES:
        path = root / relative
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        if path.is_file():
            digest.update(path.read_bytes())
        else:
            digest.update(b"<missing>")
        digest.update(b"\0")
    return digest.hexdigest()


def _architecture_record(model: nn.Module) -> dict[str, Any]:
    metadata = getattr(model, "architecture_metadata", None)
    declared = metadata() if callable(metadata) else metadata
    if declared is not None and not isinstance(declared, Mapping):
        raise TypeError("model.architecture_metadata must be a mapping or callable")
    state_shapes = {
        name: {"shape": list(tensor.shape), "dtype": str(tensor.dtype)}
        for name, tensor in model.state_dict().items()
    }
    return {
        "model_class": f"{model.__class__.__module__}.{model.__class__.__qualname__}",
        "declared": dict(declared or {}),
        "state_shapes": state_shapes,
        "no_classifier_bypass": True,
        "posterior_path": "conserved_local_rates_to_pure_birth_matrix_exponential",
    }


def _critical_config(cfg: object) -> dict[str, Any]:
    return {
        key: value
        for key, value in _config_dict(cfg).items()
        if key not in _RUNTIME_CONFIG_FIELDS
    }


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _normalized_identity_value(value: Any) -> Any:
    if isinstance(value, (tuple, list)):
        return tuple(_normalized_identity_value(item) for item in value)
    if isinstance(value, np.generic):
        return value.item()
    return value


def load_origin_v3_relation_warm_start(
    model: nn.Module,
    cfg: object,
    *,
    fold: int,
    split_signature: str | None,
) -> dict[str, Any] | None:
    """Strictly migrate one immutable v3 checkpoint into a v6 relation model.

    The source implementation hash cannot equal the current v6 source tree.
    Instead, the checkpoint is bound to caller-supplied SHA-256 content
    identity, and its own architecture/configuration records are checked for
    internal consistency before a strict-subset state migration.  A hash alone
    does not establish source authorship.  The only target keys allowed to be absent
    are the newly introduced relation-field parameters.
    """

    checkpoint_value = _cfg_get(cfg, "warm_start_checkpoint", None)
    if checkpoint_value is None:
        return None
    expected_sha256 = str(_cfg_get(cfg, "warm_start_sha256", "")).lower()
    if len(expected_sha256) != 64 or any(
        character not in "0123456789abcdef" for character in expected_sha256
    ):
        raise ValueError("v3 warm start requires an explicit hexadecimal SHA-256")
    checkpoint_path = Path(str(checkpoint_value)).expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"ORIGIN-v3 warm-start checkpoint not found: {checkpoint_path}")
    observed_sha256 = _file_sha256(checkpoint_path)
    if observed_sha256 != expected_sha256:
        raise ValueError(
            "ORIGIN-v3 warm-start SHA-256 mismatch: "
            f"expected {expected_sha256}, observed {observed_sha256}"
        )

    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(state, Mapping) or state.get("schema") != "origin-checkpoint-v3":
        raise ValueError("v6 warm start requires an origin-checkpoint-v3 source")
    if int(state.get("fold", -1)) != int(fold):
        raise ValueError("ORIGIN-v3 warm-start fold mismatch")
    if split_signature is not None and state.get("split_signature") != split_signature:
        raise ValueError("ORIGIN-v3 warm-start split signature mismatch")

    architecture = state.get("architecture")
    if not isinstance(architecture, Mapping):
        raise ValueError("ORIGIN-v3 warm-start checkpoint has no architecture record")
    if state.get("architecture_signature") != _canonical_sha256(architecture):
        raise ValueError("ORIGIN-v3 warm-start architecture signature is invalid")
    declared = architecture.get("declared")
    if not isinstance(declared, Mapping):
        raise ValueError("ORIGIN-v3 warm-start architecture declaration is missing")
    required_v3_declaration = {
        "name": "ORIGIN",
        "encoder": "convnext_tiny",
        "evidence_scales": ["s4", "s8", "s16", "s32"],
        "atom_mode": "cumulative",
        "evidence_dependency_policy": "native_convnext_stage_receptive_fields",
        "no_classifier_bypass": True,
    }
    for name, expected in required_v3_declaration.items():
        if _normalized_identity_value(declared.get(name)) != _normalized_identity_value(expected):
            raise ValueError(
                f"warm-start source is not the audited v3 baseline: {name}="
                f"{declared.get(name)!r}, expected {expected!r}"
            )
    # Current ORIGIN metadata records the disabled optional field explicitly as
    # ``relation_contract=None``.  Old v3 checkpoints may omit it entirely.
    # Either representation is a valid non-relational source; only a populated
    # contract (or enabled flag) means the source is already relational.
    if bool(declared.get("relation_enabled", False)) or declared.get(
        "relation_contract"
    ) is not None:
        raise ValueError("warm-start source already contains a relation field")

    critical = state.get("critical_config")
    if not isinstance(critical, Mapping):
        raise ValueError("ORIGIN-v3 warm-start critical configuration is missing")
    if state.get("config_signature") != _canonical_sha256(critical):
        raise ValueError("ORIGIN-v3 warm-start configuration signature is invalid")
    source_config = state.get("config")
    if not isinstance(source_config, Mapping):
        raise ValueError("ORIGIN-v3 warm-start configuration is missing")
    if _critical_config(source_config) != dict(critical):
        raise ValueError(
            "ORIGIN-v3 warm-start full and critical configurations are inconsistent"
        )
    current_config = _config_dict(cfg)
    for name in _V3_WARMSTART_MODEL_FIELDS:
        if name not in source_config:
            raise ValueError(f"ORIGIN-v3 warm-start configuration omits {name!r}")
        source_value = _normalized_identity_value(source_config[name])
        current_value = _normalized_identity_value(current_config.get(name))
        if source_value != current_value:
            raise ValueError(
                f"ORIGIN-v3 warm-start {name} mismatch: "
                f"source={source_value!r}, current={current_value!r}"
            )

    source_model_state = state.get("model_state")
    if not isinstance(source_model_state, Mapping):
        raise ValueError("ORIGIN-v3 warm-start model state is missing")
    recorded_shapes = architecture.get("state_shapes")
    if not isinstance(recorded_shapes, Mapping):
        raise ValueError("ORIGIN-v3 warm-start state-shape record is missing")
    if set(recorded_shapes) != set(source_model_state):
        raise ValueError("ORIGIN-v3 warm-start state-shape key set is inconsistent")
    for name, tensor in source_model_state.items():
        record = recorded_shapes[name]
        if not torch.is_tensor(tensor) or not isinstance(record, Mapping):
            raise ValueError(f"invalid ORIGIN-v3 state entry {name!r}")
        if list(tensor.shape) != list(record.get("shape", [])):
            raise ValueError(f"ORIGIN-v3 recorded shape differs for {name!r}")
        if str(tensor.dtype) != str(record.get("dtype")):
            raise ValueError(f"ORIGIN-v3 recorded dtype differs for {name!r}")

    target_state = model.state_dict()
    unexpected = sorted(set(source_model_state) - set(target_state))
    missing = sorted(set(target_state) - set(source_model_state))
    expected_missing = sorted(
        name for name in target_state if name.startswith(_RELATION_PARAMETER_PREFIX)
    )
    if unexpected:
        raise ValueError(f"ORIGIN-v3 warm start has unexpected model keys: {unexpected}")
    if not expected_missing or missing != expected_missing:
        raise ValueError(
            "ORIGIN-v3 warm start is not a strict relation-only migration; "
            f"missing={missing}, expected={expected_missing}"
        )
    for name, source_tensor in source_model_state.items():
        if tuple(source_tensor.shape) != tuple(target_state[name].shape):
            raise ValueError(f"ORIGIN-v3 warm-start tensor shape differs for {name!r}")
    incompatibility = model.load_state_dict(source_model_state, strict=False)
    if sorted(incompatibility.missing_keys) != expected_missing or incompatibility.unexpected_keys:
        raise AssertionError("PyTorch state migration disagrees with the audited key partition")

    implementation_signature = str(state.get("implementation_signature", ""))
    if len(implementation_signature) != 64 or any(
        character not in "0123456789abcdef" for character in implementation_signature.lower()
    ):
        raise ValueError("ORIGIN-v3 source implementation signature is malformed")
    return {
        "schema": "origin-v3-to-v6-verified-content-warm-start-v1",
        "source_checkpoint": str(checkpoint_path),
        "source_checkpoint_sha256": observed_sha256,
        "source_checkpoint_schema": str(state["schema"]),
        "source_epoch": int(state.get("epoch", -1)),
        "source_fold": int(state["fold"]),
        "source_split_signature": str(state.get("split_signature")),
        "source_implementation_signature": implementation_signature,
        "source_architecture_signature": str(state["architecture_signature"]),
        "source_config_signature": str(state["config_signature"]),
        "source_metrics": dict(state.get("metrics", {})),
        "loaded_key_count": len(source_model_state),
        "initialized_relation_keys": expected_missing,
    }


def _tensor_field(output: object, name: str) -> torch.Tensor:
    value = output.get(name) if isinstance(output, Mapping) else getattr(output, name, None)
    if not torch.is_tensor(value):
        raise TypeError(f"ORIGIN output field {name!r} must be a tensor")
    return value


def _mapping_field(output: object, name: str) -> Mapping[str, torch.Tensor]:
    value = output.get(name) if isinstance(output, Mapping) else getattr(output, name, None)
    if not isinstance(value, Mapping):
        raise TypeError(f"ORIGIN output field {name!r} must be a mapping")
    return value


def _labels_from_dataset(dataset: object) -> list[int] | None:
    """Best-effort extraction of complete fold labels for optional weighting."""

    if isinstance(dataset, Subset):
        parent = _labels_from_dataset(dataset.dataset)
        if parent is None:
            return None
        return [int(parent[int(index)]) for index in dataset.indices]
    for attribute in ("targets", "labels"):
        values = getattr(dataset, attribute, None)
        if values is not None:
            tensor = torch.as_tensor(values, dtype=torch.long).flatten()
            if len(tensor) == len(dataset):
                return [int(value) for value in tensor.tolist()]
    items = getattr(dataset, "items", None)
    if items is None:
        return None
    labels: list[int] = []
    for item in items:
        if isinstance(item, Mapping):
            value = next(
                (item[key] for key in ("label", "grade", "target", "y") if key in item),
                None,
            )
        else:
            value = next(
                (getattr(item, key) for key in ("label", "grade", "target", "y") if hasattr(item, key)),
                None,
            )
            if value is None and isinstance(item, Sequence) and not isinstance(item, (str, bytes)):
                value = item[1] if len(item) > 1 else None
        if value is None:
            return None
        labels.append(int(value))
    return labels if len(labels) == len(dataset) else None


def build_class_weights(
    labels: Sequence[int],
    num_classes: int,
    *,
    method: str,
    beta: float = 0.999,
    cap: float | None = 10.0,
) -> torch.Tensor | None:
    """Build optional fold-level weights; ``none`` preserves population NLL."""

    if method == "none":
        return None
    values = torch.as_tensor(labels, dtype=torch.long).flatten()
    if values.numel() == 0 or bool(((values < 0) | (values >= num_classes)).any()):
        raise ValueError("training labels are empty or outside the declared class range")
    counts = torch.bincount(values, minlength=num_classes).to(torch.float64)
    if bool((counts == 0).any()):
        missing = (counts == 0).nonzero(as_tuple=False).flatten().tolist()
        raise ValueError(f"cannot class-weight a fold with missing classes: {missing}")
    if method == "inverse_frequency":
        raw = counts.reciprocal()
    elif method == "effective_num":
        if not 0.0 <= beta < 1.0:
            raise ValueError("effective_num_beta must lie in [0, 1)")
        if beta == 0.0:
            raw = torch.ones_like(counts)
        else:
            log_beta = math.log(beta)
            effective = -torch.expm1(counts * log_beta) / (1.0 - beta)
            raw = effective.reciprocal()
    else:
        raise ValueError("class_weighting must be 'none', 'inverse_frequency', or 'effective_num'")
    # Scale is immaterial under weighted-mean NLL. Normalize before applying a
    # literal hard cap; renormalizing afterwards would silently violate it.
    weights = raw / raw.mean()
    if cap is not None:
        if cap < 1.0:
            raise ValueError("class_weight_cap must be at least 1.0")
        weights = weights.clamp(max=float(cap))
    return weights.to(torch.float32)


def _confusion_metrics(
    predicted: torch.Tensor,
    labels: torch.Tensor,
    num_classes: int,
) -> dict[str, Any]:
    predicted = predicted.to(dtype=torch.long, device="cpu").flatten()
    labels = labels.to(dtype=torch.long, device="cpu").flatten()
    confusion = torch.zeros(num_classes, num_classes, dtype=torch.long)
    flat = labels * num_classes + predicted
    confusion += torch.bincount(flat, minlength=num_classes * num_classes).reshape(
        num_classes, num_classes
    )
    support = confusion.sum(dim=1)
    predicted_support = confusion.sum(dim=0)
    recalls = confusion.diag().float() / support.clamp_min(1)
    precisions = confusion.diag().float() / predicted_support.clamp_min(1)
    f1 = 2.0 * recalls * precisions / (recalls + precisions).clamp_min(1e-12)
    present = support > 0
    balanced = recalls[present].mean() if bool(present.any()) else recalls.new_zeros(())
    macro_f1 = f1[present].mean() if bool(present.any()) else f1.new_zeros(())
    return {
        "balanced_acc": 100.0 * float(balanced),
        "macro_f1": float(macro_f1),
        "confusion": confusion.tolist(),
        "per_grade_recall": recalls.tolist(),
        "per_grade_support": support.tolist(),
    }


def _quadratic_weighted_kappa(
    predicted: torch.Tensor,
    labels: torch.Tensor,
    num_classes: int,
) -> float:
    confusion = torch.zeros(num_classes, num_classes, dtype=torch.float64)
    true = labels.long().cpu()
    pred = predicted.long().cpu()
    flat = true * num_classes + pred
    confusion += torch.bincount(flat, minlength=num_classes * num_classes).reshape(
        num_classes, num_classes
    )
    total = confusion.sum()
    if total == 0:
        return 0.0
    expected = torch.outer(confusion.sum(1), confusion.sum(0)) / total
    coordinates = torch.arange(num_classes, dtype=torch.float64)
    weights = (coordinates[:, None] - coordinates[None, :]).square()
    weights /= float(max(1, (num_classes - 1) ** 2))
    denominator = (weights * expected).sum()
    if float(denominator) == 0.0:
        return 0.0
    return float(1.0 - (weights * confusion).sum() / denominator)


def _expected_calibration_error(
    class_probs: torch.Tensor,
    predicted: torch.Tensor,
    labels: torch.Tensor,
    *,
    bins: int = 15,
) -> float:
    class_probs = class_probs.float().cpu()
    predicted = predicted.long().cpu()
    labels = labels.long().cpu()
    confidence = class_probs.gather(1, predicted[:, None]).squeeze(1)
    edges = torch.linspace(0.0, 1.0, bins + 1)
    ece = torch.zeros((), dtype=torch.float64)
    for index in range(bins):
        if index == 0:
            members = (confidence >= edges[index]) & (confidence <= edges[index + 1])
        else:
            members = (confidence > edges[index]) & (confidence <= edges[index + 1])
        if not bool(members.any()):
            continue
        bin_accuracy = (predicted[members] == labels[members]).float().mean()
        bin_confidence = confidence[members].mean()
        ece += members.float().mean().double() * (bin_accuracy - bin_confidence).abs().double()
    return float(ece)


def evaluate_origin_predictions(
    class_probs: torch.Tensor,
    predicted: torch.Tensor,
    labels: torch.Tensor,
) -> dict[str, Any]:
    """Dependency-free grading metrics from an explicit decision rule."""

    if class_probs.ndim != 2:
        raise ValueError("class_probs must have shape (N, K)")
    num_classes = class_probs.shape[1]
    predicted = predicted.long().cpu().flatten()
    labels = labels.long().cpu().flatten()
    if len(predicted) != len(labels) or len(predicted) != len(class_probs):
        raise ValueError("probabilities, predictions, and labels must have equal length")
    difference = (predicted - labels).abs().float()
    metrics: dict[str, Any] = {
        "acc": 100.0 * float((predicted == labels).float().mean()),
        "mae": float(difference.mean()),
        "qwk": _quadratic_weighted_kappa(predicted, labels, num_classes),
        "ece": _expected_calibration_error(class_probs, predicted, labels),
    }
    metrics.update(_confusion_metrics(predicted, labels, num_classes))
    return metrics


def validation_selection_key(
    metrics: Mapping[str, float],
    *,
    policy: str = "acc_then_qwk",
    qwk_weight: float = 0.10,
) -> tuple[float, float, float]:
    """Prospective validation model-selection rule.

    ``acc_then_qwk`` is an exact lexicographic order: accuracy is primary, QWK
    breaks an accuracy tie, and lower validation loss breaks a second tie.
    ``acc_qwk_score`` optimizes ``accuracy_fraction + qwk_weight * QWK`` and
    then uses accuracy and QWK as deterministic tie-breakers.
    """

    accuracy = float(metrics["acc"])
    qwk = float(metrics["qwk"])
    loss = float(metrics["loss"])
    if policy == "acc_then_qwk":
        return (accuracy, qwk, -loss)
    if policy == "acc_qwk_score":
        if qwk_weight < 0:
            raise ValueError("selection_qwk_weight must be nonnegative")
        score = accuracy / 100.0 + float(qwk_weight) * qwk
        return (score, accuracy, qwk)
    raise ValueError(f"unknown checkpoint_selection {policy!r}; expected {_SELECTION_POLICIES}")


class OriginTrainer:
    """Train one ORIGIN fold with validation-only model development."""

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: DataLoader | None,
        cfg: object,
        run_dir: str | Path,
        *,
        fold: int = 0,
        split_signature: str | None = None,
        device: torch.device | str | None = None,
    ) -> None:
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.cfg = cfg
        self.fold = int(fold)
        self.split_signature = split_signature
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        if device is None:
            device = (
                "cuda" if torch.cuda.is_available() else
                "mps" if torch.backends.mps.is_available() else "cpu"
            )
        self.device = torch.device(device)
        self.num_classes = int(_cfg_get(cfg, "n_classes", _cfg_get(cfg, "num_classes", 5)))
        self.decision_rule = str(_cfg_get(cfg, "decision_rule", "posterior_median"))
        if self.decision_rule not in _DECISION_RULES:
            raise ValueError(f"decision_rule must be one of {_DECISION_RULES}")
        self.selection_policy = str(_cfg_get(cfg, "checkpoint_selection", "acc_then_qwk"))
        if self.selection_policy not in _SELECTION_POLICIES:
            raise ValueError(f"checkpoint_selection must be one of {_SELECTION_POLICIES}")
        self.selection_qwk_weight = float(_cfg_get(cfg, "selection_qwk_weight", 0.10))
        scheduler_name = str(_cfg_get(cfg, "scheduler", "plateau")).lower()
        if scheduler_name not in ("plateau", "reduce_on_plateau", "reduce_lr_on_plateau"):
            raise ValueError(
                "the prospective ORIGIN protocol uses ReduceLROnPlateau on "
                "validation loss; scheduler must be 'plateau'"
            )
        if not math.isclose(float(_cfg_get(cfg, "nll_weight", 1.0)), 1.0):
            raise ValueError(
                "ORIGIN fixes primary categorical NLL weight to 1.0; scale the "
                "auxiliary RPS/budget terms instead"
            )
        if not bool(_cfg_get(cfg, "force_decoder_fp64", True)):
            raise ValueError("ORIGIN's structural decoder must run in FP64")
        self.total_rate_cap = float(_cfg_get(cfg, "total_rate_cap", 64.0))
        if not math.isfinite(self.total_rate_cap) or self.total_rate_cap <= 0.0:
            raise ValueError("total_rate_cap must be finite and positive")
        self.early_stopping_patience = int(
            _cfg_get(cfg, "early_stopping_patience", _cfg_get(cfg, "early_stop_patience", 30))
        )
        if self.early_stopping_patience < 1:
            raise ValueError("early stopping patience must be positive")

        weighting = str(_cfg_get(cfg, "class_weighting", "none"))
        if weighting != "none" and not bool(_cfg_get(cfg, "allow_weighted_likelihood", False)):
            raise ValueError(
                "weighted likelihood changes the population posterior; set "
                "allow_weighted_likelihood=True for an explicit ablation"
            )
        labels = _labels_from_dataset(train_loader.dataset)
        if weighting != "none" and labels is None:
            raise ValueError("could not derive complete training-fold labels for class weighting")
        class_weights = build_class_weights(
            labels or [],
            self.num_classes,
            method=weighting,
            beta=float(_cfg_get(cfg, "effective_num_beta", 0.999)),
            cap=_cfg_get(cfg, "class_weight_cap", 10.0),
        )
        self.criterion = OriginLoss(
            self.num_classes,
            rps_weight=float(_cfg_get(cfg, "rps_weight", 0.25)),
            evidence_budget_weight=float(_cfg_get(cfg, "evidence_budget_weight", 0.0)),
            evidence_budget_delay_epochs=int(
                _cfg_get(cfg, "evidence_budget_delay_epochs", 0)
            ),
            class_weights=class_weights,
        )
        self.stratified_batches = bool(_cfg_get(cfg, "stratified_batches", False))
        self.population_objective_proper = (
            self.criterion.configured_objective_is_proper
            and not self.stratified_batches
        )
        if self.stratified_batches:
            logger.warning(
                "stratified oversampling changes the effective training "
                "population; evaluate or correct posterior calibration explicitly"
            )

        self.model.to(self.device)
        self.criterion.to(self.device)
        self.relation_enabled = bool(_cfg_get(cfg, "relation_enabled", False))
        self.relation_only_epochs = int(_cfg_get(cfg, "relation_only_epochs", 0))
        if self.relation_only_epochs < 0:
            raise ValueError("relation_only_epochs must be non-negative")
        self.relation_parameter_names = tuple(
            name
            for name, _ in self.model.named_parameters()
            if name.startswith(_RELATION_PARAMETER_PREFIX)
        )
        if self.relation_enabled and not self.relation_parameter_names:
            raise ValueError("relation_enabled model exposes no relation-field parameters")
        if not self.relation_enabled and self.relation_parameter_names:
            raise ValueError("non-relational ORIGIN unexpectedly exposes relation parameters")
        if self.relation_only_epochs and not self.relation_parameter_names:
            raise ValueError("relation-only training requires relation-field parameters")
        self.warm_start_provenance = load_origin_v3_relation_warm_start(
            self.model,
            cfg,
            fold=self.fold,
            split_signature=self.split_signature,
        )
        self.warm_start_metric_floor_tolerance = float(
            _cfg_get(cfg, "warm_start_metric_floor_tolerance", 1e-6)
        )
        if (
            not math.isfinite(self.warm_start_metric_floor_tolerance)
            or self.warm_start_metric_floor_tolerance < 0.0
        ):
            raise ValueError(
                "warm_start_metric_floor_tolerance must be finite and non-negative"
            )
        self.warm_start_metric_safety_floor = (
            self._build_warm_start_metric_safety_floor()
        )
        self.checkpoint_schema = (
            "origin-checkpoint-v6" if self.relation_enabled else "origin-checkpoint-v3"
        )
        self.use_amp = bool(_cfg_get(cfg, "amp", True) and self.device.type == "cuda")
        self.amp_init_scale = float(_cfg_get(cfg, "amp_init_scale", 4096.0))
        self.amp_unfreeze_scale = float(_cfg_get(cfg, "amp_unfreeze_scale", 256.0))
        self.amp_growth_interval = int(_cfg_get(cfg, "amp_growth_interval", 2000))
        if not math.isfinite(self.amp_init_scale) or self.amp_init_scale <= 0.0:
            raise ValueError("amp_init_scale must be finite and positive")
        if not math.isfinite(self.amp_unfreeze_scale) or self.amp_unfreeze_scale <= 0.0:
            raise ValueError("amp_unfreeze_scale must be finite and positive")
        if self.amp_growth_interval < 1:
            raise ValueError("amp_growth_interval must be positive")
        self.scaler = GradScaler(
            "cuda",
            enabled=self.use_amp,
            init_scale=self.amp_init_scale,
            growth_interval=self.amp_growth_interval,
        )
        self.max_consecutive_amp_skips = int(_cfg_get(cfg, "amp_max_consecutive_skips", 8))
        self.amp_total_skipped_steps = 0
        self.amp_consecutive_skipped_steps = 0
        self._amp_unfreeze_reset_done = False

        encoder = getattr(self.model, "encoder", None)
        encoder_parameters = list(encoder.parameters()) if isinstance(encoder, nn.Module) else []
        encoder_ids = {id(parameter) for parameter in encoder_parameters}
        head_parameters = [
            parameter for parameter in self.model.parameters()
            if id(parameter) not in encoder_ids
        ]
        encoder_lr = float(_cfg_get(cfg, "encoder_lr", _cfg_get(cfg, "lr", 2e-4)))
        head_lr = float(_cfg_get(cfg, "head_lr", 5e-4))
        groups: list[dict[str, Any]] = []
        if encoder_parameters:
            groups.append({"params": encoder_parameters, "lr": encoder_lr, "name": "encoder"})
        if head_parameters:
            groups.append({"params": head_parameters, "lr": head_lr, "name": "generator_head"})
        if not groups:
            raise ValueError("ORIGIN model has no trainable parameters")
        self.optimizer = torch.optim.AdamW(
            groups,
            weight_decay=float(_cfg_get(cfg, "weight_decay", 1e-5)),
        )
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            factor=float(_cfg_get(cfg, "lr_factor", 0.2)),
            patience=int(_cfg_get(cfg, "lr_patience", 10)),
            min_lr=float(_cfg_get(cfg, "lr_min", 1e-7)),
        )
        self.grad_clip_norm = float(_cfg_get(cfg, "grad_clip_norm", 5.0))
        if self.grad_clip_norm <= 0:
            raise ValueError("grad_clip_norm must be positive")
        self.encoder_freeze_epochs = int(
            _cfg_get(
                cfg,
                "freeze_encoder_epochs",
                _cfg_get(cfg, "encoder_freeze_epochs", 0),
            )
        )

        self.best_path = self.run_dir / "best.pth"
        self.last_path = self.run_dir / "last.pth"
        self.history_path = self.run_dir / "history.csv"
        self.certificate_path = self.run_dir / "validation_certificates.json"
        self.implementation_signature = origin_implementation_signature()
        self.architecture = _architecture_record(self.model)
        self.architecture_signature = _canonical_sha256(self.architecture)
        self.critical_config = _critical_config(cfg)
        self.config_signature = _canonical_sha256(self.critical_config)

        forward_signature = inspect.signature(self.model.forward)
        parameters = forward_signature.parameters
        self._forward_accepts_kwargs = any(
            parameter.kind == inspect.Parameter.VAR_KEYWORD
            for parameter in parameters.values()
        )
        self._accepts_pixel_mask = "pixel_valid_mask" in parameters or self._forward_accepts_kwargs
        self._accepts_force_fp64 = "force_decoder_fp64" in parameters or self._forward_accepts_kwargs

    def _relation_only_active(self, epoch: int) -> bool:
        return self.relation_enabled and 1 <= epoch <= self.relation_only_epochs

    def _build_warm_start_metric_safety_floor(self) -> dict[str, Any] | None:
        """Declare the prospective multi-metric floor relative to v3.

        A relation candidate must improve accuracy by more than numerical
        tolerance while preserving the clinically relevant ordinal and
        imbalance-aware metrics.  The ordinary selector remains necessary as
        well, so this floor cannot make an otherwise inferior candidate win.
        """

        if self.warm_start_provenance is None:
            return None
        source_metrics = self.warm_start_provenance.get("source_metrics", {})
        required = ("acc", "qwk", "balanced_acc", "macro_f1", "mae")
        source: dict[str, float] = {}
        for name in required:
            if name not in source_metrics:
                raise ValueError(
                    "ORIGIN-v3 hash-bound warm-start checkpoint omits "
                    f"validation metric {name!r} required by the safety floor"
                )
            value = float(source_metrics[name])
            if not math.isfinite(value):
                raise ValueError(
                    f"ORIGIN-v3 safety-floor source metric {name!r} is non-finite"
                )
            source[name] = value
        tolerance = self.warm_start_metric_floor_tolerance
        return {
            "schema": "origin-v6-v3-multimetric-safety-floor-v1",
            "source_checkpoint_sha256": self.warm_start_provenance[
                "source_checkpoint_sha256"
            ],
            "absolute_numerical_tolerance": tolerance,
            "source_metrics": source,
            "requirements": {
                "acc": {
                    "comparison": "strictly_greater_than",
                    "exclusive_threshold": source["acc"] + tolerance,
                },
                "qwk": {
                    "comparison": "greater_than_or_equal",
                    "inclusive_threshold": source["qwk"] - tolerance,
                },
                "balanced_acc": {
                    "comparison": "greater_than_or_equal",
                    "inclusive_threshold": source["balanced_acc"] - tolerance,
                },
                "macro_f1": {
                    "comparison": "greater_than_or_equal",
                    "inclusive_threshold": source["macro_f1"] - tolerance,
                },
                "mae": {
                    "comparison": "less_than_or_equal",
                    "inclusive_threshold": source["mae"] + tolerance,
                },
            },
            "ordinary_checkpoint_selector_also_required": True,
        }

    def _warm_start_candidate_floor_evaluation(
        self,
        metrics: Mapping[str, Any],
        *,
        selector_improved: bool,
    ) -> dict[str, Any] | None:
        """Evaluate and serialize one candidate against the v3 safety floor."""

        floor = self.warm_start_metric_safety_floor
        if floor is None:
            return None
        source = floor["source_metrics"]
        tolerance = float(floor["absolute_numerical_tolerance"])
        candidate: dict[str, float] = {}
        for name in ("acc", "qwk", "balanced_acc", "macro_f1", "mae"):
            if name not in metrics:
                raise ValueError(
                    f"v6 validation metrics omit safety-floor value {name!r}"
                )
            value = float(metrics[name])
            if not math.isfinite(value):
                raise ValueError(f"v6 safety-floor candidate metric {name!r} is non-finite")
            candidate[name] = value
        checks = {
            "accuracy_strict_improvement": (
                candidate["acc"] > source["acc"] + tolerance
            ),
            "qwk_non_regression": candidate["qwk"] >= source["qwk"] - tolerance,
            "balanced_acc_non_regression": (
                candidate["balanced_acc"] >= source["balanced_acc"] - tolerance
            ),
            "macro_f1_non_regression": (
                candidate["macro_f1"] >= source["macro_f1"] - tolerance
            ),
            "mae_non_regression": candidate["mae"] <= source["mae"] + tolerance,
        }
        passes_metric_floor = all(checks.values())
        return {
            "schema": "origin-v6-v3-multimetric-candidate-evaluation-v1",
            "candidate_metrics": candidate,
            "checks": checks,
            "passes_metric_floor": passes_metric_floor,
            "passes_normal_checkpoint_selector": bool(selector_improved),
            "checkpoint_eligible": bool(selector_improved and passes_metric_floor),
        }

    def _set_training_phase(self, epoch: int) -> None:
        """Apply the prospective relation-only then joint optimization policy."""

        relation_only = self._relation_only_active(epoch)
        if relation_only:
            for name, parameter in self.model.named_parameters():
                parameter.requires_grad_(name.startswith(_RELATION_PARAMETER_PREFIX))
            return
        for parameter in self.model.parameters():
            parameter.requires_grad_(True)
        self._set_encoder_trainable(epoch)

    def _set_encoder_trainable(self, epoch: int) -> None:
        encoder = getattr(self.model, "encoder", None)
        if not isinstance(encoder, nn.Module):
            return
        trainable = epoch > self.encoder_freeze_epochs
        for parameter in encoder.parameters():
            parameter.requires_grad_(trainable)

    def _reset_amp_scaler_at_encoder_unfreeze(self, epoch: int) -> bool:
        """Reset loss scaling exactly when the half-precision trunk enters backprop.

        Frozen warm-up exercises only the FP64 decoder/FP32 evidence heads, so
        its successful steps can grow a scale that has never been validated by
        ConvNeXt's FP16 backward path.  The reset is a phase transition, not an
        adaptive response to validation performance, and therefore leaves the
        optimizer, learning rates, and objective unchanged.
        """

        transition_epoch = max(
            self.encoder_freeze_epochs,
            self.relation_only_epochs,
        ) + 1
        if (
            not self.use_amp
            or transition_epoch <= 1
            or epoch != transition_epoch
            or self._amp_unfreeze_reset_done
        ):
            return False
        old_scale = float(self.scaler.get_scale())
        self.scaler = GradScaler(
            "cuda",
            enabled=True,
            init_scale=self.amp_unfreeze_scale,
            growth_interval=self.amp_growth_interval,
        )
        self.amp_consecutive_skipped_steps = 0
        self._amp_unfreeze_reset_done = True
        logger.info(
            "ORIGIN AMP phase reset at encoder unfreeze: epoch=%d old_scale=%.1f new_scale=%.1f",
            epoch,
            old_scale,
            self.amp_unfreeze_scale,
        )
        return True

    def _unpack_batch(self, batch: object) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor, Any]:
        if isinstance(batch, Mapping):
            images = next((batch[key] for key in ("image", "images", "x") if key in batch), None)
            labels = next((batch[key] for key in ("label", "labels", "target", "y") if key in batch), None)
            pixel_mask = next(
                (batch[key] for key in ("pixel_valid_mask", "pixel_mask", "valid_mask") if key in batch),
                None,
            )
            indices = next((batch[key] for key in ("index", "indices", "id") if key in batch), None)
        elif isinstance(batch, (tuple, list)):
            if len(batch) == 4:
                images, pixel_mask, labels, indices = batch
            elif len(batch) == 3:
                images, labels, indices = batch
                pixel_mask = None
            elif len(batch) == 2:
                images, labels = batch
                pixel_mask = None
                indices = None
            else:
                raise ValueError("ORIGIN batch must contain 2, 3, or 4 entries")
        else:
            raise TypeError("ORIGIN batch must be a mapping, tuple, or list")
        if not torch.is_tensor(images) or not torch.is_tensor(labels):
            raise TypeError("ORIGIN images and labels must be tensors")
        images = images.to(self.device, non_blocking=True)
        labels = labels.to(self.device, dtype=torch.long, non_blocking=True)
        if pixel_mask is not None:
            if not torch.is_tensor(pixel_mask):
                raise TypeError("pixel_valid_mask must be a tensor")
            pixel_mask = pixel_mask.to(self.device, non_blocking=True)
        return images, pixel_mask, labels, indices

    def _forward(self, images: torch.Tensor, pixel_mask: torch.Tensor | None) -> object:
        kwargs: dict[str, Any] = {}
        if pixel_mask is not None and self._accepts_pixel_mask:
            kwargs["pixel_valid_mask"] = pixel_mask
        if self._accepts_force_fp64:
            kwargs["force_decoder_fp64"] = True
        output = self.model(images, **kwargs)
        # The posterior graph is required to remain FP64 through the loss.  The
        # source rate ledger intentionally remains FP32 so its local accounting
        # has the same dtype as the evidence heads.
        for name in ("class_probs", "log_class_probs", "cumulative_probs", "expected_grade"):
            tensor = _tensor_field(output, name)
            if tensor.dtype != torch.float64:
                raise TypeError(
                    f"ORIGIN decoder field {name} is {tensor.dtype}; posterior must remain FP64"
                )
            if not bool(torch.isfinite(tensor).all()):
                raise FloatingPointError(f"ORIGIN decoder field {name} contains non-finite values")
        rates = _tensor_field(output, "total_rates")
        if rates.dtype in (torch.float16, torch.bfloat16):
            raise TypeError("ORIGIN source rate ledger must be at least FP32")
        if not bool(torch.isfinite(rates).all()):
            raise FloatingPointError("ORIGIN decoder field total_rates contains non-finite values")
        return output

    def _predictions(self, output: object) -> torch.Tensor:
        if self.decision_rule == "posterior_median":
            predicted = _tensor_field(output, "posterior_median")
        elif self.decision_rule == "class_map":
            predicted = _tensor_field(output, "class_map")
        else:
            predicted = _tensor_field(output, "expected_grade").round()
        return predicted.long().clamp(0, self.num_classes - 1)

    def _nonfinite_gradient_names(self, limit: int = 8) -> str:
        offenders: list[str] = []
        for name, parameter in self.model.named_parameters():
            if parameter.grad is not None and not bool(torch.isfinite(parameter.grad).all()):
                offenders.append(name)
                if len(offenders) == limit:
                    break
        return ", ".join(offenders) if offenders else "aggregate norm overflow"

    def _run_epoch(self, loader: DataLoader, *, train: bool, epoch: int) -> dict[str, Any]:
        self.model.train(train)
        if train and self._relation_only_active(epoch):
            # Frozen v3 modules must also be behaviorally frozen: disabling
            # gradients alone would leave stochastic depth/dropout active.
            self.model.eval()
            relation_field = getattr(
                getattr(self.model, "generator", None), "relation_field", None
            )
            if not isinstance(relation_field, nn.Module):
                raise RuntimeError("relation-only phase cannot locate relation_field")
            relation_field.train()
        # ConvNeXt contains stochastic-depth blocks. During the frozen head
        # warm-up the representation must be deterministic as well as
        # gradient-free, so keep the encoder in evaluation mode.
        encoder = getattr(self.model, "encoder", None)
        if (
            train
            and not self._relation_only_active(epoch)
            and epoch <= self.encoder_freeze_epochs
            and isinstance(encoder, nn.Module)
        ):
            encoder.eval()
        loss_sum = 0.0
        sample_count = 0
        diagnostic_sums = {"nll": 0.0, "rps": 0.0, "evidence_budget": 0.0}
        probabilities: list[torch.Tensor] = []
        predictions: list[torch.Tensor] = []
        labels_all: list[torch.Tensor] = []
        expected_all: list[torch.Tensor] = []
        total_rate_sum: torch.Tensor | None = None
        total_rate_max: torch.Tensor | None = None
        relation_log_odds_abs_sum = 0.0
        relation_log_odds_count = 0
        relation_log_odds_abs_max = 0.0
        relation_rate_delta_abs_sum = 0.0
        relation_rate_delta_count = 0
        relation_rate_delta_abs_max = 0.0
        relation_edge_abs_sum = 0.0
        relation_edge_count = 0
        relation_edge_abs_max = 0.0
        skipped_steps = 0
        context = torch.enable_grad if train else torch.no_grad
        with context():
            for batch_index, batch in enumerate(loader):
                images, pixel_mask, labels, _ = self._unpack_batch(batch)
                if train:
                    self.optimizer.zero_grad(set_to_none=True)
                with autocast(device_type="cuda", enabled=self.use_amp):
                    output = self._forward(images, pixel_mask)
                if (
                    not train
                    and epoch == 0
                    and self.warm_start_provenance is not None
                ):
                    base_rates = getattr(output, "base_total_rates", None)
                    if not torch.is_tensor(base_rates):
                        raise AssertionError(
                            "v6 warm-start baseline must expose base_total_rates"
                        )
                    no_op_error = float(
                        (base_rates.double() - _tensor_field(output, "total_rates").double())
                        .abs()
                        .max()
                        .detach()
                        .cpu()
                    )
                    if no_op_error > 2e-10:
                        raise AssertionError(
                            "zero-initialized v6 relation field changed the v3 "
                            f"transition ledger (max error {no_op_error:.3e})"
                        )
                # Structural probabilities are already FP64. Keep the scoring
                # rules outside autocast so no caller can downcast the loss.
                with autocast(device_type="cuda", enabled=False):
                    loss, diagnostics = self.criterion(output, labels, epoch=epoch)
                if not bool(torch.isfinite(loss)):
                    raise FloatingPointError(
                        f"non-finite ORIGIN loss at epoch {epoch}, batch {batch_index}"
                    )

                if train:
                    self.scaler.scale(loss).backward()
                    self.scaler.unscale_(self.optimizer)
                    try:
                        grad_norm = torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(),
                            self.grad_clip_norm,
                            error_if_nonfinite=True,
                        )
                    except RuntimeError as exc:
                        offenders = self._nonfinite_gradient_names()
                        # GradScaler skips an update only when unscale_ observed
                        # a non-finite gradient element.  A finite collection of
                        # very large gradients can instead overflow only while
                        # clip_grad_norm_ aggregates their norm; in that case
                        # GradScaler's found_inf flag remains clear and calling
                        # scaler.step would incorrectly apply the bad update.
                        aggregate_only = offenders == "aggregate norm overflow"
                        if not self.use_amp or aggregate_only:
                            raise FloatingPointError(
                                f"non-finite ORIGIN gradients at epoch {epoch}, "
                                f"batch {batch_index}: {offenders}"
                            ) from exc
                        self.scaler.step(self.optimizer)  # skipped by found_inf
                        self.scaler.update()
                        self.optimizer.zero_grad(set_to_none=True)
                        self.amp_total_skipped_steps += 1
                        self.amp_consecutive_skipped_steps += 1
                        skipped_steps += 1
                        logger.warning(
                            "ORIGIN AMP overflow: epoch=%d batch=%d offenders=%s consecutive=%d",
                            epoch,
                            batch_index,
                            offenders,
                            self.amp_consecutive_skipped_steps,
                        )
                        if self.amp_consecutive_skipped_steps >= self.max_consecutive_amp_skips:
                            raise FloatingPointError(
                                "persistent ORIGIN AMP gradient overflow; latest offenders: "
                                f"{offenders}"
                            ) from exc
                    else:
                        if not bool(torch.isfinite(grad_norm)):
                            raise FloatingPointError("ORIGIN gradient norm is non-finite")
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                        self.amp_consecutive_skipped_steps = 0

                count = int(labels.numel())
                sample_count += count
                loss_sum += float(loss.detach()) * count
                for key in diagnostic_sums:
                    diagnostic_sums[key] += float(diagnostics[key]) * count
                probabilities.append(_tensor_field(output, "class_probs").detach().float().cpu())
                predictions.append(self._predictions(output).detach().cpu())
                labels_all.append(labels.detach().cpu())
                expected_all.append(_tensor_field(output, "expected_grade").detach().float().cpu())
                rates = _tensor_field(output, "total_rates").detach().float().cpu()
                relation = getattr(output, "relation_evidence", None)
                if self.relation_enabled and relation is None:
                    raise AssertionError("relational ORIGIN emitted no relation evidence")
                if not self.relation_enabled and relation is not None:
                    raise AssertionError("v3 ORIGIN unexpectedly emitted relation evidence")
                if relation is not None:
                    relation_log_odds = relation.cumulative_log_odds.detach().abs()
                    relation_log_odds_abs_sum += float(relation_log_odds.sum().cpu())
                    relation_log_odds_count += int(relation_log_odds.numel())
                    relation_log_odds_abs_max = max(
                        relation_log_odds_abs_max,
                        float(relation_log_odds.max().cpu()),
                    )
                    base_rates = _tensor_field(output, "base_total_rates").detach()
                    rate_delta = (_tensor_field(output, "total_rates").detach() - base_rates).abs()
                    relation_rate_delta_abs_sum += float(rate_delta.sum().cpu())
                    relation_rate_delta_count += int(rate_delta.numel())
                    relation_rate_delta_abs_max = max(
                        relation_rate_delta_abs_max,
                        float(rate_delta.max().cpu()),
                    )
                    valid_edges = relation.edge_valid_mask[:, None].expand_as(
                        relation.edge_messages
                    )
                    edge_values = relation.edge_messages.detach().abs().masked_select(
                        valid_edges
                    )
                    if edge_values.numel():
                        relation_edge_abs_sum += float(edge_values.sum().cpu())
                        relation_edge_count += int(edge_values.numel())
                        relation_edge_abs_max = max(
                            relation_edge_abs_max,
                            float(edge_values.max().cpu()),
                        )
                batch_rate_sum = rates.sum(dim=0)
                batch_rate_max = rates.amax(dim=0)
                total_rate_sum = (
                    batch_rate_sum if total_rate_sum is None else total_rate_sum + batch_rate_sum
                )
                total_rate_max = (
                    batch_rate_max
                    if total_rate_max is None
                    else torch.maximum(total_rate_max, batch_rate_max)
                )
        if sample_count == 0:
            raise ValueError("ORIGIN data loader produced no samples")
        class_probs = torch.cat(probabilities)
        predicted = torch.cat(predictions)
        labels_cpu = torch.cat(labels_all)
        expected = torch.cat(expected_all)
        metrics = evaluate_origin_predictions(class_probs, predicted, labels_cpu)
        metrics.update(
            {
                "loss": loss_sum / sample_count,
                "nll": diagnostic_sums["nll"] / sample_count,
                "rps": diagnostic_sums["rps"] / sample_count,
                "evidence_budget": diagnostic_sums["evidence_budget"] / sample_count,
                "budget_active": self.criterion.budget_is_active(epoch),
                "likelihood_component_unweighted": (
                    self.criterion.likelihood_component_unweighted
                ),
                "population_objective_proper": self.population_objective_proper,
                "mean_expected_grade": float(expected.mean()),
                "amp_skipped_steps": int(skipped_steps),
                "amp_total_skipped_steps": int(self.amp_total_skipped_steps),
                "n": sample_count,
            }
        )
        if total_rate_sum is None or total_rate_max is None:
            raise RuntimeError("ORIGIN rate telemetry was not collected")
        mean_rates = total_rate_sum / sample_count
        metrics["mean_total_rate"] = float(mean_rates.mean())
        metrics["max_total_rate"] = float(total_rate_max.max())
        metrics["total_rate_cap"] = self.total_rate_cap
        metrics["max_total_rate_cap_fraction"] = (
            metrics["max_total_rate"] / self.total_rate_cap
        )
        if self.relation_enabled:
            if min(
                relation_log_odds_count,
                relation_rate_delta_count,
                relation_edge_count,
            ) <= 0:
                raise AssertionError("relational ORIGIN telemetry is empty")
            metrics.update(
                {
                    "mean_abs_relation_log_odds": (
                        relation_log_odds_abs_sum / relation_log_odds_count
                    ),
                    "max_abs_relation_log_odds": relation_log_odds_abs_max,
                    "mean_abs_relation_rate_delta": (
                        relation_rate_delta_abs_sum / relation_rate_delta_count
                    ),
                    "max_abs_relation_rate_delta": relation_rate_delta_abs_max,
                    "mean_abs_relation_edge_message": (
                        relation_edge_abs_sum / relation_edge_count
                    ),
                    "max_abs_relation_edge_message": relation_edge_abs_max,
                }
            )
        for boundary in range(int(mean_rates.numel())):
            metrics[f"mean_total_rate_boundary_{boundary}"] = float(mean_rates[boundary])
            metrics[f"max_total_rate_boundary_{boundary}"] = float(total_rate_max[boundary])
        warning_threshold = _RATE_CAP_WARNING_FRACTION * self.total_rate_cap
        if metrics["max_total_rate"] > warning_threshold:
            logger.warning(
                "ORIGIN rates are approaching the architectural cap: "
                "%s max_total_rate=%.4f cap=%.4f utilization=%.1f%% "
                "boundary_maxima=%s",
                "train" if train else "validation",
                metrics["max_total_rate"],
                self.total_rate_cap,
                100.0 * metrics["max_total_rate_cap_fraction"],
                [float(value) for value in total_rate_max],
            )
        return metrics

    @staticmethod
    def _history_row(
        epoch: int,
        train_metrics: Mapping[str, Any],
        val_metrics: Mapping[str, Any],
        optimizer: torch.optim.Optimizer,
    ) -> dict[str, Any]:
        row: dict[str, Any] = {"epoch": epoch}
        for group in optimizer.param_groups:
            row[f"{group.get('name', 'group')}_lr"] = group["lr"]
        for prefix, metrics in (("train", train_metrics), ("val", val_metrics)):
            for key, value in metrics.items():
                if isinstance(value, (bool, int, float)):
                    row[f"{prefix}_{key}"] = value
        return row

    def _append_history(self, row: Mapping[str, Any]) -> None:
        exists = self.history_path.exists()
        with self.history_path.open("a", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(row))
            if not exists:
                writer.writeheader()
            writer.writerow(dict(row))

    def _reconcile_history(self, completed_epoch: int) -> None:
        if not self.history_path.exists():
            return
        with self.history_path.open(newline="") as stream:
            reader = csv.DictReader(stream)
            fields = reader.fieldnames
            if not fields or "epoch" not in fields:
                raise ValueError("ORIGIN history.csv is malformed")
            rows = {
                int(row["epoch"]): row
                for row in reader
                if int(row["epoch"]) <= completed_epoch
            }
        temporary = self.history_path.with_name(f".{self.history_path.name}.{os.getpid()}.tmp")
        try:
            with temporary.open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=fields)
                writer.writeheader()
                for epoch in sorted(rows):
                    writer.writerow(rows[epoch])
            os.replace(temporary, self.history_path)
        finally:
            if temporary.exists():
                temporary.unlink()

    def _checkpoint_payload(
        self,
        epoch: int,
        metrics: Mapping[str, Any],
        best_key: tuple[float, float, float],
        best_epoch: int,
        bad_epochs: int,
        candidate_floor_evaluation: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        sampler = getattr(self.train_loader, "batch_sampler", None)
        return {
            "schema": self.checkpoint_schema,
            "epoch": int(epoch),
            "fold": self.fold,
            "split_signature": self.split_signature,
            "implementation_signature": self.implementation_signature,
            "architecture": self.architecture,
            "architecture_signature": self.architecture_signature,
            "config": _config_dict(self.cfg),
            "critical_config": self.critical_config,
            "config_signature": self.config_signature,
            "model_state": self.model.state_dict(),
            "criterion_state": self.criterion.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "scheduler_state": self.scheduler.state_dict(),
            "scaler_state": self.scaler.state_dict(),
            "metrics": dict(metrics),
            "selection_policy": self.selection_policy,
            "selection_qwk_weight": self.selection_qwk_weight,
            "best_selection_key": tuple(float(value) for value in best_key),
            "best_epoch": int(best_epoch),
            "early_stopping_bad_epochs": int(bad_epochs),
            "amp_total_skipped_steps": int(self.amp_total_skipped_steps),
            "train_batch_sampler_epoch": getattr(sampler, "_epoch", None),
            "rng_state": {
                "python": random.getstate(),
                "numpy": np.random.get_state(),
                "torch": torch.get_rng_state(),
                "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            },
            "likelihood_component_unweighted": (
                self.criterion.likelihood_component_unweighted
            ),
            "population_objective_proper": self.population_objective_proper,
            "warm_start_provenance": self.warm_start_provenance,
            "warm_start_metric_safety_floor": self.warm_start_metric_safety_floor,
            "candidate_metric_safety_floor_evaluation": (
                None
                if candidate_floor_evaluation is None
                else dict(candidate_floor_evaluation)
            ),
            "training_phase": (
                "hash_bound_v3_floor"
                if epoch == 0 and self.warm_start_provenance is not None
                else (
                    "relation_only"
                    if self._relation_only_active(epoch)
                    else "joint"
                )
            ),
        }

    def _save_checkpoint(
        self,
        path: Path,
        *,
        epoch: int,
        metrics: Mapping[str, Any],
        best_key: tuple[float, float, float],
        best_epoch: int,
        bad_epochs: int,
        candidate_floor_evaluation: Mapping[str, Any] | None = None,
    ) -> None:
        temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        try:
            torch.save(
                self._checkpoint_payload(
                    epoch,
                    metrics,
                    best_key,
                    best_epoch,
                    bad_epochs,
                    candidate_floor_evaluation,
                ),
                temporary,
            )
            os.replace(temporary, path)
        finally:
            if temporary.exists():
                temporary.unlink()

    def _validate_resume(self, state: Mapping[str, Any]) -> None:
        if state.get("schema") != self.checkpoint_schema:
            raise ValueError(
                f"ORIGIN resume requires {self.checkpoint_schema!r}; "
                "start a fresh architecture-specific run directory"
            )
        checks = {
            "implementation_signature": self.implementation_signature,
            "architecture_signature": self.architecture_signature,
            "config_signature": self.config_signature,
        }
        for key, expected in checks.items():
            if state.get(key) != expected:
                raise ValueError(
                    f"ORIGIN resume {key.replace('_', ' ')} mismatch; start a new run directory"
                )
        if int(state.get("fold", -1)) != self.fold:
            raise ValueError("ORIGIN resume fold mismatch")
        if self.split_signature is not None and state.get("split_signature") != self.split_signature:
            raise ValueError("ORIGIN resume split signature mismatch")
        if state.get("selection_policy") != self.selection_policy:
            raise ValueError("ORIGIN resume model-selection policy mismatch")
        saved_warm_start = state.get("warm_start_provenance")
        current_warm_start = self.warm_start_provenance
        if isinstance(saved_warm_start, Mapping) and isinstance(current_warm_start, Mapping):
            saved_warm_start = dict(saved_warm_start)
            current_warm_start = dict(current_warm_start)
            saved_warm_start.pop("source_checkpoint", None)
            current_warm_start.pop("source_checkpoint", None)
        if saved_warm_start != current_warm_start:
            raise ValueError("ORIGIN resume warm-start provenance mismatch")

    @staticmethod
    def _restore_rng(state: Mapping[str, Any]) -> None:
        rng = state.get("rng_state")
        if not isinstance(rng, Mapping):
            raise ValueError("ORIGIN resume checkpoint has no RNG state")
        random.setstate(rng["python"])
        np.random.set_state(rng["numpy"])
        torch.set_rng_state(rng["torch"].cpu())
        cuda = rng.get("cuda")
        if torch.cuda.is_available() and cuda is not None:
            for device_index, value in enumerate(cuda[: torch.cuda.device_count()]):
                torch.cuda.set_rng_state(value.cpu(), device=device_index)

    def _restore_training_state(self, state: Mapping[str, Any]) -> None:
        self.model.load_state_dict(state["model_state"], strict=True)
        self.criterion.load_state_dict(state["criterion_state"], strict=True)
        self.optimizer.load_state_dict(state["optimizer_state"])
        self.scheduler.load_state_dict(state["scheduler_state"])
        if "scaler_state" in state:
            self.scaler.load_state_dict(state["scaler_state"])
        self.amp_total_skipped_steps = int(state.get("amp_total_skipped_steps", 0))
        self.amp_consecutive_skipped_steps = 0
        sampler_epoch = state.get("train_batch_sampler_epoch")
        sampler = getattr(self.train_loader, "batch_sampler", None)
        if sampler_epoch is not None and sampler is not None:
            sampler._epoch = int(sampler_epoch)
        self._restore_rng(state)

    def _restore_best(self) -> dict[str, Any]:
        state = torch.load(self.best_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(state["model_state"], strict=True)
        self.criterion.load_state_dict(state["criterion_state"], strict=True)
        return state

    @staticmethod
    def _index_value(indices: Any, index: int) -> Any:
        if indices is None:
            return index
        if torch.is_tensor(indices):
            value = indices[index].detach().cpu()
            return value.item() if value.ndim == 0 else value.tolist()
        if isinstance(indices, Sequence) and not isinstance(indices, (str, bytes)):
            value = indices[index]
            return value.item() if isinstance(value, np.generic) else value
        return indices

    @staticmethod
    def _file_sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def _write_validation_certificates(self, checkpoint_epoch: int) -> dict[str, Any]:
        replay = getattr(self.model, "replay_without", None)
        if not callable(replay):
            raise TypeError("ORIGIN model must expose replay_without for exact certificates")
        requested = max(1, int(_cfg_get(self.cfg, "certificate_samples", 4)))
        self.model.eval()
        batch = next(iter(self.val_loader))
        images, pixel_mask, labels, indices = self._unpack_batch(batch)
        with torch.no_grad():
            with autocast(device_type="cuda", enabled=self.use_amp):
                baseline = self._forward(images, pixel_mask)
            rate_maps = _mapping_field(baseline, "local_rate_maps")
            valid_masks = _mapping_field(baseline, "valid_masks")
            batch_size = int(labels.numel())
            count = min(requested, batch_size)
            removal_masks: dict[str, torch.Tensor] = {}
            for scale, rates in rate_maps.items():
                if rates.ndim < 3 or rates.shape[0] != batch_size:
                    raise ValueError(
                        f"local rate map {scale!r} must have shape (N, spatial..., K-1)"
                    )
                removal_masks[scale] = torch.zeros(
                    rates.shape[:-1], dtype=torch.bool, device=rates.device
                )
            chosen: list[tuple[str, tuple[int, ...], float]] = []
            for sample in range(count):
                best: tuple[float, str, tuple[int, ...]] | None = None
                for scale, rates in rate_maps.items():
                    scores = rates[sample].sum(dim=-1)
                    valid = valid_masks.get(scale)
                    if valid is not None:
                        if valid.shape != rates.shape[:-1]:
                            raise ValueError(f"valid mask shape mismatch at scale {scale!r}")
                        scores = scores.masked_fill(~valid[sample].bool(), -torch.inf)
                    flat_index = int(scores.reshape(-1).argmax())
                    value = float(scores.reshape(-1)[flat_index].detach().cpu())
                    spatial = tuple(int(v) for v in np.unravel_index(flat_index, scores.shape))
                    candidate = (value, str(scale), spatial)
                    if best is None or candidate > best:
                        best = candidate
                if best is None or not math.isfinite(best[0]):
                    raise ValueError("validation sample has no valid ORIGIN evidence cell")
                value, scale, spatial = best
                removal_masks[scale][(sample, *spatial)] = True
                chosen.append((scale, spatial, value))

            intervention = replay(
                baseline,
                removal_masks,
                force_decoder_fp64=True,
            )
            replayed = getattr(intervention, "output", intervention)
            removed_rates = getattr(intervention, "removed_rates", None)
            if not torch.is_tensor(removed_rates):
                raise TypeError("ORIGIN replay result must expose removed_rates")
            baseline_rates = _tensor_field(baseline, "total_rates")
            replayed_rates = _tensor_field(replayed, "total_rates")
            baseline_source_rates = getattr(baseline, "base_total_rates", baseline_rates)
            replayed_source_rates = getattr(replayed, "base_total_rates", replayed_rates)
            if not torch.is_tensor(baseline_source_rates) or not torch.is_tensor(
                replayed_source_rates
            ):
                raise TypeError("ORIGIN replay source ledgers must be tensors")
            expected_rates = baseline_source_rates - removed_rates
            replay_error = float(
                (expected_rates - replayed_source_rates).abs().max().cpu()
            )
            if not baseline_rates.is_floating_point():
                raise TypeError("ORIGIN replay ledger must be floating point")
            configured_tolerance = float(
                _cfg_get(self.cfg, "certificate_replay_tolerance", 2e-5)
            )
            if not math.isfinite(configured_tolerance) or configured_tolerance < 0.0:
                raise ValueError(
                    "certificate_replay_tolerance must be finite and non-negative"
                )
            rate_scale = max(
                1.0,
                float(baseline_rates.detach().abs().max().cpu()),
                float(replayed_rates.detach().abs().max().cpu()),
                float(baseline_source_rates.detach().abs().max().cpu()),
                float(replayed_source_rates.detach().abs().max().cpu()),
            )
            # Local ledger maps are normally FP32 even though their conserved
            # total is accumulated in FP64. Baseline-minus-removed and a fresh
            # survivor reduction are mathematically identical but can differ
            # by several source-ledger ulps after reductions over thousands of
            # cells. Never relax the posterior calculation itself.
            ledger_dtype = next(iter(rate_maps.values())).dtype
            if not ledger_dtype.is_floating_point:
                raise TypeError("ORIGIN local replay ledger must be floating point")
            roundoff_tolerance = (
                16.0 * torch.finfo(ledger_dtype).eps * rate_scale
            )
            tolerance = max(configured_tolerance, roundoff_tolerance)
            if replay_error > tolerance:
                raise AssertionError(
                    f"ORIGIN exact replay failed: rate error {replay_error:.3g} > {tolerance:.3g}"
                )

            relation_merge_error = 0.0
            relation_trace = getattr(replayed, "relation_evidence", None)
            if relation_trace is not None:
                reconstructed_final = bounded_rate_log_odds_merge(
                    replayed_source_rates,
                    relation_trace.cumulative_log_odds,
                    total_rate_cap=float(getattr(replayed, "total_rate_cap")),
                )
                relation_merge_error = float(
                    (reconstructed_final - replayed_rates).abs().max().cpu()
                )
                if relation_merge_error > 2e-10:
                    raise AssertionError(
                        "ORIGIN relational replay does not equal the stored "
                        "bounded log-odds merge"
                    )

            # Independently certify the stronger elementwise statement: every
            # stored ledger entry is partitioned exactly into a kept or removed
            # entry. This check has zero arithmetic tolerance because each side
            # is either x + 0 or 0 + x.
            replayed_maps = _mapping_field(replayed, "local_rate_maps")
            if set(replayed_maps) != set(rate_maps):
                raise AssertionError("ORIGIN replay changed the ledger scale set")
            ledger_partition_error = 0.0
            for scale, original_map in rate_maps.items():
                mask = removal_masks[scale]
                while mask.ndim < original_map.ndim:
                    mask = mask.unsqueeze(-1)
                removed_map = torch.where(
                    mask,
                    original_map,
                    torch.zeros_like(original_map),
                )
                partition_error = (
                    original_map - (replayed_maps[scale] + removed_map)
                ).abs().max()
                ledger_partition_error = max(
                    ledger_partition_error,
                    float(partition_error.detach().cpu()),
                )
            if ledger_partition_error != 0.0:
                raise AssertionError(
                    "ORIGIN replay did not exactly partition the stored local ledger"
                )

            base_probs = _tensor_field(baseline, "class_probs").detach().cpu()
            new_probs = _tensor_field(replayed, "class_probs").detach().cpu()
            base_cumulative = _tensor_field(baseline, "cumulative_probs").detach().cpu()
            new_cumulative = _tensor_field(replayed, "cumulative_probs").detach().cpu()
            base_expected = _tensor_field(baseline, "expected_grade").detach().cpu()
            new_expected = _tensor_field(replayed, "expected_grade").detach().cpu()
            base_predictions = self._predictions(baseline).detach().cpu()
            new_predictions = self._predictions(replayed).detach().cpu()
            removed_cpu = removed_rates.detach().cpu()
            metadata = getattr(baseline, "metadata", {})
            certificates = []
            for sample, (scale, spatial, value) in enumerate(chosen):
                scale_metadata = metadata.get(scale) if isinstance(metadata, Mapping) else None
                stride = getattr(scale_metadata, "output_stride", None)
                receptive_field = getattr(scale_metadata, "receptive_field", None)
                center_offset = getattr(scale_metadata, "center_offset", None)
                center_yx = None
                if (
                    len(spatial) == 2
                    and stride is not None
                    and center_offset is not None
                ):
                    center_yx = [
                        float(center_offset + spatial[0] * stride),
                        float(center_offset + spatial[1] * stride),
                    ]
                certificates.append(
                    {
                        "sample_id": self._index_value(indices, sample),
                        "label": int(labels[sample].detach().cpu()),
                        "removed_scale": scale,
                        "removed_spatial_index": list(spatial),
                        "input_center_yx": center_yx,
                        "output_stride_pixels": (
                            None if stride is None else int(stride)
                        ),
                        "receptive_field_pixels": (
                            None if receptive_field is None else int(receptive_field)
                        ),
                        "removed_local_rate_sum": value,
                        "removed_boundary_rates": removed_cpu[sample].tolist(),
                        "baseline_prediction": int(base_predictions[sample]),
                        "replayed_prediction": int(new_predictions[sample]),
                        "baseline_class_probs": base_probs[sample].tolist(),
                        "replayed_class_probs": new_probs[sample].tolist(),
                        "cumulative_probability_delta": (
                            base_cumulative[sample] - new_cumulative[sample]
                        ).tolist(),
                        "expected_grade_delta": float(
                            base_expected[sample] - new_expected[sample]
                        ),
                    }
                )

            relation_certificates: list[dict[str, Any]] = []
            relation_base_ledger_error = 0.0
            relation_partition_error = 0.0
            relation_reaggregation_error = 0.0
            relation_final_merge_error = 0.0
            relation = getattr(baseline, "relation_evidence", None)
            if relation is not None:
                replay_relations = getattr(self.model, "replay_without_relations", None)
                if not callable(replay_relations):
                    raise TypeError(
                        "relation-enabled ORIGIN must expose replay_without_relations"
                    )
                ranked = relation.edge_messages.abs().masked_fill(
                    ~relation.edge_valid_mask[:, None], -torch.inf
                ).flatten(1)
                if bool((torch.isfinite(ranked[:count]).sum(1) == 0).any()):
                    raise ValueError("validation sample has no valid relation edge")
                selected = ranked[:count].argmax(1)
                edge_removals = torch.zeros_like(
                    relation.edge_messages, dtype=torch.bool
                ).flatten(1)
                edge_removals[
                    torch.arange(count, device=edge_removals.device), selected
                ] = True
                edge_removals = edge_removals.reshape_as(relation.edge_messages)
                relation_intervention = replay_relations(
                    baseline,
                    edge_removals,
                    force_decoder_fp64=True,
                )
                relation_replayed = relation_intervention.output
                replay_trace = relation_replayed.relation_evidence
                if replay_trace is None:
                    raise AssertionError("relation replay lost its evidence trace")
                relation_base_errors = (
                    _tensor_field(baseline, "base_total_rates")
                    - _tensor_field(relation_replayed, "base_total_rates")
                ).abs().amax(dim=1)
                relation_base_ledger_error = float(
                    relation_base_errors.max().cpu()
                )
                relation_partition_errors = (
                    relation.edge_messages
                    - (
                        replay_trace.edge_messages
                        + relation_intervention.removed_edge_messages
                    )
                ).abs().flatten(1).amax(dim=1)
                relation_partition_error = float(
                    relation_partition_errors.max().cpu()
                )
                _, recomputed_incremental, recomputed_cumulative = (
                    aggregate_ordinal_pair_messages(
                        replay_trace.edge_messages,
                        replay_trace.edge_valid_mask,
                        replay_trace.region_valid_mask,
                        delta_cap=replay_trace.delta_cap,
                    )
                )
                relation_reaggregation_errors = torch.maximum(
                    (recomputed_incremental - replay_trace.incremental_log_odds)
                    .abs()
                    .amax(dim=1),
                    (recomputed_cumulative - replay_trace.cumulative_log_odds)
                    .abs()
                    .amax(dim=1),
                )
                relation_reaggregation_error = float(
                    relation_reaggregation_errors.max().cpu()
                )
                recomputed_rates = bounded_rate_log_odds_merge(
                    _tensor_field(relation_replayed, "base_total_rates"),
                    recomputed_cumulative,
                    total_rate_cap=float(relation_replayed.total_rate_cap),
                )
                relation_final_merge_errors = (
                    recomputed_rates - _tensor_field(relation_replayed, "total_rates")
                ).abs().amax(dim=1)
                relation_final_merge_error = float(
                    relation_final_merge_errors.max().cpu()
                )
                if max(
                    relation_base_ledger_error,
                    relation_partition_error,
                    relation_reaggregation_error,
                    relation_final_merge_error,
                ) > 2e-10:
                    raise AssertionError(
                        "ORIGIN relation certificate failed exact replay invariants"
                    )

                regions = relation.edge_messages.shape[-1]
                centers = relation.region_centers_yx.detach().cpu()
                baseline_relation_probs = _tensor_field(
                    baseline, "class_probs"
                ).detach().cpu()
                replay_relation_probs = _tensor_field(
                    relation_replayed, "class_probs"
                ).detach().cpu()
                baseline_relation_expected = _tensor_field(
                    baseline, "expected_grade"
                ).detach().cpu()
                replay_relation_expected = _tensor_field(
                    relation_replayed, "expected_grade"
                ).detach().cpu()
                baseline_relation_predictions = self._predictions(baseline).detach().cpu()
                replay_relation_predictions = self._predictions(
                    relation_replayed
                ).detach().cpu()
                flat_messages = relation.edge_messages.detach().cpu().flatten(1)
                for sample in range(count):
                    flat_index = int(selected[sample])
                    boundary = flat_index // (regions * regions)
                    endpoint = flat_index % (regions * regions)
                    target = endpoint // regions
                    source = endpoint % regions
                    relation_certificates.append(
                        {
                            "sample_id": self._index_value(indices, sample),
                            "label": int(labels[sample].detach().cpu()),
                            "selection": "largest_absolute_stored_pair_message",
                            "boundary": boundary,
                            "target_region": target,
                            "source_region": source,
                            "target_center_yx": centers[target].tolist(),
                            "source_center_yx": centers[source].tolist(),
                            "region_receptive_field_pixels": int(
                                relation.region_receptive_field
                            ),
                            "region_output_stride_pixels": int(
                                relation.region_output_stride
                            ),
                            "signed_pair_message": float(
                                flat_messages[sample, flat_index]
                            ),
                            "baseline_prediction": int(
                                baseline_relation_predictions[sample]
                            ),
                            "replayed_prediction": int(
                                replay_relation_predictions[sample]
                            ),
                            "prediction_delta": int(
                                baseline_relation_predictions[sample]
                                - replay_relation_predictions[sample]
                            ),
                            "baseline_class_probs": baseline_relation_probs[
                                sample
                            ].tolist(),
                            "replayed_class_probs": replay_relation_probs[
                                sample
                            ].tolist(),
                            "class_probability_delta": (
                                baseline_relation_probs[sample]
                                - replay_relation_probs[sample]
                            ).tolist(),
                            "expected_grade_delta": float(
                                baseline_relation_expected[sample]
                                - replay_relation_expected[sample]
                            ),
                            "exact_base_ledger_invariance_error": float(
                                relation_base_errors[sample].detach().cpu()
                            ),
                            "exact_edge_partition_error": float(
                                relation_partition_errors[sample].detach().cpu()
                            ),
                            "exact_reaggregation_error": float(
                                relation_reaggregation_errors[sample].detach().cpu()
                            ),
                            "exact_final_merge_error": float(
                                relation_final_merge_errors[sample].detach().cpu()
                            ),
                        }
                    )
        payload: dict[str, Any] = {
            "schema": (
                "origin-exact-validation-certificates-v6"
                if self.relation_enabled
                else "origin-exact-validation-certificates-v3"
            ),
            "scope": "inner_validation_only",
            "fold": self.fold,
            "split_signature": self.split_signature,
            "checkpoint_epoch": int(checkpoint_epoch),
            "checkpoint_sha256": self._file_sha256(self.best_path),
            "implementation_signature": self.implementation_signature,
            "architecture_signature": self.architecture_signature,
            "decision_rule": self.decision_rule,
            "selection_rule": "largest_single_generator_contribution_across_scales",
            "interpretation_scope": (
                "exact_stored_ledger_intervention_not_causal_pixel_deletion"
            ),
            "exact_replay_max_abs_source_rate_error": replay_error,
            # Backward-compatible key: in v3 source and final rates coincide;
            # in v6 this explicitly refers to the pre-relation source ledger.
            "exact_replay_max_abs_total_rate_error": replay_error,
            "exact_replay_total_rate_tolerance": tolerance,
            "exact_relation_merge_max_abs_rate_error": relation_merge_error,
            "exact_ledger_partition_max_abs_error": ledger_partition_error,
            "relation_certificates": relation_certificates,
            "exact_relation_base_ledger_invariance_error": (
                relation_base_ledger_error
            ),
            "exact_relation_edge_partition_error": relation_partition_error,
            "exact_relation_reaggregation_error": relation_reaggregation_error,
            "exact_relation_final_merge_error": relation_final_merge_error,
            "certificates": certificates,
        }
        payload["content_checksum_sha256"] = _canonical_sha256(payload)
        temporary = self.certificate_path.with_name(
            f".{self.certificate_path.name}.{os.getpid()}.tmp"
        )
        try:
            with temporary.open("w", encoding="utf-8") as stream:
                json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False, default=_json_default)
                stream.write("\n")
            os.replace(temporary, self.certificate_path)
        finally:
            if temporary.exists():
                temporary.unlink()
        return payload

    def fit(self, *, evaluate_test: bool = False) -> dict[str, Any]:
        """Fit the fold; outer-test evaluation is opt-in and never implicit."""

        epochs = int(_cfg_get(self.cfg, "epochs", 40))
        resume = bool(_cfg_get(self.cfg, "resume", False))
        best_key = (-math.inf, -math.inf, -math.inf)
        best_epoch = 0
        bad_epochs = 0
        start_epoch = 1
        if resume and not self.last_path.exists():
            raise FileNotFoundError(
                f"ORIGIN resume requested but checkpoint does not exist: {self.last_path}"
            )
        if not resume:
            existing = [
                path
                for path in (self.best_path, self.last_path, self.history_path)
                if path.exists()
            ]
            if existing:
                names = ", ".join(str(path) for path in existing)
                raise FileExistsError(
                    "fresh ORIGIN run would overwrite existing training artifacts; "
                    f"use resume or a new run directory: {names}"
                )
        if resume:
            state = torch.load(self.last_path, map_location=self.device, weights_only=False)
            self._validate_resume(state)
            self._restore_training_state(state)
            best_key = tuple(float(value) for value in state["best_selection_key"])
            best_epoch = int(state["best_epoch"])
            bad_epochs = int(state.get("early_stopping_bad_epochs", 0))
            completed = int(state["epoch"])
            self._reconcile_history(completed)
            start_epoch = completed + 1
            logger.info("resumed ORIGIN fold=%d from epoch=%d", self.fold, completed)
        elif self.warm_start_provenance is not None:
            # The inherited v3 predictor is a prospective, eligible baseline:
            # v6 cannot silently return a worse checkpoint merely because its
            # newly added zero-initialized relation branch was optimized.
            baseline_metrics = self._run_epoch(
                self.val_loader,
                train=False,
                epoch=0,
            )
            source_metrics = self.warm_start_provenance.get("source_metrics", {})
            for name in (
                "acc",
                "mae",
                "qwk",
                "balanced_acc",
                "macro_f1",
                "confusion",
            ):
                if name not in source_metrics:
                    raise ValueError(
                        f"ORIGIN-v3 warm-start checkpoint omits validation metric {name!r}"
                    )
                if not np.allclose(
                    np.asarray(baseline_metrics[name]),
                    np.asarray(source_metrics[name]),
                    rtol=0.0,
                    atol=self.warm_start_metric_floor_tolerance,
                ):
                    raise AssertionError(
                        f"v6 epoch-0 validation {name} does not reproduce the "
                        "hash-bound v3 checkpoint"
                    )
            best_key = validation_selection_key(
                baseline_metrics,
                policy=self.selection_policy,
                qwk_weight=self.selection_qwk_weight,
            )
            best_epoch = 0
            baseline_floor_evaluation = {
                "schema": "origin-v6-v3-multimetric-candidate-evaluation-v1",
                "status": "hash_bound_v3_baseline_floor",
                "candidate_metrics": {
                    name: float(baseline_metrics[name])
                    for name in ("acc", "qwk", "balanced_acc", "macro_f1", "mae")
                },
                "checks": {
                    "verified_epoch0_source_reproduction": True,
                },
                "passes_metric_floor": True,
                "passes_normal_checkpoint_selector": True,
                "checkpoint_eligible": True,
            }
            self._save_checkpoint(
                self.best_path,
                epoch=0,
                metrics=baseline_metrics,
                best_key=best_key,
                best_epoch=0,
                bad_epochs=0,
                candidate_floor_evaluation=baseline_floor_evaluation,
            )
            logger.info(
                "hash-bound ORIGIN-v3 warm start is eligible at epoch=000: "
                "val_loss=%.5f val_acc=%.2f val_qwk=%.4f val_mae=%.4f "
                "bal_acc=%.2f macro_f1=%.4f safety_floor=%s",
                baseline_metrics["loss"],
                baseline_metrics["acc"],
                baseline_metrics["qwk"],
                baseline_metrics["mae"],
                baseline_metrics["balanced_acc"],
                baseline_metrics["macro_f1"],
                self.warm_start_metric_safety_floor,
            )

        for epoch in range(start_epoch, epochs + 1):
            started = time.time()
            self._reset_amp_scaler_at_encoder_unfreeze(epoch)
            self._set_training_phase(epoch)
            train_metrics = self._run_epoch(self.train_loader, train=True, epoch=epoch)
            val_metrics = self._run_epoch(self.val_loader, train=False, epoch=epoch)
            self.scheduler.step(float(val_metrics["loss"]))
            key = validation_selection_key(
                val_metrics,
                policy=self.selection_policy,
                qwk_weight=self.selection_qwk_weight,
            )
            selector_improved = key > best_key
            candidate_floor_evaluation = (
                self._warm_start_candidate_floor_evaluation(
                    val_metrics,
                    selector_improved=selector_improved,
                )
            )
            improved = selector_improved and (
                candidate_floor_evaluation is None
                or bool(candidate_floor_evaluation["passes_metric_floor"])
            )
            if improved:
                best_key = key
                best_epoch = epoch
                bad_epochs = 0
            else:
                bad_epochs += 1
            history_row = self._history_row(
                epoch, train_metrics, val_metrics, self.optimizer
            )
            if candidate_floor_evaluation is not None:
                history_row.update(
                    {
                        "warm_start_floor_selector_improved": bool(
                            candidate_floor_evaluation[
                                "passes_normal_checkpoint_selector"
                            ]
                        ),
                        "warm_start_floor_metric_passed": bool(
                            candidate_floor_evaluation["passes_metric_floor"]
                        ),
                        "warm_start_floor_checkpoint_eligible": bool(
                            candidate_floor_evaluation["checkpoint_eligible"]
                        ),
                        **{
                            f"warm_start_floor_{name}": bool(value)
                            for name, value in candidate_floor_evaluation[
                                "checks"
                            ].items()
                        },
                    }
                )
            self._append_history(history_row)
            if improved:
                self._save_checkpoint(
                    self.best_path,
                    epoch=epoch,
                    metrics=val_metrics,
                    best_key=best_key,
                    best_epoch=best_epoch,
                    bad_epochs=bad_epochs,
                    candidate_floor_evaluation=candidate_floor_evaluation,
                )
            self._save_checkpoint(
                self.last_path,
                epoch=epoch,
                metrics=val_metrics,
                best_key=best_key,
                best_epoch=best_epoch,
                bad_epochs=bad_epochs,
                candidate_floor_evaluation=candidate_floor_evaluation,
            )
            logger.info(
                "ORIGIN epoch=%03d train_loss=%.5f val_loss=%.5f val_acc=%.2f "
                "val_qwk=%.4f val_mae=%.4f bal_acc=%.2f macro_f1=%.4f "
                "rate_mean=%.4f rate_max=%.4f budget=%s time=%.1fs",
                epoch,
                train_metrics["loss"],
                val_metrics["loss"],
                val_metrics["acc"],
                val_metrics["qwk"],
                val_metrics["mae"],
                val_metrics["balanced_acc"],
                val_metrics["macro_f1"],
                val_metrics["mean_total_rate"],
                val_metrics["max_total_rate"],
                train_metrics["budget_active"],
                time.time() - started,
            )
            if self.relation_enabled:
                logger.info(
                    "ORIGIN relation epoch=%03d phase=%s "
                    "|D|_mean=%.6f |D|_max=%.6f "
                    "|rate_delta|_mean=%.6f |rate_delta|_max=%.6f "
                    "|edge_message|_mean=%.6f |edge_message|_max=%.6f",
                    epoch,
                    "relation_only" if self._relation_only_active(epoch) else "joint",
                    val_metrics["mean_abs_relation_log_odds"],
                    val_metrics["max_abs_relation_log_odds"],
                    val_metrics["mean_abs_relation_rate_delta"],
                    val_metrics["max_abs_relation_rate_delta"],
                    val_metrics["mean_abs_relation_edge_message"],
                    val_metrics["max_abs_relation_edge_message"],
                )
            if candidate_floor_evaluation is not None:
                logger.info(
                    "ORIGIN v3 multi-metric safety floor epoch=%03d "
                    "selector_improved=%s metric_floor_passed=%s "
                    "checkpoint_eligible=%s checks=%s thresholds=%s",
                    epoch,
                    candidate_floor_evaluation[
                        "passes_normal_checkpoint_selector"
                    ],
                    candidate_floor_evaluation["passes_metric_floor"],
                    candidate_floor_evaluation["checkpoint_eligible"],
                    candidate_floor_evaluation["checks"],
                    self.warm_start_metric_safety_floor["requirements"],
                )
            if bad_epochs >= self.early_stopping_patience:
                logger.info("ORIGIN early stopping at epoch %d", epoch)
                break

        if not self.best_path.exists():
            raise RuntimeError("ORIGIN training produced no best checkpoint")
        best = self._restore_best()
        best_validation = dict(best["metrics"])
        certificates = self._write_validation_certificates(int(best["epoch"]))
        result: dict[str, Any] = {
            "best_epoch": int(best["epoch"]),
            "best_validation": best_validation,
            "best_validation_metrics": best_validation,
            "test_evaluated": False,
            "run_dir": str(self.run_dir),
            "validation_certificate_path": str(self.certificate_path),
            "validation_certificate_checksum": certificates["content_checksum_sha256"],
            "likelihood_component_unweighted": (
                self.criterion.likelihood_component_unweighted
            ),
            "population_objective_proper": self.population_objective_proper,
            "checkpoint_schema": self.checkpoint_schema,
            "warm_start_provenance": self.warm_start_provenance,
            "warm_start_metric_safety_floor": self.warm_start_metric_safety_floor,
            "selected_checkpoint_metric_safety_floor_evaluation": best.get(
                "candidate_metric_safety_floor_evaluation"
            ),
            "epoch0_warm_start_eligible": self.warm_start_provenance is not None,
            "relation_only_epochs": self.relation_only_epochs,
            "selected_checkpoint_training_phase": str(
                best.get("training_phase", "unknown")
            ),
            "best_checkpoint_is_hash_bound_v3_floor": bool(
                self.warm_start_provenance is not None and int(best["epoch"]) == 0
            ),
        }
        result.update(
            {
                f"best_val_{key}": value
                for key, value in best_validation.items()
                if isinstance(value, (bool, int, float))
            }
        )
        if not evaluate_test:
            logger.info(
                "outer test skipped; best validation acc=%.2f qwk=%.4f epoch=%d",
                best_validation["acc"],
                best_validation["qwk"],
                int(best["epoch"]),
            )
            return result
        if self.test_loader is None:
            raise ValueError("evaluate_test=True requires a test loader")
        test_metrics = self._run_epoch(self.test_loader, train=False, epoch=int(best["epoch"]))
        result["test_evaluated"] = True
        result["test"] = test_metrics
        result.update(
            {
                key: value
                for key, value in test_metrics.items()
                if isinstance(value, (bool, int, float))
            }
        )
        return result


__all__ = [
    "OriginTrainer",
    "build_class_weights",
    "evaluate_origin_predictions",
    "origin_implementation_signature",
    "validation_selection_key",
]
