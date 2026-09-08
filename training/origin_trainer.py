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

_RATE_STABILITY_WARNING = 80.0

_RUNTIME_CONFIG_FIELDS = {
    "epochs",             # may be extended after a preemption
    "resume",             # invocation policy, not numerical identity
    "evaluate_test",      # evaluation authorization only
    "run_dir",
    "output_dir",
}

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

        transition_epoch = self.encoder_freeze_epochs + 1
        if (
            not self.use_amp
            or self.encoder_freeze_epochs < 1
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
        # ConvNeXt contains stochastic-depth blocks. During the frozen head
        # warm-up the representation must be deterministic as well as
        # gradient-free, so keep the encoder in evaluation mode.
        encoder = getattr(self.model, "encoder", None)
        if train and epoch <= self.encoder_freeze_epochs and isinstance(encoder, nn.Module):
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
        skipped_steps = 0
        context = torch.enable_grad if train else torch.no_grad
        with context():
            for batch_index, batch in enumerate(loader):
                images, pixel_mask, labels, _ = self._unpack_batch(batch)
                if train:
                    self.optimizer.zero_grad(set_to_none=True)
                with autocast(device_type="cuda", enabled=self.use_amp):
                    output = self._forward(images, pixel_mask)
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
        for boundary in range(int(mean_rates.numel())):
            metrics[f"mean_total_rate_boundary_{boundary}"] = float(mean_rates[boundary])
            metrics[f"max_total_rate_boundary_{boundary}"] = float(total_rate_max[boundary])
        if metrics["max_total_rate"] > _RATE_STABILITY_WARNING:
            logger.warning(
                "ORIGIN rates entered the decoder saturation-risk regime: "
                "%s max_total_rate=%.4f boundary_maxima=%s",
                "train" if train else "validation",
                metrics["max_total_rate"],
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
    ) -> dict[str, Any]:
        sampler = getattr(self.train_loader, "batch_sampler", None)
        return {
            "schema": "origin-checkpoint-v2",
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
    ) -> None:
        temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        try:
            torch.save(
                self._checkpoint_payload(
                    epoch, metrics, best_key, best_epoch, bad_epochs
                ),
                temporary,
            )
            os.replace(temporary, path)
        finally:
            if temporary.exists():
                temporary.unlink()

    def _validate_resume(self, state: Mapping[str, Any]) -> None:
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
            expected_rates = baseline_rates - removed_rates
            replay_error = float(
                (expected_rates - replayed_rates).abs().max().cpu()
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
            )
            # The ledger is stored in FP32. Baseline-minus-removed and a fresh
            # survivor reduction are mathematically identical but can differ
            # by several ulps after reductions over thousands of cells. Scale
            # the audit bound with the source dtype; never relax the posterior
            # calculation itself, which remains FP64.
            roundoff_tolerance = (
                16.0 * torch.finfo(baseline_rates.dtype).eps * rate_scale
            )
            tolerance = max(configured_tolerance, roundoff_tolerance)
            if replay_error > tolerance:
                raise AssertionError(
                    f"ORIGIN exact replay failed: rate error {replay_error:.3g} > {tolerance:.3g}"
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
        payload: dict[str, Any] = {
            "schema": "origin-exact-validation-certificates-v2",
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
            "exact_replay_max_abs_total_rate_error": replay_error,
            "exact_replay_total_rate_tolerance": tolerance,
            "exact_ledger_partition_max_abs_error": ledger_partition_error,
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

        for epoch in range(start_epoch, epochs + 1):
            started = time.time()
            self._reset_amp_scaler_at_encoder_unfreeze(epoch)
            self._set_encoder_trainable(epoch)
            train_metrics = self._run_epoch(self.train_loader, train=True, epoch=epoch)
            val_metrics = self._run_epoch(self.val_loader, train=False, epoch=epoch)
            self.scheduler.step(float(val_metrics["loss"]))
            key = validation_selection_key(
                val_metrics,
                policy=self.selection_policy,
                qwk_weight=self.selection_qwk_weight,
            )
            improved = key > best_key
            if improved:
                best_key = key
                best_epoch = epoch
                bad_epochs = 0
            else:
                bad_epochs += 1
            self._append_history(
                self._history_row(epoch, train_metrics, val_metrics, self.optimizer)
            )
            if improved:
                self._save_checkpoint(
                    self.best_path,
                    epoch=epoch,
                    metrics=val_metrics,
                    best_key=best_key,
                    best_epoch=best_epoch,
                    bad_epochs=bad_epochs,
                )
            self._save_checkpoint(
                self.last_path,
                epoch=epoch,
                metrics=val_metrics,
                best_key=best_key,
                best_epoch=best_epoch,
                bad_epochs=bad_epochs,
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
