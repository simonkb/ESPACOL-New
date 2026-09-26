"""Truthful trainer adapter for the frozen ORIGIN fold-9 ablation.

Optimisation and model selection are inherited unchanged from
``OriginTrainer``.  This adapter only expands provenance to include ablation
sources and prevents globally pooled controls from receiving a fabricated
local-intervention certificate.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
import torch.nn as nn

from .origin_trainer import OriginTrainer, origin_implementation_signature


_ABLATION_IMPLEMENTATION_FILES = (
    "configs/origin_ablation_config.py",
    "models/origin_ablation.py",
    "training/origin_ablation_trainer.py",
    "train_origin_ablation.py",
)


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (torch.device, torch.dtype)):
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


def origin_ablation_implementation_signature() -> str:
    """Hash the base implementation identity plus all ablation-specific files."""

    root = Path(__file__).resolve().parents[1]
    digest = hashlib.sha256()
    digest.update(origin_implementation_signature().encode("ascii"))
    digest.update(b"\0")
    for relative in _ABLATION_IMPLEMENTATION_FILES:
        path = root / relative
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes() if path.is_file() else b"<missing>")
        digest.update(b"\0")
    return digest.hexdigest()


def _truthful_architecture_record(model: nn.Module) -> dict[str, Any]:
    metadata = getattr(model, "architecture_metadata", None)
    declared = metadata() if callable(metadata) else metadata
    if not isinstance(declared, Mapping):
        raise TypeError(
            "ablation model architecture_metadata must be a mapping or callable"
        )
    state_shapes = {
        name: {"shape": list(tensor.shape), "dtype": str(tensor.dtype)}
        for name, tensor in model.state_dict().items()
    }
    return {
        "model_class": f"{model.__class__.__module__}.{model.__class__.__qualname__}",
        "declared": dict(declared),
        "state_shapes": state_shapes,
        "ablation_variant": declared.get("ablation_variant"),
        "no_classifier_bypass": bool(declared.get("no_classifier_bypass", False)),
        "posterior_path": declared.get("posterior_path"),
        "supports_exact_replay": bool(declared.get("supports_exact_replay", False)),
    }


class OriginAblationTrainer(OriginTrainer):
    """OriginTrainer with ablation provenance and replay-aware certificates."""

    def __init__(self, *args: object, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)
        declared_variant = self.architecture.get("declared", {}).get(
            "ablation_variant"
        )
        configured_variant = (
            self.cfg.get("ablation_variant")
            if isinstance(self.cfg, Mapping)
            else getattr(self.cfg, "ablation_variant", None)
        )
        if declared_variant != configured_variant:
            raise ValueError(
                "ablation model/config variant mismatch: "
                f"model={declared_variant!r}, config={configured_variant!r}"
            )
        self.implementation_signature = origin_ablation_implementation_signature()
        self.architecture = _truthful_architecture_record(self.model)
        self.architecture_signature = _canonical_sha256(self.architecture)
        self._pooled_control = (
            self.architecture.get("declared", {}).get("control_family")
            == "multiscale_masked_global_pooling"
        )
        if self._pooled_control and float(
            getattr(self.cfg, "evidence_budget_weight", 0.0)
            if not isinstance(self.cfg, Mapping)
            else self.cfg.get("evidence_budget_weight", 0.0)
        ) != 0.0:
            raise ValueError(
                "pooled controls have no local rate ledger and require "
                "evidence_budget_weight=0"
            )

    def _run_epoch(self, loader, *, train: bool, epoch: int) -> dict[str, Any]:
        if self._pooled_control:
            # Suppress the base trainer's conserved-rate-cap warning: pooled
            # outputs carry posterior-equivalent hazards only, so comparing
            # them with ORIGIN's 64-rate architectural cap is category error.
            original_cap = self.total_rate_cap
            self.total_rate_cap = 1e300
            try:
                metrics = super()._run_epoch(loader, train=train, epoch=epoch)
            finally:
                self.total_rate_cap = original_cap
        else:
            metrics = super()._run_epoch(loader, train=train, epoch=epoch)
        if not self._pooled_control:
            metrics["rate_telemetry_applicable"] = True
            return metrics

        # OriginTrainer requires a nonnegative boundary-shaped tensor for a
        # common output contract. Pooled controls expose hazard-equivalent
        # posterior telemetry there, but it is not a conserved local rate and
        # must never be compared with ORIGIN's architectural rate cap. Retain
        # it under explicit names and use a zero/N-A sentinel in legacy log
        # columns so tables cannot silently report an invalid comparison.
        metrics["rate_telemetry_applicable"] = False
        metrics["mean_equivalent_hazard_telemetry"] = metrics["mean_total_rate"]
        metrics["max_equivalent_hazard_telemetry"] = metrics["max_total_rate"]
        metrics["mean_total_rate"] = 0.0
        metrics["max_total_rate"] = 0.0
        metrics["total_rate_cap"] = 0.0
        metrics["max_total_rate_cap_fraction"] = 0.0
        boundary = 0
        while f"mean_total_rate_boundary_{boundary}" in metrics:
            metrics[f"mean_equivalent_hazard_boundary_{boundary}"] = metrics[
                f"mean_total_rate_boundary_{boundary}"
            ]
            metrics[f"max_equivalent_hazard_boundary_{boundary}"] = metrics[
                f"max_total_rate_boundary_{boundary}"
            ]
            metrics[f"mean_total_rate_boundary_{boundary}"] = 0.0
            metrics[f"max_total_rate_boundary_{boundary}"] = 0.0
            boundary += 1
        return metrics

    def _write_validation_certificates(self, checkpoint_epoch: int) -> dict[str, Any]:
        if bool(getattr(self.model, "supports_exact_replay", False)):
            return super()._write_validation_certificates(checkpoint_epoch)

        payload: dict[str, Any] = {
            "schema": "origin-ablation-certificate-status-v1",
            "scope": "inner_validation_only",
            "fold": self.fold,
            "split_signature": self.split_signature,
            "checkpoint_epoch": int(checkpoint_epoch),
            "checkpoint_sha256": self._file_sha256(self.best_path),
            "implementation_signature": self.implementation_signature,
            "architecture_signature": self.architecture_signature,
            "ablation_variant": self.architecture["ablation_variant"],
            "certificate_status": "not_applicable",
            "reason": (
                "globally_pooled_control_has_no_replayable_local_rate_ledger"
            ),
            "supports_exact_replay": False,
        }
        payload["content_checksum_sha256"] = _canonical_sha256(payload)
        temporary = self.certificate_path.with_name(
            f".{self.certificate_path.name}.{os.getpid()}.tmp"
        )
        try:
            with temporary.open("w", encoding="utf-8") as stream:
                json.dump(
                    payload,
                    stream,
                    indent=2,
                    sort_keys=True,
                    allow_nan=False,
                    default=_json_default,
                )
                stream.write("\n")
            os.replace(temporary, self.certificate_path)
        finally:
            if temporary.exists():
                temporary.unlink()
        return payload


__all__ = [
    "OriginAblationTrainer",
    "origin_ablation_implementation_signature",
]
