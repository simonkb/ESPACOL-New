"""Trainer adapter for the locked acceptance-revision baseline suite."""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import random
from typing import Any, Mapping

import numpy as np
import torch

from losses.origin_acceptance_baselines import OriginAcceptanceBaselineLoss
from models.origin_acceptance_baselines import AcceptanceLocalOutput
from .origin_ablation_trainer import (
    OriginAblationTrainer,
    origin_ablation_implementation_signature,
)


_IMPLEMENTATION_FILES = (
    "configs/origin_acceptance_baseline_config.py",
    "models/origin_acceptance_baselines.py",
    "losses/origin_acceptance_baselines.py",
    "training/origin_acceptance_baseline_trainer.py",
    "scripts/origin_acceptance_baseline_common.py",
    "train_origin_acceptance_baseline.py",
    "release_origin_acceptance_baselines.py",
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


def origin_acceptance_baseline_implementation_signature() -> str:
    """Content-address every source defining this experiment family."""

    root = Path(__file__).resolve().parents[1]
    digest = hashlib.sha256()
    digest.update(origin_ablation_implementation_signature().encode("ascii"))
    digest.update(b"\0")
    for relative in _IMPLEMENTATION_FILES:
        path = root / relative
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes() if path.is_file() else b"<missing>")
        digest.update(b"\0")
    return digest.hexdigest()


def _atomic_json(path: Path, payload: Any) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
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
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _tensor_field(output: object, name: str) -> torch.Tensor:
    value = output.get(name) if isinstance(output, Mapping) else getattr(output, name, None)
    if not torch.is_tensor(value):
        raise TypeError(f"baseline output field {name!r} must be a tensor")
    return value


class OriginAcceptanceBaselineTrainer(OriginAblationTrainer):
    """Same optimization/selection policy, with honest comparator semantics."""

    def __init__(self, *args: object, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)
        declared = self.architecture.get("declared", {})
        variant = str(declared.get("baseline_variant", declared.get("ablation_variant")))
        configured = (
            self.cfg.get("baseline_variant")
            if isinstance(self.cfg, Mapping)
            else getattr(self.cfg, "baseline_variant", None)
        )
        if variant != configured:
            raise ValueError(
                f"baseline model/config mismatch: model={variant!r}, config={configured!r}"
            )
        old_weights = getattr(self.criterion, "class_weights", None)
        class_weights = (
            None
            if old_weights is None or int(old_weights.numel()) == 0
            else old_weights.detach().clone()
        )
        self.criterion = OriginAcceptanceBaselineLoss(
            self.num_classes,
            baseline_variant=variant,
            rps_weight=float(self._cfg("rps_weight", 0.25)),
            sparse_l1_weight=float(self._cfg("sparse_l1_weight", 0.0)),
            sparse_l1_delay_epochs=int(self._cfg("sparse_l1_delay_epochs", 0)),
            class_weights=class_weights,
        ).to(self.device)
        self.population_objective_proper = (
            self.criterion.configured_objective_is_proper
            and not self.stratified_batches
        )
        self.baseline_variant = variant
        rate_applicable = bool(declared.get("rate_telemetry_applicable", True))
        if declared.get("control_family") == "multiscale_masked_global_pooling":
            rate_applicable = False
        self._pooled_control = not rate_applicable
        self.implementation_signature = (
            origin_acceptance_baseline_implementation_signature()
        )

        # These filenames make the prospective selection set unambiguous:
        # epoch zero is recorded only in warm_start_floor.json, while every
        # selectable checkpoint has completed at least one optimizer epoch.
        self.best_path = self.run_dir / "best_learned.pth"
        self.last_path = self.run_dir / "last_learned.pth"
        self.history_path = self.run_dir / "learned_history.csv"
        self.certificate_path = self.run_dir / "validation_certificates.json"
        self.warm_floor_path = self.run_dir / "warm_start_floor.json"

    def _cfg(self, name: str, default: Any = None) -> Any:
        if isinstance(self.cfg, Mapping):
            return self.cfg.get(name, default)
        return getattr(self.cfg, name, default)

    def _checkpoint_payload(self, *args: object, **kwargs: object) -> dict[str, Any]:
        payload = super()._checkpoint_payload(*args, **kwargs)
        payload.update(
            {
                "acceptance_schema": "origin-acceptance-learned-checkpoint-v1",
                "checkpoint_role": "learned_epoch_only",
                "warm_start_eligible_for_selection": False,
                "baseline_variant": self.baseline_variant,
            }
        )
        if int(payload["epoch"]) < 1:
            raise AssertionError("learned checkpoint must follow an optimizer epoch")
        return payload

    def _run_epoch(self, loader, *, train: bool, epoch: int) -> dict[str, Any]:
        metrics = super()._run_epoch(loader, train=train, epoch=epoch)
        metrics["regularizer_kind"] = (
            "unpooled_valid_activation_l1"
            if self.baseline_variant == "sparse_bagnet"
            else "none"
        )
        metrics["sparse_activation_l1"] = float(metrics["evidence_budget"])
        metrics["sparse_l1_active"] = bool(metrics["budget_active"])
        # Prevent downstream tables from mistaking the inherited diagnostic
        # column for ORIGIN's local-rate budget.
        metrics["evidence_budget_applicable"] = False
        return metrics

    @staticmethod
    def _snapshot_rng() -> dict[str, Any]:
        return {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        }

    @staticmethod
    def _restore_rng_snapshot(state: Mapping[str, Any]) -> None:
        random.setstate(state["python"])
        np.random.set_state(state["numpy"])
        torch.set_rng_state(state["torch"])
        cuda = state.get("cuda")
        if torch.cuda.is_available() and cuda is not None:
            for index, value in enumerate(cuda[: torch.cuda.device_count()]):
                torch.cuda.set_rng_state(value, device=index)

    def _write_or_validate_warm_floor(self, *, resume: bool) -> dict[str, Any]:
        if resume:
            if not self.warm_floor_path.is_file():
                raise FileNotFoundError(
                    f"resume requires warm-start floor artifact: {self.warm_floor_path}"
                )
            with self.warm_floor_path.open(encoding="utf-8") as stream:
                payload = json.load(stream)
            checks = {
                "implementation_signature": self.implementation_signature,
                "architecture_signature": self.architecture_signature,
                "config_signature": self.config_signature,
                "split_signature": self.split_signature,
                "baseline_variant": self.baseline_variant,
            }
            mismatches = [key for key, value in checks.items() if payload.get(key) != value]
            if mismatches:
                raise ValueError(
                    "warm-start floor resume mismatch in " + ", ".join(mismatches)
                )
            checksum = payload.pop("content_checksum_sha256", None)
            expected = _canonical_sha256(payload)
            payload["content_checksum_sha256"] = checksum
            if checksum != expected:
                raise ValueError("warm-start floor checksum mismatch")
            return payload
        if self.warm_floor_path.exists():
            raise FileExistsError(
                f"fresh run would overwrite warm-start floor: {self.warm_floor_path}"
            )

        # Validation at epoch zero is diagnostic only. Restore all global RNG
        # streams afterwards so measuring the common floor cannot change the
        # matched training stream.
        state = self._snapshot_rng()
        try:
            metrics = self._run_epoch(self.val_loader, train=False, epoch=0)
        finally:
            self._restore_rng_snapshot(state)
        payload: dict[str, Any] = {
            "schema": "origin-acceptance-warm-start-floor-v1",
            "scope": "inner_validation_only",
            "epoch": 0,
            "eligible_for_model_selection": False,
            "baseline_variant": self.baseline_variant,
            "fold": self.fold,
            "split_signature": self.split_signature,
            "implementation_signature": self.implementation_signature,
            "architecture_signature": self.architecture_signature,
            "config_signature": self.config_signature,
            "metrics": metrics,
        }
        payload["content_checksum_sha256"] = _canonical_sha256(payload)
        _atomic_json(self.warm_floor_path, payload)
        return payload

    def _write_local_certificate(self, checkpoint_epoch: int) -> dict[str, Any]:
        requested = max(1, int(self._cfg("certificate_samples", 4)))
        self.model.eval()
        batch = next(iter(self.val_loader))
        images, pixel_mask, labels, indices = self._unpack_batch(batch)
        with torch.no_grad():
            baseline = self._forward(images, pixel_mask)
            if not isinstance(baseline, AcceptanceLocalOutput):
                raise TypeError("local certificate requires AcceptanceLocalOutput")
            maps = baseline.effective_local_maps
            masks = baseline.original_valid_masks
            count = min(requested, int(labels.numel()))
            removals = {
                name: torch.zeros_like(mask, dtype=torch.bool)
                for name, mask in masks.items()
            }
            chosen: list[tuple[str, tuple[int, int], float]] = []
            for sample in range(count):
                best: tuple[float, str, tuple[int, int]] | None = None
                for scale, local_map in maps.items():
                    score = local_map[sample].abs().sum(dim=-1)
                    score = score.masked_fill(~masks[scale][sample], -torch.inf)
                    flat = int(score.reshape(-1).argmax())
                    value = float(score.reshape(-1)[flat].cpu())
                    spatial = tuple(
                        int(value_) for value_ in np.unravel_index(flat, score.shape)
                    )
                    candidate = (value, scale, spatial)
                    if best is None or candidate > best:
                        best = candidate
                if best is None or not math.isfinite(best[0]):
                    raise ValueError("certificate sample has no valid local cell")
                value, scale, spatial = best
                removals[scale][(sample, *spatial)] = True
                chosen.append((scale, spatial, value))

            intervention = self.model.replay_without(
                baseline, removals, force_decoder_fp64=True
            )
            replayed = intervention.output
            partition_error = 0.0
            for scale, original in maps.items():
                expanded = removals[scale].unsqueeze(-1)
                removed_map = torch.where(expanded, original, torch.zeros_like(original))
                error = (
                    original - (replayed.effective_local_maps[scale] + removed_map)
                ).abs().max()
                partition_error = max(partition_error, float(error.cpu()))
            if partition_error != 0.0:
                raise AssertionError("local replay failed exact ledger partition")

            reconstructed = self.model.intercept.to(torch.float64)[None].expand_as(
                baseline.aggregated_logits
            )
            for local_map in maps.values():
                reconstructed = reconstructed + local_map.to(torch.float64).sum((1, 2))
            no_bypass_error = float(
                (reconstructed - baseline.aggregated_logits).abs().max().cpu()
            )
            expected_logits = (
                baseline.aggregated_logits - intervention.removed_contributions
            )
            replay_error = float(
                (expected_logits - replayed.aggregated_logits).abs().max().cpu()
            )
            tolerance = float(self._cfg("certificate_replay_tolerance", 2e-6))
            if no_bypass_error > tolerance or replay_error > tolerance:
                raise AssertionError(
                    "local certificate failed: "
                    f"no-bypass={no_bypass_error:.3g}, replay={replay_error:.3g}, "
                    f"tolerance={tolerance:.3g}"
                )

            base_probs = baseline.class_probs.cpu()
            new_probs = replayed.class_probs.cpu()
            base_pred = self._predictions(baseline).cpu()
            new_pred = self._predictions(replayed).cpu()
            removed = intervention.removed_contributions.cpu()
            certificates = []
            for sample, (scale, spatial, magnitude) in enumerate(chosen):
                scale_metadata = baseline.scale_metadata.get(scale)
                stride = getattr(scale_metadata, "output_stride", None)
                offset = getattr(scale_metadata, "center_offset", None)
                receptive_field = getattr(scale_metadata, "receptive_field", None)
                center = (
                    None
                    if stride is None or offset is None
                    else [
                        float(offset + spatial[0] * stride),
                        float(offset + spatial[1] * stride),
                    ]
                )
                certificates.append(
                    {
                        "sample_id": self._index_value(indices, sample),
                        "label": int(labels[sample].cpu()),
                        "removed_scale": scale,
                        "removed_spatial_index": list(spatial),
                        "selected_absolute_map_mass": magnitude,
                        "removed_logit_contributions": removed[sample].tolist(),
                        "input_center_yx": center,
                        "output_stride_pixels": stride,
                        "receptive_field_pixels": receptive_field,
                        "baseline_prediction": int(base_pred[sample]),
                        "replayed_prediction": int(new_pred[sample]),
                        "baseline_class_probs": base_probs[sample].tolist(),
                        "replayed_class_probs": new_probs[sample].tolist(),
                        "expected_grade_delta": float(
                            baseline.expected_grade[sample].cpu()
                            - replayed.expected_grade[sample].cpu()
                        ),
                    }
                )
        payload: dict[str, Any] = {
            "schema": "origin-acceptance-local-ledger-certificates-v1",
            "scope": "inner_validation_only",
            "fold": self.fold,
            "split_signature": self.split_signature,
            "checkpoint_epoch": int(checkpoint_epoch),
            "checkpoint_sha256": self._file_sha256(self.best_path),
            "implementation_signature": self.implementation_signature,
            "architecture_signature": self.architecture_signature,
            "baseline_variant": self.baseline_variant,
            "selection_rule": "largest_absolute_single_cell_map_mass_across_scales",
            "deletion_denominator": "frozen_original_valid_count",
            "interpretation_scope": "exact_stored_ledger_not_causal_pixel_deletion",
            "exact_ledger_partition_max_abs_error": partition_error,
            "aggregate_logit_replay_max_abs_error": replay_error,
            "no_classifier_bypass_max_abs_error": no_bypass_error,
            "replay_tolerance": tolerance,
            "certificates": certificates,
        }
        payload["content_checksum_sha256"] = _canonical_sha256(payload)
        _atomic_json(self.certificate_path, payload)
        return payload

    def _write_validation_certificates(self, checkpoint_epoch: int) -> dict[str, Any]:
        if self.baseline_variant in {"ordinal_additive_mil", "sparse_bagnet"}:
            return self._write_local_certificate(checkpoint_epoch)
        return super()._write_validation_certificates(checkpoint_epoch)

    def fit(self, *, evaluate_test: bool = False) -> dict[str, Any]:
        if evaluate_test:
            raise ValueError(
                "acceptance-baseline training cannot evaluate the locked outer test; "
                "use the suite release command"
            )
        resume = bool(self._cfg("resume", False))
        warm = self._write_or_validate_warm_floor(resume=resume)
        result = super().fit(evaluate_test=False)
        if int(result["best_epoch"]) < 1:
            raise AssertionError("best_learned must be selected after an optimizer epoch")
        result.update(
            {
                "baseline_variant": self.baseline_variant,
                "best_learned_checkpoint": str(self.best_path),
                "last_learned_checkpoint": str(self.last_path),
                "warm_start_floor_path": str(self.warm_floor_path),
                "warm_start_floor_metrics": warm["metrics"],
                "warm_start_eligible_for_selection": False,
                "selection_candidates": "post_update_epochs_1_through_stop_only",
                "test_evaluated": False,
            }
        )
        return result


__all__ = [
    "OriginAcceptanceBaselineTrainer",
    "origin_acceptance_baseline_implementation_signature",
]
