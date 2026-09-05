"""Training loop for MOSAIC's proof-exclusive ordinal model."""

from __future__ import annotations

import csv
import hashlib
import logging
import math
import os
import random
import time
from dataclasses import asdict
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from configs.config import MOSAICConfig
from losses.mosaic import MosaicLoss
from models.mosaic_decoder import (
    PROOF_DECISION_RULES,
    decision_rule_outputs,
    proof_only_decisions,
)
from utils.metrics import evaluate_predictions


logger = logging.getLogger(__name__)


_IMPLEMENTATION_FILES = (
    "configs/config.py",
    "Datasets/mosaic_data.py",
    "models/local_efficientnet.py",
    "models/mosaic.py",
    "models/mosaic_decoder.py",
    "models/mosaic_model.py",
    "losses/mosaic.py",
    "utils/spatial_mask.py",
    "training/mosaic_trainer.py",
)


def mosaic_implementation_signature() -> str:
    """Content-address every source file that defines a training forward pass.

    Git identity alone is insufficient while developing on a dirty worktree.
    Refusing a resume after any of these files changes prevents a numerically
    incompatible checkpoint from being silently continued under new code.
    """

    root = Path(__file__).resolve().parents[1]
    digest = hashlib.sha256()
    for relative in _IMPLEMENTATION_FILES:
        path = root / relative
        if not path.is_file():
            raise FileNotFoundError(f"MOSAIC implementation file is missing: {path}")
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _macro_metrics(predicted: torch.Tensor, labels: torch.Tensor, classes: int) -> dict[str, float]:
    """Dependency-free balanced accuracy, macro-F1, and confusion matrix."""
    confusion = torch.zeros(classes, classes, dtype=torch.long)
    for truth, prediction in zip(labels.long(), predicted.long()):
        confusion[int(truth), int(prediction)] += 1
    recalls = confusion.diag().float() / confusion.sum(dim=1).clamp_min(1)
    precisions = confusion.diag().float() / confusion.sum(dim=0).clamp_min(1)
    f1 = 2 * precisions * recalls / (precisions + recalls).clamp_min(1e-12)
    return {
        "balanced_acc": 100.0 * float(recalls.mean()),
        "macro_f1": float(f1.mean()),
        "confusion": confusion.tolist(),
        "per_grade_recall": recalls.tolist(),
    }


class _EarlyStopping:
    def __init__(self, patience: int) -> None:
        self.patience = int(patience)
        self.best = -math.inf
        self.bad_epochs = 0

    def update(self, value: float) -> bool:
        if value > self.best + 1e-8:
            self.best = value
            self.bad_epochs = 0
            return False
        self.bad_epochs += 1
        return self.bad_epochs >= self.patience


class MosaicTrainer:
    """Train, select, and evaluate one MOSAIC fold.

    Model selection and early stopping use validation accuracy, while the
    plateau scheduler follows validation loss.  The held-out test split is
    evaluated only after the best validation checkpoint is restored.
    """

    _RESUME_CRITICAL_CONFIG_FIELDS = (
        "dataset",
        "preprocessing_version",
        "n_classes",
        "n_folds",
        "val_fraction",
        "seed",
        "img_size",
        "batch_size",
        "num_workers",
        "local_stage",
        "evidence_dim",
        "pretrained",
        "grad_checkpoint",
        "max_count",
        "count_block_size",
        "count_implementation",
        "normal_expected_count",
        "proof_epsilon",
        "necessity_fraction",
        "dense_warmup_epochs",
        "proof_ramp_epochs",
        "dense_loss_weight",
        "stability_loss_weight",
        "transition_weighting",
        "effective_num_beta",
        "transition_weight_cap",
        "transition_reduction",
        "decision_rule",
        "stratified",
        "lr",
        "head_lr",
        "weight_decay",
        "lr_factor",
        "lr_patience",
        "lr_min",
        "grad_clip_norm",
        "amp",
        "amp_init_scale",
        "amp_growth_interval",
        "amp_max_consecutive_skips",
    )

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: DataLoader,
        cfg: MOSAICConfig,
        run_dir: str,
        train_labels: list[int],
        *,
        fold: int = 0,
        split_signature: Optional[str] = None,
        device: Optional[torch.device] = None,
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
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else
            "mps" if torch.backends.mps.is_available() else "cpu"
        )
        if cfg.decision_rule not in PROOF_DECISION_RULES:
            raise ValueError(
                f"unknown MOSAIC decision rule {cfg.decision_rule!r}; "
                f"expected one of {PROOF_DECISION_RULES}"
            )
        if cfg.transition_reduction == "boundary_mean" and cfg.stratified:
            raise ValueError(
                "boundary_mean transition reduction requires uniformly sampled "
                "training batches; disable stratified_batches or use the "
                "historical sample_mean reduction"
            )
        self.model.to(self.device)
        self.use_amp = bool(cfg.amp and self.device.type == "cuda")
        if cfg.amp_init_scale <= 0:
            raise ValueError("amp_init_scale must be positive")
        if cfg.amp_growth_interval < 1:
            raise ValueError("amp_growth_interval must be positive")
        if cfg.amp_max_consecutive_skips < 1:
            raise ValueError("amp_max_consecutive_skips must be positive")
        self.scaler = GradScaler(
            "cuda",
            enabled=self.use_amp,
            init_scale=float(cfg.amp_init_scale),
            growth_interval=int(cfg.amp_growth_interval),
        )
        self.amp_total_skipped_steps = 0
        self.amp_total_forward_retries = 0
        self.amp_consecutive_skipped_steps = 0
        self.amp_consecutive_forward_retries = 0

        # Only the ImageNet-initialised convolutional trunk uses the lower
        # backbone LR.  ``encoder.pointwise`` is a new, randomly initialised
        # projection/MLP and must learn with the proof head.  Grouping the
        # complete encoder at ``cfg.lr`` silently starves this local evidence
        # adapter by the same factor as the pretrained CNN.
        encoder_parameters = list(self.model.encoder.trunk.parameters())
        encoder_ids = {id(parameter) for parameter in encoder_parameters}
        head_parameters = [
            parameter for parameter in self.model.parameters()
            if id(parameter) not in encoder_ids
        ]
        self.optimizer = torch.optim.AdamW(
            [
                {"params": encoder_parameters, "lr": cfg.lr, "name": "local_encoder"},
                {"params": head_parameters, "lr": cfg.head_lr, "name": "proof_head"},
            ],
            weight_decay=cfg.weight_decay,
        )
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            factor=cfg.lr_factor,
            patience=cfg.lr_patience,
            min_lr=cfg.lr_min,
        )
        self.criterion = MosaicLoss.from_training_labels(
            train_labels,
            cfg.n_classes,
            weight_method=cfg.transition_weighting,
            weight_beta=cfg.effective_num_beta,
            max_transition_weight=cfg.transition_weight_cap,
            dense_weight=cfg.dense_loss_weight,
            stability_weight=cfg.stability_loss_weight,
            transition_reduction=cfg.transition_reduction,
        ).to(self.device)
        self._configure_model_decision()
        if cfg.stability_loss_weight > 0:
            raise ValueError(
                "stability_loss_weight requires geometry-matched dual photometric "
                "views; keep it at zero for the initial single-view experiment"
            )
        self.early_stopping = _EarlyStopping(cfg.early_stop_patience)
        self.checkpoint_path = self.run_dir / "best.pth"
        self.last_checkpoint_path = self.run_dir / "last.pth"
        self.history_path = self.run_dir / "history.csv"
        self.implementation_signature = mosaic_implementation_signature()

    def _configure_model_decision(self) -> None:
        """Bind the direct model API to the criterion's fold-level weights."""

        weights = self.criterion.transition_weights.detach()
        if bool((weights <= 0.0).any()):
            raise ValueError(
                "invalid MOSAIC training fold: every ordinal boundary must have "
                "both stop and advance examples; at least one outcome has zero "
                "weight for the declared number of grades"
            )
        configure = getattr(self.model, "configure_proof_decoder", None)
        if configure is None:
            raise TypeError("MOSAIC model does not expose configure_proof_decoder")
        configure(self.cfg.decision_rule, weights)

    def _proof_phase(self, epoch: int) -> tuple[bool, float]:
        if epoch <= self.cfg.dense_warmup_epochs:
            return False, 0.0
        ramp_epoch = epoch - self.cfg.dense_warmup_epochs - 1
        if self.cfg.proof_ramp_epochs <= 1:
            fraction = 1.0
        else:
            fraction = min(1.0, max(0.0, ramp_epoch / (self.cfg.proof_ramp_epochs - 1)))
        return True, self.cfg.proof_epsilon * fraction

    def _move_batch(self, batch):
        images, pixel_masks, labels, indices = batch
        return (
            images.to(self.device, non_blocking=True),
            pixel_masks.to(self.device, non_blocking=True),
            labels.to(self.device, non_blocking=True),
            indices,
        )

    def _nonfinite_gradient_summary(self, limit: int = 8) -> str:
        """Name the parameters containing non-finite gradients.

        This runs only on an overflow path, so the per-parameter CUDA
        synchronizations do not affect ordinary training throughput.
        """

        offenders: list[str] = []
        for name, parameter in self.model.named_parameters():
            gradient = parameter.grad
            if gradient is None:
                continue
            finite = torch.isfinite(gradient)
            if bool(finite.all()):
                continue
            nonfinite = int((~finite).sum().detach().cpu())
            offenders.append(f"{name}[nonfinite={nonfinite}]")
            if len(offenders) >= limit:
                break
        return ", ".join(offenders) if offenders else "unknown parameter"

    def _nonfinite_parameter_summary(self, limit: int = 8) -> str:
        """Name corrupt parameters or optimizer tensors on an error path."""

        offenders: list[str] = []
        for name, parameter in self.model.named_parameters():
            finite = torch.isfinite(parameter)
            if bool(finite.all()):
                continue
            offenders.append(
                f"parameter:{name}[nonfinite={int((~finite).sum().detach().cpu())}]"
            )
            if len(offenders) >= limit:
                return ", ".join(offenders)
        parameter_names = {
            id(parameter): name for name, parameter in self.model.named_parameters()
        }
        for parameter, state in self.optimizer.state.items():
            name = parameter_names.get(id(parameter), "unknown")
            for key, value in state.items():
                if not torch.is_tensor(value):
                    continue
                finite = torch.isfinite(value)
                if bool(finite.all()):
                    continue
                offenders.append(
                    f"optimizer:{name}.{key}"
                    f"[nonfinite={int((~finite).sum().detach().cpu())}]"
                )
                if len(offenders) >= limit:
                    return ", ".join(offenders)
        return ", ".join(offenders) if offenders else "all stored tensors finite"

    @staticmethod
    def _nonfinite_output_summary(output) -> str:
        """Name invalid tensors in a completed MOSAIC forward pass.

        A log stop of ``-inf`` is the mathematically valid representation of
        an exact zero lower-tail probability.  NaN and ``+inf`` are never
        valid.  Transition probabilities, by contrast, must all be finite.
        Keeping this distinction here prevents a recovery path from silently
        clipping or otherwise changing the ordinal likelihood.
        """

        checks = (
            ("transitions", output.transitions, False),
            ("dense_transitions", output.dense_transitions, False),
            ("log_stop_probabilities", output.log_stop_probabilities, True),
            (
                "dense_log_stop_probabilities",
                output.dense_log_stop_probabilities,
                True,
            ),
        )
        invalid_masks: list[tuple[str, torch.Tensor]] = []
        for name, tensor, allow_negative_infinity in checks:
            invalid = torch.isnan(tensor) | torch.isposinf(tensor)
            if not allow_negative_infinity:
                invalid = invalid | torch.isneginf(tensor)
            invalid_masks.append((name, invalid))
        # One synchronization on the ordinary valid path.  Detailed counts
        # are transferred only for the exceptional retry/error message.
        any_invalid = torch.stack(
            [invalid.any() for _, invalid in invalid_masks]
        ).any()
        if not bool(any_invalid):
            return ""
        offenders: list[str] = []
        for name, invalid in invalid_masks:
            count = int(invalid.sum().detach().cpu())
            if count:
                offenders.append(f"{name}[nonfinite={count}]")
        return ", ".join(offenders)

    def _run_epoch(self, loader: DataLoader, *, train: bool, epoch: int) -> dict:
        self.model.train(train)
        project, tolerance = self._proof_phase(epoch) if train else (True, self.cfg.proof_epsilon)
        self.model.set_proof_tolerance(tolerance)

        loss_sum = torch.zeros((), device=self.device)
        sample_count = 0
        all_labels: list[torch.Tensor] = []
        all_transitions: list[torch.Tensor] = []
        all_log_stops: list[torch.Tensor] = []
        proof_sizes: list[torch.Tensor] = []
        proof_fractions: list[torch.Tensor] = []
        sufficiency_gaps: list[torch.Tensor] = []
        complement_scores: list[torch.Tensor] = []
        sufficiency_violations: list[torch.Tensor] = []
        complement_violations: list[torch.Tensor] = []
        diagnostic_sums: dict[str, torch.Tensor] = {}
        amp_skipped_steps = 0
        amp_forward_retries = 0
        boundary_risk_sums = torch.zeros(
            self.cfg.n_classes - 1, device=self.device
        )
        boundary_advance_sums = torch.zeros_like(boundary_risk_sums)

        context = torch.enable_grad if train else torch.no_grad
        with context():
            for batch_index, batch in enumerate(loader):
                images, pixel_masks, labels, _ = self._move_batch(batch)
                if train:
                    self.optimizer.zero_grad(set_to_none=True)
                amp_forward_error: Optional[FloatingPointError] = None
                output = None
                try:
                    with autocast(device_type="cuda", enabled=self.use_amp):
                        output = self.model(
                            images,
                            pixel_valid_mask=pixel_masks,
                            project=project,
                        )
                except FloatingPointError as exc:
                    amp_forward_error = exc
                output_issues = (
                    self._nonfinite_output_summary(output)
                    if output is not None
                    else ""
                )
                forward_retried = False
                if self.use_amp and (amp_forward_error is not None or output_issues):
                    # GradScaler cannot repair a non-finite *forward* because
                    # its scale affects only backward.  Replay the same batch
                    # once in FP32, preserving the model and loss exactly while
                    # avoiding a global FP32 throughput penalty.
                    source = (
                        str(amp_forward_error)
                        if amp_forward_error is not None
                        else output_issues
                    )
                    logger.warning(
                        "AMP forward retry: fold=%d epoch=%d batch=%d "
                        "offenders=%s",
                        self.fold,
                        epoch,
                        batch_index,
                        source,
                    )
                    self.amp_consecutive_forward_retries += 1
                    if (
                        self.amp_consecutive_forward_retries
                        >= self.cfg.amp_max_consecutive_skips
                    ):
                        raise FloatingPointError(
                            "persistent non-finite MOSAIC AMP forwards: "
                            f"{self.amp_consecutive_forward_retries} consecutive "
                            f"batches by epoch {epoch}, batch {batch_index}; "
                            "GradScaler cannot repair forward overflow. "
                            f"Latest offenders: {source}"
                        )
                    if output is not None:
                        del output
                    try:
                        with autocast(device_type="cuda", enabled=False):
                            output = self.model(
                                images,
                                pixel_valid_mask=pixel_masks,
                                project=project,
                            )
                    except FloatingPointError as fp32_exc:
                        stored_state = self._nonfinite_parameter_summary()
                        raise FloatingPointError(
                            "non-finite MOSAIC forward after FP32 retry at "
                            f"epoch {epoch}, batch {batch_index}; "
                            f"AMP source: {source}; FP32 source: {fp32_exc}; "
                            f"stored state: {stored_state}"
                        ) from fp32_exc
                    forward_retried = True
                    amp_forward_retries += 1
                    self.amp_total_forward_retries += 1
                    output_issues = self._nonfinite_output_summary(output)
                elif amp_forward_error is not None:
                    raise FloatingPointError(
                        "non-finite MOSAIC FP32 forward at "
                        f"epoch {epoch}, batch {batch_index}: {amp_forward_error}"
                    ) from amp_forward_error
                else:
                    self.amp_consecutive_forward_retries = 0
                assert output is not None
                if output_issues:
                    precision = "FP32 retry" if forward_retried else "FP32 forward"
                    raise FloatingPointError(
                        "non-finite MOSAIC output after "
                        f"{precision} at epoch {epoch}, batch {batch_index}: "
                        f"{output_issues}"
                    )
                with autocast(
                    device_type="cuda",
                    enabled=self.use_amp and not forward_retried,
                ):
                    loss, diagnostics = self.criterion(
                        output.transitions,
                        labels,
                        projected_stop_probabilities=output.stop_probabilities,
                        projected_log_stop_probabilities=output.log_stop_probabilities,
                        dense_transitions=output.dense_transitions,
                        dense_stop_probabilities=output.dense_stop_probabilities,
                        dense_log_stop_probabilities=output.dense_log_stop_probabilities,
                    )
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"non-finite MOSAIC loss at epoch {epoch}")
                if train:
                    self.scaler.scale(loss).backward()
                    self.scaler.unscale_(self.optimizer)
                    # ``error_if_nonfinite`` raises before applying a clip
                    # coefficient.  Clipping an Inf norm first would multiply
                    # Inf gradients by zero and turn a recoverable AMP
                    # overflow into NaNs.
                    try:
                        grad_norm = torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(),
                            self.cfg.grad_clip_norm,
                            error_if_nonfinite=True,
                        )
                    except RuntimeError as exc:
                        if not self.use_amp:
                            raise FloatingPointError(
                                "non-finite MOSAIC gradient without AMP at "
                                f"epoch {epoch}"
                            ) from exc
                        scale_before = float(self.scaler.get_scale())
                        offenders = self._nonfinite_gradient_summary()
                        if offenders == "unknown parameter":
                            # The aggregate norm can itself overflow even when
                            # every gradient element is finite.  GradScaler then
                            # has no ``found_inf`` flag and would apply the bad
                            # step, so fail loudly instead of treating this as a
                            # recoverable mixed-precision overflow.
                            raise FloatingPointError(
                                "non-finite MOSAIC gradient norm with no "
                                "non-finite gradient elements at "
                                f"epoch {epoch}, batch {batch_index}"
                            ) from exc
                        # GradScaler recorded ``found_inf`` during unscale_.
                        # step() therefore leaves all parameters unchanged;
                        # update() lowers the scale for the next batch.
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                        scale_after = float(self.scaler.get_scale())
                        self.optimizer.zero_grad(set_to_none=True)
                        amp_skipped_steps += 1
                        self.amp_total_skipped_steps += 1
                        self.amp_consecutive_skipped_steps += 1
                        logger.warning(
                            "AMP overflow: skipped fold=%d epoch=%d batch=%d "
                            "scale=%.1f->%.1f offenders=%s",
                            self.fold,
                            epoch,
                            batch_index,
                            scale_before,
                            scale_after,
                            offenders,
                        )
                        if (
                            scale_after >= scale_before
                            or scale_after < 1.0
                            or self.amp_consecutive_skipped_steps
                            >= self.cfg.amp_max_consecutive_skips
                        ):
                            raise FloatingPointError(
                                "persistent non-finite MOSAIC AMP gradients at "
                                f"epoch {epoch}; scale {scale_before}->{scale_after}; "
                                f"offenders: {offenders}"
                            ) from exc
                        continue
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.amp_consecutive_skipped_steps = 0

                batch_size = int(labels.numel())
                loss_sum = loss_sum + loss.detach() * batch_size
                sample_count += batch_size
                all_labels.append(labels.detach())
                all_transitions.append(output.transitions.detach())
                all_log_stops.append(output.log_stop_probabilities.detach())
                for boundary in range(self.cfg.n_classes - 1):
                    risk = diagnostics[f"at_risk_boundary_{boundary}"]
                    advance_rate = diagnostics[f"advance_rate_boundary_{boundary}"]
                    boundary_risk_sums[boundary] += risk
                    boundary_advance_sums[boundary] += advance_rate * risk
                for key, value in diagnostics.items():
                    if key.startswith("at_risk_boundary_") or key.startswith(
                        "advance_rate_boundary_"
                    ):
                        continue
                    if key not in diagnostic_sums:
                        diagnostic_sums[key] = torch.zeros_like(value)
                    diagnostic_sums[key] = diagnostic_sums[key] + value * batch_size

                valid_counts = output.valid_mask.sum(dim=1).clamp_min(1)
                proof_sizes.append(output.proof.proof_size.detach().float())
                proof_fractions.append(
                    (output.proof.proof_size / valid_counts[:, None]).detach()
                )
                sufficiency_gaps.append(output.proof.sufficiency_gap.detach())
                complement_scores.append(output.proof.complement_transition.detach())
                target = (output.dense_transitions - tolerance).clamp_min(0.0)
                sufficiency_violations.append(
                    (target - output.transitions).clamp_min(0.0).detach()
                )
                complement_violations.append(
                    (
                        self.cfg.necessity_fraction * target
                        - output.proof.complement_drop
                    ).clamp_min(0.0).detach()
                )

        if sample_count == 0:
            raise FloatingPointError(
                f"MOSAIC epoch {epoch} produced no finite optimization batches"
            )
        labels_cpu = torch.cat(all_labels).cpu()
        transitions_cpu = torch.cat(all_transitions).float().cpu()
        log_stops_cpu = torch.cat(all_log_stops).float().cpu()
        decisions = proof_only_decisions(
            transitions_cpu,
            log_stops_cpu,
            self.criterion.transition_weights.detach().float().cpu(),
        )
        decision_rules = decision_rule_outputs(decisions)
        predictions, cumulative = decision_rules[self.cfg.decision_rule]
        metrics = evaluate_predictions(
            predictions.float(),
            labels_cpu,
            self.cfg.n_classes,
            ordinal_probs=cumulative,
        )
        metrics.update(_macro_metrics(predictions.long(), labels_cpu, self.cfg.n_classes))
        metrics["decision_rule"] = self.cfg.decision_rule
        # All alternatives are fixed before seeing validation labels.  Logging
        # them on the identical proof outputs distinguishes a decoder mismatch
        # from a representation bottleneck without fitting a calibration rule.
        for rule_name, (rule_prediction, rule_cumulative) in decision_rules.items():
            rule_metrics = evaluate_predictions(
                rule_prediction.float(),
                labels_cpu,
                self.cfg.n_classes,
                ordinal_probs=rule_cumulative,
            )
            for metric_name, value in rule_metrics.items():
                metrics[f"decoder_{rule_name}_{metric_name}"] = value
        proof_size_matrix = torch.cat(proof_sizes).cpu()
        sizes = proof_size_matrix.flatten()
        fractions = torch.cat(proof_fractions).flatten().cpu()
        sufficiency_gap_mean = torch.cat(sufficiency_gaps).mean().cpu()
        complement_score_mean = torch.cat(complement_scores).mean().cpu()
        sufficiency_violation_max = torch.cat(sufficiency_violations).max().cpu()
        complement_violation_max = torch.cat(complement_violations).max().cpu()
        metrics.update(
            {
                "loss": float((loss_sum / max(sample_count, 1)).cpu()),
                "proof_projected": float(project),
                "proof_tolerance": float(tolerance),
                "proof_size_mean": float(sizes.mean()),
                "proof_size_median": float(sizes.median()),
                "proof_size_p90": float(torch.quantile(sizes, 0.90)),
                "proof_fraction_mean": float(fractions.mean()),
                "proof_fraction_median": float(fractions.median()),
                "proof_fraction_p90": float(torch.quantile(fractions, 0.90)),
                "sufficiency_gap_mean": float(sufficiency_gap_mean),
                "sufficiency_violation_max": float(sufficiency_violation_max),
                "complement_score_mean": float(complement_score_mean),
                "complement_violation_max": float(complement_violation_max),
                "amp_skipped_steps": float(amp_skipped_steps),
                "amp_forward_retries": float(amp_forward_retries),
                "amp_loss_scale": float(self.scaler.get_scale()),
            }
        )
        for key, value in diagnostic_sums.items():
            metrics[key] = float((value / max(sample_count, 1)).cpu())
        boundary_risk_cpu = boundary_risk_sums.cpu()
        boundary_advance_cpu = boundary_advance_sums.cpu()
        for boundary, risk_tensor in enumerate(boundary_risk_cpu):
            risk = float(risk_tensor)
            metrics[f"at_risk_boundary_{boundary}"] = risk
            metrics[f"advance_rate_boundary_{boundary}"] = (
                float(boundary_advance_cpu[boundary]) / risk if risk > 0 else 0.0
            )
            advance = labels_cpu > boundary
            stop = labels_cpu == boundary
            zero_proof = proof_size_matrix[:, boundary] == 0
            zero_transition = transitions_cpu[:, boundary] == 0
            advance_count = int(advance.sum())
            stop_count = int(stop.sum())
            metrics[f"zero_proof_rate_boundary_{boundary}"] = float(
                zero_proof.float().mean()
            )
            metrics[f"zero_proof_advance_rate_boundary_{boundary}"] = (
                float((zero_proof & advance).sum()) / advance_count
                if advance_count > 0
                else 0.0
            )
            metrics[f"zero_proof_stop_rate_boundary_{boundary}"] = (
                float((zero_proof & stop).sum()) / stop_count
                if stop_count > 0
                else 0.0
            )
            metrics[f"zero_transition_advance_rate_boundary_{boundary}"] = (
                float((zero_transition & advance).sum()) / advance_count
                if advance_count > 0
                else 0.0
            )
        return metrics

    @staticmethod
    def _flat_history_row(epoch: int, train: dict, val: dict, lrs: list[float]) -> dict:
        row = {"epoch": epoch, "encoder_lr": lrs[0], "head_lr": lrs[1]}
        for prefix, metrics in (("train", train), ("val", val)):
            for key, value in metrics.items():
                if isinstance(value, (float, int)):
                    row[f"{prefix}_{key}"] = value
        return row

    def _append_history(self, row: dict) -> None:
        exists = self.history_path.exists()
        with self.history_path.open("a", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(row))
            if not exists:
                writer.writeheader()
            writer.writerow(row)

    def _reconcile_history(self, completed_epoch: int) -> None:
        """Make the CSV transactionally consistent with ``last.pth``.

        History is intentionally written before the resume checkpoint so a
        completed checkpoint never refers to an epoch whose metrics are
        absent.  If preemption happens in the narrow interval between those
        writes, the CSV can contain one uncommitted future row.  On resume we
        discard such rows and de-duplicate older epochs before re-running.
        """

        if not self.history_path.exists():
            return
        with self.history_path.open(newline="") as stream:
            reader = csv.DictReader(stream)
            fieldnames = reader.fieldnames
            if not fieldnames or "epoch" not in fieldnames:
                raise ValueError("MOSAIC history.csv has no epoch column")
            by_epoch: dict[int, dict[str, str]] = {}
            for row in reader:
                try:
                    epoch = int(row["epoch"])
                except (KeyError, TypeError, ValueError) as exc:
                    raise ValueError("MOSAIC history.csv has a malformed epoch") from exc
                if epoch <= completed_epoch:
                    by_epoch[epoch] = row
        temporary = self.history_path.with_name(
            f".{self.history_path.name}.{os.getpid()}.tmp"
        )
        try:
            with temporary.open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=fieldnames)
                writer.writeheader()
                for epoch in sorted(by_epoch):
                    writer.writerow(by_epoch[epoch])
            os.replace(temporary, self.history_path)
        finally:
            if temporary.exists():
                temporary.unlink()

    def _checkpoint_payload(
        self,
        epoch: int,
        metrics: dict,
        best_accuracy: float,
    ) -> dict:
        train_batch_sampler = getattr(self.train_loader, "batch_sampler", None)
        return {
            "epoch": epoch,
            "fold": self.fold,
            "split_signature": self.split_signature,
            "implementation_signature": self.implementation_signature,
            "model_state": self.model.state_dict(),
            "criterion_state": self.criterion.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "scheduler_state": self.scheduler.state_dict(),
            "scaler_state": self.scaler.state_dict(),
            "amp_total_skipped_steps": int(self.amp_total_skipped_steps),
            "amp_total_forward_retries": int(self.amp_total_forward_retries),
            "metrics": metrics,
            "best_accuracy": float(best_accuracy),
            "early_stopping_best": float(self.early_stopping.best),
            "early_stopping_bad_epochs": int(self.early_stopping.bad_epochs),
            "rng_state": {
                "python": random.getstate(),
                "numpy": np.random.get_state(),
                "torch": torch.get_rng_state(),
                "cuda": torch.cuda.get_rng_state_all()
                if torch.cuda.is_available()
                else None,
            },
            # StratifiedBatchSampler owns a deterministic epoch counter that is
            # independent of Python/Torch global RNG.  Persist it explicitly;
            # otherwise a resumed run silently restarts its batch sequence at
            # seed+0 even though the model/optimizer RNG states were restored.
            "train_batch_sampler_epoch": getattr(
                train_batch_sampler, "_epoch", None
            ),
            "config": asdict(self.cfg),
            "architecture": {
                "output_stride": self.model.output_stride,
                "receptive_field": self.model.receptive_field,
                "expected_valid_cells": self.model.expected_valid_cells,
                "preprocessing_version": self.cfg.preprocessing_version,
                "no_global_bypass": True,
                "decision_rule": self.cfg.decision_rule,
                "decision_inputs": (
                    "selected_proof_transitions",
                    "selected_proof_log_stop_probabilities",
                    "training_fold_boundary_outcome_weights",
                ),
                "transition_reduction": self.cfg.transition_reduction,
                "training_fold_at_risk_counts": (
                    self.criterion.at_risk_counts.detach().cpu().tolist()
                    if self.criterion.at_risk_counts is not None
                    else None
                ),
            },
        }

    def _validate_resume_checkpoint(self, state: dict) -> None:
        saved_implementation = state.get("implementation_signature")
        if saved_implementation is None:
            raise ValueError(
                "resume checkpoint has no MOSAIC implementation signature; "
                "start a fresh run rather than mix arithmetic versions"
            )
        if saved_implementation != self.implementation_signature:
            raise ValueError(
                "resume checkpoint was produced by a different MOSAIC "
                "implementation; start a fresh run directory"
            )

        saved_fold = state.get("fold")
        if saved_fold is None:
            raise ValueError(
                "resume checkpoint has no fold identity; start a new run rather "
                "than risk loading weights from an incompatible split"
            )
        if int(saved_fold) != self.fold:
            raise ValueError(
                f"resume checkpoint is for fold {saved_fold}, current fold is {self.fold}"
            )

        saved_signature = state.get("split_signature")
        if self.split_signature is not None:
            if saved_signature is None:
                raise ValueError(
                    "resume checkpoint has no split signature; data identity cannot "
                    "be verified"
                )
            if saved_signature != self.split_signature:
                raise ValueError(
                    "resume checkpoint split signature differs from the current "
                    "train/validation/test partition"
                )

        saved_config = state.get("config")
        if not isinstance(saved_config, dict):
            raise ValueError("resume checkpoint has no usable configuration")
        current_config = asdict(self.cfg)
        # A checkpoint without this post-audit field used the historical
        # sample-mean objective. Preserve that identity rather than allowing a
        # missing value to inherit the prospective boundary-mean default.
        saved_config_values = dict(saved_config)
        saved_config_values.setdefault("transition_reduction", "sample_mean")
        mismatches = {
            field: (saved_config_values.get(field), current_config.get(field))
            for field in self._RESUME_CRITICAL_CONFIG_FIELDS
            if saved_config_values.get(field) != current_config.get(field)
        }
        if mismatches:
            details = ", ".join(
                f"{field}: saved={saved!r}, current={current!r}"
                for field, (saved, current) in sorted(mismatches.items())
            )
            raise ValueError(
                "resume checkpoint uses incompatible training configuration (only "
                f"epochs/runtime controls may change): {details}"
            )

    @staticmethod
    def _restore_rng_state(state: dict) -> None:
        rng = state.get("rng_state")
        if not isinstance(rng, dict):
            raise ValueError(
                "resume checkpoint has no RNG state; reproducible continuation is "
                "not possible"
            )
        random.setstate(rng["python"])
        np.random.set_state(rng["numpy"])
        torch.set_rng_state(rng["torch"].cpu())
        cuda_states = rng.get("cuda")
        if torch.cuda.is_available() and cuda_states is not None:
            for device_index, cuda_state in enumerate(
                cuda_states[: torch.cuda.device_count()]
            ):
                torch.cuda.set_rng_state(cuda_state.cpu(), device=device_index)

    def _restore_training_sampler_state(self, state: dict) -> None:
        if not self.cfg.stratified:
            return
        saved_epoch = state.get("train_batch_sampler_epoch")
        sampler = getattr(self.train_loader, "batch_sampler", None)
        if saved_epoch is None:
            raise ValueError(
                "resume checkpoint has no stratified batch-sampler epoch"
            )
        # A freshly created StratifiedBatchSampler has no _epoch until its
        # first iteration.  It is nevertheless safe to restore the field.
        if sampler is None or sampler.__class__.__name__ != "StratifiedBatchSampler":
            raise ValueError(
                "current train loader does not expose the expected "
                "StratifiedBatchSampler"
            )
        sampler._epoch = int(saved_epoch)

    def _save(
        self,
        epoch: int,
        metrics: dict,
        best_accuracy: float,
        *,
        best: bool,
    ) -> None:
        path = self.checkpoint_path if best else self.last_checkpoint_path
        temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        try:
            torch.save(
                self._checkpoint_payload(epoch, metrics, best_accuracy),
                temporary,
            )
            # Same-directory replacement is atomic on the filesystems used by
            # the local workstation and cluster.  A preemption can leave only
            # the disposable temporary file, never a half-written last.pth.
            os.replace(temporary, path)
        finally:
            if temporary.exists():
                temporary.unlink()

    def _restore_best(self) -> dict:
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint["model_state"])
        self.criterion.load_state_dict(checkpoint["criterion_state"])
        self._configure_model_decision()
        return checkpoint

    def fit(self, *, evaluate_test: bool = True) -> dict:
        logger.info(
            "MOSAIC fold=%d device=%s stride=%d RF=%d decision=%s "
            "transition_reduction=%s at_risk_counts=%s transition_weights=%s",
            self.fold,
            self.device,
            self.model.output_stride,
            self.model.receptive_field,
            self.cfg.decision_rule,
            self.cfg.transition_reduction,
            (
                self.criterion.at_risk_counts.detach().cpu().tolist()
                if self.criterion.at_risk_counts is not None
                else None
            ),
            self.criterion.transition_weights.detach().cpu().tolist(),
        )
        best_accuracy = -math.inf
        start_epoch = 1
        if self.cfg.resume and self.last_checkpoint_path.exists():
            state = torch.load(
                self.last_checkpoint_path,
                map_location=self.device,
                weights_only=False,
            )
            self._validate_resume_checkpoint(state)
            self.model.load_state_dict(state["model_state"])
            self.criterion.load_state_dict(state["criterion_state"])
            self._configure_model_decision()
            self.optimizer.load_state_dict(state["optimizer_state"])
            self.scheduler.load_state_dict(state["scheduler_state"])
            if "scaler_state" in state:
                self.scaler.load_state_dict(state["scaler_state"])
            self.amp_total_skipped_steps = int(
                state.get("amp_total_skipped_steps", 0)
            )
            self.amp_total_forward_retries = int(
                state.get("amp_total_forward_retries", 0)
            )
            self.amp_consecutive_skipped_steps = 0
            best_accuracy = float(state.get("best_accuracy", -math.inf))
            self.early_stopping.best = float(
                state.get("early_stopping_best", best_accuracy)
            )
            self.early_stopping.bad_epochs = int(
                state.get("early_stopping_bad_epochs", 0)
            )
            self._restore_rng_state(state)
            self._restore_training_sampler_state(state)
            completed_epoch = int(state["epoch"])
            self._reconcile_history(completed_epoch)
            start_epoch = completed_epoch + 1
            logger.info(
                "resumed fold=%d from epoch=%d best_accuracy=%.2f",
                self.fold,
                start_epoch - 1,
                best_accuracy,
            )

        for epoch in range(start_epoch, self.cfg.epochs + 1):
            started = time.time()
            train_metrics = self._run_epoch(self.train_loader, train=True, epoch=epoch)
            val_metrics = self._run_epoch(self.val_loader, train=False, epoch=epoch)
            self.scheduler.step(val_metrics["loss"])
            lrs = [group["lr"] for group in self.optimizer.param_groups]
            row = self._flat_history_row(epoch, train_metrics, val_metrics, lrs)
            self._append_history(row)
            logger.info(
                "epoch=%03d train_loss=%.4f val_acc=%.2f val_qwk=%.4f "
                "val_mae=%.4f proof_med=%.1f proof_frac=%.4f eps=%.4f time=%.1fs",
                epoch,
                train_metrics["loss"],
                val_metrics["acc"],
                val_metrics["qwk"],
                val_metrics["mae"],
                val_metrics["proof_size_median"],
                val_metrics["proof_fraction_mean"],
                train_metrics["proof_tolerance"],
                time.time() - started,
            )
            is_best = val_metrics["acc"] > best_accuracy
            if is_best:
                best_accuracy = val_metrics["acc"]
            should_stop = self.early_stopping.update(val_metrics["acc"])
            if is_best:
                self._save(epoch, val_metrics, best_accuracy, best=True)
            self._save(epoch, val_metrics, best_accuracy, best=False)
            if should_stop:
                logger.info("early stopping after epoch %d", epoch)
                break

        checkpoint = self._restore_best()
        best_validation = checkpoint["metrics"]
        if not evaluate_test:
            result: dict = {
                "best_epoch": int(checkpoint["epoch"]),
                "test_evaluated": False,
                "best_validation_metrics": best_validation,
            }
            result.update(
                {
                    f"best_val_{key}": value
                    for key, value in best_validation.items()
                    if isinstance(value, (float, int))
                }
            )
            logger.info(
                "test evaluation skipped; best validation acc=%.2f qwk=%.4f mae=%.4f",
                best_validation["acc"],
                best_validation["qwk"],
                best_validation["mae"],
            )
            return result
        test_metrics = self._run_epoch(self.test_loader, train=False, epoch=self.cfg.epochs)
        test_metrics["best_epoch"] = int(checkpoint["epoch"])
        test_metrics["test_evaluated"] = True
        test_metrics["best_validation_metrics"] = best_validation
        test_metrics.update(
            {
                f"best_val_{key}": value
                for key, value in best_validation.items()
                if isinstance(value, (float, int))
            }
        )
        logger.info(
            "test acc=%.2f qwk=%.4f mae=%.4f balanced_acc=%.2f macro_f1=%.4f",
            test_metrics["acc"],
            test_metrics["qwk"],
            test_metrics["mae"],
            test_metrics["balanced_acc"],
            test_metrics["macro_f1"],
        )
        return test_metrics


__all__ = ["MosaicTrainer", "mosaic_implementation_signature"]
