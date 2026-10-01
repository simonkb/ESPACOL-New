"""Objective for the pre-registered ORIGIN acceptance baselines."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch
import torch.nn as nn

from .origin import categorical_nll, ranked_probability_score


def _field(output: object, name: str) -> Any:
    if isinstance(output, Mapping):
        if name not in output:
            raise KeyError(f"baseline output has no {name!r} field")
        return output[name]
    if not hasattr(output, name):
        raise AttributeError(f"baseline output has no {name!r} attribute")
    return getattr(output, name)


class OriginAcceptanceBaselineLoss(nn.Module):
    """NLL + RPS, with a declared Sparse-BagNet activation penalty."""

    def __init__(
        self,
        num_classes: int,
        *,
        baseline_variant: str,
        rps_weight: float = 0.25,
        sparse_l1_weight: float = 0.0,
        sparse_l1_delay_epochs: int = 0,
        class_weights: Sequence[float] | torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        if num_classes < 2:
            raise ValueError("num_classes must be at least two")
        if rps_weight < 0.0 or sparse_l1_weight < 0.0:
            raise ValueError("loss weights must be non-negative")
        if sparse_l1_delay_epochs < 0:
            raise ValueError("sparse_l1_delay_epochs must be non-negative")
        self.num_classes = int(num_classes)
        self.baseline_variant = str(baseline_variant).lower()
        self.rps_weight = float(rps_weight)
        self.sparse_l1_weight = float(sparse_l1_weight)
        self.sparse_l1_delay_epochs = int(sparse_l1_delay_epochs)
        if self.baseline_variant != "sparse_bagnet" and self.sparse_l1_weight:
            raise ValueError("only sparse_bagnet may use sparse_l1_weight")
        if class_weights is None:
            weights = torch.empty(0, dtype=torch.float32)
        else:
            weights = torch.as_tensor(class_weights, dtype=torch.float32).flatten()
            if tuple(weights.shape) != (self.num_classes,):
                raise ValueError(
                    f"class_weights must have shape ({self.num_classes},)"
                )
            if not bool(torch.isfinite(weights).all()) or bool((weights <= 0).any()):
                raise ValueError("class_weights must be finite and positive")
        self.register_buffer("class_weights", weights)

    @property
    def uses_weighted_likelihood(self) -> bool:
        return self.class_weights.numel() > 0

    @property
    def likelihood_component_unweighted(self) -> bool:
        return not self.uses_weighted_likelihood

    @property
    def configured_objective_is_proper(self) -> bool:
        return self.likelihood_component_unweighted and self.sparse_l1_weight == 0.0

    def budget_is_active(self, epoch: int | None) -> bool:
        """Trainer-compatibility name for the activation regularizer gate."""

        if self.sparse_l1_weight == 0.0:
            return False
        if epoch is None:
            return self.sparse_l1_delay_epochs == 0
        return int(epoch) > self.sparse_l1_delay_epochs

    def forward(
        self,
        output: object,
        labels: torch.Tensor,
        *,
        epoch: int | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor | bool]]:
        log_probs = _field(output, "log_class_probs")
        probs = _field(output, "class_probs")
        cumulative = _field(output, "cumulative_probs")
        expected_shape = (log_probs.shape[0], self.num_classes)
        if tuple(log_probs.shape) != expected_shape or tuple(probs.shape) != expected_shape:
            raise ValueError(f"posterior must have shape {expected_shape}")
        if tuple(cumulative.shape) != (log_probs.shape[0], self.num_classes - 1):
            raise ValueError("cumulative posterior has the wrong shape")
        weights = self.class_weights if self.uses_weighted_likelihood else None
        nll = categorical_nll(log_probs, labels, class_weights=weights)
        rps = ranked_probability_score(cumulative, labels)
        active = self.budget_is_active(epoch)
        if active:
            sparse_l1 = _field(output, "sparse_activation_l1")
            if not torch.is_tensor(sparse_l1) or sparse_l1.ndim != 0:
                raise TypeError("sparse_activation_l1 must be a scalar tensor")
            if not bool(torch.isfinite(sparse_l1)) or bool(sparse_l1 < 0):
                raise ValueError("sparse_activation_l1 must be finite and non-negative")
        else:
            sparse_l1 = log_probs.new_zeros(())
        total = nll + self.rps_weight * rps
        if active:
            total = total + self.sparse_l1_weight * sparse_l1
        if not bool(torch.isfinite(total)):
            raise FloatingPointError("non-finite acceptance-baseline loss")
        # ``evidence_budget`` is retained only because OriginTrainer has a
        # fixed diagnostic slot.  The specialized trainer immediately aliases
        # it to ``sparse_activation_l1`` and records its actual semantics.
        return total, {
            "nll": nll.detach(),
            "rps": rps.detach(),
            "evidence_budget": sparse_l1.detach(),
            "sparse_activation_l1": sparse_l1.detach(),
            "budget_active": active,
            "weighted_likelihood": self.uses_weighted_likelihood,
            "likelihood_component_unweighted": self.likelihood_component_unweighted,
            "configured_objective_is_proper": self.configured_objective_is_proper,
        }


__all__ = ["OriginAcceptanceBaselineLoss"]
