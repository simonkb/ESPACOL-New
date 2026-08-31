"""Mathematical core of MOSAIC ordinal proof models.

MOSAIC (Minimum Ordinal Sufficient Attribution by Intervention and Counting)
turns local categorical ordinal states into nested boundary witnesses, scores
those witnesses with an exact truncated Poisson--binomial cardinality law, and
uses a deterministic minimum top-prefix proof as the *only* prediction path.

This file intentionally contains no image encoder or global classifier.  It is
usable with cached local features and, more importantly, keeps the proof layer
independently testable.  Tensor conventions are:

* local logits / state probabilities: ``(N, P, K)``;
* regional boundary witnesses: ``(N, P, K-1)``;
* boundary transitions: ``(N, K-1)``; and
* count distributions: bins ``0, ..., R-1, >=R`` in the last dimension.

All count arithmetic is explicitly promoted to FP32.  Encoder AMP can
therefore be used without evaluating long probability recurrences in FP16.
"""

from dataclasses import dataclass
import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _require_finite(tensor: torch.Tensor, name: str) -> None:
    """Fail at the first corrupted proof-path tensor with useful provenance."""

    finite = torch.isfinite(tensor)
    if bool(finite.all()):
        return
    invalid = ~finite
    nan_count = int(torch.isnan(tensor).sum().detach().cpu())
    posinf_count = int(torch.isposinf(tensor).sum().detach().cpu())
    neginf_count = int(torch.isneginf(tensor).sum().detach().cpu())
    raise FloatingPointError(
        f"{name} contains non-finite values "
        f"(total={int(invalid.sum().detach().cpu())}, nan={nan_count}, "
        f"+inf={posinf_count}, -inf={neginf_count})"
    )


@dataclass
class LocalOrdinalEvidence:
    """Categorical local states and their nested boundary witnesses."""

    state_probabilities: torch.Tensor
    witness_probabilities: torch.Tensor
    log_witness_probabilities: torch.Tensor
    log_nonwitness_probabilities: torch.Tensor


@dataclass
class CardinalityResult:
    """Output of a boundary-wise cardinality circuit."""

    transitions: torch.Tensor
    stop_probabilities: torch.Tensor
    log_stop_probabilities: torch.Tensor
    distributions: torch.Tensor
    log_conditional_low_distributions: torch.Tensor
    log_low_survival: torch.Tensor
    tails: torch.Tensor
    alpha: torch.Tensor
    log_alpha: torch.Tensor


@dataclass
class ProofProjectionResult:
    """A deterministic dual certificate and its replayed circuit values."""

    selected_mask: torch.Tensor
    # Sorted indices use (N, K-1, P), unlike the witness tensor, because the
    # ranking is boundary-specific.
    sorted_indices: torch.Tensor
    proof_size: torch.Tensor
    dense_transition: torch.Tensor
    projected_transition: torch.Tensor
    complement_transition: torch.Tensor
    retained_distribution: torch.Tensor
    complement_distribution: torch.Tensor
    sufficiency_gap: torch.Tensor
    complement_drop: torch.Tensor


@dataclass
class MOSAICOutput:
    """Complete output of the encoder-independent ordinal proof core."""

    local_state_probabilities: torch.Tensor
    witness_probabilities: torch.Tensor
    log_witness_probabilities: torch.Tensor
    log_nonwitness_probabilities: torch.Tensor
    alpha: torch.Tensor
    log_alpha: torch.Tensor
    dense_transitions: torch.Tensor
    dense_stop_probabilities: torch.Tensor
    dense_log_stop_probabilities: torch.Tensor
    transitions: torch.Tensor
    stop_probabilities: torch.Tensor
    log_stop_probabilities: torch.Tensor
    cumulative_probabilities: torch.Tensor
    class_probabilities: torch.Tensor
    expected_grade: torch.Tensor
    predicted_grade: torch.Tensor
    argmax_grade: torch.Tensor
    proof: ProofProjectionResult
    pivotality: Optional[torch.Tensor] = None


def nested_witness_probabilities(
    local_logits: torch.Tensor,
    valid_mask: Optional[torch.Tensor] = None,
) -> LocalOrdinalEvidence:
    """Convert one local categorical state into nested ordinal witnesses.

    ``lambda[..., k] = P(L > k)`` is computed as a reverse cumulative sum of
    the categorical state probabilities.  Hence nesting is structural rather
    than encouraged by a penalty.  Invalid cells are represented coherently as
    the normal state ``(1, 0, ..., 0)`` and have exactly zero witnesses.
    """

    if local_logits.ndim != 3:
        raise ValueError(
            "local_logits must have shape (N, P, K); got "
            f"{tuple(local_logits.shape)}"
        )
    if local_logits.shape[-1] < 2:
        raise ValueError("at least two ordinal states are required")
    _require_finite(local_logits, "MOSAIC local logits")

    # Preserve both sides of every ordinal Bernoulli event in log space.
    # Converting a saturated softmax probability back through log(p) loses
    # the recovery gradient that log_softmax retains for finite logits.
    log_state = torch.log_softmax(local_logits.float(), dim=-1)
    state = log_state.exp()

    if valid_mask is not None:
        if valid_mask.shape != local_logits.shape[:2]:
            raise ValueError(
                "valid_mask must have shape (N, P); got "
                f"{tuple(valid_mask.shape)}"
            )
        valid = valid_mask.to(device=local_logits.device, dtype=torch.bool)
        normal = torch.zeros_like(state)
        normal[..., 0] = 1.0
        state = torch.where(valid.unsqueeze(-1), state, normal)
        log_normal = torch.full_like(log_state, -torch.inf)
        log_normal[..., 0] = 0.0
        log_state = torch.where(valid.unsqueeze(-1), log_state, log_normal)

    # For states 0,...,K-1 this produces [P(L>0), ..., P(L>K-2)].
    log_witnesses = []
    log_nonwitnesses = []
    for boundary in range(local_logits.shape[-1] - 1):
        log_witnesses.append(
            torch.logsumexp(log_state[..., boundary + 1 :], dim=-1)
        )
        log_nonwitnesses.append(
            torch.logsumexp(log_state[..., : boundary + 1], dim=-1)
        )
    log_witness = torch.stack(log_witnesses, dim=-1)
    log_nonwitness = torch.stack(log_nonwitnesses, dim=-1)
    witnesses = log_witness.exp().clamp(0.0, 1.0)
    return LocalOrdinalEvidence(
        state, witnesses, log_witness, log_nonwitness
    )


class LocalOrdinalStateHead(nn.Module):
    """Shared pointwise head for local categorical ordinal states.

    Input is ``(N, P, D)``.  The state-0 bias is initialized so the expected
    number of abnormal cells is approximately ``initial_abnormal_count`` for
    an ``expected_num_cells``-cell field.
    """

    def __init__(
        self,
        input_dim: int,
        num_classes: int = 5,
        expected_num_cells: int = 112 * 112,
        initial_abnormal_count: float = 0.5,
    ) -> None:
        super().__init__()
        if input_dim <= 0:
            raise ValueError("input_dim must be positive")
        if num_classes < 2:
            raise ValueError("num_classes must be at least 2")
        if expected_num_cells <= 0:
            raise ValueError("expected_num_cells must be positive")
        if not (0.0 < initial_abnormal_count < expected_num_cells):
            raise ValueError(
                "initial_abnormal_count must lie strictly between 0 and "
                "expected_num_cells"
            )

        self.input_dim = input_dim
        self.num_classes = num_classes
        self.expected_num_cells = expected_num_cells
        self.initial_abnormal_count = initial_abnormal_count
        self.linear = nn.Linear(input_dim, num_classes)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # A zero final state map makes the requested initial abnormal count
        # true for realistic nonzero pretrained features as well as for a zero
        # test tensor.  Xavier noise across 12,544 cells otherwise multiplies
        # the intended count several-fold before the first update.
        nn.init.zeros_(self.linear.weight)
        with torch.no_grad():
            self.linear.bias.zero_()
            # If all abnormal logits are equal, this makes
            # P(L>0) ~= mu_0/P (the exact logit uses odds, not its small-p
            # approximation).
            abnormal_probability = self.initial_abnormal_count / float(
                self.expected_num_cells
            )
            normal_advantage = math.log(
                (self.num_classes - 1)
                * (1.0 - abnormal_probability)
                / abnormal_probability
            )
            self.linear.bias[0] = normal_advantage

    def logits(self, local_features: torch.Tensor) -> torch.Tensor:
        if local_features.ndim != 3 or local_features.shape[-1] != self.input_dim:
            raise ValueError(
                f"local_features must have shape (N, P, {self.input_dim}); "
                f"got {tuple(local_features.shape)}"
            )
        return self.linear(local_features)

    def forward(
        self,
        local_features: torch.Tensor,
        valid_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, LocalOrdinalEvidence]:
        logits = self.logits(local_features)
        return logits, nested_witness_probabilities(logits, valid_mask)


class TruncatedPoissonBinomial(nn.Module):
    """Exact masses ``0,...,R-1,>=R`` for independent Bernoulli events.

    ``implementation='serial'`` is the simple reference recurrence.
    ``implementation='block_tree'`` performs the same computation using
    vectorized blocks and balanced exact polynomial merges.  The latter lowers
    sequential depth substantially for a large evidence lattice.
    """

    def __init__(
        self,
        max_count: int = 32,
        implementation: str = "block_tree",
        block_size: int = 64,
    ) -> None:
        super().__init__()
        if max_count < 1:
            raise ValueError("max_count must be at least 1")
        if implementation not in {"serial", "block_tree"}:
            raise ValueError("implementation must be 'serial' or 'block_tree'")
        if block_size < 1:
            raise ValueError("block_size must be positive")
        self.max_count = max_count
        self.implementation = implementation
        self.block_size = block_size

    def empty_distribution(
        self, leading_shape: Tuple[int, ...], device: torch.device
    ) -> torch.Tensor:
        result = torch.zeros(
            *leading_shape,
            self.max_count + 1,
            device=device,
            dtype=torch.float32,
        )
        result[..., 0] = 1.0
        return result

    @staticmethod
    def _normalise_distribution(distribution: torch.Tensor) -> torch.Tensor:
        """Project round-off back onto the probability simplex.

        A 112-by-112 lattice combines 12,544 Bernoulli variables.  Even a
        tiny per-operation FP32 mass error becomes material when already
        drifted block masses are multiplied in the merge tree.  Every exact
        recurrence result is non-negative and has unit mass mathematically,
        so enforcing those two invariants is a numerical correction rather
        than a modelling approximation.
        """

        distribution = distribution.float().clamp_min(0.0)
        total = distribution.sum(dim=-1, keepdim=True)
        tiny = torch.finfo(distribution.dtype).tiny
        return distribution / total.clamp_min(tiny)

    def update(self, distribution: torch.Tensor, probability: torch.Tensor) -> torch.Tensor:
        """Apply one exact Bernoulli recurrence update in FP32."""

        if distribution.shape[-1] != self.max_count + 1:
            raise ValueError("distribution has the wrong number of count bins")
        if probability.shape != distribution.shape[:-1]:
            raise ValueError("probability must match distribution leading dimensions")
        distribution = distribution.float()
        probability = probability.float()
        low = distribution[..., : self.max_count]
        e = probability.unsqueeze(-1)
        stay = (1.0 - e) * low
        shifted = torch.cat((torch.zeros_like(low[..., :1]), e * low[..., :-1]), dim=-1)
        new_low = stay + shifted
        overflow = distribution[..., self.max_count] + probability * low[..., -1]
        return self._normalise_distribution(
            torch.cat((new_low, overflow.unsqueeze(-1)), dim=-1)
        )

    def merge(self, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        """Exactly merge two independent truncated count distributions."""

        if left.shape != right.shape or left.shape[-1] != self.max_count + 1:
            raise ValueError("left and right distributions must have identical shapes")
        left = left.float()
        right = right.float()
        r = self.max_count

        # Only low+low combinations can land below R.  An outer product plus
        # scatter-add computes the first R polynomial coefficients in a small
        # number of kernels and remains differentiable.
        outer = left[..., :r].unsqueeze(-1) * right[..., :r].unsqueeze(-2)
        indices = torch.arange(r, device=left.device)
        count_index = indices[:, None] + indices[None, :]
        valid = count_index < r
        values = outer[..., valid]
        bins = count_index[valid]
        expanded_bins = bins.reshape(*((1,) * (values.ndim - 1)), -1).expand_as(values)
        low = torch.zeros(
            *left.shape[:-1], r, device=left.device, dtype=torch.float32
        )
        low = low.scatter_add(-1, expanded_bins, values)

        # Compute overflow from non-negative terms, not as ``total-low``.
        # The latter catastrophically cancels when the true overflow tail is
        # tiny (the normal-image regime that matters most for DR).
        high_low = outer[..., ~valid].sum(dim=-1)
        overflow = (
            left[..., r] * right.sum(dim=-1)
            + left[..., :r].sum(dim=-1) * right[..., r]
            + high_low
        )
        return self._normalise_distribution(
            torch.cat((low, overflow.unsqueeze(-1)), dim=-1)
        )

    def _serial(self, probabilities: torch.Tensor) -> torch.Tensor:
        distribution = self.empty_distribution(
            tuple(probabilities.shape[:-1]), probabilities.device
        )
        for index in range(probabilities.shape[-1]):
            distribution = self.update(distribution, probabilities[..., index])
        return distribution

    def _block_tree(self, probabilities: torch.Tensor) -> torch.Tensor:
        num_events = probabilities.shape[-1]
        if num_events == 0:
            return self.empty_distribution(
                tuple(probabilities.shape[:-1]), probabilities.device
            )

        block_size = min(self.block_size, num_events)
        padding = (-num_events) % block_size
        if padding:
            probabilities = F.pad(probabilities, (0, padding), value=0.0)
        num_blocks = probabilities.shape[-1] // block_size
        blocks = probabilities.reshape(
            *probabilities.shape[:-1], num_blocks, block_size
        )
        distributions = self.empty_distribution(
            tuple(blocks.shape[:-1]), probabilities.device
        )
        for offset in range(block_size):
            distributions = self.update(distributions, blocks[..., offset])

        # Balanced tree.  An unpaired final block is carried unchanged.
        while distributions.shape[-2] > 1:
            count = distributions.shape[-2]
            pair_count = count // 2
            paired = self.merge(
                distributions[..., 0 : 2 * pair_count : 2, :],
                distributions[..., 1 : 2 * pair_count : 2, :],
            )
            if count % 2:
                distributions = torch.cat(
                    (paired, distributions[..., -1:, :]), dim=-2
                )
            else:
                distributions = paired
        return distributions.squeeze(-2)

    def forward(
        self,
        probabilities: torch.Tensor,
        valid_mask: Optional[torch.Tensor] = None,
        implementation: Optional[str] = None,
    ) -> torch.Tensor:
        if probabilities.ndim < 1:
            raise ValueError("probabilities must have at least one dimension")
        # Softmax/cumulative round-off can exceed the closed interval by one
        # ULP.  Projecting to the Bernoulli domain prevents a negative
        # ``1-p`` recurrence coefficient without changing valid inputs.
        probabilities = probabilities.float().clamp(0.0, 1.0)
        if valid_mask is not None:
            if valid_mask.shape != probabilities.shape:
                raise ValueError("valid_mask must have the same shape as probabilities")
            probabilities = probabilities * valid_mask.to(
                device=probabilities.device, dtype=torch.float32
            )
        mode = self.implementation if implementation is None else implementation
        if mode == "serial":
            return self._serial(probabilities)
        if mode == "block_tree":
            return self._block_tree(probabilities)
        raise ValueError("implementation must be 'serial' or 'block_tree'")

    @staticmethod
    def _empty_log_low(
        leading_shape: Tuple[int, ...], max_count: int, device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        log_conditional = torch.full(
            (*leading_shape, max_count),
            -torch.inf,
            device=device,
            dtype=torch.float32,
        )
        log_conditional[..., 0] = 0.0
        log_survival = torch.zeros(
            leading_shape, device=device, dtype=torch.float32
        )
        return log_conditional, log_survival

    @staticmethod
    def _safe_logsumexp(log_values: torch.Tensor, dim: int) -> torch.Tensor:
        """LogSumExp with a constant ``-inf`` result for empty support.

        PyTorch's backward for an all-``-inf`` reduction is undefined.  The
        finite dummy branch is evaluated only for unsupported rows and is
        replaced by a constant, giving the mathematically correct zero
        gradient there.
        """

        supported = (~torch.isneginf(log_values)).any(dim=dim)
        safe_values = torch.where(
            supported.unsqueeze(dim), log_values, torch.zeros_like(log_values)
        )
        reduced = torch.logsumexp(safe_values, dim=dim)
        return torch.where(
            supported, reduced, torch.full_like(reduced, -torch.inf)
        )

    @staticmethod
    def _safe_logaddexp(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        supported = (~torch.isneginf(left)) | (~torch.isneginf(right))
        safe_left = torch.where(supported, left, torch.zeros_like(left))
        safe_right = torch.where(supported, right, torch.zeros_like(right))
        combined = torch.logaddexp(safe_left, safe_right)
        return torch.where(
            supported, combined, torch.full_like(combined, -torch.inf)
        )

    @staticmethod
    def _normalise_log_low(
        log_low: torch.Tensor, log_survival: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Normalize low-count log masses and accumulate their total mass."""

        log_scale = TruncatedPoissonBinomial._safe_logsumexp(log_low, dim=-1)
        finite = torch.isfinite(log_scale)
        safe_scale = torch.where(finite, log_scale, torch.zeros_like(log_scale))
        normalized = log_low - safe_scale.unsqueeze(-1)
        # Conditional probabilities are undefined after exact zero survival.
        # A deterministic delta fallback avoids NaNs; log_survival=-inf keeps
        # every resulting absolute low-count probability exactly zero.
        fallback = torch.full_like(normalized, -torch.inf)
        fallback[..., 0] = 0.0
        normalized = torch.where(finite.unsqueeze(-1), normalized, fallback)
        return normalized, log_survival + log_scale

    def _update_log_low_from_logs(
        self,
        log_conditional: torch.Tensor,
        log_survival: torch.Tensor,
        log_probability: torch.Tensor,
        log_non_probability: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        stay = log_conditional + log_non_probability.float().unsqueeze(-1)
        shifted = torch.cat(
            (
                torch.full_like(log_conditional[..., :1], -torch.inf),
                log_conditional[..., :-1]
                + log_probability.float().unsqueeze(-1),
            ),
            dim=-1,
        )
        return self._normalise_log_low(
            self._safe_logaddexp(stay, shifted), log_survival
        )

    def _update_log_low(
        self,
        log_conditional: torch.Tensor,
        log_survival: torch.Tensor,
        probability: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        probability = probability.float().clamp(0.0, 1.0)
        endpoint = (probability == 0.0) | (probability == 1.0)
        safe_probability = torch.where(
            endpoint, torch.full_like(probability, 0.5), probability
        )
        log_probability = torch.where(
            probability == 0.0,
            torch.full_like(probability, -torch.inf),
            torch.where(
                probability == 1.0,
                torch.zeros_like(probability),
                torch.log(safe_probability),
            ),
        )
        log_non_probability = torch.where(
            probability == 1.0,
            torch.full_like(probability, -torch.inf),
            torch.where(
                probability == 0.0,
                torch.zeros_like(probability),
                torch.log1p(-safe_probability),
            ),
        )
        return self._update_log_low_from_logs(
            log_conditional,
            log_survival,
            log_probability,
            log_non_probability,
        )

    def _merge_log_low(
        self,
        left_log_conditional: torch.Tensor,
        left_log_survival: torch.Tensor,
        right_log_conditional: torch.Tensor,
        right_log_survival: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if left_log_conditional.shape != right_log_conditional.shape:
            raise ValueError("scaled lower-tail distributions must have equal shapes")
        coefficients = []
        for count in range(self.max_count):
            terms = (
                left_log_conditional[..., : count + 1]
                + torch.flip(
                    right_log_conditional[..., : count + 1], dims=(-1,)
                )
            )
            coefficients.append(self._safe_logsumexp(terms, dim=-1))
        log_low = torch.stack(coefficients, dim=-1)
        return self._normalise_log_low(
            log_low, left_log_survival + right_log_survival
        )

    def scaled_log_lower_tail(
        self,
        probabilities: torch.Tensor,
        valid_mask: Optional[torch.Tensor] = None,
        implementation: Optional[str] = None,
        *,
        log_probabilities: Optional[torch.Tensor] = None,
        log_non_probabilities: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return ``log P(C=j | C<R)`` and ``log P(C<R)`` stably."""

        if probabilities.ndim < 1:
            raise ValueError("probabilities must have at least one dimension")
        probabilities = probabilities.float().clamp(0.0, 1.0)
        if (log_probabilities is None) != (log_non_probabilities is None):
            raise ValueError(
                "log_probabilities and log_non_probabilities must be provided together"
            )
        if log_probabilities is not None:
            if (
                log_probabilities.shape != probabilities.shape
                or log_non_probabilities is None
                or log_non_probabilities.shape != probabilities.shape
            ):
                raise ValueError("log Bernoulli inputs must match probabilities")
            log_probabilities = log_probabilities.float()
            log_non_probabilities = log_non_probabilities.float()
            pair_normalizer = torch.logsumexp(
                torch.stack((log_probabilities, log_non_probabilities), dim=-1),
                dim=-1,
            )
            log_probabilities = log_probabilities - pair_normalizer
            log_non_probabilities = log_non_probabilities - pair_normalizer
        if valid_mask is not None:
            if valid_mask.shape != probabilities.shape:
                raise ValueError("valid_mask must have the same shape as probabilities")
            probabilities = probabilities * valid_mask.to(
                device=probabilities.device, dtype=torch.float32
            )
            if log_probabilities is not None:
                valid = valid_mask.to(device=probabilities.device, dtype=torch.bool)
                log_probabilities = torch.where(
                    valid, log_probabilities, torch.full_like(log_probabilities, -torch.inf)
                )
                log_non_probabilities = torch.where(
                    valid, log_non_probabilities, torch.zeros_like(log_non_probabilities)
                )
        mode = self.implementation if implementation is None else implementation
        if mode not in {"serial", "block_tree"}:
            raise ValueError("implementation must be 'serial' or 'block_tree'")

        if mode == "serial":
            log_conditional, log_survival = self._empty_log_low(
                tuple(probabilities.shape[:-1]), self.max_count, probabilities.device
            )
            for index in range(probabilities.shape[-1]):
                if log_probabilities is None:
                    log_conditional, log_survival = self._update_log_low(
                        log_conditional, log_survival, probabilities[..., index]
                    )
                else:
                    log_conditional, log_survival = self._update_log_low_from_logs(
                        log_conditional,
                        log_survival,
                        log_probabilities[..., index],
                        log_non_probabilities[..., index],
                    )
            return log_conditional, log_survival

        num_events = probabilities.shape[-1]
        if num_events == 0:
            log_conditional, log_survival = self._empty_log_low(
                tuple(probabilities.shape[:-1]), self.max_count, probabilities.device
            )
            return log_conditional, log_survival
        block_size = min(self.block_size, num_events)
        padding = (-num_events) % block_size
        if padding:
            probabilities = F.pad(probabilities, (0, padding), value=0.0)
            if log_probabilities is not None:
                log_probabilities = F.pad(
                    log_probabilities, (0, padding), value=-torch.inf
                )
                log_non_probabilities = F.pad(
                    log_non_probabilities, (0, padding), value=0.0
                )
        num_blocks = probabilities.shape[-1] // block_size
        blocks = probabilities.reshape(
            *probabilities.shape[:-1], num_blocks, block_size
        )
        log_probability_blocks = None
        log_non_probability_blocks = None
        if log_probabilities is not None:
            log_probability_blocks = log_probabilities.reshape(
                *log_probabilities.shape[:-1], num_blocks, block_size
            )
            log_non_probability_blocks = log_non_probabilities.reshape(
                *log_non_probabilities.shape[:-1], num_blocks, block_size
            )
        log_conditional, log_survival = self._empty_log_low(
            tuple(blocks.shape[:-1]), self.max_count, probabilities.device
        )
        for offset in range(block_size):
            if log_probability_blocks is None:
                log_conditional, log_survival = self._update_log_low(
                    log_conditional, log_survival, blocks[..., offset]
                )
            else:
                log_conditional, log_survival = self._update_log_low_from_logs(
                    log_conditional,
                    log_survival,
                    log_probability_blocks[..., offset],
                    log_non_probability_blocks[..., offset],
                )
        while log_conditional.shape[-2] > 1:
            count = log_conditional.shape[-2]
            pair_count = count // 2
            paired_log, paired_survival = self._merge_log_low(
                log_conditional[..., 0 : 2 * pair_count : 2, :],
                log_survival[..., 0 : 2 * pair_count : 2],
                log_conditional[..., 1 : 2 * pair_count : 2, :],
                log_survival[..., 1 : 2 * pair_count : 2],
            )
            if count % 2:
                log_conditional = torch.cat(
                    (paired_log, log_conditional[..., -1:, :]), dim=-2
                )
                log_survival = torch.cat(
                    (paired_survival, log_survival[..., -1:]), dim=-1
                )
            else:
                log_conditional, log_survival = paired_log, paired_survival
        return log_conditional.squeeze(-2), log_survival.squeeze(-1)

    @staticmethod
    def log_stops_from_scaled_lower_tail(
        log_conditional_low: torch.Tensor, log_survival: torch.Tensor
    ) -> torch.Tensor:
        """Return ``log P(C<r)`` for ``r=1,...,R``."""

        if log_conditional_low.shape[-1] < 1:
            raise ValueError("conditional lower tail must contain at least one bin")
        if log_conditional_low.shape[:-1] != log_survival.shape:
            raise ValueError("conditional lower tail and log survival shapes differ")
        return (
            torch.logcumsumexp(log_conditional_low.float(), dim=-1)
            + log_survival.unsqueeze(-1)
        )

    @staticmethod
    def tails(distribution: torch.Tensor) -> torch.Tensor:
        """Return exact tail probabilities ``P(C>=1),...,P(C>=R)``."""

        if distribution.shape[-1] < 2:
            raise ValueError("distribution must include low and overflow bins")
        # Reverse cumsum avoids catastrophic cancellation from 1 - cumsum.
        tails = torch.flip(
            torch.cumsum(torch.flip(distribution[..., 1:], dims=(-1,)), dim=-1),
            dims=(-1,),
        )
        return tails.clamp(0.0, 1.0)

    @staticmethod
    def stops(distribution: torch.Tensor) -> torch.Tensor:
        """Return ``P(C<1),...,P(C<R)`` plus ``P(C<R)`` for the overflow bin.

        More precisely, element ``r-1`` is ``P(C<r)`` for
        ``r=1,...,R``.  Computing this lower tail directly preserves a usable
        grade-0 likelihood when ``P(C>=r)`` rounds to one.
        """

        if distribution.shape[-1] < 2:
            raise ValueError("distribution must include low and overflow bins")
        return torch.cumsum(distribution[..., :-1], dim=-1).clamp(0.0, 1.0)


class OrdinalCardinalityCircuit(nn.Module):
    """Boundary-specific learned mixtures of exact cardinality tails."""

    def __init__(
        self,
        num_boundaries: int = 4,
        max_count: int = 32,
        implementation: str = "block_tree",
        block_size: int = 64,
        alpha_init_count: Optional[int] = 1,
        alpha_init_strength: float = 2.0,
    ) -> None:
        super().__init__()
        if num_boundaries < 1:
            raise ValueError("num_boundaries must be positive")
        self.num_boundaries = num_boundaries
        self.max_count = max_count
        self.count = TruncatedPoissonBinomial(
            max_count=max_count,
            implementation=implementation,
            block_size=block_size,
        )
        self.alpha_logits = nn.Parameter(torch.zeros(num_boundaries, max_count))
        if alpha_init_count is not None:
            if not (1 <= alpha_init_count <= max_count):
                raise ValueError("alpha_init_count must lie in [1, max_count]")
            with torch.no_grad():
                self.alpha_logits[:, alpha_init_count - 1] = alpha_init_strength

    @property
    def alpha(self) -> torch.Tensor:
        return torch.softmax(self.alpha_logits.float(), dim=-1)

    @property
    def log_alpha(self) -> torch.Tensor:
        return torch.log_softmax(self.alpha_logits.float(), dim=-1)

    @staticmethod
    def score_distributions(
        distributions: torch.Tensor, alpha: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        tails = TruncatedPoissonBinomial.tails(distributions)
        stops = TruncatedPoissonBinomial.stops(distributions)
        if alpha.shape[-1] != tails.shape[-1]:
            raise ValueError("alpha and distribution tail dimensions differ")
        while alpha.ndim < tails.ndim:
            alpha = alpha.unsqueeze(0)
        alpha = alpha.float().clamp_min(0.0)
        alpha = alpha / alpha.sum(dim=-1, keepdim=True).clamp_min(
            torch.finfo(alpha.dtype).tiny
        )
        continuation = (tails * alpha).sum(dim=-1)
        stop = (stops * alpha).sum(dim=-1)
        # Both sides are evaluated directly.  Normalising the pair preserves
        # tiny stop probabilities that would disappear in ``1-continuation``.
        pair_total = (continuation + stop).clamp_min(
            torch.finfo(continuation.dtype).tiny
        )
        continuation = (continuation / pair_total).clamp(0.0, 1.0)
        stop = (stop / pair_total).clamp(0.0, 1.0)
        return continuation, stop, tails

    @staticmethod
    def score_log_stops(
        log_conditional_low: torch.Tensor,
        log_low_survival: torch.Tensor,
        alpha: torch.Tensor,
        log_alpha: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Mix exact log lower tails without returning to probability space."""

        log_stops = TruncatedPoissonBinomial.log_stops_from_scaled_lower_tail(
            log_conditional_low, log_low_survival
        )
        if log_alpha is None:
            alpha = alpha.float().clamp_min(0.0)
            alpha = alpha / alpha.sum(dim=-1, keepdim=True).clamp_min(
                torch.finfo(alpha.dtype).tiny
            )
            # Preserve exact zero mixture weights without creating a
            # ``log(0)`` autograd singularity for probability-space callers.
            tiny = torch.finfo(alpha.dtype).tiny
            log_alpha = torch.where(
                alpha > 0.0,
                torch.log(alpha.clamp_min(tiny)),
                torch.full_like(alpha, -torch.inf),
            )
        else:
            log_alpha = log_alpha.float()
            log_alpha = log_alpha - torch.logsumexp(
                log_alpha, dim=-1, keepdim=True
            )
        while log_alpha.ndim < log_stops.ndim:
            log_alpha = log_alpha.unsqueeze(0)
        # An exact zero stop is represented by an all-``-inf`` mixture row.
        # Its forward value is correctly ``-inf``, but torch.logsumexp has an
        # undefined backward (0/0) on that row.  The guarded reduction keeps
        # the exact value and supplies the mathematically correct zero
        # derivative for unsupported rows.
        return TruncatedPoissonBinomial._safe_logsumexp(
            log_alpha + log_stops, dim=-1
        )

    def forward(
        self,
        witnesses: torch.Tensor,
        valid_mask: Optional[torch.Tensor] = None,
        *,
        log_witnesses: Optional[torch.Tensor] = None,
        log_nonwitnesses: Optional[torch.Tensor] = None,
    ) -> CardinalityResult:
        if witnesses.ndim != 3:
            raise ValueError("witnesses must have shape (N, P, K-1)")
        if witnesses.shape[-1] != self.num_boundaries:
            raise ValueError(
                f"expected {self.num_boundaries} boundaries; got "
                f"{witnesses.shape[-1]}"
            )
        if valid_mask is not None and valid_mask.shape != witnesses.shape[:2]:
            raise ValueError("valid_mask must have shape (N, P)")
        if (log_witnesses is None) != (log_nonwitnesses is None):
            raise ValueError("both log witness tensors must be provided together")
        if log_witnesses is not None and (
            log_witnesses.shape != witnesses.shape
            or log_nonwitnesses is None
            or log_nonwitnesses.shape != witnesses.shape
        ):
            raise ValueError("log witness tensors must match witnesses")
        _require_finite(self.alpha_logits, "MOSAIC cardinality alpha logits")

        probabilities = witnesses.permute(0, 2, 1).float()
        count_mask = None
        if valid_mask is not None:
            count_mask = valid_mask[:, None, :].expand_as(probabilities)
        distributions = self.count(probabilities, count_mask)
        log_probabilities = None
        log_non_probabilities = None
        if log_witnesses is not None:
            log_probabilities = log_witnesses.permute(0, 2, 1).float()
            log_non_probabilities = log_nonwitnesses.permute(0, 2, 1).float()
        log_conditional_low, log_low_survival = self.count.scaled_log_lower_tail(
            probabilities,
            count_mask,
            log_probabilities=log_probabilities,
            log_non_probabilities=log_non_probabilities,
        )
        alpha = self.alpha
        log_alpha = self.log_alpha
        transitions, stops, tails = self.score_distributions(distributions, alpha)
        log_stops = self.score_log_stops(
            log_conditional_low, log_low_survival, alpha, log_alpha
        )
        return CardinalityResult(
            transitions,
            stops,
            log_stops,
            distributions,
            log_conditional_low,
            log_low_survival,
            tails,
            alpha,
            log_alpha,
        )


class DualProofProjection(nn.Module):
    """Minimum top-prefix proof under sufficiency and complement suppression.

    Proof membership is selected under ``no_grad`` and treated as piecewise
    constant.  The retained circuit is then replayed from the original witness
    tensors, so gradients flow through all selected witness values and through
    the learned cardinality mixture.
    """

    def __init__(
        self,
        num_boundaries: int = 4,
        max_count: int = 32,
        sufficiency_tolerance: float = 0.02,
        complement_suppression: float = 0.5,
        implementation: str = "block_tree",
        block_size: int = 64,
        comparison_atol: float = 1e-7,
    ) -> None:
        super().__init__()
        if sufficiency_tolerance < 0:
            raise ValueError("sufficiency_tolerance must be non-negative")
        if not (0.0 <= complement_suppression <= 1.0):
            raise ValueError("complement_suppression must lie in [0, 1]")
        self.num_boundaries = num_boundaries
        self.max_count = max_count
        self.sufficiency_tolerance = sufficiency_tolerance
        self.complement_suppression = complement_suppression
        self.comparison_atol = comparison_atol
        self.count = TruncatedPoissonBinomial(
            max_count=max_count,
            implementation=implementation,
            block_size=block_size,
        )

    @staticmethod
    def _score(distribution: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
        continuation, _stop, _tails = OrdinalCardinalityCircuit.score_distributions(
            distribution, alpha
        )
        return continuation

    @staticmethod
    def _stop_score(distribution: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
        _continuation, stop, _tails = OrdinalCardinalityCircuit.score_distributions(
            distribution, alpha
        )
        return stop

    def _minimum_prefix_sizes(
        self,
        sorted_probabilities: torch.Tensor,
        alpha: torch.Tensor,
        dense_score: torch.Tensor,
    ) -> torch.Tensor:
        """Block-prefix scan plus exact within-block refinement.

        Shapes are ``(L, P)``, ``(L, R)``, and ``(L,)`` respectively, where
        ``L=N*(K-1)``.  This method is called only under ``no_grad``.
        """

        num_rows, num_events = sorted_probabilities.shape
        if num_events == 0:
            return torch.zeros(num_rows, device=sorted_probabilities.device, dtype=torch.long)

        block_size = min(self.count.block_size, num_events)
        padding = (-num_events) % block_size
        padded = sorted_probabilities
        if padding:
            padded = F.pad(padded, (0, padding), value=0.0)
        num_blocks = padded.shape[-1] // block_size
        blocks = padded.reshape(num_rows, num_blocks, block_size)

        # Each block distribution is parallel over rows and blocks.
        block_distribution = self.count.empty_distribution(
            (num_rows, num_blocks), padded.device
        )
        for offset in range(block_size):
            block_distribution = self.count.update(
                block_distribution, blocks[..., offset]
            )

        empty = self.count.empty_distribution((num_rows,), padded.device)
        prefixes = [empty]
        for block in range(num_blocks):
            prefixes.append(self.count.merge(prefixes[-1], block_distribution[:, block]))
        suffixes = [empty for _ in range(num_blocks + 1)]
        suffixes[num_blocks] = empty
        for block in range(num_blocks - 1, -1, -1):
            suffixes[block] = self.count.merge(
                block_distribution[:, block], suffixes[block + 1]
            )

        prefix_stack = torch.stack(prefixes, dim=1)
        suffix_stack = torch.stack(suffixes, dim=1)
        retained_scores = self._score(prefix_stack, alpha[:, None, :])
        complement_scores = self._score(suffix_stack, alpha[:, None, :])

        # These endpoints are identities, not independently approximated
        # computations.  Pinning them prevents a tiny merge-order discrepancy
        # from incorrectly rejecting the mathematically valid empty proof when
        # the dense transition lies inside the sufficiency tolerance.
        retained_scores[:, 0] = 0.0
        complement_scores[:, 0] = dense_score
        retained_scores[:, -1] = dense_score
        complement_scores[:, -1] = 0.0

        target = torch.clamp(
            dense_score - self.sufficiency_tolerance, min=0.0
        )
        sufficient = retained_scores + self.comparison_atol >= target[:, None]
        suppressed = (
            dense_score[:, None] - complement_scores + self.comparison_atol
            >= self.complement_suppression * target[:, None]
        )
        feasible = sufficient & suppressed
        # m=P is always feasible mathematically.  The fallback is defensive
        # against unusual NaNs rather than an alternative rule.
        has_feasible = feasible.any(dim=1)
        first_boundary = feasible.to(torch.int64).argmax(dim=1)
        first_boundary = torch.where(
            has_feasible,
            first_boundary,
            torch.full_like(first_boundary, num_blocks),
        )

        zero_proof = first_boundary == 0
        selected_block = torch.clamp(first_boundary - 1, min=0)
        row_index = torch.arange(num_rows, device=padded.device)
        chosen_values = blocks[row_index, selected_block]
        outer_prefix = prefix_stack[row_index, selected_block]
        outer_suffix = suffix_stack[row_index, selected_block + 1]

        within_prefixes = [empty]
        for offset in range(block_size):
            within_prefixes.append(
                self.count.update(within_prefixes[-1], chosen_values[:, offset])
            )
        within_suffixes = [empty for _ in range(block_size + 1)]
        within_suffixes[block_size] = empty
        for offset in range(block_size - 1, -1, -1):
            within_suffixes[offset] = self.count.update(
                within_suffixes[offset + 1], chosen_values[:, offset]
            )

        fine_feasible = []
        for kept in range(1, block_size + 1):
            retained_distribution = self.count.merge(
                outer_prefix, within_prefixes[kept]
            )
            complement_distribution = self.count.merge(
                within_suffixes[kept], outer_suffix
            )
            retained = self._score(retained_distribution, alpha)
            complement = self._score(complement_distribution, alpha)
            fine_feasible.append(
                (retained + self.comparison_atol >= target)
                & (
                    dense_score - complement + self.comparison_atol
                    >= self.complement_suppression * target
                )
            )
        fine_feasible_tensor = torch.stack(fine_feasible, dim=1)
        has_fine = fine_feasible_tensor.any(dim=1)
        within_size = fine_feasible_tensor.to(torch.int64).argmax(dim=1) + 1
        within_size = torch.where(
            has_fine,
            within_size,
            torch.full_like(within_size, block_size),
        )
        proof_size = selected_block * block_size + within_size
        proof_size = torch.clamp(proof_size, max=num_events)
        return torch.where(zero_proof, torch.zeros_like(proof_size), proof_size)

    def forward(
        self,
        witnesses: torch.Tensor,
        alpha: torch.Tensor,
        valid_mask: Optional[torch.Tensor] = None,
        dense_result: Optional[CardinalityResult] = None,
    ) -> ProofProjectionResult:
        if witnesses.ndim != 3:
            raise ValueError("witnesses must have shape (N, P, K-1)")
        n, p, boundaries = witnesses.shape
        if boundaries != self.num_boundaries:
            raise ValueError("unexpected number of ordinal boundaries")
        if alpha.shape != (boundaries, self.max_count):
            raise ValueError(
                f"alpha must have shape {(boundaries, self.max_count)}"
            )
        if valid_mask is None:
            valid_mask = torch.ones(n, p, device=witnesses.device, dtype=torch.bool)
        elif valid_mask.shape != (n, p):
            raise ValueError("valid_mask must have shape (N, P)")
        else:
            valid_mask = valid_mask.to(device=witnesses.device, dtype=torch.bool)

        flat = witnesses.permute(0, 2, 1).reshape(n * boundaries, p).float()
        flat_valid = (
            valid_mask[:, None, :]
            .expand(n, boundaries, p)
            .reshape(n * boundaries, p)
        )
        flat = flat * flat_valid.float()
        alpha_flat = (
            alpha[None, :, :]
            .expand(n, boundaries, self.max_count)
            .reshape(n * boundaries, self.max_count)
            .float()
        )

        if dense_result is None:
            dense_distribution_flat = self.count(flat)
            dense_score_flat = self._score(dense_distribution_flat, alpha_flat)
        else:
            expected_distribution_shape = (n, boundaries, self.max_count + 1)
            if dense_result.distributions.shape != expected_distribution_shape:
                raise ValueError(
                    "dense_result distributions must have shape "
                    f"{expected_distribution_shape}"
                )
            if dense_result.transitions.shape != (n, boundaries):
                raise ValueError("dense_result transitions have an incompatible shape")
            dense_distribution_flat = dense_result.distributions.reshape(
                n * boundaries, self.max_count + 1
            )
            dense_score_flat = dense_result.transitions.reshape(n * boundaries)

        with torch.no_grad():
            sortable = flat.detach().masked_fill(~flat_valid, -1.0)
            try:
                sorted_indices_flat = torch.argsort(
                    sortable, dim=-1, descending=True, stable=True
                )
            except TypeError:  # pragma: no cover - older PyTorch fallback
                sorted_indices_flat = torch.argsort(
                    sortable, dim=-1, descending=True
                )
            sorted_probabilities = torch.gather(flat.detach(), 1, sorted_indices_flat)
            proof_size_flat = self._minimum_prefix_sizes(
                sorted_probabilities, alpha_flat.detach(), dense_score_flat.detach()
            )
            valid_count = flat_valid.sum(dim=-1)
            proof_size_flat = torch.minimum(proof_size_flat, valid_count)
            ranks = torch.arange(p, device=witnesses.device)[None, :]
            selected_sorted = ranks < proof_size_flat[:, None]
            selected_flat = torch.zeros_like(selected_sorted).scatter(
                1, sorted_indices_flat, selected_sorted
            )
            selected_flat &= flat_valid

        retained_flat = flat * selected_flat.float()
        complement_flat = flat * ((~selected_flat) & flat_valid).float()
        retained_distribution_flat = self.count(retained_flat)
        complement_distribution_flat = self.count(complement_flat)
        projected_flat = self._score(retained_distribution_flat, alpha_flat)
        complement_score_flat = self._score(complement_distribution_flat, alpha_flat)

        def nb(tensor: torch.Tensor) -> torch.Tensor:
            return tensor.reshape(n, boundaries, *tensor.shape[1:])

        selected_mask = selected_flat.reshape(n, boundaries, p).permute(0, 2, 1)
        dense_transition = dense_score_flat.reshape(n, boundaries)
        projected_transition = projected_flat.reshape(n, boundaries)
        complement_transition = complement_score_flat.reshape(n, boundaries)
        target_gap = dense_transition - projected_transition
        complement_drop = dense_transition - complement_transition
        return ProofProjectionResult(
            selected_mask=selected_mask,
            sorted_indices=sorted_indices_flat.reshape(n, boundaries, p),
            proof_size=proof_size_flat.reshape(n, boundaries),
            dense_transition=dense_transition,
            projected_transition=projected_transition,
            complement_transition=complement_transition,
            retained_distribution=nb(retained_distribution_flat),
            complement_distribution=nb(complement_distribution_flat),
            sufficiency_gap=target_gap,
            complement_drop=complement_drop,
        )


def continuation_probabilities(
    transitions: torch.Tensor,
    stop_probabilities: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Convert conditional continuation transitions to cumulative/classes.

    For ``B=K-1`` transitions, returns cumulative ``P(Y>k)`` with shape
    ``(..., B)`` and a normalized class distribution with shape ``(..., K)``.
    """

    if transitions.ndim < 1 or transitions.shape[-1] < 1:
        raise ValueError("at least one continuation transition is required")
    transitions = transitions.float().clamp(0.0, 1.0)
    if stop_probabilities is None:
        stops = 1.0 - transitions
    else:
        if stop_probabilities.shape != transitions.shape:
            raise ValueError("stop_probabilities must match transitions")
        stops = stop_probabilities.float().clamp_min(0.0)
        pair_total = (transitions + stops).clamp_min(
            torch.finfo(transitions.dtype).tiny
        )
        transitions = transitions / pair_total
        stops = stops / pair_total
    cumulative = torch.cumprod(transitions, dim=-1)
    first = stops[..., :1]
    if transitions.shape[-1] > 1:
        middle = cumulative[..., :-1] * stops[..., 1:]
        classes = torch.cat((first, middle, cumulative[..., -1:]), dim=-1)
    else:
        classes = torch.cat((first, cumulative), dim=-1)
    classes = classes.clamp_min(0.0)
    classes = classes / classes.sum(dim=-1, keepdim=True).clamp_min(
        torch.finfo(classes.dtype).tiny
    )
    return cumulative, classes


def fixed_proof_pivotality(
    witnesses: torch.Tensor,
    proof: ProofProjectionResult,
    alpha: torch.Tensor,
    max_count: int,
) -> torch.Tensor:
    """Exact fixed-proof effect of removing each selected witness.

    Returns ``delta`` with shape ``(N, P, K-1)``.  Unselected cells are zero.
    This computes

    ``lambda_i * sum_r alpha_r P(C_without_i = r-1)``

    without division or deconvolution, so probabilities at exactly 0 or 1 are
    handled safely.
    """

    n, p, boundaries = witnesses.shape
    if alpha.shape != (boundaries, max_count):
        raise ValueError("alpha has an incompatible shape")
    count = TruncatedPoissonBinomial(max_count=max_count, implementation="serial")
    flat = witnesses.permute(0, 2, 1).reshape(n * boundaries, p).float()
    sorted_indices = proof.sorted_indices.reshape(n * boundaries, p)
    sizes = proof.proof_size.reshape(n * boundaries)
    max_size = int(sizes.max().item()) if sizes.numel() else 0
    if max_size == 0:
        return torch.zeros_like(witnesses, dtype=torch.float32)

    selected_values = torch.gather(flat, 1, sorted_indices[:, :max_size])
    active = (
        torch.arange(max_size, device=witnesses.device)[None, :] < sizes[:, None]
    )
    selected_values = selected_values * active.float()
    rows = n * boundaries
    empty = count.empty_distribution((rows,), witnesses.device)

    prefixes = [empty]
    for index in range(max_size):
        prefixes.append(count.update(prefixes[-1], selected_values[:, index]))
    suffixes = [empty for _ in range(max_size + 1)]
    suffixes[max_size] = empty
    for index in range(max_size - 1, -1, -1):
        suffixes[index] = count.update(
            suffixes[index + 1], selected_values[:, index]
        )

    alpha_flat = (
        alpha[None, :, :]
        .expand(n, boundaries, max_count)
        .reshape(rows, max_count)
        .float()
    )
    # Merge every prefix/suffix pair in one batched operation.  Keeping the
    # recurrence exact while avoiding one R-by-R GPU merge launch per selected
    # cell is important when an early-training certificate is still large.
    prefix_stack = torch.stack(prefixes[:-1], dim=1)
    suffix_stack = torch.stack(suffixes[1:], dim=1)
    derivative_chunks = []
    merge_chunk_size = 256
    for start in range(0, max_size, merge_chunk_size):
        stop = min(max_size, start + merge_chunk_size)
        without_i = count.merge(
            prefix_stack[:, start:stop],
            suffix_stack[:, start:stop],
        )
        derivative_chunks.append(
            (
                alpha_flat[:, None, :]
                * without_i[..., :max_count]
            ).sum(dim=-1)
        )
    derivative = torch.cat(derivative_chunks, dim=1)
    sorted_delta = selected_values * derivative * active.float()
    delta_flat = torch.zeros_like(flat).scatter(
        1, sorted_indices[:, :max_size], sorted_delta
    )
    return delta_flat.reshape(n, boundaries, p).permute(0, 2, 1)


class MOSAICOrdinalCore(nn.Module):
    """Encoder-independent proof-exclusive ordinal classifier.

    Call :meth:`forward` with local categorical logits.  There is deliberately
    no argument for a pooled image feature and no residual/global output path.
    """

    def __init__(
        self,
        num_classes: int = 5,
        max_count: int = 32,
        sufficiency_tolerance: float = 0.02,
        complement_suppression: float = 0.5,
        implementation: str = "block_tree",
        block_size: int = 64,
        alpha_init_count: Optional[int] = 1,
    ) -> None:
        super().__init__()
        if num_classes < 2:
            raise ValueError("num_classes must be at least 2")
        self.num_classes = num_classes
        self.num_boundaries = num_classes - 1
        self.max_count = max_count
        self.circuit = OrdinalCardinalityCircuit(
            num_boundaries=self.num_boundaries,
            max_count=max_count,
            implementation=implementation,
            block_size=block_size,
            alpha_init_count=alpha_init_count,
        )
        self.projector = DualProofProjection(
            num_boundaries=self.num_boundaries,
            max_count=max_count,
            sufficiency_tolerance=sufficiency_tolerance,
            complement_suppression=complement_suppression,
            implementation=implementation,
            block_size=block_size,
        )

    def forward(
        self,
        local_logits: torch.Tensor,
        valid_mask: Optional[torch.Tensor] = None,
        project: bool = True,
        return_pivotality: bool = False,
    ) -> MOSAICOutput:
        if local_logits.shape[-1] != self.num_classes:
            raise ValueError(
                f"expected {self.num_classes} local states, got "
                f"{local_logits.shape[-1]}"
            )
        evidence = nested_witness_probabilities(local_logits, valid_mask)
        dense = self.circuit(
            evidence.witness_probabilities,
            valid_mask,
            log_witnesses=evidence.log_witness_probabilities,
            log_nonwitnesses=evidence.log_nonwitness_probabilities,
        )

        if project:
            proof = self.projector(
                evidence.witness_probabilities,
                dense.alpha,
                valid_mask=valid_mask,
                dense_result=dense,
            )
        else:
            n, p, boundaries = evidence.witness_probabilities.shape
            if valid_mask is None:
                selected = torch.ones(
                    n, p, boundaries, device=local_logits.device, dtype=torch.bool
                )
            else:
                selected = valid_mask[:, :, None].expand(n, p, boundaries).bool()
            complement_probabilities = evidence.witness_probabilities * (~selected).float()
            complement = self.circuit(complement_probabilities)
            sort_values = evidence.witness_probabilities.permute(0, 2, 1)
            if valid_mask is not None:
                sort_values = sort_values.masked_fill(
                    ~valid_mask[:, None, :].bool(), -1.0
                )
            try:
                sorted_indices = torch.argsort(
                    sort_values,
                    dim=-1,
                    descending=True,
                    stable=True,
                )
            except TypeError:  # pragma: no cover
                sorted_indices = torch.argsort(
                    sort_values,
                    dim=-1,
                    descending=True,
                )
            proof_size = selected.sum(dim=1)
            proof = ProofProjectionResult(
                selected_mask=selected,
                sorted_indices=sorted_indices,
                proof_size=proof_size,
                dense_transition=dense.transitions,
                projected_transition=dense.transitions,
                complement_transition=complement.transitions,
                retained_distribution=dense.distributions,
                complement_distribution=complement.distributions,
                sufficiency_gap=torch.zeros_like(dense.transitions),
                complement_drop=dense.transitions - complement.transitions,
            )

        projected_stops = self.projector._stop_score(
            proof.retained_distribution,
            dense.alpha,
        )
        if project:
            projected_probabilities = (
                evidence.witness_probabilities * proof.selected_mask.float()
            ).permute(0, 2, 1)
            selected = proof.selected_mask.permute(0, 2, 1)
            projected_log_probabilities = torch.where(
                selected,
                evidence.log_witness_probabilities.permute(0, 2, 1),
                torch.full_like(projected_probabilities, -torch.inf),
            )
            projected_log_non_probabilities = torch.where(
                selected,
                evidence.log_nonwitness_probabilities.permute(0, 2, 1),
                torch.zeros_like(projected_probabilities),
            )
            projected_log_conditional_low, projected_log_survival = (
                self.circuit.count.scaled_log_lower_tail(
                    projected_probabilities,
                    log_probabilities=projected_log_probabilities,
                    log_non_probabilities=projected_log_non_probabilities,
                )
            )
            projected_log_stops = self.circuit.score_log_stops(
                projected_log_conditional_low,
                projected_log_survival,
                dense.alpha,
                dense.log_alpha,
            )
        else:
            projected_log_stops = dense.log_stop_probabilities
        cumulative, classes = continuation_probabilities(
            proof.projected_transition,
            projected_stops,
        )
        pivotality = None
        if return_pivotality:
            pivotality = fixed_proof_pivotality(
                evidence.witness_probabilities,
                proof,
                dense.alpha,
                self.max_count,
            )
        class_axis = torch.arange(
            self.num_classes, device=classes.device, dtype=classes.dtype
        )
        expected = (classes * class_axis).sum(dim=-1)
        predicted = expected.round().long().clamp(0, self.num_classes - 1)
        return MOSAICOutput(
            local_state_probabilities=evidence.state_probabilities,
            witness_probabilities=evidence.witness_probabilities,
            log_witness_probabilities=evidence.log_witness_probabilities,
            log_nonwitness_probabilities=evidence.log_nonwitness_probabilities,
            alpha=dense.alpha,
            log_alpha=dense.log_alpha,
            dense_transitions=dense.transitions,
            dense_stop_probabilities=dense.stop_probabilities,
            dense_log_stop_probabilities=dense.log_stop_probabilities,
            transitions=proof.projected_transition,
            stop_probabilities=projected_stops,
            log_stop_probabilities=projected_log_stops,
            cumulative_probabilities=cumulative,
            class_probabilities=classes,
            expected_grade=expected,
            predicted_grade=predicted,
            argmax_grade=classes.argmax(dim=-1),
            proof=proof,
            pivotality=pivotality,
        )


class MOSAICProofHead(nn.Module):
    """Pointwise local-state head followed by :class:`MOSAICOrdinalCore`."""

    def __init__(
        self,
        input_dim: int,
        num_classes: int = 5,
        expected_num_cells: int = 112 * 112,
        initial_abnormal_count: float = 0.5,
        max_count: int = 32,
        sufficiency_tolerance: float = 0.02,
        complement_suppression: float = 0.5,
        implementation: str = "block_tree",
        block_size: int = 64,
        alpha_init_count: Optional[int] = 1,
    ) -> None:
        super().__init__()
        self.local_state_head = LocalOrdinalStateHead(
            input_dim=input_dim,
            num_classes=num_classes,
            expected_num_cells=expected_num_cells,
            initial_abnormal_count=initial_abnormal_count,
        )
        self.ordinal_core = MOSAICOrdinalCore(
            num_classes=num_classes,
            max_count=max_count,
            sufficiency_tolerance=sufficiency_tolerance,
            complement_suppression=complement_suppression,
            implementation=implementation,
            block_size=block_size,
            alpha_init_count=alpha_init_count,
        )

    def forward(
        self,
        local_features: torch.Tensor,
        valid_mask: Optional[torch.Tensor] = None,
        project: bool = True,
        return_pivotality: bool = False,
    ) -> MOSAICOutput:
        # The core computes probabilities once; avoid an otherwise redundant
        # softmax in LocalOrdinalStateHead.forward.  The local-state reduction
        # spans the complete evidence lattice, so both its Linear backward and
        # the exact count circuit must remain FP32 even when the image encoder
        # runs under autocast.
        _require_finite(local_features, "MOSAIC local features")
        with torch.autocast(device_type=local_features.device.type, enabled=False):
            local_logits = self.local_state_head.logits(local_features.float())
            return self.ordinal_core(
                local_logits,
                valid_mask=valid_mask,
                project=project,
                return_pivotality=return_pivotality,
            )


# A concise alias for downstream experiment code.
MOSAICHead = MOSAICProofHead


__all__ = [
    "CardinalityResult",
    "DualProofProjection",
    "LocalOrdinalEvidence",
    "LocalOrdinalStateHead",
    "MOSAICHead",
    "MOSAICOrdinalCore",
    "MOSAICOutput",
    "MOSAICProofHead",
    "OrdinalCardinalityCircuit",
    "ProofProjectionResult",
    "TruncatedPoissonBinomial",
    "continuation_probabilities",
    "fixed_proof_pivotality",
    "nested_witness_probabilities",
]
