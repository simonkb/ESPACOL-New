#!/usr/bin/env python3
"""State the exact distinction between ORIGIN's two ledger decoders.

The audit is intentionally adversarial to an over-broad CTMC claim.  Both the
pure-birth and sequential-hazard decoders are valid, monotone, saturated
fixed-horizon ordinal parameterizations.  The CTMC-specific contract is the
shared-clock Markov semigroup, not exact ledger replay.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Sequence

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models.origin import decode_pure_birth_rates, fit_pure_birth_rates, pure_birth_generator


def sequential_posterior(rates: torch.Tensor, horizon: float = 1.0) -> torch.Tensor:
    rates = torch.as_tensor(rates, dtype=torch.float64)
    continuation = -torch.expm1(-rates * float(horizon))
    tails = continuation.cumprod(-1)
    return torch.cat((1.0 - tails[..., :1], tails[..., :-1] - tails[..., 1:], tails[..., -1:]), -1)


def sequential_transition(rates: torch.Tensor, horizon: float = 1.0) -> torch.Tensor:
    """Natural restartable sequential transition kernel from every state."""

    rates = torch.as_tensor(rates, dtype=torch.float64)
    if rates.ndim != 1:
        raise ValueError("rates must be one-dimensional")
    classes = rates.numel() + 1
    matrix = torch.zeros(classes, classes, dtype=torch.float64)
    matrix[-1, -1] = 1.0
    for initial in range(classes - 1):
        posterior = sequential_posterior(rates[initial:], horizon)
        matrix[initial, initial:] = posterior
    return matrix


def sequential_inverse(class_probs: torch.Tensor) -> torch.Tensor:
    probabilities = torch.as_tensor(class_probs, dtype=torch.float64)
    probabilities = probabilities / probabilities.sum()
    tails = torch.flip(torch.flip(probabilities, (0,)).cumsum(0), (0,))[1:]
    conditional = torch.empty_like(tails)
    conditional[0] = tails[0]
    conditional[1:] = tails[1:] / tails[:-1]
    return -torch.log1p(-conditional)


def run_contract_audit(seed: int = 20261001, trials: int = 64) -> dict[str, object]:
    generator = torch.Generator().manual_seed(seed)
    maximum_tail_order_error = 0.0
    maximum_ctmc_semigroup_error = 0.0
    minimum_sequential_semigroup_failure = math.inf
    maximum_ctmc_simplex_error = 0.0
    maximum_sequential_simplex_error = 0.0
    maximum_monotonicity_violation = 0.0
    for _ in range(trials):
        rates = 0.1 + 3.9 * torch.rand(4, generator=generator, dtype=torch.float64)
        ctmc = decode_pure_birth_rates(rates, force_fp64=True)
        sequential = sequential_posterior(rates)
        ctmc_tails = ctmc.cumulative_probs
        sequential_tails = 1.0 - sequential.cumsum(-1)[:-1]
        # Boundary zero is equal; every deeper same-rate tail is strictly
        # smaller because a shared clock must accommodate all transitions.
        maximum_tail_order_error = max(
            maximum_tail_order_error,
            float((ctmc_tails[1:] - sequential_tails[1:]).amax()),
        )

        first = float(torch.rand((), generator=generator) + 0.1)
        second = float(torch.rand((), generator=generator) + 0.1)
        q = pure_birth_generator(rates)
        p_first = torch.matrix_exp(first * q)
        p_second = torch.matrix_exp(second * q)
        p_sum = torch.matrix_exp((first + second) * q)
        maximum_ctmc_semigroup_error = max(
            maximum_ctmc_semigroup_error,
            float((p_sum - p_first @ p_second).abs().amax()),
        )
        s_first = sequential_transition(rates, first)
        s_second = sequential_transition(rates, second)
        s_sum = sequential_transition(rates, first + second)
        minimum_sequential_semigroup_failure = min(
            minimum_sequential_semigroup_failure,
            float((s_sum - s_first @ s_second).abs().amax()),
        )

        target = torch.rand(5, generator=generator, dtype=torch.float64) + 0.05
        target = target / target.sum()
        ctmc_rates = fit_pure_birth_rates(target, tolerance=1e-12)
        reconstructed_ctmc = decode_pure_birth_rates(ctmc_rates, force_fp64=True).class_probs
        sequential_rates = sequential_inverse(target)
        reconstructed_sequential = sequential_posterior(sequential_rates)
        maximum_ctmc_simplex_error = max(
            maximum_ctmc_simplex_error,
            float((reconstructed_ctmc - target).abs().amax()),
        )
        maximum_sequential_simplex_error = max(
            maximum_sequential_simplex_error,
            float((reconstructed_sequential - target).abs().amax()),
        )

        increment = 0.5 * torch.rand(4, generator=generator, dtype=torch.float64)
        larger_ctmc = decode_pure_birth_rates(rates + increment, force_fp64=True)
        larger_seq = sequential_posterior(rates + increment)
        larger_seq_tails = 1.0 - larger_seq.cumsum(-1)[:-1]
        maximum_monotonicity_violation = max(
            maximum_monotonicity_violation,
            float((ctmc.cumulative_probs - larger_ctmc.cumulative_probs).amax()),
            float((sequential_tails - larger_seq_tails).amax()),
        )

    # The small-horizon coefficient for crossing m transitions is
    # product(lambda[:m]) / m!.  Test m=1..4 at a horizon that remains
    # representable in float64.
    reference_rates = torch.tensor([0.7, 1.1, 1.3, 0.9], dtype=torch.float64)
    horizon = 1e-3
    tiny = torch.matrix_exp(horizon * pure_birth_generator(reference_rates))[0]
    factorial_relative_errors: list[float] = []
    for transitions in range(1, 5):
        observed_tail = float(tiny[transitions:].sum())
        leading = (
            horizon**transitions
            * float(reference_rates[:transitions].prod())
            / math.factorial(transitions)
        )
        factorial_relative_errors.append(abs(observed_tail - leading) / leading)

    passed = (
        maximum_tail_order_error <= 1e-12
        and maximum_ctmc_semigroup_error <= 1e-12
        and minimum_sequential_semigroup_failure >= 1e-6
        and maximum_ctmc_simplex_error <= 2e-9
        and maximum_sequential_simplex_error <= 1e-12
        and maximum_monotonicity_violation <= 2e-12
        and max(factorial_relative_errors) <= 0.01
    )
    return {
        "schema": "origin-decoder-contract-audit-v1",
        "seed": seed,
        "trials": trials,
        "fixed_horizon_conclusion": (
            "both decoders are saturated normalized monotone ordinal maps; "
            "exact replay belongs to the additive ledger"
        ),
        "ctmc_specific_contract": (
            "shared finite clock and homogeneous Markov semigroup across horizons"
        ),
        "maximum_same_rate_tail_order_error": maximum_tail_order_error,
        "maximum_ctmc_semigroup_error": maximum_ctmc_semigroup_error,
        "minimum_sequential_semigroup_failure": minimum_sequential_semigroup_failure,
        "maximum_ctmc_simplex_reconstruction_error": maximum_ctmc_simplex_error,
        "maximum_sequential_simplex_reconstruction_error": maximum_sequential_simplex_error,
        "maximum_monotonicity_violation": maximum_monotonicity_violation,
        "factorial_small_horizon_relative_errors": factorial_relative_errors,
        "passed": passed,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=20261001)
    parser.add_argument("--trials", type=int, default=64)
    args = parser.parse_args()
    payload = run_contract_audit(args.seed, args.trials)
    if not payload["passed"]:
        raise AssertionError(json.dumps(payload, indent=2))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    args.output.write_text(encoded)
    checksum = hashlib.sha256(encoded.encode()).hexdigest()
    print(json.dumps({"output": str(args.output), "sha256": checksum, **payload}, indent=2))


if __name__ == "__main__":
    main()
