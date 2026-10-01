#!/usr/bin/env python3
"""Independent numerical audit for ORIGIN's pure-birth decoder.

The deployed implementation uses a differentiable FP64 Taylor
scaling-and-squaring exponential.  This audit deliberately obtains its
reference values from mpmath at high precision and checks forward
probabilities plus finite-difference derivatives of the expected grade.
It never trains a model or reads patient data.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any, Iterable, Sequence

import mpmath as mp
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models.origin import decode_pure_birth_rates


SCHEMA = "origin-decoder-independent-numerical-audit-v1"


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _reference_transition(rates: Sequence[float], *, dps: int) -> np.ndarray:
    """Return ``exp(Q)`` from an implementation independent of PyTorch."""

    if dps < 50:
        raise ValueError("reference precision must be at least 50 decimal digits")
    values = [float(value) for value in rates]
    if not values or any(not math.isfinite(value) or value < 0.0 for value in values):
        raise ValueError("reference rates must be finite, nonnegative, and nonempty")
    with mp.workdps(dps):
        size = len(values) + 1
        generator = mp.matrix(size, size)
        for index, value in enumerate(values):
            rate = mp.mpf(repr(value))
            generator[index, index] = -rate
            generator[index, index + 1] = rate
        transition = mp.expm(generator)
        return np.asarray(
            [
                [float(transition[row, column]) for column in range(size)]
                for row in range(size)
            ],
            dtype=np.float64,
        )


def _reference_expected_grade(rates: Sequence[float], *, dps: int) -> float:
    transition = _reference_transition(rates, dps=dps)
    return float(np.dot(transition[0], np.arange(transition.shape[0], dtype=np.float64)))


def _reference_expected_grade_mp(rates: Sequence[mp.mpf]) -> mp.mpf:
    """High-precision expected grade used inside numerical differentiation."""

    size = len(rates) + 1
    generator = mp.matrix(size, size)
    for index, rate in enumerate(rates):
        generator[index, index] = -rate
        generator[index, index + 1] = rate
    transition = mp.expm(generator)
    return mp.fsum(mp.mpf(column) * transition[0, column] for column in range(size))


def _rate_cases(samples: int, seed: int, cap: float) -> list[list[float]]:
    if samples < 1:
        raise ValueError("samples must be positive")
    if not math.isfinite(cap) or cap <= 0.0:
        raise ValueError("rate cap must be finite and positive")
    adversarial = [
        [0.0, 0.0, 0.0, 0.0],
        [cap, cap, cap, cap],
        [cap, 0.0, cap, 0.0],
        [0.0, cap, 0.0, cap],
        [1e-12, 1e-9, 1e-6, 1e-3],
        [1e-3, 1e-6, 1e-9, 1e-12],
        [0.5, 0.5, 0.5, 0.5],
        [1.0, 1.0 + 1e-10, 1.0 - 1e-10, 1.0],
        [cap, cap / 4.0, 1.0, 1e-6],
        [1e-6, 1.0, cap / 4.0, cap],
    ]
    rng = np.random.default_rng(seed)
    cases = list(adversarial)
    for index in range(samples):
        if index % 2:
            values = rng.uniform(0.0, cap, size=4)
        else:
            exponent = rng.uniform(-12.0, math.log10(cap), size=4)
            values = np.power(10.0, exponent)
        cases.append([float(value) for value in values])
    return cases


def _finite_difference_gradient(
    rates: Sequence[float], *, dps: int
) -> np.ndarray:
    # mpmath's ``step`` differentiator evaluates a high-precision finite
    # difference independently of PyTorch/autograd.  It also avoids converting
    # nearly-null deep-boundary derivatives to FP64 before subtraction.
    with mp.workdps(dps):
        base = [mp.mpf(repr(float(value))) for value in rates]
        result: list[float] = []
        for index, value in enumerate(base):
            def function(candidate: mp.mpf) -> mp.mpf:
                current = list(base)
                current[index] = candidate
                return _reference_expected_grade_mp(current)

            derivative = mp.diff(function, value, method="step", addprec=30)
            result.append(float(derivative))
    return np.asarray(result, dtype=np.float64)


def audit_decoder(
    *,
    samples: int = 64,
    gradient_samples: int = 8,
    seed: int = 20261001,
    rate_cap: float = 64.0,
    dps: int = 90,
) -> dict[str, Any]:
    """Audit forward probabilities and expected-grade derivatives."""

    cases = _rate_cases(samples, seed, rate_cap)
    forward_rows: list[dict[str, Any]] = []
    max_abs = 0.0
    max_rel = 0.0
    max_row_error = 0.0
    minimum_probability = math.inf
    for case_index, rates in enumerate(cases):
        tensor = torch.tensor([rates], dtype=torch.float64)
        observed = decode_pure_birth_rates(tensor, force_fp64=True)
        probability = observed.class_probs[0].detach().cpu().numpy()
        reference = _reference_transition(rates, dps=dps)[0]
        absolute = np.abs(probability - reference)
        relative = absolute / np.maximum(np.abs(reference), 1e-300)
        row_error = abs(float(probability.sum()) - 1.0)
        record = {
            "case": case_index,
            "rates": rates,
            "max_absolute_probability_error": float(absolute.max()),
            "max_relative_probability_error": float(relative.max()),
            "row_sum_error": row_error,
            "minimum_probability": float(probability.min()),
        }
        forward_rows.append(record)
        max_abs = max(max_abs, record["max_absolute_probability_error"])
        max_rel = max(max_rel, record["max_relative_probability_error"])
        max_row_error = max(max_row_error, row_error)
        minimum_probability = min(minimum_probability, record["minimum_probability"])

    gradient_rows: list[dict[str, Any]] = []
    max_gradient_abs = 0.0
    max_gradient_rel = 0.0
    # Exclude all-zero and saturated adversarial cases from a relative-gradient
    # summary; their true derivatives can be effectively zero.
    eligible = [case for case in cases if max(case) <= 0.75 * rate_cap and sum(case) > 1e-8]
    for case_index, rates in enumerate(eligible[:gradient_samples]):
        tensor = torch.tensor([rates], dtype=torch.float64, requires_grad=True)
        expected = decode_pure_birth_rates(tensor, force_fp64=True).expected_grade.sum()
        (gradient,) = torch.autograd.grad(expected, tensor)
        observed = gradient[0].detach().cpu().numpy()
        reference = _finite_difference_gradient(rates, dps=dps)
        absolute = np.abs(observed - reference)
        relative = absolute / np.maximum(np.abs(reference), 1e-10)
        record = {
            "case": case_index,
            "rates": rates,
            "autograd": observed.tolist(),
            "finite_difference": reference.tolist(),
            "max_absolute_gradient_error": float(absolute.max()),
            "max_relative_gradient_error": float(relative.max()),
        }
        gradient_rows.append(record)
        max_gradient_abs = max(max_gradient_abs, record["max_absolute_gradient_error"])
        max_gradient_rel = max(max_gradient_rel, record["max_relative_gradient_error"])

    tolerances = {
        "forward_absolute": 2e-12,
        "forward_row_sum": 5e-12,
        "gradient_absolute": 2e-6,
        "gradient_relative": 2e-4,
    }
    passed = bool(
        max_abs <= tolerances["forward_absolute"]
        and max_row_error <= tolerances["forward_row_sum"]
        and minimum_probability >= 0.0
        and max_gradient_abs <= tolerances["gradient_absolute"]
        and max_gradient_rel <= tolerances["gradient_relative"]
    )
    payload: dict[str, Any] = {
        "schema": SCHEMA,
        "implementation_under_test": "models.origin.decode_pure_birth_rates",
        "independent_reference": "mpmath.expm",
        "reference_decimal_digits": int(dps),
        "seed": int(seed),
        "rate_domain": [0.0, float(rate_cap)],
        "n_forward_cases": len(forward_rows),
        "n_gradient_cases": len(gradient_rows),
        "summary": {
            "max_absolute_probability_error": max_abs,
            "max_relative_probability_error": max_rel,
            "max_row_sum_error": max_row_error,
            "minimum_probability": minimum_probability,
            "max_absolute_gradient_error": max_gradient_abs,
            "max_relative_gradient_error": max_gradient_rel,
            "passed": passed,
        },
        "tolerances": tolerances,
        "forward_cases": forward_rows,
        "gradient_cases": gradient_rows,
    }
    payload["content_checksum_sha256"] = _canonical_sha256(payload)
    return payload


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    temporary.replace(path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=64)
    parser.add_argument("--gradient-samples", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20261001)
    parser.add_argument("--rate-cap", type=float, default=64.0)
    parser.add_argument("--dps", type=int, default=90)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("runs/origin_acceptance/numerical_decoder_audit.json"),
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    payload = audit_decoder(
        samples=args.samples,
        gradient_samples=args.gradient_samples,
        seed=args.seed,
        rate_cap=args.rate_cap,
        dps=args.dps,
    )
    _write_json(args.output, payload)
    print(json.dumps({"output": str(args.output), **payload["summary"]}, indent=2))
    if not payload["summary"]["passed"]:
        raise SystemExit("ORIGIN decoder numerical audit failed")


if __name__ == "__main__":
    main()
