#!/usr/bin/env python3
"""Gate-0 correctness and runtime preflight for the exact MOSAIC proof core."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from losses.mosaic import MosaicLoss
from models.mosaic import MOSAICOrdinalCore, TruncatedPoissonBinomial


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def calibrated_local_logits(
    batch_size: int,
    cells: int,
    classes: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Make a non-saturated, mixed-label proof-learning preflight batch.

    Standard-normal logits over a 12,544-cell field imply thousands of
    abnormal Bernoulli events.  Every count tail then saturates and a finite
    backward pass can misleadingly have exactly zero learning signal.  This
    fixture instead starts from the model's intended near-normal regime and
    injects a small, grade-dependent number of high-confidence abnormal cells.
    It is synthetic, but it exercises both stop and continuation outcomes.
    """

    if batch_size < 2:
        raise ValueError("batch_size must be at least 2 for the mixed-label preflight")
    labels = torch.linspace(
        0, classes - 1, batch_size, device=device
    ).round().long()

    background_abnormal_count = 0.25
    abnormal_probability = background_abnormal_count / float(cells)
    base = torch.full(
        (batch_size, cells, classes),
        abnormal_probability / float(classes - 1),
        device=device,
        dtype=torch.float32,
    )
    base[..., 0] = 1.0 - abnormal_probability

    for sample, grade in enumerate(labels.tolist()):
        if grade == 0:
            continue
        # Increasing evidence burden across grades, while staying far from
        # the max-count overflow regime.  A state-g cell supports exactly the
        # nested boundaries 0,...,g-1.
        witness_count = min(cells, 2 * grade + 1)
        local = torch.full(
            (witness_count, classes),
            1e-6,
            device=device,
            dtype=torch.float32,
        )
        local[:, 0] = 0.10
        local[:, grade] = 0.90
        local = local / local.sum(dim=-1, keepdim=True)
        base[sample, :witness_count] = local

    return base.clamp_min(torch.finfo(base.dtype).tiny).log(), labels


def main() -> None:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--cells", type=int, default=112 * 112)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--classes", type=int, default=5)
    parser.add_argument("--max_count", type=int, default=32)
    parser.add_argument("--block_size", type=int, default=64)
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--json_out", default=None)
    args = parser.parse_args()

    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available()
        else "mps" if args.device == "auto" and torch.backends.mps.is_available()
        else "cpu" if args.device == "auto" else args.device
    )
    torch.manual_seed(17)

    # Independent implementation agreement before the large benchmark.
    check = torch.rand(2, 3, 127, device=device)
    counter = TruncatedPoissonBinomial(
        args.max_count, implementation="block_tree", block_size=args.block_size
    ).to(device)
    serial = counter(check, implementation="serial")
    tree = counter(check, implementation="block_tree")
    max_error = float((serial - tree).abs().max())
    if max_error > 2e-5:
        raise AssertionError(f"block-tree/serial mismatch: {max_error}")

    core = MOSAICOrdinalCore(
        num_classes=args.classes,
        max_count=args.max_count,
        implementation="block_tree",
        block_size=args.block_size,
        sufficiency_tolerance=0.02,
        complement_suppression=0.5,
    ).to(device)
    valid = torch.ones(args.batch_size, args.cells, device=device, dtype=torch.bool)

    criterion = MosaicLoss(args.classes, dense_weight=0.1).to(device)
    timings = []
    backward_timings = []
    local_gradient_norms = []
    alpha_gradient_norms = []
    final = None
    for iteration in range(args.iterations + 1):
        logits, labels = calibrated_local_logits(
            args.batch_size, args.cells, args.classes, device
        )
        logits.requires_grad_(True)
        core.zero_grad(set_to_none=True)
        synchronize(device)
        start = time.perf_counter()
        output = core(logits, valid_mask=valid, project=True)
        synchronize(device)
        forward_seconds = time.perf_counter() - start
        loss, _ = criterion(
            output.transitions,
            labels,
            projected_stop_probabilities=output.stop_probabilities,
            projected_log_stop_probabilities=output.log_stop_probabilities,
            dense_transitions=output.dense_transitions,
            dense_stop_probabilities=output.dense_stop_probabilities,
            dense_log_stop_probabilities=output.dense_log_stop_probabilities,
        )
        synchronize(device)
        start = time.perf_counter()
        loss.backward()
        synchronize(device)
        backward_seconds = time.perf_counter() - start
        local_gradient_norm = float(logits.grad.float().norm())
        alpha_gradient = core.circuit.alpha_logits.grad
        alpha_gradient_norm = (
            float(alpha_gradient.float().norm()) if alpha_gradient is not None else 0.0
        )
        if iteration > 0:
            timings.append(forward_seconds)
            backward_timings.append(backward_seconds)
            local_gradient_norms.append(local_gradient_norm)
            alpha_gradient_norms.append(alpha_gradient_norm)
        final = output

    # Keep the original standard-normal field as a separate stress case.  It
    # creates a nearly all-abnormal lattice and therefore measures a large
    # proof, but is intentionally not used as evidence that training works.
    stress_logits = torch.randn(
        args.batch_size, args.cells, args.classes, device=device
    )
    synchronize(device)
    start = time.perf_counter()
    with torch.no_grad():
        stress_output = core(stress_logits, valid_mask=valid, project=True)
    synchronize(device)
    stress_forward_seconds = time.perf_counter() - start

    assert final is not None
    target = (final.proof.dense_transition - 0.02).clamp_min(0)
    sufficient = final.transitions + 2e-6 >= target
    suppressed = (
        final.proof.dense_transition - final.proof.complement_transition + 2e-6
        >= 0.5 * target
    )
    if not sufficient.all() or not suppressed.all():
        raise AssertionError("projected proof violates its certificate inequalities")
    if not torch.isfinite(logits.grad).all():
        raise AssertionError("non-finite backward gradient")
    if not torch.isfinite(final.log_stop_probabilities).all():
        raise AssertionError("non-finite projected log-stop likelihood")
    if not torch.isfinite(final.dense_log_stop_probabilities).all():
        raise AssertionError("non-finite dense log-stop likelihood")
    if min(local_gradient_norms) <= 0.0:
        raise AssertionError("calibrated proof loss produced zero local-logit gradient")
    if min(alpha_gradient_norms) <= 0.0:
        raise AssertionError("calibrated proof loss produced zero cardinality gradient")

    report = {
        "status": "pass",
        "device": str(device),
        "batch_size": args.batch_size,
        "cells": args.cells,
        "classes": args.classes,
        "max_count": args.max_count,
        "block_size": args.block_size,
        "serial_tree_max_error": max_error,
        "forward_seconds_mean": sum(timings) / len(timings),
        "backward_seconds_mean": sum(backward_timings) / len(backward_timings),
        "proof_size_mean": float(final.proof.proof_size.float().mean()),
        "local_gradient_norm_min": min(local_gradient_norms),
        "alpha_gradient_norm_min": min(alpha_gradient_norms),
        "projected_log_stop_min": float(
            final.log_stop_probabilities.detach().min()
        ),
        "dense_log_stop_min": float(
            final.dense_log_stop_probabilities.detach().min()
        ),
        "stress_random_forward_seconds": stress_forward_seconds,
        "stress_random_proof_size_mean": float(
            stress_output.proof.proof_size.float().mean()
        ),
        "peak_cuda_memory_gib": (
            torch.cuda.max_memory_allocated(device) / 2**30 if device.type == "cuda" else None
        ),
    }
    print(json.dumps(report, indent=2))
    if args.json_out:
        Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.json_out, "w") as stream:
            json.dump(report, stream, indent=2)


if __name__ == "__main__":
    main()
