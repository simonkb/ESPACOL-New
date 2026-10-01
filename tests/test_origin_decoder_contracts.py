from __future__ import annotations

import torch

from models.origin import decode_pure_birth_rates, pure_birth_generator
from tools.audit_origin_decoder_contracts import (
    run_contract_audit,
    sequential_posterior,
    sequential_transition,
)


def test_same_rates_imply_stricter_deep_ctmc_tails() -> None:
    rates = torch.tensor([0.7, 1.1, 1.3, 0.9], dtype=torch.float64)
    ctmc = decode_pure_birth_rates(rates, force_fp64=True)
    sequential = sequential_posterior(rates)
    sequential_tails = 1.0 - sequential.cumsum(-1)[:-1]
    assert torch.allclose(ctmc.cumulative_probs[:1], sequential_tails[:1])
    assert bool((ctmc.cumulative_probs[1:] < sequential_tails[1:]).all())


def test_ctmc_semigroup_and_sequential_failure() -> None:
    rates = torch.ones(2, dtype=torch.float64)
    q = pure_birth_generator(rates)
    half = torch.matrix_exp(0.5 * q)
    whole = torch.matrix_exp(q)
    assert torch.allclose(whole, half @ half, atol=1e-14, rtol=1e-14)
    sequential_half = sequential_transition(rates, 0.5)
    sequential_whole = sequential_transition(rates, 1.0)
    assert float((sequential_whole - sequential_half @ sequential_half).abs().amax()) > 0.05


def test_complete_decoder_contract_audit() -> None:
    payload = run_contract_audit(seed=17, trials=8)
    assert payload["passed"] is True

