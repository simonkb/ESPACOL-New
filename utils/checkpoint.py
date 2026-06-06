"""Checkpoint utilities for saving and restoring model state."""

from __future__ import annotations

import os
from typing import Optional

import torch


def save_checkpoint(
    path: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: dict,
    is_best: bool = False,
    text_encoder: Optional[torch.nn.Module] = None,
) -> None:
    """Save model, optimizer, metrics, and optional text encoder state.

    If is_best is True, also write a copy named best_model.pth in
    the same directory.
    """
    state = {
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "metrics": metrics,
    }

    if text_encoder is not None:
        state["text_encoder_state"] = text_encoder.state_dict()

    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(state, path)

    if is_best:
        best_path = os.path.join(os.path.dirname(path), "best_model.pth")
        torch.save(state, best_path)


def load_checkpoint(
    path: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    text_encoder: Optional[torch.nn.Module] = None,
    device: torch.device = torch.device("cpu"),
    strict: bool = True,
) -> dict:
    """Load checkpoint from path.

    Returns the full checkpoint state so callers can inspect metrics, epoch,
    and optional text_encoder_state.
    """
    state = torch.load(path, map_location=device)

    model.load_state_dict(state["model_state"], strict=strict)

    if optimizer is not None and "optimizer_state" in state:
        optimizer.load_state_dict(state["optimizer_state"])

    if text_encoder is not None and "text_encoder_state" in state:
        text_encoder.load_state_dict(state["text_encoder_state"], strict=strict)

    return state