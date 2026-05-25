"""Checkpoint utilities for saving and restoring model state."""

from __future__ import annotations

import os
import torch


def save_checkpoint(
    path: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: dict,
    is_best: bool = False,
) -> None:
    """Save model + optimizer state to *path*.

    If *is_best* is True, also write a copy named 'best_model.pth' in
    the same directory.
    """
    state = {
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "metrics": metrics,
    }
    torch.save(state, path)

    if is_best:
        best_path = os.path.join(os.path.dirname(path), "best_model.pth")
        torch.save(state, best_path)


def load_checkpoint(
    path: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    device: torch.device = torch.device("cpu"),
) -> dict:
    """Load checkpoint from *path*.

    Returns the stored metrics dict so callers can inspect best_val_loss etc.
    """
    state = torch.load(path, map_location=device)
    model.load_state_dict(state["model_state"])
    if optimizer is not None and "optimizer_state" in state:
        optimizer.load_state_dict(state["optimizer_state"])
    return state.get("metrics", {})
