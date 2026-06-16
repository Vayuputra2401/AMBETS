"""
checkpoint.py — Save/load model checkpoints and expert reinitialization.

save_checkpoint : saves model + optimizer + scheduler + epoch + metrics + config
load_checkpoint : restores model (and optionally optimizer/scheduler), returns (epoch, metrics)
reinitialize_experts : resets fc2 weights of collapsed experts (called by collapse monitor)
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    epoch: int,
    metrics: Dict,
    config,
    ckpt_dir: str,
) -> str:
    """
    Save full training state to <ckpt_dir>/epoch_<epoch>.pth.

    Returns
    -------
    str   path to saved checkpoint
    """
    os.makedirs(ckpt_dir, exist_ok=True)
    path = os.path.join(ckpt_dir, f"epoch_{epoch:04d}.pth")
    torch.save(
        {
            "epoch":            epoch,
            "model_state":      model.state_dict(),
            "optimizer_state":  optimizer.state_dict(),
            "scheduler_state":  scheduler.state_dict() if scheduler is not None else None,
            "metrics":          metrics,
            "config":           config,
        },
        path,
    )
    return path


def load_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler=None,
) -> Tuple[int, Dict]:
    """
    Load checkpoint from path into model (and optionally optimizer/scheduler).

    Parameters
    ----------
    path      : path to .pth checkpoint file
    model     : model instance to load weights into (in-place)
    optimizer : optional optimizer to restore state
    scheduler : optional LR scheduler to restore state

    Returns
    -------
    (start_epoch, metrics)
        start_epoch : epoch to resume from (saved_epoch + 1)
        metrics     : metrics dict saved at that checkpoint
    """
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    if optimizer is not None and ckpt.get("optimizer_state") is not None:
        optimizer.load_state_dict(ckpt["optimizer_state"])
    if scheduler is not None and ckpt.get("scheduler_state") is not None:
        scheduler.load_state_dict(ckpt["scheduler_state"])
    return ckpt["epoch"] + 1, ckpt.get("metrics", {})


def reinitialize_experts(ccr_module: nn.Module, collapsed_names: List[str]) -> None:
    """
    Reinitialize fc2 weights of named collapsed experts.

    Matches the gain=0.01 from CCRBottleneckModule._init_weights() to keep
    the reinitialized expert in the same dynamic regime as training start.

    Parameters
    ----------
    ccr_module      : CCRBottleneckModule instance
    collapsed_names : list of concept_name strings whose experts collapsed
    """
    for expert in ccr_module.experts:
        if expert.concept_name in collapsed_names:
            nn.init.xavier_uniform_(expert.fc2.weight, gain=0.01)
            nn.init.zeros_(expert.fc2.bias)
