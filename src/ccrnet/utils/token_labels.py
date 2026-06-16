"""
token_labels.py — Downsample GT label maps to the CCR token grid.

Called in the training loop after model.forward() and before CCRTotalLoss:
  token_labels = downsample_labels_to_tokens(label, model.get_grid_shape())
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn.functional as F


def downsample_labels_to_tokens(
    labels: torch.Tensor,
    grid_shape: Tuple[int, int, int],
) -> torch.Tensor:
    """
    Nearest-neighbor downsample a GT label map to the CCR token grid.

    Parameters
    ----------
    labels     : [B, H, W, D]  int64 ground-truth label map
    grid_shape : (h, w, d)     target spatial size, e.g. (16, 16, 16)

    Returns
    -------
    [B, N]  int64 token labels, where N = h * w * d  (e.g. 4096 for 16³)
    """
    B = labels.shape[0]
    labels_f = labels.float().unsqueeze(1)              # [B, 1, H, W, D]
    down = F.interpolate(labels_f, size=grid_shape, mode="nearest")
    return down.long().squeeze(1).reshape(B, -1)        # [B, N]
