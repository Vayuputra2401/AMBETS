"""
boundary_head.py — BoundaryRefinementHead: thin residual 3D conv head.

Sharpens segmentation boundaries without overriding routing assignments.
<100K parameters. Runs after the MONAI UNetr decoder on seg_logits.

Architecture:
  Conv3d(K→16, 3×3×3, padding=1) → InstanceNorm3d(16) → ReLU
  → Conv3d(16→K, 1×1×1)
  output = input + residual    (residual connection prevents overwrite)
"""

from __future__ import annotations

import torch
import torch.nn as nn


class BoundaryRefinementHead(nn.Module):
    """
    Edge-sharpening residual head applied to decoder output logits.

    Parameters
    ----------
    num_concepts    : int   Number of segmentation classes K (default 4).
    hidden_channels : int   Internal feature width (default 16).

    Input / Output
    --------------
    [B, K, H, W, D] → [B, K, H, W, D]   (identical shape, residual refinement)
    """

    def __init__(self, num_concepts: int = 4, hidden_channels: int = 16) -> None:
        super().__init__()
        self.refine = nn.Sequential(
            nn.Conv3d(num_concepts, hidden_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(hidden_channels, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv3d(hidden_channels, num_concepts, kernel_size=1, bias=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : [B, K, H, W, D]   logits from decoder

        Returns
        -------
        [B, K, H, W, D]   refined logits (input + residual)
        """
        return x + self.refine(x)
