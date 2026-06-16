"""
boundary.py — L_boundary: boundary-aware cross-entropy for 3D segmentation.

Tumor sub-region boundaries are the clinically most critical and geometrically
most difficult regions to segment accurately.  Standard cross-entropy treats every
voxel equally, giving the optimiser no special incentive to get boundaries right.

L_boundary addresses this with a spatially-varying weight map: voxels near the
boundary between two classes receive higher cross-entropy weight than interior voxels.

Boundary detection (GPU-compatible, no scipy)
---------------------------------------------
For a 3D integer label map:
  1. One-hot encode labels: [B, H, W, D] → [B, K, H, W, D]
  2. Morphological dilation: max_pool3d with 3×3×3 kernel, padding=1
     dilated[b, k, h, w, d] = max over 3×3×3 neighbourhood
  3. Morphological erosion: −max_pool3d(−x) with same kernel
     eroded[b, k, h, w, d] = min over 3×3×3 neighbourhood
  4. boundary_k = |dilated_k − eroded_k| > 0.5   (voxels at class-k boundary)
  5. boundary   = OR over all k (any-class boundary)
  6. Optional: iterative dilation of boundary mask to widen the boundary zone.

Weight map:
    weight(b, h, w, d) = base_weight + boost_factor × is_boundary(b, h, w, d)
    Interior: base_weight (e.g. 1.0)
    Boundary: base_weight + boost_factor (e.g. 1.0 + 5.0 = 6.0)

Loss:
    L_boundary = mean_{b,h,w,d} [ weight(b,h,w,d) × CE(pred_logits, labels) ]

The weight computation is wrapped in torch.no_grad() — it is not differentiated.

Plan reference: CCR-Net_Research_Plan.md Section 7.1 — L_boundary description.
Standard BraTS boundary-aware segmentation literature (e.g. Myronenko 2018).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ccr.config.ccr_config import LossConfig


class BoundaryAwareLoss(nn.Module):
    """
    Distance-weighted cross-entropy for 3D multi-class segmentation.

    Parameters
    ----------
    num_classes : int
        K — number of segmentation classes.
    dilation_iters : int
        Number of morphological dilation passes that widen the boundary zone.
        More iterations → wider zone → more voxels boosted.
        Default 3 balances precision vs. zone coverage.
    base_weight : float
        Cross-entropy weight applied uniformly to all non-boundary voxels.
    boost_factor : float
        Additional weight added at boundary voxels.
        Total boundary weight = base_weight + boost_factor.

    Inputs
    ------
    pred_logits : Tensor [B, K, H, W, D]   raw (pre-softmax) logits
    labels      : Tensor [B, H, W, D]      integer class labels in [0, K)

    Returns
    -------
    Tensor []   scalar boundary-weighted cross-entropy loss  ≥ 0
    """

    _POOL_KERNEL: int = 3
    _POOL_PAD:    int = 1

    def __init__(
        self,
        num_classes: int,
        dilation_iters: int = 3,
        base_weight: float = 1.0,
        boost_factor: float = 5.0,
    ) -> None:
        super().__init__()
        self.num_classes    = num_classes
        self.dilation_iters = dilation_iters
        self.base_weight    = base_weight
        self.boost_factor   = boost_factor

    @torch.no_grad()
    def _compute_boundary_mask(self, labels: torch.Tensor) -> torch.Tensor:
        """
        Detect boundary voxels using morphological dilation and erosion.

        A voxel is on the class boundary if its one-hot class membership changes
        between the dilated and eroded versions of the label map.

        Parameters
        ----------
        labels : Tensor [B, H, W, D]   integer labels

        Returns
        -------
        Tensor [B, H, W, D]  boolean — True at boundary voxels
        """
        B, H, W, D = labels.shape
        K = self.num_classes

        # One-hot encode: [B, H, W, D] → [B, K, H, W, D]
        labels_oh = (
            F.one_hot(labels.long(), num_classes=K)
            .permute(0, 4, 1, 2, 3)
            .float()
        )  # [B, K, H, W, D]

        # Morphological dilation: max in 3×3×3 neighbourhood (sliding-window)
        # F.max_pool3d operates on a [B*K, 1, H, W, D] tensor
        BK = B * K
        x_flat = labels_oh.reshape(BK, 1, H, W, D)

        dilated_flat = F.max_pool3d(
            x_flat,
            kernel_size=self._POOL_KERNEL,
            stride=1,
            padding=self._POOL_PAD,
        )  # [B*K, 1, H, W, D]

        # Morphological erosion: min in neighbourhood = −max_pool3d(−x)
        eroded_flat = -F.max_pool3d(
            -x_flat,
            kernel_size=self._POOL_KERNEL,
            stride=1,
            padding=self._POOL_PAD,
        )  # [B*K, 1, H, W, D]

        dilated = dilated_flat.reshape(B, K, H, W, D)
        eroded  = eroded_flat.reshape(B, K, H, W, D)

        # A voxel is at the boundary of class k if dilated_k ≠ eroded_k
        # Take the union over all classes: any-class boundary
        boundary = ((dilated - eroded).abs() > 0.5).any(dim=1)  # [B, H, W, D] bool

        # Widen the boundary zone with additional dilation passes
        for _ in range(self.dilation_iters - 1):
            boundary_float = boundary.float().unsqueeze(1)  # [B, 1, H, W, D]
            boundary = (
                F.max_pool3d(
                    boundary_float,
                    kernel_size=self._POOL_KERNEL,
                    stride=1,
                    padding=self._POOL_PAD,
                ).squeeze(1)
                > 0.5
            )  # [B, H, W, D] bool

        return boundary  # [B, H, W, D]

    def forward(
        self,
        pred_logits: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute boundary-aware cross-entropy.

        Parameters
        ----------
        pred_logits : Tensor [B, K, H, W, D]
        labels      : Tensor [B, H, W, D]

        Returns
        -------
        Tensor []   scalar
        """
        # Step 1: Detect boundary voxels (no gradient through this)
        boundary_mask = self._compute_boundary_mask(labels)  # [B, H, W, D] bool

        # Step 2: Build per-voxel weight map
        weights = labels.new_full(
            labels.shape, self.base_weight, dtype=torch.float
        )  # [B, H, W, D]
        weights[boundary_mask] = weights[boundary_mask] + self.boost_factor

        # Detach weights: they do not carry gradients through the boundary detection.
        # The weight map guides the loss landscape but is treated as a fixed scalar.
        weights = weights.detach()

        # Step 3: Per-voxel cross-entropy (unreduced)
        ce_per_voxel = F.cross_entropy(
            pred_logits, labels.long(), reduction="none"
        )  # [B, H, W, D]

        # Step 4: Weighted mean
        return (weights * ce_per_voxel).mean()
