"""
segmentation.py — L_seg = 0.5 × DiceLoss + 0.5 × FocalLoss(γ=2).

Both losses operate on 3D volumetric segmentation outputs or token-space predictions.
Input convention throughout:
    pred_logits : Tensor [B, K, *spatial]  or  [B, N, K] for token space
    labels      : Tensor [B, *spatial]     or  [B, N]     integer class indices

Plan reference: CCR-Net_Research_Plan.md Section 7.1 — L_seg formula.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ccr.config.ccr_config import LossConfig


class DiceLoss(nn.Module):
    """
    Soft multi-class Dice loss, macro-averaged over all K classes.

    Math
    ----
    For each class c ∈ {0, …, K-1}:

        p_c = softmax(pred_logits)[..., c]       predicted probability for class c
        y_c = (labels == c).float()              binary ground-truth mask for class c

        dice_c = (2 × Σ(p_c × y_c) + ε) / (Σ(p_c²) + Σ(y_c²) + ε)

    The squared denominator form (p² + y²) is numerically more stable than the
    linear form (p + y) when predictions are soft (not binary), because both
    terms vanish at the same rate as the numerator when pred matches label.

        L_dice = 1 − (1/K) × Σ_c dice_c

    Parameters
    ----------
    num_classes : int
        K — must equal the size of the class dimension in pred_logits.
    smooth : float
        Laplace smoothing ε.  Prevents 0/0 on empty classes (e.g., ET in small crops).

    Inputs
    ------
    pred_logits : Tensor [B, K, *spatial]   raw (pre-softmax) logits
    labels      : Tensor [B, *spatial]      integer class labels in [0, K)

    Returns
    -------
    Tensor []   scalar Dice loss in [0, 1]
    """

    def __init__(self, num_classes: int, smooth: float = 1e-5) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.smooth = smooth

    def forward(
        self,
        pred_logits: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        # pred_logits: [B, K, *spatial]
        # Convert logits → probabilities along the class dimension
        probs = F.softmax(pred_logits, dim=1)  # [B, K, *spatial]

        # Flatten spatial dimensions: [B, K, *spatial] → [B, K, M]
        # where M = product of all spatial dimensions
        B, K = probs.shape[:2]
        probs_flat  = probs.reshape(B, K, -1)        # [B, K, M]
        labels_flat = labels.reshape(B, -1).long()   # [B, M]

        dice_per_class = probs_flat.new_zeros(K)

        for c in range(K):
            p_c = probs_flat[:, c, :]                          # [B, M] predicted prob for class c
            y_c = (labels_flat == c).float()                   # [B, M] binary GT for class c

            # Numerator: 2 × Σ(p_c × y_c) summed over batch and spatial dims
            numerator = 2.0 * (p_c * y_c).sum(dim=[0, 1])     # scalar

            # Denominator: Σ(p_c²) + Σ(y_c²) — squared form for stable soft Dice
            denominator = (p_c * p_c).sum(dim=[0, 1]) + (y_c * y_c).sum(dim=[0, 1])

            dice_per_class[c] = (numerator + self.smooth) / (denominator + self.smooth)

        # Macro-average: 1 − mean Dice across all classes
        return 1.0 - dice_per_class.mean()


class FocalLoss(nn.Module):
    """
    Multi-class focal loss for class-imbalanced dense prediction.

    Math (per Lin et al., 2017, adapted to multi-class)
    ---------------------------------------------------
    For each voxel at position t with ground-truth class c_t:

        p_t      = softmax(pred_logits)[..., c_t]    predicted prob at GT class
        FL(t)    = −α × (1 − p_t)^γ × log(p_t + ε)

        L_focal  = mean_t FL(t)

    The modulating factor (1 − p_t)^γ:
      - Down-weights easy examples: when p_t ≈ 1, the factor → 0.
      - Focuses training on hard examples: when p_t ≈ 0, the factor → 1.

    For BraTS, ET voxels occupy < 5 % of a typical volume.  Focal loss prevents
    the background class from dominating gradients.

    Parameters
    ----------
    gamma : float  focusing exponent γ (default 2.0)
    alpha : float  class-balancing weight α (default 0.25)

    Inputs
    ------
    pred_logits : Tensor [B, K, *spatial]   raw logits
    labels      : Tensor [B, *spatial]      integer GT labels in [0, K)

    Returns
    -------
    Tensor []   scalar focal loss  ≥ 0
    """

    _LOG_EPS: float = 1e-8

    def __init__(self, gamma: float = 2.0, alpha: float = 0.25) -> None:
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(
        self,
        pred_logits: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        # pred_logits: [B, K, *spatial]
        # Compute per-class probabilities
        probs = F.softmax(pred_logits, dim=1)  # [B, K, *spatial]

        # Gather p_t: probability at the ground-truth class for each voxel
        # labels needs to be [B, 1, *spatial] for gather
        labels_expanded = labels.long().unsqueeze(1)               # [B, 1, *spatial]
        p_t = probs.gather(dim=1, index=labels_expanded).squeeze(1)  # [B, *spatial]

        # Focal loss: −α (1−p_t)^γ log(p_t)
        focal_weight  = (1.0 - p_t) ** self.gamma                  # [B, *spatial]
        log_p_t       = (p_t + self._LOG_EPS).log()                 # [B, *spatial]
        loss_per_voxel = -self.alpha * focal_weight * log_p_t       # [B, *spatial]

        return loss_per_voxel.mean()


class SegmentationLoss(nn.Module):
    """
    L_seg = 0.5 × DiceLoss + 0.5 × FocalLoss(γ=2)

    Combined segmentation loss from CCR plan Section 7.1.

    The Dice term penalises volumetric overlap errors globally.
    The Focal term concentrates gradient signal on hard (boundary/minority) voxels.
    Equal weighting is standard in BraTS literature.

    Parameters
    ----------
    config : LossConfig
        focal_gamma, focal_alpha, dice_smooth, num_classes (inferred from pred_logits).

    Inputs
    ------
    pred_logits : Tensor [B, K, *spatial]
    labels      : Tensor [B, *spatial]

    Returns
    -------
    dict with keys:
        total : scalar   L_seg
        dice  : scalar   DiceLoss component
        focal : scalar   FocalLoss component
    """

    def __init__(self, config: LossConfig) -> None:
        super().__init__()
        self._config = config
        # num_classes inferred from pred_logits.shape[1] at forward time
        self._focal = FocalLoss(gamma=config.focal_gamma, alpha=config.focal_alpha)
        self._smooth = config.dice_smooth

    def forward(
        self,
        pred_logits: torch.Tensor,
        labels: torch.Tensor,
    ) -> dict:
        num_classes = pred_logits.shape[1]

        dice_loss_fn = DiceLoss(num_classes=num_classes, smooth=self._smooth)
        dice_val  = dice_loss_fn(pred_logits, labels)
        focal_val = self._focal(pred_logits, labels)

        total = 0.5 * dice_val + 0.5 * focal_val

        return {
            "total": total,
            "dice":  dice_val,
            "focal": focal_val,
        }
