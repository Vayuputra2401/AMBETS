"""
alignment.py — L_align: concept alignment loss (the core CCR interpretability loss).

L_align forces Expert k to activate for exactly the voxels belonging to clinical
concept k in the ground-truth annotation.  Without this loss, routing is arbitrary
— tokens may be routed to any expert regardless of clinical meaning.  With it,
routing becomes clinically interpretable: the routing map for concept k IS the
model's per-voxel prediction of subregion k.

This loss is the mechanism that creates the Concept Alignment Score (CAS):
    CAS(k) = Pearson(P(t→k), M_k(t))
After sufficient training with L_align, CAS(k) ≥ 0.85 for all k (BraTS target).

Base math (from CCR plan Section 7.1)
--------------------------------------
For each concept k ∈ {0, …, K-1}:

    gt_k(b, t)  = 1  if token (b, t) has ground-truth label k, else 0   [B, N]
    p_k(b, t)   = routing_probs[b, t, k]                                  [B, N]
    loss_k      = BCE(p_k, gt_k)

    L_align = (1/K) × Σ_k BCE(p_k, gt_k)

Phase 1 improvements (docs/phase1_analysis.md §5, Improvements 1 and 4)
------------------------------------------------------------------------
1. Foreground-only (foreground_only=True, default):
   Background tokens (~80% of BraTS volume) dominate standard BCE and make the
   gradient easy to satisfy.  The hard alignment problem is distinguishing NCR
   from Edema from ET.  With foreground_only=True:
     - Only non-background tokens (labels > 0) contribute to the loss.
     - Only concepts k=1..K-1 are included (background concept is excluded).
   This concentrates the entire L_align gradient signal on the tumor subregions,
   directly improving CAS_fg(k=1,2,3).

2. Focal modulation (focal_gamma > 0, default 2.0):
   Standard BCE weights all tokens equally.  Interior NCR/Edema/ET tokens far
   from boundaries are easy and quickly reach p_k ≈ 1.  Focal BCE down-weights
   these easy tokens and focuses training on hard boundary/transition tokens:
       bce_focal = (1 − p_correct)^γ × BCE
   where p_correct = p_k for GT-positive tokens, (1−p_k) for GT-negative tokens.
   ET is ~2% of foreground; focal modulation prevents the large Edema mass from
   dominating gradient signal and improves CAS(NCR) and CAS(ET).

Plan reference: CCR-Net_Research_Plan.md Section 7.1, alignment_loss code block.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConceptAlignmentLoss(nn.Module):
    """
    L_align — per-concept BCE between routing probabilities and GT subregion masks.

    Supports foreground-only computation and focal modulation.
    Both are enabled by default (see module docstring for rationale).

    Parameters
    ----------
    num_concepts : int
        K — number of clinical concepts.
    foreground_only : bool
        If True (default), only non-background tokens (labels > 0) contribute
        to the loss, and only concepts k=1..K-1 are included.
        Background (k=0) is excluded because it dominates BCE and makes the
        loss trivially easy to minimise without improving tumor-subregion routing.
    focal_gamma : float
        Focal modulation exponent γ ≥ 0.  Set to 0.0 to use plain BCE.
        When γ > 0: bce_focal(t) = (1−p_correct(t))^γ × BCE(t)
        Easy tokens (p_correct ≈ 1) receive weight ≈ 0; hard tokens receive
        weight ≈ 1.  γ=2 is standard (Lin et al., 2017).

    Inputs
    ------
    routing_probs : Tensor [B, N, K]
        Soft routing probabilities from ClinicalConceptRouter.
        Values ∈ (0, 1), sum to 1 along the K dimension.
    seg_labels : Tensor [B, N]
        Integer ground-truth concept labels per token, values ∈ {0, …, K-1}.
        Obtained by downsampling the 3D segmentation label map to token resolution.

    Returns
    -------
    Tensor []   scalar alignment loss  ≥ 0
    """

    _PROB_EPS: float = 1e-7
    _PROB_MAX: float = 1.0 - 1e-7
    _BG_CLASS: int  = 0    # Background concept index — excluded with foreground_only

    def __init__(
        self,
        num_concepts: int,
        foreground_only: bool = True,
        focal_gamma: float = 2.0,
    ) -> None:
        super().__init__()
        self.num_concepts    = num_concepts
        self.foreground_only = foreground_only
        self.focal_gamma     = focal_gamma

    def forward(
        self,
        routing_probs: torch.Tensor,
        seg_labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute L_align.

        With foreground_only=True and focal_gamma=2.0 (defaults):
            L_align = (1/(K-1)) × Σ_{k=1}^{K-1}  mean_{fg tokens} [
                          (1−p_correct(t))^γ × BCE(p_k(t), gt_k(t))
                      ]
        where fg tokens are those with GT label ≠ 0 (non-background).

        Parameters
        ----------
        routing_probs : Tensor [B, N, K]
        seg_labels    : Tensor [B, N]    integer labels in {0, …, K-1}

        Returns
        -------
        Tensor []   scalar
        """
        B, N, K = routing_probs.shape
        assert seg_labels.shape == (B, N), (
            f"seg_labels shape {seg_labels.shape} does not match "
            f"routing_probs batch/token shape ({B}, {N})"
        )
        assert K == self.num_concepts, (
            f"routing_probs K={K} does not match num_concepts={self.num_concepts}"
        )

        # Determine which concepts and tokens contribute to the loss.
        if self.foreground_only:
            # Only non-background tokens contribute.
            fg_mask  = (seg_labels != self._BG_CLASS).float()   # [B, N]
            fg_count = fg_mask.sum() + 1e-8                     # scalar
            k_start  = 1                                        # skip k=0 (background)
        else:
            fg_mask  = torch.ones(B, N, dtype=routing_probs.dtype,
                                  device=routing_probs.device)
            fg_count = float(B * N)
            k_start  = 0

        n_concepts = K - k_start    # number of concepts in the average
        loss       = routing_probs.new_zeros(())

        for k in range(k_start, K):
            # Binary GT for concept k: 1 where token belongs to k, else 0.
            gt_k = (seg_labels == k).float()                          # [B, N]

            # Routing probability for concept k, clamped for BCE stability.
            p_k  = routing_probs[..., k].clamp(self._PROB_EPS, self._PROB_MAX)  # [B, N]

            # Per-token BCE: -[gt·log(p) + (1-gt)·log(1-p)]
            bce = -(gt_k * p_k.log() + (1.0 - gt_k) * (1.0 - p_k).log())  # [B, N]

            # Focal modulation: down-weight tokens the router already handles well.
            # p_correct = routing probability for the correct class.
            # (1-p_correct)^γ → 0 for easy tokens; → 1 for hard tokens.
            if self.focal_gamma > 0.0:
                p_correct    = torch.where(gt_k.bool(), p_k, 1.0 - p_k)  # [B, N]
                focal_weight = (1.0 - p_correct).pow(self.focal_gamma)    # [B, N]
                bce          = focal_weight * bce

            # Apply foreground mask and accumulate mean per foreground token.
            loss = loss + (fg_mask * bce).sum() / fg_count

        return loss / n_concepts
