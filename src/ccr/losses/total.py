"""
total.py — CCRTotalLoss: curriculum-weighted combination of all CCR loss components.

L_total = L_seg + λ₁·L_align + λ₂·L_diversity + λ₃·L_entropy_reg + λ₄·L_boundary
        + tau_reg_weight·(τ_actual − τ_target)²   [only when tau_current is provided]

The weight tuple (λ₁, λ₂, λ₃, λ₄) transitions through three phases defined by
CurriculumWeightScheduler.  L_seg is always weighted 1.0.

The curriculum prevents expert collapse during early training:
  - Warmup (epochs 1-10):     λ₁=0   — encoder learns features before routing is forced
  - Alignment (epochs 11-50): λ₁=1.0 — routing semantics enforced
  - Refinement (epochs 51-80): λ₁=0.5, λ₄=1.0 — boundary precision phase

Temperature annealing regularization (Improvement 3, docs/phase1_analysis.md §5):
  Pass tau_current = module.router.temperature to activate soft annealing toward
  the phase-appropriate τ target (warmup=2.0 → alignment=1.0 → refinement=0.5).
  If tau_current is None (default), tau regularization is skipped — backward
  compatible with all existing tests and usage.

Note: L_boundary requires 3D spatial predictions [B, K, H, W, D].  When operating
in pure token space (e.g. before building the full network), pass boundary_weight=0
or use SegmentationLoss directly.

Plan reference: CCR-Net_Research_Plan.md Section 7.1 and Table 7.2.
Improvement reference: docs/phase1_analysis.md §5, Improvement 3.
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn

from ccr.config.ccr_config import CCRConfig
from ccr.losses.alignment    import ConceptAlignmentLoss
from ccr.losses.boundary     import BoundaryAwareLoss
from ccr.losses.contrastive  import ContrastiveConceptLoss
from ccr.losses.diversity    import ExpertDiversityLoss
from ccr.losses.entropy      import EntropyRegularizationLoss
from ccr.losses.segmentation import SegmentationLoss
from ccr.utils.weight_schedule import CurriculumWeightScheduler


class CCRTotalLoss(nn.Module):
    """
    Curriculum-weighted total loss for CCR training.

    The five components are:
        L_seg          — segmentation accuracy  (always weighted 1.0)
        L_align        — routing-concept alignment  (λ₁, foreground-only focal)
        L_diversity    — expert diversity / load balance  (λ₂)
        L_entropy_reg  — entropy regularisation  (λ₃)
        L_boundary     — boundary-aware cross-entropy  (λ₄)
        L_tau_reg      — soft temperature annealing  (optional)

    Parameters
    ----------
    config : CCRConfig
        Provides LossConfig (individual hyperparameters) and CurriculumConfig
        (epoch boundaries and temperature targets).

    Usage
    -----
        loss_fn = CCRTotalLoss(config)
        loss_fn.set_epoch(current_epoch)   # once per epoch, before training loop

        outputs = loss_fn(
            pred_logits   = ...,            # [B, K, H, W, D]
            labels        = ...,            # [B, H, W, D]
            routing_probs = ...,            # [B, N, K]
            token_labels  = ...,            # [B, N]
            tau_current   = module.router.temperature,  # optional Tensor scalar
        )
        outputs['total'].backward()

    Inputs
    ------
    pred_logits   : Tensor [B, K, H, W, D]
        Final 3D segmentation logits (after boundary head / decoder).
    labels        : Tensor [B, H, W, D]
        Integer ground-truth class labels.
    routing_probs : Tensor [B, N, K]
        Soft routing probabilities from CCRBottleneckModule.
    token_labels  : Tensor [B, N]
        Ground-truth class label for each bottleneck token.
        Obtained by downsampling the full-resolution label map to token resolution
        (nearest-neighbour 3D downsampling preserves class boundaries).
    tau_current : Tensor scalar, optional
        Current temperature value (module.router.temperature).
        When provided, adds L_tau_reg = tau_reg_weight·(τ−τ_target)² to total.
        When None (default), tau regularization is skipped.

    Returns
    -------
    dict with keys:
        total       : Tensor []   L_total (call .backward() on this)
        seg         : Tensor []   L_seg component
        align       : Tensor []   L_align component (0 during warmup)
        diversity   : Tensor []   L_diversity component
        entropy     : Tensor []   L_entropy_reg component (0 during warmup)
        boundary    : Tensor []   L_boundary component (0 during warmup)
        contrastive : Tensor []   L_contrastive NCR-ET component (0 during warmup or if weight=0)
        tau_reg     : Tensor []   L_tau_reg component (0.0 if tau_current is None)
        phase       : str         current curriculum phase name
        weights     : tuple       (λ₁, λ₂, λ₃, λ₄) active this epoch
        tau_target: float       current τ target (0.0 if tau_current is None)
    """

    def __init__(self, config: CCRConfig) -> None:
        super().__init__()
        self._config     = config
        num_concepts     = config.router.num_concepts
        lc               = config.loss

        self.seg_loss       = SegmentationLoss(lc)
        self.align_loss     = ConceptAlignmentLoss(
            num_concepts    = num_concepts,
            foreground_only = lc.align_foreground_only,
            focal_gamma     = lc.align_focal_gamma,
            concept_weights = lc.align_concept_weights,
        )
        self.diversity_loss = ExpertDiversityLoss(
            num_concepts = num_concepts,
            eps          = lc.diversity_eps,
        )
        self.entropy_loss      = EntropyRegularizationLoss(eps=lc.entropy_eps)
        self.boundary_loss     = BoundaryAwareLoss(
            num_classes    = num_concepts,
            dilation_iters = lc.boundary_dilation_iters,
            base_weight    = lc.boundary_base_weight,
            boost_factor   = lc.boundary_boost_factor,
        )
        self.contrastive_loss  = ContrastiveConceptLoss(ncr_idx=1, et_idx=num_concepts - 1)
        self.scheduler         = CurriculumWeightScheduler(config.curriculum, lc.weights)
        self._tau_reg_weight   = lc.tau_reg_weight
        self._contrastive_w    = lc.contrastive_ncr_et_weight
        self._current_epoch: int = 1

    def set_epoch(self, epoch: int) -> None:
        """
        Set the current training epoch to determine active loss weights and τ target.
        Must be called once before each epoch's training loop.
        """
        self._current_epoch = epoch

    def forward(
        self,
        pred_logits:   torch.Tensor,
        labels:        torch.Tensor,
        routing_probs: torch.Tensor,
        token_labels:  torch.Tensor,
        tau_current:   Optional[torch.Tensor] = None,
    ) -> Dict[str, object]:
        """
        Compute L_total and all components for the current epoch.
        """
        lam_align, lam_diversity, lam_entropy, lam_boundary = (
            self.scheduler.get_weights(self._current_epoch)
        )
        phase      = self.scheduler.get_phase_name(self._current_epoch)
        tau_target = self.scheduler.get_tau_target(self._current_epoch)

        # --- L_seg (always computed) ---
        seg_out = self.seg_loss(pred_logits, labels)
        l_seg   = seg_out["total"]

        # --- L_align (skip during warmup for speed and collapse prevention) ---
        if lam_align > 0.0:
            l_align        = self.align_loss(routing_probs, token_labels)
            l_contrastive  = self.contrastive_loss(routing_probs, token_labels)
        else:
            l_align        = pred_logits.new_zeros(())
            l_contrastive  = pred_logits.new_zeros(())

        # --- L_diversity (always computed; prevents collapse from epoch 1) ---
        l_diversity = self.diversity_loss(routing_probs)

        # --- L_entropy_reg ---
        if lam_entropy > 0.0:
            l_entropy = self.entropy_loss(routing_probs)
        else:
            l_entropy = pred_logits.new_zeros(())

        # --- L_boundary ---
        if lam_boundary > 0.0:
            l_boundary = self.boundary_loss(pred_logits, labels)
        else:
            l_boundary = pred_logits.new_zeros(())

        # --- L_tau_reg (optional temperature annealing regularization) ---
        # Soft pull: (τ_actual − τ_target)² penalises deviation from the
        # phase-appropriate temperature target.  Only active when training
        # loop passes tau_current = module.router.temperature.
        if tau_current is not None:
            tau_clamped = tau_current.clamp(0.1, 5.0)
            l_tau_reg   = self._tau_reg_weight * (tau_clamped - tau_target) ** 2
        else:
            l_tau_reg = pred_logits.new_zeros(())

        # --- Weighted sum ---
        l_total = (
            l_seg
            + lam_align              * l_align
            + lam_diversity          * l_diversity
            + lam_entropy            * l_entropy
            + lam_boundary           * l_boundary
            + self._contrastive_w    * l_contrastive
            + l_tau_reg
        )

        return {
            "total":       l_total,
            "seg":         l_seg,
            "align":       l_align,
            "diversity":   l_diversity,
            "entropy":     l_entropy,
            "boundary":    l_boundary,
            "contrastive": l_contrastive,
            "tau_reg":     l_tau_reg,
            "phase":       phase,
            "weights":     (lam_align, lam_diversity, lam_entropy, lam_boundary),
            "tau_target":  tau_target,
        }
