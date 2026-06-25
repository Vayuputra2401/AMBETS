"""
weight_schedule.py — CurriculumWeightScheduler: phase-based loss weight transitions.

Returns the active (λ_align, λ_diversity, λ_entropy_reg, λ_boundary) tuple for
any given training epoch, based on the three-phase curriculum in CCR plan Table 7.2.
Also returns the temperature target τ for soft annealing (Improvement 3).

Phase definitions
-----------------
  Warmup     epoch ∈ [1, warmup_end]:             λ₁=0,   λ₂=0.5, λ₃=0,    λ₄=0
  Alignment  epoch ∈ (warmup_end, align_end]:     λ₁=1.0, λ₂=0.5, λ₃=0.01, λ₄=0.3
  Refinement epoch ∈ (align_end, total_epochs]:   λ₁=0.5, λ₂=0.1, λ₃=0.01, λ₄=1.0

Temperature targets (Improvement 3, docs/phase1_analysis.md §5)
----------------------------------------------------------------
Coupling τ to the curriculum makes the three phases structurally coherent:
  Warmup     τ_target = tau_warmup    (default 2.0)  → diffuse routing, free exploration
  Alignment  τ_target = tau_alignment (default 1.0)  → routing tightens as CAS rises
  Refinement τ_target = tau_refinement(default 0.5)  → crisp concept maps at inference

The soft tau regularization is applied in CCRTotalLoss when tau_current is passed:
    L_tau_reg = tau_reg_weight × (τ_actual − τ_target)²

λ₁=0 during warmup is critical.  If routing alignment is enforced before the encoder
has learned discriminative features, the router receives gradient signals from a
random encoder and collapses all experts to the same attractor.  The warmup
phase lets the encoder learn useful representations first.

Plan reference: CCR-Net_Research_Plan.md Section 7.2, Table 7.2.
Improvement reference: docs/phase1_analysis.md §5, Improvement 3.
"""

from __future__ import annotations

from typing import Tuple

from ccr.config.ccr_config import CurriculumConfig


class CurriculumWeightScheduler:
    """
    Returns per-epoch loss weight tuples (λ₁, λ₂, λ₃, λ₄) and τ targets.

    Tuple index mapping:
        [0] λ_align      — ConceptAlignmentLoss
        [1] λ_diversity  — ExpertDiversityLoss
        [2] λ_entropy    — EntropyRegularizationLoss
        [3] λ_boundary   — BoundaryAwareLoss

    Parameters
    ----------
    config : CurriculumConfig
        warmup_end_epoch    — last warmup epoch (inclusive)
        alignment_end_epoch — last alignment epoch (inclusive)
        total_epochs        — last refinement epoch (inclusive)
        tau_warmup          — temperature target for warmup phase
        tau_alignment       — temperature target for alignment phase
        tau_refinement      — temperature target for refinement phase
    """

    def __init__(self, config: CurriculumConfig, weights=None) -> None:
        self._warmup_end     = config.warmup_end_epoch
        self._align_end      = config.alignment_end_epoch
        self._total          = config.total_epochs
        self._tau_warmup     = config.tau_warmup
        self._tau_alignment  = config.tau_alignment
        self._tau_refinement = config.tau_refinement
        # Per-phase loss-weight tuples MUST come from the (possibly overridden)
        # config — otherwise ablations that zero a weight are silently ignored.
        # Falls back to defaults when not provided (identical to prior behaviour).
        from ccr.config.ccr_config import LossWeights
        self._weights = weights if weights is not None else LossWeights()

    def get_weights(self, epoch: int) -> Tuple[float, float, float, float]:
        """
        Return (λ_align, λ_diversity, λ_entropy_reg, λ_boundary) for this epoch.

        Parameters
        ----------
        epoch : int   1-indexed training epoch

        Returns
        -------
        Tuple[float, float, float, float]

        Raises
        ------
        ValueError  if epoch < 1 or epoch > total_epochs
        """
        if epoch < 1:
            raise ValueError(f"epoch must be ≥ 1, got {epoch}")
        if epoch > self._total:
            raise ValueError(
                f"epoch {epoch} exceeds total_epochs {self._total}"
            )

        if epoch <= self._warmup_end:
            return self._weights.warmup
        elif epoch <= self._align_end:
            return self._weights.alignment
        else:
            return self._weights.refinement

    def get_phase_name(self, epoch: int) -> str:
        """
        Return the curriculum phase name for the given epoch.

        Returns
        -------
        str   one of {'warmup', 'alignment', 'refinement'}
        """
        if epoch < 1 or epoch > self._total:
            raise ValueError(
                f"epoch {epoch} out of range [1, {self._total}]"
            )
        if epoch <= self._warmup_end:
            return "warmup"
        elif epoch <= self._align_end:
            return "alignment"
        else:
            return "refinement"

    def get_tau_target(self, epoch: int) -> float:
        """
        Return the soft temperature target τ for the current curriculum phase.

        The temperature annealing regularization in CCRTotalLoss uses this target
        to gently pull the learned τ toward the phase-appropriate value:
            L_tau_reg = tau_reg_weight × (τ_actual − τ_target)²

        Phase targets:
          Warmup     → tau_warmup    (default 2.0): diffuse routing, max exploration
          Alignment  → tau_alignment (default 1.0): moderate sharpening with CAS
          Refinement → tau_refinement(default 0.5): crisp maps for clean inference

        Parameters
        ----------
        epoch : int   1-indexed training epoch

        Returns
        -------
        float   τ target value
        """
        phase = self.get_phase_name(epoch)
        return {
            "warmup":     self._tau_warmup,
            "alignment":  self._tau_alignment,
            "refinement": self._tau_refinement,
        }[phase]

    def is_align_active(self, epoch: int) -> bool:
        """
        Return True if L_align should be computed this epoch (i.e. λ₁ > 0).

        After warmup ends, routing semantics are enforced by L_align.
        Before warmup ends, computing L_align is wasteful (λ₁=0) and risks
        destabilising the encoder through noisy gradients.
        """
        return epoch > self._warmup_end
