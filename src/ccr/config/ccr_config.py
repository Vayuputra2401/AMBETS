"""
ccr_config.py — Typed configuration dataclasses for the Clinical Concept Routing (CCR) system.

Every hyperparameter that appears in the CCR research plan is surfaced here with its
default value, type, and a reference to the plan section that justifies it.  All other
code imports from here rather than hard-coding values.

References
----------
CCR-Net_Research_Plan.md:
  - Section 6.1  : router/expert architecture and embed_dim
  - Section 7.1  : loss function formulas and parameters
  - Section 7.2  : curriculum phase weight table (Table 7.2)
  - Section 9    : dataset-specific K values
  - Section 16.1 : expert collapse threshold (< 5 % after epoch 20)

Phase 1 improvements (2026-05-28):
  - RouterConfig.use_prototypes        : prototype-augmented routing
  - LossConfig.align_foreground_only   : L_align on foreground tokens only (k>0)
  - LossConfig.align_focal_gamma       : focal modulation in L_align
  - LossConfig.tau_reg_weight          : weight for temperature annealing regularization
  - CurriculumConfig.tau_{phase}       : temperature targets per training phase
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple


# ---------------------------------------------------------------------------
# Sub-configurations — each maps to one part of the CCR system
# ---------------------------------------------------------------------------


@dataclass
class RouterConfig:
    """
    Configuration for ClinicalConceptRouter.

    The router is a lightweight MLP placed at the encoder bottleneck.
    It maps each token embedding to a probability distribution over K clinical
    concepts.  The resulting distribution IS the per-voxel explanation.

    Plan reference: Section 6.1 — ClinicalConceptRouter architecture.

    Attributes
    ----------
    embed_dim : int
        Dimensionality of bottleneck token embeddings (D).
        Default 192 matches Swin-B encoder output dimension.
    num_concepts : int
        Number of clinical concepts K.
        K=4 for BraTS: {Background, NCR, Edema, ET}.
        K=3 for LiTS : {Background, Liver, Liver Tumor}.
    hidden_dim_factor : float
        Router hidden dimension = embed_dim * hidden_dim_factor.
        Factor 2.0 → hidden_dim = 384 for embed_dim = 192.
    temperature_init : float
        Initial value of the learnable temperature parameter τ.
        τ controls the sharpness of the routing distribution:
          τ → 0  :  one-hot (maximally sharp / confident)
          τ → ∞  :  uniform (maximally diffuse / uncertain)
        Clamped to [0.1, 5.0] during forward pass to prevent degeneracy.
    temperature_learnable : bool
        If True, τ is an nn.Parameter updated by the optimizer.
        If False, τ is a fixed constant (temperature_init) for ablations.
    use_prototypes : bool
        If True, augment MLP routing logits with distance-to-prototype logits.
        K learnable prototype vectors {c_k} ∈ ℝ^D are maintained.  Routing
        blends MLP logits with negative squared-distance to the nearest prototype:
            logits = α·logits_mlp + (1−α)·logits_proto
        where α is a learnable blend scalar and logits_proto = −||x − c_k||² / √D.
        Prototypes converge under L_align to the mean bottleneck representation
        of each clinical concept, making routing geometrically interpretable.
    """

    embed_dim: int = 192
    num_concepts: int = 4
    hidden_dim_factor: float = 2.0
    temperature_init: float = 1.0
    temperature_learnable: bool = True
    use_prototypes: bool = True


@dataclass
class ExpertConfig:
    """
    Configuration for ClinicalConceptExpert (one instance per concept).

    Each expert is a 2-layer residual MLP.  The residual connection prevents
    information loss during early training before experts have specialised.

    Plan reference: Section 6.1 — ClinicalConceptExpert architecture.

    Attributes
    ----------
    embed_dim : int
        Must equal RouterConfig.embed_dim (both operate on the same tokens).
    num_concepts : int
        Must equal RouterConfig.num_concepts (one expert instantiated per concept).
    expert_hidden_factor : float
        Expert hidden dimension = embed_dim * expert_hidden_factor.
        Default 2.667 → hidden_dim ≈ 512 for embed_dim = 192.
    """

    embed_dim: int = 192
    num_concepts: int = 4
    expert_hidden_factor: float = 2.667


@dataclass
class LossWeights:
    """
    Curriculum-phase loss weight tuples (λ₁, λ₂, λ₃, λ₄).

    Tuple index mapping:
        [0] λ_align      — ConceptAlignmentLoss
        [1] λ_diversity  — ExpertDiversityLoss
        [2] λ_entropy    — EntropyRegularizationLoss
        [3] λ_boundary   — BoundaryAwareLoss

    L_seg is always weighted 1.0 and is not included in this tuple.

    Plan reference: Section 7.2, Table 7.2.
    """

    warmup: Tuple[float, float, float, float] = (0.0, 0.5, 0.00, 0.0)
    """Epochs 1-10.  L_align = 0 so the encoder learns features before routing."""

    alignment: Tuple[float, float, float, float] = (1.0, 0.5, 0.01, 0.3)
    """Epochs 11-50.  Full alignment loss; λ_entropy kept at 0.01 (small)."""

    refinement: Tuple[float, float, float, float] = (1.0, 0.1, 0.01, 1.0)
    """Epochs alignment_end+1 .. total.  L_boundary raised to 1.0; lam_align kept at 1.0.
    Rationale (Run 2 finding): halving lam_align to 0.5 caused NCR CAS to drift from its
    peak (0.706→0.675 over 30 epochs).  Keeping lam_align=1.0 maintains routing pressure
    through the full run without hurting Dice (Edema/ET CAS already ≥0.90 by this phase)."""


@dataclass
class LossConfig:
    """
    All hyperparameters for the five CCR loss components.

    Plan reference: Section 7.1 — loss functions and formulas.

    L_total = L_seg + λ₁·L_align + λ₂·L_diversity + λ₃·L_entropy_reg + λ₄·L_boundary
            + τ_reg_weight · (τ − τ_target)²   [optional, when tau_current is provided]

    Attributes
    ----------
    focal_gamma : float
        Focusing parameter γ in FocalLoss.  γ=2 is standard per Lin et al. 2017.
    focal_alpha : float
        Class-balancing weight α in FocalLoss.  Applied uniformly across classes.
    dice_smooth : float
        Laplace smoothing ε in DiceLoss to avoid 0/0.
    diversity_eps : float
        Numerical stability ε in cosine similarity computation of L_diversity.
    entropy_eps : float
        Numerical stability ε inside log() in L_entropy_reg to prevent log(0).
    boundary_dilation_iters : int
        Number of morphological dilation iterations that widen the boundary zone.
        More iterations → wider boundary zone → more boundary-weighted voxels.
    boundary_base_weight : float
        Cross-entropy weight for non-boundary voxels.
    boundary_boost_factor : float
        Additional cross-entropy weight added at boundary voxels.
        Total boundary weight = base_weight + boost_factor.
    align_foreground_only : bool
        If True, L_align is computed only on non-background tokens (labels > 0)
        and only for concepts k=1..K-1.  Background tokens dominate BCE and the
        hard alignment problem is distinguishing tumor subregions.  Foreground-only
        L_align concentrates gradient signal where the paper's claim lives.
    align_focal_gamma : float
        Focal modulation exponent for L_align BCE.  Down-weights already-correct
        tokens; focuses alignment gradient on hard boundary/transition tokens.
        Set to 0.0 to disable focal modulation (plain BCE).
    tau_reg_weight : float
        Weight for the soft temperature annealing regularization term.
        Loss += tau_reg_weight × (τ_actual − τ_target)².
        Only applied when tau_current is passed to CCRTotalLoss.forward().
    weights : LossWeights
        Curriculum phase weight tuples.
    """

    focal_gamma: float = 2.0
    focal_alpha: float = 0.25
    dice_smooth: float = 1e-5
    diversity_eps: float = 1e-8
    entropy_eps: float = 1e-8
    boundary_dilation_iters: int = 3
    boundary_base_weight: float = 1.0
    boundary_boost_factor: float = 5.0
    align_foreground_only: bool = True
    align_focal_gamma: float = 2.0
    align_concept_weights: Optional[Tuple[float, ...]] = None
    contrastive_ncr_et_weight: float = 0.0
    tau_reg_weight: float = 0.01
    weights: LossWeights = field(default_factory=LossWeights)


@dataclass
class CurriculumConfig:
    """
    Epoch boundaries for the three training phases, and temperature targets.

    Plan reference: Section 7.2 — curriculum training strategy.

    Attributes
    ----------
    warmup_end_epoch : int
        Epochs 1..warmup_end_epoch  → warmup phase  (L_align weight = 0).
    alignment_end_epoch : int
        Epochs warmup+1..alignment_end → alignment phase (L_align weight = 1.0).
    total_epochs : int
        Epochs alignment+1..total_epochs → refinement phase (L_boundary = 1.0).
    tau_warmup : float
        Soft temperature target during warmup.  High value → diffuse routing →
        experts explore freely before specialisation pressure is applied.
    tau_alignment : float
        Soft temperature target during alignment.  Routing tightens as CAS rises.
    tau_refinement : float
        Soft temperature target during refinement.  Low value → crisp routing →
        deterministic concept maps at inference.

    The warmup rationale: starting L_align from epoch 1 causes expert collapse
    because the encoder has not yet learned discriminative features.  The routing
    MLP would receive random gradients and all experts collapse to the same solution.
    """

    warmup_end_epoch: int = 10
    alignment_end_epoch: int = 50
    total_epochs: int = 80
    tau_warmup: float = 2.0
    tau_alignment: float = 1.0
    tau_refinement: float = 0.5


@dataclass
class CCRConfig:
    """
    Top-level configuration that aggregates all sub-configurations.

    This is the single object passed to CCRBottleneckModule, CCRTotalLoss,
    CurriculumWeightScheduler, and all metric trackers.

    Plan reference: Section 6 (architecture), Section 7 (training), Section 9 (datasets).

    Attributes
    ----------
    router : RouterConfig
    expert : ExpertConfig
    loss : LossConfig
    curriculum : CurriculumConfig
    concept_names : Tuple[str, ...]
        Human-readable clinical concept labels in concept-index order.
        Must have length == num_concepts.
        BraTS: ("background", "necrotic_core", "edema", "enhancing_tumor")
        LiTS : ("background", "liver", "liver_tumor")
    hard_routing_inference : bool
        If True, the dispatcher uses argmax (hard) routing during model.eval().
        If False, soft routing is used throughout (useful for calibration analysis).
    """

    router: RouterConfig = field(default_factory=RouterConfig)
    expert: ExpertConfig = field(default_factory=ExpertConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    curriculum: CurriculumConfig = field(default_factory=CurriculumConfig)
    concept_names: Tuple[str, ...] = (
        "background",
        "necrotic_core",
        "edema",
        "enhancing_tumor",
    )
    hard_routing_inference: bool = True
    enabled: bool = True
    """If False, CCRNet runs as a plain Swin+UNETR backbone (no router/experts, no
    routing losses) — the no-CCR segmentation control (W4) that shows CCR is
    accuracy-neutral. Set via configs/ablations/no_ccr.yaml."""

    def __post_init__(self) -> None:
        # expert.embed_dim must always match router.embed_dim (single source of truth).
        # Auto-sync so callers only need to set router.embed_dim (e.g. from YAML or tests).
        if self.expert.embed_dim != self.router.embed_dim:
            self.expert.embed_dim = self.router.embed_dim
        if self.router.num_concepts != self.expert.num_concepts:
            raise ValueError(
                f"router.num_concepts ({self.router.num_concepts}) must equal "
                f"expert.num_concepts ({self.expert.num_concepts})"
            )
        if self.router.num_concepts != len(self.concept_names):
            raise ValueError(
                f"num_concepts ({self.router.num_concepts}) must equal "
                f"len(concept_names) ({len(self.concept_names)})"
            )
