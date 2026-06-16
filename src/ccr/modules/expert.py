"""
expert.py — ClinicalConceptExpert: per-concept feature refinement network.

Each expert is a 2-layer residual MLP.  One expert instance is created for each
clinical concept k ∈ {0, …, K-1}.  During training (soft dispatch), expert_k
processes every token and its output is weighted by P(t→k).  During inference
(hard dispatch), only the assigned expert processes each token.

Architecture
------------
Input tokens X̃ ∈ ℝ^{M×D}  (M = number of tokens dispatched to this expert)

    norm_out = LayerNorm(X̃)                       [M, D]
    h        = GELU(norm_out @ W₁ + b₁)            [M, H]   H = D × expert_hidden_factor
    out      = h @ W₂ + b₂                         [M, D]
    result   = X̃ + out                             [M, D]   ← residual connection

The residual connection is critical during early training.  Before experts have
specialised, the correction `out` is near-zero and the module acts as an identity.
This prevents gradient instability in the first ~10 epochs (warmup phase).

Plan reference: CCR-Net_Research_Plan.md Section 6.1 — ClinicalConceptExpert.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ccr.config.ccr_config import ExpertConfig


class ClinicalConceptExpert(nn.Module):
    """
    Single per-concept feature refinement expert.

    Refines the bottleneck representations of tokens that the router assigns
    to clinical concept `concept_id`.  The refinement is additive (residual),
    preserving the backbone's original features while adding concept-specific
    adjustments.

    Parameters
    ----------
    config : ExpertConfig
        embed_dim           D — token dimension
        expert_hidden_factor  hidden_dim = embed_dim × expert_hidden_factor
    concept_id : int
        Zero-based clinical concept index.
        BraTS: 0=Background, 1=NCR, 2=Edema, 3=EnhancingTumor.
    concept_name : str
        Human-readable name used in logs and metric outputs.

    Inputs
    ------
    tokens : Tensor [M, D]
        Token embeddings for the M tokens dispatched to this expert.
        In soft dispatch: M = B × N (all tokens, weighted externally).
        In hard dispatch: M = number of tokens whose argmax = concept_id.

    Returns
    -------
    Tensor [M, D]
        Concept-refined token embeddings.
    """

    def __init__(
        self,
        config: ExpertConfig,
        concept_id: int,
        concept_name: str,
    ) -> None:
        super().__init__()
        self.concept_id = concept_id
        self.concept_name = concept_name

        hidden_dim = int(config.embed_dim * config.expert_hidden_factor)

        # Pre-normalisation before the projection (Pre-LN style)
        self.norm = nn.LayerNorm(config.embed_dim)
        self.fc1  = nn.Linear(config.embed_dim, hidden_dim)
        self.fc2  = nn.Linear(hidden_dim, config.embed_dim)

        self._init_weights()

    def _init_weights(self) -> None:
        """
        Near-zero initialisation for fc2 so that at the start of training the
        expert correction is close to zero (identity-like residual).
        This prevents the router from receiving large misleading gradients in
        the first few epochs before experts have begun to specialise.
        """
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.zeros_(self.fc1.bias)
        nn.init.xavier_uniform_(self.fc2.weight, gain=0.01)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        Apply concept-specific feature refinement via a residual MLP.

        Parameters
        ----------
        tokens : Tensor [M, D]
            Token embeddings dispatched to this expert.

        Returns
        -------
        Tensor [M, D]
            Refined embeddings: tokens + expert_correction(tokens).
        """
        # Residual branch: norm → up-project → activate → down-project
        correction = self.fc2(F.gelu(self.fc1(self.norm(tokens))))  # [M, D]

        # Residual addition: output = input + correction
        return tokens + correction  # [M, D]
