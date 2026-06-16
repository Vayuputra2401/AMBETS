"""
diversity.py — L_diversity: expert diversity loss preventing two MoE failure modes.

MoE training has two well-known failure modes:
  1. Expert collapse   — all tokens route to a single expert (load imbalance)
  2. Expert redundancy — multiple experts learn identical representations (overlap)

L_diversity directly addresses both.

Math (from CCR plan Section 7.1, diversity_loss code block)
-----------------------------------------------------------
Given routing_probs P ∈ ℝ^{B×N×K}:

Component 1 — Load Balancing:
    mean_k = (1/BN) × Σ_{b,t} P(b, t → k)     per-expert average usage  [K]
    load_balance = (1/K) × Σ_k (mean_k − 1/K)²
    Minimised when all experts receive exactly 1/K fraction of tokens (uniform load).

Component 2 — Cosine Overlap:
    Flatten P to F ∈ ℝ^{K × BN}  (one routing vector per expert over all tokens)
    G = F @ Fᵀ                    Gram matrix  [K, K]
    cosine_sim[i,j] = G[i,j] / (||F_i|| × ||F_j|| + ε)
    cosine_overlap = mean of cosine_sim[i≠j]
    Penalises pairs of experts with similar routing patterns (high cosine similarity).
    Minimised when all pairs of expert routing vectors are orthogonal.

    L_diversity = load_balance + cosine_overlap

Plan reference: CCR-Net_Research_Plan.md Section 7.1.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ExpertDiversityLoss(nn.Module):
    """
    L_diversity = load_balance_loss + cosine_overlap_loss

    Parameters
    ----------
    num_concepts : int
        K — must equal routing_probs.shape[-1].
    eps : float
        Numerical stability term in cosine similarity denominator.

    Inputs
    ------
    routing_probs : Tensor [B, N, K]
        Soft routing probabilities from ClinicalConceptRouter (training mode).

    Returns
    -------
    Tensor []   scalar diversity loss  ≥ 0
    """

    def __init__(self, num_concepts: int, eps: float = 1e-8) -> None:
        super().__init__()
        self.num_concepts = num_concepts
        self.eps = eps

    def forward(self, routing_probs: torch.Tensor) -> torch.Tensor:
        """
        Compute L_diversity = load_balance + cosine_overlap.

        Parameters
        ----------
        routing_probs : Tensor [B, N, K]

        Returns
        -------
        Tensor []   scalar  ≥ 0
        """
        B, N, K = routing_probs.shape
        assert K == self.num_concepts

        # ----------------------------------------------------------------
        # Component 1: Load Balancing Loss
        # Target: each expert receives 1/K of all tokens on average.
        # ----------------------------------------------------------------

        # Average routing probability per expert over all tokens and batch items
        mean_probs = routing_probs.mean(dim=[0, 1])         # [K]  ∈ [0, 1]

        # Squared deviation from uniform load (1/K per expert)
        load_balance = ((mean_probs - 1.0 / K) ** 2).mean()  # scalar

        # ----------------------------------------------------------------
        # Component 2: Cosine Overlap Loss
        # Reshape to [K, B*N]: one routing-probability vector per expert.
        # Penalise high cosine similarity between any two expert vectors.
        # ----------------------------------------------------------------

        # [B, N, K] → [B*N, K] → [K, B*N]  (contiguous for matmul)
        flat = routing_probs.reshape(B * N, K).T.contiguous()  # [K, B*N]

        # Gram matrix: G[i,j] = F_i · F_j (dot product between expert routing vectors)
        gram = flat @ flat.T                              # [K, K]

        # Norms of each expert routing vector: [K, 1]
        norms = gram.diagonal().sqrt().unsqueeze(1)       # [K, 1]

        # Normalised cosine similarity: G[i,j] / (||F_i|| × ||F_j||)
        cosine_sim = gram / (norms @ norms.T + self.eps)  # [K, K]

        # Mask out diagonal (self-similarity = 1, not penalised)
        off_diag_mask = ~torch.eye(K, dtype=torch.bool, device=routing_probs.device)

        # Mean of all off-diagonal cosine similarities
        # Penalises expert pairs with similar routing patterns
        cosine_overlap = cosine_sim[off_diag_mask].mean()  # scalar

        return load_balance + cosine_overlap
