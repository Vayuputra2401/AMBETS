"""
entropy.py — L_entropy_reg: routing entropy regularization.

L_entropy_reg encourages the model to express genuine confidence (low entropy)
for unambiguous interior voxels and genuine uncertainty (high entropy) at tumor
boundaries.  This keeps routing entropy meaningful as a calibrated uncertainty
signal, which is required for Proposition 2 to hold.

Without this regularisation, routing entropy can drift toward a constant value
that carries no information about spatial uncertainty.

Math (from CCR plan Section 7.1)
---------------------------------
    H(b, t) = −Σ_k P(b, t→k) × log(P(b, t→k) + ε)    [B, N]
    L_entropy_reg = (1/BN) × Σ_{b,t} H(b, t)

    H(t) ∈ [0, log K]:
      H = 0       when routing is maximally confident (one-hot, single expert)
      H = log K   when routing is maximally uncertain (uniform across all experts)

IMPORTANT constraint on λ₃:
    λ₃ is kept at 0.01 — intentionally small.  Larger λ₃ pushes the model toward
    uniform routing (maximum entropy), destroying Proposition 2 and the concept
    alignment signal built by L_align.  The small value maintains a weak pressure
    toward confident routing without suppressing genuine boundary uncertainty.

Plan reference: CCR-Net_Research_Plan.md Section 7.1, entropy_regularization code block.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class EntropyRegularizationLoss(nn.Module):
    """
    L_entropy_reg = mean routing entropy over all tokens.

    Parameters
    ----------
    eps : float
        Numerical stability term added inside log to prevent log(0).

    Inputs
    ------
    routing_probs : Tensor [B, N, K]
        Soft routing probabilities.  Values ∈ (0, 1), sum to 1 along K.

    Returns
    -------
    Tensor []   scalar mean entropy  ∈ [0, log K]
    """

    def __init__(self, eps: float = 1e-8) -> None:
        super().__init__()
        self.eps = eps

    def forward(self, routing_probs: torch.Tensor) -> torch.Tensor:
        """
        Compute mean routing entropy.

        H(b, t) = −Σ_k P(b, t→k) × log(P(b, t→k) + ε)
        L_entropy_reg = mean_{b,t} H(b, t)

        Parameters
        ----------
        routing_probs : Tensor [B, N, K]

        Returns
        -------
        Tensor []   scalar ∈ [0, log K]
        """
        # Per-token entropy: sum over the concept dimension K
        entropy = -(
            routing_probs * (routing_probs + self.eps).log()
        ).sum(dim=-1)  # [B, N]

        # Mean over all tokens and batch items
        return entropy.mean()  # scalar
