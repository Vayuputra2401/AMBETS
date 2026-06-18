"""
contrastive.py — L_contrastive: pairwise NCR-ET routing disambiguation loss.

Problem this solves
-------------------
At the 16³ bottleneck, NCR and ET tokens are adjacent in space and share a
similar T1ce appearance at coarse resolution (8³ voxels/token).  L_align
treats each concept independently and does not explicitly push NCR and ET
routing probabilities apart.  The router can achieve moderate L_align by
assigning similar probabilities to both concepts for Tumor Core tokens.

L_contrastive adds a direct disambiguation signal: for tokens whose GT label
is NCR (k=1) or ET (k=3), it maximises the log-ratio of the correct concept
over the competing one in a binary softmax over {NCR, ET}:

    For NCR token:  loss_t = -log( P(t→NCR) / (P(t→NCR) + P(t→ET)) )
    For ET  token:  loss_t = -log( P(t→ET)  / (P(t→NCR) + P(t→ET)) )

This is equivalent to binary cross-entropy restricted to the NCR-ET pair.
A perfect router gives loss_t → 0; an ambiguous router (P_NCR ≈ P_ET) gives
loss_t → log(2) ≈ 0.693.

Only NCR and ET tokens contribute — background and edema are unaffected.
The loss is zero during warmup (called only when lam_align > 0 in total.py).

Plan reference: CCR-Net_Research_Plan.md — Run 2 NCR fix, contrastive term.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ContrastiveConceptLoss(nn.Module):
    """
    Binary contrastive loss separating NCR and ET routing probabilities.

    Parameters
    ----------
    ncr_idx : int
        Concept index for Necrotic Core.  Default 1 (BraTS).
    et_idx : int
        Concept index for Enhancing Tumor.  Default 3 (BraTS).
    eps : float
        Numerical stability epsilon inside log().

    Inputs
    ------
    routing_probs : Tensor [B, N, K]
        Soft routing probabilities.
    seg_labels : Tensor [B, N]
        Integer ground-truth labels per token ∈ {0, …, K-1}.

    Returns
    -------
    Tensor []   scalar contrastive loss ≥ 0.
    """

    def __init__(
        self,
        ncr_idx: int = 1,
        et_idx:  int = 3,
        eps:     float = 1e-8,
    ) -> None:
        super().__init__()
        self.ncr_idx = ncr_idx
        self.et_idx  = et_idx
        self.eps     = eps

    def forward(
        self,
        routing_probs: torch.Tensor,
        seg_labels:    torch.Tensor,
    ) -> torch.Tensor:
        ncr_mask = (seg_labels == self.ncr_idx).float()   # [B, N]
        et_mask  = (seg_labels == self.et_idx).float()    # [B, N]
        n_tokens = (ncr_mask + et_mask).sum() + self.eps

        p_ncr = routing_probs[..., self.ncr_idx]           # [B, N]
        p_et  = routing_probs[..., self.et_idx]            # [B, N]
        denom = p_ncr + p_et + self.eps

        # Binary log-softmax over {NCR, ET} pair
        loss_ncr = -torch.log(p_ncr / denom + self.eps) * ncr_mask
        loss_et  = -torch.log(p_et  / denom + self.eps) * et_mask

        return (loss_ncr + loss_et).sum() / n_tokens
