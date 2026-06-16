"""
router.py — ClinicalConceptRouter: the core CCR routing mechanism.

This module implements the routing MLP that sits at the encoder bottleneck and
assigns each image token to a probability distribution over K clinically-labeled
expert networks.  The routing probability IS the per-voxel explanation — not a
post-hoc approximation of it.

Mathematical operation
----------------------
Given bottleneck tokens X ∈ ℝ^{B×N×D}:

    1. Normalise:   X̂ = LayerNorm(X)                          [B, N, D]
    2. Project up:  H  = GELU(X̂ @ W₁ + b₁)                   [B, N, H]
                         where H = D × hidden_dim_factor
    3. Project to K: logits_mlp = H @ W₂ + b₂                 [B, N, K]

If use_prototypes=True (default), prototype-augmented routing is applied:

    4. Prototype logits:
           diff         = X̂ − c_k  for each prototype c_k ∈ ℝ^D    [B, N, K, D]
           dist_sq      = ||diff||²                                   [B, N, K]
           logits_proto = −dist_sq / √D                               [B, N, K]
    5. Blend:
           α            = sigmoid(blend_alpha_param)                  scalar ∈ (0,1)
           logits       = α·logits_mlp + (1−α)·logits_proto

    After training with L_align, prototype c_k converges to the mean bottleneck
    representation of concept k tokens, making routing geometrically interpretable:
    you can visualise what each expert's "template" looks like in feature space.

    6. Temperature:  logits_τ = logits / clamp(τ, 0.1, 5.0)   [B, N, K]
    7. Normalise:    P(t→k) = softmax(logits_τ, dim=-1)        [B, N, K]
    8. Entropy:      H(t) = -Σ_k P(t→k) log(P(t→k) + ε)       [B, N]

At training   → returns soft probabilities (differentiable, gradients flow to encoder).
At inference  → caller (CCRBottleneckModule) may convert to hard one-hot via argmax.

Why the bottleneck?
-------------------
Routing at the bottleneck means the routing decision is causally upstream of every
decoder operation.  The decoder receives concept-conditioned representations and
makes predictions that are CONDITIONED on the routing.  This is the opposite of
post-hoc attribution, where explanation follows prediction.

Plan reference: CCR-Net_Research_Plan.md Section 6.1.
Phase 1 improvement: prototype-augmented router (docs/phase1_analysis.md §5, Improvement 2).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ccr.config.ccr_config import RouterConfig


class ClinicalConceptRouter(nn.Module):
    """
    Routing MLP (optionally prototype-augmented) placed at the encoder bottleneck.

    Maps each bottleneck token X[b, t, :] ∈ ℝ^D to a probability distribution
    P(t → k) over K clinical concepts.  This distribution directly constitutes
    the model's clinical concept assignment for token t — it is not derived from
    the prediction; it determines the prediction.

    When use_prototypes=True (default), K learnable prototype vectors augment
    the MLP logits with a distance-to-prototype signal.  Prototypes converge
    under L_align to the mean bottleneck representation of each clinical concept,
    providing a geometrically interpretable explanation of routing decisions.

    Proposition 1a (CCR-Net):  faithfulness = CAS  (exact, by construction)
    Proposition 1b (Retrofit):  faithfulness ≥ CAS × (1 − DD)  (bounded, measurable)

    Parameters
    ----------
    config : RouterConfig
        embed_dim         D — bottleneck token dimension (Swin-B = 192)
        num_concepts      K — clinical labels (BraTS K=4, LiTS K=3)
        hidden_dim_factor   hidden_dim = embed_dim × hidden_dim_factor
        temperature_init  τ₀ — initial temperature (learnable or fixed)
        temperature_learnable — whether τ is an nn.Parameter
        use_prototypes    — whether to augment routing with prototype distances

    Inputs
    ------
    bottleneck_tokens : Tensor [B, N, D]
        Encoder bottleneck token sequence.

    Returns
    -------
    dict with:
        routing_probs : [B, N, K]  soft routing probabilities, sums to 1 along K
        entropy       : [B, N]     routing entropy per token (uncertainty signal)
        logits        : [B, N, K]  pre-softmax blended logits (for debugging)
    """

    _TEMP_MIN: float = 0.1
    _TEMP_MAX: float = 5.0

    def __init__(self, config: RouterConfig) -> None:
        super().__init__()
        self.num_concepts    = config.num_concepts
        self._embed_dim      = config.embed_dim
        self._entropy_eps    = 1e-8
        self._use_prototypes = config.use_prototypes

        hidden_dim = int(config.embed_dim * config.hidden_dim_factor)

        # Routing MLP: LayerNorm → Linear → GELU → Linear
        # Pre-LN style stabilises training for transformer-like inputs.
        self.norm = nn.LayerNorm(config.embed_dim)
        self.fc1  = nn.Linear(config.embed_dim, hidden_dim)
        self.fc2  = nn.Linear(hidden_dim, config.num_concepts)

        # Prototype-augmented routing (Improvement 2, phase1_analysis.md §5).
        # K prototype vectors in ℝ^D, one per clinical concept.
        # blend_alpha: learnable scalar controlling MLP vs. prototype contribution.
        if self._use_prototypes:
            self.prototypes  = nn.Parameter(
                torch.empty(config.num_concepts, config.embed_dim)
            )
            # blend_alpha controls the MLP / prototype mix via sigmoid:
            #   α = sigmoid(blend_alpha) ∈ (0,1)
            #   logits = α·logits_mlp + (1−α)·logits_proto
            # Initialised at 0.0 → α ≈ 0.5: equal initial blend.
            self.blend_alpha = nn.Parameter(torch.zeros(1))

        # Temperature τ: scalar learnable parameter or fixed constant.
        # Clamped during forward to (0.1, 5.0) to prevent degenerate distributions.
        if config.temperature_learnable:
            self.temperature = nn.Parameter(
                torch.full((1,), config.temperature_init)
            )
        else:
            self.register_buffer(
                "temperature",
                torch.full((1,), config.temperature_init),
            )

        self._init_weights()

    def _init_weights(self) -> None:
        """
        Xavier uniform initialisation for routing MLP weights.
        Near-zero fc2 init produces near-uniform initial routing.
        Prototype init spreads K concept vectors evenly across ℝ^D.
        """
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.zeros_(self.fc1.bias)
        nn.init.xavier_uniform_(self.fc2.weight, gain=0.1)
        nn.init.zeros_(self.fc2.bias)

        if self._use_prototypes:
            # Xavier uniform on [K, D]: spreads prototypes across the embedding space.
            # Treats K as out-features and D as in-features for fan calculation.
            nn.init.xavier_uniform_(self.prototypes)

    def forward(self, bottleneck_tokens: torch.Tensor) -> dict:
        """
        Compute routing probabilities for each bottleneck token.

        Parameters
        ----------
        bottleneck_tokens : Tensor [B, N, D]

        Returns
        -------
        dict
            routing_probs : Tensor [B, N, K]  — probability simplex along K
            entropy       : Tensor [B, N]     — H(t) = -Σ P log P  ∈ [0, log K]
            logits        : Tensor [B, N, K]  — blended pre-temperature logits
        """
        # [B, N, D] → normalise → [B, N, D]
        x_norm = self.norm(bottleneck_tokens)

        # ---- MLP branch ----
        # [B, N, D] → [B, N, H] → [B, N, K]
        logits_mlp = self.fc2(F.gelu(self.fc1(x_norm)))  # [B, N, K]

        # ---- Prototype branch (optional) ----
        if self._use_prototypes:
            # Distance of each token to each prototype vector.
            # x_norm: [B, N, D]; prototypes: [K, D]
            # diff:   [B, N, K, D]
            diff    = x_norm.unsqueeze(2) - self.prototypes.unsqueeze(0).unsqueeze(0)
            dist_sq = (diff * diff).sum(dim=-1)                      # [B, N, K]

            # Negative distance as logit, scaled by √D to match MLP logit magnitudes.
            # Higher similarity (smaller distance) → higher logit.
            logits_proto = -dist_sq / (self._embed_dim ** 0.5)       # [B, N, K]

            # Learnable blend: α = sigmoid(blend_alpha) ∈ (0, 1)
            alpha  = self.blend_alpha.sigmoid()                       # scalar
            logits = alpha * logits_mlp + (1.0 - alpha) * logits_proto  # [B, N, K]
        else:
            logits = logits_mlp

        # Temperature scaling: clamp to [0.1, 5.0] to prevent degeneracy
        tau = self.temperature.clamp(self._TEMP_MIN, self._TEMP_MAX)
        routing_probs = F.softmax(logits / tau, dim=-1)              # [B, N, K]

        # Routing entropy: H(t) = -Σ_k P(t→k) log P(t→k)
        entropy = -(
            routing_probs * (routing_probs + self._entropy_eps).log()
        ).sum(dim=-1)                                                 # [B, N]

        return {
            "routing_probs": routing_probs,  # [B, N, K]
            "entropy":       entropy,        # [B, N]
            "logits":        logits,         # [B, N, K]  blended
        }
