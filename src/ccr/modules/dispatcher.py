"""
dispatcher.py — CCRBottleneckModule: the complete CCR bottleneck replacement.

Combines ClinicalConceptRouter with K ClinicalConceptExperts into a single
drop-in module for the encoder bottleneck.  This is the architectural unit
that is swapped into any segmentation model to add faithful intrinsic explanation.

Dispatch modes
--------------
SOFT dispatch (training, model.train()):
    expert_output(t) = Σ_k P(t→k) · Expert_k(X[b, t, :])

    Every expert processes every token; outputs are weighted-summed by routing
    probabilities.  Fully differentiable: gradients flow through:
        expert outputs → weighting by P(t→k) → routing probs → router MLP → encoder.
    This end-to-end gradient flow is required for L_align to shape routing semantics.

HARD dispatch (inference, model.eval(), when config.hard_routing_inference=True):
    k*(b, t) = argmax_k P(t→k)
    expert_output(t) = Expert_{k*(b,t)}(X[b, t, :])

    Each token is processed by exactly one expert.  This produces crisp,
    deterministic concept maps: every voxel is assigned to exactly one clinical
    concept without ambiguity.  Not differentiable; only used for inference.

The soft→hard switch is automatic based on self.training.

Plan reference: CCR-Net_Research_Plan.md Section 6.1 and 6.3.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Dict, Iterator, Optional

import torch
import torch.nn as nn

from ccr.config.ccr_config import CCRConfig
from ccr.modules.expert import ClinicalConceptExpert
from ccr.modules.router import ClinicalConceptRouter


@contextmanager
def routing_intervention(
    module: "CCRBottleneckModule",
    override: Optional[torch.Tensor],
) -> Iterator[None]:
    """
    Temporarily force tokens through experts the router did not choose.

    `override` is an integer tensor [B, N] of target concept ids, where -1 leaves a token
    untouched. Inside the block the module hard-dispatches according to `override`; on exit
    the hook is always cleared, including on exception, so an intervention can never leak
    into a subsequent forward pass.

    This is the causal test of the CCR claim. The routing is asserted to be causally
    upstream of the prediction; if that is true, re-routing tokens from concept a to expert
    b must move the decoder's posterior toward class b over exactly those voxels. The
    experiment is impossible for post-hoc attribution -- there is nothing in a saliency map
    to intervene on -- which is why it isolates what CCR provides.

    Example
    -------
    >>> override = torch.full_like(assignments, -1)
    >>> override[assignments == 1] = 2          # necrosis tokens -> edema expert
    >>> with routing_intervention(model.ccr, override):
    ...     out = model(image)
    """
    previous = module.route_override
    module.route_override = override
    try:
        yield
    finally:
        module.route_override = previous


class CCRBottleneckModule(nn.Module):
    """
    Complete CCR bottleneck replacement module.

    Drop-in replacement for the encoder bottleneck in any encoder-decoder
    segmentation model.  Accepts bottleneck token sequences, performs routing,
    dispatches tokens to concept experts, and returns concept-conditioned
    representations together with the routing evidence.

    Parameters
    ----------
    config : CCRConfig
        Complete CCR configuration (router, expert, curriculum, concept names).

    Inputs
    ------
    bottleneck_tokens : Tensor [B, N, D]
        Encoder bottleneck token sequence.
        B = batch size
        N = number of spatial tokens
        D = token embedding dimension (config.router.embed_dim)

    Returns
    -------
    dict with keys:
        routing_probs  : Tensor [B, N, K]  — THE EXPLANATION.  P(token t → concept k).
                          Soft during training; hard one-hot during eval (if configured).
        expert_outputs : Tensor [B, N, D]  — concept-conditioned bottleneck features.
                          These replace or augment the backbone bottleneck in CCR-Retrofit.
        entropy        : Tensor [B, N]     — routing entropy per token (Proposition 2).
                          High entropy = uncertain concept assignment = boundary region.
        assignments    : Tensor [B, N]     — integer hard concept assignment (argmax).
                          Used for ExpertUtilizationTracker and hard-routing inference.
        logits         : Tensor [B, N, K]  — pre-softmax router logits (diagnostics).
    """

    def __init__(self, config: CCRConfig) -> None:
        super().__init__()
        self.config = config

        # Causal-intervention hook (see `routing_intervention` below). When set to an
        # integer tensor [B, N], it REPLACES the argmax assignment before dispatch, so the
        # token is processed by an expert the router did not choose. Entries of -1 mean
        # "leave this token alone". None (the default) disables the mechanism entirely, so
        # training and ordinary inference are untouched.
        #
        # This is what makes CCR's explanation testable in a way a saliency map is not:
        # you cannot intervene on a heatmap, but you can re-route a token and watch the
        # prediction move. Not a plain attribute on purpose -- it is consumed by forward()
        # and must be cleared by the caller; use the context manager.
        self.route_override: torch.Tensor | None = None

        self.router = ClinicalConceptRouter(config.router)

        self.experts = nn.ModuleList(
            [
                ClinicalConceptExpert(
                    config.expert,
                    concept_id=k,
                    concept_name=name,
                )
                for k, name in enumerate(config.concept_names)
            ]
        )

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def forward(self, bottleneck_tokens: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Route tokens and dispatch to concept experts.

        Parameters
        ----------
        bottleneck_tokens : Tensor [B, N, D]

        Returns
        -------
        dict  (keys: routing_probs, expert_outputs, entropy, assignments, logits)
        """
        B, N, D = bottleneck_tokens.shape

        # --- Step 1: Compute routing probabilities ---
        router_out = self.router(bottleneck_tokens)
        routing_probs = router_out["routing_probs"]   # [B, N, K]
        entropy       = router_out["entropy"]         # [B, N]
        logits        = router_out["logits"]          # [B, N, K]

        # Hard assignment for metrics and hard-dispatch inference
        assignments = routing_probs.argmax(dim=-1)    # [B, N]

        # --- Step 1b: Causal intervention (optional) ---
        # Overriding the assignment forces tokens through an expert the router did not
        # choose. `routing_probs` below is left as the router's ACTUAL belief (the
        # explanation is what the router said); `assignments` reports what was really
        # dispatched, so a caller can always tell the two apart.
        intervened = self.route_override is not None
        if intervened:
            override = self.route_override.to(assignments.device)
            if override.shape != assignments.shape:
                raise ValueError(
                    f"route_override shape {tuple(override.shape)} does not match "
                    f"assignments {tuple(assignments.shape)}"
                )
            keep = override < 0                       # -1 == leave this token alone
            assignments = torch.where(keep, assignments, override.to(assignments.dtype))

            # `effective_logits` is what the routing decision WOULD have been had the router
            # chosen the override. Needed because a model may consume routing through more
            # than one path: dispatch (which expert runs) AND, in direct_mode, an additive
            # term in the output. Overriding only the dispatch would leave that second path
            # reading the router's original belief, so the intervention would silently do
            # nothing through it. Overridden tokens get a one-hot at the original logit
            # scale, so magnitudes stay comparable.
            scale = logits.abs().amax(dim=-1, keepdim=True)            # [B, N, 1]
            one_hot = torch.zeros_like(logits).scatter_(
                -1, assignments.unsqueeze(-1), 1.0) * scale
            effective_logits = torch.where(keep.unsqueeze(-1), logits, one_hot)

        # --- Step 2: Dispatch to experts ---
        # An intervention forces HARD dispatch: the point is that exactly one expert (the
        # one we chose) processes the token. Soft dispatch would blend all K experts and
        # dilute the manipulation into a reweighting, which is not the causal test.
        if intervened:
            expert_outputs = self._hard_dispatch(bottleneck_tokens, assignments)
        elif self.training or not self.config.hard_routing_inference:
            # Soft dispatch: differentiable, gradients flow to all experts and router
            expert_outputs = self._soft_dispatch(bottleneck_tokens, routing_probs)
        else:
            # Hard dispatch: each token goes to exactly one expert
            expert_outputs = self._hard_dispatch(bottleneck_tokens, assignments)

        return {
            "routing_probs":  routing_probs,   # [B, N, K]  the router's ACTUAL belief
            "expert_outputs": expert_outputs,  # [B, N, D]
            "entropy":        entropy,         # [B, N]
            "assignments":    assignments,     # [B, N]  what was really dispatched
            "logits":         logits,          # [B, N, K]  the router's actual logits
            # Equals `logits` normally; under intervention it reflects the FORCED decision.
            # Consumers that act on the routing (direct_mode) must use this; consumers that
            # REPORT the explanation must use `logits`/`routing_probs`.
            "effective_logits": effective_logits if intervened else logits,
        }

    # ------------------------------------------------------------------
    # Dispatch implementations
    # ------------------------------------------------------------------

    def _soft_dispatch(
        self,
        tokens: torch.Tensor,
        routing_probs: torch.Tensor,
    ) -> torch.Tensor:
        """
        Weighted combination of all expert outputs.

        expert_output(b, t) = Σ_k  P(b, t → k) · Expert_k(tokens[b, t])

        All K experts process every token.  The contribution of Expert_k to
        token t is weighted by the routing probability P(b, t → k).

        Parameters
        ----------
        tokens        : Tensor [B, N, D]
        routing_probs : Tensor [B, N, K]

        Returns
        -------
        Tensor [B, N, D]
        """
        B, N, D = tokens.shape

        # Reshape for expert input: [B*N, D]
        flat_tokens = tokens.reshape(B * N, D)

        # Accumulate weighted expert contributions
        output = torch.zeros_like(flat_tokens)  # [B*N, D]

        for k, expert in enumerate(self.experts):
            # Expert_k processes all tokens: [B*N, D] → [B*N, D]
            expert_k_out = expert(flat_tokens)  # [B*N, D]

            # Weight by P(b, t → k): reshape routing_probs[..., k] to [B*N, 1]
            p_k = routing_probs[..., k].reshape(B * N, 1)  # [B*N, 1]

            output = output + p_k * expert_k_out  # [B*N, D]

        return output.reshape(B, N, D)  # [B, N, D]

    def _hard_dispatch(
        self,
        tokens: torch.Tensor,
        assignments: torch.Tensor,
    ) -> torch.Tensor:
        """
        Route each token to its single highest-probability expert.

        expert_output(b, t) = Expert_{k*(b,t)}(tokens[b, t])
        where k*(b,t) = argmax_k P(b, t → k)

        Not differentiable.  Used for inference only.

        Parameters
        ----------
        tokens      : Tensor [B, N, D]
        assignments : Tensor [B, N]     integer concept indices

        Returns
        -------
        Tensor [B, N, D]
        """
        B, N, D = tokens.shape

        flat_tokens      = tokens.reshape(B * N, D)        # [B*N, D]
        flat_assignments = assignments.reshape(B * N)      # [B*N]
        output           = torch.zeros_like(flat_tokens)   # [B*N, D]

        for k, expert in enumerate(self.experts):
            # Select only the tokens assigned to expert k
            mask = flat_assignments == k   # [B*N] boolean

            if mask.any():
                # Expert_k processes only the tokens assigned to it
                output[mask] = expert(flat_tokens[mask])   # [mask.sum(), D]

        return output.reshape(B, N, D)  # [B, N, D]
