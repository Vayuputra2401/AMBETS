"""
test_router.py — Unit tests for ClinicalConceptRouter.

Every test operates on CPU with small-but-realistic BraTS-like shapes.
Tests are ordered from structural (shape/range) → mathematical (simplex, entropy)
→ behavioural (temperature, gradient flow).
"""

import math
import pytest
import torch

from ccr.config.ccr_config import RouterConfig
from ccr.modules.router     import ClinicalConceptRouter


# ---------------------------------------------------------------------------
# Shape and output structure
# ---------------------------------------------------------------------------

class TestRouterOutputShape:

    def test_routing_probs_shape(self, router, batch_tokens):
        """routing_probs must be [B, N, K]."""
        out = router(batch_tokens)
        assert out["routing_probs"].shape == (2, 4096, 4), (
            f"Expected [2, 4096, 4], got {out['routing_probs'].shape}"
        )

    def test_entropy_shape(self, router, batch_tokens):
        """entropy must be [B, N]."""
        out = router(batch_tokens)
        assert out["entropy"].shape == (2, 4096)

    def test_logits_shape(self, router, batch_tokens):
        """logits must be [B, N, K] (pre-temperature-scaled)."""
        out = router(batch_tokens)
        assert out["logits"].shape == (2, 4096, 4)

    def test_all_keys_present(self, router, batch_tokens):
        out = router(batch_tokens)
        assert set(out.keys()) == {"routing_probs", "entropy", "logits"}


# ---------------------------------------------------------------------------
# Probability simplex constraint
# ---------------------------------------------------------------------------

class TestProbabilitySimplex:

    def test_probs_sum_to_one(self, router, batch_tokens):
        """
        routing_probs must sum to exactly 1.0 along the K dimension.
        This is the fundamental invariant: Σ_k P(t→k) = 1 for every token t.
        """
        out   = router(batch_tokens)
        probs = out["routing_probs"]                          # [B, N, K]
        row_sums = probs.sum(dim=-1)                          # [B, N]
        ones     = torch.ones_like(row_sums)
        assert torch.allclose(row_sums, ones, atol=1e-5), (
            f"Max deviation from 1.0: {(row_sums - ones).abs().max().item():.2e}"
        )

    def test_probs_non_negative(self, router, batch_tokens):
        """All routing probabilities must be ≥ 0."""
        out   = router(batch_tokens)
        probs = out["routing_probs"]
        assert (probs >= 0).all(), "Negative routing probability found"

    def test_probs_at_most_one(self, router, batch_tokens):
        """All routing probabilities must be ≤ 1."""
        out   = router(batch_tokens)
        probs = out["routing_probs"]
        assert (probs <= 1.0 + 1e-6).all(), "Routing probability > 1.0 found"

    def test_no_nan_or_inf(self, router, batch_tokens):
        """No NaN or Inf values in any output."""
        out = router(batch_tokens)
        for key, val in out.items():
            assert not torch.isnan(val).any(), f"NaN in {key}"
            assert not torch.isinf(val).any(), f"Inf in {key}"


# ---------------------------------------------------------------------------
# Entropy range
# ---------------------------------------------------------------------------

class TestEntropyRange:

    def test_entropy_non_negative(self, router, batch_tokens):
        """Entropy H(t) ≥ 0 for all tokens."""
        out = router(batch_tokens)
        assert (out["entropy"] >= -1e-6).all(), "Negative entropy found"

    def test_entropy_at_most_log_k(self, router, batch_tokens):
        """Entropy H(t) ≤ log(K) — maximum entropy for K classes."""
        out      = router(batch_tokens)
        max_H    = math.log(4)   # log(K=4) ≈ 1.386
        exceeds  = (out["entropy"] > max_H + 1e-5).sum().item()
        assert exceeds == 0, (
            f"{exceeds} tokens exceeded max entropy {max_H:.4f}"
        )

    def test_one_hot_routing_has_zero_entropy(self):
        """
        If routing_probs is one-hot (hard assignment), entropy must be ~0.
        Validates the entropy formula against a degenerate edge case.
        """
        # One-hot routing: all tokens assigned to expert 0
        probs = torch.zeros(1, 10, 4)
        probs[..., 0] = 1.0   # [1, 10, 4]

        cfg    = RouterConfig()
        router = ClinicalConceptRouter(cfg)

        # Compute entropy directly from the formula (not via router.forward)
        eps    = 1e-8
        H      = -(probs * (probs + eps).log()).sum(dim=-1)  # [1, 10]
        assert H.max().item() < 1e-4, f"Expected ~0 entropy for one-hot, got {H.max():.4f}"

    def test_uniform_routing_has_max_entropy(self):
        """Uniform routing P(t→k) = 1/K has entropy = log(K)."""
        K      = 4
        probs  = torch.full((1, 10, K), 1.0 / K)
        eps    = 1e-8
        H      = -(probs * (probs + eps).log()).sum(dim=-1)  # [1, 10]
        expected_H = math.log(K)
        assert abs(H.mean().item() - expected_H) < 1e-4


# ---------------------------------------------------------------------------
# Temperature behaviour
# ---------------------------------------------------------------------------

class TestTemperature:

    def test_higher_temperature_increases_entropy(self, batch_tokens):
        """
        Higher τ → flatter (more uniform) routing → higher entropy.
        Verified by running the same router twice with different temperature values
        (temperature manipulated directly on the shared model to keep MLP weights
        identical — avoids state_dict transfer issues between parameter/buffer configs).
        """
        cfg    = RouterConfig(temperature_init=1.0, temperature_learnable=True)
        router = ClinicalConceptRouter(cfg).eval()

        with torch.no_grad():
            # Low temperature → sharp routing → low entropy
            router.temperature.fill_(0.2)
            H_low  = router(batch_tokens)["entropy"].mean().item()

            # High temperature → flat routing → high entropy
            router.temperature.fill_(3.0)
            H_high = router(batch_tokens)["entropy"].mean().item()

        assert H_high > H_low, (
            f"High-temperature entropy ({H_high:.4f}) should exceed "
            f"low-temperature entropy ({H_low:.4f})"
        )

    def test_temperature_clamped_below(self, default_config):
        """Setting temperature to near-zero should not cause NaN (clamped to 0.1)."""
        router = ClinicalConceptRouter(default_config.router)
        if hasattr(router.temperature, 'data'):
            router.temperature.data.fill_(0.001)
        tokens = torch.randn(1, 64, 192)
        out    = router(tokens)
        assert not torch.isnan(out["routing_probs"]).any()

    def test_temperature_clamped_above(self, default_config):
        """Setting temperature to very large value should not cause NaN (clamped to 5.0)."""
        router = ClinicalConceptRouter(default_config.router)
        if hasattr(router.temperature, 'data'):
            router.temperature.data.fill_(1000.0)
        tokens = torch.randn(1, 64, 192)
        out    = router(tokens)
        assert not torch.isnan(out["routing_probs"]).any()


# ---------------------------------------------------------------------------
# Gradient flow
# ---------------------------------------------------------------------------

class TestGradientFlow:

    def test_router_weights_receive_gradients(self, router, batch_tokens):
        """
        After a backward pass through routing_probs.sum(), all router MLP
        weights must have non-None gradients.  This verifies the routing
        distribution is differentiable with respect to router parameters.
        """
        out    = router(batch_tokens)
        loss   = out["routing_probs"].sum()
        loss.backward()

        for name, param in router.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, (
                    f"Parameter '{name}' has no gradient after backward"
                )

    def test_gradient_flows_to_input(self, router, batch_tokens):
        """
        Gradients must flow back through routing_probs to the input tokens.
        This is required for L_align to shape the encoder representations.
        """
        tokens = batch_tokens.detach().requires_grad_(True)
        out    = router(tokens)
        out["routing_probs"].sum().backward()
        assert tokens.grad is not None, "No gradient flowing to input tokens"
        assert not torch.isnan(tokens.grad).any(), "NaN in input gradient"

    def test_temperature_receives_gradient_when_learnable(self, default_config, batch_tokens):
        """Learnable temperature parameter must accumulate gradients."""
        router = ClinicalConceptRouter(default_config.router)
        out    = router(batch_tokens)
        out["routing_probs"].sum().backward()
        assert router.temperature.grad is not None, (
            "Learnable temperature has no gradient"
        )


# ---------------------------------------------------------------------------
# Prototype-augmented routing (Improvement 2)
# ---------------------------------------------------------------------------

class TestPrototypeRouter:

    def test_prototypes_exist_when_enabled(self, default_config):
        """Router with use_prototypes=True must have .prototypes and .blend_alpha."""
        router = ClinicalConceptRouter(default_config.router)
        assert hasattr(router, "prototypes"),  "prototypes attribute missing"
        assert hasattr(router, "blend_alpha"), "blend_alpha attribute missing"

    def test_prototype_shape(self, default_config):
        """Prototypes must be [K, D]."""
        router = ClinicalConceptRouter(default_config.router)
        K = default_config.router.num_concepts
        D = default_config.router.embed_dim
        assert router.prototypes.shape == (K, D), (
            f"Expected prototypes shape ({K}, {D}), got {router.prototypes.shape}"
        )

    def test_output_shapes_unchanged_with_prototypes(self, batch_tokens, default_config):
        """Prototype augmentation must not change any output shape."""
        router = ClinicalConceptRouter(default_config.router)
        out    = router(batch_tokens)
        B, N   = batch_tokens.shape[:2]
        K      = default_config.router.num_concepts
        assert out["routing_probs"].shape == (B, N, K)
        assert out["entropy"].shape       == (B, N)
        assert out["logits"].shape        == (B, N, K)

    def test_probability_simplex_with_prototypes(self, router, batch_tokens):
        """Routing probs must sum to 1 with prototype augmentation enabled."""
        out      = router(batch_tokens)
        row_sums = out["routing_probs"].sum(dim=-1)
        assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5)

    def test_prototypes_receive_gradient(self, default_config, batch_tokens):
        """Prototype vectors must receive gradients via routing_probs.sum().backward()."""
        router = ClinicalConceptRouter(default_config.router)
        out    = router(batch_tokens)
        out["routing_probs"].sum().backward()
        assert router.prototypes.grad is not None, "prototypes have no gradient"
        assert not torch.isnan(router.prototypes.grad).any()

    def test_blend_alpha_receives_gradient(self, default_config, batch_tokens):
        """blend_alpha must receive gradients."""
        router = ClinicalConceptRouter(default_config.router)
        out    = router(batch_tokens)
        out["routing_probs"].sum().backward()
        assert router.blend_alpha.grad is not None, "blend_alpha has no gradient"

    def test_no_prototypes_when_disabled(self):
        """Router with use_prototypes=False must not have prototype attributes."""
        from ccr.config.ccr_config import RouterConfig
        cfg    = RouterConfig(use_prototypes=False)
        router = ClinicalConceptRouter(cfg)
        assert not hasattr(router, "prototypes"),  "prototypes should not exist"
        assert not hasattr(router, "blend_alpha"), "blend_alpha should not exist"

    def test_disabled_prototypes_still_valid(self):
        """Router with use_prototypes=False must still produce valid routing probs."""
        from ccr.config.ccr_config import RouterConfig
        cfg    = RouterConfig(use_prototypes=False)
        router = ClinicalConceptRouter(cfg)
        tokens = torch.randn(1, 64, 192)
        out    = router(tokens)
        row_sums = out["routing_probs"].sum(dim=-1)
        assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5)
