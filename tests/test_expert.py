"""
test_expert.py — Unit tests for ClinicalConceptExpert.

Validates shape preservation, residual connection behaviour, gradient flow,
and that distinct experts learn distinct transformations.
"""

import pytest
import torch

from ccr.config.ccr_config import ExpertConfig
from ccr.modules.expert     import ClinicalConceptExpert


# ---------------------------------------------------------------------------
# Fixtures (supplement conftest.py)
# ---------------------------------------------------------------------------

@pytest.fixture
def expert(expert_config):
    """Single Background expert (concept_id=0)."""
    return ClinicalConceptExpert(expert_config, concept_id=0, concept_name="background")


@pytest.fixture
def flat_tokens():
    """Flat token batch: [M=512, D=192] simulating hard-dispatch input."""
    torch.manual_seed(42)
    return torch.randn(512, 192)


# ---------------------------------------------------------------------------
# Output shape
# ---------------------------------------------------------------------------

class TestExpertShape:

    def test_output_shape_matches_input(self, expert, flat_tokens):
        """Expert output must be identical shape to input."""
        out = expert(flat_tokens)
        assert out.shape == flat_tokens.shape, (
            f"Expected {flat_tokens.shape}, got {out.shape}"
        )

    def test_single_token(self, expert):
        """Expert handles single-token input (M=1)."""
        token = torch.randn(1, 192)
        out   = expert(token)
        assert out.shape == (1, 192)

    def test_large_batch(self, expert_config):
        """Expert handles large batch without shape issues."""
        exp   = ClinicalConceptExpert(expert_config, concept_id=1, concept_name="ncr")
        large = torch.randn(8192, 192)
        out   = exp(large)
        assert out.shape == (8192, 192)


# ---------------------------------------------------------------------------
# Residual connection
# ---------------------------------------------------------------------------

class TestResidualConnection:

    def test_output_differs_from_input(self, expert, flat_tokens):
        """
        After training, expert output ≠ input (the correction is non-zero).

        Note: with near-zero fc2 initialisation, a freshly constructed expert
        may output values very close to input.  We test that the correction is
        at least applied (output shape OK) and will diverge after training.
        This test uses a deliberately non-zero correction by constructing an
        expert and applying a large random update to fc2.
        """
        # Force fc2 weights to be non-trivial
        with torch.no_grad():
            expert.fc2.weight.fill_(0.1)
        out = expert(flat_tokens)
        # With non-trivial fc2, output must differ from input
        max_diff = (out - flat_tokens).abs().max().item()
        assert max_diff > 1e-4, (
            f"Expert output too close to input (max diff = {max_diff:.2e}), "
            "residual correction may not be applied"
        )

    def test_identity_at_zero_correction(self, expert_config):
        """
        If fc2 weights are zero, expert output equals input (pure identity).
        Validates that the residual connection is implemented correctly.
        """
        exp = ClinicalConceptExpert(expert_config, concept_id=0, concept_name="background")
        with torch.no_grad():
            exp.fc2.weight.zero_()
            exp.fc2.bias.zero_()

        tokens = torch.randn(64, 192)
        out    = exp(tokens)

        assert torch.allclose(out, tokens, atol=1e-5), (
            "Expert with zero fc2 should act as identity (pure residual)"
        )


# ---------------------------------------------------------------------------
# Concept identity attributes
# ---------------------------------------------------------------------------

class TestConceptIdentity:

    def test_concept_id_stored(self, expert_config):
        for k in range(4):
            exp = ClinicalConceptExpert(expert_config, concept_id=k, concept_name=f"c{k}")
            assert exp.concept_id == k

    def test_concept_name_stored(self, expert_config):
        names = ["background", "necrotic_core", "edema", "enhancing_tumor"]
        for k, name in enumerate(names):
            exp = ClinicalConceptExpert(expert_config, concept_id=k, concept_name=name)
            assert exp.concept_name == name


# ---------------------------------------------------------------------------
# Gradient flow
# ---------------------------------------------------------------------------

class TestExpertGradients:

    def test_fc1_receives_gradient(self, expert, flat_tokens):
        out = expert(flat_tokens)
        out.sum().backward()
        assert expert.fc1.weight.grad is not None

    def test_fc2_receives_gradient(self, expert, flat_tokens):
        out = expert(flat_tokens)
        out.sum().backward()
        assert expert.fc2.weight.grad is not None

    def test_gradient_flows_to_input(self, expert):
        tokens = torch.randn(64, 192, requires_grad=True)
        out    = expert(tokens)
        out.sum().backward()
        assert tokens.grad is not None
        assert not torch.isnan(tokens.grad).any()


# ---------------------------------------------------------------------------
# Different experts produce different outputs
# ---------------------------------------------------------------------------

class TestExpertDiversity:

    def test_two_experts_produce_different_outputs(self, expert_config):
        """
        Two experts with default (random) initialisation should produce
        different outputs on the same input.  This tests that experts are
        independently parameterised.
        """
        exp0 = ClinicalConceptExpert(expert_config, concept_id=0, concept_name="background")
        exp1 = ClinicalConceptExpert(expert_config, concept_id=1, concept_name="ncr")

        # Set fc2 to non-trivial values to ensure outputs differ
        with torch.no_grad():
            exp0.fc2.weight.fill_(0.05)
            exp1.fc2.weight.fill_(-0.05)

        tokens = torch.randn(32, 192)
        out0   = exp0(tokens)
        out1   = exp1(tokens)

        assert not torch.allclose(out0, out1, atol=1e-4), (
            "Two experts with different weights should produce different outputs"
        )
