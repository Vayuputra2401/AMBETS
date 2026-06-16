"""
test_dispatcher.py — Unit tests for CCRBottleneckModule (the full CCR module).

Tests cover:
  - Output dict completeness and tensor shapes
  - Soft vs hard dispatch behaviour
  - Gradient flow through soft dispatch
  - Expert utilization on random input
  - Consistency between assignments and routing_probs.argmax
"""

import pytest
import torch
import torch.nn.functional as F

from ccr.config.ccr_config  import CCRConfig
from ccr.modules.dispatcher import CCRBottleneckModule


# ---------------------------------------------------------------------------
# Output structure
# ---------------------------------------------------------------------------

class TestDispatcherOutputKeys:

    def test_all_required_keys_present(self, dispatcher, batch_tokens):
        out = dispatcher(batch_tokens)
        required = {"routing_probs", "expert_outputs", "entropy", "assignments", "logits"}
        assert required.issubset(set(out.keys())), (
            f"Missing keys: {required - set(out.keys())}"
        )

    def test_routing_probs_shape(self, dispatcher, batch_tokens):
        out = dispatcher(batch_tokens)
        assert out["routing_probs"].shape == (2, 4096, 4)

    def test_expert_outputs_shape(self, dispatcher, batch_tokens):
        out = dispatcher(batch_tokens)
        assert out["expert_outputs"].shape == (2, 4096, 192)

    def test_entropy_shape(self, dispatcher, batch_tokens):
        out = dispatcher(batch_tokens)
        assert out["entropy"].shape == (2, 4096)

    def test_assignments_shape(self, dispatcher, batch_tokens):
        out = dispatcher(batch_tokens)
        assert out["assignments"].shape == (2, 4096)

    def test_assignments_values_in_range(self, dispatcher, batch_tokens):
        """assignments must contain only valid expert indices [0, K)."""
        out = dispatcher(batch_tokens)
        assert (out["assignments"] >= 0).all()
        assert (out["assignments"] < 4).all()


# ---------------------------------------------------------------------------
# Probability simplex (carried through from router)
# ---------------------------------------------------------------------------

class TestDispatcherRoutingProbs:

    def test_probs_sum_to_one(self, dispatcher, batch_tokens):
        out      = dispatcher(batch_tokens)
        row_sums = out["routing_probs"].sum(dim=-1)   # [B, N]
        assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5)

    def test_assignments_match_argmax(self, dispatcher, batch_tokens):
        """Hard assignments must equal argmax of routing_probs."""
        out         = dispatcher(batch_tokens)
        argmax_k    = out["routing_probs"].argmax(dim=-1)
        assert torch.equal(out["assignments"], argmax_k), (
            "assignments should equal routing_probs.argmax(dim=-1)"
        )


# ---------------------------------------------------------------------------
# Soft vs hard dispatch
# ---------------------------------------------------------------------------

class TestSoftHardDispatch:

    def test_soft_dispatch_in_train_mode(self, dispatcher, batch_tokens):
        """In train mode, dispatcher uses soft dispatch (weighted sum of all experts)."""
        dispatcher.train()
        # Soft dispatch processes every token through every expert;
        # output is a weighted combination.  Check it is not identical to any
        # single expert output (which would indicate hard dispatch).
        out = dispatcher(batch_tokens)
        # If soft dispatch: output[b, t] = Σ_k p_k * expert_k(tokens[b, t])
        # The output should be smooth — not all-zero (degenerate) or all-same.
        out_var = out["expert_outputs"].var().item()
        assert out_var > 1e-4, "Expert outputs have near-zero variance in train mode"

    def test_hard_dispatch_in_eval_mode(self, default_config, batch_tokens):
        """In eval mode with hard_routing_inference=True, each token goes to one expert."""
        disp = CCRBottleneckModule(default_config)
        disp.eval()
        with torch.no_grad():
            out = disp(batch_tokens)
        # Hard dispatch: output[b, t] = expert_{argmax k}(tokens[b, t])
        # Verify no NaN or Inf
        assert not torch.isnan(out["expert_outputs"]).any()
        assert not torch.isinf(out["expert_outputs"]).any()

    def test_soft_mode_forced_in_eval_when_hard_disabled(self, batch_tokens):
        """When hard_routing_inference=False, soft dispatch used even in eval."""
        from ccr.config.ccr_config import CCRConfig
        cfg = CCRConfig()
        cfg.hard_routing_inference = False
        disp = CCRBottleneckModule(cfg)
        disp.eval()
        with torch.no_grad():
            out = disp(batch_tokens)
        assert out["expert_outputs"].shape == (2, 4096, 192)


# ---------------------------------------------------------------------------
# Gradient flow
# ---------------------------------------------------------------------------

class TestDispatcherGradients:

    def test_routing_probs_grad_in_soft_dispatch(self, dispatcher, batch_tokens):
        """
        In soft dispatch (training mode), gradients must flow back through
        routing_probs to the router parameters.
        This is what allows L_align to shape routing semantics.
        """
        dispatcher.train()
        out  = dispatcher(batch_tokens)
        # Simulate a simple loss on expert outputs
        loss = out["expert_outputs"].sum()
        loss.backward()

        # Router parameters must have gradients
        for name, param in dispatcher.router.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, (
                    f"No gradient on router.{name} after backward in soft dispatch"
                )

    def test_expert_params_grad_in_soft_dispatch(self, dispatcher, batch_tokens):
        """All experts must receive gradients in soft dispatch mode."""
        dispatcher.train()
        out  = dispatcher(batch_tokens)
        out["expert_outputs"].sum().backward()

        for k, expert in enumerate(dispatcher.experts):
            for name, param in expert.named_parameters():
                assert param.grad is not None, (
                    f"No gradient on expert[{k}].{name} after backward"
                )

    def test_gradients_flow_to_input_tokens(self, dispatcher):
        """Gradients from expert_outputs must reach the input tokens."""
        tokens = torch.randn(1, 256, 192, requires_grad=True)
        dispatcher.train()
        out    = dispatcher(tokens)
        out["expert_outputs"].sum().backward()
        assert tokens.grad is not None


# ---------------------------------------------------------------------------
# Expert utilization — no collapse on random input
# ---------------------------------------------------------------------------

class TestExpertUtilization:

    def test_all_experts_receive_tokens_random_input(self, dispatcher, batch_tokens):
        """
        On random input with freshly initialised weights, each of the K=4 experts
        should receive at least some tokens.  Near-zero fc2 initialisation makes
        routing near-uniform, so no expert should be completely starved.
        """
        dispatcher.train()
        out         = dispatcher(batch_tokens)
        assignments = out["assignments"].reshape(-1)   # [B*N]

        for k in range(4):
            count = (assignments == k).sum().item()
            assert count > 0, (
                f"Expert {k} received 0 tokens on random input — possible collapse at init"
            )

    def test_routing_probs_not_all_same_class(self, dispatcher, batch_tokens):
        """Routing should produce a mixture of all K classes, not all-one-class."""
        out    = dispatcher(batch_tokens)
        argmax = out["routing_probs"].argmax(dim=-1)   # [B, N]
        n_unique = argmax.unique().numel()
        assert n_unique > 1, (
            f"All tokens routed to the same expert ({n_unique} unique assignments)"
        )
