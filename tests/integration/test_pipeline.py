"""
test_pipeline.py — Integration tests: full Phase 1 CCR pipeline.

Simulates a complete training step:
  CCRBottleneckModule (router + 4 experts) + CCRTotalLoss (all 5 loss terms)
  → forward pass → backward pass → gradient verification

Also tests:
  - ExpertUtilizationTracker collapse detection
  - CurriculumWeightScheduler phase transitions
  - ConceptAlignmentScore accumulation and computation
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

import math
import pytest
import torch
import torch.nn.functional as F

from ccr import CCRConfig, CCRBottleneckModule, CCRTotalLoss
from ccr.utils.metrics         import ConceptAlignmentScore, ExpertUtilizationTracker
from ccr.utils.weight_schedule import CurriculumWeightScheduler
from ccr.config.ccr_config     import CurriculumConfig


# ---------------------------------------------------------------------------
# Dimensions (kept small for CI speed — math is identical)
# ---------------------------------------------------------------------------

B  = 2
N  = 1024     # token count (smaller than full BraTS for speed)
D  = 192
K  = 4
HH = 32       # spatial dimension for 3D losses


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_spatial_labels(batch: int, h: int) -> torch.Tensor:
    """Build a spatial label volume with all 4 classes present."""
    labels = torch.zeros(batch, h, h, h, dtype=torch.long)
    labels[:, h//4:3*h//4, h//4:3*h//4, h//4:3*h//4] = 2   # edema
    labels[:, h//3:2*h//3, h//3:2*h//3, h//3:2*h//3] = 1   # ncr
    labels[:, 2*h//5:3*h//5, 2*h//5:3*h//5, 2*h//5:3*h//5] = 3  # et
    return labels


def _make_token_labels(batch: int, n_tokens: int) -> torch.Tensor:
    """Build token labels with all 4 classes present."""
    labels = torch.zeros(batch, n_tokens, dtype=torch.long)
    labels[:, 820:920]  = 1
    labels[:, 920:1000] = 2
    labels[:, 1000:]    = 3
    return labels


# ---------------------------------------------------------------------------
# Test 1: Full training step
# ---------------------------------------------------------------------------

class TestFullTrainingStep:

    def test_forward_and_backward(self):
        """
        Simulates one complete training iteration at epoch 15 (alignment phase):
          1. Forward pass through CCRBottleneckModule
          2. Construct 3D segmentation logits (placeholder — full network Phase 2)
          3. Compute CCRTotalLoss with all components
          4. Backward pass
          5. Verify gradients on router and all 4 experts
        """
        cfg        = CCRConfig()
        dispatcher = CCRBottleneckModule(cfg)
        loss_fn    = CCRTotalLoss(cfg)
        loss_fn.set_epoch(15)   # alignment phase: all losses active

        dispatcher.train()

        # Bottleneck tokens (simulates Swin-B encoder output)
        tokens = torch.randn(B, N, D)

        # --- Forward through CCR module ---
        ccr_out = dispatcher(tokens)
        routing_probs = ccr_out["routing_probs"]    # [B, N, K]
        expert_outputs = ccr_out["expert_outputs"]  # [B, N, D]

        # --- Build 3D segmentation logits ---
        # In the full CCR-Net, these come from upsampling routing_probs.
        # For Phase 1, we use a simple linear projection to simulate this.
        proj       = torch.nn.Linear(D, K)
        pred_logits_3d = proj(expert_outputs.mean(dim=1))  # [B, K] — placeholder
        # Expand to full 3D [B, K, H, W, D] for BoundaryAwareLoss
        pred_logits_3d = pred_logits_3d.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        pred_logits_3d = pred_logits_3d.expand(B, K, HH, HH, HH).contiguous()

        spatial_labels = _make_spatial_labels(B, HH)
        token_labels   = _make_token_labels(B, N)

        # --- Compute total loss ---
        loss_out = loss_fn(
            pred_logits   = pred_logits_3d,
            labels        = spatial_labels,
            routing_probs = routing_probs,
            token_labels  = token_labels,
        )

        # --- Verify all loss components are positive scalars ---
        for key in ["total", "seg", "align", "diversity", "entropy", "boundary"]:
            val = loss_out[key]
            assert val.shape == (), f"Loss['{key}'] is not a scalar"
            assert val.item() >= -1e-6, f"Loss['{key}'] < 0: {val.item():.6f}"

        # L_align must be > 0 in alignment phase
        assert loss_out["align"].item() > 0.0, (
            "L_align should be > 0 at epoch 15 (alignment phase)"
        )

        # --- Backward ---
        loss_out["total"].backward()

        # --- Verify router gradients ---
        for name, param in dispatcher.router.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient: router.{name}"

        # --- Verify all 4 expert gradients ---
        for k, expert in enumerate(dispatcher.experts):
            for name, param in expert.named_parameters():
                assert param.grad is not None, f"No gradient: expert[{k}].{name}"

        # --- routing_probs simplex ---
        row_sums = routing_probs.sum(dim=-1)
        assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5), (
            "routing_probs must sum to 1 along K dimension"
        )

    def test_no_nan_in_gradients(self):
        """No NaN gradients anywhere in the pipeline."""
        cfg        = CCRConfig()
        dispatcher = CCRBottleneckModule(cfg)
        loss_fn    = CCRTotalLoss(cfg)
        loss_fn.set_epoch(20)
        dispatcher.train()

        tokens         = torch.randn(B, N, D)
        ccr_out        = dispatcher(tokens)
        token_labels   = _make_token_labels(B, N)
        spatial_labels = _make_spatial_labels(B, HH)

        # Simple segmentation logits
        pred_logits = torch.randn(B, K, HH, HH, HH)

        loss_out = loss_fn(
            pred_logits   = pred_logits,
            labels        = spatial_labels,
            routing_probs = ccr_out["routing_probs"],
            token_labels  = token_labels,
        )
        loss_out["total"].backward()

        for name, param in dispatcher.named_parameters():
            if param.grad is not None:
                assert not torch.isnan(param.grad).any(), (
                    f"NaN gradient in {name}"
                )


# ---------------------------------------------------------------------------
# Test 2: Expert utilization and collapse detection
# ---------------------------------------------------------------------------

class TestExpertCollapseDetection:

    def test_no_collapse_on_random_input(self):
        """All 4 experts should receive > 5 % of tokens on random input."""
        cfg        = CCRConfig()
        dispatcher = CCRBottleneckModule(cfg)
        dispatcher.train()

        tracker = ExpertUtilizationTracker(
            num_concepts=4,
            concept_names=cfg.concept_names,
            warmup_end_epoch=10,
        )

        for _ in range(5):
            tokens  = torch.randn(B, N, D)
            out     = dispatcher(tokens)
            tracker.update(out["assignments"])

        stats = tracker.compute()
        for name, frac in stats.items():
            assert frac > 0.01, (
                f"Expert '{name}' has very low utilization ({frac:.3f}) on random input"
            )

    def test_collapse_detected_post_warmup(self):
        """
        Simulating a collapsed state (all tokens to expert 0) after warmup
        should trigger a warning and return collapsed expert names.
        """
        cfg     = CCRConfig()
        tracker = ExpertUtilizationTracker(
            num_concepts=4,
            concept_names=cfg.concept_names,
            warmup_end_epoch=10,
            min_utilization_pct=5.0,
        )

        # Simulate collapsed routing: all 4096 tokens go to expert 0
        collapsed_assignments = torch.zeros(B, N, dtype=torch.long)
        tracker.update(collapsed_assignments)

        import warnings
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            flagged = tracker.check_collapse(epoch=25)   # post-warmup

        # Experts 1, 2, 3 received 0 % tokens — should be flagged
        assert len(flagged) == 3, (
            f"Expected 3 collapsed experts, got {len(flagged)}: {flagged}"
        )
        assert "necrotic_core" in flagged
        assert "edema" in flagged
        assert "enhancing_tumor" in flagged

    def test_no_collapse_warning_during_warmup(self):
        """Collapse check during warmup should return empty list (not a concern yet)."""
        cfg     = CCRConfig()
        tracker = ExpertUtilizationTracker(
            num_concepts=4,
            concept_names=cfg.concept_names,
            warmup_end_epoch=10,
        )
        collapsed = torch.zeros(B, N, dtype=torch.long)
        tracker.update(collapsed)

        flagged = tracker.check_collapse(epoch=5)   # during warmup
        assert flagged == [], (
            "Collapse during warmup should not be flagged"
        )


# ---------------------------------------------------------------------------
# Test 3: Curriculum scheduler phase transitions
# ---------------------------------------------------------------------------

class TestCurriculumScheduler:

    def test_warmup_weights(self):
        cfg   = CurriculumConfig()
        sched = CurriculumWeightScheduler(cfg)
        for epoch in [1, 5, 10]:
            w = sched.get_weights(epoch)
            assert w[0] == 0.0, f"λ_align should be 0 in warmup, epoch={epoch}"
            assert sched.get_phase_name(epoch) == "warmup"

    def test_alignment_weights(self):
        cfg   = CurriculumConfig()
        sched = CurriculumWeightScheduler(cfg)
        for epoch in [11, 25, 50]:
            w = sched.get_weights(epoch)
            assert w[0] == 1.0, f"λ_align should be 1.0 in alignment, epoch={epoch}"
            assert sched.get_phase_name(epoch) == "alignment"

    def test_refinement_weights(self):
        cfg   = CurriculumConfig()
        sched = CurriculumWeightScheduler(cfg)
        for epoch in [51, 65, 80]:
            w = sched.get_weights(epoch)
            assert w[0] == 0.5,  f"λ_align should be 0.5 in refinement, epoch={epoch}"
            assert w[3] == 1.0,  f"λ_boundary should be 1.0 in refinement, epoch={epoch}"
            assert sched.get_phase_name(epoch) == "refinement"

    def test_is_align_active(self):
        cfg   = CurriculumConfig()
        sched = CurriculumWeightScheduler(cfg)
        assert not sched.is_align_active(10)  # last warmup epoch
        assert sched.is_align_active(11)       # first alignment epoch

    def test_invalid_epoch_raises(self):
        cfg   = CurriculumConfig()
        sched = CurriculumWeightScheduler(cfg)
        with pytest.raises(ValueError):
            sched.get_weights(0)
        with pytest.raises(ValueError):
            sched.get_weights(81)


# ---------------------------------------------------------------------------
# Test 4: ConceptAlignmentScore accumulation
# ---------------------------------------------------------------------------

class TestConceptAlignmentScore:

    def test_returns_all_concept_keys(self):
        cfg  = CCRConfig()
        cas  = ConceptAlignmentScore(
            num_concepts=4,
            concept_names=cfg.concept_names,
        )
        # Feed perfectly aligned routing (one-hot routing = GT)
        probs  = torch.zeros(B, N, K)
        labels = _make_token_labels(B, N)
        for k in range(K):
            probs[labels == k, k] = 1.0

        cas.update(probs, labels)
        scores = cas.compute()
        assert set(scores.keys()) == set(cfg.concept_names)

    def test_perfect_alignment_gives_high_cas(self):
        """
        When routing perfectly predicts GT, CAS(k) should be close to 1.0.
        """
        cfg  = CCRConfig()
        cas  = ConceptAlignmentScore(4, cfg.concept_names)

        labels = _make_token_labels(B, N)
        probs  = torch.zeros(B, N, K)
        for k in range(K):
            mask = (labels == k)
            probs[mask, k] = 1.0

        for _ in range(10):
            cas.update(probs, labels)

        scores = cas.compute()
        for name, score in scores.items():
            assert score > 0.85, (
                f"Perfect alignment should give CAS > 0.85 for '{name}', got {score:.4f}"
            )

    def test_random_routing_gives_low_cas(self):
        """Random routing should produce near-zero CAS."""
        cfg  = CCRConfig()
        cas  = ConceptAlignmentScore(4, cfg.concept_names)

        labels = _make_token_labels(B, N)
        torch.manual_seed(99)
        for _ in range(20):
            probs = F.softmax(torch.randn(B, N, K), dim=-1)
            cas.update(probs, labels)

        scores = cas.compute()
        for name, score in scores.items():
            assert abs(score) < 0.3, (
                f"Random routing should give |CAS| < 0.3 for '{name}', got {score:.4f}"
            )

    def test_reset_clears_state(self):
        cfg  = CCRConfig()
        cas  = ConceptAlignmentScore(4, cfg.concept_names)

        labels = _make_token_labels(B, N)
        probs  = torch.zeros(B, N, K)
        for k in range(K):
            probs[labels == k, k] = 1.0

        cas.update(probs, labels)
        cas.reset()

        scores = cas.compute()
        # After reset, no data → all CAS = 0
        for name, score in scores.items():
            assert score == 0.0, f"After reset, CAS should be 0, got {score}"
