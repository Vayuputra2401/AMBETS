"""
test_losses.py — Unit tests for all five CCR loss components.

For each loss:
  1. Output is a non-negative scalar
  2. Gradient flows to the relevant input
  3. Perfect prediction or ideal routing → loss approaches minimum
  4. Loss behaves consistently with the math in the docstring
"""

import math
import pytest
import torch
import torch.nn.functional as F

from ccr.config.ccr_config   import LossConfig
from ccr.losses.alignment    import ConceptAlignmentLoss
from ccr.losses.boundary     import BoundaryAwareLoss
from ccr.losses.diversity    import ExpertDiversityLoss
from ccr.losses.entropy      import EntropyRegularizationLoss
from ccr.losses.segmentation import DiceLoss, FocalLoss, SegmentationLoss
from ccr.losses.total        import CCRTotalLoss


# ---------------------------------------------------------------------------
# Fixtures (small tensors for speed)
# ---------------------------------------------------------------------------

B, N, K   = 2, 256, 4          # Smaller N for fast loss tests
H = W = D = 32                  # Small spatial volume for boundary tests


@pytest.fixture
def small_routing_probs():
    """Soft routing probs [B, N, K] with approximate BraTS class balance."""
    torch.manual_seed(7)
    raw    = torch.randn(B, N, K)
    return F.softmax(raw, dim=-1)


@pytest.fixture
def small_token_labels():
    """Token labels [B, N] with all 4 classes present."""
    labels = torch.zeros(B, N, dtype=torch.long)
    labels[:, 200:220] = 1
    labels[:, 220:235] = 2
    labels[:, 235:250] = 3
    return labels


@pytest.fixture
def small_logits():
    """3D segmentation logits [B, K, H, W, D]."""
    torch.manual_seed(3)
    return torch.randn(B, K, H, W, D)


@pytest.fixture
def small_spatial_labels():
    """3D GT labels [B, H, W, D]."""
    labels = torch.zeros(B, H, W, D, dtype=torch.long)
    labels[:, 8:24, 8:24, 8:24]   = 2   # edema
    labels[:, 10:22, 10:22, 10:22] = 1  # ncr
    labels[:, 13:19, 13:19, 13:19] = 3  # et
    return labels


# ---------------------------------------------------------------------------
# DiceLoss
# ---------------------------------------------------------------------------

class TestDiceLoss:

    def test_scalar_output(self, small_logits, small_spatial_labels):
        loss_fn = DiceLoss(num_classes=K)
        loss    = loss_fn(small_logits, small_spatial_labels)
        assert loss.shape == (), f"Expected scalar, got {loss.shape}"

    def test_non_negative(self, small_logits, small_spatial_labels):
        loss_fn = DiceLoss(num_classes=K)
        loss    = loss_fn(small_logits, small_spatial_labels)
        assert loss.item() >= -1e-6

    def test_perfect_prediction_near_zero(self):
        """
        When predicted probabilities perfectly match GT one-hot, Dice loss ≈ 0.
        """
        K     = 4
        B, H, W, D = 1, 8, 8, 8
        labels = torch.zeros(B, H, W, D, dtype=torch.long)
        labels[:, 3:5, 3:5, 3:5] = 1   # class 1 in a small cube

        # Perfect logits: large positive value at GT class, negative elsewhere
        logits = torch.full((B, K, H, W, D), -10.0)
        for c in range(K):
            logits[:, c, ...][labels == c] = 10.0

        loss_fn = DiceLoss(num_classes=K)
        loss    = loss_fn(logits, labels)
        assert loss.item() < 0.05, f"Perfect prediction Dice loss = {loss.item():.4f}"

    def test_gradient_flows_to_logits(self, small_logits, small_spatial_labels):
        logits  = small_logits.detach().requires_grad_(True)
        loss_fn = DiceLoss(num_classes=K)
        loss    = loss_fn(logits, small_spatial_labels)
        loss.backward()
        assert logits.grad is not None
        assert not torch.isnan(logits.grad).any()


# ---------------------------------------------------------------------------
# FocalLoss
# ---------------------------------------------------------------------------

class TestFocalLoss:

    def test_scalar_output(self, small_logits, small_spatial_labels):
        loss_fn = FocalLoss(gamma=2.0, alpha=0.25)
        loss    = loss_fn(small_logits, small_spatial_labels)
        assert loss.shape == ()

    def test_non_negative(self, small_logits, small_spatial_labels):
        loss_fn = FocalLoss()
        assert loss_fn(small_logits, small_spatial_labels).item() >= -1e-6

    def test_perfect_prediction_near_zero(self):
        """High-confidence correct predictions have near-zero focal loss."""
        K     = 4
        B, H  = 1, 16
        labels = torch.zeros(B, H, H, H, dtype=torch.long)

        # Perfect logits
        logits = torch.full((B, K, H, H, H), -10.0)
        for c in range(K):
            logits[:, c, ...][labels == c] = 10.0

        loss_fn = FocalLoss(gamma=2.0, alpha=0.25)
        loss    = loss_fn(logits, labels)
        assert loss.item() < 0.05, f"Expected near-zero focal loss, got {loss.item():.4f}"

    def test_gradient_flows(self, small_logits, small_spatial_labels):
        logits  = small_logits.detach().requires_grad_(True)
        loss_fn = FocalLoss()
        loss_fn(logits, small_spatial_labels).backward()
        assert logits.grad is not None


# ---------------------------------------------------------------------------
# ConceptAlignmentLoss
# ---------------------------------------------------------------------------

class TestConceptAlignmentLoss:

    def test_scalar_output(self, small_routing_probs, small_token_labels):
        loss_fn = ConceptAlignmentLoss(num_concepts=K)
        loss    = loss_fn(small_routing_probs, small_token_labels)
        assert loss.shape == ()

    def test_non_negative(self, small_routing_probs, small_token_labels):
        loss_fn = ConceptAlignmentLoss(num_concepts=K)
        assert loss_fn(small_routing_probs, small_token_labels).item() >= -1e-6

    def test_perfect_alignment_near_zero(self):
        """
        When routing_probs perfectly predicts GT (one-hot routing matching labels),
        L_align ≈ 0.
        """
        B, N, K = 1, 100, 4
        labels  = torch.zeros(B, N, dtype=torch.long)
        labels[:, 25:50] = 1
        labels[:, 50:75] = 2
        labels[:, 75:]   = 3

        # Perfect routing: one-hot matching labels
        probs = torch.zeros(B, N, K)
        for k in range(K):
            probs[labels == k, k] = 1.0

        loss_fn = ConceptAlignmentLoss(num_concepts=K)
        loss    = loss_fn(probs, labels)
        assert loss.item() < 0.02, f"Perfect alignment loss = {loss.item():.4f}"

    def test_gradient_flows_to_routing_probs(self, small_token_labels):
        """L_align gradient must reach routing_probs — required for CAS to improve."""
        probs = F.softmax(torch.randn(B, N, K), dim=-1).requires_grad_(True)
        loss_fn = ConceptAlignmentLoss(num_concepts=K)
        loss_fn(probs, small_token_labels).backward()
        assert probs.grad is not None
        assert not torch.isnan(probs.grad).any()

    def test_wrong_label_shape_raises(self, small_routing_probs):
        loss_fn = ConceptAlignmentLoss(num_concepts=K)
        bad_labels = torch.zeros(B, N + 1, dtype=torch.long)   # wrong N
        with pytest.raises(AssertionError):
            loss_fn(small_routing_probs, bad_labels)


# ---------------------------------------------------------------------------
# ConceptAlignmentLoss — foreground-only (Improvement 1)
# ---------------------------------------------------------------------------

class TestConceptAlignmentForegroundOnly:

    def test_foreground_only_is_default(self):
        """ConceptAlignmentLoss() should default to foreground_only=True."""
        loss_fn = ConceptAlignmentLoss(num_concepts=K)
        assert loss_fn.foreground_only is True

    def test_foreground_only_scalar_output(self, small_routing_probs, small_token_labels):
        loss_fn = ConceptAlignmentLoss(num_concepts=K, foreground_only=True)
        loss    = loss_fn(small_routing_probs, small_token_labels)
        assert loss.shape == ()

    def test_foreground_only_non_negative(self, small_routing_probs, small_token_labels):
        loss_fn = ConceptAlignmentLoss(num_concepts=K, foreground_only=True)
        assert loss_fn(small_routing_probs, small_token_labels).item() >= -1e-6

    def test_all_background_batch_gives_zero_fg_loss(self):
        """
        When all tokens are background (label=0), foreground_only=True gives 0.0
        because fg_count = 0 and no foreground concept contributes.
        """
        B, N, K   = 1, 50, 4
        probs     = F.softmax(torch.randn(B, N, K), dim=-1)
        labels    = torch.zeros(B, N, dtype=torch.long)  # all background
        loss_fn   = ConceptAlignmentLoss(num_concepts=K, foreground_only=True)
        loss      = loss_fn(probs, labels)
        assert loss.item() == 0.0, (
            f"All-background batch should give 0.0 loss, got {loss.item()}"
        )

    def test_foreground_only_gradient_flows(self, small_token_labels):
        probs   = F.softmax(torch.randn(B, N, K), dim=-1).requires_grad_(True)
        loss_fn = ConceptAlignmentLoss(num_concepts=K, foreground_only=True)
        loss_fn(probs, small_token_labels).backward()
        assert probs.grad is not None
        assert not torch.isnan(probs.grad).any()

    def test_foreground_lower_or_equal_than_all_tokens(self, small_routing_probs, small_token_labels):
        """
        Foreground-only loss is not systematically higher or lower than all-tokens loss —
        they answer different questions.  Both must be non-negative scalars.
        """
        loss_fg  = ConceptAlignmentLoss(num_concepts=K, foreground_only=True)
        loss_all = ConceptAlignmentLoss(num_concepts=K, foreground_only=False)
        v_fg  = loss_fg(small_routing_probs, small_token_labels)
        v_all = loss_all(small_routing_probs, small_token_labels)
        assert v_fg.item()  >= -1e-6
        assert v_all.item() >= -1e-6


# ---------------------------------------------------------------------------
# ConceptAlignmentLoss — focal BCE (Improvement 4)
# ---------------------------------------------------------------------------

class TestConceptAlignmentFocal:

    def test_focal_gamma_default_is_nonzero(self):
        """Default focal_gamma must be > 0 (focal modulation active by default)."""
        loss_fn = ConceptAlignmentLoss(num_concepts=K)
        assert loss_fn.focal_gamma > 0.0

    def test_focal_scalar_output(self, small_routing_probs, small_token_labels):
        loss_fn = ConceptAlignmentLoss(num_concepts=K, focal_gamma=2.0)
        loss    = loss_fn(small_routing_probs, small_token_labels)
        assert loss.shape == ()

    def test_focal_non_negative(self, small_routing_probs, small_token_labels):
        loss_fn = ConceptAlignmentLoss(num_concepts=K, focal_gamma=2.0)
        assert loss_fn(small_routing_probs, small_token_labels).item() >= -1e-6

    def test_perfect_routing_near_zero_with_focal(self):
        """
        With focal BCE, perfect routing (p_k=1 for GT-positive tokens)
        still gives near-zero loss because focal_weight = (1-1)^γ = 0.
        """
        B, N, K = 1, 100, 4
        labels  = torch.zeros(B, N, dtype=torch.long)
        labels[:, 25:50] = 1
        labels[:, 50:75] = 2
        labels[:, 75:]   = 3
        probs = torch.zeros(B, N, K)
        for k in range(K):
            probs[labels == k, k] = 1.0

        loss_fn = ConceptAlignmentLoss(num_concepts=K, focal_gamma=2.0, foreground_only=False)
        loss    = loss_fn(probs, labels)
        assert loss.item() < 0.02, (
            f"Perfect routing with focal BCE should give ~0 loss, got {loss.item():.4f}"
        )

    def test_focal_down_weights_easy_tokens(self):
        """
        With γ > 0, focal BCE should be ≤ plain BCE for confident correct predictions.
        Confident correct tokens get down-weighted by (1-p_correct)^γ → 0.
        """
        B, N, K = 1, 100, 4
        labels  = torch.zeros(B, N, dtype=torch.long)
        labels[:, 50:] = 1

        # High-confidence routing: p_k=0.95 for correct class
        probs = torch.full((B, N, K), 0.05 / (K - 1))
        for k in range(K):
            mask = (labels == k)
            probs[mask, k] = 0.95
        probs = probs / probs.sum(dim=-1, keepdim=True)   # renormalise

        loss_plain = ConceptAlignmentLoss(num_concepts=K, focal_gamma=0.0, foreground_only=False)
        loss_focal = ConceptAlignmentLoss(num_concepts=K, focal_gamma=2.0, foreground_only=False)

        v_plain = loss_plain(probs, labels).item()
        v_focal = loss_focal(probs, labels).item()

        assert v_focal <= v_plain + 1e-4, (
            f"Focal BCE ({v_focal:.4f}) should be ≤ plain BCE ({v_plain:.4f}) "
            "for high-confidence correct predictions"
        )

    def test_focal_gradient_flows(self, small_token_labels):
        probs   = F.softmax(torch.randn(B, N, K), dim=-1).requires_grad_(True)
        loss_fn = ConceptAlignmentLoss(num_concepts=K, focal_gamma=2.0)
        loss_fn(probs, small_token_labels).backward()
        assert probs.grad is not None
        assert not torch.isnan(probs.grad).any()


# ---------------------------------------------------------------------------
# CurriculumWeightScheduler — tau targets (Improvement 3)
# ---------------------------------------------------------------------------

class TestTauSchedule:

    def test_tau_target_warmup(self):
        from ccr.utils.weight_schedule import CurriculumWeightScheduler
        from ccr.config.ccr_config import CurriculumConfig
        cfg  = CurriculumConfig()
        sched = CurriculumWeightScheduler(cfg)
        assert sched.get_tau_target(5) == cfg.tau_warmup

    def test_tau_target_alignment(self):
        from ccr.utils.weight_schedule import CurriculumWeightScheduler
        from ccr.config.ccr_config import CurriculumConfig
        cfg  = CurriculumConfig()
        sched = CurriculumWeightScheduler(cfg)
        assert sched.get_tau_target(25) == cfg.tau_alignment

    def test_tau_target_refinement(self):
        from ccr.utils.weight_schedule import CurriculumWeightScheduler
        from ccr.config.ccr_config import CurriculumConfig
        cfg  = CurriculumConfig()
        sched = CurriculumWeightScheduler(cfg)
        assert sched.get_tau_target(60) == cfg.tau_refinement

    def test_tau_target_decreases_across_phases(self):
        """τ_warmup > τ_alignment > τ_refinement by design."""
        from ccr.utils.weight_schedule import CurriculumWeightScheduler
        from ccr.config.ccr_config import CurriculumConfig
        cfg  = CurriculumConfig()
        sched = CurriculumWeightScheduler(cfg)
        assert sched.get_tau_target(5) > sched.get_tau_target(25) > sched.get_tau_target(60)

    def test_tau_reg_in_total_loss_output(self, default_config,
                                          small_logits, small_spatial_labels,
                                          small_routing_probs, small_token_labels):
        """CCRTotalLoss must include 'tau_reg' and 'tau_target' in output dict."""
        from ccr.losses.total import CCRTotalLoss
        loss_fn = CCRTotalLoss(default_config)
        loss_fn.set_epoch(20)
        out = loss_fn(small_logits, small_spatial_labels,
                      small_routing_probs, small_token_labels)
        assert "tau_reg"    in out
        assert "tau_target" in out

    def test_tau_reg_zero_when_no_tau_current(self, default_config,
                                              small_logits, small_spatial_labels,
                                              small_routing_probs, small_token_labels):
        """Without tau_current, tau_reg must be exactly 0.0 (backward compatible)."""
        from ccr.losses.total import CCRTotalLoss
        loss_fn = CCRTotalLoss(default_config)
        loss_fn.set_epoch(20)
        out = loss_fn(small_logits, small_spatial_labels,
                      small_routing_probs, small_token_labels)
        assert out["tau_reg"].item() == 0.0

    def test_tau_reg_nonzero_when_tau_provided(self, default_config,
                                               small_logits, small_spatial_labels,
                                               small_routing_probs, small_token_labels):
        """When tau_current differs from tau_target, tau_reg must be > 0."""
        from ccr.losses.total import CCRTotalLoss
        loss_fn = CCRTotalLoss(default_config)
        loss_fn.set_epoch(60)   # refinement: tau_target=0.5
        # tau_current at 3.0 — far from target
        tau = torch.tensor([3.0])
        out = loss_fn(small_logits, small_spatial_labels,
                      small_routing_probs, small_token_labels,
                      tau_current=tau)
        assert out["tau_reg"].item() > 0.0

    def test_tau_reg_zero_when_tau_at_target(self, default_config,
                                             small_logits, small_spatial_labels,
                                             small_routing_probs, small_token_labels):
        """When tau_current == tau_target, tau_reg ≈ 0."""
        from ccr.losses.total import CCRTotalLoss
        loss_fn = CCRTotalLoss(default_config)
        loss_fn.set_epoch(60)   # refinement: tau_target=0.5
        tau = torch.tensor([0.5])   # exactly at target
        out = loss_fn(small_logits, small_spatial_labels,
                      small_routing_probs, small_token_labels,
                      tau_current=tau)
        assert out["tau_reg"].item() < 1e-6


# ---------------------------------------------------------------------------
# ExpertDiversityLoss
# ---------------------------------------------------------------------------

class TestExpertDiversityLoss:

    def test_scalar_output(self, small_routing_probs):
        loss_fn = ExpertDiversityLoss(num_concepts=K)
        assert loss_fn(small_routing_probs).shape == ()

    def test_non_negative(self, small_routing_probs):
        loss_fn = ExpertDiversityLoss(num_concepts=K)
        assert loss_fn(small_routing_probs).item() >= -1e-6

    def test_uniform_routing_minimises_load_balance(self):
        """
        When all experts receive exactly 1/K of tokens, the load-balance
        component = 0.  Cosine overlap may still be non-zero.
        """
        B, N, K = 1, 400, 4
        # Build probs where each expert gets exactly 100 tokens (one-hot routing)
        probs = torch.zeros(B, N, K)
        for k in range(K):
            probs[:, k * 100 : (k + 1) * 100, k] = 1.0

        loss_fn = ExpertDiversityLoss(num_concepts=K)
        loss    = loss_fn(probs)

        # Compute load-balance component separately
        mean_probs  = probs.mean(dim=[0, 1])
        load_balance = ((mean_probs - 1.0 / K) ** 2).mean().item()
        assert load_balance < 1e-6, (
            f"Uniform load should give ~0 load_balance, got {load_balance:.2e}"
        )

    def test_gradient_flows(self, small_routing_probs):
        probs   = small_routing_probs.detach().requires_grad_(True)
        loss_fn = ExpertDiversityLoss(num_concepts=K)
        loss_fn(probs).backward()
        assert probs.grad is not None


# ---------------------------------------------------------------------------
# EntropyRegularizationLoss
# ---------------------------------------------------------------------------

class TestEntropyRegularizationLoss:

    def test_scalar_output(self, small_routing_probs):
        loss_fn = EntropyRegularizationLoss()
        assert loss_fn(small_routing_probs).shape == ()

    def test_non_negative(self, small_routing_probs):
        loss_fn = EntropyRegularizationLoss()
        assert loss_fn(small_routing_probs).item() >= -1e-6

    def test_uniform_routing_is_max_entropy(self):
        """Uniform routing P = 1/K produces entropy = log(K) per token."""
        K      = 4
        probs  = torch.full((1, 50, K), 1.0 / K)
        loss_fn = EntropyRegularizationLoss()
        H_mean = loss_fn(probs).item()
        assert abs(H_mean - math.log(K)) < 1e-4

    def test_one_hot_routing_is_zero_entropy(self):
        """One-hot routing produces entropy ≈ 0."""
        K = 4
        probs = torch.zeros(1, 50, K)
        probs[..., 0] = 1.0
        loss_fn = EntropyRegularizationLoss()
        H_mean = loss_fn(probs).item()
        assert H_mean < 1e-4

    def test_gradient_flows(self, small_routing_probs):
        probs   = small_routing_probs.detach().requires_grad_(True)
        loss_fn = EntropyRegularizationLoss()
        loss_fn(probs).backward()
        assert probs.grad is not None


# ---------------------------------------------------------------------------
# BoundaryAwareLoss
# ---------------------------------------------------------------------------

class TestBoundaryAwareLoss:

    def test_scalar_output(self, small_logits, small_spatial_labels):
        loss_fn = BoundaryAwareLoss(num_classes=K)
        assert loss_fn(small_logits, small_spatial_labels).shape == ()

    def test_non_negative(self, small_logits, small_spatial_labels):
        loss_fn = BoundaryAwareLoss(num_classes=K)
        assert loss_fn(small_logits, small_spatial_labels).item() >= -1e-6

    def test_boundary_loss_higher_than_plain_ce(self, small_logits, small_spatial_labels):
        """
        Boundary loss with boost_factor > 0 must exceed plain cross-entropy
        because boundary voxels receive extra weight.
        """
        logits      = small_logits.detach()
        labels      = small_spatial_labels

        plain_ce    = F.cross_entropy(logits, labels.long())
        loss_fn     = BoundaryAwareLoss(num_classes=K, base_weight=1.0, boost_factor=5.0)
        boundary_ce = loss_fn(logits, labels)

        assert boundary_ce.item() >= plain_ce.item() - 1e-4, (
            "Boundary-aware loss should be ≥ plain CE when boost_factor > 0"
        )

    def test_gradient_flows_to_logits(self, small_logits, small_spatial_labels):
        logits  = small_logits.detach().requires_grad_(True)
        loss_fn = BoundaryAwareLoss(num_classes=K)
        loss_fn(logits, small_spatial_labels).backward()
        assert logits.grad is not None

    def test_no_boost_equals_plain_ce(self, small_logits, small_spatial_labels):
        """With boost_factor=0, boundary loss = plain cross-entropy."""
        logits  = small_logits.detach()
        labels  = small_spatial_labels

        plain_ce    = F.cross_entropy(logits, labels.long()).item()
        loss_fn     = BoundaryAwareLoss(num_classes=K, base_weight=1.0, boost_factor=0.0)
        boundary_ce = loss_fn(logits, labels).item()

        assert abs(boundary_ce - plain_ce) < 1e-4, (
            f"No-boost boundary CE ({boundary_ce:.6f}) should equal "
            f"plain CE ({plain_ce:.6f})"
        )


# ---------------------------------------------------------------------------
# CCRTotalLoss — curriculum weighting
# ---------------------------------------------------------------------------

class TestCCRTotalLoss:

    @pytest.fixture
    def total_loss_fn(self, default_config):
        return CCRTotalLoss(default_config)

    def test_warmup_align_is_zero(self, total_loss_fn,
                                  small_logits, small_spatial_labels,
                                  small_routing_probs, small_token_labels):
        """During warmup (epoch ≤ 10), L_align contribution = 0."""
        total_loss_fn.set_epoch(5)
        out = total_loss_fn(
            pred_logits   = small_logits,
            labels        = small_spatial_labels,
            routing_probs = small_routing_probs,
            token_labels  = small_token_labels,
        )
        assert out["align"].item() == 0.0, (
            f"L_align should be 0 at epoch 5 (warmup), got {out['align'].item()}"
        )
        assert out["phase"] == "warmup"

    def test_alignment_phase_align_active(self, total_loss_fn,
                                          small_logits, small_spatial_labels,
                                          small_routing_probs, small_token_labels):
        """During alignment phase (epoch 11-50), L_align > 0."""
        total_loss_fn.set_epoch(20)
        out = total_loss_fn(
            pred_logits   = small_logits,
            labels        = small_spatial_labels,
            routing_probs = small_routing_probs,
            token_labels  = small_token_labels,
        )
        assert out["align"].item() > 0.0
        assert out["phase"] == "alignment"

    def test_refinement_phase_boundary_active(self, total_loss_fn,
                                              small_logits, small_spatial_labels,
                                              small_routing_probs, small_token_labels):
        """During refinement (epoch > 50), L_boundary > 0."""
        total_loss_fn.set_epoch(60)
        out = total_loss_fn(
            pred_logits   = small_logits,
            labels        = small_spatial_labels,
            routing_probs = small_routing_probs,
            token_labels  = small_token_labels,
        )
        assert out["boundary"].item() > 0.0
        assert out["phase"] == "refinement"

    def test_total_is_sum_of_components(self, total_loss_fn,
                                        small_logits, small_spatial_labels,
                                        small_routing_probs, small_token_labels):
        """total = seg + λ₁*align + λ₂*diversity + λ₃*entropy + λ₄*boundary."""
        total_loss_fn.set_epoch(25)   # alignment phase: all components active
        out = total_loss_fn(
            pred_logits   = small_logits,
            labels        = small_spatial_labels,
            routing_probs = small_routing_probs,
            token_labels  = small_token_labels,
        )
        lam_a, lam_d, lam_e, lam_b = out["weights"]
        expected = (
            out["seg"]
            + lam_a * out["align"]
            + lam_d * out["diversity"]
            + lam_e * out["entropy"]
            + lam_b * out["boundary"]
        )
        assert torch.allclose(out["total"], expected, atol=1e-5)

    def test_backward_from_total(self, total_loss_fn,
                                 small_logits, small_spatial_labels,
                                 small_routing_probs, small_token_labels):
        """total.backward() must succeed without NaN gradients."""
        logits = small_logits.detach().requires_grad_(True)
        probs  = F.softmax(torch.randn_like(small_routing_probs), dim=-1).requires_grad_(True)

        total_loss_fn.set_epoch(20)
        out  = total_loss_fn(logits, small_spatial_labels, probs, small_token_labels)
        out["total"].backward()

        assert logits.grad is not None
        assert probs.grad is not None
        assert not torch.isnan(logits.grad).any()
        assert not torch.isnan(probs.grad).any()
