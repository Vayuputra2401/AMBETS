"""
test_evaluation.py — Tests for src/ccrnet/utils/evaluation.py

All tests run on CPU with small synthetic inputs. No MONAI / real BraTS needed.
A lightweight MockCCRNet is used for deletion/insertion AUC tests to avoid
instantiating the full SwinTransformer.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "data"))

from ccrnet.utils.evaluation import (
    upsample_routing_to_volume,
    compute_auroc_per_concept,
    compute_deletion_auc,
    compute_insertion_auc,
    compute_ece,
    mc_dropout_entropy,
)


# ---------------------------------------------------------------------------
# Lightweight mock model (avoids full Swin stack)
# ---------------------------------------------------------------------------

class MockCCRNet(nn.Module):
    """
    Minimal CCRNet standin for evaluation function tests.
    Uses 32³ image, 4³ token grid (N=64 tokens).
    Outputs are deterministic given a fixed seed to keep tests repeatable.
    """

    def __init__(self, K: int = 4, H: int = 32, grid: int = 4):
        super().__init__()
        self.K = K
        self.H = H
        self.grid = grid
        self._dummy = nn.Linear(1, 1)  # so model has parameters

    def forward(self, x: torch.Tensor) -> dict:
        B = x.shape[0]
        N = self.grid ** 3
        K = self.K
        H = self.H

        # Deterministic but input-dependent outputs (so masking has an effect)
        seed_val = x.mean().item()
        g = torch.Generator()
        g.manual_seed(int(abs(seed_val) * 1e6) % (2**31))

        raw = torch.randn(B, N, K, generator=g)
        routing_probs = torch.softmax(raw, dim=-1)
        assignments = routing_probs.argmax(dim=-1)
        entropy = -(routing_probs * (routing_probs + 1e-8).log()).sum(dim=-1)
        seg_logits = torch.randn(B, K, H, H, H, generator=g)

        return {
            "seg_logits":    seg_logits,
            "routing_probs": routing_probs,
            "entropy":       entropy,
            "assignments":   assignments,
        }

    def get_grid_shape(self):
        return (self.grid, self.grid, self.grid)


# ---------------------------------------------------------------------------
# Constants for small tests
# ---------------------------------------------------------------------------

K   = 4
N   = 4 * 4 * 4    # 64 tokens
H   = 32
GRID = (4, 4, 4)
VOL  = (H, H, H)
CONCEPT_NAMES = ("background", "necrotic_core", "edema", "enhancing_tumor")


@pytest.fixture
def mock_model():
    m = MockCCRNet(K=K, H=H, grid=4)
    m.eval()
    return m


@pytest.fixture
def synthetic_image():
    torch.manual_seed(0)
    return torch.randn(1, 4, H, H, H)


@pytest.fixture
def synthetic_label():
    label = torch.zeros(1, H, H, H, dtype=torch.long)
    label[:, 8:24, 8:24, 8:24] = 2   # edema
    label[:, 12:20, 12:20, 12:20] = 1  # NCR
    label[:, 14:18, 14:18, 14:18] = 3  # ET
    return label


@pytest.fixture
def synthetic_routing_probs():
    torch.manual_seed(42)
    raw = torch.randn(1, N, K)
    return torch.softmax(raw, dim=-1)


@pytest.fixture
def synthetic_token_labels():
    labels = torch.zeros(1, N, dtype=torch.long)
    labels[0, :15] = 2  # edema
    labels[0, :5]  = 1  # NCR (first 5 tokens)
    labels[0, :2]  = 3  # ET (first 2 tokens)
    return labels


# ---------------------------------------------------------------------------
# 1. upsample_routing_to_volume
# ---------------------------------------------------------------------------

def test_upsample_routing_shape():
    probs = torch.softmax(torch.randn(1, N, K), dim=-1)
    out = upsample_routing_to_volume(probs, GRID, VOL)
    assert out.shape == (1, K, H, H, H)


def test_upsample_routing_sums_to_one():
    """After upsampling, per-voxel probability sums should still be ~1."""
    probs = torch.softmax(torch.randn(2, N, K), dim=-1)
    out = upsample_routing_to_volume(probs, GRID, VOL)
    # Sum over K at each voxel: should be close to 1 (trilinear doesn't perfectly preserve sum)
    voxel_sums = out.sum(dim=1)  # [B, H, W, D]
    assert (voxel_sums - 1.0).abs().max().item() < 0.1


def test_upsample_routing_wrong_N_raises():
    probs = torch.randn(1, N + 1, K)
    with pytest.raises(ValueError):
        upsample_routing_to_volume(probs, GRID, VOL)


# ---------------------------------------------------------------------------
# 2. compute_auroc_per_concept
# ---------------------------------------------------------------------------

def test_auroc_returns_correct_keys(synthetic_routing_probs, synthetic_token_labels):
    result = compute_auroc_per_concept(
        synthetic_routing_probs, synthetic_token_labels, CONCEPT_NAMES
    )
    # Background skipped; foreground concepts present
    assert "necrotic_core" in result
    assert "edema" in result
    assert "enhancing_tumor" in result
    assert "background" not in result


def test_auroc_perfect_routing():
    """One-hot routing perfectly aligned with GT labels → AUROC = 1.0."""
    labels = torch.zeros(1, N, dtype=torch.long)
    labels[0, :10] = 1   # NCR
    labels[0, 10:30] = 2  # Edema
    labels[0, 30:40] = 3  # ET

    # One-hot routing matching labels exactly
    probs = torch.zeros(1, N, K)
    for i in range(N):
        probs[0, i, labels[0, i].item()] = 1.0

    result = compute_auroc_per_concept(probs, labels, CONCEPT_NAMES)
    for name in ("necrotic_core", "edema", "enhancing_tumor"):
        assert result[name] == pytest.approx(1.0, abs=1e-6), f"{name} AUROC != 1.0: {result[name]}"


def test_auroc_no_positive_tokens_returns_nan():
    """A concept with no GT tokens should return nan."""
    labels = torch.zeros(1, N, dtype=torch.long)  # all background
    probs = torch.softmax(torch.randn(1, N, K), dim=-1)
    result = compute_auroc_per_concept(probs, labels, CONCEPT_NAMES)
    for name in ("necrotic_core", "edema", "enhancing_tumor"):
        assert math.isnan(result[name])


# ---------------------------------------------------------------------------
# 3. compute_deletion_auc
# ---------------------------------------------------------------------------

def test_deletion_auc_returns_correct_keys(mock_model, synthetic_image, synthetic_label,
                                            synthetic_routing_probs):
    result = compute_deletion_auc(
        mock_model, synthetic_image, synthetic_label,
        synthetic_routing_probs, GRID, torch.device("cpu"),
        CONCEPT_NAMES,
        concept_indices=(1, 2, 3),
        thresholds=(0.10, 0.30, 0.50),
    )
    assert isinstance(result, dict)
    assert "necrotic_core" in result
    assert "edema" in result
    assert "enhancing_tumor" in result


def test_deletion_auc_returns_floats(mock_model, synthetic_image, synthetic_label,
                                      synthetic_routing_probs):
    result = compute_deletion_auc(
        mock_model, synthetic_image, synthetic_label,
        synthetic_routing_probs, GRID, torch.device("cpu"),
        CONCEPT_NAMES,
        thresholds=(0.10, 0.30),
    )
    for v in result.values():
        assert isinstance(v, float)
        assert not math.isnan(v)


# ---------------------------------------------------------------------------
# 4. compute_insertion_auc
# ---------------------------------------------------------------------------

def test_insertion_auc_returns_correct_keys(mock_model, synthetic_image,
                                             synthetic_routing_probs):
    result = compute_insertion_auc(
        mock_model, synthetic_image,
        synthetic_routing_probs, GRID, torch.device("cpu"),
        CONCEPT_NAMES,
        concept_indices=(1, 2, 3),
        thresholds=(0.10, 0.30, 0.50),
    )
    assert isinstance(result, dict)
    assert set(result.keys()) == {"necrotic_core", "edema", "enhancing_tumor"}


def test_insertion_auc_returns_floats(mock_model, synthetic_image, synthetic_routing_probs):
    result = compute_insertion_auc(
        mock_model, synthetic_image,
        synthetic_routing_probs, GRID, torch.device("cpu"),
        CONCEPT_NAMES,
        thresholds=(0.10, 0.30),
    )
    for v in result.values():
        assert isinstance(v, float)
        assert not math.isnan(v)


# ---------------------------------------------------------------------------
# 5. compute_ece
# ---------------------------------------------------------------------------

def test_ece_perfect_routing_near_zero():
    """
    One-hot routing exactly matching labels → confidence=1.0, accuracy=1.0 everywhere
    → ECE ≈ 0 (all tokens in the highest confidence bin are correct).
    """
    labels = torch.zeros(1, N, dtype=torch.long)
    labels[0, :10] = 1
    labels[0, 10:30] = 2
    labels[0, 30:40] = 3

    probs = torch.zeros(1, N, K)
    for i in range(N):
        probs[0, i, labels[0, i].item()] = 1.0

    # Entropy = 0 (perfectly certain)
    entropy = torch.zeros(1, N)

    ece = compute_ece(probs, labels, entropy)
    assert isinstance(ece, float)
    assert ece < 0.05, f"ECE should be near-zero for perfect routing, got {ece}"


def test_ece_uniform_routing():
    """Uniform routing → confidence=0 everywhere, accuracy~0.25 → positive ECE."""
    labels = torch.zeros(1, N, dtype=torch.long)
    labels[0, :N//4] = 1
    labels[0, N//4:N//2] = 2
    labels[0, N//2:3*N//4] = 3

    probs = torch.full((1, N, K), 1.0 / K)
    # Entropy = log(K) = maximum
    entropy = torch.full((1, N), math.log(K))

    ece = compute_ece(probs, labels, entropy)
    assert isinstance(ece, float)
    # Confidence = 0, accuracy = 0.25 → |0 - 0.25| = 0.25
    assert ece > 0.0


def test_ece_returns_float_with_random_inputs(synthetic_routing_probs, synthetic_token_labels):
    entropy = -(synthetic_routing_probs * (synthetic_routing_probs + 1e-8).log()).sum(dim=-1)
    ece = compute_ece(synthetic_routing_probs, synthetic_token_labels, entropy)
    assert isinstance(ece, float)
    assert 0.0 <= ece <= 1.0


# ---------------------------------------------------------------------------
# 6. mc_dropout_entropy
# ---------------------------------------------------------------------------

def test_mc_dropout_entropy_shape(mock_model, synthetic_image):
    result = mc_dropout_entropy(mock_model, synthetic_image, n_passes=3)
    assert result.shape == (N,), f"Expected [{N}], got {result.shape}"


def test_mc_dropout_entropy_non_negative(mock_model, synthetic_image):
    result = mc_dropout_entropy(mock_model, synthetic_image, n_passes=3)
    assert (result >= 0).all(), "Entropy must be non-negative"


def test_mc_dropout_entropy_bounded(mock_model, synthetic_image):
    result = mc_dropout_entropy(mock_model, synthetic_image, n_passes=3)
    max_entropy = math.log(K)
    assert (result <= max_entropy + 1e-5).all(), f"Entropy exceeds log(K)={max_entropy:.4f}"


# ---------------------------------------------------------------------------
# 7. Ablation config parsing
# ---------------------------------------------------------------------------

def test_no_align_yaml_parses_correctly():
    """A1 ablation: lam_align=0 in all phases must parse correctly."""
    from ccrnet.config.phase2_config import Phase2Config
    cfg_path = str(ROOT / "configs" / "ablations" / "no_align.yaml")
    cfg = Phase2Config.from_yaml(cfg_path)
    w = cfg.ccr.loss.weights
    assert w.warmup[0] == 0.0,     f"warmup lam_align should be 0.0, got {w.warmup[0]}"
    assert w.alignment[0] == 0.0,  f"alignment lam_align should be 0.0, got {w.alignment[0]}"
    assert w.refinement[0] == 0.0, f"refinement lam_align should be 0.0, got {w.refinement[0]}"


def test_no_warmup_yaml_parses_correctly():
    """A4 ablation: warmup_end_epoch=0 must parse correctly."""
    from ccrnet.config.phase2_config import Phase2Config
    cfg_path = str(ROOT / "configs" / "ablations" / "no_warmup.yaml")
    cfg = Phase2Config.from_yaml(cfg_path)
    assert cfg.ccr.curriculum.warmup_end_epoch == 0
    assert cfg.ccr.loss.weights.warmup[0] == 1.0, "Warmup lam_align should be 1.0 in no_warmup ablation"
