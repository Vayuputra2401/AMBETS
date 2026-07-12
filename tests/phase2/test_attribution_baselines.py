"""
test_attribution_baselines.py — Tests for the new W3 baselines (IG + Occlusion).

A tiny differentiable model (Conv3d) stands in for CCRNet so gradients actually flow
to the input (the MockCCRNet in test_evaluation.py emits noise unrelated to x, which
would break Integrated Gradients). Grad-CAM is not tested here (it needs a real decoder).
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "data"))
sys.path.insert(0, str(ROOT / "pipeline"))

from attribution_baselines import (  # noqa: E402
    integrated_gradients,
    occlusion_saliency,
    explanation_maps,
)

K = 4
H = 16


class DiffMock(nn.Module):
    """Differentiable stand-in: seg_logits depend on the input (so IG has a path)."""

    def __init__(self, in_ch: int = 4, k: int = K):
        super().__init__()
        self.conv = nn.Conv3d(in_ch, k, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> dict:
        return {
            "seg_logits":    self.conv(x),
            "routing_probs": None,
            "entropy":       None,
            "assignments":   None,
        }


def _model_and_image():
    torch.manual_seed(0)
    m = DiffMock().eval()
    img = torch.randn(1, 4, H, H, H)
    return m, img


def test_integrated_gradients_shape_and_nonneg():
    m, img = _model_and_image()
    sal = integrated_gradients(m, img, k=1, steps=4)
    assert sal.shape == (H, H, H)
    assert torch.isfinite(sal).all()
    assert (sal >= 0).all(), "|IG| summed over channels must be non-negative"


def test_integrated_gradients_zero_baseline_zero_input():
    """With a zero baseline and a zero image, IG = (x - x') * grad = 0 everywhere."""
    m, _ = _model_and_image()
    img = torch.zeros(1, 4, H, H, H)
    sal = integrated_gradients(m, img, k=2, steps=4)
    assert torch.allclose(sal, torch.zeros_like(sal), atol=1e-6)


def test_occlusion_saliency_shape_and_nonneg():
    m, img = _model_and_image()
    with torch.no_grad():
        baseline_probs = torch.softmax(m(img)["seg_logits"], dim=1)
    sal = occlusion_saliency(m, img, k=1, baseline_probs=baseline_probs, grid=2)
    assert sal.shape == (H, H, H)
    assert torch.isfinite(sal).all()
    assert (sal >= 0).all()


def test_occlusion_empty_pred_region_returns_zeros():
    m, img = _model_and_image()
    # Force a baseline where concept-3 is never the argmax → empty predicted region.
    baseline_probs = torch.zeros(1, K, H, H, H)
    baseline_probs[0, 0] = 1.0   # everything predicted background
    sal = occlusion_saliency(m, img, k=3, baseline_probs=baseline_probs, grid=2)
    assert torch.count_nonzero(sal) == 0


def test_explanation_maps_dispatch_ig_and_occlusion():
    m, img = _model_and_image()
    with torch.no_grad():
        baseline_probs = torch.softmax(m(img)["seg_logits"], dim=1)
    for method in ("ig", "occlusion"):
        maps = explanation_maps(
            method, m, None, img, (1, 2, 3), (H, H, H),
            baseline_probs=baseline_probs, occ_grid=2, ig_steps=3,
        )
        assert set(maps.keys()) == {1, 2, 3}
        for v in maps.values():
            assert v.shape == (H, H, H)


def test_explanation_maps_unknown_method_raises():
    m, img = _model_and_image()
    try:
        explanation_maps("bogus", m, None, img, (1,), (H, H, H))
        assert False, "expected ValueError"
    except ValueError:
        pass
