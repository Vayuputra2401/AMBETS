"""
test_ccrnet.py — CCRNet full model tests.

All run on CPU with 64³ synthetic volumes.
"""

from __future__ import annotations

import torch
import pytest


@pytest.fixture
def model(phase2_config):
    from ccrnet.models.ccrnet import CCRNet
    m = CCRNet(phase2_config)
    m.train()
    return m


# ---------------------------------------------------------------------------
# Shape tests
# ---------------------------------------------------------------------------

def test_seg_logits_shape(model, phase2_config, dummy_image):
    out = model(dummy_image)
    B, _, H, W, D = dummy_image.shape
    K = phase2_config.ccr.router.num_concepts
    assert out["seg_logits"].shape == (B, K, H, W, D), (
        f"seg_logits shape {out['seg_logits'].shape} != ({B},{K},{H},{W},{D})"
    )


def test_routing_probs_shape(model, phase2_config, dummy_image):
    out = model(dummy_image)
    B   = dummy_image.shape[0]
    K   = phase2_config.ccr.router.num_concepts
    h   = phase2_config.swin.img_size[0] // 8
    N   = h ** 3
    assert out["routing_probs"].shape == (B, N, K), (
        f"routing_probs shape {out['routing_probs'].shape} != ({B},{N},{K})"
    )


def test_routing_probs_sum_to_one(model, dummy_image):
    out   = model(dummy_image)
    probs = out["routing_probs"]              # [B, N, K]
    sums  = probs.sum(dim=-1)                 # [B, N]
    assert (sums - 1.0).abs().max().item() < 1e-4, (
        f"routing_probs rows don't sum to 1; max deviation {(sums-1).abs().max():.6f}"
    )


def test_entropy_shape(model, phase2_config, dummy_image):
    out = model(dummy_image)
    B   = dummy_image.shape[0]
    h   = phase2_config.swin.img_size[0] // 8
    N   = h ** 3
    assert out["entropy"].shape == (B, N), f"entropy shape {out['entropy'].shape} != ({B},{N})"


def test_assignments_shape(model, phase2_config, dummy_image):
    out = model(dummy_image)
    B   = dummy_image.shape[0]
    h   = phase2_config.swin.img_size[0] // 8
    N   = h ** 3
    assert out["assignments"].shape == (B, N), (
        f"assignments shape {out['assignments'].shape} != ({B},{N})"
    )


# ---------------------------------------------------------------------------
# Routing behaviour
# ---------------------------------------------------------------------------

def test_eval_hard_dispatch(phase2_config, dummy_image):
    """In eval mode, assignments must be integer class indices in [0, K)."""
    from ccrnet.models.ccrnet import CCRNet
    m = CCRNet(phase2_config)
    m.eval()
    with torch.no_grad():
        out = m(dummy_image)
    K = phase2_config.ccr.router.num_concepts
    assert out["assignments"].dtype in (torch.long, torch.int64, torch.int32), (
        f"assignments dtype {out['assignments'].dtype} not integer"
    )
    assert out["assignments"].min() >= 0, "Negative assignment index"
    assert out["assignments"].max() < K, f"Assignment index >= K={K}"


def test_get_grid_shape(model, dummy_image):
    model(dummy_image)
    gs = model.get_grid_shape()
    assert gs is not None, "get_grid_shape() returned None after forward()"
    assert len(gs) == 3, f"Expected 3-tuple, got {gs}"
    expected = dummy_image.shape[-1] // 8
    assert all(s == expected for s in gs), (
        f"Expected grid_shape ({expected},{expected},{expected}), got {gs}"
    )


# ---------------------------------------------------------------------------
# Gradient tests
# ---------------------------------------------------------------------------

def test_gradient_reaches_encoder(phase2_config, dummy_image):
    """Loss gradient must flow back to Swin patch_embed."""
    from ccrnet.models.ccrnet import CCRNet
    m = CCRNet(phase2_config)
    m.train()
    out  = m(dummy_image)
    out["seg_logits"].sum().backward()
    grad = m.encoder.swin.patch_embed.proj.weight.grad
    assert grad is not None, "No gradient on encoder patch_embed"
    assert not torch.isnan(grad).any(), "NaN in encoder gradient"


def test_gradient_reaches_ccr_router(phase2_config, dummy_image):
    """L_align path: gradient must reach CCR router parameters."""
    from ccrnet.models.ccrnet import CCRNet
    m = CCRNet(phase2_config)
    m.train()
    out  = m(dummy_image)
    out["routing_probs"].sum().backward()
    router_grad = None
    for p in m.ccr.router.parameters():
        if p.grad is not None:
            router_grad = p.grad
            break
    assert router_grad is not None, "No gradient on any CCR router parameter"


# ---------------------------------------------------------------------------
# Correctness
# ---------------------------------------------------------------------------

def test_no_nan_inf_in_outputs(model, dummy_image):
    model.eval()
    with torch.no_grad():
        out = model(dummy_image)
    for k, v in out.items():
        if isinstance(v, torch.Tensor):
            assert not torch.isnan(v).any(), f"NaN in output['{k}']"
            assert not torch.isinf(v).any(), f"Inf in output['{k}']"


def test_ccr_receives_correct_token_shape(phase2_config, dummy_image):
    """CCRBottleneckModule must receive [B, N, 192] — not the raw spatial tensor."""
    from ccrnet.models.ccrnet import CCRNet
    received_shapes = []

    class ShapeCapture(torch.nn.Module):
        def __init__(self, wrapped):
            super().__init__()
            self._wrapped = wrapped

        def forward(self, tokens):
            received_shapes.append(tokens.shape)
            return self._wrapped(tokens)

    m = CCRNet(phase2_config)
    m.ccr = ShapeCapture(m.ccr)
    m.train()
    m(dummy_image)

    assert len(received_shapes) == 1
    shape = received_shapes[0]
    B = dummy_image.shape[0]
    h = phase2_config.swin.img_size[0] // 8
    N = h ** 3
    D = phase2_config.ccr.router.embed_dim
    assert shape == (B, N, D), f"CCR received {shape} != expected ({B},{N},{D})"


def test_full_loss_forward(phase2_config, dummy_image, dummy_label):
    """Full CCRTotalLoss forward + backward must complete without error."""
    from ccrnet.models.ccrnet import CCRNet
    from ccr.losses.total import CCRTotalLoss
    from ccrnet.utils.token_labels import downsample_labels_to_tokens

    m  = CCRNet(phase2_config)
    m.train()
    loss_fn = CCRTotalLoss(phase2_config.ccr)
    loss_fn.set_epoch(1)

    out          = m(dummy_image)
    token_labels = downsample_labels_to_tokens(dummy_label, m.get_grid_shape())
    losses       = loss_fn(
        pred_logits   = out["seg_logits"],
        labels        = dummy_label,
        routing_probs = out["routing_probs"],
        token_labels  = token_labels,
    )
    losses["total"].backward()
    assert losses["total"].item() > 0, "Total loss should be positive"
