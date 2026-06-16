"""
test_encoder.py — SwinEncoder3D unit tests.

All run on CPU with 64³ volumes. No pretrained weights required.
"""

from __future__ import annotations

import torch
import pytest


def test_encoder_returns_five_hidden_states(phase2_config, dummy_image):
    from ccrnet.models.encoder import SwinEncoder3D
    enc = SwinEncoder3D(phase2_config.swin)
    enc.eval()
    with torch.no_grad():
        hs = enc(dummy_image)
    assert isinstance(hs, (list, tuple)), "forward() must return a list of tensors"
    assert len(hs) == 5, f"Expected 5 hidden states, got {len(hs)}"


def test_encoder_stage2_shape(phase2_config, dummy_image):
    """hidden_states_out[2] must be [B, 4*embed_dim, H/8, W/8, D/8]."""
    from ccrnet.models.encoder import SwinEncoder3D
    enc = SwinEncoder3D(phase2_config.swin)
    enc.eval()
    with torch.no_grad():
        hs = enc(dummy_image)
    B, _, H, W, D = dummy_image.shape
    expected_C = 4 * phase2_config.swin.embed_dim          # 192
    expected_spatial = H // 8                               # 8 for 64³
    s2 = hs[2]
    assert s2.shape == (B, expected_C, expected_spatial, expected_spatial, expected_spatial), (
        f"Stage-2 shape {s2.shape} != expected ({B}, {expected_C}, "
        f"{expected_spatial}, {expected_spatial}, {expected_spatial})"
    )


def test_encoder_all_stage_shapes(phase2_config, dummy_image):
    """All 5 hidden states must have the expected decreasing spatial resolutions."""
    from ccrnet.models.encoder import SwinEncoder3D
    enc = SwinEncoder3D(phase2_config.swin)
    enc.eval()
    with torch.no_grad():
        hs = enc(dummy_image)
    B, _, H, _, _ = dummy_image.shape
    f = phase2_config.swin.embed_dim  # 48
    expected_shapes = [
        (B,  f,    H // 2,  H // 2,  H // 2 ),
        (B,  f*2,  H // 4,  H // 4,  H // 4 ),
        (B,  f*4,  H // 8,  H // 8,  H // 8 ),
        (B,  f*8,  H // 16, H // 16, H // 16),
        (B,  f*16, H // 32, H // 32, H // 32),
    ]
    for i, (hs_i, exp) in enumerate(zip(hs, expected_shapes)):
        assert hs_i.shape == exp, f"hs[{i}] shape {hs_i.shape} != expected {exp}"


def test_encoder_no_nan(phase2_config, dummy_image):
    from ccrnet.models.encoder import SwinEncoder3D
    enc = SwinEncoder3D(phase2_config.swin)
    enc.eval()
    with torch.no_grad():
        hs = enc(dummy_image)
    for i, h in enumerate(hs):
        assert not torch.isnan(h).any(), f"NaN in hs[{i}]"
        assert not torch.isinf(h).any(), f"Inf in hs[{i}]"


def test_encoder_gradient_flows(phase2_config, dummy_image):
    """Gradient must reach patch_embed after backprop through encoder."""
    from ccrnet.models.encoder import SwinEncoder3D
    enc = SwinEncoder3D(phase2_config.swin)
    enc.train()
    hs  = enc(dummy_image)
    loss = hs[2].sum()
    loss.backward()
    patch_embed_grad = enc.swin.patch_embed.proj.weight.grad
    assert patch_embed_grad is not None, "No gradient on patch_embed.proj.weight"
    assert not torch.isnan(patch_embed_grad).any(), "NaN in patch_embed gradient"


def test_encoder_stage2_token_count(phase2_config, dummy_image):
    """Stage-2 flatten must give exactly N tokens matching CCR router.embed_dim."""
    from ccrnet.models.encoder import SwinEncoder3D
    enc = SwinEncoder3D(phase2_config.swin)
    enc.eval()
    with torch.no_grad():
        hs = enc(dummy_image)
    stage2 = hs[2]                          # [B, 192, h, w, d]
    B, C, h, w, d = stage2.shape
    tokens = stage2.flatten(2).transpose(1, 2)  # [B, N, C]
    assert tokens.shape[2] == phase2_config.ccr.router.embed_dim, (
        f"Token dim {tokens.shape[2]} != ccr.router.embed_dim "
        f"{phase2_config.ccr.router.embed_dim}"
    )
