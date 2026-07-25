"""
test_ccr_segresnet.py — CCR-SegResNet (CNN backbone) tests.

Mirrors test_ccrnet.py: the same CCR module dropped into a MONAI SegResNet must produce the
same output dict (seg_logits + routing_probs + entropy + assignments) and train end-to-end.
Runs on CPU with 64^3 volumes (bottleneck 8^3, 192-d).
"""

from __future__ import annotations

import copy
import torch


def test_segresnet_seg_logits_shape(phase2_config, dummy_image):
    from ccrnet.models.ccr_segresnet import CCRSegResNet
    m = CCRSegResNet(phase2_config)
    m.eval()
    with torch.no_grad():
        out = m(dummy_image)
    B, _, H, W, D = dummy_image.shape
    K = phase2_config.ccr.router.num_concepts
    assert out["seg_logits"].shape == (B, K, H, W, D)


def test_segresnet_routing_probs(phase2_config, dummy_image):
    from ccrnet.models.ccr_segresnet import CCRSegResNet
    m = CCRSegResNet(phase2_config)
    out = m(dummy_image)
    B = dummy_image.shape[0]
    K = phase2_config.ccr.router.num_concepts
    gs = m.get_grid_shape()
    N = gs[0] * gs[1] * gs[2]
    assert out["routing_probs"].shape == (B, N, K)
    sums = out["routing_probs"].sum(dim=-1)
    assert (sums - 1.0).abs().max().item() < 1e-4


def test_segresnet_loss_backward(phase2_config, dummy_image, dummy_label):
    from ccrnet.models.ccr_segresnet import CCRSegResNet
    from ccr.losses.total import CCRTotalLoss
    from ccrnet.utils.token_labels import downsample_labels_to_tokens

    m = CCRSegResNet(phase2_config)
    m.train()
    loss_fn = CCRTotalLoss(phase2_config.ccr)
    loss_fn.set_epoch(15)

    out = m(dummy_image)
    token_labels = downsample_labels_to_tokens(dummy_label, m.get_grid_shape())
    losses = loss_fn(
        pred_logits=out["seg_logits"], labels=dummy_label,
        routing_probs=out["routing_probs"], token_labels=token_labels,
        tau_current=m.ccr.router.temperature,
    )
    losses["total"].backward()
    assert losses["total"].item() > 0


def test_segresnet_factory(phase2_config):
    from ccrnet.models.factory import build_model
    from ccrnet.models.ccr_segresnet import CCRSegResNet
    from ccrnet.models.ccrnet import CCRNet
    assert isinstance(build_model(phase2_config, "segresnet"), CCRSegResNet)
    assert isinstance(build_model(phase2_config, "ccrnet"), CCRNet)


def test_segresnet_no_ccr_bypass(phase2_config, dummy_image):
    from ccrnet.models.ccr_segresnet import CCRSegResNet
    cfg = copy.deepcopy(phase2_config)
    cfg.ccr.enabled = False
    m = CCRSegResNet(cfg)
    assert m.ccr is None
    m.eval()
    with torch.no_grad():
        out = m(dummy_image)
    assert out["routing_probs"] is None
    assert out["seg_logits"].shape[1] == phase2_config.ccr.router.num_concepts
