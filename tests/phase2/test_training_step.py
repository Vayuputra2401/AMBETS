"""
test_training_step.py — Integration tests for a complete training step.

All run on CPU with synthetic 64³ data. No GPU or real BraTS required.
"""

from __future__ import annotations

import os
import tempfile

import pytest
import torch
import torch.nn as nn


@pytest.fixture
def model_and_optim(phase2_config):
    from ccrnet.models.ccrnet import CCRNet
    m = CCRNet(phase2_config)
    m.train()
    opt = torch.optim.AdamW(m.parameters(), lr=1e-4, weight_decay=1e-5)
    return m, opt


# ---------------------------------------------------------------------------
# Training step
# ---------------------------------------------------------------------------

def test_forward_backward_step(model_and_optim, phase2_config, dummy_image, dummy_label):
    """Full forward + backward + optimizer.step() must run without error."""
    from ccr.losses.total import CCRTotalLoss
    from ccrnet.utils.token_labels import downsample_labels_to_tokens

    model, optimizer = model_and_optim
    loss_fn = CCRTotalLoss(phase2_config.ccr)
    loss_fn.set_epoch(1)

    optimizer.zero_grad()
    out = model(dummy_image)
    token_labels = downsample_labels_to_tokens(dummy_label, model.get_grid_shape())
    losses = loss_fn(
        pred_logits   = out["seg_logits"],
        labels        = dummy_label,
        routing_probs = out["routing_probs"],
        token_labels  = token_labels,
    )
    losses["total"].backward()
    optimizer.step()


def test_all_params_receive_gradients(model_and_optim, phase2_config, dummy_image, dummy_label):
    """Every trainable parameter must have a gradient after backward()."""
    from ccr.losses.total import CCRTotalLoss
    from ccrnet.utils.token_labels import downsample_labels_to_tokens

    model, optimizer = model_and_optim
    loss_fn = CCRTotalLoss(phase2_config.ccr)
    loss_fn.set_epoch(15)   # alignment phase — all loss components active

    optimizer.zero_grad()
    out = model(dummy_image)
    token_labels = downsample_labels_to_tokens(dummy_label, model.get_grid_shape())
    losses = loss_fn(
        pred_logits   = out["seg_logits"],
        labels        = dummy_label,
        routing_probs = out["routing_probs"],
        token_labels  = token_labels,
    )
    losses["total"].backward()

    no_grad = [name for name, p in model.named_parameters()
               if p.requires_grad and p.grad is None]
    assert len(no_grad) == 0, (
        f"{len(no_grad)} parameters have no gradient: {no_grad[:5]}"
    )


def test_optimizer_step_changes_weights(model_and_optim, phase2_config, dummy_image, dummy_label):
    """Model weights must change after optimizer.step()."""
    from ccr.losses.total import CCRTotalLoss
    from ccrnet.utils.token_labels import downsample_labels_to_tokens

    model, optimizer = model_and_optim
    loss_fn = CCRTotalLoss(phase2_config.ccr)
    loss_fn.set_epoch(1)

    # snapshot a parameter that will definitely receive gradient
    tracked_param = next(model.boundary_head.parameters())
    weight_before = tracked_param.data.clone()

    optimizer.zero_grad()
    out = model(dummy_image)
    token_labels = downsample_labels_to_tokens(dummy_label, model.get_grid_shape())
    losses = loss_fn(
        pred_logits   = out["seg_logits"],
        labels        = dummy_label,
        routing_probs = out["routing_probs"],
        token_labels  = token_labels,
    )
    losses["total"].backward()
    optimizer.step()

    weight_after = tracked_param.data
    assert not torch.allclose(weight_before, weight_after), (
        "Weights did not change after optimizer.step() — gradient might be zero"
    )


# ---------------------------------------------------------------------------
# Checkpoint
# ---------------------------------------------------------------------------

def test_save_load_checkpoint(model_and_optim, phase2_config, dummy_image, dummy_label):
    """save_checkpoint → load_checkpoint must restore identical model state."""
    from ccr.losses.total import CCRTotalLoss
    from ccrnet.utils.token_labels import downsample_labels_to_tokens
    from ccrnet.utils.checkpoint import save_checkpoint, load_checkpoint
    from ccrnet.models.ccrnet import CCRNet

    model, optimizer = model_and_optim
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=8)
    loss_fn = CCRTotalLoss(phase2_config.ccr)
    loss_fn.set_epoch(1)

    # one step
    out = model(dummy_image)
    token_labels = downsample_labels_to_tokens(dummy_label, model.get_grid_shape())
    losses = loss_fn(
        pred_logits   = out["seg_logits"],
        labels        = dummy_label,
        routing_probs = out["routing_probs"],
        token_labels  = token_labels,
    )
    losses["total"].backward()
    optimizer.step()
    optimizer.zero_grad()

    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_path = save_checkpoint(
            model, optimizer, scheduler,
            epoch=1, metrics={"dice_wt": 0.5},
            config=phase2_config, ckpt_dir=tmpdir,
        )
        assert os.path.exists(ckpt_path), "Checkpoint file not created"

        # fresh model — load and compare
        m2 = CCRNet(phase2_config)
        start_epoch, metrics = load_checkpoint(ckpt_path, m2)

        assert start_epoch == 2, f"Expected start_epoch=2, got {start_epoch}"

        for (n1, p1), (n2, p2) in zip(model.named_parameters(), m2.named_parameters()):
            assert torch.allclose(p1, p2), f"Parameter mismatch at {n1}"


# ---------------------------------------------------------------------------
# Expert reinitialization
# ---------------------------------------------------------------------------

def test_reinitialize_experts(phase2_config):
    """reinitialize_experts must change fc2 weights of named experts only."""
    from ccrnet.models.ccrnet import CCRNet
    from ccrnet.utils.checkpoint import reinitialize_experts

    model = CCRNet(phase2_config)
    ccr   = model.ccr

    # record weights before re-init
    concept_names = [e.concept_name for e in ccr.experts]
    weights_before = {e.concept_name: e.fc2.weight.data.clone() for e in ccr.experts}

    target = concept_names[0]
    reinitialize_experts(ccr, [target])

    for e in ccr.experts:
        if e.concept_name == target:
            assert not torch.allclose(weights_before[e.concept_name], e.fc2.weight.data), (
                f"Expert '{target}' fc2 weight unchanged after reinit"
            )
        else:
            assert torch.allclose(weights_before[e.concept_name], e.fc2.weight.data), (
                f"Expert '{e.concept_name}' fc2 weight changed but should not have"
            )


# ---------------------------------------------------------------------------
# Token label downsampling
# ---------------------------------------------------------------------------

def test_downsample_labels_shape(dummy_label):
    from ccrnet.utils.token_labels import downsample_labels_to_tokens
    grid_shape = (8, 8, 8)
    token_labels = downsample_labels_to_tokens(dummy_label, grid_shape)
    B = dummy_label.shape[0]
    N = 8 * 8 * 8
    assert token_labels.shape == (B, N), (
        f"token_labels shape {token_labels.shape} != ({B},{N})"
    )
    assert token_labels.dtype == torch.long


def test_downsample_labels_values(dummy_label):
    from ccrnet.utils.token_labels import downsample_labels_to_tokens
    token_labels = downsample_labels_to_tokens(dummy_label, (8, 8, 8))
    assert token_labels.min() >= 0, "Negative token label after downsample"
    assert token_labels.max() <= 3, "Token label > 3 after downsample"
