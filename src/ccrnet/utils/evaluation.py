"""
evaluation.py — CCR-Net evaluation utilities for WACV 2027 submission.

Faithfulness metrics:
  compute_auroc_per_concept : AUROC of routing_probs[k] as classifier for GT concept k
  compute_deletion_auc      : progressive masking by explanation score (high = faithful)
  compute_insertion_auc     : progressive revelation from zero baseline (high = faithful)

Uncertainty calibration:
  compute_ece               : Expected Calibration Error of routing confidence vs accuracy
  mc_dropout_entropy        : MC-Dropout entropy baseline (requires model with dropout > 0)

Utilities:
  upsample_routing_to_volume : trilinear upsample token routing map to full image resolution
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Utility: upsample token-level routing to full image resolution
# ---------------------------------------------------------------------------

def upsample_routing_to_volume(
    routing_probs: torch.Tensor,        # [B, N, K]
    grid_shape: Tuple[int, int, int],   # (h, w, d) e.g. (16, 16, 16)
    target_shape: Tuple[int, int, int], # (H, W, D) e.g. (128, 128, 128)
) -> torch.Tensor:                       # [B, K, H, W, D]
    """
    Trilinear upsample token routing probabilities to full image resolution.
    Used to build spatial attribution maps for deletion/insertion AUC.
    """
    B, N, K = routing_probs.shape
    h, w, d = grid_shape
    if N != h * w * d:
        raise ValueError(f"N={N} does not match grid_shape product {h*w*d}")

    # [B, N, K] → [B, K, h, w, d]
    spatial = routing_probs.permute(0, 2, 1).reshape(B, K, h, w, d).float()
    # [B, K, h, w, d] → [B, K, H, W, D]
    return F.interpolate(spatial, size=target_shape, mode="trilinear", align_corners=False)


# ---------------------------------------------------------------------------
# AUROC per concept
# ---------------------------------------------------------------------------

def compute_auroc_per_concept(
    routing_probs: torch.Tensor,        # [B, N, K] accumulated over dataset
    token_labels: torch.Tensor,         # [B, N] int64
    concept_names: Tuple[str, ...],
    skip_background: bool = True,       # k=0 is background — skip by default
) -> Dict[str, float]:
    """
    One-vs-rest AUROC of routing_probs[:,:,k] as a binary classifier for GT concept k.

    A high AUROC (>0.90) means the router assigns high probability to concept k
    specifically where the GT label is k — direct measure of routing specificity.
    Background (k=0) excluded: dominated by class-imbalance signal.

    Returns {concept_name: auroc} for k=1..K-1.
    Concepts with no positive GT tokens get auroc=nan.
    """
    try:
        from sklearn.metrics import roc_auc_score
    except ImportError:
        raise ImportError("scikit-learn required: pip install scikit-learn")

    B, N, K = routing_probs.shape
    probs_np = routing_probs.cpu().float().numpy().reshape(-1, K)   # [B*N, K]
    labels_np = token_labels.cpu().long().numpy().reshape(-1)       # [B*N]

    results: Dict[str, float] = {}
    start_k = 1 if skip_background else 0
    for k in range(start_k, K):
        gt_k = (labels_np == k).astype(int)
        if gt_k.sum() == 0 or (1 - gt_k).sum() == 0:
            results[concept_names[k]] = float("nan")
            continue
        try:
            results[concept_names[k]] = float(roc_auc_score(gt_k, probs_np[:, k]))
        except Exception:
            results[concept_names[k]] = float("nan")

    return results


# ---------------------------------------------------------------------------
# Deletion / Insertion AUC — explanation-map agnostic core
# ---------------------------------------------------------------------------
#
# The deletion/insertion methodology is identical regardless of WHERE the
# explanation comes from (CCR routing, GradCAM, gradient saliency, ...). The
# only method-specific input is the per-concept explanation volume `expl_k`.
# These two functions are the single source of truth so that CCR and the
# post-hoc baselines are compared under exactly the same protocol.

_DEFAULT_THRESHOLDS: Tuple[float, ...] = (0.05, 0.10, 0.20, 0.30, 0.40, 0.50)


def deletion_auc_from_map(
    model,
    image: torch.Tensor,                # [1, C, H, W, D]
    label: torch.Tensor,                # [1, H, W, D] int64
    expl_k: torch.Tensor,               # [H, W, D] explanation scores for concept k
    k: int,
    thresholds: Tuple[float, ...] = _DEFAULT_THRESHOLDS,
    baseline_fill: float = 0.0,
    baseline_probs: Optional[torch.Tensor] = None,  # [1, K, H, W, D] softmax of unmasked image
) -> float:
    """
    Deletion AUC for a single concept k from an arbitrary explanation map.

    Mask the top-τ fraction of voxels (ranked by expl_k) and measure the drop in
    concept-k confidence. Higher AUC = a more faithful explanation (important
    voxels concentrated in the top rank → fast, large confidence drop when removed).
    """
    _, C, H, W, D = image.shape
    total_voxels = H * W * D

    model.eval()
    with torch.no_grad():
        if baseline_probs is None:
            baseline_probs = torch.softmax(model(image)["seg_logits"], dim=1)

        sorted_idx = expl_k.detach().reshape(-1).argsort(descending=True)

        gt_mask_k = (label[0] == k)
        if gt_mask_k.sum() > 0:
            baseline_conf = baseline_probs[0, k][gt_mask_k].mean().item()
        else:
            baseline_conf = baseline_probs[0, k].mean().item()

        drops: List[float] = []
        for tau in thresholds:
            n_mask = max(1, int(tau * total_voxels))
            mask_idx = sorted_idx[:n_mask]

            masked = image.clone()
            for ch in range(C):
                flat = masked[0, ch].flatten().clone()
                flat[mask_idx] = baseline_fill
                masked[0, ch] = flat.reshape(H, W, D)

            probs = torch.softmax(model(masked)["seg_logits"], dim=1)
            if gt_mask_k.sum() > 0:
                conf = probs[0, k][gt_mask_k].mean().item()
            else:
                conf = probs[0, k].mean().item()
            drops.append(float(baseline_conf - conf))

    return float(np.trapezoid(drops, list(thresholds)))


def insertion_auc_from_map(
    model,
    image: torch.Tensor,                # [1, C, H, W, D]
    expl_k: torch.Tensor,               # [H, W, D] explanation scores for concept k
    k: int,
    thresholds: Tuple[float, ...] = _DEFAULT_THRESHOLDS,
) -> float:
    """
    Insertion AUC for a single concept k from an arbitrary explanation map.

    Reveal the top-τ voxels (ranked by expl_k) from a zero baseline and measure
    the rise in concept-k confidence. Higher AUC = the explanation identifies the
    minimal sufficient voxels.
    """
    _, C, H, W, D = image.shape
    total_voxels = H * W * D

    model.eval()
    with torch.no_grad():
        sorted_idx = expl_k.detach().reshape(-1).argsort(descending=True)

        confidences: List[float] = []
        for tau in thresholds:
            n_insert = max(1, int(tau * total_voxels))
            insert_idx = sorted_idx[:n_insert]

            partial = torch.zeros_like(image)
            for ch in range(C):
                partial_flat = partial[0, ch].flatten()
                orig_flat = image[0, ch].flatten()
                partial_flat[insert_idx] = orig_flat[insert_idx]
                partial[0, ch] = partial_flat.reshape(H, W, D)

            probs = torch.softmax(model(partial)["seg_logits"], dim=1)
            confidences.append(probs[0, k].mean().item())

    return float(np.trapezoid(confidences, list(thresholds)))


# ---------------------------------------------------------------------------
# Deletion AUC (CCR routing)
# ---------------------------------------------------------------------------

def compute_deletion_auc(
    model,
    image: torch.Tensor,                # [1, C, H, W, D]
    label: torch.Tensor,                # [1, H, W, D] int64
    routing_probs: torch.Tensor,        # [1, N, K]
    grid_shape: Tuple[int, int, int],
    device: torch.device,
    concept_names: Tuple[str, ...],
    concept_indices: Tuple[int, ...] = (1, 2, 3),
    thresholds: Tuple[float, ...] = _DEFAULT_THRESHOLDS,
    baseline_fill: float = 0.0,
) -> Dict[str, float]:
    """
    Deletion AUC for CCR routing: builds the per-concept explanation volume from
    routing_probs (upsampled to image resolution) and calls the shared core.

    Returns {concept_name: deletion_auc_float} for each index in concept_indices.
    """
    _, C, H, W, D = image.shape
    target_shape = (H, W, D)
    routing_volume = upsample_routing_to_volume(routing_probs, grid_shape, target_shape)

    model.eval()
    with torch.no_grad():
        baseline_probs = torch.softmax(model(image)["seg_logits"], dim=1)

    results: Dict[str, float] = {}
    for k in concept_indices:
        name = concept_names[k] if k < len(concept_names) else f"concept_{k}"
        results[name] = deletion_auc_from_map(
            model, image, label, routing_volume[0, k], k,
            thresholds=thresholds, baseline_fill=baseline_fill,
            baseline_probs=baseline_probs,
        )
    return results


# ---------------------------------------------------------------------------
# Insertion AUC
# ---------------------------------------------------------------------------

def compute_insertion_auc(
    model,
    image: torch.Tensor,                # [1, C, H, W, D]
    routing_probs: torch.Tensor,        # [1, N, K]
    grid_shape: Tuple[int, int, int],
    device: torch.device,
    concept_names: Tuple[str, ...],
    concept_indices: Tuple[int, ...] = (1, 2, 3),
    thresholds: Tuple[float, ...] = _DEFAULT_THRESHOLDS,
) -> Dict[str, float]:
    """
    Insertion AUC for CCR routing: builds the per-concept explanation volume from
    routing_probs (upsampled to image resolution) and calls the shared core.

    Returns {concept_name: insertion_auc_float} for each index in concept_indices.
    """
    _, C, H, W, D = image.shape
    target_shape = (H, W, D)
    routing_volume = upsample_routing_to_volume(routing_probs, grid_shape, target_shape)

    model.eval()
    results: Dict[str, float] = {}
    for k in concept_indices:
        name = concept_names[k] if k < len(concept_names) else f"concept_{k}"
        results[name] = insertion_auc_from_map(
            model, image, routing_volume[0, k], k, thresholds=thresholds,
        )
    return results


# ---------------------------------------------------------------------------
# Expected Calibration Error
# ---------------------------------------------------------------------------

def compute_ece(
    routing_probs: torch.Tensor,  # [B, N, K] or [N, K]
    token_labels: torch.Tensor,   # [B, N] or [N] int64
    entropy: torch.Tensor,        # [B, N] or [N] — H(t) = -Σ P log P in nats ∈ [0, log K]
    n_bins: int = 15,
) -> float:
    """
    Expected Calibration Error of routing confidence vs token-level accuracy.

    Confidence(t) = 1 - H(t) / log(K)      (1 = certain, 0 = uniform)
    Accuracy(t)   = 1[argmax routing_probs == token_label]

    ECE = Σ_b (|b| / N) × |mean_conf(b) − mean_acc(b)|

    A well-calibrated router assigns high confidence where its hard routing decision
    is correct, and low confidence at uncertain boundary regions. Proposition 2 claims
    routing entropy ECE < MC-Dropout ECE at T=10 (zero additional overhead).
    """
    K = routing_probs.shape[-1]
    max_entropy = math.log(K)  # log(4) ≈ 1.386 nats for K=4

    entropy_flat = entropy.detach().flatten().float()
    labels_flat = token_labels.detach().flatten().long()
    probs_flat = routing_probs.detach().reshape(-1, K).float()

    # Confidence ∈ [0, 1]: 1 = perfectly certain routing
    normalized_entropy = (entropy_flat / max_entropy).clamp(0.0, 1.0)
    confidence = 1.0 - normalized_entropy  # [B*N]

    # Binary accuracy of hard routing decision
    assignments = probs_flat.argmax(dim=-1)  # [B*N]
    accuracy = (assignments == labels_flat).float()  # [B*N]

    # Equal-width bins on confidence
    n = confidence.numel()
    ece = 0.0
    bin_edges = torch.linspace(0.0, 1.0, n_bins + 1, device=confidence.device)

    for i in range(n_bins):
        lo = bin_edges[i].item()
        hi = bin_edges[i + 1].item()
        # Include upper edge only in the last bin
        if i < n_bins - 1:
            in_bin = (confidence >= lo) & (confidence < hi)
        else:
            in_bin = (confidence >= lo) & (confidence <= hi)

        count = in_bin.sum().item()
        if count == 0:
            continue

        bin_conf = confidence[in_bin].mean().item()
        bin_acc = accuracy[in_bin].mean().item()
        weight = count / n
        ece += weight * abs(bin_conf - bin_acc)

    return float(ece)


# ---------------------------------------------------------------------------
# MC-Dropout entropy baseline
# ---------------------------------------------------------------------------

def mc_dropout_entropy(
    model,
    image: torch.Tensor,    # [1, C, H, W, D]
    n_passes: int = 10,
) -> torch.Tensor:           # [N] entropy of the MC-averaged routing distribution
    """
    MC-Dropout entropy baseline for Proposition 2 comparison.

    Enables all nn.Dropout modules in train mode (so they randomly zero activations)
    while keeping the rest of the model in eval mode (frozen BN/InstanceNorm stats).
    Runs n_passes forward passes and computes entropy of the MEAN routing distribution.

    Returns the per-token entropy [N=4096] of the MC-averaged routing probs.

    Note: requires at least one nn.Dropout layer with p > 0 for meaningful variance.
    MONAI SwinTransformer defaults to drop_rate=0.0 — for the paper's Prop 2
    comparison, use a model retrained with drop_rate=0.1 or dropout added to the
    CCR router.
    """
    model.eval()
    for m in model.modules():
        if isinstance(m, (nn.Dropout, nn.Dropout2d, nn.Dropout3d)):
            m.train()

    routing_probs_list: List[torch.Tensor] = []
    with torch.no_grad():
        for _ in range(n_passes):
            out = model(image)
            routing_probs_list.append(out["routing_probs"][0].float())  # [N, K]

    model.eval()

    stacked = torch.stack(routing_probs_list, dim=0)  # [T, N, K]
    mean_probs = stacked.mean(dim=0)                   # [N, K]

    entropy = -(mean_probs * (mean_probs + 1e-8).log()).sum(dim=-1)  # [N]
    return entropy
