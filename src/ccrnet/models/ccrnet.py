"""
ccrnet.py — CCRNet: full model connecting Swin encoder, CCR bottleneck, and MONAI decoder.

Data flow (Run 4 — CCR at stage-2)
------------------------------------
  [B, 4, H, W, D]
    ↓ SwinEncoder3D
  5 hidden states at decreasing spatial resolution
    ↓ extract hs[2] = [B, 192, 16, 16, 16]  (CCR_STAGE=2)
    ↓ reshape → [B, 4096, 192]
    ↓ CCRBottleneckModule   (Phase 1 — untouched)
  expert_outputs [B, 4096, 192]
    ↓ reshape → [B, 192, 16, 16, 16]  (ccr_spatial)
    ↓ SwinUNETRDecoder  (x_in, hs, ccr_spatial as enc3 skip — stage-2 path)
    ↓ BoundaryRefinementHead
  seg_logits [B, K, H, W, D]

Stage choice rationale
----------------------
Stage-2 (16³ tokens, 8³ voxels/token, 192-dim) is preferred over stage-1 because
the 192-dim features have more global semantic context after an extra patch merge,
making NCR/ET discrimination more reliable despite fewer tokens.
Run 3 (stage-1) showed that lower-level 96-dim features caused over-routing to NCR
(util 32% vs true ~12%), dragging CAS down to 0.53 vs 0.71 at stage-2.

Run history
-----------
  Run 1:   _CCR_STAGE=2, embed_dim=192, lam_align_refine=0.5, align_end=50. NCR final=0.580
  Run 2:   _CCR_STAGE=2, embed_dim=192, lam_align_refine=0.5, align_end=50. NCR peak=0.706, final=0.698
  Run 3:   _CCR_STAGE=1, embed_dim=96,  lam_align_refine=1.0, align_end=60. NCR plateau=0.534 (stage-1 worse)
  Run 4+:  _CCR_STAGE=2, embed_dim=192, lam_align_refine=1.0, align_end=60. Target: hold 0.706+ through refinement

Output dict keys
----------------
  seg_logits    : [B, K, 128, 128, 128]   → CCRTotalLoss.pred_logits
  routing_probs : [B, 32768, K]            → CCRTotalLoss.routing_probs  (stage-1)
  entropy       : [B, 32768]              → uncertainty maps
  assignments   : [B, 32768]              → ExpertUtilizationTracker
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

from ccrnet.config.phase2_config import Phase2Config
from ccrnet.models.encoder import SwinEncoder3D
from ccrnet.models.decoder import SwinUNETRDecoder
from ccrnet.models.boundary_head import BoundaryRefinementHead


class CCRNet(nn.Module):
    """
    CCR-Net: Swin-B encoder + CCRBottleneckModule at stage-2 + MONAI UNetr decoder.

    Parameters
    ----------
    config : Phase2Config
        Full configuration.  ccr.router.embed_dim must equal 2×swin.embed_dim (stage-1)
        or 4×swin.embed_dim (stage-2) — enforced by Phase2Config.__post_init__.

    Usage
    -----
    model = CCRNet(config)
    out   = model(image)          # image: [B, 4, 128, 128, 128]
    # out['seg_logits']:    [B, 4, 128, 128, 128]
    # out['routing_probs']: [B, 4096, 4]   (stage-2: 16³=4096 tokens)
    # out['entropy']:       [B, 4096]
    # out['assignments']:   [B, 4096]
    """

    _CCR_STAGE: int = 2   # Swin hidden_states_out index for CCR insertion.
                           # 2 → 16³ tokens, 192-dim (Run 1–2, Run 4+ — preferred)
                           # 1 → 32³ tokens, 96-dim  (Run 3 — tried, worse due to feature ambiguity)

    def __init__(self, config: Phase2Config) -> None:
        super().__init__()

        # No-CCR control (W4): when disabled, the model is a plain Swin+UNETR
        # backbone — raw stage-2 features feed the decoder enc3 skip, with no router,
        # no experts, and no routing losses. Used to show CCR is accuracy-neutral.
        self.ccr_enabled = getattr(config.ccr, "enabled", True)

        self.encoder = SwinEncoder3D(config.swin)
        if self.ccr_enabled:
            from ccr.modules.dispatcher import CCRBottleneckModule
            self.ccr = CCRBottleneckModule(config.ccr)
        else:
            self.ccr = None
        self.decoder = SwinUNETRDecoder(
            embed_dim=config.swin.embed_dim,
            num_concepts=config.ccr.router.num_concepts,
            spatial_dims=3,
            in_channels=config.swin.in_channels,
            ccr_stage=self._CCR_STAGE,
        )
        self.boundary_head = BoundaryRefinementHead(
            num_concepts=config.ccr.router.num_concepts,
        )
        self._grid_shape: Optional[Tuple[int, int, int]] = None

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Parameters
        ----------
        x : [B, 4, H, W, D]   multi-modal MRI volume (H=W=D=128)

        Returns
        -------
        dict with keys: seg_logits, routing_probs, entropy, assignments
        """
        hs = self.encoder(x)

        stage_feat = hs[self._CCR_STAGE]            # [B, C, h, w, d]
        B, C, h, w, d = stage_feat.shape            # C=96 (stage-1) or 192 (stage-2)
        self._grid_shape = (h, w, d)                # (32,32,32) or (16,16,16)

        if self.ccr_enabled:
            tokens = stage_feat.flatten(2).transpose(1, 2)   # [B, h*w*d, C]
            ccr_out = self.ccr(tokens)
            ccr_spatial = (
                ccr_out["expert_outputs"]               # [B, h*w*d, C]
                .transpose(1, 2)
                .reshape(B, C, h, w, d)
            )
            routing_probs = ccr_out["routing_probs"]
            entropy       = ccr_out["entropy"]
            assignments   = ccr_out["assignments"]
        else:
            # No-CCR control: raw stage-2 features are the enc3 skip; no routing.
            ccr_spatial   = stage_feat
            routing_probs = None
            entropy       = None
            assignments   = None

        seg_logits = self.decoder(x, hs, ccr_spatial)
        seg_logits = self.boundary_head(seg_logits)

        return {
            "seg_logits":    seg_logits,
            "routing_probs": routing_probs,
            "entropy":       entropy,
            "assignments":   assignments,
        }

    def get_grid_shape(self) -> Optional[Tuple[int, int, int]]:
        """Returns spatial grid shape of the CCR stage after forward().
        (32,32,32) for stage-1; (16,16,16) for stage-2 on 128³ input."""
        return self._grid_shape
