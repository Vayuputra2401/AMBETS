"""
ccrnet.py — CCRNet: full model connecting Swin encoder, CCR bottleneck, and MONAI decoder.

Data flow (Run 3 — CCR at stage-1)
------------------------------------
  [B, 4, H, W, D]
    ↓ SwinEncoder3D
  5 hidden states at decreasing spatial resolution
    ↓ extract hs[1] = [B, 96, 32, 32, 32]  (CCR_STAGE=1)
    ↓ reshape → [B, 32768, 96]
    ↓ CCRBottleneckModule   (Phase 1 — untouched)
  expert_outputs [B, 32768, 96]
    ↓ reshape → [B, 96, 32, 32, 32]  (ccr_spatial)
    ↓ SwinUNETRDecoder  (x_in, hs, ccr_spatial as enc2 skip — stage-1 path)
    ↓ BoundaryRefinementHead
  seg_logits [B, K, H, W, D]

Stage choice rationale
----------------------
Stage-1 (32³ tokens, 4³ voxels/token) gives 80–480 NCR tokens per BraTS case
vs stage-2 (16³, 8³ voxels/token) which gives only 10–60 NCR tokens.  Pearson
correlation for CAS_fg(NCR) requires at minimum ~50 tokens per case to be
statistically reliable.  Stage-1 resolves the hard ceiling observed in Run 1–2
(NCR CAS ≈ 0.58–0.71) where the structural limit is the number of tokens,
not the quality of the routing loss.

Run history
-----------
  Run 1/2: _CCR_STAGE = 2, embed_dim=192, 16³ tokens, NCR CAS ceiling ~0.58–0.71
  Run 3+:  _CCR_STAGE = 1, embed_dim=96,  32³ tokens, target NCR CAS ≥ 0.85

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
    CCR-Net: Swin-B encoder + CCRBottleneckModule at stage-1 + MONAI UNetr decoder.

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
    # out['routing_probs']: [B, 32768, 4]   (stage-1: 32³=32768 tokens)
    # out['entropy']:       [B, 32768]
    # out['assignments']:   [B, 32768]
    """

    _CCR_STAGE: int = 1   # Swin hidden_states_out index for CCR insertion.
                           # 1 → 32³ tokens, 96-dim  (Run 3+, preferred for NCR CAS)
                           # 2 → 16³ tokens, 192-dim (Run 1–2 baseline)

    def __init__(self, config: Phase2Config) -> None:
        super().__init__()
        from ccr.modules.dispatcher import CCRBottleneckModule

        self.encoder = SwinEncoder3D(config.swin)
        self.ccr = CCRBottleneckModule(config.ccr)
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

        tokens = stage_feat.flatten(2).transpose(1, 2)   # [B, h*w*d, C]

        ccr_out = self.ccr(tokens)

        ccr_spatial = (
            ccr_out["expert_outputs"]               # [B, h*w*d, C]
            .transpose(1, 2)
            .reshape(B, C, h, w, d)
        )

        seg_logits = self.decoder(x, hs, ccr_spatial)
        seg_logits = self.boundary_head(seg_logits)

        return {
            "seg_logits":    seg_logits,
            "routing_probs": ccr_out["routing_probs"],
            "entropy":       ccr_out["entropy"],
            "assignments":   ccr_out["assignments"],
        }

    def get_grid_shape(self) -> Optional[Tuple[int, int, int]]:
        """Returns spatial grid shape of the CCR stage after forward().
        (32,32,32) for stage-1; (16,16,16) for stage-2 on 128³ input."""
        return self._grid_shape
