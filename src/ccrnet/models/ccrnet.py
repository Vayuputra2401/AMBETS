"""
ccrnet.py — CCRNet: full model connecting Swin encoder, CCR bottleneck, and MONAI decoder.

Data flow
---------
  [B, 4, H, W, D]
    ↓ SwinEncoder3D
  5 hidden states at decreasing spatial resolution
    ↓ extract hs[2] = [B, 192, 16, 16, 16]  (CCR_STAGE=2)
    ↓ reshape → [B, 4096, 192]
    ↓ CCRBottleneckModule   (Phase 1 — untouched)
  expert_outputs [B, 4096, 192]
    ↓ reshape → [B, 192, 16, 16, 16]  (ccr_spatial)
    ↓ SwinUNETRDecoder  (x_in, hs, ccr_spatial as enc3 skip)
    ↓ BoundaryRefinementHead
  seg_logits [B, K, H, W, D]

Output dict keys
----------------
  seg_logits    : [B, K, 128, 128, 128]   → CCRTotalLoss.pred_logits
  routing_probs : [B, 4096, K]             → CCRTotalLoss.routing_probs
  entropy       : [B, 4096]               → uncertainty maps
  assignments   : [B, 4096]               → ExpertUtilizationTracker
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

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
        Full configuration with ccr, swin, data, and training sub-configs.
        The invariant `ccr.router.embed_dim == 4 * swin.embed_dim` is enforced
        by Phase2Config.__post_init__.

    Usage
    -----
    model = CCRNet(config)
    out   = model(image)          # image: [B, 4, 128, 128, 128]
    # out['seg_logits']:    [B, 4, 128, 128, 128]
    # out['routing_probs']: [B, 4096, 4]
    # out['entropy']:       [B, 4096]
    # out['assignments']:   [B, 4096]
    """

    _CCR_STAGE: int = 2

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

        stage2 = hs[self._CCR_STAGE]
        B, C, h, w, d = stage2.shape
        self._grid_shape = (h, w, d)

        tokens = stage2.flatten(2).transpose(1, 2)      # [B, h*w*d, C]

        ccr_out = self.ccr(tokens)

        ccr_spatial = (
            ccr_out["expert_outputs"]                    # [B, h*w*d, C]
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
        """Returns spatial grid shape of CCR stage (e.g. (16,16,16)) after forward()."""
        return self._grid_shape
