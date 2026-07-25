"""
ccr_segresnet.py — CCR-SegResNet: the CNN instantiation of the CCR principle.

Demonstrates that CCR is a drop-in bottleneck *module*, not a Swin-specific model. A MONAI
SegResNet (ResNet-style CNN encoder-decoder) has the SAME CCRBottleneckModule inserted at its
bottleneck. For a 128^3 input the SegResNet bottleneck is $192\times16^3$ --- identical in shape
to the Swin stage-2 insertion in CCRNet --- so the CCR module, all five losses, the curriculum,
and every metric are reused byte-for-byte. Only the encoder/decoder wrapper differs.

Data flow
---------
  [B, 4, 128, 128, 128]
    -> SegResNet.encode           -> bottleneck [B, 192, 16, 16, 16] + skip features
    -> flatten to tokens [B, 4096, 192]
    -> CCRBottleneckModule        (unchanged)
    -> reshape -> [B, 192, 16, 16, 16]
    -> SegResNet.decode(ccr, skips) -> [B, K, 128, 128, 128]
    -> BoundaryRefinementHead
  seg_logits [B, K, 128, 128, 128]

Selected via `--model segresnet` (see ccrnet.models.factory.build_model). Reuses Phase2Config:
`ccr.router.embed_dim` (192) sets the bottleneck width, so the stage invariant still holds.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
from monai.networks.nets import SegResNet

from ccrnet.config.phase2_config import Phase2Config
from ccrnet.models.boundary_head import BoundaryRefinementHead

_BLOCKS_DOWN = (1, 2, 2, 4)   # 3 downsamples -> bottleneck at input/8 (16^3 for 128^3)


class CCRSegResNet(nn.Module):
    """CNN backbone (MONAI SegResNet) + the CCR bottleneck module at its deepest level."""

    def __init__(self, config: Phase2Config) -> None:
        super().__init__()
        self.ccr_enabled = getattr(config.ccr, "enabled", True)

        embed_dim = config.ccr.router.embed_dim          # bottleneck channels (192)
        factor = 2 ** (len(_BLOCKS_DOWN) - 1)            # 8
        if embed_dim % factor != 0:
            raise ValueError(f"ccr.router.embed_dim ({embed_dim}) must be divisible by {factor}")
        init_filters = embed_dim // factor               # 24 -> bottleneck 24*8 = 192

        self.segresnet = SegResNet(
            spatial_dims=3,
            in_channels=config.swin.in_channels,
            out_channels=config.ccr.router.num_concepts,
            init_filters=init_filters,
            blocks_down=_BLOCKS_DOWN,
            blocks_up=(1, 1, 1),
            dropout_prob=(config.swin.drop_path_rate or None),
        )

        if self.ccr_enabled:
            from ccr.modules.dispatcher import CCRBottleneckModule
            self.ccr = CCRBottleneckModule(config.ccr)
        else:
            self.ccr = None

        self.boundary_head = BoundaryRefinementHead(num_concepts=config.ccr.router.num_concepts)
        self._grid_shape: Optional[Tuple[int, int, int]] = None

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        bottleneck, down_x = self.segresnet.encode(x)    # bottleneck [B, C, h, w, d]
        B, C, h, w, d = bottleneck.shape
        self._grid_shape = (h, w, d)

        if self.ccr_enabled:
            tokens = bottleneck.flatten(2).transpose(1, 2)          # [B, h*w*d, C]
            ccr_out = self.ccr(tokens)
            ccr_spatial = (
                ccr_out["expert_outputs"].transpose(1, 2).reshape(B, C, h, w, d)
            )
            routing_probs = ccr_out["routing_probs"]
            entropy       = ccr_out["entropy"]
            assignments   = ccr_out["assignments"]
        else:
            ccr_spatial   = bottleneck
            routing_probs = entropy = assignments = None

        # decode() consumes the reversed skip list; down_x[0] (the raw bottleneck) is unused,
        # so passing ccr_spatial as the starting feature routes the CCR output into the decoder.
        seg_logits = self.segresnet.decode(ccr_spatial, list(reversed(down_x)))
        seg_logits = self.boundary_head(seg_logits)

        return {
            "seg_logits":    seg_logits,
            "routing_probs": routing_probs,
            "entropy":       entropy,
            "assignments":   assignments,
        }

    def get_grid_shape(self) -> Optional[Tuple[int, int, int]]:
        return self._grid_shape
