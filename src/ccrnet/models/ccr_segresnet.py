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

        # Skip-connection gate (see evals/diagnostics_202608). CCR sits at SegResNet's true
        # bottleneck, so nothing DEEPER bypasses it -- but every shallower skip in `down_x`
        # runs straight from encoder to decoder, and the diagnostic showed the decoder
        # reconstructs the segmentation through them: zeroing the entire CCR bottleneck cost
        # only ~1.5 Dice on edema. alpha scales those bypass paths:
        #   1.0 -> current behaviour, routing is decorative
        #   0.0 -> the CCR bottleneck is the only route from encoder to decoder
        # Intermediate values trace the accuracy/causal-control trade-off. The skips carry
        # spatial detail as well as semantics, so alpha=0 is expected to cost boundary
        # quality -- quantifying that cost is the point of the sweep.
        self.skip_gate = float(getattr(config.ccr, "skip_gate", 1.0))

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
        skips = list(reversed(down_x))
        if self.skip_gate != 1.0:
            # Attenuate the bypass paths so the decoder must rely on the routed bottleneck.
            # down_x[0] is the raw bottleneck and is unused by decode(), so gating it would
            # be a no-op; the remaining entries are the shallower skips that bypass CCR.
            skips = [s * self.skip_gate for s in skips]
        seg_logits = self.segresnet.decode(ccr_spatial, skips)
        seg_logits = self.boundary_head(seg_logits)

        return {
            "seg_logits":    seg_logits,
            "routing_probs": routing_probs,
            "entropy":       entropy,
            "assignments":   assignments,
        }

    def get_grid_shape(self) -> Optional[Tuple[int, int, int]]:
        return self._grid_shape
