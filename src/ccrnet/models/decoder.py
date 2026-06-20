"""
decoder.py — SwinUNETRDecoder: MONAI UNet-style decoder for CCR-Net.

Uses MONAI's UnetrBasicBlock, UnetrUpBlock, and UnetOutBlock directly —
the same decoder components used in the original SwinUNETR paper.

Architecture (ccr_stage=1, embed_dim=48, img_size=128)   ← Run 3 default
-------------------------------------------------------
CCR inserted at stage-1 (32³ tokens, 96-dim).  POST-CCR 96-dim features
feed enc2 (32³); raw hs[2] (192-dim) feeds enc3 (16³).

  x_in  [B,  4, 128, 128, 128]  → enc0: [B, 48,  128, 128, 128]
  hs[0] [B, 48,  64,  64,  64]  → enc1: [B, 48,   64,  64,  64]
  ccr   [B, 96,  32,  32,  32]  → enc2: [B, 96,   32,  32,  32]  ← POST-CCR
  hs[2] [B,192,  16,  16,  16]  → enc3: [B,192,   16,  16,  16]  (raw Swin)
  hs[4] [B,768,   4,   4,   4]  } → enc4: [B,384,   8,   8,   8]
  hs[3] [B,384,   8,   8,   8]  }

Architecture (ccr_stage=2, embed_dim=48, img_size=128)   ← Run 1–2 baseline
-------------------------------------------------------
CCR inserted at stage-2 (16³ tokens, 192-dim).  POST-CCR 192-dim features
feed enc3 (16³); raw hs[1] (96-dim) feeds enc2 (32³).

  x_in  [B,  4, 128, 128, 128]  → enc0: [B, 48,  128, 128, 128]
  hs[0] [B, 48,  64,  64,  64]  → enc1: [B, 48,   64,  64,  64]
  hs[1] [B, 96,  32,  32,  32]  → enc2: [B, 96,   32,  32,  32]  (raw Swin)
  ccr   [B,192,  16,  16,  16]  → enc3: [B,192,   16,  16,  16]  ← POST-CCR
  hs[4] [B,768,   4,   4,   4]  } → enc4: [B,384,   8,   8,   8]
  hs[3] [B,384,   8,   8,   8]  }

Decoder path (identical for both stages):
  dec3 = dec(enc4, enc3) → [B, 192,  16,  16,  16]
  dec2 = dec(dec3, enc2) → [B,  96,  32,  32,  32]
  dec1 = dec(dec2, enc1) → [B,  48,  64,  64,  64]
  dec0 = dec(dec1, enc0) → [B,  24, 128, 128, 128]
  out  = conv(dec0)      → [B,   K, 128, 128, 128]

Channel invariant: enc2 (96-dim) and enc3 (192-dim) match both stages because
stage-1 CCR output is 96 (=2×embed_dim) and stage-2 CCR output is 192 (=4×embed_dim).
The __init__ block definitions are identical; only the forward() routing differs.
"""

from __future__ import annotations

from typing import List

import torch
import torch.nn as nn


class SwinUNETRDecoder(nn.Module):
    """
    MONAI UNet-style decoder matching the SwinUNETR architecture.

    The CCR expert_outputs are passed as `ccr_feat` and replace either the
    raw Swin stage-1 output (enc2 skip, ccr_stage=1) or the raw Swin stage-2
    output (enc3 skip, ccr_stage=2) in the decoder path.

    Parameters
    ----------
    embed_dim    : int   Swin base embed_dim (default 48).
    num_concepts : int   Number of output classes K (default 4 for BraTS).
    spatial_dims : int   Always 3 for volumetric medical images.
    in_channels  : int   MRI input modalities (default 4).
    ccr_stage    : int   Which Swin stage CCR was inserted at (1 or 2).
                         Determines whether ccr_feat feeds enc2 (stage-1, 96-dim)
                         or enc3 (stage-2, 192-dim).  Default 1 (Run 3+).

    Inputs
    ------
    x_in     : Tensor [B, in_channels, H, W, D]
    hs       : List[Tensor]  5 SwinEncoder3D hidden states
    ccr_feat : Tensor  CCR expert_outputs reshaped to spatial form
                       [B, 96, 32³] for stage-1, [B, 192, 16³] for stage-2

    Returns
    -------
    Tensor [B, K, H, W, D]   segmentation logits
    """

    def __init__(
        self,
        embed_dim: int = 48,
        num_concepts: int = 4,
        spatial_dims: int = 3,
        in_channels: int = 4,
        ccr_stage: int = 1,
    ) -> None:
        super().__init__()
        from monai.networks.blocks.unetr_block import UnetrBasicBlock, UnetrUpBlock
        from monai.networks.blocks.dynunet_block import UnetOutBlock

        if ccr_stage not in (1, 2):
            raise ValueError(f"ccr_stage must be 1 or 2, got {ccr_stage}")
        self.ccr_stage = ccr_stage

        f = embed_dim  # 48

        # Skip-connection encoders — refine raw Swin hidden states or CCR output.
        # Channel dims are the same for both stages (enc2=96, enc3=192).
        self.enc0 = UnetrBasicBlock(
            spatial_dims=spatial_dims, in_channels=in_channels,
            out_channels=f, kernel_size=3, stride=1,
            norm_name="instance", res_block=True,
        )
        self.enc1 = UnetrBasicBlock(
            spatial_dims=spatial_dims, in_channels=f,
            out_channels=f, kernel_size=3, stride=1,
            norm_name="instance", res_block=True,
        )
        # enc2: receives ccr_feat (stage-1) or hs[1] (stage-2)
        self.enc2 = UnetrBasicBlock(
            spatial_dims=spatial_dims, in_channels=f * 2,
            out_channels=f * 2, kernel_size=3, stride=1,
            norm_name="instance", res_block=True,
        )
        # enc3: receives hs[2] (stage-1) or ccr_feat (stage-2)
        self.enc3 = UnetrBasicBlock(
            spatial_dims=spatial_dims, in_channels=f * 4,
            out_channels=f * 4, kernel_size=3, stride=1,
            norm_name="instance", res_block=True,
        )
        # enc4 fuses the two deepest Swin hidden states before decoding
        self.enc4 = UnetrUpBlock(
            spatial_dims=spatial_dims, in_channels=f * 16,
            out_channels=f * 8, kernel_size=3, upsample_kernel_size=2,
            norm_name="instance", res_block=True,
        )

        # Decoder up-blocks — each doubles spatial resolution
        self.dec3 = UnetrUpBlock(
            spatial_dims=spatial_dims, in_channels=f * 8,
            out_channels=f * 4, kernel_size=3, upsample_kernel_size=2,
            norm_name="instance", res_block=True,
        )
        self.dec2 = UnetrUpBlock(
            spatial_dims=spatial_dims, in_channels=f * 4,
            out_channels=f * 2, kernel_size=3, upsample_kernel_size=2,
            norm_name="instance", res_block=True,
        )
        self.dec1 = UnetrUpBlock(
            spatial_dims=spatial_dims, in_channels=f * 2,
            out_channels=f, kernel_size=3, upsample_kernel_size=2,
            norm_name="instance", res_block=True,
        )
        self.dec0 = UnetrUpBlock(
            spatial_dims=spatial_dims, in_channels=f,
            out_channels=f, kernel_size=3, upsample_kernel_size=2,
            norm_name="instance", res_block=True,
        )

        # Final 1×1×1 conv to output K class logits
        self.out = UnetOutBlock(
            spatial_dims=spatial_dims,
            in_channels=f,
            out_channels=num_concepts,
        )

    def forward(
        self,
        x_in: torch.Tensor,
        hs: List[torch.Tensor],
        ccr_feat: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        x_in     : [B, 4,   H, W, D]   raw input (deepest skip)
        hs       : list of 5 tensors from SwinEncoder3D
        ccr_feat : [B, C, h, w, d]  CCR expert_outputs in spatial form.
                   C=96 (stage-1) or C=192 (stage-2); h=H/4 or H/8 respectively.

        Returns
        -------
        [B, K, H, W, D]   segmentation logits
        """
        enc0 = self.enc0(x_in)           # [B,  48, H,   W,   D  ]

        if self.ccr_stage == 1:
            # Stage-1 CCR (Run 3+): POST-CCR features at 32³ go into enc2 skip.
            # Raw hs[2] (16³, 192-dim) goes into enc3 skip as before.
            enc1 = self.enc1(hs[0])      # [B,  48, H/2, W/2, D/2]
            enc2 = self.enc2(ccr_feat)   # [B,  96, H/4, W/4, D/4]  ← POST-CCR
            enc3 = self.enc3(hs[2])      # [B, 192, H/8, W/8, D/8]
        else:
            # Stage-2 CCR (Run 1–2 baseline): POST-CCR features at 16³ go into enc3.
            # Raw hs[1] (32³, 96-dim) goes into enc2 skip.
            enc1 = self.enc1(hs[0])      # [B,  48, H/2, W/2, D/2]
            enc2 = self.enc2(hs[1])      # [B,  96, H/4, W/4, D/4]
            enc3 = self.enc3(ccr_feat)   # [B, 192, H/8, W/8, D/8]  ← POST-CCR

        enc4 = self.enc4(hs[4], hs[3])  # [B, 384, H/16, W/16, D/16]

        dec3 = self.dec3(enc4, enc3)    # [B, 192, H/8,  W/8,  D/8 ]
        dec2 = self.dec2(dec3, enc2)    # [B,  96, H/4,  W/4,  D/4 ]
        dec1 = self.dec1(dec2, enc1)    # [B,  48, H/2,  W/2,  D/2 ]
        dec0 = self.dec0(dec1, enc0)    # [B,  24, H,    W,    D   ]

        return self.out(dec0)           # [B,   K, H,    W,    D   ]
