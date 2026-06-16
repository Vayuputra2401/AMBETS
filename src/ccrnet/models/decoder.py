"""
decoder.py — SwinUNETRDecoder: MONAI UNet-style decoder for CCR-Net.

Uses MONAI's UnetrBasicBlock, UnetrUpBlock, and UnetOutBlock directly —
the same decoder components used in the original SwinUNETR paper.

Architecture
------------
Mirrors the SwinUNETR decoder exactly, with one modification:
the skip connection at dec2 (enc3 in SwinUNETR terms) receives the
POST-CCR features (expert_outputs reshaped) instead of the raw Swin
stage-2 hidden state.  All other skip connections are unchanged.

Feature dimensions (embed_dim=48, img_size=128):

  x_in  [B,  4, 128, 128, 128]  → enc0: [B, 48,  128, 128, 128]
  hs[0] [B, 48,  64,  64,  64]  → enc1: [B, 48,   64,  64,  64]
  hs[1] [B, 96,  32,  32,  32]  → enc2: [B, 96,   32,  32,  32]
  ccr   [B,192,  16,  16,  16]  → enc3: [B,192,   16,  16,  16]  ← POST-CCR
  hs[4] [B,768,   4,   4,   4]  } → enc4: [B,384,    8,   8,   8]
  hs[3] [B,384,   8,   8,   8]  }

  dec3 = dec(enc4, enc3) → [B, 192,  16,  16,  16]
  dec2 = dec(dec3, enc2) → [B,  96,  32,  32,  32]
  dec1 = dec(dec2, enc1) → [B,  48,  64,  64,  64]
  dec0 = dec(dec1, enc0) → [B,  24, 128, 128, 128]
  out  = conv(dec0)      → [B,   K, 128, 128, 128]

Plan reference: Phase 2 plan — decoder section.
"""

from __future__ import annotations

from typing import List, Tuple

import torch
import torch.nn as nn


class SwinUNETRDecoder(nn.Module):
    """
    MONAI UNet-style decoder matching the SwinUNETR architecture.

    The CCR expert_outputs are passed as `ccr_feat` and replace the
    raw Swin stage-2 output (hs[2]) in the decoder path.  This ensures
    all decoder operations are conditioned on concept-specific features.

    Parameters
    ----------
    embed_dim    : int   Swin base embed_dim (default 48).  All channel counts
                         are multiples of this value.
    num_concepts : int   Number of output classes K (default 4 for BraTS).
    spatial_dims : int   Always 3 for volumetric medical images.

    Inputs
    ------
    x_in     : Tensor [B, in_channels, H, W, D]   raw input image (deepest skip)
    hs       : List[Tensor]   5 SwinEncoder3D hidden states
    ccr_feat : Tensor [B, 4*embed_dim, H/8, W/8, D/8]   CCR expert outputs (spatial)

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
    ) -> None:
        super().__init__()
        from monai.networks.blocks.unetr_block import UnetrBasicBlock, UnetrUpBlock
        from monai.networks.blocks.dynunet_block import UnetOutBlock

        f = embed_dim  # 48

        # Skip connection encoders — refine raw Swin hidden states
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
        self.enc2 = UnetrBasicBlock(
            spatial_dims=spatial_dims, in_channels=f * 2,
            out_channels=f * 2, kernel_size=3, stride=1,
            norm_name="instance", res_block=True,
        )
        # enc3 receives post-CCR features (ccr_feat = expert_outputs spatial)
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
        x_in     : [B, 4,   128, 128, 128]   raw input (deepest skip)
        hs       : list of 5 tensors from SwinEncoder3D
        ccr_feat : [B, 192,  16,  16,  16]   CCR expert_outputs (spatial)

        Returns
        -------
        [B, K, 128, 128, 128]   segmentation logits
        """
        enc0 = self.enc0(x_in)           # [B,  48, 128, 128, 128]
        enc1 = self.enc1(hs[0])          # [B,  48,  64,  64,  64]
        enc2 = self.enc2(hs[1])          # [B,  96,  32,  32,  32]
        enc3 = self.enc3(ccr_feat)       # [B, 192,  16,  16,  16]  ← post-CCR
        enc4 = self.enc4(hs[4], hs[3])   # [B, 384,   8,   8,   8]

        dec3 = self.dec3(enc4, enc3)     # [B, 192,  16,  16,  16]
        dec2 = self.dec2(dec3, enc2)     # [B,  96,  32,  32,  32]
        dec1 = self.dec1(dec2, enc1)     # [B,  48,  64,  64,  64]
        dec0 = self.dec0(dec1, enc0)     # [B,  24, 128, 128, 128]

        return self.out(dec0)            # [B,   K, 128, 128, 128]
