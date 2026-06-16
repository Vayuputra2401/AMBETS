"""
encoder.py — SwinEncoder3D: MONAI SwinTransformer wrapper for CCR-Net.

Uses monai.networks.nets.swin_unetr.SwinTransformer directly — the same
encoder used inside SwinUNETR, unchanged.

CCR insertion point
-------------------
For img_size=128, embed_dim=48, patch_size=(2,2,2), depths=(2,2,2,2):

  hidden_states_out[0]: [B,  48, 64, 64, 64]
  hidden_states_out[1]: [B,  96, 32, 32, 32]
  hidden_states_out[2]: [B, 192, 16, 16, 16]  ← CCR_STAGE
  hidden_states_out[3]: [B, 384,  8,  8,  8]
  hidden_states_out[4]: [B, 768,  4,  4,  4]

Stage 2 gives exactly 192-dim at 16^3 = 4096 tokens — the CCRBottleneckModule
input shape (embed_dim=192) matches this with zero projection overhead.

Pretrained weights
------------------
MONAI provides SwinUNETR pretrained on Task01_BrainTumour (Medical Segmentation
Decathlon). Set pretrained_path to the downloaded .pt file. Keys from the full
SwinUNETR checkpoint prefixed with 'swinViT.' are mapped to the SwinTransformer.

Plan reference: Phase 2 plan — encoder section.
"""

from __future__ import annotations

from typing import List

import torch
import torch.nn as nn

from ccrnet.config.phase2_config import SwinConfig


class SwinEncoder3D(nn.Module):
    """
    MONAI Swin-B 3D encoder for CCR-Net.

    Wraps monai.networks.nets.swin_unetr.SwinTransformer to produce a list of
    5 hidden-state tensors at progressively reduced spatial resolutions.
    CCRNet extracts hidden_states_out[CCR_STAGE=2] as the CCR insertion point.

    Parameters
    ----------
    config : SwinConfig
        MONAI SwinTransformer hyperparameters.

    Inputs
    ------
    x : Tensor [B, 4, H, W, D]
        Multi-modal MRI volume. 4 channels = (T1c, T1n, T2w, T2f) for BraTS 2024
        or (T1ce, T1, T2, FLAIR) for BraTS 2021.

    Returns
    -------
    List[Tensor]   5 feature tensors:
        [0]: [B,  C,   H/2,  W/2,  D/2 ]   stage-0
        [1]: [B, 2C,   H/4,  W/4,  D/4 ]   stage-1
        [2]: [B, 4C,   H/8,  W/8,  D/8 ]   stage-2  ← CCR insertion
        [3]: [B, 8C,   H/16, W/16, D/16]   stage-3
        [4]: [B, 16C,  H/32, W/32, D/32]   bottleneck
        where C = config.embed_dim (default 48)
    """

    CCR_STAGE: int = 2

    def __init__(self, config: SwinConfig) -> None:
        super().__init__()
        from monai.networks.nets.swin_unetr import SwinTransformer as MonaiSwinTransformer

        # MONAI >= 1.4 removed img_size from SwinTransformer.__init__
        self.swin = MonaiSwinTransformer(
            in_chans=config.in_channels,
            embed_dim=config.embed_dim,
            window_size=config.window_size,
            patch_size=config.patch_size,
            depths=config.depths,
            num_heads=config.num_heads,
            mlp_ratio=config.mlp_ratio,
            qkv_bias=True,
            drop_rate=0.0,
            attn_drop_rate=0.0,
            drop_path_rate=config.drop_path_rate,
            norm_layer=nn.LayerNorm,
            use_checkpoint=False,
            spatial_dims=3,
        )

        if config.pretrained_path:
            self._load_pretrained(config.pretrained_path)

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Run the Swin encoder and return all 5 hidden states.

        Parameters
        ----------
        x : Tensor [B, 4, H, W, D]

        Returns
        -------
        List of 5 Tensors — one per encoder stage.
        """
        return self.swin(x, normalize=True)

    def _load_pretrained(self, path: str) -> None:
        """
        Load MONAI SSL pretrained SwinTransformer weights (model_swinvit.pt).

        The checkpoint was saved with DataParallel, so all keys start with 'module.'.
        MONAI 1.5 renamed MLP layers from fc1/fc2 to linear1/linear2.
        Both transformations are applied before loading.

        SSL-only heads (contrastive_head, rotation_head, convTrans3d) are not
        present in our model and are silently ignored via strict=False.

        PyTorch 2.6+ requires weights_only=False for this checkpoint.
        """
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        state = ckpt.get("state_dict", ckpt)

        mapped = {}
        for k, v in state.items():
            k2 = k.removeprefix("module.")           # strip DataParallel prefix
            k2 = k2.replace("mlp.fc1", "mlp.linear1").replace("mlp.fc2", "mlp.linear2")
            mapped[k2] = v

        if not mapped:
            raise ValueError(f"[SwinEncoder3D] Empty state dict after remapping from {path}.")

        # Handle patch_embed channel mismatch (SSL trained on 1-ch, model uses 4-ch).
        # Tile the 1-channel weight across in_channels and divide by in_channels
        # so a uniform 4-channel input produces identical activations as the
        # 1-channel pretrained model (standard ViT multi-channel inflation).
        pe_key = "patch_embed.proj.weight"
        model_sd = self.swin.state_dict()
        if pe_key in mapped and pe_key in model_sd:
            w_ckpt  = mapped[pe_key]   # [C_out, 1, kD, kH, kW]  or [C_out, C_in_ckpt, ...]
            w_model = model_sd[pe_key] # [C_out, C_in_model, kD, kH, kW]
            if w_ckpt.shape != w_model.shape:
                c_ckpt  = w_ckpt.shape[1]
                c_model = w_model.shape[1]
                mapped[pe_key] = w_ckpt.repeat(1, c_model, 1, 1, 1)[:, :c_model] / c_ckpt
                print(
                    f"[SwinEncoder3D] patch_embed inflated: {c_ckpt}-ch -> {c_model}-ch "
                    f"(tiled + /={c_ckpt})."
                )

        missing, unexpected = self.swin.load_state_dict(mapped, strict=False)
        n_model  = len(model_sd)
        n_loaded = n_model - len(missing)
        print(
            f"[SwinEncoder3D] Loaded {n_loaded}/{n_model} pretrained keys "
            f"({len(unexpected)} checkpoint-only SSL heads ignored)."
        )
        if len(missing) > 0:
            import warnings
            warnings.warn(
                f"[SwinEncoder3D] {len(missing)} model keys not found in checkpoint: {missing[:3]}",
                RuntimeWarning,
            )
