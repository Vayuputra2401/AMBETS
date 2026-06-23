"""
phase2_config.py — Typed configuration dataclasses for Phase 2 (CCR-Net full model).

Extends Phase 1's CCRConfig with four new sub-configs:
  - SwinConfig     : MONAI SwinTransformer hyperparameters
  - DataConfig     : BraTS dataset paths and split fractions
  - TrainingConfig : optimizer, scheduler, AMP, checkpointing

Phase 2 constraint: ccr.router.embed_dim == 4 * swin.embed_dim
  MONAI SwinTransformer with embed_dim=48 produces 4*48=192-dim features at
  stage-2 (hidden_states_out[2]), which is exactly the CCR insertion point.

YAML I/O: Phase2Config.from_yaml(path) and Phase2Config.to_yaml(path).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, asdict
from typing import Tuple

import yaml

from ccr.config.ccr_config import CCRConfig


@dataclass
class SwinConfig:
    """
    MONAI SwinTransformer 3D configuration.

    embed_dim=48 with 4 patch-merge stages produces feature dimensions:
      stage-0: [B,  48, 64, 64, 64]
      stage-1: [B,  96, 32, 32, 32]
      stage-2: [B, 192, 16, 16, 16]  ← CCR insertion (4*48=192 = CCRConfig.embed_dim)
      stage-3: [B, 384,  8,  8,  8]
      stage-4: [B, 768,  4,  4,  4]

    pretrained_path: path to a SwinUNETR .pt checkpoint from the MONAI model zoo.
    When set, the encoder portion of the checkpoint (keys prefixed 'swinViT.')
    is loaded into the SwinTransformer with strict=False.
    Leave empty ("") to train from scratch.
    """

    img_size: Tuple[int, int, int] = (128, 128, 128)
    in_channels: int = 4
    embed_dim: int = 48
    window_size: Tuple[int, int, int] = (7, 7, 7)
    patch_size: Tuple[int, int, int] = (2, 2, 2)
    depths: Tuple[int, ...] = (2, 2, 2, 2)
    num_heads: Tuple[int, ...] = (3, 6, 12, 24)
    mlp_ratio: float = 4.0
    drop_path_rate: float = 0.0
    pretrained_path: str = ""
    use_checkpoint: bool = False  # gradient checkpointing — trades compute for VRAM (set True on <24GB GPUs)


@dataclass
class DataConfig:
    """
    BraTS dataset configuration.

    brats_version controls the NIfTI filename convention:
      "2024": {case}-t1c / -t1n / -t2w / -t2f / -seg  (BraTS 2024 GLI)
      "2021": {case}_t1ce / _t1 / _t2 / _flair / _seg  (BraTS 2021)

    data_root must be set before training:
      The root should contain one sub-directory per patient, e.g.:
        data_root/
          BraTS-GLI-00001-000/
            BraTS-GLI-00001-000-t1c.nii.gz
            ...
          BraTS-GLI-00002-000/
            ...
    """

    data_root: str = ""
    brats_official_val_root: str = ""  # official BraTS val (no GT) — for leaderboard submission
    brats_version: str = "2024"
    spatial_size: Tuple[int, int, int] = (128, 128, 128)
    val_fraction: float = 0.125
    test_fraction: float = 0.125
    num_workers: int = 4
    seed: int = 42


@dataclass
class TrainingConfig:
    """
    Optimizer, scheduler, AMP, and checkpointing settings.

    Optimizer  : AdamW  (lr=1e-4, weight_decay=1e-5)
    Scheduler  : CosineAnnealingLR (T_max=80, eta_min=1e-6)
    AMP        : torch.cuda.amp.GradScaler (enabled by default)
    Checkpoints: saved to checkpoint_dir every checkpoint_every epochs
    """

    batch_size: int = 2
    grad_accum_steps: int = 1  # effective batch = batch_size * grad_accum_steps
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    grad_clip: float = 1.0
    amp: bool = True
    checkpoint_dir: str = "checkpoints/"
    checkpoint_every: int = 5
    log_every: int = 10


@dataclass
class Phase2Config:
    """
    Top-level Phase 2 configuration.

    Aggregates CCRConfig (Phase 1, unchanged) with three new sub-configs.

    Critical invariant (checked in __post_init__):
        ccr.router.embed_dim ∈ {2 * swin.embed_dim, 4 * swin.embed_dim}

    Stage-1 insertion (Run 3+): embed_dim=96  (32³ tokens, 4³ voxels/token)
    Stage-2 insertion (Run 1-2): embed_dim=192 (16³ tokens, 8³ voxels/token)

    Stage-1 is preferred for NCR CAS because it provides 8× more tokens over
    the NCR region (80-480 vs 10-60 per case), making Pearson correlation reliable.
    """

    ccr: CCRConfig = field(default_factory=CCRConfig)
    swin: SwinConfig = field(default_factory=SwinConfig)
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

    def __post_init__(self) -> None:
        # CCR can be inserted at Swin stage-1 (2×embed_dim=96) or stage-2 (4×embed_dim=192).
        # Stage-1 gives 32³ tokens (4³ voxels/token) — better NCR resolution.
        # Stage-2 gives 16³ tokens (8³ voxels/token) — original baseline.
        stage1_dim = 2 * self.swin.embed_dim
        stage2_dim = 4 * self.swin.embed_dim
        if self.ccr.router.embed_dim not in (stage1_dim, stage2_dim):
            raise ValueError(
                f"ccr.router.embed_dim ({self.ccr.router.embed_dim}) must be "
                f"2× swin.embed_dim ({stage1_dim}, stage-1) or "
                f"4× swin.embed_dim ({stage2_dim}, stage-2)."
            )
        if self.ccr.curriculum.total_epochs != 80:
            raise ValueError(
                "ccr.curriculum.total_epochs must be 80 (the Phase 1 curriculum is fixed)."
            )

    # ------------------------------------------------------------------
    # YAML I/O
    # ------------------------------------------------------------------

    @classmethod
    def from_yaml(cls, path: str) -> "Phase2Config":
        """Load config from a YAML file, merging with defaults."""
        with open(path, "r") as fh:
            raw = yaml.safe_load(fh) or {}

        ccr_raw  = raw.pop("ccr", {})
        swin_raw = raw.pop("swin", {})
        data_raw = raw.pop("data", {})
        train_raw = raw.pop("training", {})

        ccr_cfg = CCRConfig()
        if ccr_raw:
            router_raw     = ccr_raw.pop("router", {})
            expert_raw     = ccr_raw.pop("expert", {})
            curriculum_raw = ccr_raw.pop("curriculum", {})
            loss_raw       = ccr_raw.pop("loss", {})
            if router_raw:
                from ccr.config.ccr_config import RouterConfig
                for k, v in router_raw.items():
                    if hasattr(ccr_cfg.router, k):
                        setattr(ccr_cfg.router, k, v)
                # auto-sync expert.embed_dim after router is applied from YAML
                ccr_cfg.expert.embed_dim = ccr_cfg.router.embed_dim
            if expert_raw:
                for k, v in expert_raw.items():
                    if hasattr(ccr_cfg.expert, k):
                        setattr(ccr_cfg.expert, k, v)
            if curriculum_raw:
                for k, v in curriculum_raw.items():
                    if hasattr(ccr_cfg.curriculum, k):
                        setattr(ccr_cfg.curriculum, k, v)
            if loss_raw:
                loss_weights_raw = loss_raw.pop("weights", {})
                for k, v in loss_raw.items():
                    if hasattr(ccr_cfg.loss, k):
                        cur = getattr(ccr_cfg.loss, k)
                        if isinstance(cur, tuple) and isinstance(v, list):
                            v = tuple(v)
                        setattr(ccr_cfg.loss, k, v)
                if loss_weights_raw:
                    for k, v in loss_weights_raw.items():
                        if hasattr(ccr_cfg.loss.weights, k):
                            if isinstance(v, list):
                                v = tuple(v)
                            setattr(ccr_cfg.loss.weights, k, v)

        def _apply(dcls, d):
            obj = dcls()
            for k, v in d.items():
                if hasattr(obj, k):
                    cur = getattr(obj, k)
                    if isinstance(cur, tuple) and isinstance(v, list):
                        v = tuple(v)
                    setattr(obj, k, v)
            return obj

        swin_cfg  = _apply(SwinConfig, swin_raw)
        data_cfg  = _apply(DataConfig, data_raw)
        train_cfg = _apply(TrainingConfig, train_raw)

        return cls(ccr=ccr_cfg, swin=swin_cfg, data=data_cfg, training=train_cfg)

    def to_yaml(self, path: str) -> None:
        """Serialize config to YAML."""
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        d = {
            "swin": {
                k: (list(v) if isinstance(v, tuple) else v)
                for k, v in asdict(self.swin).items()
            },
            "data": asdict(self.data),
            "training": asdict(self.training),
            "ccr": {
                "router": {
                    k: v for k, v in asdict(self.ccr.router).items()
                },
                "curriculum": {
                    k: v for k, v in asdict(self.ccr.curriculum).items()
                },
            },
        }
        with open(path, "w") as fh:
            yaml.dump(d, fh, default_flow_style=False, sort_keys=False)
