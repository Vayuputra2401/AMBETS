"""
conftest.py — Shared pytest fixtures for Phase 2 tests.

All tests run on CPU with small/synthetic inputs so no GPU or real BraTS data
is required.  MONAI's SwinTransformer is instantiated with the smallest valid
configuration that still produces the correct hidden_states_out structure.

Key fixture: `phase2_config` — a full Phase2Config with tiny Swin for speed.
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path
from typing import List

import numpy as np
import pytest
import torch

# Ensure src/ and data/ are importable
ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "data"))

from ccrnet.config.phase2_config import Phase2Config, SwinConfig, DataConfig, TrainingConfig
from ccr.config.ccr_config import CCRConfig, RouterConfig, CurriculumConfig

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

B         = 1        # batch size
IN_CH     = 4        # MRI modalities
IMG_SIZE  = 64       # 64³ instead of 128³ for speed
EMBED_DIM = 48       # must match; CCR embed_dim = 4*48 = 192
K         = 4        # concepts


# ---------------------------------------------------------------------------
# Config fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def phase2_config() -> Phase2Config:
    """Minimal Phase2Config using 64³ volumes for fast CPU tests."""
    router_cfg = RouterConfig(embed_dim=192, num_concepts=K)
    curriculum  = CurriculumConfig(
        warmup_end_epoch=2, alignment_end_epoch=10, total_epochs=80
    )
    ccr_cfg = CCRConfig(router=router_cfg, curriculum=curriculum)

    swin_cfg = SwinConfig(
        img_size=(IMG_SIZE, IMG_SIZE, IMG_SIZE),
        in_channels=IN_CH,
        embed_dim=EMBED_DIM,
        window_size=(4, 4, 4),     # must divide img_size/patch_size^stage
        patch_size=(2, 2, 2),
        depths=(2, 2, 2, 2),
        num_heads=(3, 6, 12, 24),
        drop_path_rate=0.0,
        pretrained_path="",
    )

    data_cfg     = DataConfig(data_root="", brats_version="2024",
                              spatial_size=(IMG_SIZE, IMG_SIZE, IMG_SIZE))
    training_cfg = TrainingConfig(batch_size=1, amp=False, checkpoint_dir="checkpoints_test/")

    return Phase2Config(ccr=ccr_cfg, swin=swin_cfg, data=data_cfg, training=training_cfg)


@pytest.fixture
def dummy_image() -> torch.Tensor:
    """Synthetic MRI volume [B, 4, 64, 64, 64]."""
    torch.manual_seed(42)
    return torch.randn(B, IN_CH, IMG_SIZE, IMG_SIZE, IMG_SIZE)


@pytest.fixture
def dummy_label() -> torch.Tensor:
    """Synthetic label map [B, 64, 64, 64] with BraTS-like structure."""
    label = torch.zeros(B, IMG_SIZE, IMG_SIZE, IMG_SIZE, dtype=torch.long)
    label[:, 16:48, 16:48, 16:48] = 2   # edema
    label[:, 22:42, 22:42, 22:42] = 1   # NCR
    label[:, 28:36, 28:36, 28:36] = 3   # ET
    return label


# ---------------------------------------------------------------------------
# Synthetic NIfTI dataset helper (used by test_brats_dataset.py)
# ---------------------------------------------------------------------------

def _write_nifti(path: Path, arr: np.ndarray) -> None:
    import nibabel as nib
    img = nib.Nifti1Image(arr.astype(np.float32), affine=np.eye(4))
    nib.save(img, str(path))


def make_synthetic_brats_dir(
    root: Path,
    n_cases: int = 6,
    version: str = "2024",
    vol_shape: tuple = (64, 64, 64),
) -> Path:
    """
    Write synthetic NIfTI files to root so BraTSDataset can load them.

    Returns root.
    """
    rng = np.random.default_rng(0)
    for i in range(n_cases):
        case_name = f"BraTS-SYN-{i:05d}-000" if version == "2024" else f"BraTS2021_{i:05d}"
        case_dir  = root / case_name
        case_dir.mkdir(parents=True, exist_ok=True)

        if version == "2024":
            suffixes = ["-t1c", "-t1n", "-t2w", "-t2f"]
            seg_suf  = "-seg"
        else:
            suffixes = ["_t1ce", "_t1", "_t2", "_flair"]
            seg_suf  = "_seg"

        for suf in suffixes:
            _write_nifti(case_dir / f"{case_name}{suf}.nii.gz",
                         rng.standard_normal(vol_shape).astype(np.float32))

        seg = np.zeros(vol_shape, dtype=np.int16)
        seg[16:48, 16:48, 16:48] = 2
        seg[22:42, 22:42, 22:42] = 1
        seg[28:36, 28:36, 28:36] = 4 if version == "2024" else 3
        _write_nifti(case_dir / f"{case_name}{seg_suf}.nii.gz", seg.astype(np.float32))

    return root
