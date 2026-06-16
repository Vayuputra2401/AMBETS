"""
test_brats_dataset.py — BraTSDataset unit tests using synthetic NIfTI files.

No real BraTS data required. Synthetic cases are written to a temp directory.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "data"))

from tests.phase2.conftest import make_synthetic_brats_dir


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def synthetic_root_2024(tmp_path_factory):
    root = tmp_path_factory.mktemp("brats2024")
    make_synthetic_brats_dir(root, n_cases=6, version="2024", vol_shape=(64, 64, 64))
    return root


@pytest.fixture(scope="module")
def synthetic_root_2021(tmp_path_factory):
    root = tmp_path_factory.mktemp("brats2021")
    make_synthetic_brats_dir(root, n_cases=6, version="2021", vol_shape=(64, 64, 64))
    return root


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_dataset_len(synthetic_root_2024):
    from brats_dataset import BraTSDataset, get_patient_dirs, split_patients
    dirs = get_patient_dirs(str(synthetic_root_2024))
    splits = split_patients(dirs, val_frac=0.166, test_frac=0.166, seed=42)
    ds = BraTSDataset(splits["train"], split="train",
                      spatial_size=(64, 64, 64), brats_version="2024")
    assert len(ds) > 0, "Train split dataset is empty"


def test_dataset_image_shape(synthetic_root_2024):
    from brats_dataset import BraTSDataset, get_patient_dirs, split_patients
    dirs   = get_patient_dirs(str(synthetic_root_2024))
    splits = split_patients(dirs, val_frac=0.166, test_frac=0.166, seed=42)
    ds = BraTSDataset(splits["train"], split="train",
                      spatial_size=(64, 64, 64), brats_version="2024")
    item = ds[0]
    assert item["image"].shape == (4, 64, 64, 64), (
        f"image shape {item['image'].shape} != (4,64,64,64)"
    )
    assert item["image"].dtype == torch.float32


def test_dataset_label_shape_dtype(synthetic_root_2024):
    from brats_dataset import BraTSDataset, get_patient_dirs, split_patients
    dirs   = get_patient_dirs(str(synthetic_root_2024))
    splits = split_patients(dirs, val_frac=0.166, test_frac=0.166, seed=42)
    ds = BraTSDataset(splits["train"], split="train",
                      spatial_size=(64, 64, 64), brats_version="2024")
    item = ds[0]
    assert item["label"].shape == (64, 64, 64), (
        f"label shape {item['label'].shape} != (64,64,64)"
    )
    assert item["label"].dtype == torch.long


def test_label_values_valid_2024(synthetic_root_2024):
    """BraTS 2024 has label 4 (ET) which must be remapped to 3."""
    from brats_dataset import BraTSDataset, get_patient_dirs, split_patients
    dirs   = get_patient_dirs(str(synthetic_root_2024))
    splits = split_patients(dirs, val_frac=0.166, test_frac=0.166, seed=42)
    ds = BraTSDataset(splits["train"], split="train",
                      spatial_size=(64, 64, 64), brats_version="2024")
    for i in range(len(ds)):
        lbl = ds[i]["label"]
        assert lbl.min() >= 0, f"Negative label value in sample {i}"
        assert lbl.max() <= 3, f"Label value > 3 in sample {i}: max={lbl.max()}"


def test_brats_2021_naming(synthetic_root_2021):
    """BraTS 2021 naming (_t1ce, _t1, _t2, _flair) must load correctly."""
    from brats_dataset import BraTSDataset, get_patient_dirs, split_patients
    dirs   = get_patient_dirs(str(synthetic_root_2021))
    splits = split_patients(dirs, val_frac=0.166, test_frac=0.166, seed=42)
    ds = BraTSDataset(splits["train"], split="train",
                      spatial_size=(64, 64, 64), brats_version="2021")
    assert len(ds) > 0
    item = ds[0]
    assert item["image"].shape == (4, 64, 64, 64)


def test_train_split_has_augmentation(synthetic_root_2024):
    """Train and val transforms must be distinct (val has no random flips)."""
    from monai.transforms import RandFlipd
    from brats_dataset import _build_transforms
    train_t = _build_transforms("train", (64, 64, 64), "2024")
    val_t   = _build_transforms("val",   (64, 64, 64), "2024")

    def _has_flip(t):
        for tr in t.transforms:
            if isinstance(tr, RandFlipd):
                return True
        return False

    assert _has_flip(train_t), "Train transforms missing RandFlipd"
    assert not _has_flip(val_t), "Val transforms should not have RandFlipd"


def test_patient_id_returned(synthetic_root_2024):
    from brats_dataset import BraTSDataset, get_patient_dirs, split_patients
    dirs   = get_patient_dirs(str(synthetic_root_2024))
    splits = split_patients(dirs, val_frac=0.166, test_frac=0.166, seed=42)
    ds = BraTSDataset(splits["train"], split="train",
                      spatial_size=(64, 64, 64), brats_version="2024")
    item = ds[0]
    assert "patient_id" in item, "patient_id not returned"
    assert isinstance(item["patient_id"], str) and len(item["patient_id"]) > 0


def test_split_counts_sum_to_total(synthetic_root_2024):
    from brats_dataset import get_patient_dirs, split_patients
    dirs   = get_patient_dirs(str(synthetic_root_2024))
    splits = split_patients(dirs, val_frac=0.166, test_frac=0.166, seed=42)
    total = len(dirs)
    split_total = sum(len(v) for v in splits.values())
    assert split_total == total, (
        f"Split counts {split_total} != total patients {total}"
    )
