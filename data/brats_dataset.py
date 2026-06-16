"""
brats_dataset.py — BraTSDataset and get_dataloader() using MONAI transforms.

Supports BraTS 2024 (GLI) and BraTS 2021 naming conventions.

BraTS 2024 filenames: {case}-t1c / -t1n / -t2w / -t2f / -seg .nii.gz
BraTS 2021 filenames: {case}_t1ce / _t1 / _t2 / _flair / _seg .nii.gz

Labels: 0=Background, 1=NCR, 2=Edema, 3=ET — matches CCRConfig.concept_names.

Preprocessing (all splits):
  LoadImaged → EnsureChannelFirstd → NormalizeIntensityd (z-score, nonzero, per-channel)
  → CropForegroundd (margin=10) → SpatialPadd(128) → RandCropByPosNegLabeld

Training-only augmentation:
  RandFlipd + RandRotate90d + RandScaleIntensityd + RandShiftIntensityd + RandGaussianNoised
"""

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from monai.data import CacheDataset
from monai.transforms import (
    CropForegroundd,
    EnsureChannelFirstd,
    EnsureTyped,
    LoadImaged,
    NormalizeIntensityd,
    RandCropByPosNegLabeld,
    RandFlipd,
    RandGaussianNoised,
    RandRotate90d,
    RandScaleIntensityd,
    RandShiftIntensityd,
    SpatialPadd,
    Compose,
    MapTransform,
)


# ---------------------------------------------------------------------------
# Custom transform: map BraTS 2024 label 4 → 3  (ET in 2024 is label 4)
# ---------------------------------------------------------------------------

class RemapBraTS2024Labels(MapTransform):
    """
    BraTS 2024 uses label 4 for Enhancing Tumor (ET).
    Remap to label 3 so all splits use {0,1,2,3} = {BG, NCR, ED, ET}.
    """

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            seg = d[key]
            seg[seg == 4] = 3
            d[key] = seg
        return d


# ---------------------------------------------------------------------------
# File-system helpers
# ---------------------------------------------------------------------------

def _modality_filenames(case_name: str, brats_version: str) -> Dict[str, str]:
    if brats_version == "2024":
        return {
            "t1c":   f"{case_name}-t1c.nii.gz",
            "t1n":   f"{case_name}-t1n.nii.gz",
            "t2w":   f"{case_name}-t2w.nii.gz",
            "t2f":   f"{case_name}-t2f.nii.gz",
            "label": f"{case_name}-seg.nii.gz",
        }
    else:  # "2021"
        return {
            "t1c":   f"{case_name}_t1ce.nii.gz",
            "t1n":   f"{case_name}_t1.nii.gz",
            "t2w":   f"{case_name}_t2.nii.gz",
            "t2f":   f"{case_name}_flair.nii.gz",
            "label": f"{case_name}_seg.nii.gz",
        }


def get_patient_dirs(data_root: str) -> List[Path]:
    """Return sorted list of patient subdirectories under data_root."""
    root = Path(data_root)
    return sorted([p for p in root.iterdir() if p.is_dir()])


def split_patients(
    patient_dirs: List[Path],
    val_frac: float,
    test_frac: float,
    seed: int,
) -> Dict[str, List[Path]]:
    """Deterministic train / val / test split by patient."""
    rng = random.Random(seed)
    dirs = list(patient_dirs)
    rng.shuffle(dirs)
    n = len(dirs)
    n_test = max(1, int(n * test_frac))
    n_val  = max(1, int(n * val_frac))
    return {
        "test":  dirs[:n_test],
        "val":   dirs[n_test : n_test + n_val],
        "train": dirs[n_test + n_val :],
    }


# ---------------------------------------------------------------------------
# MONAI transforms
# ---------------------------------------------------------------------------

def _build_transforms(
    split: str,
    spatial_size: Tuple[int, int, int],
    brats_version: str,
) -> Compose:
    image_keys = ["t1c", "t1n", "t2w", "t2f"]
    all_keys   = image_keys + ["label"]

    base = [
        LoadImaged(keys=all_keys),
        EnsureChannelFirstd(keys=all_keys),
    ]

    if brats_version == "2024":
        base.append(RemapBraTS2024Labels(keys=["label"]))

    base += [
        NormalizeIntensityd(
            keys=image_keys, nonzero=True, channel_wise=True
        ),
        CropForegroundd(
            keys=all_keys, source_key="label", margin=10
        ),
        SpatialPadd(
            keys=all_keys, spatial_size=list(spatial_size)
        ),
        RandCropByPosNegLabeld(
            keys=all_keys,
            label_key="label",
            spatial_size=list(spatial_size),
            pos=1,
            neg=1,
            num_samples=1,
            image_key="t1c",
            image_threshold=0,
        ),
        EnsureTyped(keys=image_keys, dtype=torch.float32),
        EnsureTyped(keys=["label"],  dtype=torch.long),
    ]

    if split == "train":
        base += [
            RandFlipd(keys=all_keys, prob=0.5, spatial_axis=0),
            RandFlipd(keys=all_keys, prob=0.5, spatial_axis=1),
            RandFlipd(keys=all_keys, prob=0.5, spatial_axis=2),
            RandRotate90d(keys=all_keys, prob=0.5, max_k=3),
            RandScaleIntensityd(keys=image_keys, factors=0.1, prob=0.5),
            RandShiftIntensityd(keys=image_keys, offsets=0.1, prob=0.5),
            RandGaussianNoised(keys=image_keys, prob=0.1, std=0.01),
        ]

    return Compose(base)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class BraTSDataset(Dataset):
    """
    BraTS segmentation dataset backed by MONAI CacheDataset.

    Returns per __getitem__:
        {
            'image':      Tensor [4, 128, 128, 128] float32
            'label':      Tensor [128, 128, 128]    int64
            'patient_id': str
        }
    """

    def __init__(
        self,
        patient_dirs: List[Path],
        split: str,
        spatial_size: Tuple[int, int, int] = (128, 128, 128),
        brats_version: str = "2024",
        cache_rate: float = 0.0,
        num_workers: int = 0,
    ) -> None:
        self.patient_dirs = patient_dirs
        self.brats_version = brats_version
        transforms = _build_transforms(split, spatial_size, brats_version)

        data_dicts = []
        for p in patient_dirs:
            files = _modality_filenames(p.name, brats_version)
            entry = {k: str(p / v) for k, v in files.items()}
            entry["patient_id"] = p.name
            data_dicts.append(entry)

        self._cache_dataset = CacheDataset(
            data=data_dicts,
            transform=transforms,
            cache_rate=cache_rate,
            num_workers=num_workers,
        )

    def __len__(self) -> int:
        return len(self._cache_dataset)

    def __getitem__(self, idx: int) -> Dict:
        result = self._cache_dataset[idx]
        # RandCropByPosNegLabeld with num_samples=1 wraps output in a list
        item = result[0] if isinstance(result, list) else result
        # Concatenate 4 modality channels into single image tensor
        image = torch.cat(
            [item["t1c"], item["t1n"], item["t2w"], item["t2f"]], dim=0
        )  # [4, H, W, D]
        label = item["label"].squeeze(0).long()  # [H, W, D]
        return {
            "image":      image,
            "label":      label,
            "patient_id": item["patient_id"],
        }


# ---------------------------------------------------------------------------
# Dataloader factory
# ---------------------------------------------------------------------------

def get_dataloader(
    data_root: str,
    split: str,
    batch_size: int = 2,
    num_workers: int = 4,
    spatial_size: Tuple[int, int, int] = (128, 128, 128),
    brats_version: str = "2024",
    val_fraction: float = 0.125,
    test_fraction: float = 0.125,
    seed: int = 42,
    cache_rate: float = 0.0,
) -> DataLoader:
    """
    Build a DataLoader for the requested split.

    Parameters
    ----------
    data_root    : path to BraTS root directory (one sub-dir per patient)
    split        : "train", "val", or "test"
    batch_size   : samples per batch
    num_workers  : DataLoader workers
    spatial_size : crop/pad target (H, W, D)
    brats_version: "2024" or "2021"
    val_fraction : fraction of patients for validation
    test_fraction: fraction of patients for test
    seed         : random seed for split reproducibility
    cache_rate   : MONAI CacheDataset rate (0.0 = no caching)

    Returns
    -------
    torch.utils.data.DataLoader
    """
    patient_dirs = get_patient_dirs(data_root)
    splits = split_patients(patient_dirs, val_fraction, test_fraction, seed)
    dataset = BraTSDataset(
        patient_dirs=splits[split],
        split=split,
        spatial_size=spatial_size,
        brats_version=brats_version,
        cache_rate=cache_rate,
        num_workers=num_workers,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == "train"),
        num_workers=num_workers,
        pin_memory=True,
        drop_last=(split == "train"),
        persistent_workers=(num_workers > 0),
        prefetch_factor=4 if num_workers > 0 else None,
    )
