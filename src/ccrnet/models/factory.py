"""
factory.py — build the CCR model for a given backbone.

  build_model(config, "ccrnet")    -> CCRNet         (Swin transformer encoder + UNETR decoder)
  build_model(config, "segresnet") -> CCRSegResNet   (MONAI SegResNet CNN encoder-decoder)

Both wrap the SAME CCRBottleneckModule at a $192\times16^3$ bottleneck; only the backbone
differs. Selected on the CLI via --model in train.py / evaluate.py / concept_alignment.py.
"""

from __future__ import annotations

ARCHES = ("ccrnet", "segresnet")


def build_model(config, arch: str = "ccrnet"):
    if arch == "ccrnet":
        from ccrnet.models.ccrnet import CCRNet
        return CCRNet(config)
    if arch == "segresnet":
        from ccrnet.models.ccr_segresnet import CCRSegResNet
        return CCRSegResNet(config)
    raise ValueError(f"unknown --model {arch!r}; choose from {ARCHES}")
