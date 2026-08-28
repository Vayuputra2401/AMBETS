"""
env_config.py — Environment-specific path overrides for CCR-Net.

Loads configs/env/{env}.yaml and applies path overrides to a Phase2Config.
The base YAML (brats_phase2.yaml) holds all hyperparameters and is never
modified per-environment. The env YAML holds only paths.

Usage in train.py / evaluate.py:
    config = Phase2Config.from_yaml(config_path)
    apply_env(config, env="gcp", config_dir=Path(config_path).parent)
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import yaml


def _expand(path: str) -> str:
    """Expand ``~`` and ``$VAR``/``${VAR}`` so env paths need no hardcoded username.

    Lets configs use portable paths like ``~/data/training_data1_v2`` that resolve to
    the current user's home on any instance (the VM username is not stable across
    machines).
    """
    return os.path.expanduser(os.path.expandvars(path))


def apply_env(config, env: str, config_dir: Optional[Path] = None) -> dict:
    """
    Load configs/env/{env}.yaml and override paths in config in-place.

    Parameters
    ----------
    config     : Phase2Config  — modified in-place
    env        : "local" | "gcp" | any name matching a file in configs/env/
    config_dir : directory that contains the env/ subfolder
                 (defaults to the configs/ dir adjacent to this file's package root)

    Returns
    -------
    dict  — the full parsed env YAML (useful for printing GCP metadata)
    """
    if config_dir is None:
        # src/ccrnet/config/ → ../../configs/
        config_dir = Path(__file__).parent.parent.parent.parent / "configs"

    env_path = Path(config_dir) / "env" / f"{env}.yaml"
    if not env_path.exists():
        raise FileNotFoundError(
            f"Environment config not found: {env_path}\n"
            f"Available envs: {[p.stem for p in env_path.parent.glob('*.yaml')]}"
        )

    with open(env_path) as f:
        env_cfg = yaml.safe_load(f) or {}

    paths = env_cfg.get("paths", {})
    if paths.get("data_root"):
        config.data.data_root = _expand(paths["data_root"])
    if paths.get("brats_official_val_root"):
        config.data.brats_official_val_root = _expand(paths["brats_official_val_root"])
    if paths.get("checkpoint_dir"):
        config.training.checkpoint_dir = _expand(paths["checkpoint_dir"])
    if paths.get("pretrained_path"):
        config.swin.pretrained_path = _expand(paths["pretrained_path"])

    return env_cfg
