"""
evaluate.py — CCR-Net Phase 2 evaluation entry point.

Usage
-----
    python pipeline/evaluate.py --checkpoint checkpoints/epoch_0080.pth --split val
    python pipeline/evaluate.py --checkpoint checkpoints/epoch_0080.pth --split test

Outputs
-------
  - Per-patient Dice WT/TC/ET and HD95 (console + CSV)
  - Routing probability maps saved as NIfTI for 5 sample cases
  - Summary: mean ± std Dice for WT, TC, ET

CLI args
--------
    --checkpoint  path to .pth checkpoint (required)
    --config      path to YAML config (default: configs/brats_phase2.yaml)
    --split       "val" or "test"
    --data_root   override data root from YAML
    --device      cuda / cpu
    --save_dir    where to write per-patient NIfTIs (default: evals/<split>/)
    --n_save      number of cases to save routing maps for (default: 5)
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent / "data"))

from ccrnet.config.phase2_config import Phase2Config
from ccrnet.config.env_config import apply_env
from ccrnet.models.ccrnet import CCRNet
from ccrnet.utils.token_labels import downsample_labels_to_tokens
from ccrnet.utils.checkpoint import load_checkpoint
from brats_dataset import get_dataloader

from ccr.utils.metrics import ConceptAlignmentScore


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

def _dice(pred_mask: torch.Tensor, gt_mask: torch.Tensor, smooth: float = 1e-6) -> float:
    p = pred_mask.float().reshape(-1)
    g = gt_mask.float().reshape(-1)
    return ((2.0 * (p * g).sum() + smooth) / (p.sum() + g.sum() + smooth)).item()


def dice_wt(pred, label): return _dice(pred > 0, label > 0)
def dice_tc(pred, label): return _dice(pred.eq(1) | pred.eq(3), label.eq(1) | label.eq(3))
def dice_et(pred, label): return _dice(pred.eq(3), label.eq(3))


def _hd95(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
    """95th-percentile Hausdorff distance. Returns inf if either mask is empty."""
    try:
        from scipy.ndimage import distance_transform_edt
        if pred_mask.sum() == 0 or gt_mask.sum() == 0:
            return float("inf")
        d1 = distance_transform_edt(~gt_mask).flatten()[pred_mask.flatten()]
        d2 = distance_transform_edt(~pred_mask).flatten()[gt_mask.flatten()]
        return float(np.percentile(np.concatenate([d1, d2]), 95))
    except Exception:
        return float("nan")


# ---------------------------------------------------------------------------
# Routing map saver
# ---------------------------------------------------------------------------

def _save_routing_nifti(
    routing_probs: torch.Tensor,  # [1, N, K]
    grid_shape: tuple,
    patient_id: str,
    save_dir: str,
    concept_names: tuple,
) -> None:
    try:
        import nibabel as nib
    except ImportError:
        return

    os.makedirs(save_dir, exist_ok=True)
    B, N, K = routing_probs.shape
    h, w, d = grid_shape
    probs_spatial = routing_probs[0].T.reshape(K, h, w, d).cpu().numpy()  # [K, h, w, d]

    for k, name in enumerate(concept_names):
        arr = probs_spatial[k]
        img = nib.Nifti1Image(arr.astype(np.float32), affine=np.eye(4))
        fname = os.path.join(save_dir, f"{patient_id}_routing_{name}.nii.gz")
        nib.save(img, fname)


# ---------------------------------------------------------------------------
# Evaluation loop
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(
    model: CCRNet,
    data_loader,
    cas: ConceptAlignmentScore,
    device: torch.device,
    save_dir: str,
    n_save: int,
    concept_names: tuple,
) -> List[Dict]:
    model.eval()
    cas.reset()
    results = []
    saved = 0

    for batch in tqdm(data_loader, desc="  evaluating"):
        image = batch["image"].to(device, non_blocking=True)
        label = batch["label"].to(device, non_blocking=True)
        pid   = batch["patient_id"][0] if isinstance(batch["patient_id"], list) else batch["patient_id"]

        out          = model(image)
        token_labels = downsample_labels_to_tokens(label, model.get_grid_shape())
        cas.update(out["routing_probs"], token_labels)

        pred = out["seg_logits"].argmax(dim=1)  # [B, H, W, D]
        p_np = pred[0].cpu().numpy().astype(bool)
        l_np = label[0].cpu().numpy()

        row = {
            "patient_id": pid,
            "dice_wt":    dice_wt(pred[0], label[0]),
            "dice_tc":    dice_tc(pred[0], label[0]),
            "dice_et":    dice_et(pred[0], label[0]),
            "hd95_wt":    _hd95(p_np > 0, l_np > 0),
            "hd95_tc":    _hd95((p_np == 1) | (p_np == 3), (l_np == 1) | (l_np == 3)),
            "hd95_et":    _hd95(p_np == 3, l_np == 3),
        }
        results.append(row)

        if saved < n_save:
            _save_routing_nifti(
                out["routing_probs"], model.get_grid_shape(),
                pid, save_dir, concept_names,
            )
            saved += 1

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    _default_config = str(Path(__file__).parent.parent / "configs" / "brats_phase2.yaml")

    parser = argparse.ArgumentParser(description="CCR-Net Phase 2 evaluation")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to .pth checkpoint")
    parser.add_argument("--env",        type=str, default="local",
                        help="Environment: 'local' or 'gcp' (loads configs/env/{env}.yaml)")
    parser.add_argument("--config",     type=str, default=_default_config,
                        help="Path to base YAML config")
    parser.add_argument("--split",      type=str, default="val", choices=["val", "test"])
    parser.add_argument("--save_dir",   type=str, default="",
                        help="Output dir for NIfTIs/CSV (default: {checkpoint_dir}/evals/{split})")
    parser.add_argument("--device",     type=str, default="")
    parser.add_argument("--n_save",     type=int, default=5)
    args = parser.parse_args()

    config = Phase2Config.from_yaml(args.config)
    apply_env(config, env=args.env, config_dir=Path(args.config).parent)

    device = torch.device(
        args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    save_dir = args.save_dir or str(
        Path(config.training.checkpoint_dir) / "evals" / args.split
    )

    model = CCRNet(config).to(device)
    start_epoch, metrics = load_checkpoint(args.checkpoint, model)
    print(f"Loaded checkpoint (epoch {start_epoch - 1})")

    loader = get_dataloader(
        data_root    = config.data.data_root,
        split        = args.split,
        batch_size   = 1,
        num_workers  = config.data.num_workers,
        spatial_size = config.data.spatial_size,
        brats_version= config.data.brats_version,
        val_fraction = config.data.val_fraction,
        test_fraction= config.data.test_fraction,
        seed         = config.data.seed,
    )

    concept_names = ("background", "necrotic_core", "edema", "enhancing_tumor")
    cas = ConceptAlignmentScore(config.ccr.router.num_concepts, concept_names)

    print(f"Evaluating {len(loader.dataset)} patients on {args.split} split ...")
    results = evaluate(model, loader, cas, device, save_dir, args.n_save, concept_names)

    # Print per-patient results
    for r in results:
        print(
            f"{r['patient_id']}: "
            f"Dice WT={r['dice_wt']:.3f} TC={r['dice_tc']:.3f} ET={r['dice_et']:.3f} | "
            f"HD95 WT={r['hd95_wt']:.1f} TC={r['hd95_tc']:.1f} ET={r['hd95_et']:.1f}"
        )

    # Summary statistics
    for metric in ["dice_wt", "dice_tc", "dice_et"]:
        vals = [r[metric] for r in results if not np.isnan(r[metric])]
        label_short = metric.replace("dice_", "").upper()
        print(f"Mean Dice {label_short}: {np.mean(vals):.3f} ± {np.std(vals):.3f}")

    cas_fg = cas.compute_fg()
    print(f"CAS_fg: {cas_fg}")

    # Save CSV
    os.makedirs(save_dir, exist_ok=True)
    csv_path = os.path.join(save_dir, f"{args.split}_results.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)
    print(f"Results saved to {csv_path}")
    print(f"Routing NIfTIs (first {args.n_save} cases) saved to {save_dir}/")


if __name__ == "__main__":
    main()
