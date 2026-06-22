"""
evaluate.py — CCR-Net Phase 2 evaluation entry point.

Usage
-----
    # Standard eval (Dice, HD95, CAS, AUROC, ECE):
    python pipeline/evaluate.py --checkpoint checkpoints/epoch_0080.pth --split val

    # Full faithfulness eval (adds deletion AUC, insertion AUC — expensive: ~19 passes/patient):
    python pipeline/evaluate.py --checkpoint checkpoints/epoch_0080.pth --split val --full

    # With MC-Dropout uncertainty comparison (T=10):
    python pipeline/evaluate.py --checkpoint ... --split val --mc_passes 10

Outputs
-------
  - Per-patient Dice WT/TC/ET, HD95, AUROC per concept, ECE
  - Summary JSON: all metrics mean ± std
  - Routing probability NIfTI maps (first --n_save cases)
  - CSV with per-patient rows

CLI args
--------
    --checkpoint  path to .pth checkpoint (required)
    --config      path to YAML config (default: configs/brats_phase2.yaml)
    --env         local | gcp
    --split       val | test
    --save_dir    output dir (default: {checkpoint_dir}/evals/{split})
    --device      cuda | cpu
    --n_save      number of cases to save routing NIfTIs (default: 5)
    --full        enable deletion AUC and insertion AUC (expensive)
    --mc_passes   MC-Dropout passes for uncertainty ECE comparison (default: 0 = disabled)
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

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
from ccrnet.utils.evaluation import (
    compute_auroc_per_concept,
    compute_deletion_auc,
    compute_insertion_auc,
    compute_ece,
    mc_dropout_entropy,
)
from brats_dataset import get_dataloader

from ccr.utils.metrics import ConceptAlignmentScore


# ---------------------------------------------------------------------------
# Segmentation metric helpers
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
    full_eval: bool = False,
    mc_passes: int = 0,
) -> Dict:
    """
    Run evaluation over the full split.

    Returns a dict with keys:
      per_patient : List[Dict] — per-patient metrics (Dice, HD95, ECE, deletion AUC if --full)
      routing_probs_all : [total_N, K] tensor — for global AUROC computation
      token_labels_all  : [total_N] tensor — for global AUROC computation
      mc_ece            : float or None — MC-Dropout ECE (if mc_passes > 0)
    """
    model.eval()
    cas.reset()
    per_patient: List[Dict] = []
    saved = 0

    # Accumulated for global AUROC (one shot at end rather than per-patient)
    all_routing_probs: List[torch.Tensor] = []
    all_token_labels: List[torch.Tensor]  = []

    # For global ECE accumulation
    all_entropy: List[torch.Tensor] = []
    all_ece_labels: List[torch.Tensor] = []

    # MC-Dropout entropy accumulation
    all_mc_entropy: List[torch.Tensor] = []
    all_mc_labels: List[torch.Tensor] = []

    for batch in tqdm(data_loader, desc="  evaluating"):
        image = batch["image"].to(device, non_blocking=True)
        label = batch["label"].to(device, non_blocking=True)
        pid   = batch["patient_id"][0] if isinstance(batch["patient_id"], list) else batch["patient_id"]

        out          = model(image)
        grid_shape   = model.get_grid_shape()
        token_labels = downsample_labels_to_tokens(label, grid_shape)

        cas.update(out["routing_probs"], token_labels)

        # Accumulate for global metrics
        all_routing_probs.append(out["routing_probs"].cpu())  # [1, N, K]
        all_token_labels.append(token_labels.cpu())           # [1, N]
        all_entropy.append(out["entropy"].cpu())              # [1, N]
        all_ece_labels.append(token_labels.cpu())

        # Segmentation metrics
        pred   = out["seg_logits"].argmax(dim=1)  # [B, H, W, D]
        p_np   = pred[0].cpu().numpy().astype(bool)
        l_np   = label[0].cpu().numpy()

        row: Dict = {
            "patient_id": pid,
            "dice_wt":    dice_wt(pred[0], label[0]),
            "dice_tc":    dice_tc(pred[0], label[0]),
            "dice_et":    dice_et(pred[0], label[0]),
            "hd95_wt":    _hd95(p_np > 0, l_np > 0),
            "hd95_tc":    _hd95((p_np == 1) | (p_np == 3), (l_np == 1) | (l_np == 3)),
            "hd95_et":    _hd95(p_np == 3, l_np == 3),
        }

        # Per-patient ECE
        row["ece"] = compute_ece(
            out["routing_probs"], token_labels, out["entropy"]
        )

        # Expensive faithfulness metrics (only if --full)
        if full_eval:
            del_auc = compute_deletion_auc(
                model, image, label, out["routing_probs"],
                grid_shape, device, concept_names,
            )
            ins_auc = compute_insertion_auc(
                model, image, out["routing_probs"],
                grid_shape, device, concept_names,
            )
            for name, val in del_auc.items():
                row[f"del_auc_{name}"] = val
            for name, val in ins_auc.items():
                row[f"ins_auc_{name}"] = val

        # MC-Dropout uncertainty comparison
        if mc_passes > 0:
            mc_ent = mc_dropout_entropy(model, image, n_passes=mc_passes)  # [N]
            all_mc_entropy.append(mc_ent.cpu())
            all_mc_labels.append(token_labels[0].cpu())

        per_patient.append(row)

        if saved < n_save:
            _save_routing_nifti(
                out["routing_probs"], grid_shape,
                pid, save_dir, concept_names,
            )
            saved += 1

    # Global AUROC (all patients combined)
    routing_probs_all = torch.cat(all_routing_probs, dim=0)   # [total_B, N, K]
    token_labels_all  = torch.cat(all_token_labels, dim=0)    # [total_B, N]
    entropy_all       = torch.cat(all_entropy, dim=0)

    # Global ECE (all patients combined)
    global_ece = compute_ece(routing_probs_all, token_labels_all, entropy_all)

    # MC-Dropout ECE (if requested)
    mc_ece: Optional[float] = None
    if mc_passes > 0 and all_mc_entropy:
        mc_ent_all = torch.stack(all_mc_entropy, dim=0)  # [total_B, N]
        mc_lab_all = torch.stack(all_mc_labels, dim=0)   # [total_B, N]

        # For MC-Dropout ECE, treat entropy as "uncertainty" and compute calibration
        # (compare against routing ECE for Prop 2 validation)
        import math
        K = routing_probs_all.shape[-1]
        max_ent = math.log(K)
        mc_conf = 1.0 - (mc_ent_all / max_ent).clamp(0.0, 1.0)
        rout_probs_for_mc = routing_probs_all  # use same routing probs for accuracy reference
        assignments_all = rout_probs_for_mc.reshape(-1, K).argmax(dim=-1)
        labels_flat = token_labels_all.reshape(-1).long()
        acc_all = (assignments_all == labels_flat).float()
        conf_flat = mc_conf.reshape(-1)
        n = conf_flat.numel()
        ece_mc = 0.0
        bin_edges = torch.linspace(0.0, 1.0, 16)
        for i in range(15):
            lo, hi = bin_edges[i].item(), bin_edges[i+1].item()
            mask = (conf_flat >= lo) & (conf_flat < hi if i < 14 else conf_flat <= hi)
            if mask.sum() == 0:
                continue
            ece_mc += (mask.sum().item() / n) * abs(conf_flat[mask].mean().item() - acc_all[mask].mean().item())
        mc_ece = float(ece_mc)

    return {
        "per_patient":       per_patient,
        "routing_probs_all": routing_probs_all,
        "token_labels_all":  token_labels_all,
        "global_ece":        global_ece,
        "mc_ece":            mc_ece,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    _default_config = str(Path(__file__).parent.parent / "configs" / "brats_phase2.yaml")

    parser = argparse.ArgumentParser(description="CCR-Net Phase 2 evaluation")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--env",        type=str, default="local")
    parser.add_argument("--config",     type=str, default=_default_config)
    parser.add_argument("--split",      type=str, default="val", choices=["val", "test"])
    parser.add_argument("--save_dir",   type=str, default="")
    parser.add_argument("--device",     type=str, default="")
    parser.add_argument("--n_save",     type=int, default=5)
    parser.add_argument("--full",       action="store_true",
                        help="Enable deletion/insertion AUC (~19 forward passes/patient)")
    parser.add_argument("--mc_passes",  type=int, default=0,
                        help="MC-Dropout passes for uncertainty ECE comparison (requires dropout in model)")
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
    start_epoch, _ = load_checkpoint(args.checkpoint, model)
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

    concept_names = config.ccr.concept_names
    cas = ConceptAlignmentScore(config.ccr.router.num_concepts, concept_names)

    if args.full:
        print(f"Full eval enabled (deletion + insertion AUC). Expected ~{19 * len(loader.dataset)} forward passes.")

    print(f"Evaluating {len(loader.dataset)} patients on {args.split} split ...")
    t0 = time.time()

    eval_out = evaluate(
        model, loader, cas, device, save_dir,
        args.n_save, concept_names,
        full_eval=args.full,
        mc_passes=args.mc_passes,
    )

    results = eval_out["per_patient"]
    elapsed = time.time() - t0
    print(f"Eval done in {elapsed:.1f}s")

    # --- Segmentation metrics ---
    print("\n=== Segmentation ===")
    for r in results:
        print(
            f"{r['patient_id']}: "
            f"Dice WT={r['dice_wt']:.3f} TC={r['dice_tc']:.3f} ET={r['dice_et']:.3f} | "
            f"HD95 WT={r['hd95_wt']:.1f} TC={r['hd95_tc']:.1f} ET={r['hd95_et']:.1f}"
        )

    summary: Dict = {}
    for metric in ["dice_wt", "dice_tc", "dice_et"]:
        vals = [r[metric] for r in results if not np.isnan(r[metric])]
        label_short = metric.replace("dice_", "").upper()
        mean_v, std_v = np.mean(vals), np.std(vals)
        print(f"Mean Dice {label_short}: {mean_v:.3f} ± {std_v:.3f}")
        summary[metric] = {"mean": float(mean_v), "std": float(std_v)}

    # --- CAS ---
    print("\n=== CAS_fg ===")
    cas_fg = cas.compute_fg()
    for k, v in cas_fg.items():
        print(f"  {k}: {v:.4f}")
        summary[f"cas_fg_{k}"] = float(v)

    # --- AUROC per concept (global) ---
    print("\n=== AUROC per concept ===")
    auroc = compute_auroc_per_concept(
        eval_out["routing_probs_all"],
        eval_out["token_labels_all"],
        concept_names,
    )
    for name, val in auroc.items():
        print(f"  {name}: {val:.4f}")
        summary[f"auroc_{name}"] = float(val)

    # --- ECE ---
    print("\n=== Calibration ===")
    global_ece = eval_out["global_ece"]
    print(f"  Routing ECE: {global_ece:.4f}")
    summary["routing_ece"] = float(global_ece)

    if eval_out["mc_ece"] is not None:
        print(f"  MC-Dropout ECE (T={args.mc_passes}): {eval_out['mc_ece']:.4f}")
        summary[f"mc_dropout_ece_T{args.mc_passes}"] = float(eval_out["mc_ece"])

    # --- Deletion / Insertion AUC ---
    if args.full:
        print("\n=== Faithfulness (Deletion AUC) ===")
        del_keys = [k for k in results[0] if k.startswith("del_auc_")]
        for dk in del_keys:
            concept = dk.replace("del_auc_", "")
            vals = [r[dk] for r in results if not np.isnan(r[dk])]
            mean_v = np.mean(vals)
            print(f"  {concept}: {mean_v:.4f}")
            summary[dk] = {"mean": float(mean_v), "std": float(np.std(vals))}

        print("\n=== Faithfulness (Insertion AUC) ===")
        ins_keys = [k for k in results[0] if k.startswith("ins_auc_")]
        for ik in ins_keys:
            concept = ik.replace("ins_auc_", "")
            vals = [r[ik] for r in results if not np.isnan(r[ik])]
            mean_v = np.mean(vals)
            print(f"  {concept}: {mean_v:.4f}")
            summary[ik] = {"mean": float(mean_v), "std": float(np.std(vals))}

    # --- Save CSV ---
    os.makedirs(save_dir, exist_ok=True)
    csv_path = os.path.join(save_dir, f"{args.split}_results.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)
    print(f"\nPer-patient CSV: {csv_path}")

    # --- Save JSON summary ---
    summary["n_patients"] = len(results)
    summary["checkpoint"] = args.checkpoint
    summary["split"] = args.split
    summary["elapsed_s"] = float(elapsed)

    json_path = os.path.join(save_dir, f"{args.split}_summary.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary JSON:    {json_path}")
    print(f"Routing NIfTIs (first {args.n_save} cases): {save_dir}/")


if __name__ == "__main__":
    main()
