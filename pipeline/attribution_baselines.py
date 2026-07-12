"""
attribution_baselines.py — Post-hoc attribution baselines for Figure 3 (WACV 2027).

Computes deletion AUC + insertion AUC for post-hoc explanation methods on the
SAME CCR-Net model and SAME test cases as `evaluate.py --full`, so the numbers
are directly comparable. This is the "other half" of Figure 3: CCR routing
(from evaluate.py --full) vs the post-hoc baselines computed here.

Methods (spanning 3 attribution families — answers the "all gradient-based" critique)
-------
  gradcam   : Grad-CAM at the post-CCR bottleneck block (decoder.enc3, 16³, 192-dim)
              — the same location the routing explanation lives, for a fair contrast.
  gradient  : Vanilla gradient saliency — |∂ mean(seg_logits[:,k]) / ∂ input|,
              max over the 4 MRI channels → 128³ map.
  ig        : Integrated Gradients — path-integrated, axiomatic (completeness/
              sensitivity); Σ|IG| over channels → 128³ map.
  occlusion : Coarse 3D occlusion — perturbation family (no gradients): zero each of
              grid³ cells, saliency = drop in predicted-k confidence.

All explanation maps are scored with the IDENTICAL deletion/insertion protocol used
for CCR (evaluation.deletion_auc_from_map / insertion_auc_from_map), under BOTH
measurement windows — region_mode="gt" (GT==k, primary) and region_mode="pred"
(predicted==k, removes the GT-supervision confound). Output keys are
{method}_{del|ins}_{gt|pred}_{concept}.

Usage
-----
    # Full test set (n=0), all four methods, both region modes → resolves NCR n and
    # the confound in one pass. This is the W3/W6 run.
    python pipeline/attribution_baselines.py \
        --checkpoint /path/epoch_0080.pth --env gcp --split test --n_cases 0

    # Cheaper subset / single method:
    python pipeline/attribution_baselines.py \
        --checkpoint ... --env gcp --split test --n_cases 40 --methods gradcam,ig

CLI args
--------
    --checkpoint   path to .pth checkpoint (required)
    --env          local | gcp
    --config       path to YAML config (default: configs/brats_phase2.yaml)
    --split        val | test
    --methods      comma list from {gradcam,gradient,ig,occlusion} (default all four)
    --n_cases      number of cases (default 40; 0 = all)
    --occ_grid     occlusion grid per axis (default 4 → 4³=64 passes/concept)
    --ig_steps     Integrated-Gradients Riemann steps (default 32)
    --save_dir     output dir (default: {checkpoint_dir}/evals/{split}_baselines)
    --device       cuda | cpu
    --no_sync_gcs  disable automatic GCS sync
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent / "data"))

from ccrnet.config.phase2_config import Phase2Config
from ccrnet.config.env_config import apply_env
from ccrnet.models.ccrnet import CCRNet
from ccrnet.utils.checkpoint import load_checkpoint
from ccrnet.utils.evaluation import (
    deletion_auc_from_map,
    insertion_auc_from_map,
    _DEFAULT_THRESHOLDS,
)
from brats_dataset import get_dataloader


# Measurement windows scored for every method/concept (W2): "gt" is the primary
# protocol, "pred" removes the GT-supervision confound.
REGION_MODES = ("gt", "pred")


# ---------------------------------------------------------------------------
# Grad-CAM at the post-CCR bottleneck (decoder.enc3)
# ---------------------------------------------------------------------------

class GradCAM3D:
    """
    Grad-CAM on a target conv block. Captures the block's activation (forward)
    and its gradient (backward), then forms the class-k CAM.

    target_layer should output a [B, C, h, w, d] tensor — here decoder.enc3.
    """

    def __init__(self, model: CCRNet, target_layer: nn.Module) -> None:
        self.model = model
        self._act: torch.Tensor = None
        self._grad: torch.Tensor = None
        target_layer.register_forward_hook(self._forward_hook)

    def _forward_hook(self, module, inputs, output) -> None:
        self._act = output
        # The hook fires on EVERY forward, including the no_grad baseline/deletion/
        # insertion passes where `output` doesn't require grad. Only attach the
        # gradient hook when grad is enabled (i.e. during cam()'s own forward).
        if output.requires_grad:
            output.register_hook(self._save_grad)

    def _save_grad(self, grad: torch.Tensor) -> None:
        self._grad = grad

    def cam(self, image: torch.Tensor, k: int, target_shape) -> torch.Tensor:
        """Returns the class-k CAM upsampled to target_shape: [H, W, D]."""
        self.model.zero_grad(set_to_none=True)
        out = self.model(image)
        score = out["seg_logits"][0, k].mean()
        score.backward()

        # channel weights = global-average-pooled gradients
        weights = self._grad.mean(dim=(2, 3, 4), keepdim=True)       # [1, C, 1, 1, 1]
        cam = torch.relu((weights * self._act).sum(dim=1, keepdim=True))  # [1, 1, h, w, d]
        cam = F.interpolate(cam, size=target_shape, mode="trilinear", align_corners=False)
        return cam[0, 0].detach()                                     # [H, W, D]


# ---------------------------------------------------------------------------
# Vanilla gradient saliency
# ---------------------------------------------------------------------------

def gradient_saliency(model: CCRNet, image: torch.Tensor, k: int) -> torch.Tensor:
    """|∂ mean(seg_logits[:,k]) / ∂ input|, max over channels → [H, W, D]."""
    model.zero_grad(set_to_none=True)
    img = image.clone().detach().requires_grad_(True)
    out = model(img)
    score = out["seg_logits"][0, k].mean()
    score.backward()
    sal = img.grad[0].abs().amax(dim=0)   # [H, W, D] — max over 4 MRI channels
    return sal.detach()


# ---------------------------------------------------------------------------
# Integrated Gradients (axiomatic, path-integrated — a different sub-family)
# ---------------------------------------------------------------------------

def integrated_gradients(
    model: CCRNet,
    image: torch.Tensor,
    k: int,
    steps: int = 32,
    baseline: torch.Tensor = None,
) -> torch.Tensor:
    """
    Integrated Gradients for concept k → [H, W, D] (Σ_channels |IG|).

    IG_i = (x_i − x'_i) · ∫₀¹ ∂F_k/∂x_i(x' + α(x−x')) dα, approximated by a Riemann
    sum over `steps` interior points. F_k = mean(seg_logits[:,k]); baseline x' = 0.
    Unlike vanilla gradient saliency this satisfies completeness/sensitivity axioms —
    added specifically to answer "both baselines are gradient-family" with a
    stronger gradient method; Occlusion adds the perturbation family.
    """
    if baseline is None:
        baseline = torch.zeros_like(image)
    total_grad = torch.zeros_like(image)
    for s in range(1, steps + 1):
        alpha = s / steps
        interp = (baseline + alpha * (image - baseline)).detach().requires_grad_(True)
        model.zero_grad(set_to_none=True)
        score = model(interp)["seg_logits"][0, k].mean()
        grad, = torch.autograd.grad(score, interp)
        total_grad = total_grad + grad.detach()
    avg_grad = total_grad / steps
    ig = (image - baseline) * avg_grad          # [1, C, H, W, D]
    return ig[0].abs().sum(dim=0).detach()      # [H, W, D] — Σ over MRI channels


# ---------------------------------------------------------------------------
# Occlusion saliency (perturbation family — no gradients)
# ---------------------------------------------------------------------------

def occlusion_saliency(
    model: CCRNet,
    image: torch.Tensor,
    k: int,
    baseline_probs: torch.Tensor,
    grid: int = 4,
    baseline_fill: float = 0.0,
) -> torch.Tensor:
    """
    Coarse 3D occlusion saliency for concept k → [H, W, D].

    Partition the volume into grid³ cells; zero each cell, re-run, and set the cell's
    saliency to the drop in mean class-k confidence over the model's predicted-k
    region (GT-free). Perturbation-family attribution — the non-gradient baseline that
    directly rebuts the "all baselines are gradient-based" critique. The coarse grid
    bounds the cost to grid³ forward passes per concept (default 4³ = 64).
    """
    _, C, H, W, D = image.shape
    region = (baseline_probs[0].argmax(dim=0) == k)      # predicted-k region
    sal = torch.zeros(H, W, D, device=image.device)
    if region.sum() == 0:
        return sal
    base = baseline_probs[0, k][region].mean().item()

    def _bounds(n: int):
        edges = [int(round(i * n / grid)) for i in range(grid + 1)]
        return [(edges[i], edges[i + 1]) for i in range(grid)]

    with torch.no_grad():
        for (z0, z1) in _bounds(H):
            for (y0, y1) in _bounds(W):
                for (x0, x1) in _bounds(D):
                    occ = image.clone()
                    occ[:, :, z0:z1, y0:y1, x0:x1] = baseline_fill
                    probs = torch.softmax(model(occ)["seg_logits"], dim=1)
                    score = probs[0, k][region].mean().item()
                    sal[z0:z1, y0:y1, x0:x1] = base - score   # confidence drop
    return sal.clamp(min=0.0)   # importance = positive contribution to concept k


# ---------------------------------------------------------------------------
# Per-method explanation volume for all concepts
# ---------------------------------------------------------------------------

def explanation_maps(
    method: str,
    model: CCRNet,
    gradcam: GradCAM3D,
    image: torch.Tensor,
    concept_indices,
    target_shape,
    baseline_probs: torch.Tensor = None,
    occ_grid: int = 4,
    ig_steps: int = 32,
) -> Dict[int, torch.Tensor]:
    """Returns {k: expl_k [H,W,D]} for the given method."""
    maps: Dict[int, torch.Tensor] = {}
    for k in concept_indices:
        if method == "gradcam":
            maps[k] = gradcam.cam(image, k, target_shape)
        elif method == "gradient":
            maps[k] = gradient_saliency(model, image, k)
        elif method == "ig":
            maps[k] = integrated_gradients(model, image, k, steps=ig_steps)
        elif method == "occlusion":
            maps[k] = occlusion_saliency(model, image, k, baseline_probs, grid=occ_grid)
        else:
            raise ValueError(f"unknown method: {method}")
    return maps


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    _default_config = str(Path(__file__).parent.parent / "configs" / "brats_phase2.yaml")

    parser = argparse.ArgumentParser(description="Post-hoc attribution baselines (Figure 3)")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--env",        type=str, default="local")
    parser.add_argument("--config",     type=str, default=_default_config)
    parser.add_argument("--split",      type=str, default="test", choices=["val", "test"])
    parser.add_argument("--methods",    type=str, default="gradcam,gradient,ig,occlusion",
                        help="comma list from {gradcam,gradient,ig,occlusion}")
    parser.add_argument("--n_cases",    type=int, default=40,
                        help="Number of cases (deterministic first-N of split). 0 = all.")
    parser.add_argument("--occ_grid",   type=int, default=4,
                        help="Occlusion grid resolution per axis (grid^3 forward passes/concept).")
    parser.add_argument("--ig_steps",   type=int, default=32,
                        help="Integrated-Gradients Riemann-sum steps.")
    parser.add_argument("--save_dir",   type=str, default="")
    parser.add_argument("--device",     type=str, default="")
    parser.add_argument("--no_sync_gcs", action="store_true")
    args = parser.parse_args()

    methods = [m.strip() for m in args.methods.split(",") if m.strip()]

    config = Phase2Config.from_yaml(args.config)
    env_cfg = apply_env(config, env=args.env,
                        config_dir=Path(__file__).parent.parent / "configs")

    device = torch.device(
        args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    save_dir = args.save_dir or str(
        Path(config.training.checkpoint_dir) / "evals" / f"{args.split}_baselines"
    )
    os.makedirs(save_dir, exist_ok=True)

    model = CCRNet(config).to(device)
    start_epoch, _ = load_checkpoint(args.checkpoint, model)
    model.eval()
    print(f"Loaded checkpoint (epoch {start_epoch - 1}) on {device}")

    gradcam = GradCAM3D(model, model.decoder.enc3)   # post-CCR bottleneck block

    concept_names = config.ccr.concept_names
    concept_indices = (1, 2, 3)   # NCR, Edema, ET (skip background)

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

    n_total = len(loader.dataset)
    n_cases = n_total if args.n_cases == 0 else min(args.n_cases, n_total)
    print(f"Methods: {methods}  |  cases: {n_cases}/{n_total}  |  concepts: "
          f"{[concept_names[k] for k in concept_indices]}")
    print(f"Thresholds: {_DEFAULT_THRESHOLDS}")

    # rows[method] -> list of per-patient dicts
    per_patient: List[Dict] = []
    t0 = time.time()

    seen = 0
    for batch in tqdm(loader, total=n_cases, desc="  baselines"):
        if seen >= n_cases:
            break
        image = batch["image"].to(device)
        label = batch["label"].to(device)
        pid   = batch["patient_id"][0] if isinstance(batch["patient_id"], list) else batch["patient_id"]
        target_shape = tuple(image.shape[2:])

        # Baseline (unmasked) softmax, reused across concepts for deletion.
        with torch.no_grad():
            baseline_probs = torch.softmax(model(image)["seg_logits"], dim=1)

        row: Dict = {"patient_id": pid}
        for method in methods:
            maps = explanation_maps(
                method, model, gradcam, image, concept_indices, target_shape,
                baseline_probs=baseline_probs, occ_grid=args.occ_grid, ig_steps=args.ig_steps,
            )
            for k in concept_indices:
                name = concept_names[k]
                # Score under BOTH measurement windows: "gt" (GT==k) is the primary
                # protocol; "pred" (predicted==k) removes the GT-supervision confound
                # (W2) so CCR's routing — trained to align with GT — is compared fairly.
                for rmode in REGION_MODES:
                    d = deletion_auc_from_map(
                        model, image, label, maps[k], k,
                        baseline_probs=baseline_probs, region_mode=rmode,
                    )
                    ins = insertion_auc_from_map(
                        model, image, label, maps[k], k,
                        baseline_probs=baseline_probs, region_mode=rmode,
                    )
                    row[f"{method}_del_{rmode}_{name}"] = d
                    row[f"{method}_ins_{rmode}_{name}"] = ins
        per_patient.append(row)
        seen += 1

    elapsed = time.time() - t0
    print(f"\nDone {seen} cases in {elapsed:.1f}s")

    # --- Aggregate ---
    summary: Dict = {}
    for rmode in REGION_MODES:
        print(f"\n=== Deletion / Insertion AUC — region={rmode} (higher = more faithful) ===")
        for method in methods:
            print(f"  [{method}]")
            for k in concept_indices:
                name = concept_names[k]
                # NaN-filter: a case lacking the region (GT==k absent, or the model
                # predicts no k) returns NaN by design; plain np.mean would poison the
                # whole concept (esp. NCR, often absent). nanmean averages only the
                # cases where the concept is defined.
                dvals = np.array([r[f"{method}_del_{rmode}_{name}"] for r in per_patient], float)
                ivals = np.array([r[f"{method}_ins_{rmode}_{name}"] for r in per_patient], float)
                n_used = int(np.isfinite(dvals).sum())
                dmean, dstd = float(np.nanmean(dvals)), float(np.nanstd(dvals))
                imean, istd = float(np.nanmean(ivals)), float(np.nanstd(ivals))
                print(f"     {name:16s} deletion={dmean:.4f}±{dstd:.4f}  "
                      f"insertion={imean:.4f}  (n={n_used}/{len(per_patient)})")
                summary[f"{method}_del_{rmode}_{name}"] = {"mean": dmean, "std": dstd, "n": n_used}
                summary[f"{method}_ins_{rmode}_{name}"] = {"mean": imean, "std": istd, "n": n_used}

    # --- Save CSV + JSON ---
    csv_path = os.path.join(save_dir, f"{args.split}_baselines.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(per_patient[0].keys()))
        writer.writeheader()
        writer.writerows(per_patient)

    summary["methods"] = methods
    summary["n_cases"] = seen
    summary["checkpoint"] = args.checkpoint
    summary["split"] = args.split
    summary["thresholds"] = list(_DEFAULT_THRESHOLDS)
    summary["elapsed_s"] = float(elapsed)
    json_path = os.path.join(save_dir, f"{args.split}_baselines_summary.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nCSV:  {csv_path}")
    print(f"JSON: {json_path}")

    # --- GCS sync (rsync, no nesting) ---
    gcs_ckpt_dir = env_cfg.get("paths", {}).get("gcs_checkpoint_dir", "")
    if gcs_ckpt_dir and not args.no_sync_gcs:
        run_id  = Path(args.checkpoint).parent.name
        gcs_dst = f"{gcs_ckpt_dir.rstrip('/')}/{run_id}/evals/{args.split}_baselines"
        print(f"\nSyncing to GCS: {save_dir} → {gcs_dst}")
        try:
            subprocess.run(["gsutil", "-m", "rsync", "-r", save_dir, gcs_dst], check=True)
            print("  GCS sync complete.")
        except Exception as e:
            print(f"  [warn] GCS sync failed: {e}")


if __name__ == "__main__":
    main()
