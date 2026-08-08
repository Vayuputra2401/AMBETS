"""
concept_alignment.py — concept-discriminability of each explanation method (W3 follow-up).

Deletion/insertion AUC rewards input *sensitivity*, which IG/Occlusion optimize directly;
they win it without producing a CONCEPT explanation. This script measures the complementary,
clinically relevant axis: does each method's per-concept map localize the *clinical concept*?

For every method (CCR routing + the four post-hoc baselines) and concept k we compute the
AUROC of the map e_k as a classifier for GT concept k, in two windows:
  - foreground  : concept k vs the OTHER tumor voxels (GT>0, GT!=k) -> concept-discriminability
                  (the metric a same-blob-for-every-class sensitivity map should fail, ~0.5)
  - all         : concept k vs everything (mostly background) -> region localization (easy)

CCR is trained to align with concepts (CAS); the hypothesis is that CCR >> IG/Occlusion on the
foreground metric. This is scored WITHOUT deletion/insertion masking, so it is far cheaper than
attribution_baselines.py (map generation only).

Usage
-----
    python pipeline/concept_alignment.py --checkpoint /path/epoch_0080.pth --env gcp \
        --split test --n_cases 0
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
import torch.nn.functional as F
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent / "data"))
sys.path.insert(0, str(Path(__file__).parent))

from ccrnet.config.phase2_config import Phase2Config
from ccrnet.config.env_config import apply_env
from ccrnet.models.ccrnet import CCRNet
from ccrnet.models.factory import build_model, ARCHES
from ccrnet.utils.checkpoint import load_checkpoint
from ccrnet.utils.evaluation import upsample_routing_to_volume, concept_auroc_from_volume
from attribution_baselines import GradCAM3D, explanation_maps
from brats_dataset import get_dataloader

WINDOWS = ("fg", "all")          # foreground (discriminability) + all (localization)
BASELINES = ("gradcam", "gradient", "ig", "occlusion")


def main() -> None:
    _default_config = str(Path(__file__).parent.parent / "configs" / "brats_phase2.yaml")
    ap = argparse.ArgumentParser(description="Concept-discriminability of explanation methods")
    ap.add_argument("--checkpoint", type=str, required=True)
    ap.add_argument("--env",        type=str, default="local")
    ap.add_argument("--config",     type=str, default=_default_config)
    ap.add_argument("--split",      type=str, default="test", choices=["val", "test"])
    ap.add_argument("--n_cases",    type=int, default=0, help="0 = all")
    ap.add_argument("--occ_grid",   type=int, default=4)
    ap.add_argument("--ig_steps",   type=int, default=16)
    ap.add_argument("--methods",    type=str, default="ccr,gradcam,gradient,ig,occlusion",
                    help="comma list from {ccr,gradcam,gradient,ig,occlusion,segsoftmax,"
                         "segsoftmax_16}; use 'ccr' alone for a non-Swin backbone (skips the "
                         "Grad-CAM decoder hook). segsoftmax = the model's own posterior (the "
                         "prediction itself, not a post-hoc method); segsoftmax_16 = the same "
                         "posterior pooled to the router's token grid (granularity-matched).")
    ap.add_argument("--model",      type=str, default="ccrnet", choices=list(ARCHES))
    ap.add_argument("--save_dir",   type=str, default="")
    ap.add_argument("--device",     type=str, default="")
    ap.add_argument("--no_sync_gcs", action="store_true")
    args = ap.parse_args()

    config = Phase2Config.from_yaml(args.config)
    env_cfg = apply_env(config, env=args.env, config_dir=Path(__file__).parent.parent / "configs")
    device = torch.device(args.device if args.device else
                          ("cuda" if torch.cuda.is_available() else "cpu"))
    save_dir = args.save_dir or str(
        Path(config.training.checkpoint_dir) / "evals" / f"{args.split}_alignment")
    os.makedirs(save_dir, exist_ok=True)

    model = build_model(config, args.model).to(device)
    start_epoch, _ = load_checkpoint(args.checkpoint, model)
    model.eval()
    print(f"Loaded checkpoint (epoch {start_epoch - 1}) on {device} [{args.model}]")

    methods = tuple(m.strip() for m in args.methods.split(",") if m.strip())
    # Grad-CAM hooks the Swin decoder's enc3 block; build it only if requested AND available
    # (a CNN backbone like SegResNet has no decoder.enc3 -> run with --methods ccr).
    gradcam = (GradCAM3D(model, model.decoder.enc3)
               if "gradcam" in methods and hasattr(getattr(model, "decoder", None), "enc3")
               else None)

    concept_names = config.ccr.concept_names
    concept_indices = (1, 2, 3)

    loader = get_dataloader(
        data_root=config.data.data_root, split=args.split, batch_size=1,
        num_workers=config.data.num_workers, spatial_size=config.data.spatial_size,
        brats_version=config.data.brats_version, val_fraction=config.data.val_fraction,
        test_fraction=config.data.test_fraction, seed=config.data.seed,
    )
    n_total = len(loader.dataset)
    n_cases = n_total if args.n_cases == 0 else min(args.n_cases, n_total)
    print(f"Methods: {methods} | cases: {n_cases}/{n_total} | windows: {WINDOWS}")

    per_patient: List[Dict] = []
    t0 = time.time(); seen = 0
    for batch in tqdm(loader, total=n_cases, desc="  alignment"):
        if seen >= n_cases:
            break
        image = batch["image"].to(device)
        label = batch["label"].to(device)          # [1, H, W, D]
        pid = batch["patient_id"][0] if isinstance(batch["patient_id"], list) else batch["patient_id"]
        target_shape = tuple(image.shape[2:])

        with torch.no_grad():
            out = model(image)
            grid = model.get_grid_shape()
            baseline_probs = torch.softmax(out["seg_logits"], dim=1)
            ccr_vol = upsample_routing_to_volume(out["routing_probs"], grid, target_shape)  # [1,K,H,W,D]

        row: Dict = {"patient_id": pid}
        for method in methods:
            if method == "ccr":
                maps = {k: ccr_vol[0, k] for k in concept_indices}
            elif method == "segsoftmax":
                # the model's own per-class segmentation posterior as a "concept map"
                # (the intrinsic baseline: a plain segmentation head also discriminates concepts).
                maps = {k: baseline_probs[0, k] for k in concept_indices}
            elif method == "segsoftmax_16":
                # GRANULARITY-MATCHED control. Plain segsoftmax decides over 128^3 = 2.1M voxels
                # while CCR routing decides over 16^3 = 4096 tokens -> 512x fewer decision units,
                # so the raw comparison confounds "carries more concept information" with "has
                # finer spatial resolution". Here the posterior is average-pooled DOWN to the
                # router's own token grid and pushed back up through the identical trilinear
                # operator used for routing (upsample_routing_to_volume), so both maps carry
                # exactly the same spatial degrees of freedom and only the information differs.
                pooled = F.adaptive_avg_pool3d(baseline_probs, grid)          # [1,K,h,w,d]
                flat = pooled.flatten(2).permute(0, 2, 1)                     # [1,N,K]
                ss16 = upsample_routing_to_volume(flat, grid, target_shape)   # [1,K,H,W,D]
                maps = {k: ss16[0, k] for k in concept_indices}
            else:
                maps = explanation_maps(method, model, gradcam, image, concept_indices,
                                        target_shape, baseline_probs=baseline_probs,
                                        occ_grid=args.occ_grid, ig_steps=args.ig_steps)
            for k in concept_indices:
                name = concept_names[k]
                for win in WINDOWS:
                    row[f"{method}_auroc_{win}_{name}"] = concept_auroc_from_volume(
                        maps[k], label[0], k, foreground_only=(win == "fg"))
        per_patient.append(row); seen += 1

    elapsed = time.time() - t0
    print(f"\nDone {seen} cases in {elapsed:.1f}s")

    # --- aggregate (nanmean; concept absent -> NaN) ---
    summary: Dict = {}
    for win in WINDOWS:
        label_win = "concept-discriminability (vs other tumor)" if win == "fg" else "localization (vs all)"
        print(f"\n=== Concept AUROC — {win}: {label_win} (0.5 = chance) ===")
        for method in methods:
            cells = []
            for k in concept_indices:
                name = concept_names[k]
                vals = np.array([r[f"{method}_auroc_{win}_{name}"] for r in per_patient], float)
                n_used = int(np.isfinite(vals).sum())
                mean_v, std_v = float(np.nanmean(vals)), float(np.nanstd(vals))
                summary[f"{method}_auroc_{win}_{name}"] = {"mean": mean_v, "std": std_v, "n": n_used}
                cells.append(f"{name.split('_')[0]}={mean_v:.3f}")
            print(f"  {method:10s} " + "  ".join(cells))

    csv_path = os.path.join(save_dir, f"{args.split}_alignment.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(per_patient[0].keys()))
        w.writeheader(); w.writerows(per_patient)
    summary.update({"methods": list(methods), "n_cases": seen, "checkpoint": args.checkpoint,
                    "split": args.split, "elapsed_s": float(elapsed)})
    json_path = os.path.join(save_dir, f"{args.split}_alignment_summary.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nCSV:  {csv_path}\nJSON: {json_path}")

    gcs = env_cfg.get("paths", {}).get("gcs_checkpoint_dir", "")
    if gcs and not args.no_sync_gcs:
        run_id = Path(args.checkpoint).parent.name
        dst = f"{gcs.rstrip('/')}/{run_id}/evals/{args.split}_alignment"
        print(f"\nSyncing to GCS: {save_dir} -> {dst}")
        try:
            subprocess.run(["gsutil", "-m", "rsync", "-r", save_dir, dst], check=True)
            print("  GCS sync complete.")
        except Exception as e:
            print(f"  [warn] GCS sync failed: {e}")


if __name__ == "__main__":
    main()
