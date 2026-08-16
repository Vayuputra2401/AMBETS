"""
error_detection.py — is the routing a useful SECOND OPINION on the segmentation?

Motivation. Three independent observations say the routing is not simply a blurred copy of
the output: DD is large (0.73/0.53/0.40), CCR beats the segmentation posterior on insertion
for necrotic core (+0.133), and qualitatively the routing has been seen flagging enhancing
tumor that the decoder under-segments. If routing/output DISAGREEMENT predicts where the
final mask is WRONG, that is a free, intrinsic error-detection signal -- something the
segmentation cannot provide about itself, and clinically far more useful than another
heatmap.

This matters because the causal claim did not survive: re-routing tokens barely moves the
prediction (pipeline/routing_intervention.py). "The routing drives the output" is dead, but
"the routing knows something the output does not" is a different claim, and it is the one
the evidence actually points at.

The test. Over the tumor neighbourhood (predicted OR ground-truth foreground -- scoring the
whole volume would let the vast correct background inflate every method), predict the binary
event "the final prediction is wrong here" from each signal, and report AUROC.

Signals
-------
  disagree_hard  : routing argmax != predicted label            (the CCR-specific signal)
  routing_entropy: per-voxel entropy of the routing distribution (CCR's free uncertainty)
  disagree_soft  : 1 - routing probability of the PREDICTED class
  seg_confidence : 1 - max_k softmax_k   <-- THE BASELINE THAT MATTERS

seg_confidence is the honest competitor: every segmentation model already has it, for free,
with no CCR module. If routing disagreement does not beat it, the second-opinion framing is
dead too, and we should know that before building a paper on it.
"""

from __future__ import annotations

import argparse
import csv
import json
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
sys.path.insert(0, str(Path(__file__).parent))

from ccrnet.config.phase2_config import Phase2Config
from ccrnet.config.env_config import apply_env
from ccrnet.models.factory import build_model, ARCHES
from ccrnet.utils.checkpoint import load_checkpoint
from ccrnet.utils.evaluation import upsample_routing_to_volume
from brats_dataset import get_dataloader

SIGNALS = ("disagree_hard", "routing_entropy", "disagree_soft", "seg_confidence")


def _auroc(score: torch.Tensor, positive: torch.Tensor) -> float:
    """AUROC of `score` predicting the boolean `positive`, via the rank identity."""
    n_pos = int(positive.sum())
    n_neg = int((~positive).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    s = score.float().flatten()
    ranks = torch.empty_like(s)
    order = torch.argsort(s)
    ranks[order] = torch.arange(1, s.numel() + 1, dtype=s.dtype, device=s.device)
    # average ranks over ties so a binary signal is not scored arbitrarily
    uniq, inv = torch.unique(s, return_inverse=True)
    mean_r = torch.zeros_like(uniq).scatter_reduce_(
        0, inv, ranks, reduce="mean", include_self=False)
    ranks = mean_r[inv]
    r_pos = ranks[positive.flatten()].sum()
    return float((r_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def main() -> None:
    _default_config = str(Path(__file__).parent.parent / "configs" / "brats_phase2.yaml")
    ap = argparse.ArgumentParser(description="Does routing disagreement predict segmentation error?")
    ap.add_argument("--checkpoint", type=str, required=True)
    ap.add_argument("--env",        type=str, default="local")
    ap.add_argument("--config",     type=str, default=_default_config)
    ap.add_argument("--split",      type=str, default="test", choices=["val", "test"])
    ap.add_argument("--n_cases",    type=int, default=0, help="0 = all")
    ap.add_argument("--model",      type=str, default="ccrnet", choices=list(ARCHES))
    ap.add_argument("--save_dir",   type=str, default="")
    ap.add_argument("--device",     type=str, default="")
    args = ap.parse_args()

    config = Phase2Config.from_yaml(args.config)
    apply_env(config, env=args.env, config_dir=Path(__file__).parent.parent / "configs")
    device = torch.device(args.device if args.device else
                          ("cuda" if torch.cuda.is_available() else "cpu"))
    save_dir = args.save_dir or str(
        Path(config.training.checkpoint_dir) / "evals" / f"{args.split}_errdet")
    os.makedirs(save_dir, exist_ok=True)

    model = build_model(config, args.model).to(device)
    start_epoch, _ = load_checkpoint(args.checkpoint, model)
    model.eval()
    if getattr(model, "ccr", None) is None:
        raise SystemExit("Model has no CCR module — no routing to compare against.")
    print(f"Loaded checkpoint (epoch {start_epoch - 1}) on {device} [{args.model}]")

    loader = get_dataloader(
        data_root=config.data.data_root, split=args.split, batch_size=1,
        num_workers=config.data.num_workers, spatial_size=config.data.spatial_size,
        brats_version=config.data.brats_version, val_fraction=config.data.val_fraction,
        test_fraction=config.data.test_fraction, seed=config.data.seed,
    )
    n_cases = len(loader.dataset) if args.n_cases == 0 else min(args.n_cases, len(loader.dataset))

    per_patient: List[Dict] = []
    t0 = time.time(); seen = 0
    for batch in tqdm(loader, total=n_cases, desc="  err-detect"):
        if seen >= n_cases:
            break
        image = batch["image"].to(device)
        label = batch["label"].to(device)[0]                       # [H,W,D]
        pid = batch["patient_id"][0] if isinstance(batch["patient_id"], list) else batch["patient_id"]

        with torch.no_grad():
            out = model(image)
            grid = model.get_grid_shape()
            probs = torch.softmax(out["seg_logits"], dim=1)[0]     # [K,H,W,D]
            pred = probs.argmax(0)                                 # [H,W,D]
            rvol = upsample_routing_to_volume(
                out["routing_probs"], grid, tuple(image.shape[2:]))[0]   # [K,H,W,D]

        # Scoring region: the tumour neighbourhood. Over the whole volume the correct
        # background would dominate and every signal would look excellent.
        region = (pred > 0) | (label > 0)
        wrong = (pred != label) & region
        row: Dict = {"patient_id": pid,
                     "n_region": int(region.sum()), "n_wrong": int(wrong.sum()),
                     "err_rate": float(wrong.sum() / region.sum()) if int(region.sum()) else float("nan")}

        if int(region.sum()) == 0 or int(wrong.sum()) == 0:
            for s in SIGNALS:
                row[f"auroc_{s}"] = float("nan")
        else:
            r_arg = rvol.argmax(0)
            p_of_pred = rvol.gather(0, pred.unsqueeze(0))[0]        # routing prob of predicted class
            ent = -(rvol.clamp_min(1e-8) * rvol.clamp_min(1e-8).log()).sum(0)
            sig = {
                "disagree_hard":  (r_arg != pred).float(),
                "routing_entropy": ent,
                "disagree_soft":  1.0 - p_of_pred,
                "seg_confidence": 1.0 - probs.max(0).values,
            }
            for s in SIGNALS:
                row[f"auroc_{s}"] = _auroc(sig[s][region], wrong[region])

        per_patient.append(row); seen += 1

    elapsed = time.time() - t0
    print(f"\nDone {seen} cases in {elapsed:.1f}s")

    summary: Dict = {}
    print("\n=== AUROC predicting 'the final mask is wrong here' (tumour neighbourhood) ===")
    print("    0.5 = chance. seg_confidence is the free baseline every model already has.")
    for s in SIGNALS:
        vals = np.array([r[f"auroc_{s}"] for r in per_patient], float)
        m = float(np.nanmean(vals))
        summary[f"auroc_{s}"] = {"mean": m, "std": float(np.nanstd(vals)),
                                 "n": int(np.isfinite(vals).sum())}
        mark = "  <-- baseline" if s == "seg_confidence" else ""
        print(f"  {s:18s} {m:.4f}  (n={int(np.isfinite(vals).sum())}){mark}")

    base = summary["auroc_seg_confidence"]["mean"]
    best_ccr = max(summary[f"auroc_{s}"]["mean"] for s in SIGNALS if s != "seg_confidence")
    summary["best_ccr_minus_baseline"] = best_ccr - base
    print(f"\n  best CCR signal - seg_confidence = {best_ccr - base:+.4f}")
    print("  > 0 : routing carries error information the output does not -> second-opinion claim")
    print("  <=0 : the posterior already knows where it is wrong -> claim is dead, report it")

    err = np.array([r["err_rate"] for r in per_patient], float)
    summary["mean_error_rate_in_region"] = float(np.nanmean(err))
    print(f"  (mean error rate in the scored region: {np.nanmean(err):.4f})")

    csv_path = os.path.join(save_dir, f"{args.split}_errdet.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(per_patient[0].keys()))
        w.writeheader(); w.writerows(per_patient)
    summary.update({"checkpoint": args.checkpoint, "split": args.split, "n_cases": seen,
                    "elapsed_s": float(elapsed)})
    json_path = os.path.join(save_dir, f"{args.split}_errdet_summary.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nCSV:  {csv_path}\nJSON: {json_path}")


if __name__ == "__main__":
    main()
