"""
routing_intervention.py — the causal test of the CCR claim.

Every other experiment in this repo is OBSERVATIONAL: we measure how well the routing
correlates with clinical concepts (CAS), how well it discriminates them (concept AUROC),
how much removing high-routing voxels hurts (deletion). None of those can separate "the
routing drives the prediction" from "the routing and the prediction are both downstream of
the same features". This script settles it by intervening.

The manipulation
----------------
At the bottleneck each token t has a hard concept assignment k*(t). We force the tokens
currently assigned to concept `a` through the expert for concept `b` instead, changing
nothing else, and re-run the decoder. If the routing is causally upstream of the output --
the paper's central structural claim -- the decoder's posterior over exactly those voxels
must move toward class b.

This experiment is IMPOSSIBLE for post-hoc attribution. There is nothing in a Grad-CAM or
integrated-gradients map to intervene on: the map is a readout, so editing it changes
nothing about the model. That asymmetry is the point.

Controls (all reported, all necessary)
--------------------------------------
- identity (a -> a)   : must be exactly 0. Validates that the effect comes from re-routing
                        and not from the soft/hard dispatch switch. The baseline is run
                        through the SAME hard-dispatch path (all -1 override) so this holds
                        by construction rather than by luck.
- off-target classes  : classes other than a and b should barely move -- otherwise we are
                        measuring generic perturbation, not concept-specific control.
- outside the region  : voxels whose tokens were NOT re-routed should barely move --
                        otherwise the effect is diffuse rather than localized.

Outputs a K x K matrix of effects; the diagonal is the identity control and the
off-diagonal is the result.

SCOPE (matters for how the headline number reads)
-------------------------------------------------
By default the intervention is restricted to tokens overlapping the TUMOUR NEIGHBOURHOOD.
The router assigns most of the volume to background and normal tissue (~26% of all tokens
go to edema alone), so re-routing every token of a concept repaints the whole brain. That
is genuine causal control, but it mostly demonstrates "the output follows the routing
everywhere" -- which is close to tautological in direct_mode, where the routing logits are
literally an additive term of the output. The defensible claim is concept control WHERE IT
IS CLINICALLY MEANINGFUL, i.e. inside the tumour. `--all_tokens` restores the unrestricted
behaviour; `scope` is recorded in the summary JSON so a CSV can never be misread.

Usage
-----
    python pipeline/routing_intervention.py \
        --checkpoint ~/checkpoints/20260621_120750/epoch_0080.pth \
        --model ccrnet --config configs/brats_phase2.yaml --env gcp \
        --split test --n_cases 0 --save_dir ~/checkpoints/evals/test_intervention
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

from ccr.modules.dispatcher import routing_intervention
from ccrnet.config.phase2_config import Phase2Config
from ccrnet.config.env_config import apply_env
from ccrnet.models.factory import build_model, ARCHES
from ccrnet.utils.checkpoint import load_checkpoint
from brats_dataset import get_dataloader


def _token_mask_to_voxels(mask_tokens: torch.Tensor, grid, target_shape) -> torch.Tensor:
    """[N] bool over tokens -> [H,W,D] bool over voxels (nearest, i.e. each token's block)."""
    m = mask_tokens.reshape(1, 1, *grid).float()
    return F.interpolate(m, size=target_shape, mode="nearest")[0, 0] > 0.5


def main() -> None:
    _default_config = str(Path(__file__).parent.parent / "configs" / "brats_phase2.yaml")
    ap = argparse.ArgumentParser(description="Causal routing-intervention experiment")
    ap.add_argument("--checkpoint", type=str, required=True)
    ap.add_argument("--env",        type=str, default="local")
    ap.add_argument("--config",     type=str, default=_default_config)
    ap.add_argument("--split",      type=str, default="test", choices=["val", "test"])
    ap.add_argument("--n_cases",    type=int, default=0, help="0 = all")
    ap.add_argument("--model",      type=str, default="ccrnet", choices=list(ARCHES))
    ap.add_argument("--concepts",   type=str, default="1,2,3",
                    help="concept ids to use as intervention sources/targets "
                         "(default: the three foreground concepts; add 0 for background)")
    ap.add_argument("--all_tokens", action="store_true",
                    help="re-route EVERY token assigned to the source concept, including "
                         "background. Default restricts to the tumour neighbourhood -- see "
                         "the note in the module docstring on why that is the honest headline.")
    ap.add_argument("--save_dir",   type=str, default="")
    ap.add_argument("--device",     type=str, default="")
    ap.add_argument("--no_sync_gcs", action="store_true",
                    help="skip mirroring results to GCS")
    args = ap.parse_args()

    config = Phase2Config.from_yaml(args.config)
    env_cfg = apply_env(config, env=args.env, config_dir=Path(__file__).parent.parent / "configs")
    device = torch.device(args.device if args.device else
                          ("cuda" if torch.cuda.is_available() else "cpu"))
    save_dir = args.save_dir or str(
        Path(config.training.checkpoint_dir) / "evals" / f"{args.split}_intervention")
    os.makedirs(save_dir, exist_ok=True)

    model = build_model(config, args.model).to(device)
    start_epoch, _ = load_checkpoint(args.checkpoint, model)
    model.eval()
    print(f"Loaded checkpoint (epoch {start_epoch - 1}) on {device} [{args.model}]")

    ccr_module = getattr(model, "ccr", None)
    if ccr_module is None:
        raise SystemExit("This model has no CCR module (ccr.enabled=false) — nothing to intervene on.")

    names = config.ccr.concept_names
    ids = [int(c) for c in args.concepts.split(",") if c.strip() != ""]

    loader = get_dataloader(
        data_root=config.data.data_root, split=args.split, batch_size=1,
        num_workers=config.data.num_workers, spatial_size=config.data.spatial_size,
        brats_version=config.data.brats_version, val_fraction=config.data.val_fraction,
        test_fraction=config.data.test_fraction, seed=config.data.seed,
    )
    n_total = len(loader.dataset)
    n_cases = n_total if args.n_cases == 0 else min(args.n_cases, n_total)
    print(f"Concepts: {[names[i] for i in ids]} | cases: {n_cases}/{n_total} | "
          f"{len(ids)*len(ids)} interventions/case")

    per_patient: List[Dict] = []
    t0 = time.time(); seen = 0
    for batch in tqdm(loader, total=n_cases, desc="  intervention"):
        if seen >= n_cases:
            break
        image = batch["image"].to(device)
        label = batch["label"].to(device)[0]
        pid = batch["patient_id"][0] if isinstance(batch["patient_id"], list) else batch["patient_id"]
        target_shape = tuple(image.shape[2:])

        with torch.no_grad():
            # One probe forward to learn the token grid (the model records it during
            # forward), then the real baseline through the SAME hard-dispatch path as the
            # interventions -- so any measured effect is due to re-routing and not to the
            # soft/hard dispatch switch.
            n_tok = model(image)["assignments"].shape[1]
            grid = model.get_grid_shape()
            null_override = torch.full((1, n_tok), -1, dtype=torch.long, device=device)
            with routing_intervention(ccr_module, null_override):
                out0 = model(image)
            p0 = torch.softmax(out0["seg_logits"], dim=1)[0]      # [K,H,W,D]
            a0 = out0["assignments"][0]                           # [N]

        # The router sends most of the volume (~26% of tokens to edema alone) to background
        # and normal tissue, so re-routing EVERY token of a concept repaints the whole brain.
        # That is real causal control but it mostly demonstrates "the output follows the
        # routing everywhere", which is near-tautological in direct mode. Restricting to the
        # tumour neighbourhood measures concept control where it is clinically meaningful.
        tumour_tok = None
        if not args.all_tokens:
            lab_grid = F.interpolate((label > 0).float().reshape(1, 1, *target_shape),
                                     size=grid, mode="area")[0, 0]
            tumour_tok = (lab_grid.flatten() > 0.0)

        row: Dict = {"patient_id": pid}
        for a in ids:
            src = (a0 == a)
            if tumour_tok is not None:
                src = src & tumour_tok
            n_src = int(src.sum())
            row[f"n_tokens_{names[a]}"] = n_src
            if n_src == 0:
                for b in ids:
                    for f in ("dtarget", "dsource", "doff", "doutside", "flip"):
                        row[f"{f}_{names[a]}_to_{names[b]}"] = float("nan")
                continue

            region = _token_mask_to_voxels(src, grid, target_shape)   # [H,W,D] bool
            outside = ~region
            n_reg = int(region.sum())

            for b in ids:
                override = torch.full((1, n_tok), -1, dtype=torch.long, device=device)
                override[0, src] = b
                with torch.no_grad(), routing_intervention(ccr_module, override):
                    p1 = torch.softmax(model(image)["seg_logits"], dim=1)[0]

                d = p1 - p0                                            # [K,H,W,D]
                dtarget = float(d[b][region].mean())                   # want > 0
                dsource = float(d[a][region].mean())                   # want < 0 (a != b)
                others = [c for c in range(len(names)) if c not in (a, b)]
                doff = (float(torch.stack([d[c][region].abs() for c in others]).mean())
                        if others else float("nan"))
                doutside = (float(d.abs().mean(0)[outside].mean())
                            if int(outside.sum()) > 0 else float("nan"))
                flip = float(((p1.argmax(0) == b) & region).sum() / max(n_reg, 1)
                             - ((p0.argmax(0) == b) & region).sum() / max(n_reg, 1))

                row[f"dtarget_{names[a]}_to_{names[b]}"]  = dtarget
                row[f"dsource_{names[a]}_to_{names[b]}"]  = dsource
                row[f"doff_{names[a]}_to_{names[b]}"]     = doff
                row[f"doutside_{names[a]}_to_{names[b]}"] = doutside
                row[f"flip_{names[a]}_to_{names[b]}"]     = flip

        per_patient.append(row); seen += 1

    elapsed = time.time() - t0
    print(f"\nDone {seen} cases in {elapsed:.1f}s")

    # --- aggregate ---
    summary: Dict = {}
    for f, label in (("dtarget", "Delta posterior of the TARGET class in the re-routed region"),
                     ("flip",    "Fraction of re-routed voxels whose predicted label became the target"),
                     ("dsource", "Delta posterior of the SOURCE class in the re-routed region"),
                     ("doff",    "Mean |Delta| of OFF-TARGET classes (specificity; want ~0)"),
                     ("doutside","Mean |Delta| OUTSIDE the re-routed region (locality; want ~0)")):
        print(f"\n=== {f}: {label} ===")
        corner = "from / to"
        print(f"{corner:>18s}" + "".join(f"{names[b][:9]:>11s}" for b in ids))
        for a in ids:
            cells = []
            for b in ids:
                vals = np.array([r.get(f"{f}_{names[a]}_to_{names[b]}", np.nan)
                                 for r in per_patient], float)
                mean_v = float(np.nanmean(vals)) if np.isfinite(vals).any() else float("nan")
                summary[f"{f}_{names[a]}_to_{names[b]}"] = {
                    "mean": mean_v,
                    "std": float(np.nanstd(vals)) if np.isfinite(vals).any() else float("nan"),
                    "n": int(np.isfinite(vals).sum()),
                }
                cells.append(f"{mean_v:>11.4f}")
            print(f"{names[a][:17]:>18s}" + "".join(cells))

    csv_path = os.path.join(save_dir, f"{args.split}_intervention.csv")
    with open(csv_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(per_patient[0].keys()))
        w.writeheader(); w.writerows(per_patient)
    summary.update({"scope": "all_tokens" if args.all_tokens else "tumour_neighbourhood",
                    "concepts": [names[i] for i in ids], "n_cases": seen,
                    "checkpoint": args.checkpoint, "split": args.split,
                    "elapsed_s": float(elapsed)})
    json_path = os.path.join(save_dir, f"{args.split}_intervention_summary.json")
    with open(json_path, "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"\nCSV:  {csv_path}\nJSON: {json_path}")

    # Mirror results to GCS. Without this the outputs live only on the VM disk -- which is
    # how the direct-mode intervention and diagnostics were nearly lost when the instance
    # was stopped. The destination is keyed on the SAVE-DIR basename rather than a hardcoded
    # suffix (the flaw in concept_alignment.py), so two runs writing to different --save_dir
    # values cannot overwrite each other on GCS.
    gcs = env_cfg.get("paths", {}).get("gcs_checkpoint_dir", "")
    if gcs and not args.no_sync_gcs:
        run_id = Path(args.checkpoint).parent.name
        dst = f"{gcs.rstrip(chr(47))}/{run_id}/evals/{Path(save_dir).name}"
        print("")
        print(f"Syncing to GCS: {save_dir} -> {dst}")
        try:
            subprocess.run(["gsutil", "-m", "rsync", "-r", save_dir, dst], check=True)
            print("  GCS sync complete.")
        except Exception as e:
            print(f"  [warn] GCS sync failed: {e}")


if __name__ == "__main__":
    main()
