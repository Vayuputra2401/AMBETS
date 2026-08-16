"""
linear_probe.py — does the CCR architecture add anything a linear probe would not?

This is the control that decides whether there is an architectural contribution at all.
CCR's router is an MLP with learnable concept prototypes, temperature annealing, and a bank
of K experts behind it, and it scores 0.87 concept-discriminability. But the router reads
bottleneck features h that a pretrained segmentation encoder produced. If a plain multinomial
logistic regression on those same h reaches ~0.87, then none of that machinery is doing work:
the concept information was already linearly decodable, and "routing" is an elaborate readout.

Two checkpoints answer two different questions:

  full model  -- probe vs router on the SAME features. Isolates whether the router's
                 architecture (prototypes, temperature, expert bank) beats a linear map.
                 Note these features were themselves shaped by L_align gradients, so this
                 is the weaker of the two comparisons.

  A1 (no L_align) -- features trained WITHOUT any concept supervision at the bottleneck.
                 If a probe here still reaches ~0.85, the concept structure was already
                 present in an ordinary segmentation encoder and L_align is not adding it
                 either. That is the strong version of the control.

To keep the number directly comparable to every other discriminability figure in the repo,
the probe's per-token posterior is upsampled to voxels through the SAME operator used for
routing and scored with the SAME concept_auroc_from_volume in the SAME foreground window.

Usage
-----
    python pipeline/linear_probe.py \
        --checkpoint ~/checkpoints/20260621_120750/epoch_0080.pth \
        --model ccrnet --config configs/brats_phase2.yaml --env gcp \
        --fit_split val --n_fit 40 --n_cases 0
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
from ccrnet.utils.token_labels import downsample_labels_to_tokens
from ccrnet.utils.evaluation import upsample_routing_to_volume, concept_auroc_from_volume
from brats_dataset import get_dataloader


def _loader(config, split: str):
    return get_dataloader(
        data_root=config.data.data_root, split=split, batch_size=1,
        num_workers=config.data.num_workers, spatial_size=config.data.spatial_size,
        brats_version=config.data.brats_version, val_fraction=config.data.val_fraction,
        test_fraction=config.data.test_fraction, seed=config.data.seed,
    )


def main() -> None:
    _default_config = str(Path(__file__).parent.parent / "configs" / "brats_phase2.yaml")
    ap = argparse.ArgumentParser(description="Linear-probe control for the CCR router")
    ap.add_argument("--checkpoint", type=str, required=True)
    ap.add_argument("--env",        type=str, default="local")
    ap.add_argument("--config",     type=str, default=_default_config)
    ap.add_argument("--split",      type=str, default="test", choices=["val", "test"])
    ap.add_argument("--fit_split",  type=str, default="val", choices=["train", "val"],
                    help="split used to FIT the probe (never the eval split)")
    ap.add_argument("--n_fit",      type=int, default=40, help="cases used to fit the probe")
    ap.add_argument("--n_cases",    type=int, default=0, help="eval cases, 0 = all")
    ap.add_argument("--max_tokens", type=int, default=300000,
                    help="cap on training tokens (subsampled) to keep the solver fast")
    ap.add_argument("--model",      type=str, default="ccrnet", choices=list(ARCHES))
    ap.add_argument("--save_dir",   type=str, default="")
    ap.add_argument("--device",     type=str, default="")
    args = ap.parse_args()

    from sklearn.linear_model import LogisticRegression

    config = Phase2Config.from_yaml(args.config)
    apply_env(config, env=args.env, config_dir=Path(__file__).parent.parent / "configs")
    device = torch.device(args.device if args.device else
                          ("cuda" if torch.cuda.is_available() else "cpu"))
    save_dir = args.save_dir or str(
        Path(config.training.checkpoint_dir) / "evals" / f"{args.split}_probe")
    os.makedirs(save_dir, exist_ok=True)

    model = build_model(config, args.model).to(device)
    start_epoch, _ = load_checkpoint(args.checkpoint, model)
    model.eval()
    ccr = getattr(model, "ccr", None)
    if ccr is None:
        raise SystemExit("Model has no CCR module — nothing to probe.")
    names = config.ccr.concept_names
    K = len(names)
    concept_indices = (1, 2, 3)
    print(f"Loaded checkpoint (epoch {start_epoch - 1}) on {device} [{args.model}]")

    # Grab the tokens ENTERING the CCR module — the features the router sees.
    grabbed: Dict[str, torch.Tensor] = {}
    ccr.register_forward_pre_hook(lambda m, inp: grabbed.__setitem__("h", inp[0].detach()))

    # ---------------- fit ----------------
    fit_loader = _loader(config, args.fit_split)
    n_fit = min(args.n_fit, len(fit_loader.dataset))
    X: List[np.ndarray] = []
    y: List[np.ndarray] = []
    seen = 0
    for batch in tqdm(fit_loader, total=n_fit, desc=f"  fit[{args.fit_split}]"):
        if seen >= n_fit:
            break
        image = batch["image"].to(device)
        label = batch["label"].to(device)
        with torch.no_grad():
            model(image)
            grid = model.get_grid_shape()
            h = grabbed["h"][0]                                   # [N, D]
            tl = downsample_labels_to_tokens(label, grid)[0]      # [N]
        X.append(h.cpu().numpy()); y.append(tl.cpu().numpy())
        seen += 1

    X = np.concatenate(X); y = np.concatenate(y)
    if X.shape[0] > args.max_tokens:                              # subsample, keep classes
        rng = np.random.default_rng(0)
        idx = rng.choice(X.shape[0], args.max_tokens, replace=False)
        X, y = X[idx], y[idx]
    print(f"Fitting probe on {X.shape[0]} tokens, dim {X.shape[1]}, "
          f"class counts {np.bincount(y, minlength=K).tolist()}")

    probe = LogisticRegression(max_iter=2000, multi_class="multinomial", n_jobs=-1)
    probe.fit(X, y)
    print(f"  train accuracy {probe.score(X, y):.4f}")

    # sklearn drops classes absent from the fit set; rebuild the full K-column posterior
    present = list(probe.classes_)

    # ---------------- evaluate ----------------
    eval_loader = _loader(config, args.split)
    n_cases = len(eval_loader.dataset) if args.n_cases == 0 else min(args.n_cases,
                                                                    len(eval_loader.dataset))
    per_patient: List[Dict] = []
    t0 = time.time(); seen = 0
    for batch in tqdm(eval_loader, total=n_cases, desc=f"  eval[{args.split}]"):
        if seen >= n_cases:
            break
        image = batch["image"].to(device)
        label = batch["label"].to(device)
        pid = batch["patient_id"][0] if isinstance(batch["patient_id"], list) else batch["patient_id"]
        target_shape = tuple(image.shape[2:])

        with torch.no_grad():
            out = model(image)
            grid = model.get_grid_shape()
            h = grabbed["h"][0].cpu().numpy()                      # [N, D]
            probe_p = np.zeros((h.shape[0], K), dtype=np.float32)
            probe_p[:, present] = probe.predict_proba(h)
            probe_t = torch.from_numpy(probe_p).unsqueeze(0).to(device)   # [1,N,K]

            # identical downstream path as the router, so the numbers are comparable
            probe_vol = upsample_routing_to_volume(probe_t, grid, target_shape)
            ccr_vol = upsample_routing_to_volume(out["routing_probs"], grid, target_shape)

        row: Dict = {"patient_id": pid}
        for k in concept_indices:
            for tag, vol in (("probe", probe_vol), ("ccr", ccr_vol)):
                row[f"{tag}_auroc_fg_{names[k]}"] = concept_auroc_from_volume(
                    vol[0, k], label[0], k, foreground_only=True)
        per_patient.append(row); seen += 1

    elapsed = time.time() - t0
    print(f"\nDone {seen} cases in {elapsed:.1f}s")

    summary: Dict = {}
    print("\n=== Concept-discriminability (fg AUROC): linear probe vs CCR router ===")
    for tag in ("ccr", "probe"):
        cells = []
        for k in concept_indices:
            vals = np.array([r[f"{tag}_auroc_fg_{names[k]}"] for r in per_patient], float)
            m = float(np.nanmean(vals))
            summary[f"{tag}_auroc_fg_{names[k]}"] = {
                "mean": m, "std": float(np.nanstd(vals)), "n": int(np.isfinite(vals).sum())}
            cells.append(f"{names[k].split('_')[0]}={m:.3f}")
        mean_all = float(np.nanmean([summary[f'{tag}_auroc_fg_{names[k]}']['mean']
                                     for k in concept_indices]))
        summary[f"{tag}_mean"] = mean_all
        print(f"  {tag:6s} " + "  ".join(cells) + f"   mean={mean_all:.4f}")

    gap = summary["ccr_mean"] - summary["probe_mean"]
    print(f"\n  router - probe = {gap:+.4f}")
    print("  |gap| ~ 0  -> the router's architecture adds nothing over a linear readout;")
    print("               the concept structure was already linearly decodable in h.")

    csv_path = os.path.join(save_dir, f"{args.split}_probe.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(per_patient[0].keys()))
        w.writeheader(); w.writerows(per_patient)
    summary.update({"checkpoint": args.checkpoint, "split": args.split,
                    "fit_split": args.fit_split, "n_fit": n_fit, "n_cases": seen,
                    "n_fit_tokens": int(X.shape[0]), "router_minus_probe": gap})
    json_path = os.path.join(save_dir, f"{args.split}_probe_summary.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nCSV:  {csv_path}\nJSON: {json_path}")


if __name__ == "__main__":
    main()
