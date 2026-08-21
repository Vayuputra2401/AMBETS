"""
intervention_figure.py — the money shot: what a causal intervention LOOKS like.

The intervention matrix (routing_intervention.py) says re-routing a token flips ~82% of the
affected voxels to the target concept. That is the paper's central result and it currently
exists only as a table. This renders it.

Layout
------
Top row (context):   T1ce | ground truth | routing argmax | baseline prediction
Grid (3x3):          rows = SOURCE concept, cols = TARGET concept. Each cell is the
                     segmentation AFTER forcing every token the router assigned to the row
                     concept through the column concept's expert. The re-routed region is
                     outlined in white.

Read it as the intervention matrix made visible:
  - the DIAGONAL (a -> a) is the identity control and must be pixel-identical to the
    baseline. If a diagonal cell differs, the experiment is invalid -- so the figure carries
    its own control, and the caption reports the measured changed-voxel fraction for each.
  - OFF-DIAGONAL cells should show the outlined region change colour to the column concept.
    That is causal control you can see rather than infer.

This figure is impossible for post-hoc attribution: there is nothing in a saliency map to
intervene on. That asymmetry is the argument.

Usage
-----
    python pipeline/intervention_figure.py \
        --checkpoint ~/checkpoints/20260817_212748/epoch_0075.pth \
        --model segresnet --config configs/causal/segresnet_direct.yaml --env gcp \
        --split test --out figures/fig_intervention.png
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent / "data"))
sys.path.insert(0, str(Path(__file__).parent))

from ccr.modules.dispatcher import routing_intervention
from ccrnet.config.phase2_config import Phase2Config
from ccrnet.config.env_config import apply_env
from ccrnet.models.factory import build_model, ARCHES
from ccrnet.utils.checkpoint import load_checkpoint
from brats_dataset import get_dataloader

# Same palette as qualitative_maps.py so the paper's figures stay consistent.
SEG_CMAP = ListedColormap(["#00000000", "#e04141", "#3bb143", "#3b6bdb"])  # bg, NCR, Edema, ET
CONCEPTS = [(1, "NCR"), (2, "Edema"), (3, "ET")]


def _norm(a):
    """Robust min-max to [0,1] (1st-99th pct) so bright outliers do not crush the image."""
    a = a.astype(np.float32)
    fin = a[np.isfinite(a)]
    if fin.size == 0:
        return a
    lo, hi = np.percentile(fin, 1), np.percentile(fin, 99)
    return np.clip((a - lo) / (hi - lo + 1e-8), 0.0, 1.0)


def _tokens_to_voxels(mask_tokens, grid, target_shape):
    """[N] bool over tokens -> [H,W,D] bool over voxels (each token = one nearest block)."""
    m = mask_tokens.reshape(1, 1, *grid).float()
    return F.interpolate(m, size=target_shape, mode="nearest")[0, 0] > 0.5


def main() -> None:
    _cfg = str(Path(__file__).parent.parent / "configs" / "brats_phase2.yaml")
    ap = argparse.ArgumentParser(description="Qualitative causal-intervention figure")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--env", default="local")
    ap.add_argument("--config", default=_cfg)
    ap.add_argument("--model", default="ccrnet", choices=list(ARCHES))
    ap.add_argument("--split", default="test", choices=["val", "test"])
    ap.add_argument("--case_id", default="",
                    help="patient id; default = auto-pick the most concept-balanced case "
                         "among the first --search (a single-concept case shows nothing)")
    ap.add_argument("--search", type=int, default=25, help="cases to scan when auto-picking")
    ap.add_argument("--out", default="figures/fig_intervention.png")
    ap.add_argument("--device", default="")
    args = ap.parse_args()

    config = Phase2Config.from_yaml(args.config)
    apply_env(config, env=args.env, config_dir=Path(__file__).parent.parent / "configs")
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    model = build_model(config, args.model).to(device)
    ep, _ = load_checkpoint(args.checkpoint, model)
    model.eval()
    ccr = getattr(model, "ccr", None)
    if ccr is None:
        raise SystemExit("Model has no CCR module -- nothing to intervene on.")
    print(f"Loaded checkpoint (epoch {ep - 1}) on {device} [{args.model}]")

    loader = get_dataloader(
        data_root=config.data.data_root, split=args.split, batch_size=1,
        num_workers=config.data.num_workers, spatial_size=config.data.spatial_size,
        brats_version=config.data.brats_version, val_fraction=config.data.val_fraction,
        test_fraction=config.data.test_fraction, seed=config.data.seed,
    )

    # --- pick a case: all three concepts must be present or most cells show nothing ---
    best_score, batch, pid = -1.0, None, None
    for i, b in enumerate(loader):
        this_pid = b["patient_id"][0] if isinstance(b["patient_id"], list) else b["patient_id"]
        if args.case_id:
            if this_pid == args.case_id:
                batch, pid = b, this_pid
                break
            continue
        if i >= args.search:
            break
        lab = b["label"][0]
        frac = [float((lab == k).sum()) / lab.numel() for k, _ in CONCEPTS]
        if min(frac) <= 0:
            continue
        score = min(frac) / max(frac)          # 1.0 == perfectly balanced
        if score > best_score:
            best_score, batch, pid = score, b, this_pid
    if batch is None:
        raise SystemExit("No suitable case found (need all three concepts present).")
    print(f"Case: {pid}")

    image = batch["image"].to(device)
    label = batch["label"].to(device)[0]
    shape = tuple(image.shape[2:])

    with torch.no_grad():
        n_tok = model(image)["assignments"].shape[1]
        grid = model.get_grid_shape()
        # The baseline runs through the SAME hard-dispatch path as the interventions (all -1
        # override), so the diagonal is a genuine control and not a comparison against a
        # different dispatch mode.
        null = torch.full((1, n_tok), -1, dtype=torch.long, device=device)
        with routing_intervention(ccr, null):
            out0 = model(image)
        base_pred = out0["seg_logits"].argmax(1)[0]
        a0 = out0["assignments"][0]
        rmap = F.interpolate(a0.reshape(1, 1, *grid).float(), size=shape,
                             mode="nearest")[0, 0].long()

    # slice with the largest tumour cross-section
    z = int(torch.argmax((label > 0).sum(dim=(0, 1))).item())
    t1c = _norm(image[0, 1].cpu().numpy()[:, :, z])
    gt = label.cpu().numpy()[:, :, z]
    bp = base_pred.cpu().numpy()[:, :, z]
    rm = rmap.cpu().numpy()[:, :, z]

    # crop to the tumour bbox (+ margin); on a full slice the change is invisible
    ys, xs = np.where(gt > 0)
    m = 18
    y0, y1 = max(ys.min() - m, 0), min(ys.max() + m, gt.shape[0])
    x0, x1 = max(xs.min() - m, 0), min(xs.max() + m, gt.shape[1])

    def crop(a):
        return a[y0:y1, x0:x1]

    # --- run the 3x3 interventions ---
    cells, diag_err = {}, {}
    for a, aname in CONCEPTS:
        src = (a0 == a)
        for b_id, bname in CONCEPTS:
            if int(src.sum()) == 0:
                cells[(a, b_id)] = (None, None)
                continue
            ov = torch.full((1, n_tok), -1, dtype=torch.long, device=device)
            ov[0, src] = b_id
            with torch.no_grad(), routing_intervention(ccr, ov):
                pred = model(image)["seg_logits"].argmax(1)[0]
            reg = _tokens_to_voxels(src, grid, shape)
            cells[(a, b_id)] = (pred.cpu().numpy()[:, :, z], reg.cpu().numpy()[:, :, z])
            if a == b_id:
                diag_err[aname] = float((pred != base_pred).float().mean())

    # --- render ---
    fig, axes = plt.subplots(4, 4, figsize=(10.5, 10.8))
    for ax in axes.ravel():
        ax.axis("off")

    def show(ax, seg=None, region=None, title=None):
        ax.imshow(np.rot90(crop(t1c)), cmap="gray")
        if seg is not None:
            sm = np.ma.masked_where(crop(seg) == 0, crop(seg))
            ax.imshow(np.rot90(sm), cmap=SEG_CMAP, vmin=0, vmax=3, alpha=0.85,
                      interpolation="nearest")
        if region is not None:
            ax.contour(np.rot90(crop(region)), levels=[0.5], colors="white", linewidths=1.1)
        if title:
            ax.set_title(title, fontsize=10)

    def row_label(ax, text):
        ax.axis("on")
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
        ax.set_ylabel(text, fontsize=11)

    # context row
    axes[0, 0].imshow(np.rot90(crop(t1c)), cmap="gray")
    axes[0, 0].set_title("T1ce", fontsize=10)
    show(axes[0, 1], seg=gt, title="Ground truth")
    show(axes[0, 2], seg=rm, title="Routing (argmax)")
    show(axes[0, 3], seg=bp, title="Prediction (baseline)")

    # 3x3 intervention grid: col 0 carries the row label, cols 1-3 the three targets
    for r, (a, aname) in enumerate(CONCEPTS, start=1):
        row_label(axes[r, 0], f"from {aname}")
        for c, (b_id, bname) in enumerate(CONCEPTS):
            pred, reg = cells[(a, b_id)]
            ax = axes[r, c + 1]
            if pred is None:
                ax.set_title(f"{aname}→{bname}  (absent)", fontsize=9)
                continue
            tag = f"{aname}→{bname}" + ("  (control)" if a == b_id else "")
            show(ax, seg=pred, region=reg, title=tag)

    ctrl = ", ".join(f"{k} {v:.0e}" for k, v in diag_err.items())
    fig.suptitle(
        f"Causal routing intervention - {pid} (axial z={z})\n"
        f"white outline = re-routed region;  diagonal = identity control "
        f"(changed-voxel fraction: {ctrl})",
        fontsize=11, y=0.985)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig.savefig(args.out, dpi=170, bbox_inches="tight")
    print(f"\nSaved: {args.out}")
    print(f"Identity controls (must be 0.0): {diag_err}")


if __name__ == "__main__":
    main()
