"""
qualitative_maps.py — the money-shot figure: CCR concept partition vs. baseline sensitivity maps.

For one representative case and axial slice, renders each method's per-concept map
(NCR / Edema / ET) side by side. The visual claim: CCR's three concept maps DIFFER and land on
the correct GT sub-regions (a genuine concept partition), whereas the sensitivity methods
(IG, occlusion, gradient, Grad-CAM) show nearly the SAME high-signal blob for every concept ->
they are not concept explanations. This is the qualitative counterpart to the
concept-discriminability result (Table tab:compare).

Rows = {context, CCR, Grad-CAM, Gradient, IG, Occlusion}; columns = {NCR, Edema, ET}.
Each map cell is min-max normalized for shape (not magnitude), masked to the brain, with the
GT concept-k boundary overlaid.

Usage
-----
    python pipeline/qualitative_maps.py --checkpoint /path/epoch_0080.pth --env gcp --split test
    # optionally pin a case / slice:
    python pipeline/qualitative_maps.py ... --case_id BraTS-GLI-02284-XXX --slice 78
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent / "data"))
sys.path.insert(0, str(Path(__file__).parent))

from ccrnet.config.phase2_config import Phase2Config
from ccrnet.config.env_config import apply_env
from ccrnet.models.ccrnet import CCRNet
from ccrnet.utils.checkpoint import load_checkpoint
from ccrnet.utils.evaluation import upsample_routing_to_volume
from attribution_baselines import GradCAM3D, explanation_maps
from brats_dataset import get_dataloader

METHODS = ["CCR", "Grad-CAM", "Gradient", "IG", "Occlusion"]
MKEY = {"Grad-CAM": "gradcam", "Gradient": "gradient", "IG": "ig", "Occlusion": "occlusion"}
CONCEPTS = [(1, "NCR"), (2, "Edema"), (3, "ET")]
SEG_CMAP = ListedColormap(["#00000000", "#e04141", "#3bb143", "#3b6bdb"])  # bg,NCR,Edema,ET


def _norm(a):
    a = a.astype(np.float32)
    lo, hi = np.nanmin(a), np.nanmax(a)
    return (a - lo) / (hi - lo + 1e-8)


def main() -> None:
    _cfg = str(Path(__file__).parent.parent / "configs" / "brats_phase2.yaml")
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--env", default="local")
    ap.add_argument("--config", default=_cfg)
    ap.add_argument("--split", default="test", choices=["val", "test"])
    ap.add_argument("--case_id", default="")
    ap.add_argument("--slice", type=int, default=-1)
    ap.add_argument("--occ_grid", type=int, default=6)   # finer for a crisp map
    ap.add_argument("--ig_steps", type=int, default=32)
    ap.add_argument("--save_dir", default="")
    ap.add_argument("--device", default="")
    ap.add_argument("--no_sync_gcs", action="store_true")
    args = ap.parse_args()

    config = Phase2Config.from_yaml(args.config)
    env_cfg = apply_env(config, env=args.env, config_dir=Path(__file__).parent.parent / "configs")
    device = torch.device(args.device if args.device else
                          ("cuda" if torch.cuda.is_available() else "cpu"))
    save_dir = args.save_dir or str(Path(config.training.checkpoint_dir) / "evals" / f"{args.split}_qualitative")
    os.makedirs(save_dir, exist_ok=True)

    model = CCRNet(config).to(device)
    ep, _ = load_checkpoint(args.checkpoint, model)
    model.eval()
    gradcam = GradCAM3D(model, model.decoder.enc3)
    print(f"Loaded checkpoint (epoch {ep - 1}) on {device}")

    loader = get_dataloader(
        data_root=config.data.data_root, split=args.split, batch_size=1,
        num_workers=config.data.num_workers, spatial_size=config.data.spatial_size,
        brats_version=config.data.brats_version, val_fraction=config.data.val_fraction,
        test_fraction=config.data.test_fraction, seed=config.data.seed)

    # --- pick a case with all three concepts well represented ---
    chosen = None
    for batch in loader:
        pid = batch["patient_id"][0] if isinstance(batch["patient_id"], list) else batch["patient_id"]
        lab = batch["label"][0].numpy()
        counts = {k: int((lab == k).sum()) for k, _ in CONCEPTS}
        if args.case_id:
            if args.case_id in str(pid):        # substring match (suffix-agnostic)
                chosen = (batch, pid, lab); break
        elif all(c > 300 for c in counts.values()):
            chosen = (batch, pid, lab); break
    if chosen is None:
        raise SystemExit("No suitable case found (need all 3 concepts). Try --case_id.")
    batch, pid, lab = chosen
    print(f"Case: {pid}  concept voxels: " +
          ", ".join(f"{n}={int((lab==k).sum())}" for k, n in CONCEPTS))

    image = batch["image"].to(device)
    H, W, D = image.shape[2:]
    z = args.slice if args.slice >= 0 else int(np.argmax((lab > 0).sum(axis=(0, 1))))
    print(f"Slice z={z}")

    with torch.no_grad():
        out = model(image)
        grid = model.get_grid_shape()
        base = torch.softmax(out["seg_logits"], dim=1)
        ccr_vol = upsample_routing_to_volume(out["routing_probs"], grid, (H, W, D))
        pred = out["seg_logits"].argmax(1)[0].cpu().numpy()

    maps = {"CCR": {k: ccr_vol[0, k].cpu().numpy() for k, _ in CONCEPTS}}
    for disp, key in MKEY.items():
        mm = explanation_maps(key, model, gradcam, image, (1, 2, 3), (H, W, D),
                              baseline_probs=base, occ_grid=args.occ_grid, ig_steps=args.ig_steps)
        maps[disp] = {k: mm[k].cpu().numpy() for k, _ in CONCEPTS}

    t1c = image[0, 0].cpu().numpy()[:, :, z]
    brain = t1c > (t1c.min() + 1e-3)
    gt_sl = lab[:, :, z]
    pred_sl = pred[:, :, z]

    # --- figure: (context + 5 methods) rows x 3 concept columns ---
    nrows = 1 + len(METHODS)
    fig, axes = plt.subplots(nrows, 3, figsize=(7.5, 2.4 * nrows))

    def _panel(ax, img, cmap="gray", seg=False, contour_k=None, title=None, ylabel=None):
        ax.imshow(np.rot90(t1c), cmap="gray")
        if seg:
            m = np.ma.masked_where(np.rot90(img) == 0, np.rot90(img))
            ax.imshow(m, cmap=SEG_CMAP, vmin=0, vmax=3, alpha=0.85)
        elif img is not None:
            disp = np.rot90(_norm(np.where(brain, img, np.nan)))
            ax.imshow(disp, cmap="magma", alpha=0.80)
        if contour_k is not None:
            ax.contour(np.rot90(gt_sl == contour_k), levels=[0.5], colors="cyan", linewidths=0.8)
        if title:
            ax.set_title(title, fontsize=11)
        if ylabel:
            ax.set_ylabel(ylabel, fontsize=11)
        ax.set_xticks([]); ax.set_yticks([])

    # context row: T1c | GT | Pred
    axes[0, 0].imshow(np.rot90(t1c), cmap="gray"); axes[0, 0].set_title("T1ce", fontsize=11)
    axes[0, 0].set_ylabel("input / seg", fontsize=11)
    _panel(axes[0, 1], gt_sl, seg=True, title="Ground truth")
    _panel(axes[0, 2], pred_sl, seg=True, title="Prediction")
    for c in range(3):
        axes[0, c].set_xticks([]); axes[0, c].set_yticks([])

    # method rows
    for r, method in enumerate(METHODS, start=1):
        for c, (k, cname) in enumerate(CONCEPTS):
            _panel(axes[r, c], maps[method][k][:, :, z], contour_k=k,
                   title=(cname if r == 1 else None),
                   ylabel=(method if c == 0 else None))

    fig.suptitle(f"Per-concept explanation maps  (case {pid}, slice {z})   "
                 "— cyan = GT concept boundary", fontsize=12, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.985])
    out_png = os.path.join(save_dir, f"{args.split}_qualitative_{pid}.png")
    fig.savefig(out_png, dpi=170, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_png}")

    gcs = env_cfg.get("paths", {}).get("gcs_checkpoint_dir", "")
    if gcs and not args.no_sync_gcs:
        run_id = Path(args.checkpoint).parent.name
        dst = f"{gcs.rstrip('/')}/{run_id}/evals/{args.split}_qualitative"
        try:
            subprocess.run(["gsutil", "-m", "rsync", "-r", save_dir, dst], check=True)
            print(f"Synced to {dst}")
        except Exception as e:
            print(f"[warn] GCS sync failed: {e}")


if __name__ == "__main__":
    main()
