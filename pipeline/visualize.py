"""
pipeline/visualize.py — Figure 2 generator for WACV 2027 paper.

Generates the 7-panel qualitative figure:
  T1c | GT labels | CCR-Net pred | Routing NCR | Routing Edema | Routing ET | Entropy

Usage
-----
    # Visualize a specific patient directory:
    python pipeline/visualize.py --checkpoint checkpoints/epoch_0080.pth \
        --patient_dir /path/to/BraTS-GLI-00005-100 --env local

    # Auto-select N cases from val split (first N by default):
    python pipeline/visualize.py --checkpoint checkpoints/epoch_0080.pth \
        --n_cases 3 --env local

    # With explicit slice:
    python pipeline/visualize.py ... --slice_idx 64

Outputs (per patient case)
--------------------------
    <output_dir>/<patient_id>/
        figure2_combined.png        — all 7 panels in one row, 300 DPI
        panel_t1c.png
        panel_gt.png
        panel_pred.png
        panel_routing_NCR.png       (or whatever concept_names[1] is)
        panel_routing_Edema.png
        panel_routing_ET.png
        panel_entropy.png

CLI args
--------
    --checkpoint   path to .pth checkpoint (required; gs:// auto-downloads via gsutil)
    --config       path to YAML config (default: configs/brats_phase2.yaml)
    --env          local | gcp
    --patient_dir  path to one BraTS patient directory (overrides --n_cases)
    --n_cases      number of val cases to visualize when --patient_dir not given (default: 3)
    --slice_idx    axial slice index; default: auto (richest label slice along last axis)
    --output_dir   where to save panels (default: visualizations/figure2)
    --device       cuda | cpu
    --dpi          output DPI (default: 300)
"""

from __future__ import annotations

import argparse
import math
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent / "data"))

from ccrnet.config.phase2_config import Phase2Config
from ccrnet.config.env_config import apply_env
from ccrnet.models.ccrnet import CCRNet
from ccrnet.utils.checkpoint import load_checkpoint
from ccrnet.utils.evaluation import upsample_routing_to_volume
from brats_dataset import (
    RemapBraTS2024Labels,
    _modality_filenames,
    get_patient_dirs,
    split_patients,
)


# ---------------------------------------------------------------------------
# Display names for panel titles (concept_names use snake_case internally)
# ---------------------------------------------------------------------------

_DISPLAY_NAMES: Dict[str, str] = {
    "background":       "Background",
    "necrotic_core":    "NCR",
    "edema":            "Edema",
    "enhancing_tumor":  "ET",
}


# ---------------------------------------------------------------------------
# Label overlay colors: NCR=red, Edema=yellow, ET=cyan (alpha=0.70)
# ---------------------------------------------------------------------------

_LABEL_RGBA: Dict[int, np.ndarray] = {
    0: np.array([0.0, 0.0, 0.0, 0.0]),   # Background — transparent
    1: np.array([0.9, 0.1, 0.1, 0.70]),  # NCR
    2: np.array([1.0, 1.0, 0.0, 0.70]),  # Edema
    3: np.array([0.0, 0.9, 0.9, 0.70]),  # ET
}


# ---------------------------------------------------------------------------
# Single-patient loader: deterministic center crop, no augmentation
# ---------------------------------------------------------------------------

def load_patient_for_vis(
    patient_dir: Path,
    spatial_size: Tuple[int, int, int],
    brats_version: str,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    Load and preprocess one BraTS case for visualization.

    Uses CenterSpatialCropd (deterministic) instead of RandCropByPosNegLabeld.

    Works with BOTH training cases (have GT seg file) and official BraTS validation
    cases (no GT seg file — BraTS competition val data has no ground truth).

    Returns
    -------
    image : [4, H, W, D] float32   (channels: t1c, t1n, t2w, t2f)
    label : [H, W, D] int64  OR  None if no seg file exists (official BraTS val)
    """
    from monai.transforms import (
        CenterSpatialCropd,
        Compose,
        CropForegroundd,
        EnsureChannelFirstd,
        EnsureTyped,
        LoadImaged,
        NormalizeIntensityd,
        SpatialPadd,
    )

    image_keys = ["t1c", "t1n", "t2w", "t2f"]
    files = _modality_filenames(patient_dir.name, brats_version)

    # Check whether GT label exists (training split) or not (official BraTS val)
    seg_path = patient_dir / files["label"]
    has_label = seg_path.exists()

    if has_label:
        all_keys = image_keys + ["label"]
        crop_source = "label"
    else:
        all_keys = image_keys
        crop_source = "t1c"   # crop by T1c foreground when no label

    transforms_list = [
        LoadImaged(keys=all_keys),
        EnsureChannelFirstd(keys=all_keys),
    ]
    if has_label and brats_version == "2024":
        transforms_list.append(RemapBraTS2024Labels(keys=["label"]))
    transforms_list += [
        NormalizeIntensityd(keys=image_keys, nonzero=True, channel_wise=True),
        CropForegroundd(keys=all_keys, source_key=crop_source, margin=10),
        SpatialPadd(keys=all_keys, spatial_size=list(spatial_size)),
        CenterSpatialCropd(keys=all_keys, roi_size=list(spatial_size)),
        EnsureTyped(keys=image_keys, dtype=torch.float32),
    ]
    if has_label:
        transforms_list.append(EnsureTyped(keys=["label"], dtype=torch.long))
    transforms = Compose(transforms_list)

    data_dict = {k: str(patient_dir / v) for k, v in files.items() if k != "label" or has_label}
    data_dict["patient_id"] = patient_dir.name
    result = transforms(data_dict)

    image = torch.cat([result[k] for k in image_keys], dim=0)   # [4, H, W, D]
    label = result["label"].squeeze(0).long() if has_label else None
    return image, label


# ---------------------------------------------------------------------------
# Slice and display helpers
# ---------------------------------------------------------------------------

def _best_axial_slice(label: np.ndarray) -> int:
    """Find the axial (last-axis) slice with the most foreground label voxels."""
    per_slice = (label > 0).sum(axis=(0, 1))  # [D]
    return int(np.argmax(per_slice))


def _norm_for_display(arr: np.ndarray) -> np.ndarray:
    """Clip [1st–99th pct of nonzero] and rescale to [0, 1]."""
    nz = arr[arr > 0]
    if len(nz) == 0:
        return np.zeros_like(arr, dtype=np.float32)
    lo, hi = np.percentile(nz, [1, 99])
    return np.clip((arr - lo) / (hi - lo + 1e-8), 0.0, 1.0).astype(np.float32)


def _label_to_rgba(label_2d: np.ndarray) -> np.ndarray:
    """[H, W] int → [H, W, 4] RGBA float32."""
    H, W = label_2d.shape
    rgba = np.zeros((H, W, 4), dtype=np.float32)
    for val, color in _LABEL_RGBA.items():
        rgba[label_2d == val] = color
    return rgba


# ---------------------------------------------------------------------------
# GCS checkpoint download (no-op if path is local)
# ---------------------------------------------------------------------------

def _resolve_checkpoint(path: str) -> str:
    """If path is a gs:// URI, download to a temp file and return local path."""
    if not path.startswith("gs://"):
        return path
    tmp = tempfile.mktemp(suffix=".pth")
    print(f"  Downloading checkpoint from GCS …")
    subprocess.run(["gsutil", "cp", path, tmp], check=True)
    return tmp


# ---------------------------------------------------------------------------
# Panel rendering
# ---------------------------------------------------------------------------

def _render_panels(
    t1c_norm: np.ndarray,                  # [H, W] float32, already in [0,1]
    label_2d: np.ndarray,                  # [H, W] int
    pred_2d: np.ndarray,                   # [H, W] int
    routing_argmax_2d: np.ndarray,         # [H, W] int — hard expert assignment, brain-masked
    routing_maps: Dict[str, np.ndarray],   # {internal_name: [H,W] float in [0,1]}, brain-masked
    entropy_norm: np.ndarray,              # [H, W] float32 in [0,1]
    brain_mask_2d: np.ndarray,             # [H, W] bool — True inside brain tissue
    concept_names_fg: List[str],           # e.g. ["necrotic_core","edema","enhancing_tumor"]
    patient_id: str,
    save_dir: str,
    dpi: int = 300,
) -> str:
    """
    Render all 8 panels as individual PNGs and one combined figure.

    Heatmap/entropy panels are masked to the brain so background leakage and the
    crop-boundary frame do not appear. Panel 4 (Routing argmax) is the hard expert
    assignment in the same palette as GT/Pred — the visual proof of routing=prediction.

    Returns path to the combined PNG.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(save_dir, exist_ok=True)

    # Panel spec: (title, base_img, overlay_rgba_or_None, heatmap_or_None, heatmap_cmap)
    gt_overlay = _label_to_rgba(label_2d) if label_2d is not None else None
    panels = [
        ("T1c",             t1c_norm, None,                            None,  None),
        ("GT",              t1c_norm, gt_overlay,                      None,  None),
        ("CCR-Net Pred",    t1c_norm, _label_to_rgba(pred_2d),         None,  None),
        ("Routing (argmax)",t1c_norm, _label_to_rgba(routing_argmax_2d), None, None),
    ]
    for name in concept_names_fg:
        display = _DISPLAY_NAMES.get(name, name)
        hmap = routing_maps.get(name)
        panels.append((f"Routing {display}", t1c_norm, None, hmap, "hot"))
    panels.append(("Entropy", t1c_norm, None, entropy_norm, "inferno"))

    # --- Individual panels ---
    safe_titles = {
        "T1c":              "panel_t1c.png",
        "GT":               "panel_gt.png",
        "CCR-Net Pred":     "panel_pred.png",
        "Routing (argmax)": "panel_routing_argmax.png",
        "Entropy":          "panel_entropy.png",
    }
    for name in concept_names_fg:
        display = _DISPLAY_NAMES.get(name, name)
        safe_titles[f"Routing {display}"] = f"panel_routing_{display}.png"

    for title, base, ov, hmap, hcmap in panels:
        fig_s, ax_s = plt.subplots(1, 1, figsize=(2.4, 2.4))
        fig_s.patch.set_facecolor("black")
        ax_s.imshow(base, cmap="gray", origin="lower", aspect="equal")
        if ov is not None:
            ax_s.imshow(ov, origin="lower", aspect="equal")
        if hmap is not None:
            hmap_disp = np.ma.masked_where(~brain_mask_2d, hmap)
            ax_s.imshow(hmap_disp, cmap=hcmap, alpha=0.65, origin="lower",
                        aspect="equal", vmin=0.0, vmax=1.0)
        ax_s.axis("off")
        fig_s.tight_layout(pad=0)
        fname = safe_titles.get(title, f"panel_{title.replace(' ', '_')}.png")
        fig_s.savefig(os.path.join(save_dir, fname), dpi=dpi,
                      bbox_inches="tight", facecolor="black", edgecolor="none")
        plt.close(fig_s)

    # --- Combined figure ---
    n = len(panels)
    fig, axes = plt.subplots(1, n, figsize=(n * 2.4, 2.8))
    fig.patch.set_facecolor("black")

    for ax, (title, base, ov, hmap, hcmap) in zip(axes, panels):
        ax.imshow(base, cmap="gray", origin="lower", aspect="equal")
        if ov is not None:
            ax.imshow(ov, origin="lower", aspect="equal")
        if hmap is not None:
            hmap_disp = np.ma.masked_where(~brain_mask_2d, hmap)
            ax.imshow(hmap_disp, cmap=hcmap, alpha=0.65, origin="lower",
                      aspect="equal", vmin=0.0, vmax=1.0)
        ax.set_title(title, fontsize=7, color="white", pad=2)
        ax.axis("off")

    fig.suptitle(patient_id, color="white", fontsize=8, y=1.01)
    fig.tight_layout(pad=0.3)
    combined = os.path.join(save_dir, "figure2_combined.png")
    fig.savefig(combined, dpi=dpi, bbox_inches="tight",
                facecolor="black", edgecolor="none")
    plt.close(fig)

    return combined


# ---------------------------------------------------------------------------
# Per-case inference + visualisation
# ---------------------------------------------------------------------------

@torch.no_grad()
def visualize_case(
    model: CCRNet,
    patient_dir: Path,
    config: Phase2Config,
    device: torch.device,
    output_dir: str,
    slice_idx: Optional[int],
    dpi: int,
) -> Dict:
    """
    Run inference on one patient and generate all Figure 2 panels.

    Returns a summary dict for the calling loop.
    """
    model.eval()

    image, label = load_patient_for_vis(
        patient_dir,
        tuple(config.data.spatial_size),
        config.data.brats_version,
    )

    image_dev = image.unsqueeze(0).to(device)   # [1, 4, H, W, D]
    out = model(image_dev)
    grid_shape = model.get_grid_shape()          # (16, 16, 16)

    # Routing → full volume  [1, K, H, W, D]
    routing_vol = upsample_routing_to_volume(
        out["routing_probs"], grid_shape, tuple(config.data.spatial_size)
    )

    # Entropy → full volume  [H, W, D]
    B, N = out["entropy"].shape
    h, w, d = grid_shape
    entropy_vol = F.interpolate(
        out["entropy"].reshape(1, 1, h, w, d).float(),
        size=tuple(config.data.spatial_size),
        mode="trilinear", align_corners=False,
    )[0, 0]

    pred = out["seg_logits"].argmax(dim=1)[0]   # [H, W, D]

    # Numpy
    image_np   = image.cpu().numpy()              # [4, H, W, D]
    label_np   = label.cpu().numpy()             # [H, W, D]
    pred_np    = pred.cpu().numpy()              # [H, W, D]
    routing_np = routing_vol[0].cpu().numpy()    # [K, H, W, D]
    entropy_np = entropy_vol.cpu().numpy()       # [H, W, D]

    # T1c = channel 0
    t1c_vol = image_np[0]                        # [H, W, D]

    # Auto-select axial slice
    if slice_idx is None:
        if label_np is not None:
            chosen_slice = _best_axial_slice(label_np)
        else:
            # No GT: pick center slice (official BraTS val cases)
            chosen_slice = entropy_np.shape[2] // 2
    else:
        chosen_slice = slice_idx

    # 2D slices along last axis
    t1c_2d     = t1c_vol[:, :, chosen_slice]
    label_2d   = label_np[:, :, chosen_slice] if label_np is not None else None
    pred_2d    = pred_np[:, :, chosen_slice]
    entropy_2d = entropy_np[:, :, chosen_slice]

    # Brain mask: after NormalizeIntensityd(nonzero=True), background/pad is exactly 0.
    # Masking to brain removes the crop-boundary frame and background routing leakage.
    brain_mask_2d = (t1c_2d != 0)

    # Hard routing assignment (argmax over experts), restricted to brain tissue.
    # Background class (0) is transparent in _LABEL_RGBA → only foreground routing shows.
    routing_argmax_2d = routing_np.argmax(axis=0)[:, :, chosen_slice].astype(int)
    routing_argmax_2d[~brain_mask_2d] = 0

    # Entropy normalised to [0, 1], masked to brain
    entropy_norm = (np.clip(entropy_2d / math.log(4), 0.0, 1.0).astype(np.float32)
                    * brain_mask_2d)

    # Routing maps (foreground concepts only: k=1..K-1), masked to brain and
    # renormalised within brain (99th pct) so contrast reflects in-tissue variation.
    concept_names  = config.ccr.concept_names    # ("Background","NCR","Edema","ET")
    fg_names       = list(concept_names[1:])     # ["NCR","Edema","ET"]
    routing_slices = {}
    for k in range(1, len(concept_names)):
        m = routing_np[k, :, :, chosen_slice].astype(np.float32) * brain_mask_2d
        in_brain = m[brain_mask_2d]
        hi = float(np.percentile(in_brain, 99)) if in_brain.size else 1.0
        if hi > 1e-6:
            m = np.clip(m / hi, 0.0, 1.0)
        routing_slices[concept_names[k]] = m

    t1c_norm = _norm_for_display(t1c_2d)

    patient_id = patient_dir.name
    save_dir   = os.path.join(output_dir, patient_id)
    combined   = _render_panels(
        t1c_norm, label_2d, pred_2d,
        routing_argmax_2d, routing_slices, entropy_norm,
        brain_mask_2d, fg_names, patient_id, save_dir, dpi,
    )

    # Quick per-case summary
    if label_np is not None:
        fg_mask  = label_np > 0
        ncr_mask = label_np == 1
        ncr_voxels   = int(ncr_mask.sum())
        entropy_fg   = float(entropy_np[fg_mask].mean()) if fg_mask.any() else 0.0
    else:
        ncr_voxels = -1    # no GT — unknown
        entropy_fg = float(entropy_np.mean())

    summary = {
        "patient_id":      patient_id,
        "slice_idx":       chosen_slice,
        "has_gt":          label_np is not None,
        "ncr_tokens":      ncr_voxels,
        "routing_ncr_max": float(routing_np[1].max()),
        "mean_entropy_fg": entropy_fg,
        "combined_png":    combined,
    }

    _print_case_summary(summary)
    return summary


def _print_case_summary(s: Dict) -> None:
    print(
        f"  {s['patient_id']}  slice={s['slice_idx']}"
        f"  NCR_voxels={s['ncr_tokens']}"
        f"  routing_NCR_max={s['routing_ncr_max']:.3f}"
        f"  entropy_fg={s['mean_entropy_fg']:.3f}"
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    _default_config = str(Path(__file__).parent.parent / "configs" / "brats_phase2.yaml")

    parser = argparse.ArgumentParser(description="CCR-Net Figure 2 generator")
    parser.add_argument("--checkpoint",  type=str, required=True)
    parser.add_argument("--env",         type=str, default="local")
    parser.add_argument("--config",      type=str, default=_default_config)
    parser.add_argument("--patient_dir", type=str, default="",
                        help="Path to one BraTS patient directory. "
                             "Omit to auto-select from val split.")
    parser.add_argument("--official_val", action="store_true",
                        help="Visualise from official BraTS val data (no GT labels). "
                             "Uses brats_official_val_root from env config.")
    parser.add_argument("--n_cases",     type=int, default=3,
                        help="Cases to visualise when --patient_dir not given.")
    parser.add_argument("--slice_idx",   type=int, default=None,
                        help="Axial slice index. Default: auto (richest label slice).")
    parser.add_argument("--output_dir",  type=str, default="visualizations/figure2")
    parser.add_argument("--device",      type=str, default="")
    parser.add_argument("--dpi",         type=int, default=300)
    args = parser.parse_args()

    config = Phase2Config.from_yaml(args.config)
    apply_env(config, env=args.env, config_dir=Path(args.config).parent)

    device = torch.device(
        args.device if args.device
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )

    ckpt_path = _resolve_checkpoint(args.checkpoint)
    model     = CCRNet(config).to(device)
    epoch, _  = load_checkpoint(ckpt_path, model)
    model.eval()
    print(f"Loaded checkpoint (epoch {epoch - 1}) on {device}")

    # Determine which patients to visualise
    if args.patient_dir:
        patient_dirs = [Path(args.patient_dir)]
    elif args.official_val:
        official_root = getattr(config.data, "brats_official_val_root", "")
        if not official_root:
            raise ValueError(
                "brats_official_val_root not set in env config. "
                "Add it to configs/env/local.yaml or configs/env/gcp.yaml."
            )
        patient_dirs = get_patient_dirs(official_root)[: args.n_cases]
        print(f"Using official BraTS val data (no GT): {official_root}")
        print(f"Selected {len(patient_dirs)} cases")
    else:
        all_dirs = get_patient_dirs(config.data.data_root)
        splits   = split_patients(
            all_dirs,
            config.data.val_fraction,
            config.data.test_fraction,
            config.data.seed,
        )
        patient_dirs = splits["val"][: args.n_cases]
        print(f"Auto-selected {len(patient_dirs)} internal val cases (have GT)")

    os.makedirs(args.output_dir, exist_ok=True)
    summaries = []

    for pdir in patient_dirs:
        try:
            s = visualize_case(
                model, pdir, config, device,
                args.output_dir, args.slice_idx, args.dpi,
            )
            summaries.append(s)
        except Exception as exc:
            print(f"  [error] {pdir.name}: {exc}")

    print(f"\nDone. {len(summaries)} case(s) saved to: {args.output_dir}/")
    print("\nPaper case selection guide:")
    print("  Best case  = high routing_NCR_max (>0.60) + correct pred + all 3 regions")
    print("  Hard case  = high entropy_fg (boundary uncertainty)")
    print("  Check combined PNGs to confirm routing maps visually align with GT labels")


if __name__ == "__main__":
    main()
