"""
paper_figures.py — the two data figures for the WACV submission.

Both are rendered from committed result files, so the paper's figures and its tables cannot
drift apart. No GPU, no VM.

Fig A  CAS emergence.  V0's routing sits at the floor until L_align switches on at epoch 11.
       DIRECT's climbs during warm-up with L_align OFF, because the routing logits are part
       of the output and L_seg therefore trains them. This is the visual answer to "CAS only
       measures what L_align trains".

Fig B  Intervention matrices.  V2 and DIRECT side by side on a shared colour scale, so the
       ~56x difference in causal control is legible at a glance rather than as a table of
       small numbers. Diagonals are the identity control and are exactly zero.

Usage
-----
    python pipeline/paper_figures.py --out_dir "wacv2027/figures"
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).parent.parent
CONCEPTS = ["necrotic_core", "edema", "enhancing_tumor"]
SHORT = {"necrotic_core": "NCR", "edema": "Edema", "enhancing_tumor": "ET"}

# Colour-blind-safe (Okabe-Ito). WACV's kit explicitly warns that a significant subset of
# reviewers have a colour-vision deficiency, red-green being the most common.
C_V0, C_DIRECT = "#0072B2", "#D55E00"


def _val_series(metrics_path: Path):
    """-> (epochs, {concept: [cas]}) from a training metrics.json."""
    d = json.load(open(metrics_path))
    eps, cas = [], {c: [] for c in CONCEPTS}
    for e in d:
        v = e.get("val")
        if not v or "cas_fg" not in v:
            continue
        eps.append(e["epoch"])
        for c in CONCEPTS:
            cas[c].append(v["cas_fg"][c])
    return np.array(eps), {c: np.array(v, float) for c, v in cas.items()}


def fig_cas_emergence(out_png: Path):
    e0, c0 = _val_series(ROOT / "evals/segresnet_train/metrics.json")       # V0
    e1, c1 = _val_series(ROOT / "evals/causal_direct/metrics_train.json")   # DIRECT

    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.35), sharey=True)
    for ax, c in zip(axes, CONCEPTS):
        ax.axvspan(0, 10, color="0.92", zorder=0)
        ax.plot(e0, c0[c], "o-", color=C_V0, ms=3, lw=1.6, label="V0 (natural)")
        ax.plot(e1, c1[c], "s-", color=C_DIRECT, ms=3, lw=1.6, label="DIRECT (additive)")
        ax.axvline(11, color="0.35", ls="--", lw=1.0)
        ax.set_title(SHORT[c], fontsize=10)
        ax.set_xlabel("epoch", fontsize=9)
        ax.tick_params(labelsize=8)
        ax.set_xlim(0, 80)
        ax.grid(alpha=0.25, lw=0.5)
    axes[0].set_ylabel(r"CAS$_{\rm fg}$", fontsize=9)
    axes[0].set_ylim(-0.08, 1.0)
    # annotate the warm-up band vertically inside it: the top-left of panel 0 is occupied
    # by the y-label, and the curves leave the band's middle free in every panel.
    axes[0].text(5.2, 0.55, "warm-up: $\\mathcal{L}_{\\rm align}$ off",
                 fontsize=7, ha="center", va="center", color="0.35", rotation=90)
    axes[-1].legend(fontsize=7.5, loc="lower right", framealpha=0.95)
    fig.tight_layout()
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    print(f"  {out_png}   (V0 ep10 -> "
          f"{'/'.join(f'{c0[c][e0==10][0]:.2f}' for c in CONCEPTS)} | DIRECT ep10 -> "
          f"{'/'.join(f'{c1[c][e1==10][0]:.2f}' for c in CONCEPTS)})")


def _flip_matrix(summary_path: Path):
    d = json.load(open(summary_path))
    M = np.zeros((3, 3))
    for i, a in enumerate(CONCEPTS):
        for j, b in enumerate(CONCEPTS):
            M[i, j] = d[f"flip_{a}_to_{b}"]["mean"]
    return M


def fig_intervention_matrices(out_png: Path):
    v2 = _flip_matrix(ROOT / "evals/causal_v2_intervention_tumour/test_intervention_summary.json")
    dr = _flip_matrix(ROOT / "evals/causal_direct_intervention_tumour/test_intervention_summary.json")

    # wspace: without it the right panel's y-tick labels collide with the left panel's edge
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.9), gridspec_kw={"wspace": 0.35})
    for ax, M, title in ((axes[0], v2, "V2 (subtractive repair)"),
                         (axes[1], dr, "DIRECT (additive routing)")):
        im = ax.imshow(M, cmap="viridis", vmin=0, vmax=1)   # shared scale: the point is the gap
        ax.set_xticks(range(3)); ax.set_yticks(range(3))
        ax.set_xticklabels([SHORT[c] for c in CONCEPTS], fontsize=8.5)
        ax.set_yticklabels([SHORT[c] for c in CONCEPTS], fontsize=8.5)
        ax.set_xlabel("re-routed TO", fontsize=9)
        ax.set_title(title, fontsize=9.5)
        for i in range(3):
            for j in range(3):
                ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=8.5,
                        color="white" if M[i, j] < 0.55 else "black")
    axes[0].set_ylabel("re-routed FROM", fontsize=9)
    fig.colorbar(im, ax=axes, fraction=0.030, pad=0.02,
                 label="fraction of voxels flipped to target")
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    print(f"  {out_png}   (V2 off-diag mean {v2[~np.eye(3,dtype=bool)].mean():.4f} | "
          f"DIRECT {dr[~np.eye(3,dtype=bool)].mean():.4f}; diagonals "
          f"{np.diag(v2).sum():.1f}/{np.diag(dr).sum():.1f})")


def main():
    ap = argparse.ArgumentParser(description="Render the paper's data figures")
    ap.add_argument("--out_dir", default="wacv2027/figures")
    a = ap.parse_args()
    out = ROOT / a.out_dir
    os.makedirs(out, exist_ok=True)
    print("Rendering:")
    fig_cas_emergence(out / "fig_cas_emergence.png")
    fig_intervention_matrices(out / "fig_intervention_matrices.png")


if __name__ == "__main__":
    main()
