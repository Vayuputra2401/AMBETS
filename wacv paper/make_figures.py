"""Generate fig5 (W2 window-invariance) and fig6 (Prop 1b bound) from the test summary.

Numbers are from evals/test/test_summary.json (Run 4 ep80, --full, n=168). Run:
    python "wacv paper/make_figures.py"
Style matches fig3/fig4: black bar edges, light y-grid, dotted 0.5 ref.
Colorblind-safe: blue/orange (GT/pred windows), green/gray (measured/bound) + hatch for print.
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({"font.size": 13, "axes.edgecolor": "#333333",
                     "axes.linewidth": 0.9, "figure.dpi": 200})

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")

BLUE, ORANGE = "#4c72b0", "#dd8452"   # GT / predicted window
GREEN, GRAY  = "#2ca02c", "#9aa0a6"   # measured / bound

concepts = ["Edema", "ET", "NCR"]
x = np.arange(len(concepts)); w = 0.38


def _bars(ax, gt, pred, title, ylabel=None):
    b1 = ax.bar(x - w/2, gt,   w, label="GT region", color=BLUE,
                edgecolor="black", linewidth=0.8)
    b2 = ax.bar(x + w/2, pred, w, label="Predicted region", color=ORANGE,
                edgecolor="black", linewidth=0.8, hatch="////")
    ax.axhline(0.5, ls=":", color="black", lw=1.1, zorder=0)
    ax.set_xticks(x); ax.set_xticklabels(concepts)
    ax.set_ylim(0, 1.0); ax.set_title(title)
    ax.grid(axis="y", alpha=0.3, zorder=0)
    if ylabel:
        ax.set_ylabel(ylabel)
    for b in list(b1) + list(b2):
        ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.012,
                f"{b.get_height():.3f}", ha="center", va="bottom", fontsize=9)


# fig5 — window invariance (W2)
fig, axes = plt.subplots(1, 2, figsize=(10, 3.9))
_bars(axes[0], [0.866, 0.815, 0.675], [0.866, 0.819, 0.667],
      "Deletion AUC", ylabel="AUC (higher = more faithful)")
_bars(axes[1], [0.831, 0.754, 0.542], [0.828, 0.730, 0.516], "Insertion AUC")
axes[0].legend(loc="lower left", framealpha=0.95, fontsize=11)
fig.tight_layout(); fig.savefig(f"{OUT}/fig5_window_invariance.png", bbox_inches="tight")
plt.close(fig)

# fig6 — Prop 1b bound: measured deletion AUC (GT) vs CAS*(1-DD)
meas  = [0.866, 0.815, 0.675]
bound = [0.9353*(1-0.5311), 0.9305*(1-0.3948), 0.6876*(1-0.7327)]  # 0.439, 0.563, 0.184
fig, ax = plt.subplots(figsize=(6.6, 4.0))
bm = ax.bar(x - w/2, meas,  w, label="Measured deletion AUC", color=GREEN,
            edgecolor="black", linewidth=0.8)
bb = ax.bar(x + w/2, bound, w, label=r"$\mathrm{CAS}\cdot(1-\mathrm{DD})$ bound",
            color=GRAY, edgecolor="black", linewidth=0.8, hatch="....")
ax.axhline(0.5, ls=":", color="black", lw=1.1, zorder=0)
ax.set_xticks(x); ax.set_xticklabels(concepts)
ax.set_ylim(0, 1.0); ax.set_ylabel("Deletion AUC")
ax.set_title(r"Measured deletion AUC exceeds the $\mathrm{CAS}\cdot(1-\mathrm{DD})$ bound")
ax.grid(axis="y", alpha=0.3, zorder=0)
ax.legend(loc="upper right", framealpha=0.95, fontsize=11)
for b in list(bm) + list(bb):
    ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.012,
            f"{b.get_height():.2f}", ha="center", va="bottom", fontsize=9)
fig.tight_layout(); fig.savefig(f"{OUT}/fig6_prop1b_bound.png", bbox_inches="tight")
plt.close(fig)
print("wrote fig5_window_invariance.png + fig6_prop1b_bound.png")
