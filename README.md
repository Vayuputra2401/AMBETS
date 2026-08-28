# Aligned but Inert: Concept Bottlenecks in Dense Prediction, and How to Make Them Causal

Code, results and supplementary material for WACV 2027 submission **#2592** (Applications
track). Anonymized for review: absolute paths, cloud resource names and usernames have been
stripped, so checkpoint references appear as `checkpoints/<run-id>/epoch_XXXX.pth`.

## The paper in one page

Concept bottleneck models promise intrinsic interpretability: route the prediction through a
human-readable concept layer, and the concept activation *is* the explanation. We build one
for dense prediction — a routing bottleneck assigning every encoder token to one of *K*
clinically labeled experts — and train it for 3D brain-tumor segmentation on BraTS 2024.

**By every measure this literature reports, it works.** Routing tracks the clinical concepts
at CAS_fg up to 0.96, and separates necrosis, edema and enhancing tumor at AUROC 0.88 against
0.74 for the strongest post-hoc attribution.

**Yet the routing is causally inert.** Forcing tokens through a different expert changes 1.1%
of voxel labels. Zeroing the bottleneck outright costs at most 3.1 Dice. The decoder
reconstructs the segmentation without ever consulting the concept layer.

Alignment metrics cannot detect this, because alignment is compatible with a bottleneck the
model never consults — and the alignment loss trains precisely what those metrics score. The
cause is not degenerate experts, which compute genuinely distinct transformations, but two
leaks that encoder–decoder architectures invite: a residual path *through* the module, and
skip connections *around* it.

**The repair.** Removing the decoder's alternatives does not work; the optimizer compensates.
Making the routing an *additive term of the output* does:

| | causal control (label-flip) | Dice WT/TC/ET |
|---|---|---|
| NATURAL (residual expert, skips intact) | 0.001 | 0.927 / 0.883 / 0.869 |
| GATED (subtractive repair) | 0.011 `[0.009, 0.014]` | 0.925 / 0.872 / 0.854 |
| **DIRECT (additive routing)** | **0.636** `[0.621, 0.651]` | 0.927 / 0.873 / 0.858 |

Wilcoxon *p* = 2.6 × 10⁻²⁹ over 168 matched cases, for a cost of ≈1 Dice point.

> A concept layer is an explanation only if the model consults it, and showing that requires
> intervening, not correlating.

## Repository map

```
supplementary/       supp.pdf — the supplementary document submitted with the paper
src/ccr/             the CCR module: router, experts, dispatcher, five losses, curriculum
src/ccrnet/          full models — CCR-SegResNet and CCR-Swin/UNETR
pipeline/            training, evaluation, and the three diagnostic tests
configs/             hyperparameters; configs/causal/ holds the three configurations
evals/               every result file the paper and supplement are rendered from
tests/               187 unit and integration tests
```

## The diagnostic protocol

The paper's reusable contribution: three cheap tests that separate a bottleneck the model
uses from one it ignores. None needs retraining.

```bash
# Test 1 — causal intervention. Re-route tokens from concept a through concept b's expert.
python pipeline/routing_intervention.py --checkpoint <ckpt> --model segresnet \
       --config configs/causal/segresnet_direct.yaml --split test

# Test 2 — bottleneck ablation. Each path is ablated separately; see --help.
python pipeline/expert_diagnostics.py  --checkpoint <ckpt> --model segresnet --split test

# Test 3 — linear-probe control. Fit on features ENTERING the module, score as the router is.
python pipeline/linear_probe.py        --checkpoint <ckpt> --model segresnet \
       --fit_split val --split test
```

`routing_intervention.py` defaults to the tumor-restricted scope, which is the number the
paper quotes; pass `--all_tokens` for the whole-volume scope reported alongside it.

## Reproducing the tables

Every number is computed from a committed per-case file under `evals/`. No GPU is needed to
re-derive a table from them.

| Paper | Source |
|---|---|
| Tab. 1 — alignment metrics | `evals/segresnet/`, `evals/segresnet_alignment/`, `evals/test_baselines/` |
| Tab. 2 — causal control | `evals/causal_{v2,direct}_intervention_tumour/` |
| Tab. 3 — bottleneck necessity | `evals/causal_direct_expert_diag/` |
| Tab. 5 — protocol summary | all of the above + `evals/diagnostics_202608/probe_*/` |
| Fig. 2, 3 — data figures | `python pipeline/paper_figures.py` |
| Supp. S6 — ablations | `evals/ablation_a1/`, `evals/ablation_a4/` |
| Supp. S7 — error detection | `evals/diagnostics_202608/error_detection/` |

The headline label-flip is a per-case clustered estimate: mean the off-diagonal cells within
each case, then average across cases, with a bootstrap CI over cases.

## Setup

```bash
pip install -r requirements_phase2.txt      # torch, monai, einops, nibabel, scikit-learn
python -m pytest tests/ -q                  # 228 tests, ~3 min on CPU
```

MONAI 1.5+ is required: several tests exercise API that moved in that release (`img_size`
removed from the Swin constructor, `mlp.fc1/fc2` renamed). An older MONAI fails ~36 tests.

Data is BraTS 2024 adult glioma; the split is patient-wise 75/12.5/12.5 with seed 42, giving
168 test cases. It is our own split, so absolute Dice is not comparable to BraTS leaderboard
results — every comparison in the paper is within-split.

## A note on what we report

Two comparisons in this work go against the method, and both are in the paper rather than
omitted. Post-hoc attribution beats routing on deletion and insertion AUC (supplement §S5).
Plain segmentation confidence predicts model error better than any routing-derived signal we
tried (§S7). The paper's central finding is a negative result about our own system, and the
supplement collects every declared limitation in one place (§S9).
