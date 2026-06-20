# AMBETS — CCR Research Project

## What this project is

**Clinical Concept Routing (CCR)**: A research paper proposing routing-as-faithful-explanation for 3D medical image segmentation. The routing decision at the encoder bottleneck IS the segmentation prediction, IS the per-voxel explanation, IS the calibrated uncertainty — one forward pass, no post-hoc step.

**Master plan**: `CCR-Net_Research_Plan.md` — read this before touching any code or writing any text.

## Current status (2026-06-20)

**Phase 0 — COMPLETE**
- BraTS data downloaded (official). EDA done (`data/load_brats_sample.py`).
- Novelty confirmed: no concurrent paper routes to clinically-labeled experts at segmentation bottleneck for faithful explanation in 3D dense prediction.
- Comparison tables 13.1–13.4 updated with 2025-2026 papers.
- Introduction drafted (~700 words, NeurIPS style). Related Work drafted (4 subsections).
- Deliverable: `paper_draft_phase0.md`

**Phase 1 — COMPLETE + IMPROVED**
- 136/136 tests pass (7.1s CPU). All 5 Phase 1 improvements implemented and tested.
- Core: CCRBottleneckModule (router + experts + dispatcher), all 5 loss functions, curriculum scheduler, CAS metric, ExpertUtilizationTracker.
- Improvements (2026-05-28):
  - Prototype-augmented router (K learnable concept prototypes, blend_alpha)
  - Foreground-only L_align (BCE on non-background tokens only, k=1..K-1)
  - Focal-BCE for L_align (γ=2.0 by default, down-weights easy tokens)
  - Temperature annealing (τ targets per phase: warmup=2.0, alignment=1.0, refinement=0.5)
  - CAS_fg metric (foreground-only Pearson — primary reported metric)
- New files: `tests/test_metrics.py` (+20 tests for CAS_all, CAS_fg, ExpertUtilizationTracker)
- Documentation: `docs/phase1_architecture.md`, `docs/phase1_guide.md`, `docs/phase1_analysis.md`
- Deliverable: `src/ccr/`, `tests/`

**Phase 2 — COMPLETE + PRETRAINED (2026-06-01)**
- 169/169 tests pass (136 Phase 1 + 33 Phase 2). All model components implemented and tested.
- Architecture: MONAI SwinTransformer (Swin-B 3D) encoder → CCRBottleneckModule at stage-2 → MONAI UNetr decoder → BoundaryRefinementHead.
- CCR insertion: hs[2] [B,192,16,16,16] → reshape [B,4096,192] → CCRBottleneckModule → reshape back → decoder enc3 skip.
- Pretrained weights: `pretrained/model_swinvit.pt` (MONAI SSL, 392MB). 126/126 encoder keys load. Three remappings required (see below).
- Critical fixes: MONAI 1.5+ `img_size` removed; `einops` required; pretrained checkpoint needs key remapping.
- New files: `src/ccrnet/`, `data/brats_dataset.py`, `configs/brats_phase2.yaml`, `pipeline/train.py`, `pipeline/evaluate.py`, `tests/phase2/`, `requirements_phase2.txt`, `pretrained/`, `configs/env/`.
- BraTS data on **E: drive** at `E:\BraTS2024-BraTS-GLI-TrainingData\training_data1_v2`. EDA in `visualizations/BraTS-GLI-00005-100/`. Raw volume: (182,218,182). Labels {0,2,4} → remapped {0,1,2,3}.
- Test/run venv: `C:\Users\pathi\envs\ai_research` (torch 2.10+cu126, monai 1.5.2, einops)

**Pretrained checkpoint remapping (model_swinvit.pt → SwinEncoder3D)**:
1. Strip `module.` prefix (saved with DataParallel)
2. Rename `mlp.fc1` → `mlp.linear1`, `mlp.fc2` → `mlp.linear2` (MONAI 1.5 rename)
3. Inflate `patch_embed.proj.weight` from [48,1,2,2,2] → [48,4,2,2,2] (SSL trained on 1-ch; tile + divide by 1)

**Training commands (env flag selects config):**
```
# Local  (paths from configs/env/local.yaml)
"C:\Users\pathi\envs\ai_research\Scripts\python.exe" pipeline/train.py --env local

# GCP    (paths from configs/env/gcp.yaml)
python pipeline/train.py --env gcp
```

**Pipeline fixes applied (2026-06-08):**
- Bug fixed: `tracker.compute()` was called after `tracker.reset()` → expert utilisation always logged 0. Fixed order.
- Old unrelated folders renamed: `aws-agent/`, `gcp-agent/`, `detector_agent-agent/` — ignore these.
- Data pipeline cross-checked against EDA: filenames, label remap, shape handling all confirmed correct.
- Config generalised for GCP: `--env local/gcp` flag, `configs/env/local.yaml` + `configs/env/gcp.yaml`, `src/ccrnet/config/env_config.py`.

**Run 2 training (20260618_181043) — COMPLETE (ep80, 2026-06-20):**
- NCR CAS peak: 0.706 (ep40). Final ep80: NCR=0.698, Edema=0.901, ET=0.893.
- Dice WT=0.907 (best), TC=0.823, ET=0.812 at ep80.
- Hard ceiling confirmed: 16³ = 10-60 NCR tokens/case. Refinement oscillation: 0.675-0.706.
- Checkpoint: `gs://research-brats/checkpoints/20260618_181043/epoch_0080.pth`

**Run 3 — COMMITTED (acc613a, 2026-06-20), ready to start on GCP after Run 2 finishes:**
- Stage-1 CCR (hs[1], 32³ tokens, embed_dim=96). Fixes structural NCR ceiling: 80-480 tokens/case.
- lam_align=1.0 in refinement (was 0.5). alignment_end_epoch=60.
- All Run 2 loss fixes kept (concept weights [1,3,1,1], γ=1.0, contrastive=0.3).
- **INCOMPATIBLE with Run 1/2 checkpoints** — must train from scratch.
- 169/169 tests pass. Start: `python3 pipeline/train.py --env gcp`

**Next**: Start Run 3 from scratch on GCP immediately → Phase 3 (CCR-Retrofit) after Run 3 confirms NCR CAS ≥ 0.85.

**Phase sequence**: 0 (lit review) → 1 (CCR module) → 2 (CCR-Net training) → **3 (CCR-Retrofit)** → 4 (experiments) → 5 (generalization+agent) → 6 (paper writing) → 7 (submission)

## Directory layout

```
AMBETS/
├── CCR-Net_Research_Plan.md            # Master plan — all decisions, losses, experiments
├── NeuroReAct_Implementation_Plan.md   # Separate unpublished system — DO NOT cite in CCR paper
├── system.md                           # NeuroReAct system doc
├── paper_draft_phase0.md               # Phase 0: novelty table, comparison tables, intro, related work
├── requirements_phase1.txt             # torch, numpy, pytest
├── requirements_phase2.txt             # + monai, nibabel, einops, pyyaml, tqdm, scipy
├── configs/
│   ├── brats_phase2.yaml               # Hyperparameters only (no paths)
│   └── env/
│       ├── local.yaml                  # Local paths (E: drive)
│       └── gcp.yaml                    # GCP bucket paths + instance metadata
├── pipeline/
│   ├── train.py                        # Entry: python pipeline/train.py --env local|gcp
│   └── evaluate.py                     # Eval: python pipeline/evaluate.py --checkpoint ... --env local|gcp
├── data/
│   ├── load_brats_sample.py            # BraTS data loader + EDA script
│   ├── brats_dataset.py                # BraTSDataset + get_dataloader() (Phase 2)
│   ├── BraTS-GLI-00005-100/            # Sample case (BraTS 2023 GLI format)
│   └── requirements.txt
├── src/
│   ├── ccr/                            # Phase 1 deliverable — the CCR module (untouched)
│   │   ├── __init__.py
│   │   ├── config/ccr_config.py        # All hyperparameters (typed dataclasses)
│       ├── modules/
│       │   ├── router.py               # ClinicalConceptRouter
│       │   ├── expert.py               # ClinicalConceptExpert
│       │   └── dispatcher.py           # CCRBottleneckModule
│       ├── losses/
│       │   ├── segmentation.py         # DiceLoss + FocalLoss + SegmentationLoss
│       │   ├── alignment.py            # ConceptAlignmentLoss (L_align — the core loss)
│       │   ├── diversity.py            # ExpertDiversityLoss
│       │   ├── entropy.py              # EntropyRegularizationLoss
│       │   ├── boundary.py             # BoundaryAwareLoss (GPU, no scipy)
│       │   └── total.py                # CCRTotalLoss with curriculum
│       └── utils/
│           ├── weight_schedule.py      # CurriculumWeightScheduler
│           └── metrics.py              # ConceptAlignmentScore, ExpertUtilizationTracker
├── tests/
│   ├── conftest.py
│   ├── test_router.py, test_expert.py, test_losses.py, test_dispatcher.py
│   └── integration/test_pipeline.py
├── docs/
│   ├── phase1_architecture.md          # Dense technical reference (Phase 1)
│   ├── phase1_guide.md                 # Readable narrative guide (Phase 1)
│   ├── phase1_analysis.md              # Theory validation + 5 improvement proposals (Phase 1)
│   ├── phase1_onboarding.md            # Intermediate ML engineer onboarding guide (Phase 1)
│   ├── phase2_architecture.md          # Model design, bottleneck insertion, decoder, training, data (Phase 2)
│   └── experiments.md                  # Complete experiment specs, metrics, success criteria, tables needed
└── visualizations/                     # EDA output plots
```

## Hard constraints — do not violate

- **No efficiency framing**: FLOPs/params reported in supplementary only, never as contributions.
- **Bottleneck placement only**: CCR goes at the encoder bottleneck. NOT at output, NOT at intermediate layers.
- **NeuroReAct is unpublished**: Do not cite it anywhere in the CCR paper. Section 8 uses standalone Gemini 1.5 Pro only.
- **K=4 for BraTS**: {Background, NCR, ED, ET} — matches BraTS annotation protocol exactly.
- **Warm up before L_align**: Do not activate alignment loss until epoch 11. Starting earlier causes expert collapse.
- **λ₃ (entropy) ≤ 0.01**: Larger values push routing toward uniform, destroying CAS and Proposition 2.

## Key metrics to track during training

- **CAS_fg(k)** for k=1,2,3 (NCR, Edema, ET foreground only) — primary metric, target ≥ 0.85
- **CAS_all(k)** — report in supplementary for completeness (includes background)
- **Do NOT report CAS(background)** as evidence of routing quality — low variance in M_0 makes Pearson unreliable
- Expert utilization % — re-init any expert <5% usage after epoch 20
- L_align value — should decrease monotonically after warmup ends
- DD (Decoder Divergence) — report honestly; high DD is not hidden, it is explained

## Formal claims (Propositions — do not relax without theory change)

- **Prop 1a** (CCR-Net): faithfulness = CAS. Exact by construction.
- **Prop 1b** (CCR-Retrofit): faithfulness ≥ CAS × (1 − DD). Bounded, measurable.
- **Prop 2**: Routing entropy ECE < MC-Dropout ECE at T=10, zero overhead. **Empirical claim — requires Phase 4 validation, not derivable from current code.**

## Known theoretical limitations (be aware, do not hide in paper)

- **Soft dispatch means training explanation is not fully causal**: During training, all experts contribute to every token via weighting. The paper's faithfulness claim is about inference-time hard dispatch, not training-time soft routing. Deletion AUC experiments must use hard-routing explanation maps.
- **L_diversity prevents routing redundancy, not functional redundancy**: Expert outputs could still be similar even with orthogonal routing vectors. Functional specialization is driven by L_align, not L_diversity alone. State this clearly in the paper.
- **K=4 imposes hard concept boundaries**: Transition zone tokens are forced into one concept. This is expected and matches BraTS annotation protocol.

## Phase 1 improvements — ALL IMPLEMENTED (2026-05-28)

All 5 improvements are live in `src/ccr/`. 136/136 tests pass.

| # | What | Where | Default |
|---|---|---|---|
| 1 | Foreground-only L_align | `losses/alignment.py` | `foreground_only=True` |
| 2 | Prototype-augmented router | `modules/router.py` | `use_prototypes=True` |
| 3 | Temperature annealing | `utils/weight_schedule.py` + `losses/total.py` | τ: 2.0→1.0→0.5 |
| 4 | Focal-BCE for L_align | `losses/alignment.py` | `focal_gamma=2.0` |
| 5 | CAS_fg metric | `utils/metrics.py` | `cas.compute_fg()` |

To use tau annealing in training loop: `loss_fn(..., tau_current=module.router.temperature)`

## Target venues

1. MICCAI 2026 (8-page, ~Feb 2026 deadline) — submit first
2. NeurIPS 2026 (full version, ~May 2026 deadline) — primary target
