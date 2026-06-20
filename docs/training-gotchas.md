# CCR-Net Training Gotchas

Reference document for starting new runs and designing ablation studies.
Updated after each training run.

---

## Run 1 — 20260616_190302

**Hardware**: GCP `asia-east1-c`, T4 (15GB), 8 vCPU, 30GB RAM
**Config**: batch_size=1, grad_accum_steps=4 (effective batch=4), AMP+grad_checkpointing, num_workers=6

### Results summary at key epochs

| Epoch | Phase | NCR CAS | Edema CAS | ET CAS | Dice WT | Dice TC | Dice ET |
|-------|-------|---------|-----------|--------|---------|---------|---------|
| 2  | warmup | — | — | — | 0.822 | 0.650 | 0.644 |
| 10 | warmup | ~0.0 | ~0.0 | ~0.0 | 0.861 | 0.727 | 0.718 |
| 15 | alignment | 0.398 | 0.861 | 0.840 | 0.872 | 0.740 | 0.737 |
| 30 | alignment | 0.501 | 0.896 | 0.863 | 0.888 | 0.797 | 0.778 |
| 35 | alignment | 0.530 | 0.897 | 0.871 | 0.895 | 0.798 | 0.786 |
| 40 | alignment | **0.530** | 0.900 | 0.871 | 0.893 | 0.807 | 0.793 |
| 60 | refinement | 0.589 | 0.902 | 0.877 | 0.904 | 0.814 | 0.801 |
| 65 | refinement | 0.579 | 0.903 | 0.877 | 0.905 | 0.817 | 0.803 |
| 70 | refinement | 0.579 | 0.900 | 0.876 | 0.905 | 0.810 | 0.797 |
| 75 | refinement | 0.578 | 0.901 | 0.876 | 0.905 | 0.811 | 0.798 |
| **80** | **refinement** | **0.580** | **0.900** | **0.877** | **0.905** | **0.816** | **0.802** |

**Targets**: CAS_fg ≥ 0.85 all three, Dice WT>0.85 / TC>0.75 / ET>0.65

**Final status (ep80)**: Edema ✓, ET ✓, NCR ✗ (0.580 — structural limit). All Dice targets ✓.
**NCR plateau**: 0.530 (ep35-40) → 0.589 (ep60, refinement helped) → 0.580 (ep80 final, Pearson noise).
**Run 1 ceiling**: NCR CAS ~0.58. Fix in Run 2: concept weights 3×NCR + γ=1.0 + contrastive loss.

---

## Gotcha 1 — NCR CAS Plateaus Around 0.53

**Symptom**: `cas_fg_ncr` stops improving after the alignment phase stabilises. Edema and ET continue to climb. NCR stays exactly flat for 5+ consecutive epochs.

**Root causes (in order of fundamentality)**:

### 1. Spatial resolution at bottleneck
Stage-2 CCR runs at 16³ tokens, each covering 8×8×8 voxels. NCR is typically 5,000–30,000 voxels per BraTS case → only **10–60 NCR tokens per case**. This makes Pearson correlation statistically noisy regardless of how well the router is trained — the denominator is too small.

Additionally, the NCR-ET interface tokens physically contain both labels. Nearest-neighbour downsampling assigns one arbitrarily to M_NCR(t), directly injecting label noise into the L_align training signal.

Edema is 10–20× larger → more tokens per case → stable Pearson → CAS converges reliably.

### 2. Class imbalance inside L_align
L_align uses focal BCE per concept independently across foreground tokens. Even within the foreground mask, NCR is the minority. Edema and ET dominate gradient magnitude each batch. The router learns edema and ET routing fast, settles into a stable state, and NCR gets stranded — there simply isn't enough NCR gradient to push further.

### 3. Focal γ=2.0 is wrong for rare-class routing alignment
Focal loss was designed for object detection (rare positives globally). Here:
- NCR tokens are rare → focal loss should upweight them ✓
- But the absolute count of NCR tokens is so small that even with upweighting, total NCR gradient is swamped by the many more edema/ET tokens (which are downweighted but still dominate by count)

### 4. NCR-ET semantic overlap at coarse resolution
T1ce distinguishes NCR (necrotic = dark) from ET (enhancing = bright), but at 8³ voxel resolution the 192-dim Swin features encode a neighbourhood average. The NCR-ET boundary signal exists but is diluted across the token. The router cannot confidently commit to NCR vs ET at this scale.

---

## Does the NCR plateau break paper claims?

**No — but it requires honest reporting.**

| Claim | Affected? | Notes |
|-------|-----------|-------|
| Prop 1a: faithfulness = CAS | ✗ | Holds exactly regardless of CAS magnitude. CAS=0.53 means explanation IS what model computed. |
| Success gate: CAS_fg ≥ 0.85 for NCR | ✓ MISS | Must be reported honestly in paper. |
| Segmentation quality | ✗ | TC Dice=0.807 despite diffuse NCR routing. Decouples segmentation from routing quality. |
| Prop 2: ECE calibration | ✗ | Unrelated. |

**Paper framing (use in limitations section)**:

> "NCR concept alignment is limited by the 16³ bottleneck resolution — each spatial token covers an 8³ voxel neighbourhood, insufficient to cleanly isolate the NCR-ET interface. CAS_fg(NCR) ≈ 0.53 reflects this structural limit, while Dice TC = 0.807 confirms segmentation quality is unaffected. Notably, the faithfulness guarantee holds regardless: even at CAS=0.53, the CCR explanation is exactly what the model computed — not an approximation of it. Higher-resolution bottleneck insertion (stage-1, 32³ tokens, 4³ voxels/token) or multi-scale routing is a direct path to improving NCR alignment."

---

## Fixes for Next Run

### Fix 1 — Per-concept weighting in L_align (implement before Run 2)

Inverse-frequency weighting for NCR tokens in `ConceptAlignmentLoss`.

```python
# src/ccr/losses/alignment.py
# Add concept_weights: Dict[int, float] param
# k=0 (bg): 0.0 (already skipped by foreground_only)
# k=1 (ncr): 3.0  — upweight proportional to inverse frequency
# k=2 (edema): 1.0
# k=3 (et): 1.0
```

Static config in `configs/brats_phase2.yaml`:
```yaml
ccr:
  loss:
    align_concept_weights: [0.0, 3.0, 1.0, 1.0]  # per-concept L_align weight
```

Or compute dynamically per batch from token label frequencies:
`weight[k] = n_foreground_tokens / (n_foreground_concepts * count_k)`

**Expected impact**: NCR gets 3× more gradient from L_align per batch. Directly targets the imbalance without changing architecture.

### Fix 2 — Reduce focal gamma (implement before Run 2)

```yaml
# configs/brats_phase2.yaml
ccr:
  loss:
    align_focal_gamma: 1.0   # was 2.0 — gentler downweighting of easy tokens
```

γ=2.0 was appropriate for dense detection but over-suppresses the many moderate-difficulty NCR tokens. γ=1.0 keeps more of the NCR signal.

---

## Ablation Studies for Paper (Phase 4)

### Ablation A — CCR stage placement

Run CCR at stage-1, stage-2, stage-3, report CAS_fg and Dice for each.

| Stage | Tokens | Voxels/token | NCR tokens/case | Notes |
|-------|--------|-------------|-----------------|-------|
| Stage-1 | 32³ = 32768 | 4³ | 80–480 | **Current (Run 3+)** |
| Stage-2 | 16³ = 4096 | 8³ | 10–60 | Run 1–2 baseline |
| Stage-3 | 8³ = 512 | 16³ | 1–8 | Too coarse for NCR |

**Config for stage-2 (baseline ablation)**: In `src/ccrnet/models/ccrnet.py`, set `_CCR_STAGE = 2`. Set `router.embed_dim = 192` (= 4 × swin.embed_dim) in YAML. This is the Run 1/2 configuration.

### Ablation B — L_align concept weighting

| Config | NCR weight | Notes |
|--------|-----------|-------|
| Uniform | 1.0 | Baseline (Run 1) |
| 3× NCR | 3.0 | Fix 1 above |
| Inverse freq | dynamic | Computed per batch |

### Ablation C — Focal gamma

| γ | Notes |
|---|-------|
| 0.0 | Standard BCE — no focal |
| 1.0 | Mild focal |
| 2.0 | Current (Run 1) |

### Ablation D — Hierarchical routing (Phase 5)

Stage 1: route to {Background, TumorCore, Edema}.
Stage 2: within TumorCore tokens only, route to {NCR, ET}.

Respects BraTS label nesting (TC = NCR ∪ ET). Eliminates edema tokens from the NCR-ET routing decision entirely — expected to significantly improve NCR CAS.

---

## Gotcha 2 — T4 OOM at batch_size=2

**Symptom**: `torch.OutOfMemoryError` during Swin stage-4 forward pass. Softmax allocation fails at ~14.55/14.56GB.

**Fix**: `use_checkpoint: true` in `configs/brats_phase2.yaml` (gradient checkpointing) + `batch_size: 1`. Both changes required simultaneously. On ≥24GB GPU (V100/A100), set `use_checkpoint: false` and `batch_size: 2+` for faster training.

---

## Gotcha 3 — Expert collapse in smoke tests

**Symptom**: Smoke test (2 batches) shows ~99% token routing to one expert (ET). Looks alarming.

**Not a bug**: 2 batches is nowhere near enough for diversity loss to spread experts. Full epoch (1014 batches) resolves this naturally — Run 1 showed balanced utilisation (bg=23%, ncr=20%, edema=25%, et=31%) by epoch 2.

**Real collapse** only matters after epoch 20 (`warmup_end_epoch=10` + buffer). `ExpertUtilizationTracker.check_collapse()` only fires for epoch > 10. Watch `util_ncr` specifically — NCR diffuse routing means it may drift low in refinement phase.

---

## Gotcha 4 — GCS checkpoint sync fails silently

**Symptom**: Checkpoint saved locally, GCS sync step reports `[warn] checkpoint GCS sync failed`.

**Cause**: VM's attached service account has `devstorage.read_only` scope — no GCS writes possible through metadata server credentials regardless of IAM permissions.

**Fix**: Authenticate a separate user account on the instance:
```bash
gcloud auth login --no-launch-browser
```
Then `sync_checkpoint_to_gcs()` in `src/ccrnet/utils/checkpoint.py` uses `gcloud storage cp` which picks up this user token instead of the VM metadata token.

**Per-run GCS path**: `gs://research-brats/checkpoints/<run_id>/epoch_NNNN.pth`

---

## Timing Reference (T4, BraTS 2024 GLI, 1014 train / 168 val)

| Phase | Per-epoch time |
|-------|---------------|
| Training only | ~32 min (1014 batches × 1.89s/batch) |
| Validation (every 5th epoch) | ~3.5 min (168 patients forward-only) |
| Total with validation | ~35.5 min |
| Full 80-epoch run | ~47 hours (~2 days) |

Validation runs every epoch when using `--epochs N` flag. Validation is skipped for non-checkpoint epochs in the full run (`checkpoint_every: 5` in config).

---

## Run 2 — 20260618_181043

**Config changes vs Run 1**: align_focal_gamma 2.0→1.0, align_concept_weights [1,3,1,1] (NCR 3×), contrastive_ncr_et_weight=0.3.

### Run 2 results at key epochs

| Epoch | Phase | NCR CAS | Edema CAS | ET CAS | Dice WT | Dice TC | Dice ET |
|-------|-------|---------|-----------|--------|---------|---------|---------|
| 10 | warmup | ~0.0 | ~0.0 | ~0.0 | 0.867 | 0.740 | 0.735 |
| 15 | alignment | **0.512** | 0.866 | 0.853 | 0.871 | 0.755 | 0.745 |
| 30 | alignment | **0.670** | 0.896 | 0.885 | 0.890 | 0.782 | 0.771 |
| 40 | alignment | **0.706** (peak) | 0.903 | 0.896 | 0.898 | 0.801 | 0.789 |
| 55 | refinement | 0.691 | 0.904 | 0.895 | 0.904 | 0.812 | 0.802 |
| 60 | refinement | 0.675 (drifting) | 0.903 | 0.893 | 0.905 | 0.814 | 0.803 |

**vs Run 1 at ep15**: NCR +28% (0.512 vs 0.398). NCR fixes working from first alignment epoch.
**Refinement problem**: lam_align halved to 0.5 at ep51 → NCR drifted from peak 0.706 (ep40) to 0.675 (ep60). Run 1 got +0.059 boost in refinement; Run 2 got -0.031. Fix in Run 3: keep lam_align=1.0.
**Hard ceiling remains**: 16³ resolution still limits NCR to 10-60 tokens/case. Fix in Run 3: stage-1 (32³).

---

## Run 3 — Config committed 2026-06-20

**Primary goal**: Push NCR CAS from peak ~0.706 (Run 2) to ≥ 0.85 by fixing both root causes.

### Two problems fixed in Run 3

**Problem 1 — Stage resolution limit (primary)**

Stage-2 (16³) gives 10–60 NCR tokens/case → Pearson CAS unreliable.
Stage-1 (32³) gives 80–480 NCR tokens/case → Pearson CAS reliable.
CCR moved from hs[2] (192-dim) to hs[1] (96-dim).

**Problem 2 — Refinement phase NCR drift**

Run 2: lam_align 1.0→0.5 at ep51 caused NCR CAS to drift 0.706→0.675.
Fix: `LossWeights.refinement = (1.0, 0.1, 0.01, 1.0)` — lam_align stays 1.0 all run.
Also: alignment phase extended to ep60 (was ep50).

### Config changes vs Run 2

| Param | Run 2 | Run 3 |
|-------|-------|-------|
| `_CCR_STAGE` | 2 | **1** |
| `router.embed_dim` | 192 | **96** |
| `alignment_end_epoch` | 50 | **60** |
| `LossWeights.refinement[0]` (lam_align) | 0.5 | **1.0** |
| `align_focal_gamma` | 1.0 | 1.0 (unchanged) |
| `align_concept_weights` | [1,3,1,1] | [1,3,1,1] (unchanged) |
| `contrastive_ncr_et_weight` | 0.3 | 0.3 (unchanged) |

### Files changed for Run 3

| File | Change |
|------|--------|
| `src/ccr/config/ccr_config.py` | `LossWeights.refinement`: `(0.5,...)→(1.0,...)`. `CCRConfig.__post_init__` auto-syncs `expert.embed_dim = router.embed_dim` instead of raising. |
| `src/ccrnet/config/phase2_config.py` | `__post_init__` accepts `2×` or `4×` swin.embed_dim (stage-1 and stage-2). |
| `src/ccrnet/models/ccrnet.py` | `_CCR_STAGE = 1`. Passes `ccr_stage` to decoder. |
| `src/ccrnet/models/decoder.py` | Added `ccr_stage: int = 1` param. stage-1: ccr_feat→enc2, hs[2]→enc3. stage-2: old behavior. Channel dims unchanged. |
| `configs/brats_phase2.yaml` | `router.embed_dim: 96`, `alignment_end_epoch: 60`. |
| `tests/integration/test_pipeline.py` | Refinement weight assertion `0.5→1.0`. |
| `tests/phase2/conftest.py` | `RouterConfig(embed_dim=96)`. |
| `tests/phase2/test_ccrnet.py` | Token count formula uses `CCRNet._CCR_STAGE` (stage-aware). |
| `tests/phase2/test_encoder.py` | `test_encoder_stage2_token_count` renamed to `test_encoder_ccr_stage_token_dim` (stage-aware). |

**All 169 tests pass.**

### Decoder channel invariant (why __init__ is unchanged)

- Stage-1 CCR output: 96-dim (=2×f). enc2 has `in_channels=f*2=96` ✓
- Stage-2 CCR output: 192-dim (=4×f). enc3 has `in_channels=f*4=192` ✓
- Only forward() routing changes — no new layers needed.

### IMPORTANT: Run 3 is incompatible with Run 1/2 checkpoints

embed_dim changed 192→96. Cannot resume from Run 1 or Run 2 checkpoints.
Always start Run 3 from scratch:
```bash
python3 pipeline/train.py --env gcp
```
