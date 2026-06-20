# CCR-Net Post-Edit Checklist

Do this after EVERY code change before committing. Survive auto-compact by reading this first.

---

## After any Python edit in src/ or tests/

```
"C:\Users\pathi\envs\ai_research\Scripts\python.exe" -m pytest tests/ -q --tb=short
```
Expected: **169 passed** (136 Phase 1 + 33 Phase 2). If any fail, fix before committing.

---

## After editing configs/brats_phase2.yaml

- [ ] Does `router.embed_dim` match `_CCR_STAGE`?
  - Stage-1 → embed_dim=96 (=2×swin.embed_dim=48)
  - Stage-2 → embed_dim=192 (=4×swin.embed_dim=48)
- [ ] Does `alignment_end_epoch < total_epochs`? (Currently 60 < 80)
- [ ] Run `Phase2Config.from_yaml("configs/brats_phase2.yaml")` — no ValueError

Quick smoke check:
```
"C:\Users\pathi\envs\ai_research\Scripts\python.exe" -c "
from src.ccrnet.config.phase2_config import Phase2Config
cfg = Phase2Config.from_yaml('configs/brats_phase2.yaml')
print('OK router.embed_dim=', cfg.ccr.router.embed_dim,
      'alignment_end_epoch=', cfg.ccr.curriculum.alignment_end_epoch)
"
```

---

## After editing ccrnet.py or decoder.py

- [ ] `CCRNet._CCR_STAGE` must match `router.embed_dim` in YAML:
  - `_CCR_STAGE=1` → `embed_dim=96`
  - `_CCR_STAGE=2` → `embed_dim=192`
- [ ] `SwinUNETRDecoder(ccr_stage=self._CCR_STAGE)` — decoder must receive the stage param.
- [ ] Run phase2 tests: `pytest tests/phase2/ -q`

---

## After editing ccr_config.py (LossWeights, CCRConfig)

- [ ] `LossWeights.refinement[0]` (lam_align) = 1.0 (Run 3+ design decision)
- [ ] `CCRConfig.__post_init__` auto-syncs `expert.embed_dim = router.embed_dim`
- [ ] All 169 tests pass.

---

## After any change that affects training behavior (loss weights, scheduler, model)

- [ ] Update `docs/training-gotchas.md` with what changed and why.
- [ ] Update `CLAUDE.md` current status section.
- [ ] Update relevant memory file in `C:\Users\pathi\.claude\projects\c--Users-pathi-OneDrive-Desktop-AMBETS\memory\`.

---

## Before every `git commit`

```
git diff --stat          # review what changed
git add <specific files> # never git add -A
git status               # verify no accidental .env / secrets / binaries
```

Commit message format:
```
<short imperative> (50 chars max)

<what changed and why — the research rationale>

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>
```

---

## Session state as of 2026-06-20

| Item | State |
|------|-------|
| Tests | 169/169 pass |
| Run 2 | ep65/80. NCR=0.700 (ep65). Projection ep80: ~0.69-0.72. LR~9e-6. |
| Run 3 | Config committed acc613a. Start from scratch after Run 2. |
| GCP instance | `ccr-research`, asia-east1-c, T4. Local data at `/home/g21cs2026/data/` |
| Run 3 start cmd | `python3 pipeline/train.py --env gcp` |
| Run 3 checkpoint | Incompatible with Run 1/2 (embed_dim 192→96) |
| Next milestone | ep70, ep75, ep80 val for Run 2. Then Run 3. |
| After Run 3 | Phase 3 — CCR-Retrofit (frozen SwinUNETR + CCR, measure DD) |

---

## Key file locations

| What | Where |
|------|-------|
| CCR stage | `src/ccrnet/models/ccrnet.py:_CCR_STAGE` |
| Loss weights | `src/ccr/config/ccr_config.py:LossWeights` |
| YAML config | `configs/brats_phase2.yaml` |
| GCP paths | `configs/env/gcp.yaml` |
| Training entry | `pipeline/train.py` |
| Decoder skip logic | `src/ccrnet/models/decoder.py:forward()` |
| embed_dim assertion | `src/ccrnet/config/phase2_config.py:__post_init__` |
| Run 2 checkpoint | `gs://research-brats/checkpoints/20260618_181043/` |
| Run 3 doc | `docs/training-gotchas.md` (Run 3 section at bottom) |

---

## What NOT to touch

- `src/ccr/` — Phase 1 module (untouched design constraint)
- `aws-agent/`, `gcp-agent/`, `detector_agent-agent/` — old unrelated folders, ignore
- `NeuroReAct_Implementation_Plan.md` / `system.md` — separate system, never cite in CCR paper
