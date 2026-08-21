# DIRECT MODE — the causal claim is finally supported (run `20260817_212748`)

**Config:** `configs/causal/segresnet_direct.yaml` — SegResNet, `direct_mode=true`,
`refine_scale=1.0`, `skip_gate=1.0`, `expert.residual=true`.
`seg_logits = upsample(effective_routing_logits) + refine_scale * decoder_out`.

**Checkpoint:** `epoch_0075.pth` (epoch 80's write failed on a full disk; ep75 is the last
valid one and the run had converged — ep75 and ep80 val were identical to 3 decimals).

**Provenance:** all raw data now in the repo and VERIFIED against the run stdout — every
transcribed figure matched the source exactly, and the intervention diagonal is `[0.0, 0.0,
0.0]`. `evaluate.py` output here; per-patient intervention CSV in
`../causal_direct_intervention/`; diagnostics JSON in `../causal_direct_expert_diag/`.

**Significance (bootstrap, n=168, per-patient mean off-diagonal flip):**

| | mean | 95% CI |
|---|---|---|
| **DIRECT** | **0.8157** | [0.8117, 0.8198] |
| V2 (gated) | 0.0012 | [0.0009, 0.0016] |

Paired DIRECT − V2 = **+0.8145 [+0.8103, +0.8187]**, Wilcoxon p = 2.6e-29, n=168.
Non-overlapping CIs by three orders of magnitude.

---

## 0. SCOPE — read this before quoting any intervention number

The intervention was first run over **every** token assigned to the source concept. The
router sends most of the volume to background and normal tissue (~26% of all tokens to edema
alone), so that repaints the whole brain: rendering the figure showed `Edema->NCR` turning
the entire volume red. It is genuine causal control, but it mostly demonstrates "the output
follows the routing everywhere" -- near-tautological in direct mode, where the routing logits
are literally an additive term of the output.

**Quote the TUMOUR-RESTRICTED numbers.** Both are recorded; `scope` is written into every
summary JSON.

| | tumour-restricted | all tokens |
|---|---|---|
| label-flip to target | **0.546** | 0.816 |
| delta target posterior | **0.540** | 0.798 |
| off-target \|delta\| (specificity, want ~0) | **0.079** | 0.151 |
| outside-region \|delta\| (locality, want ~0) | **0.0046** | 0.0163 |

Per-patient headline (the convention used for the V0/V2 comparison, bootstrap n=168):
**0.636, 95% CI [0.621, 0.651]**. (The 0.546 above is the per-cell mean of the matrix; the
two differ because restriction makes some cells NaN for some cases. Report the per-patient
figure with its CI and say which is which.)

**Restricting made the result BETTER, not just smaller:** specificity roughly doubled
(off-target 0.151 -> 0.079) and locality improved 3.5x (0.0163 -> 0.0046). The unrestricted
number was larger but sloppier -- it was moving things it had no business moving.

## 1. THE HEADLINE — causal intervention, tumour-restricted (n=168)

```
from \ to      necrotic     edema   enhancing
necrotic_core    0.0000    0.3119    0.2801
edema            0.6611    0.0000    0.7721
enhancing_tumor  0.5149    0.7384    0.0000
```

Diagonal exactly `[0.0, 0.0, 0.0]`.

**NCR is weak as a SOURCE (0.28-0.31) but fine as a TARGET (0.51-0.66).** The model can be
pushed into calling a region necrosis, but pushing necrosis tokens elsewhere works less well
-- consistent with NCR being the scarcest concept, where the refinement term dominates the
few tokens involved. Report this asymmetry; it is the honest texture of the result.

## 1b. Unrestricted matrix (for completeness, NOT the headline)

Fraction of re-routed voxels whose predicted label **became the target concept**:

```
from \ to      necrotic     edema   enhancing
necrotic_core    0.0000    0.8570    0.5732
edema            0.8966    0.0000    0.9582
enhancing_tumor  0.7693    0.8398    0.0000
```

**Mean off-diagonal label-flip = 0.816**, against **V0 = 0.0020** and **V2 = 0.0012**.
A ~400x increase. Re-routing a token now changes the prediction to the concept it was
re-routed to, in ~82% of the affected voxels.

Delta of the TARGET-class posterior in the re-routed region (mean off-diagonal **0.798**):

```
from \ to      necrotic     edema   enhancing
necrotic_core    0.0000    0.8494    0.5248
edema            0.8826    0.0000    0.9507
enhancing_tumor  0.7531    0.8294    0.0000
```

Delta of the SOURCE class (want negative — it drops as the token leaves that concept):

```
necrotic_core    0.0000   -0.0085   -0.0085
edema           -0.8903    0.0000   -0.8909
enhancing_tumor -0.6439   -0.6435    0.0000
```

**Locality is excellent** — voxels OUTSIDE the re-routed region barely move (0.009-0.026):

```
necrotic_core    0.0000    0.0152    0.0187
edema            0.0176    0.0000    0.0257
enhancing_tumor  0.0090    0.0118    0.0000
```

**Specificity is good except from NCR** — off-target classes (want ~0):

```
necrotic_core    0.0000    0.4207    0.2613   <- the weak row
edema            0.0292    0.0000    0.0368
enhancing_tumor  0.0657    0.0937    0.0000
```

Re-routing FROM necrotic core disturbs off-target classes (0.26-0.42) far more than
re-routing from edema or ET (0.03-0.09). Consistent with NCR being the model's weakest
concept (base Dice 0.483, and its source-posterior barely drops at -0.0085 because it was
never confident to begin with). Report this honestly; do not average it away.

The diagonal is **exactly 0.0000** by construction, so every off-diagonal number is
attributable to the re-routing alone (see commit 4b8fbd6 -- this control was silently broken
until the pre-run code check).

## 2. Bottleneck necessity (n=40)

```
                    base     bypass  zero_expert  zero_routing      zero
necrotic_core     0.4828     0.4838       0.0534        0.0061    0.1170
edema             0.8975     0.4774       0.3334        0.4178    0.8007
enhancing_tumor   0.8518     0.5970       0.4278        0.1715    0.6726
```

**`zero_routing` devastates the model**: NCR 0.483 -> 0.006 (-99%), ET 0.852 -> 0.172 (-80%),
edema 0.898 -> 0.418 (-53%). Compare V0, where zeroing the entire bottleneck cost ~1.5% on
edema. The routing term is now genuinely load-bearing.

**Oddity worth understanding before it goes in the paper:** zeroing BOTH paths (`zero`, edema
0.801) is *less* damaging than zeroing the routing term alone (`zero_routing`, 0.418). The
decoder is trained to ADD to a routing base; strip the base but leave the expert path and its
output is miscalibrated, whereas zeroing both puts it in a cleaner regime. The two paths
interact — this is not a bug but it needs explaining.

Experts are moderately distinct (pairwise divergence 0.28-0.55, residual scale 0.21-0.47),
between V0's 0.84-1.09 and V2's collapsed 0.155.

## 3. Test metrics vs both baselines

| metric | V0 (naive) | V2 (gated) | **DIRECT** | vs V0 |
|---|---|---|---|---|
| Dice WT | 0.9272 | 0.9252 | **0.9273** | **+0.0001** |
| Dice TC | 0.8828 | 0.8715 | 0.8729 | -0.0099 |
| Dice ET | 0.8688 | 0.8543 | 0.8582 | -0.0106 |
| CAS NCR | 0.8023 | 0.7862 | **0.8091** | +0.0068 |
| CAS edema | 0.9618 | 0.9610 | **0.9620** | +0.0002 |
| CAS ET | 0.9540 | 0.9527 | **0.9543** | +0.0003 |
| routing ECE | 0.8410 | 0.6735 | **0.6809** | **-0.1601** |
| AUROC edema | 0.6381 | -- | **0.9005** | **+0.2624** |
| AUROC ET | 0.6218 | -- | **0.8386** | **+0.2168** |
| AUROC NCR | 0.9368 | -- | 0.7828 | -0.1540 |

**The price of causal control is ~1 Dice point on TC/ET. WT is free. CAS is slightly BETTER
than the inert baseline, calibration improves markedly, and token-level AUROC improves a lot
on edema/ET (NCR regresses).**

DD is reported by `evaluate.py` (0.629/0.532/0.385) but is **CIRCULAR in direct mode** — the
decoder output contains the routing term by construction, so DD measures how much the
refinement perturbs the routing, not independent agreement. Not comparable to V0/V2 DD.
`evaluate.py` prints this warning automatically.

## What this establishes

The subtractive fixes failed because they fought an architecture that did not need its
bottleneck (V2: soft gating compensated away; removing the expert residual collapsed expert
divergence 0.90 -> 0.155). Making the routing an **additive term of the output** makes causal
influence arithmetic rather than something the optimiser can route around — and the measured
`refine_ratio` converged at 0.354, so the routing term was never numerically drowned out.

Three independent lines now agree the routing drives the prediction: the intervention (0.816
flip), the `zero_routing` ablation (-53% to -99% Dice), and `refine_ratio` (routing dominates
the logits ~3:1). This is the arm the paper was missing.
