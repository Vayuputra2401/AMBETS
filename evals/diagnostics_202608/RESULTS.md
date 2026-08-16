# Diagnostic suite — 2026-08, CCR-Net Run 4 (`20260621_120750/epoch_0080.pth`)

**Provenance:** transcribed verbatim from the run stdout on the GCP VM. The raw CSV/JSON
still live on the VM at `~/checkpoints/evals/{test_expert_diag,test_probe,test_probe_a1,
test_errdet,test_intervention}` and should be rsynced in when SSH recovers (it failed
during the pull attempt). These are the printed aggregates, not re-derived numbers.

These four experiments were run to answer "what is CCR actually?" after the causal
intervention came back null. Together they refute the paper's central claim.

---

## 1. Causal routing intervention (`routing_intervention.py`, n=5)

Re-route the tokens assigned to concept *a* through concept *b*'s expert; measure whether
the decoder's posterior over those voxels moves toward *b*.

```
dtarget (Δ posterior of TARGET class in re-routed region)
                   necrotic_     edema  enhancing
  necrotic_core       0.0000    0.0003     0.0002
  edema               0.0009    0.0000     0.0012
  enhancing_tumor     0.0018    0.0014     0.0000

flip (fraction of re-routed voxels whose predicted label became the target)
  max off-diagonal 0.0020   (i.e. 0.2% of voxels)

doff  (off-target classes)   up to 0.0052  -- LARGER than dtarget in several cells
doutside (outside region)    up to 0.0012  -- same magnitude as dtarget
```

Diagonal is exactly 0.0000 (identity control), so the mechanism is correct and the null
is real. **Re-routing does not move the prediction, is not concept-specific, and is not
spatially localized.**

## 2. Expert diagnostics (`expert_diagnostics.py`, n=20)

```
1. Residual scale  ||E_k(h) - h|| / ||h||
     background 0.6147 | necrotic 0.7375 | edema 0.5908 | enhancing 0.7434

2. Pairwise expert divergence  ||E_a(h) - E_b(h)|| / ||h||
                   backgroun  necrotic_     edema  enhancing
  background          0.0000     0.9207    0.8366     0.9250
  necrotic_core       0.9207     0.0000    0.9366     1.0872
  edema               0.8366     0.9366    0.0000     0.8850
  enhancing_tumor     0.9250     1.0872    0.8850     0.0000

3. Bottleneck necessity: per-concept Dice when the CCR output is replaced
                      base    bypass      zero
  necrotic_core     0.2555    0.2327    0.2335
  edema             0.8790    0.8699    0.8645
  enhancing_tumor   0.8620    0.8364    0.8062
```

**The "experts collapsed to identity" hypothesis is REFUTED** — experts move tokens by
60–74% of their norm and are strongly distinct from each other (divergence ~0.9).

**The real cause: zeroing the ENTIRE bottleneck barely changes the output.** Edema
0.879→0.865, ET 0.862→0.806, NCR 0.256→0.234. The UNETR decoder reconstructs the
segmentation from its other encoder skips; the CCR branch is nearly vestigial. This is
architectural, not an optimisation failure.

## 3. Linear probe (`linear_probe.py`, fit on 40 val cases, eval n=168)

Multinomial logistic regression on the same bottleneck features the router sees, scored
through the identical upsample + `concept_auroc_from_volume` path.

```
FULL model (20260621_120750):
  ccr    necrotic 0.854  edema 0.891  enhancing 0.872   mean 0.8721
  probe  necrotic 0.850  edema 0.778  enhancing 0.888   mean 0.8387
  router - probe = +0.0334

A1, no L_align (20260722_202827):
  ccr    necrotic 0.506  edema 0.684  enhancing 0.658   mean 0.6160
  probe  necrotic 0.826  edema 0.753  enhancing 0.855   mean 0.8115
  router - probe = -0.1955
```

**The decisive number is A1's probe: 0.8115.** On a model trained with NO concept
supervision at the bottleneck, a linear probe recovers concept structure at 0.81 —
already above every post-hoc method (best occlusion 0.741). The full CCR stack buys
0.872 vs 0.811, i.e. **+0.06 over probing an ordinary segmentation encoder**. The router
also only beats a linear map on its own features by +0.033, and loses to it on ET.

## 4. Error detection (`error_detection.py`, n=168)

AUROC predicting "the final mask is wrong here", over the tumour neighbourhood
(mean error rate in region 0.171).

```
  disagree_hard      0.7258
  routing_entropy    0.5603
  disagree_soft      0.7863
  seg_confidence     0.8831   <-- free baseline, every model already has it
  best CCR - baseline = -0.0968
```

**The second-opinion claim is dead.** The posterior's own confidence predicts its errors
better than any routing-derived signal.

---

## What these establish, jointly

| Claim in the paper | Status |
|---|---|
| Routing is causally upstream / "the route is the reason" | **Refuted** (1, 2) — the decoder bypasses the bottleneck |
| CCR creates the concept structure | **Largely refuted** (3) — 0.81 of it exists without any concept supervision |
| Routing beats post-hoc on concept-discriminability | **Holds** (0.87 vs 0.74) but is mostly a property of the features, not the architecture |
| Alignment is learned from L_align | **Holds** (A1 router 0.616 vs full 0.872) |
| Routing gives useful free uncertainty | **Refuted** (4), and ECE was already 0.73 |
| Backbone-agnostic | **Holds** (Swin + SegResNet) |

The consistent picture: the CCR module is a **well-trained, concept-aligned side branch
that the decoder largely ignores**. That single fact explains every result — why L_align
can drive CAS to 0.94 without changing Dice, why A1 destroys CAS while Dice barely moves,
why interventions do nothing, and why zeroing the bottleneck is nearly free.
