# What CCR Explains — Positioning & Reviewer Rebuttals

This doc answers the sharpest question a reviewer (or collaborator) asks about CCR:

> *"Routing says 'edema' and the segmentation says 'edema.' Isn't the explanation just the
> prediction restated? How is CCR an explanation at all?"*

It is the honest, defensible framing. Use it when writing/revising the paper and in rebuttals.

---

## 1. Two different things people call "explanation"

1. **A post-hoc map *about* a black box** — Grad-CAM, Integrated Gradients, Occlusion, SHAP.
   The model is opaque; you compute a separate heatmap *afterward* to approximate what it used.
   Always an approximation, and — as our deletion/insertion + concept-alignment results show —
   an unfaithful, non-concept one.
2. **An inherently interpretable *decision*** — the model makes its decision in terms you can
   read directly, so there is nothing to reconstruct. This is Rudin's distinction
   ("stop explaining black boxes; build interpretable models", `rudin2019stop`).

**CCR is type 2, not type 1.** The correct answer to "how is it an explanation" is therefore
**not** "it produces a good heatmap" — it is "the model's decision process is legible by
construction."

## 2. What CCR actually explains

CCR-Net is *architecturally forced* to do this, in order:

1. **commit every region to one named clinical concept** — route each token to one of K experts
   {Background, NCR, Edema, ET};
2. process that region with that concept's **dedicated expert**;
3. produce the mask from the routed features.

It cannot emit a segmentation without first making a concept commitment. So the explanation for
"this region is tumor" is: *"the model routed it to the ET expert with p=0.9 and low entropy
(confident); that routing is what conditioned the decoder to produce this mask."* A
**mechanistic, concept-level account of the decision**, faithful because it *is* the computation.

## 3. "But a plain segmentation softmax also says 'edema' — what's different?"

The real question. The honest difference:

- A plain model's per-class softmax **is the output** — the thing you are trying to explain.
  "It's edema because the softmax says edema" is circular; it gives no access to the *decision*.
- CCR's routing is a **distinct, upstream, modular decision variable**:
  - it lives at the **bottleneck**, causally upstream of the decoder;
  - it is **not identical to the output** — we *measure* this (Decoder Divergence, DD = 0.73 /
    0.53 / 0.40 for NCR / edema / ET; the decoder refines the routing rather than echoing it);
  - each concept is a **separate expert you can inspect and intervene on** — ablate the ET
    expert and watch what breaks (the deletion analysis); read *which* expert fired and how
    confidently. A monolithic softmax gives none of this mechanistic handle.

## 4. CCR's four claims to being an *explanation* (precise)

1. **Faithful by construction** — it is the actual mechanism, provably no post-hoc gap (Prop 1a).
2. **Concept-modular** — separate, inspectable, intervenable experts, not one entangled head.
3. **Free + per-voxel + with uncertainty** — routing entropy is a decision-level uncertainty
   map, one forward pass, no extra cost.
4. **Prototype-grounded** — learnable concept prototypes let you ask what the model's canonical
   "edema" looks like.

## 5. The honest limit (state it, don't hide it)

- CCR explains at the level of clinical **concepts** (which of K experts, how confidently), not
  a finer account of the router's internal reasoning.
- The concept *content* necessarily overlaps the segmentation it conditions.
- Therefore the accurate description is **"an inherently interpretable segmentation model whose
  concept-routing decision is exposed as a faithful, intrinsic explanation"** — *not* "a saliency
  method that out-heatmaps IG." Its advantage over a heatmap is faithfulness + concept labels +
  modularity + free uncertainty, **not** spatial sharpness.

## 6. Why this is exactly the reframe the results force

- **Deletion/insertion AUC rewards input *sensitivity***, which IG/Occlusion optimize directly;
  they win it (≈0.95) *without producing a concept explanation*. It is a sensitivity test on
  their home turf; occlusion is near-circular with it.
- **Concept-alignment** (`pipeline/concept_alignment.py`, foreground AUROC: concept k vs *other
  tumor*) is the axis a concept explanation must win. Hypothesis: CCR ≫ IG/Occlusion, because
  their per-class maps highlight the same high-leverage tumor structure for every class and
  cannot separate NCR / edema / ET. This is the experiment that turns the argument above into a
  measured result.

## 7. Rebuttal-ready answers

- *"Isn't routing just the prediction?"* → No: it is the causally-upstream, modular decision
  variable that *produces* the prediction, provably distinct from it (DD > 0), and it is
  interpretable by construction (Rudin), not a post-hoc reconstruction.
- *"IG/Occlusion beat you on deletion AUC."* → Deletion measures input sensitivity, which those
  methods optimize; occlusion is near-tautological with it. On concept-discriminability — what a
  clinical explanation needs — they collapse toward chance while CCR aligns (report the
  concept-alignment table). We report deletion honestly and do not claim to win it.
- *"Why not a plain segmentation model + post-hoc saliency?"* → Post-hoc saliency on that model
  is unfaithful (our deletion result for Grad-CAM ≈ 0.40) and gives no concept decision; CCR
  gives a faithful, concept-labeled, free, uncertainty-bearing decision, at (per the no-CCR
  control) no accuracy cost.
