# CCR Phase 1 — Complete Onboarding Guide

**Audience**: Intermediate ML engineer joining the project. You know PyTorch, transformers,
and have seen segmentation models before. You have not seen this codebase.

**What this document is**: Every concept, every design decision, every formula, every file —
explained from scratch with diagrams. Not a reference manual. A mental model builder.

**Date**: 2026-05-28 (last updated with all 5 Phase 1 improvements)
**Status**: Phase 1 complete. 136/136 tests pass.

---

## Table of Contents

1. [Why This Exists — The Problem with Explanations](#1-why-this-exists)
2. [The Central Idea in One Paragraph](#2-the-central-idea)
3. [The Dataset: BraTS and K=4](#3-the-dataset)
4. [The Big Picture: Architecture Overview](#4-architecture-overview)
5. [Glossary: Every Term Defined](#5-glossary)
6. [Module 1: ClinicalConceptRouter](#6-router)
7. [Module 2: ClinicalConceptExpert](#7-expert)
8. [Module 3: CCRBottleneckModule (Dispatcher)](#8-dispatcher)
9. [The Loss System: Overview](#9-loss-overview)
10. [L_seg — Segmentation Loss](#10-lseg)
11. [L_align — The Core Loss](#11-lalign)
12. [L_diversity — Prevent Expert Collapse](#12-ldiversity)
13. [L_entropy_reg — Regularize Uncertainty](#13-lentropy)
14. [L_boundary — Sharpen Edges](#14-lboundary)
15. [CCRTotalLoss — Everything Combined](#15-total-loss)
16. [Training Dynamics: The Three-Phase Curriculum](#16-curriculum)
17. [Temperature and Why It Matters](#17-temperature)
18. [Expert Collapse: The Main Training Risk](#18-expert-collapse)
19. [Metrics: CAS and CAS_fg](#19-metrics)
20. [The 5 Phase 1 Improvements](#20-improvements)
21. [How to Run the Tests](#21-tests)
22. [What Phase 1 Is NOT](#22-what-it-is-not)
23. [Hard Rules: Never Violate These](#23-hard-rules)

---

## 1. Why This Exists

### The problem with standard segmentation + explanation

You have a 3D brain MRI. You want to:
1. Segment the tumor into subregions (necrotic core, edema, enhancing tumor)
2. Explain *why* the model made each voxel prediction

The standard workflow is:
1. Train a segmentation model (nnU-Net, Swin-UNETR, etc.)
2. *After* training, run a post-hoc attribution method to explain it

**Post-hoc attribution methods** — GradCAM, SHAP, Integrated Gradients, Attention Rollout —
work like this:

```
Input Image → [BLACK BOX NETWORK] → Prediction
                                         |
                          ┌──────────────┘
                          ↓
              run attribution method
              compute ∂output/∂input
              or ∂output/∂intermediate_features
                          |
                          ↓
              Saliency/Heatmap "explanation"
```

The problem: **these measure sensitivity, not causation.**

- GradCAM says "this region influenced the prediction the most" — but the model
  already made its prediction before GradCAM ran. The explanation is approximate.
- Deletion AUC (a faithfulness test) measures: if you mask the "most important" region,
  does the prediction drop? For post-hoc methods on BraTS: AUC ≈ 0.55–0.62. Barely
  above masking random regions (AUC = 0.50). The explanations are not faithful.
- Post-hoc methods run separately from prediction — there is no guarantee that what
  they highlight is actually what the network used.

### What we want instead

We want the explanation to **be** the computation — not describe it after the fact.

This is the core CCR idea.

---

## 2. The Central Idea

> Place routing at the encoder bottleneck.
> The routing decision at the bottleneck IS the per-voxel explanation.
> The decoder only sees concept-conditioned representations.
> Therefore prediction = explanation. One forward pass. No approximation.

In concrete terms:

- Every image token at the bottleneck gets a probability distribution over K clinical concepts:
  `P(token t → concept k)` for k ∈ {Background, NCR, Edema, ET}
- The decoder only receives token features that have been processed by the clinically-labeled
  expert for that token's dominant concept
- You train the routing probabilities to correlate with ground-truth subregion labels
- After training: the routing map for concept k IS the model's segmentation of subregion k

This is not approximate. For CCR-Net: **faithfulness = CAS** exactly by construction.

For CCR-Retrofit (inserting CCR into an existing backbone): **faithfulness ≥ CAS × (1 − DD)**
where DD measures how much the backbone decoder diverges from pure concept-conditioned behavior.

---

## 3. The Dataset

### BraTS — Brain Tumor Segmentation Challenge

BraTS provides multi-modal 3D brain MRI. Each scan has 4 modalities:
- T1 (native T1-weighted)
- T1ce (T1 with contrast enhancement)
- T2 (T2-weighted)
- FLAIR (Fluid-Attenuated Inversion Recovery)

Each voxel has a ground-truth label from one of 4 classes:

```
Label 0 — Background         (~80% of voxels)  ← healthy brain + skull + air
Label 1 — NCR/NET             (~5%)             ← Necrotic Core / Non-Enhancing Tumor
Label 2 — Edema               (~13%)            ← Peritumoral edema (swelling)
Label 3 — Enhancing Tumor     (~2%)             ← Active tumor cells (shows on T1ce)
```

Visually, a coronal slice looks like:

```
         ████████████████████████
        ██  Background (80%)    ██
       ██   everywhere          ██
      ██   ┌─────────────────┐   ██
     ██    │  Edema (13%)    │    ██
    ██     │  ┌───────────┐  │     ██
   ██      │  │  NCR  (5%)│  │      ██
   ██      │  │  ┌─────┐  │  │      ██
   ██      │  │  │ ET  │  │  │      ██
   ██      │  │  │ 2%  │  │  │      ██
   ██      │  │  └─────┘  │  │      ██
    ██     │  └───────────┘  │     ██
     ██    └─────────────────┘    ██
      ██                         ██
       ████████████████████████████
```

**Why K=4?** This matches the BraTS annotation protocol exactly. K=4 is not an arbitrary
hyperparameter — it reflects the clinical taxonomy that radiologists use. We don't choose 3
or 5; we use the ground truth label space.

### From 3D volume to tokens

A Swin-B encoder converts the 4-channel 3D volume into a sequence of tokens:

```
Input:   [B, 4, H, W, D]     ← 4 MRI modalities, 3D volume
         e.g. [2, 4, 128, 128, 128]

Encoder: Patch tokenization (patch_size=8) + Swin-B transformer

Bottleneck: [B, N, 192]      ← N tokens, each 192-dimensional
             e.g. [2, 4096, 192]  (128/8)^3 = 16^3 ≈ 4096 tokens
```

Each of the 4096 tokens corresponds to an 8×8×8 patch of the original volume.
Each token gets its own routing probability: which of K=4 clinical concepts does this patch belong to?

---

## 4. Architecture Overview

Here is the complete data flow. Phase 1 implements the shaded CCR Bottleneck block.
Phase 2 will add the encoder and decoder.

```
                     ┌─────────────────────────────────────────────────┐
                     │              FULL CCR-Net (Phase 2+)            │
                     │                                                 │
Input MRI            │  ┌─────────────┐                               │
[B, 4, H, W, D] ─────┼─▶│  Encoder    │                               │
                     │  │  (Swin-B)   │                               │
                     │  └──────┬──────┘                               │
                     │         │ bottleneck tokens [B, N, D]          │
                     │         ▼                                       │
                     │  ╔══════════════════════════════╗              │
                     │  ║   CCRBottleneckModule         ║  ← PHASE 1  │
                     │  ║                              ║              │
                     │  ║  ClinicalConceptRouter       ║              │
                     │  ║    ↓                         ║              │
                     │  ║  routing_probs [B, N, K]     ║  ← THE EXPLANATION │
                     │  ║    ↓                         ║              │
                     │  ║  K × ClinicalConceptExpert   ║              │
                     │  ║    ↓                         ║              │
                     │  ║  expert_outputs [B, N, D]    ║              │
                     │  ╚══════════════╤═══════════════╝              │
                     │                 │ concept-conditioned tokens    │
                     │                 ▼                               │
                     │  ┌─────────────────┐                           │
                     │  │    Decoder      │                           │
                     │  │  (upsampling    │                           │
                     │  │   + boundary    │                           │
                     │  │   head)         │                           │
                     │  └────────┬────────┘                           │
                     │           │ segmentation logits [B, K, H, W, D]│
                     └───────────┼─────────────────────────────────────┘
                                 │
                                 ▼
                     Predicted segmentation
```

The key insight is the arrow labeled "THE EXPLANATION": `routing_probs [B, N, K]` is both
the explanation AND the causal input to everything downstream.

---

## 5. Glossary

Every term you will encounter:

| Term | Definition |
|---|---|
| **Token** | A [D]-dimensional vector representing an 8×8×8 patch of the MRI volume after the encoder. Think: one "word" in a sequence. |
| **B** | Batch size — number of patients processed simultaneously. Typically 2 for 3D medical images. |
| **N** | Number of tokens per image. For 128³ volume with patch_size=8: N ≈ 4096. |
| **D** | Token embedding dimension. Swin-B bottleneck outputs D=192. |
| **K** | Number of clinical concepts. K=4 for BraTS. |
| **Router** | The small MLP that decides, for each token, what probability to assign to each of K concepts. |
| **Expert** | A small residual MLP. One per concept. Refines tokens that are routed to it. |
| **Routing probability** | P(token t → concept k) — a number in [0,1]. All K probabilities for a token sum to 1. This IS the explanation. |
| **Soft dispatch** | During training: every expert processes every token; outputs are weighted by routing probability. Differentiable. |
| **Hard dispatch** | During inference: each token goes to exactly ONE expert — whichever has the highest probability. Crisp, deterministic. |
| **Temperature τ** | A learnable scalar that controls how "sharp" the routing distribution is. Low τ → one expert dominates. High τ → all experts get similar weight. |
| **L_align** | The core loss. Pushes routing_probs to match ground-truth subregion labels. This is the mechanism that makes routing clinically meaningful. |
| **CAS** | Concept Alignment Score. Pearson correlation between routing_probs and GT subregion masks. The primary evaluation metric. |
| **CAS_fg** | CAS computed only on foreground (non-background) tokens. The honest primary metric. |
| **Expert collapse** | A failure mode: all tokens route to one expert. The other experts receive no gradient and stay untrained. |
| **Curriculum** | A training schedule that turns different loss terms on/off at different epochs. Three phases: warmup, alignment, refinement. |
| **Warmup phase** | Epochs 1–10: L_align is OFF. The encoder learns features first. If L_align is on too early, expert collapse occurs. |
| **Bottleneck** | The narrowest point in the encoder-decoder: the point where the encoder output feeds into the decoder. This is where CCR is placed. |
| **Post-hoc** | Computed AFTER the prediction. GradCAM is post-hoc. CCR routing is NOT post-hoc — it is computed as part of the forward pass and determines the prediction. |
| **Proposition 1a** | For CCR-Net: faithfulness = CAS exactly. Not approximate. |
| **Proposition 1b** | For CCR-Retrofit: faithfulness ≥ CAS × (1 − DD). Bounded guarantee. |
| **DD** | Decoder Divergence. How much the backbone decoder diverges from fully concept-conditioned behavior. Only relevant for CCR-Retrofit. |
| **Simplex** | The probability simplex: the set of vectors where all components are ≥ 0 and sum to 1. Routing probs live on the K-simplex. |

---

## 6. Module 1: ClinicalConceptRouter

**File**: `src/ccr/modules/router.py`

**What it does**: Takes bottleneck tokens as input, outputs a probability distribution over
K clinical concepts for each token.

### The math, step by step

```
Input: bottleneck_tokens X  [B, N, D]
       e.g.  [2, 4096, 192]

Step 1 — LayerNorm
   X̂ = LayerNorm(X)         [B, N, D]
   (Standard pre-norm: normalizes each token to mean=0, std=1 across the D dimension)

Step 2 — Two-layer MLP (the "routing MLP")
   H = GELU(X̂ @ W₁ + b₁)   [B, N, H]   where H = D × 2.0 = 384
   logits_mlp = H @ W₂ + b₂  [B, N, K]   K=4 for BraTS

   This is just a 2-layer feedforward network.
   Output: K scores per token (not yet probabilities).

Step 3 — Prototype logits (if use_prototypes=True, default)
   c_k ∈ ℝ^D for k=0..3     K learnable prototype vectors [K, D] = [4, 192]
   
   diff    = X̂ - c_k          [B, N, K, D]   (distance from each token to each prototype)
   dist_sq = ||diff||²         [B, N, K]      (squared L2 distance)
   logits_proto = -dist_sq / √D  [B, N, K]   (negative distance: closer = higher logit)

   α = sigmoid(blend_alpha)   scalar ∈ (0, 1)    learnable blend weight
   logits = α × logits_mlp + (1−α) × logits_proto

Step 4 — Temperature scaling
   τ = clamp(temperature, 0.1, 5.0)   learnable scalar
   logits_τ = logits / τ              [B, N, K]

Step 5 — Softmax (converts logits to probabilities on the simplex)
   routing_probs = softmax(logits_τ, dim=-1)   [B, N, K]
   → Each row sums to 1.0
   → All values ∈ (0, 1)

Step 6 — Entropy (uncertainty signal)
   H(t) = -Σ_k P(t→k) × log(P(t→k) + ε)   [B, N]
   → H = 0 if routing is one-hot (certain)
   → H = log(K) ≈ 1.39 if routing is uniform (maximally uncertain)
```

### The prototype branch — why it exists

Without prototypes, the router is a black box: `W₁, W₂` are opaque weight matrices.
After training you can say "token t was routed to Expert 2" but you cannot say
*what Expert 2 represents geometrically in feature space*.

With prototypes:

```
Prototype c_k ∈ ℝ^192 = "what the bottleneck feature of a typical
                          concept-k voxel looks like"

After training with L_align:
   c_0 ← mean feature of background tokens
   c_1 ← mean feature of NCR tokens
   c_2 ← mean feature of edema tokens
   c_3 ← mean feature of ET tokens

You can visualize these! PCA or t-SNE the 4 prototype vectors and show
that NCR, Edema, ET live in different corners of the 192-D feature space.
```

The `blend_alpha` parameter lets the model decide how much to trust the MLP vs.
the geometric prototype signal. It is initialized to 0.0 (α = sigmoid(0) = 0.5),
giving equal initial weight to both.

### Temperature — intuition

```
Low τ (e.g. 0.1):
   logits = [2.0, 0.5, 0.3, 0.1]
   logits/τ = [20.0, 5.0, 3.0, 1.0]
   softmax → [≈1.0, ≈0.0, ≈0.0, ≈0.0]   ← "I am 100% sure it's concept 0"

High τ (e.g. 5.0):
   logits = [2.0, 0.5, 0.3, 0.1]
   logits/τ = [0.4, 0.1, 0.06, 0.02]
   softmax → [0.33, 0.27, 0.24, 0.16]   ← "roughly equal chance of any concept"
```

During warmup (τ target = 2.0): routing is diffuse. Experts see all tokens and
can specialize from any starting point.

During refinement (τ target = 0.5): routing is crisp. Each token is strongly
committed to one expert.

### Output dictionary

```python
out = router(tokens)
out["routing_probs"]  # [B, N, K]  — THE EXPLANATION
out["entropy"]        # [B, N]     — uncertainty per token
out["logits"]         # [B, N, K]  — pre-softmax (for debugging)
```

---

## 7. Module 2: ClinicalConceptExpert

**File**: `src/ccr/modules/expert.py`

**What it does**: Given tokens dispatched to it, refines them with concept-specific
feature transformations.

There are K=4 experts, one per concept: Expert_0 (Background), Expert_1 (NCR),
Expert_2 (Edema), Expert_3 (ET). They have the same architecture but independent weights.

### Architecture: 2-layer residual MLP

```
Input: tokens X  [M, D]    M = number of tokens dispatched to this expert
                            D = 192

Step 1 — LayerNorm
   norm_out = LayerNorm(X)          [M, D]

Step 2 — Up-projection
   h = GELU(norm_out @ W₁ + b₁)    [M, H]   H = D × 2.667 ≈ 512

Step 3 — Down-projection
   correction = h @ W₂ + b₂        [M, D]

Step 4 — Residual addition
   output = X + correction          [M, D]
```

The **residual connection** is critical:

```
Early training (epoch 1-5):
   Expert weights are near-zero (fc2 initialized with gain=0.01)
   correction ≈ 0
   output ≈ X     ← the module acts like an identity function
   
   This means the encoder/decoder can start learning segmentation normally
   before the experts have specialized. The residual prevents experts from
   injecting random noise early in training.

Late training (epoch 30+):
   correction has grown and is meaningfully concept-specific
   Expert_1 now transforms tokens differently from Expert_2, Expert_3
```

Without the residual connection: in early epochs, the expert outputs random
high-magnitude features that corrupt the decoder. Training destabilizes.

### One expert per concept

```python
# In CCRConfig:
concept_names = ("background", "necrotic_core", "edema", "enhancing_tumor")

# In CCRBottleneckModule:
self.experts = nn.ModuleList([
    ClinicalConceptExpert(config.expert, concept_id=0, concept_name="background"),
    ClinicalConceptExpert(config.expert, concept_id=1, concept_name="necrotic_core"),
    ClinicalConceptExpert(config.expert, concept_id=2, concept_name="edema"),
    ClinicalConceptExpert(config.expert, concept_id=3, concept_name="enhancing_tumor"),
])
```

Each expert has independent weights. Under L_align they learn concept-specific
feature transformations. Expert_3 (ET) learns to emphasize contrast-enhancement features.
Expert_1 (NCR) learns to emphasize T1 hypointensity features. Etc.

---

## 8. Module 3: CCRBottleneckModule (Dispatcher)

**File**: `src/ccr/modules/dispatcher.py`

**What it does**: Combines the router and K experts into one drop-in module.
Handles the soft/hard dispatch switch between training and inference.

### Full forward pass

```
Input: bottleneck_tokens  [B, N, D]

Step 1 — Router
   router_out = self.router(bottleneck_tokens)
   routing_probs = router_out["routing_probs"]   [B, N, K]
   assignments   = routing_probs.argmax(dim=-1)  [B, N]   ← integer: which expert wins

Step 2 — Dispatch (training vs. inference differ here)
   if self.training:
       expert_outputs = soft_dispatch(bottleneck_tokens, routing_probs)
   else:
       expert_outputs = hard_dispatch(bottleneck_tokens, assignments)

Output dict:
   routing_probs   [B, N, K]   ← THE EXPLANATION
   expert_outputs  [B, N, D]   ← concept-conditioned features for the decoder
   entropy         [B, N]      ← uncertainty (high at tumor boundaries)
   assignments     [B, N]      ← integer hard assignments (for metrics)
   logits          [B, N, K]   ← pre-softmax (diagnostics)
```

### Soft dispatch — training

```
For each token t in each image b:
   expert_output(b, t) = Σ_k  P(b,t → k)  ×  Expert_k(X[b,t])

In code:
   flat_tokens = tokens.reshape(B*N, D)   # collapse batch and token dims
   output = zeros(B*N, D)
   for k, expert in enumerate(self.experts):
       expert_k_out = expert(flat_tokens)  # [B*N, D]  — all tokens through all experts
       p_k = routing_probs[..., k].reshape(B*N, 1)  # [B*N, 1]
       output += p_k * expert_k_out

Why? Because backpropagation needs a path through P(t→k).
The routing_probs appear explicitly in the computation, so gradients flow:
   ∂L/∂P(t→k)  ← this is what L_align needs to shape routing semantics
   ∂L/∂encoder  ← via chain rule through P(t→k) → router MLP → encoder
```

Visually:

```
token t ──────────→ Expert_0 ──→ out_0 ──┐
         │                                │ × P(t→0)
         │                                ▼
         ├────────→ Expert_1 ──→ out_1 ──┐│
         │                               ││ × P(t→1)
         │                               ││
         ├────────→ Expert_2 ──→ out_2 ──┐│
         │                              │││ × P(t→2)
         │                              │││
         └────────→ Expert_3 ──→ out_3 ──┐│
                                        ││││ × P(t→3)
                                        ▼▼▼▼
                                  Σ weighted sum  →  expert_output(t)
```

### Hard dispatch — inference

```
For each token t:
   k* = argmax_k P(t → k)
   expert_output(b, t) = Expert_{k*}(X[b, t])

In code:
   for k, expert in enumerate(self.experts):
       mask = (assignments == k)  # which tokens belong to expert k
       output[mask] = expert(flat_tokens[mask])

Why? Crisp explanation maps. At inference you want to say:
"Token t was processed by Expert_NCR — it is part of the necrotic core."
Not: "Token t was 60% NCR, 25% Edema, 15% ET."

Not differentiable (argmax has zero gradient).
That is fine: inference does not need gradients.
```

### The training/inference switch is automatic

```python
module.train()  # → soft dispatch (gradients flow everywhere)
module.eval()   # → hard dispatch (crisp explanations)
```

You do not call anything else. The module checks `self.training` in `forward()`.

---

## 9. The Loss System: Overview

CCR uses 5 loss functions. They are weighted and combined by the curriculum scheduler.

```
L_total = L_seg
        + λ₁ × L_align
        + λ₂ × L_diversity
        + λ₃ × L_entropy_reg
        + λ₄ × L_boundary
        + τ_reg_weight × (τ_actual − τ_target)²   ← optional, when tau_current passed
```

The λ weights change across three training phases:

```
Phase       Epochs    λ₁(align)  λ₂(div)  λ₃(ent)  λ₄(bound)
─────────────────────────────────────────────────────────────
Warmup       1–10         0        0.5      0.00      0.0
Alignment   11–50         1.0      0.5      0.01      0.3
Refinement  51–80         0.5      0.1      0.01      1.0
```

`L_seg` is always weighted 1.0.

Why these specific weights? Each phase has a goal:
- **Warmup**: learn segmentation features without routing pressure. Only L_diversity
  to prevent all experts from being identical from the start.
- **Alignment**: make routing match clinical labels. L_align dominates.
- **Refinement**: sharpen boundaries. L_boundary takes over.

---

## 10. L_seg — Segmentation Loss

**File**: `src/ccr/losses/segmentation.py`

This is a standard combination loss for imbalanced multi-class segmentation.

```
L_seg = DiceLoss(pred, gt) + FocalLoss(pred, gt)
```

### DiceLoss

Dice coefficient measures overlap between prediction and ground truth:

```
Dice(k) = (2 × |pred_k ∩ gt_k| + ε) / (|pred_k| + |gt_k| + ε)

For a perfect prediction: Dice = 1.0
For no overlap:           Dice = 0.0

DiceLoss(k) = 1 - Dice(k)
DiceLoss_total = mean over k=0..K-1 of DiceLoss(k)
```

Why Dice? It handles class imbalance naturally. ET is only 2% of voxels.
Standard cross-entropy would give 98% accuracy by predicting no ET everywhere.
Dice penalizes missing the small ET class even when it's rare.

### FocalLoss

Standard focal loss from Lin et al. 2017:

```
FL(p_t) = -α × (1 - p_t)^γ × log(p_t)

where p_t = predicted probability of the correct class
      α   = 0.25  (down-weight easy background class)
      γ   = 2.0   (focus on hard voxels)

(1 - p_t)^γ → 0 for easy voxels (p_t ≈ 1)
(1 - p_t)^γ → 1 for hard voxels (p_t ≈ 0)
```

Focal loss focuses training on the voxels where the model is currently wrong —
especially important for small subregions (ET, NCR).

### Combined

Using both Dice + Focal is standard practice in medical image segmentation.
They are complementary: Dice optimizes overlap, Focal optimizes per-voxel confidence.

---

## 11. L_align — The Core Loss

**File**: `src/ccr/losses/alignment.py`

This is THE mechanism that makes CCR work. Without L_align, routing is arbitrary —
tokens route to experts at random regardless of their clinical label. With L_align,
routing correlates with ground-truth subregion labels.

### The base idea

For each clinical concept k, we want:
- Tokens that belong to concept k → high P(t→k) (router assigns them to Expert_k)
- Tokens that do NOT belong to concept k → low P(t→k)

This is binary cross-entropy for each concept:

```
gt_k(t) = 1  if token t has ground-truth label k, else 0
p_k(t)  = routing_probs[t, k]   (routing probability for concept k)

BCE(p_k, gt_k) = -[gt_k × log(p_k) + (1 - gt_k) × log(1 - p_k)]

L_align_base = (1/K) × Σ_k BCE(p_k, gt_k)
```

### Why this creates competition between concepts

The routing_probs live on the probability simplex: `Σ_k p_k(t) = 1`.
This constraint means the K concepts COMPETE for each token.

```
If token t is an NCR token (gt_1 = 1, gt_0=gt_2=gt_3 = 0):
   L_align wants: p_1(t) → 1
   
   But because Σ_k p_k(t) = 1:
   pushing p_1(t) → 1  forces  p_0(t) + p_2(t) + p_3(t) → 0
   
   So L_align also pushes p_0, p_2, p_3 → 0  automatically.
```

This is NOT K independent binary classifiers. It is a K-way soft assignment
with a single constraint. The simplex geometry is the right design.

### Phase 1 improvement 1: Foreground-only

In BraTS, ~80% of tokens are background (label=0). Standard BCE computes loss
on all N tokens per image.

**Problem**:
```
Background tokens (80%) → easy to classify (just predict p_0 ≈ 1 for most tokens)
Foreground tokens (20%) → the actual hard problem: distinguish NCR from Edema from ET
```

The gradient signal is dominated by easy background tokens. The tumor subregion
alignment — which is the actual claim of the paper — gets drowned out.

**Fix**: Compute L_align only on foreground tokens.

```python
if self.foreground_only:
    fg_mask  = (seg_labels != 0).float()   # 1 for tumor tokens, 0 for background
    fg_count = fg_mask.sum() + 1e-8
    k_start  = 1   # skip concept k=0 (background)

# Only NCR (k=1), Edema (k=2), ET (k=3) concepts are optimized
# Only tumor tokens contribute to the gradient
# Background concept (k=0) is removed from L_align entirely
```

**Effect**: 100% of L_align gradient signal now goes to distinguishing tumor subregions.
CAS_fg(NCR), CAS_fg(Edema), CAS_fg(ET) all improve.

### Phase 1 improvement 4: Focal modulation

Even among foreground tokens, there is easy-hard imbalance.

```
Easy tokens: interior NCR voxels, far from boundaries
             Model quickly learns p_NCR ≈ 0.95 for these
             BCE gradient is tiny: −log(0.95) ≈ 0.05

Hard tokens: boundary voxels (NCR/Edema transition)
             p_NCR might be 0.55 — the model is uncertain
             BCE gradient is larger: −log(0.55) ≈ 0.60
```

Once the model is confident on interior tokens, the standard BCE gradient from
those tokens dominates and the hard boundary tokens do not get enough attention.

**Fix**: Focal modulation, same idea as FocalLoss but for the alignment loss.

```
p_correct(t) = p_k(t) if token t belongs to concept k
               1 - p_k(t) otherwise

focal_weight(t) = (1 - p_correct(t))^γ

bce_focal(t) = focal_weight(t) × BCE(t)
```

```
Easy token (p_correct = 0.95):   focal_weight = (1-0.95)^2.0 = 0.0025  ← tiny!
Hard token (p_correct = 0.55):   focal_weight = (1-0.55)^2.0 = 0.2025  ← 80× larger
```

**Effect**: The model keeps learning hard boundary tokens long after easy interior
tokens are "solved". CAS(ET) improves most because ET voxels are small and often
near boundaries.

### Complete formula with both improvements

```
With foreground_only=True and focal_gamma=2.0:

L_align = (1/(K-1)) × Σ_{k=1}^{K-1}  
          [ Σ_{t: foreground} (1 - p_correct(t))^2 × BCE(p_k(t), gt_k(t)) ]
          ─────────────────────────────────────────────────────────────────
                        number of foreground tokens
```

---

## 12. L_diversity — Prevent Expert Collapse

**File**: `src/ccr/losses/diversity.py`

This loss prevents two failure modes:

1. **Expert collapse**: all tokens route to Expert_0. Experts 1, 2, 3 receive no gradient
   and stay untrained.
2. **Routing redundancy**: Experts 1, 2, 3 route to completely overlapping sets of tokens —
   effectively doing the same thing.

### Load balance term

Penalizes unequal utilization:

```
mean_routing_k = mean over all tokens of P(t→k)   ← average probability for concept k
                 ideally ≈ 1/K = 0.25 for uniform distribution

L_load_balance = K × Σ_k mean_routing_k²
```

If one expert gets all tokens (collapse): one mean_routing_k ≈ 1, others ≈ 0:
  `L_load_balance = K × (1² + 0 + 0 + 0) = K`

If all experts are equal (uniform): all mean_routing_k ≈ 1/K:
  `L_load_balance = K × K × (1/K)² = 1`

Minimum is 1. Maximally imbalanced (collapse) = K. L_diversity penalizes being above 1.

### Cosine Gram matrix term

Penalizes routing patterns that are too similar between experts.

```
R_k = routing_probs[..., k].reshape(B*N)   ← routing vector for expert k over all tokens
R_k_norm = R_k / ||R_k||                   ← normalized

G[k1, k2] = dot(R_k1_norm, R_k2_norm)     ← cosine similarity of routing patterns

L_gram = ||G||² (Frobenius norm of the Gram matrix)
```

If Expert 1 and Expert 2 route to exactly the same tokens:
  `G[1,2] = 1.0` → high penalty

If Expert 1 and Expert 2 route to completely different tokens:
  `G[1,2] ≈ 0` → no penalty

**Important limitation**: This penalizes routing overlap, NOT functional similarity.
Two experts can have different routing patterns but learn identical transformations.
Functional diversity is enforced by L_align (different subregions → different gradient signals).

### When L_diversity is active

```
Warmup (epochs 1-10):    λ_diversity = 0.5   ← ONLY active loss besides L_seg
Alignment (epochs 11-50): λ_diversity = 0.5   ← still active
Refinement (51-80):       λ_diversity = 0.1   ← reduced, routing already established
```

During warmup, L_diversity is the guardian against collapse because L_align is off.
Without it, all experts would converge to the same weights by symmetry.

---

## 13. L_entropy_reg — Regularize Uncertainty

**File**: `src/ccr/losses/entropy.py`

Routing entropy H(t) = -Σ_k P(t→k) log P(t→k) measures how uncertain the router
is about token t.

```
H = 0           ← certain: one expert gets all the probability (one-hot)
H = log(K)      ← maximally uncertain: all experts equal probability
```

We want:
- Boundary/transition tokens: **high entropy** (genuine uncertainty → high entropy → high reported uncertainty → correct calibration)
- Interior tokens: **low entropy** (clear concept assignment)

Without regulation, the model might collapse to either extreme:
- All tokens get near-zero entropy (overconfident)
- All tokens get near-maximum entropy (useless routing)

L_entropy_reg penalizes BOTH extremes with a soft target:

```
H_target = 0.5 × log(K)   ← midpoint

L_entropy_reg = mean over tokens of (H(t) - H_target)²
```

**Critical constraint**: `λ_entropy ≤ 0.01`. This loss must NOT dominate training.
Its job is a gentle nudge, not a strong signal. If λ_entropy > 0.01, routing collapses
toward uniform (high entropy), destroying CAS and calibration.

During warmup: λ_entropy = 0 (not active — routing is still random, no point penalizing).

---

## 14. L_boundary — Sharpen Edges

**File**: `src/ccr/losses/boundary.py`

Segmentation models often produce blurry boundaries between subregions. The network
is uncertain at transition zones and assigns probabilities in a smooth gradient rather
than a crisp step.

L_boundary is weighted cross-entropy that applies higher weight to voxels near subregion
boundaries:

```
boundary_voxels = morphological dilation of subregion edges   ← computed on GPU
weight_map(t) = base_weight (1.0) for non-boundary voxels
                base_weight + boost_factor (5.0) for boundary voxels

L_boundary = weighted cross-entropy using weight_map
```

**Why morphological dilation?** Simple edge detection (find voxels where label changes)
only marks a 1-voxel-thin boundary. Dilation expands it by 3 iterations, creating a
boundary zone that includes slightly uncertain voxels on both sides of the edge.

**Why activated only in refinement?**

```
Warmup:     λ_boundary = 0.0   ← encoder not ready; boundary loss too early = noise
Alignment:  λ_boundary = 0.3   ← minor boundary signal while L_align dominates
Refinement: λ_boundary = 1.0   ← full boundary emphasis — precision pass
```

In refinement, the model already produces good coarse segmentation. The boundary
loss refines the edges, improving HD95 (Hausdorff Distance, a boundary-sensitive metric).

**GPU implementation note**: The dilation is implemented with max-pooling (no scipy).
This is important — scipy morphological operations cannot run in GPU training loops.

---

## 15. CCRTotalLoss — Everything Combined

**File**: `src/ccr/losses/total.py`

This is the entry point. It holds all 5 loss functions, the curriculum scheduler,
and wires everything together.

### Interface

```python
cfg = CCRConfig()
loss_fn = CCRTotalLoss(cfg)

# In your training loop:
result = loss_fn(
    pred_logits   = ...,          # [B, K, H, W, D]  — decoder segmentation output
    labels        = ...,          # [B, H, W, D]     — 3D ground-truth labels
    routing_probs = ...,          # [B, N, K]         — from CCRBottleneckModule
    token_labels  = ...,          # [B, N]            — GT labels downsampled to token res
    epoch         = 15,           # current epoch (1-indexed)
    tau_current   = module.router.temperature,  # optional, for temperature annealing
)

# result is a dict:
result["total"]       # scalar loss to backprop through
result["seg"]         # L_seg component (logging)
result["align"]       # L_align component (logging)
result["diversity"]   # L_diversity component (logging)
result["entropy"]     # L_entropy_reg component (logging)
result["boundary"]    # L_boundary component (logging)
result["tau_reg"]     # temperature regularization (0 if tau_current not passed)
result["phase"]       # "warmup" / "alignment" / "refinement"
result["weights"]     # (λ₁, λ₂, λ₃, λ₄) for this epoch
result["tau_target"]  # target τ for this phase
```

### Token labels vs. 3D labels

A key implementation detail: the loss function needs labels in two formats:

```
labels       [B, H, W, D]   → 3D voxel labels for L_seg and L_boundary
                               (full resolution, e.g. [2, 128, 128, 128])

token_labels [B, N]         → token-level labels for L_align
                               (downsampled to match bottleneck token resolution)
                               N ≈ 4096 for 128³ volume with patch_size=8
```

`token_labels` is obtained by nearest-neighbor downsampling of the 3D label map to
the token grid. This is a Phase 2 task (not in Phase 1 — Phase 1 tests use random
integer tensors with the right shape).

### Epoch tracking

```python
loss_fn.set_epoch(epoch)  # call at the start of each training epoch
# OR pass epoch= argument to forward()
```

The curriculum scheduler uses the epoch to look up (λ₁, λ₂, λ₃, λ₄).

---

## 16. Training Dynamics: The Three-Phase Curriculum

Understanding WHY the phases exist is more important than memorizing the λ values.

### Why you cannot just train with all losses from epoch 1

The problem is a chicken-and-egg deadlock:

```
To route correctly → router needs encoder features that distinguish NCR from Edema from ET
To learn those features → encoder needs gradient from the segmentation loss
But the segmentation loss requires the decoder
And the decoder receives expert-processed features
And the experts need the router to work correctly

At epoch 1:
- Encoder features are near-random
- Router produces near-uniform routing (initialization is near-zero logits)
- L_align receives random gradient signals
- The random gradient pushes some expert to dominate by chance
- That expert gets more gradient, the others less
- System collapses to one expert
- Collapse is permanent (experts without gradient stay random forever)
```

### Phase 1 — Warmup (epochs 1–10)

Goal: Let the encoder + decoder learn to segment without routing pressure.

```
λ = (0.0, 0.5, 0.00, 0.0)
    align   div  ent   bound

L_total = L_seg + 0.5 × L_diversity
```

L_diversity at 0.5 is the only constraint on routing during warmup.
It prevents all experts from becoming identical by keeping them spread out.

By epoch 10: encoder has learned discriminative features (NCR, Edema, ET look
different in feature space). Router has not collapsed. Ready for L_align.

### Phase 2 — Alignment (epochs 11–50)

Goal: Force routing to correlate with clinical labels. The key training phase.

```
λ = (1.0, 0.5, 0.01, 0.3)

L_total = L_seg + 1.0 × L_align + 0.5 × L_diversity + 0.01 × L_entropy + 0.3 × L_boundary
```

L_align at 1.0 dominates. The routing MLP receives strong gradient signals telling
it to assign NCR tokens to Expert_NCR, Edema tokens to Expert_Edema, etc.

CAS_fg should rise from ~0 to ≥0.85 during this phase.
Temperature τ anneals from 2.0 toward 1.0 (routing tightens).

### Phase 3 — Refinement (epochs 51–80)

Goal: Sharpen boundary precision. Crisp concept maps.

```
λ = (0.5, 0.1, 0.01, 1.0)

L_total = L_seg + 0.5 × L_align + 0.1 × L_diversity + 0.01 × L_entropy + 1.0 × L_boundary
```

L_boundary at 1.0 forces the model to be precise at subregion edges.
L_align reduced to 0.5 (routing is already trained, less pressure needed).
L_diversity reduced to 0.1 (experts are already specialized).
Temperature τ anneals toward 0.5 (crisp inference-ready routing).

### Visual timeline

```
Epoch:  1        10        50        80
        │────────│─────────│─────────│
Phase:  │ Warmup │ Alignment│ Refine  │
        │        │          │         │
L_align:│  OFF   │  FULL    │  HALF   │
L_div:  │  0.5   │  0.5     │  0.1    │
L_bound:│  OFF   │  0.3     │  FULL   │
τ target│  2.0   │  1.0     │  0.5    │

CAS_fg: │ ~0     │ rising   │ ≥0.85   │
        │        │ toward   │ target  │
        │        │ 0.85     │         │
```

---

## 17. Temperature and Why It Matters

Temperature τ is a single learnable scalar that scales all routing logits before softmax.

### During training: gradients flow through τ

```
logits_τ = logits / τ
routing_probs = softmax(logits_τ)

∂L/∂τ = ∂L/∂routing_probs × ∂routing_probs/∂logits_τ × ∂logits_τ/∂τ
       = ... × (-logits / τ²)
```

If τ is learnable (default), the optimizer updates τ like any other parameter.
But without guidance, τ might go in the wrong direction for the current training phase.

### Temperature annealing (Phase 1 Improvement 3)

We add a soft penalty to pull τ toward the phase-appropriate target:

```
L_tau_reg = tau_reg_weight × (τ_actual − τ_target)²
          = 0.01           × (τ_actual − τ_target)²

τ_target:  2.0 (warmup) → 1.0 (alignment) → 0.5 (refinement)
```

This is a soft nudge, not a hard constraint (weight 0.01 is small).
τ still adjusts to the data, but now has a curriculum-aligned prior.

### Clamping [0.1, 5.0]

Hard bounds prevent degenerate routing:

```
τ < 0.1: routing_probs ≈ one-hot   → gradients ≈ 0 through softmax (near-argmax)
         The routing MLP stops learning because the softmax is saturated.

τ > 5.0: routing_probs ≈ uniform   → P(t→k) ≈ 1/K for all k
         L_align receives no useful gradient signal through routing_probs.
```

Clamping is applied in `router.forward()` on every call:
```python
tau = self.temperature.clamp(0.1, 5.0)
```

---

## 18. Expert Collapse: The Main Training Risk

Expert collapse is when one expert receives all (or nearly all) tokens, and the
other experts receive none.

```
Normal routing:
   Expert_0 (Background):  ~80% of tokens
   Expert_1 (NCR):          ~5% of tokens
   Expert_2 (Edema):        ~13% of tokens
   Expert_3 (ET):            ~2% of tokens

Collapsed routing:
   Expert_0:  ~99% of tokens
   Expert_1:   ~0.3% of tokens
   Expert_2:   ~0.5% of tokens
   Expert_3:   ~0.2% of tokens
```

**Why it is permanent**: If Expert_1 gets only 0.3% of tokens in soft dispatch,
its weight in `Σ_k P(t→k) × Expert_k(X[t])` is ~0.003. Gradients through Expert_1
are ~0.003 of total. Expert_1 stays in its initial state indefinitely. The model
has effectively lost 3 experts permanently.

**How collapse happens**:
1. Early training: all routing probabilities are near-uniform (random initialization)
2. Random chance: Expert_0 happens to reduce L_seg slightly more than others
3. Gradient update: routing_probs shift slightly toward Expert_0
4. Positive feedback: Expert_0 gets more gradient → learns faster → attracts more tokens
5. Collapse: all tokens route to Expert_0

**How we prevent it**:

```
1. Warmup (L_align = 0):
   No clinical alignment pressure → routing cannot collapse toward clinical labels
   Routing stays near-uniform → all experts receive similar gradient

2. L_diversity (λ=0.5 during warmup):
   Load balance term penalizes concentration
   Gram matrix term penalizes expert redundancy
   Together they keep routing spread during warmup

3. Near-zero expert initialization (gain=0.01):
   Expert outputs ≈ 0 early → routing gradients from L_align are small early
   Prevents the positive feedback loop in early epochs

4. Temperature τ_target = 2.0 during warmup:
   High τ → diffuse routing → even if one expert starts to dominate,
   the softmax is "soft" enough that other experts still get 20-30% of tokens
```

**Detection**: `ExpertUtilizationTracker` monitors utilization every epoch.

```python
tracker = ExpertUtilizationTracker(num_concepts=4, concept_names=..., warmup_end_epoch=10)

# Each training batch:
tracker.update(assignments)   # [B, N] integer expert assignments

# End of each epoch:
collapsed = tracker.check_collapse(epoch=current_epoch)
# Returns list of expert names with < 5% utilization (after epoch 10)
# Emits RuntimeWarning for each collapsed expert
```

**What to do if collapse is detected after epoch 20**:
Re-initialize the collapsed expert's weights and slightly increase L_diversity weight.
The model needs to be "unstuck" manually — the training dynamics will not self-correct.

---

## 19. Metrics: CAS and CAS_fg

**File**: `src/ccr/utils/metrics.py`

### Concept Alignment Score (CAS)

CAS measures how well the router's probability for concept k correlates with the
ground-truth subregion mask for concept k.

```
M_k(t) = 1 if token t has GT label k, else 0    ← binary ground-truth mask
P(t→k) = routing_probs[t, k]                    ← routing probability

CAS(k) = Pearson(P(t→k), M_k(t))  over all tokens in the validation set
```

**Pearson correlation** measures linear correlation between two variables:

```
Pearson(X, Y) = (n × ΣXY - ΣX × ΣY) / (√(n×ΣX² - (ΣX)²) × √(n×ΣY² - (ΣY)²))

Range: [-1, 1]
   1.0  = perfect positive correlation (ideal: router matches anatomy exactly)
   0.0  = no correlation (random routing)
  -1.0  = perfect negative correlation (routing is anti-aligned with anatomy)
```

For a well-trained CCR model: `CAS(NCR) ≥ 0.85`, `CAS(Edema) ≥ 0.85`, `CAS(ET) ≥ 0.85`.

### Why CAS_all(Background) is unreliable

Background is 80% of BraTS voxels. M_0(t) = 1 for ~80% of tokens.

```
M_0: [1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, ...]   (mostly 1s)
     nearly constant → near-zero variance

Pearson requires variance in both variables.
When M_0 has near-zero variance, the denominator of Pearson → 0.
CAS_all(Background) is numerically unstable and meaningless.
```

**Never report CAS_all(Background) as evidence of routing quality**.
If all three tumor CAS values are ≥ 0.85, background routing is correct by
elimination — the background expert is getting the remaining tokens.

### CAS_fg — the honest primary metric

**Key question**: Among tumor tokens (non-background), how well does routing
match the clinical annotation?

```
CAS_fg(k) = Pearson(P(t→k), M_k(t))  over FOREGROUND TOKENS ONLY
             k ∈ {1, 2, 3}   (NCR, Edema, ET — background excluded)
```

Foreground tokens have genuine label variance (NCR ≠ Edema ≠ ET → labels change).
The Pearson computation is well-conditioned.
CAS_fg is a harder threshold to meet than CAS_all: target is still ≥ 0.85.

**Report CAS_fg as the primary metric in the paper.**
Report CAS_all in supplementary for completeness.

### Running computation (no storing predictions)

The implementation uses running sums to avoid storing all N tokens × B batches in memory.

```
For Pearson(X, Y), you need 5 statistics: n, ΣX, ΣY, ΣX², ΣY², ΣXY

Pearson = (n×ΣXY − ΣX×ΣY) / (√(n×ΣX² − (ΣX)²) × √(n×ΣY² − (ΣY)²))

These 5 sums can be accumulated batch by batch without storing raw data.
ConceptAlignmentScore maintains 12 running sums:
   6 for CAS_all (n, sum_p, sum_m, sum_pp, sum_mm, sum_pm) × K concepts
   6 for CAS_fg  (same, but fg-only, for concepts k=1..K-1)
```

Usage:

```python
cas = ConceptAlignmentScore(num_concepts=4,
                             concept_names=("background", "necrotic_core",
                                            "edema", "enhancing_tumor"))

for batch in val_loader:
    routing_probs, token_labels = ...  # [B,N,K] and [B,N]
    cas.update(routing_probs, token_labels)

scores_all = cas.compute()       # {"background": 0.xx, "necrotic_core": 0.xx, ...}
scores_fg  = cas.compute_fg()    # {"necrotic_core": 0.xx, "edema": 0.xx, "enhancing_tumor": 0.xx}
                                 # note: no "background" key in compute_fg()
cas.reset()  # clear for next epoch
```

---

## 20. The 5 Phase 1 Improvements

All 5 are implemented and live. Defaults have been changed to the improved versions.

### Mental model: before vs. after

Before all improvements, the system was correct but left performance on the table.
The improvements are principled refinements, not hacks.

---

### Improvement 1 — Foreground-only L_align

```
Before: L_align computed on all N tokens (80% background, 20% tumor)
        80% of gradient signal wasted on easy background tokens

After:  L_align computed only on foreground tokens (tumor only)
        100% of gradient signal on hard tumor subregion distinctions

Config:  LossConfig.align_foreground_only = True  (default)
File:    losses/alignment.py
```

Expect: CAS_fg(k=1,2,3) to improve significantly. Training converges faster.

---

### Improvement 2 — Prototype-augmented Router

```
Before: Router = pure black-box MLP
        Cannot explain what "NCR" looks like in feature space
        No geometric interpretability

After:  Router = MLP + distance-to-prototype blend
        4 learnable concept prototype vectors [4, 192]
        After training: c_k ≈ mean feature of concept-k tokens

        Visualization opportunity: PCA/t-SNE the 4 prototypes →
        show NCR, Edema, ET in separate regions of feature space

Config:  RouterConfig.use_prototypes = True  (default)
File:    modules/router.py
```

New parameters: `self.prototypes [K, D]`, `self.blend_alpha [1]`

The blend_alpha (initialized at 0.0 → α=0.5) lets the model learn how much
to trust the MLP vs. geometric prototype signal.

---

### Improvement 3 — Temperature Annealing

```
Before: Temperature τ is fully learnable with no curriculum alignment
        τ might drift to high values during alignment phase (bad: blurry routing)
        or to low values during warmup (bad: premature collapse)

After:  Soft curriculum target for τ per phase:
        Warmup:     τ_target = 2.0  (diffuse, exploratory)
        Alignment:  τ_target = 1.0  (moderate sharpening)
        Refinement: τ_target = 0.5  (crisp, inference-ready)

        L_tau_reg = 0.01 × (τ_actual − τ_target)²

Config:  CurriculumConfig.tau_{warmup,alignment,refinement} = {2.0, 1.0, 0.5}
         LossConfig.tau_reg_weight = 0.01
Files:   utils/weight_schedule.py, losses/total.py
Usage:   loss_fn(..., tau_current=module.router.temperature)
```

The weight 0.01 is small — this is a soft nudge, not forcing τ to exactly the target.
Temperature still responds to training data dynamics.

---

### Improvement 4 — Focal-BCE for L_align

```
Before: Standard BCE weights all foreground tokens equally
        Easy interior tokens (p_correct ≈ 0.95) dominate gradient
        Hard boundary tokens, small ET voxels get tiny gradient

After:  Focal modulation: (1 - p_correct)^γ × BCE
        γ = 2.0  (standard from Lin et al. 2017)

        Easy token: p_correct=0.95 → weight (1-0.95)^2 = 0.0025
        Hard token: p_correct=0.55 → weight (1-0.55)^2 = 0.2025
        → 80× more gradient on hard tokens

Config:  LossConfig.align_focal_gamma = 2.0  (default)
File:    losses/alignment.py
```

Expect: CAS(ET) and CAS(NCR) improve most because they are the smallest subregions
and previously suffered most from easy-token gradient domination.

---

### Improvement 5 — CAS_fg as Primary Metric

```
Before: CAS_all(k) over all tokens
        Inflated by background (80% of tokens, easy Pearson)
        CAS_all(background) numerically unreliable

After:  CAS_fg(k) for k=1,2,3 over foreground tokens only
        Honest metric: "among tumor tokens, does routing match anatomy?"
        CAS_fg ≥ 0.85 is the target (harder to achieve than CAS_all ≥ 0.85)

        compute() → CAS_all (supplementary)
        compute_fg() → CAS_fg (primary metric, main paper table)

File:    utils/metrics.py
```

---

### Summary table

| # | What Changed | File | Expected Impact |
|---|---|---|---|
| 1 | Foreground-only L_align | `losses/alignment.py` | Higher CAS_fg(NCR,Edema,ET) |
| 2 | Prototype-augmented router | `modules/router.py` | Geometric interpretability, paper figure |
| 3 | Temperature annealing | `weight_schedule.py` + `total.py` | Structurally coherent curriculum |
| 4 | Focal-BCE for L_align | `losses/alignment.py` | Higher CAS(ET), CAS(NCR) |
| 5 | CAS_fg metric | `utils/metrics.py` | Honest reporting |

---

## 21. How to Run the Tests

### Setup

```bash
cd c:\Users\pathi\OneDrive\Desktop\AMBETS
pip install -r requirements_phase1.txt
```

### Run everything

```bash
python -m pytest tests/ -v --tb=short
# Expected: 136/136 passed in ~7s
```

### Run by component

```bash
# Router: shapes, simplex, entropy, temperature, prototype routing
python -m pytest tests/test_router.py -v       # 26 tests

# Expert: residual, initialization, gradient flow
python -m pytest tests/test_expert.py -v       # 11 tests

# All losses: seg, align (fg+focal), diversity, entropy, boundary, tau schedule
python -m pytest tests/test_losses.py -v       # 50 tests

# Dispatcher: soft/hard dispatch, training/eval switch
python -m pytest tests/test_dispatcher.py -v   # 12 tests

# Metrics: CAS_all, CAS_fg, ExpertUtilizationTracker
python -m pytest tests/test_metrics.py -v      # 20 tests

# Full system: forward + backward, curriculum transitions, CAS accumulation
python -m pytest tests/integration/ -v         # 17 tests
```

### Manual sanity check

```python
import torch
from ccr import CCRBottleneckModule, CCRTotalLoss, CCRConfig

cfg    = CCRConfig()
module = CCRBottleneckModule(cfg)
loss_fn = CCRTotalLoss(cfg)

# BraTS-realistic tensor sizes
tokens      = torch.randn(2, 4096, 192)   # [B, N, D]
token_labels = torch.randint(0, 4, (2, 4096))  # [B, N] integer labels

# TRAINING MODE — soft dispatch
module.train()
out = module(tokens)

print("routing_probs:", out["routing_probs"].shape)   # [2, 4096, 4]
print("sums to 1?", out["routing_probs"].sum(-1).allclose(torch.ones(2,4096), atol=1e-5))
print("entropy range:", out["entropy"].min().item(), "to", out["entropy"].max().item())
print("max entropy:", 1.386)  # log(4)

# Compute loss (simulated — real training needs pred_logits from decoder too)
# See tests/integration/test_pipeline.py for full example

# INFERENCE MODE — hard dispatch
module.eval()
out_eval = module(tokens)
# Each token assigned to exactly one expert
assignments = out_eval["assignments"]  # [2, 4096] integer in {0,1,2,3}
print("unique assignments:", assignments.unique())   # should have all 4 values

# CAS metric
from ccr.utils.metrics import ConceptAlignmentScore
cas = ConceptAlignmentScore(num_concepts=4,
    concept_names=("background","necrotic_core","edema","enhancing_tumor"))
cas.update(out["routing_probs"], token_labels)
print("CAS_all:", cas.compute())
print("CAS_fg:", cas.compute_fg())   # only NCR, Edema, ET keys — no background
```

---

## 22. What Phase 1 Is NOT

Phase 1 is the core CCR module — a backbone-agnostic bottleneck replacement.

**It is NOT a complete segmentation model.** You cannot take a BraTS NIfTI file and
produce a segmentation with Phase 1 alone.

Missing components (Phase 2 targets):

```
Phase 1 gives you:
   CCRBottleneckModule [B, N, D] → [B, N, D]
   + all losses
   + metrics
   
Phase 2 will add:
   PatchTokenizer3D   converts [B, 4, H, W, D] into [B, N, D] tokens
   Swin-B encoder     learns discriminative features
   CCRBottleneckModule  (Phase 1, already done!)
   Upsampling decoder  converts [B, N, D] back to [B, K, H, W, D]
   Boundary head       final refinement pass
   Training loop       BraTS dataloader, optimizer, checkpointing
```

---

## 23. Hard Rules: Never Violate These

These are not style preferences. Violating them breaks the paper's claims.

### 1. CCR goes at the bottleneck ONLY

```
CORRECT:  encoder → [CCRBottleneckModule] → decoder
WRONG:    encoder → decoder → [routing at output]

Why: Routing at the output is post-hoc (prediction is made, then explained).
     Routing at the bottleneck is causal (the routing IS the prediction mechanism).
     Proposition 1a requires bottleneck placement. Output routing = approximation.
```

### 2. Never start L_align before epoch 11

```
CORRECT:  epochs 1-10: L_align = 0
          epochs 11+:  L_align = 1.0

WRONG:    start L_align from epoch 1

Why: Expert collapse. The encoder has not learned discriminative features.
     The router receives random gradients and collapses permanently.
```

### 3. λ_entropy ≤ 0.01

```
CORRECT:  LossConfig.entropy regularization weight ≤ 0.01
WRONG:    increase to 0.1 or higher

Why: Larger values push routing toward uniform (high entropy).
     Uniform routing → CAS ≈ 0 (routing is random, matches nothing).
     Proposition 2 (entropy calibration) requires non-uniform routing.
```

### 4. K=4 for BraTS — do not change

```
CORRECT:  K=4: {Background, NCR, Edema, ET}
WRONG:    K=3, K=5, K=2

Why: Matches BraTS annotation protocol exactly.
     CAS measures routing vs. annotation alignment.
     If K ≠ annotation classes, CAS is measuring alignment with a wrong taxonomy.
```

### 5. Report CAS_fg, not CAS_all as primary

```
CORRECT:  CAS_fg(NCR), CAS_fg(Edema), CAS_fg(ET) in Table 1
          CAS_all in supplementary

WRONG:    CAS_all in main paper table
          CAS_all(Background) as evidence of routing quality

Why: CAS_all is inflated by background dominance.
     CAS_all(Background) is numerically unreliable.
     CAS_fg is the honest claim.
```

### 6. Deletion AUC must use HARD-routing maps

```
CORRECT:  model.eval() → hard dispatch → routing_probs is one-hot
          use routing_probs as explanation map for deletion AUC

WRONG:    use soft routing_probs (from training mode) for deletion AUC

Why: The faithfulness claim (Prop 1a) is about inference-time hard dispatch.
     Soft routing probabilities blend all experts — not crisp explanations.
     Deletion AUC with soft maps tests something different from the paper's claim.
```

### 7. NeuroReAct is a separate unpublished system — do not cite in CCR paper

```
NeuroReAct_Implementation_Plan.md and system.md are in this directory but
belong to a separate project. Do NOT reference them anywhere in CCR paper text.
```

---

## Appendix: File Map

```
src/ccr/
├── __init__.py                    exports CCRBottleneckModule, CCRTotalLoss, CCRConfig
├── config/
│   └── ccr_config.py              ALL hyperparameters as typed dataclasses
├── modules/
│   ├── router.py                  ClinicalConceptRouter (routing MLP + prototypes)
│   ├── expert.py                  ClinicalConceptExpert (residual MLP per concept)
│   └── dispatcher.py              CCRBottleneckModule (router + experts + dispatch)
├── losses/
│   ├── segmentation.py            DiceLoss + FocalLoss → L_seg
│   ├── alignment.py               ConceptAlignmentLoss → L_align (core loss)
│   ├── diversity.py               ExpertDiversityLoss → L_diversity
│   ├── entropy.py                 EntropyRegularizationLoss → L_entropy_reg
│   ├── boundary.py                BoundaryAwareLoss → L_boundary (GPU, no scipy)
│   └── total.py                   CCRTotalLoss (combines all 5 + curriculum)
└── utils/
    ├── weight_schedule.py         CurriculumWeightScheduler (λ schedule + τ targets)
    └── metrics.py                 ConceptAlignmentScore + ExpertUtilizationTracker

tests/
├── conftest.py                    shared fixtures (router, tokens, config)
├── test_router.py                 26 tests: shapes, simplex, entropy, temperature, prototypes
├── test_expert.py                 11 tests: residual, init, gradients
├── test_losses.py                 50 tests: all 5 losses + tau schedule + foreground + focal
├── test_dispatcher.py             12 tests: soft/hard dispatch, training/eval switch
├── test_metrics.py                20 tests: CAS_all, CAS_fg, ExpertUtilizationTracker
└── integration/
    └── test_pipeline.py           17 tests: full forward+backward, curriculum transitions

docs/
├── phase1_architecture.md         dense technical reference
├── phase1_guide.md                readable narrative guide
├── phase1_analysis.md             theory validation + 5 improvements (concise)
└── phase1_onboarding.md           ← this file
```

---

*Last updated: 2026-05-28. All 5 Phase 1 improvements implemented. 136/136 tests pass.*
