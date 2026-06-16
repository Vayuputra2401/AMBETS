# CCR Phase 1 — What Was Built and Why

**Date completed**: 2026-05-26
**Tests**: 91/91 passing (5.2 s on CPU)
**Entry point**: `src/ccr/`

---

## The One Paragraph You Need to Read First

CCR proposes that the routing decision at the encoder bottleneck IS the segmentation
prediction, IS the per-voxel explanation, and IS the calibrated uncertainty estimate —
all in one forward pass.  Phase 1 builds this routing mechanism in isolation.  It does not
yet build the full encoder or the boundary head.  What it does build is the core module that
will be plugged into any segmentation backbone in Phase 2 and Phase 3.  Everything in Phase 1
is designed to be backbone-agnostic.

---

## Part 1 — The Problem This Code Solves

Before reading any code, understand why each piece exists.

### Why routing at all?

Standard segmentation models produce a prediction and then, separately, you run GradCAM or
SHAP to explain it.  The explanation is computed *after* the decision, from gradients or
attention weights that are byproducts of the computation.  These approximations are documented
to have deletion AUC 0.55–0.62 — barely better than random masking.

CCR's answer: make the explanation the computation itself.  A routing layer at the bottleneck
assigns each image token to one of K clinical expert networks.  The routing probability
*P(t → k)* — the probability that token *t* was sent to expert *k* — is the model's statement
that the tissue at position *t* belongs to clinical concept *k*.  This is not derived from the
prediction; it determines the prediction.

### Why K = 4 for BraTS?

BraTS annotates four subregions: Background, Necrotic Core (NCR), Peritumoral Edema, and
Enhancing Tumor (ET).  We set K = 4 to match this protocol exactly.  Expert 0 learns
Background, Expert 1 learns NCR, Expert 2 learns Edema, Expert 3 learns ET.  The experts are
not free to discover arbitrary concepts — they are constrained to clinical concepts by the
alignment loss (L_align) during training.

---

## Part 2 — The Three Core Classes

### `ClinicalConceptRouter`  (`src/ccr/modules/router.py`)

**What it does**: Takes the encoder's bottleneck token sequence and outputs, for every token,
a probability distribution over the K clinical concepts.

**Input / Output**:
```
Input:  bottleneck tokens  [B, N, D]
            B = batch size
            N = number of tokens (e.g. 4096 for 128³ volume with patch_size=8)
            D = encoder embedding dimension (192 for Swin-B)

Output: routing_probs  [B, N, K]  — sums to 1 along K for every token
        entropy        [B, N]     — routing uncertainty per token
        logits         [B, N, K]  — pre-softmax values (for debugging)
```

**The architecture**: A four-layer MLP applied identically to every token:

```
LayerNorm(D)                     stabilises inputs (tokens come from a transformer)
→ Linear(D → H)                  project up.  H = 384 for D=192, factor=2.0
→ GELU                           non-linearity
→ Linear(H → K)                  project to number of concepts
→ divide by temperature τ        controls sharpness of distribution
→ Softmax                        converts to probability distribution
```

**The temperature parameter τ** is learnable.  It is clamped to [0.1, 5.0].
- Low τ (approaching 0.1): routing becomes nearly one-hot — very confident assignment
- High τ (approaching 5.0): routing approaches uniform — maximum uncertainty

At the start of training, τ = 1.0.  The model learns to increase it where it is genuinely
uncertain (boundaries) and decrease it where it is confident (tumor core centers).

**What "sum to 1" means practically**: For every single token in every batch, the four
routing probabilities sum to exactly 1.0.  If token *t* has routing_probs = [0.02, 0.85, 0.08, 0.05],
that means the model is 85% confident this token belongs to the Necrotic Core.  That number
is the explanation for this voxel.  No post-hoc computation is needed.

**Entropy**: Computed as −Σ_k P(t→k) log P(t→k).  Ranges from 0 (one-hot, certain) to
log(4) ≈ 1.386 (uniform, maximally uncertain).  This entropy is the free calibrated uncertainty
estimate for every voxel — Proposition 2 of the paper.

---

### `ClinicalConceptExpert`  (`src/ccr/modules/expert.py`)

**What it does**: One expert per clinical concept.  Receives the bottleneck tokens assigned to
its concept and refines them with concept-specific computation.

**Input / Output**:
```
Input:  tokens  [M, D]   where M = number of tokens dispatched to this expert
Output: tokens  [M, D]   refined
```

**The architecture**: A 2-layer residual MLP:

```
LayerNorm(D)
→ Linear(D → H_expert)    H_expert ≈ 512 for D=192 (factor 2.667)
→ GELU
→ Linear(H_expert → D)
+ residual (add input back)
```

**Why residual?** At the very start of training, before the experts have learned anything
useful, the correction term is near-zero (the second linear layer is initialised near zero).
So the expert acts almost like an identity function.  This prevents large, misleading gradient
signals from corrupting the encoder in the first few epochs.  As training progresses and the
routing becomes meaningful, the experts learn non-trivial corrections.

**Each expert is a separate set of weights**.  Expert 1 (NCR) and Expert 2 (Edema) have
entirely independent parameters.  They start from the same random initialisation but diverge
during training as the alignment loss forces them to specialise.

---

### `CCRBottleneckModule`  (`src/ccr/modules/dispatcher.py`)

**What it does**: Combines the router and the K experts into the single module that replaces
the bottleneck in a segmentation model.

**Input / Output**:
```
Input:   bottleneck_tokens  [B, N, D]

Output:
    routing_probs  [B, N, K]   — THE EXPLANATION
    expert_outputs [B, N, D]   — concept-conditioned features (fed to decoder)
    entropy        [B, N]      — routing uncertainty
    assignments    [B, N]      — hard concept assignment (argmax of routing_probs)
    logits         [B, N, K]   — pre-softmax router logits
```

**Two dispatch modes**:

*Soft dispatch (training)*:

```
expert_output(b, t) = Σ_k  P(b, t→k) · Expert_k(tokens[b, t])
```

Every expert processes every token.  Expert *k*'s output is weighted by P(t→k).
The result is a weighted average of all four experts' outputs.

This is the training mode because it is fully differentiable.  Gradients flow
from the expert outputs, through the routing probability weights, back to the router MLP,
and then back to the encoder.  This is what allows L_align to reshape how the encoder
represents the image.

*Hard dispatch (inference)*:

```
k*(b, t) = argmax_k P(b, t→k)
expert_output(b, t) = Expert_{k*(b,t)}(tokens[b, t])
```

Each token is processed by exactly one expert — the one with the highest routing probability.
This produces crisp, deterministic concept maps: every single voxel is unambiguously assigned
to exactly one clinical concept.  This is the inference mode.

The switch is automatic: `model.train()` → soft dispatch.  `model.eval()` → hard dispatch.

---

## Part 3 — The Five Loss Functions

Training CCR requires five loss components that work together.  Each serves a specific role.

### Loss 1: L_seg — Segmentation Loss  (`src/ccr/losses/segmentation.py`)

**Purpose**: Keep the model accurate at segmentation.  This is the baseline loss that all
other losses are added to.  It ensures CCR does not sacrifice accuracy for interpretability.

**Formula**:
```
L_seg = 0.5 × DiceLoss + 0.5 × FocalLoss(γ=2)
```

**DiceLoss** measures volumetric overlap.  For each class *c*:
- Compute predicted probability *p_c* and binary ground-truth *y_c*
- Dice = (2 × overlap) / (total predicted + total GT)
- Average across all K classes

The squared denominator form (*p²* + *y²*) is used instead of the linear form (*p* + *y*)
because it is numerically more stable when predictions are soft probabilities.

**FocalLoss** down-weights easy examples.  When the model is already very confident and
correct on a voxel, the gradient from that voxel is small.  When the model is wrong or
uncertain, the gradient is large.  This is critical for BraTS because the Enhancing Tumor
subregion occupies < 5% of the volume — without focal weighting, the background would
dominate every gradient update.

**L_seg is always weighted 1.0**.  It does not change across curriculum phases.

---

### Loss 2: L_align — Concept Alignment Loss  (`src/ccr/losses/alignment.py`)

**Purpose**: Force Expert *k* to activate for the clinical subregion *k*.

This is the most important loss in the paper.  Without it, the router is arbitrary — tokens
could be routed to any expert regardless of their clinical meaning.  L_align is what
transforms the routing from an efficiency mechanism into an interpretability mechanism.

**What it computes**:

For each concept *k* (Background, NCR, Edema, ET):
1. Build a binary ground-truth map: *gt_k(t)* = 1 if the annotation says token *t* is concept *k*, else 0
2. Take the routing probability: *p_k(t)* = routing_probs[t, k]
3. Compute binary cross-entropy between them: BCE(*p_k*, *gt_k*)
4. Average across all K concepts

**Why binary cross-entropy?** BCE(*p_k*, *gt_k*) is minimised when *p_k(t)* = 1 for tokens
in subregion *k* and *p_k(t)* = 0 for all other tokens.  Because the routing probabilities
sum to 1 (they are a softmax), pushing *p_k* up for a token automatically pushes all other
*p_{k'≠k}* down for that same token.  So training with L_align forces a one-vs-rest
alignment between routing and the clinical annotation.

**What L_align achieves after training**: The Concept Alignment Score (CAS) — the Pearson
correlation between routing probabilities and GT masks — should reach ≥ 0.85 for all four
experts.  CAS = 0.85 means the routing map for a concept explains 85% of the variance in
the ground-truth mask for that concept.  By Proposition 1a, this directly equals the
faithfulness of the explanation for CCR-Net.

**L_align is ZERO during the warmup phase (epochs 1–10)**.  This is intentional.  See
Part 4 (Curriculum) for the reason.

---

### Loss 3: L_diversity — Expert Diversity Loss  (`src/ccr/losses/diversity.py`)

**Purpose**: Prevent two failure modes of Mixture-of-Experts training.

Mixture-of-Experts models have two well-known failure modes:

**Failure mode 1 — Expert collapse**: All tokens route to a single expert.  The other experts
receive no gradient and never learn anything.  The model degrades to a single-expert model.

**Failure mode 2 — Expert redundancy**: Two or more experts learn identical representations.
The effective number of concepts drops below K.  For BraTS this would mean two experts
learning the same subregion while another goes completely unused.

L_diversity directly penalises both:

*Load balancing component*: Computes the average fraction of tokens each expert receives.
In a perfect world, each of the K=4 experts should receive 25% of tokens.  The loss penalises
deviation from this 25% target: Σ_k (actual_fraction_k − 1/K)².

*Cosine overlap component*: Computes cosine similarity between every pair of expert routing
vectors (treating each expert's routing probabilities over all tokens as a vector).  Penalises
any two experts that have similar routing patterns — forcing them to specialise on different
tokens.

L_diversity is active throughout all training phases (λ₂ = 0.5 in warmup and alignment,
0.1 in refinement).  It is the main guard against collapse even before L_align is introduced.

---

### Loss 4: L_entropy_reg — Entropy Regularisation  (`src/ccr/losses/entropy.py`)

**Purpose**: Keep routing entropy meaningful as a calibrated uncertainty signal.

**What it computes**: Mean routing entropy across all tokens.  Routing entropy is already
computed in the router forward pass (H(t) = −Σ_k P log P).  This loss simply adds a penalty
proportional to the mean of those entropy values.

**Why it is needed**: Without this loss, the routing entropy can drift toward a constant value
— neither high nor low — that carries no information about where the model is uncertain.
With a small entropy regularisation, the model is gently encouraged to be confident where it
can be (interior of tumor subregions) and uncertain where it genuinely should be (boundaries,
multi-region voxels).

**Critical constraint**: λ₃ = 0.01 — intentionally very small.  If λ₃ is increased
significantly, the model is pushed toward uniform routing (maximum entropy) for all tokens,
which destroys both the CAS (alignment is lost) and the Proposition 2 claim (calibrated
uncertainty).  The value 0.01 is the right order of magnitude: small enough to not interfere
with L_align, large enough to prevent entropy from collapsing to a flat constant.

---

### Loss 5: L_boundary — Boundary-Aware Loss  (`src/ccr/losses/boundary.py`)

**Purpose**: Give the optimiser a stronger incentive to get voxels near tumor boundaries right.

Standard cross-entropy weights every voxel equally.  But the clinical importance of getting a
voxel right scales with how close it is to a subregion boundary.  Getting an interior NCR voxel
wrong has a smaller clinical impact than getting a boundary voxel wrong.

**How it works**:

Step 1 — Detect boundary voxels using morphological operations (GPU-compatible, no scipy):
- One-hot encode the label map → [B, K, H, W, D]
- Morphological dilation: take the maximum value over each 3×3×3 neighbourhood
- Morphological erosion: take the minimum value over each 3×3×3 neighbourhood
- A voxel is on a boundary if dilation ≠ erosion for any class
- Optionally widen the boundary zone by repeating the dilation step

Step 2 — Build a weight map:
- Interior voxels: weight = 1.0 (base_weight)
- Boundary voxels: weight = 6.0 (base_weight 1.0 + boost_factor 5.0)

Step 3 — Compute weighted cross-entropy:
- Standard CE loss with per-voxel weights
- Boundary voxels contribute 6× to the total loss

**This is not differentiated through** — the weight map is computed in `torch.no_grad()` and
detached from the computation graph.  The weight map is a fixed scalar multiplier, not a
learned function.  Only the CE values carry gradients.

L_boundary is most important in the refinement phase (λ₄ = 1.0, epochs 51–80) when the model
has already learned coarse concept alignment and needs to sharpen boundary precision.

---

## Part 4 — The Training Curriculum

**Why a curriculum?**

If you activate L_align from epoch 1, the encoder has not yet learned discriminative features.
The router receives gradients from a random encoder, all experts collapse to the same point,
and training fails.  This is a known failure mode of Mixture-of-Experts training.

The solution is a three-phase curriculum that introduces losses in the right order:

```
Phase        Epochs    What happens
─────────────────────────────────────────────────────────────────────────────────
Warmup       1–10      Only L_seg + L_diversity are active.
                       The encoder learns to segment.
                       L_diversity prevents early collapse.
                       L_align = 0 — the encoder is not ready for alignment yet.

Alignment    11–50     L_align is switched on at full weight (λ₁ = 1.0).
                       The routing probabilities are forced to match the GT masks.
                       CAS rises from ~0.2 (random) toward ≥ 0.85 (target).
                       L_entropy_reg is also activated (λ₃ = 0.01).

Refinement   51–80     L_align is reduced (λ₁ = 0.5) — alignment is mostly learned.
                       L_boundary is raised to full weight (λ₄ = 1.0).
                       Focus shifts to sharpening boundary predictions.
                       L_diversity is reduced (λ₂ = 0.1) — collapse is no longer a risk.
─────────────────────────────────────────────────────────────────────────────────
```

In code, the curriculum is managed by `CurriculumWeightScheduler` in `utils/weight_schedule.py`
and applied by `CCRTotalLoss` in `losses/total.py`:

```python
loss_fn = CCRTotalLoss(config)
loss_fn.set_epoch(current_epoch)   # called once per epoch before the training loop
```

---

## Part 5 — The Metrics

### Concept Alignment Score (CAS)  (`src/ccr/utils/metrics.py`)

CAS is the primary metric for CCR.  It measures how well the routing probabilities correlate
with the ground-truth segmentation masks.

**Formula**: Pearson correlation between P(t→k) and M_k(t) over the validation set.

```
CAS(k) = Pearson( P(t→k|x),  M_k(t) )
       = (n × ΣPM − ΣP × ΣM) / (sqrt(n ΣP² − (ΣP)²) × sqrt(n ΣM² − (ΣM)²))

where n = total number of tokens accumulated
```

**Why Pearson?** It is invariant to scale and offset, which matters because P(t→k) is a soft
probability (continuous, in [0, 1]) while M_k(t) is a binary GT mask (0 or 1).  Pearson
measures correlation between the shapes of the two distributions, not their absolute values.

**Target**: CAS ≥ 0.85 for all four concepts on the BraTS 2021 validation set.

**Implementation**: Running sums are used instead of storing all predictions.  This means you
can accumulate over a large validation set without running out of memory.

```python
cas = ConceptAlignmentScore(num_concepts=4, concept_names=config.concept_names)
for batch in val_loader:
    cas.update(routing_probs, token_labels)   # accumulate
scores = cas.compute()   # compute Pearson from running sums
# {'background': 0.91, 'necrotic_core': 0.87, 'edema': 0.85, 'enhancing_tumor': 0.86}
cas.reset()
```

**Proposition 1a connection**: For CCR-Net (where routing = prediction), faithfulness = CAS
exactly.  CAS is not just a training metric — it is the formal measure of faithfulness.

---

### Expert Utilization Tracker  (`src/ccr/utils/metrics.py`)

**What it tracks**: What fraction of all tokens each expert receives each epoch.

**Collapse threshold** (from plan Section 16.1): If any expert receives < 5% of tokens after
epoch 20 (post-warmup), the model is showing signs of expert collapse.  The tracker emits a
`RuntimeWarning` and returns the collapsed expert names so you can re-initialise those experts.

```python
tracker = ExpertUtilizationTracker(
    num_concepts=4,
    concept_names=config.concept_names,
    warmup_end_epoch=10,
)
for batch in train_loader:
    tracker.update(ccr_out["assignments"])   # assignments: [B, N] hard routing
stats   = tracker.compute()   # {'background': 0.79, 'necrotic_core': 0.09, ...}
flagged = tracker.check_collapse(epoch=current_epoch)
tracker.reset()
```

If `flagged` is non-empty, re-initialise the weights of those experts before continuing.

---

## Part 6 — The Configuration

Everything is controlled from a single config object.  No magic numbers in the model code.

```python
from ccr import CCRConfig

cfg = CCRConfig()   # BraTS defaults: K=4, embed_dim=192
```

The full config tree looks like this:

```
CCRConfig
├── router: RouterConfig
│   ├── embed_dim = 192           Swin-B bottleneck dimension
│   ├── num_concepts = 4          K = 4 for BraTS
│   ├── hidden_dim_factor = 2.0   Router hidden = 192 × 2 = 384
│   ├── temperature_init = 1.0    Starting τ
│   └── temperature_learnable = True
│
├── expert: ExpertConfig
│   ├── embed_dim = 192
│   ├── num_concepts = 4
│   └── expert_hidden_factor = 2.667  Expert hidden = 192 × 2.667 ≈ 512
│
├── loss: LossConfig
│   ├── focal_gamma = 2.0
│   ├── focal_alpha = 0.25
│   ├── dice_smooth = 1e-5
│   ├── diversity_eps = 1e-8
│   ├── entropy_eps = 1e-8
│   ├── boundary_dilation_iters = 3
│   ├── boundary_base_weight = 1.0
│   ├── boundary_boost_factor = 5.0
│   └── weights: LossWeights
│       ├── warmup     = (0.0, 0.5, 0.00, 0.0)
│       ├── alignment  = (1.0, 0.5, 0.01, 0.3)
│       └── refinement = (0.5, 0.1, 0.01, 1.0)
│
├── curriculum: CurriculumConfig
│   ├── warmup_end_epoch = 10
│   ├── alignment_end_epoch = 50
│   └── total_epochs = 80
│
└── concept_names = ("background", "necrotic_core", "edema", "enhancing_tumor")
```

To use for LiTS instead of BraTS:

```python
from ccr.config.ccr_config import RouterConfig, ExpertConfig

cfg_lits = CCRConfig(
    router        = RouterConfig(num_concepts=3),
    expert        = ExpertConfig(num_concepts=3),
    concept_names = ("background", "liver", "liver_tumor"),
)
```

The `__post_init__` method on CCRConfig will raise a `ValueError` immediately if you create
an inconsistent config (e.g., router has K=4 but concept_names has 3 entries).

---

## Part 7 — The Tests

**91 tests across 5 test files, all passing in 5.2 seconds on CPU.**

### `tests/test_router.py` — 18 tests

Covers the fundamental invariants that must hold throughout training:
- Routing probabilities sum to 1.0 for every token (the probability simplex)
- No NaN or Inf values in any output
- Entropy is bounded in [0, log(K)]
- One-hot routing has zero entropy; uniform routing has maximum entropy
- Higher temperature produces higher entropy (confirmed by filling the same router with two different τ values)
- Gradients flow through routing_probs back to the input tokens (required for L_align to shape the encoder)

### `tests/test_expert.py` — 11 tests

- Output shape matches input shape
- With zero fc2 weights, expert acts as pure identity (residual = 0)
- With non-zero fc2 weights, expert modifies the features
- All weights receive gradients after backward
- Different experts produce different outputs on the same input

### `tests/test_losses.py` — 26 tests

For each of the five losses and the total loss:
- Returns a non-negative scalar
- Perfect prediction → loss near zero (validates the math is correct)
- Gradient flows back to the relevant input
- CCRTotalLoss: L_align = 0 at epoch 5 (warmup), L_align > 0 at epoch 20 (alignment)
- CCRTotalLoss: total = sum of weighted components (validates the weighting formula)
- BoundaryAwareLoss with boost_factor=0 equals plain cross-entropy (validates the weight map logic)

### `tests/test_dispatcher.py` — 12 tests

- All output dictionary keys are present with correct shapes
- assignments == routing_probs.argmax(dim=-1) in all modes
- Soft dispatch in train mode; hard dispatch in eval mode
- Gradient flows through soft dispatch to all router and expert parameters
- On random input, all 4 experts receive at least some tokens (no collapse at init)

### `tests/integration/test_pipeline.py` — 24 tests

The integration test simulates a complete training step with all components active:

1. Create CCRBottleneckModule (router + 4 experts)
2. Create CCRTotalLoss with epoch=15 (alignment phase — all losses active)
3. Forward pass with BraTS-like token inputs [B=2, N=1024, D=192]
4. Build placeholder 3D segmentation logits [B=2, K=4, H=32, W=32, D=32]
5. Compute all 5 loss components
6. Backward pass
7. Verify: routing_probs sums to 1, all losses are positive, all router and expert weights have gradients, no NaN gradients

Also tests:
- **Collapse detection**: Simulating all-expert-0 routing triggers a RuntimeWarning
  and the tracker returns the 3 collapsed expert names
- **Curriculum transitions**: L_align = 0 at epoch 5, = 1.0 at epoch 25, = 0.5 at epoch 65
- **CAS accumulation**: Perfect alignment routing gives CAS > 0.85; random routing gives |CAS| < 0.3

---

## Part 8 — How to Run

From the `AMBETS/` directory:

```bash
# Full test suite
python -m pytest tests/ -v

# Just the integration pipeline
python -m pytest tests/integration/test_pipeline.py -v

# Check a specific invariant
python -m pytest tests/test_router.py::TestProbabilitySimplex -v

# Check all loss math
python -m pytest tests/test_losses.py -v
```

---

## Part 9 — What Phase 1 Does NOT Include

Phase 1 is the CCR module in isolation.  It does not include:

- The 3D patch tokenizer (PatchTokenizer3D with patch_size=8) — Phase 2
- The Swin-B 3D encoder — Phase 2
- The thin boundary refinement / upsampling head — Phase 2
- The full CCR-Net end-to-end training loop — Phase 2
- CCR-Retrofit (wrapping existing backbones) — Phase 3
- The BraTS data pipeline (beyond the existing EDA code) — Phase 2
- Actual training on BraTS 2021 — Phase 2

What Phase 1 provides is the drop-in module that Phase 2 and Phase 3 will plug into their
respective architectures.  The interface is clean:

```python
from ccr import CCRBottleneckModule, CCRConfig, CCRTotalLoss

cfg        = CCRConfig()
ccr_module = CCRBottleneckModule(cfg)    # drop into any encoder bottleneck
loss_fn    = CCRTotalLoss(cfg)

# One step of training:
ccr_out  = ccr_module(bottleneck_tokens)   # [B, N, D]
loss_out = loss_fn(pred_logits, labels, ccr_out["routing_probs"], token_labels)
loss_out["total"].backward()
```

---

## Part 10 — Invariants That Must Never Be Changed

These constraints are locked in.  Changing any of them breaks the theoretical claims.

| Constraint | Why it is locked |
|---|---|
| Σ_k P(t→k) = 1 for all tokens | The routing must be a probability distribution.  Violation makes CAS undefined and L_align gradients meaningless. |
| L_align = 0 during epochs 1–10 | Activating L_align before the encoder learns anything causes expert collapse.  The warmup epoch count can be tuned but must not be set to 0. |
| CCR placed at the encoder bottleneck only | Placing routing at the output means the prediction is made first and routing is derived from it — that is post-hoc.  Only bottleneck placement is causally upstream. |
| K = 4 for BraTS | The alignment loss uses the BraTS annotation labels directly.  K must match the number of annotated subregions. |
| λ₃ (entropy) must stay ≤ 0.01 | Larger values push routing toward maximum entropy (uniform), destroying both CAS and Proposition 2. |
| No efficiency claims as contributions | FLOPs and parameter counts are reported for completeness but are not contributions.  This is an interpretability paper. |

---

*Phase 1 complete.  Phase 2: PatchTokenizer3D + Swin-B encoder + CCR-Net full training loop on BraTS 2021.*
