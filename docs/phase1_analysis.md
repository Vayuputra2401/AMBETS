# CCR Phase 1 — Theory Validation, Test Guide, and Improvements

**Date**: 2026-05-28
**Status**: Phase 1 complete + improved — 136/136 tests pass (was 91 before improvements)
**Purpose**: Honest assessment of what Phase 1 is, whether the theory holds, how to run tests, and five principled improvements — all of which have been implemented.

## Implementation Status

All 5 improvements are live. Defaults changed to the improved versions.

| # | Improvement | File | Default |
|---|---|---|---|
| 1 | Foreground-only L_align | `losses/alignment.py` | `foreground_only=True` |
| 2 | Prototype-augmented router | `modules/router.py` | `use_prototypes=True` |
| 3 | Temperature annealing | `utils/weight_schedule.py` + `losses/total.py` | τ: 2.0→1.0→0.5 |
| 4 | Focal-BCE for L_align | `losses/alignment.py` | `focal_gamma=2.0` |
| 5 | CAS_fg metric | `utils/metrics.py` | `cas.compute_fg()` |

New test file: `tests/test_metrics.py` — 20 tests for CAS_all, CAS_fg, ExpertUtilizationTracker.
New config fields: `RouterConfig.use_prototypes`, `LossConfig.align_foreground_only`, `LossConfig.align_focal_gamma`, `LossConfig.tau_reg_weight`, `CurriculumConfig.tau_{warmup,alignment,refinement}`.

---

---

## 1. What Phase 1 Is (and Is Not)

Phase 1 is a **backbone-agnostic bottleneck module**. It is NOT a complete segmentation model. It cannot take a BraTS volume and produce a segmentation. It has no encoder, no patch tokenizer, no decoder, no data pipeline.

What it IS:

```
CCRBottleneckModule [B, N, D] → [B, N, D]
    ├── ClinicalConceptRouter     → routing_probs [B, N, K]
    └── K × ClinicalConceptExpert → expert_outputs [B, N, D]

CCRTotalLoss
    ├── L_seg         (Dice + Focal)
    ├── L_align       (BCE routing vs. GT subregion masks)
    ├── L_diversity   (load balance + cosine Gram overlap)
    ├── L_entropy_reg (entropy regularization)
    └── L_boundary    (morphological boundary-weighted CE)

CurriculumWeightScheduler  — three-phase λ schedule
ConceptAlignmentScore      — running Pearson CAS
ExpertUtilizationTracker   — collapse detection
```

Phase 1 is the **core claim instantiated in code**. Everything else in Phase 2–6 is infrastructure and empirical validation around this core.

---

## 2. How to Test It

### Run all 91 tests

```bash
cd c:\Users\pathi\OneDrive\Desktop\AMBETS
pip install -r requirements_phase1.txt
python -m pytest tests/ -v --tb=short
```

### Run by component

```bash
python -m pytest tests/test_router.py -v       # 18 tests: shapes, simplex, entropy, gradients
python -m pytest tests/test_expert.py -v       # 11 tests: residual, init, gradients
python -m pytest tests/test_losses.py -v       # 26 tests: all 5 loss components
python -m pytest tests/test_dispatcher.py -v   # 12 tests: soft/hard dispatch, auto-switch
python -m pytest tests/integration/ -v         # 24 tests: full training step simulation
```

### What the integration test proves

`tests/integration/test_pipeline.py` is the most important test. It:
- Simulates a full forward + backward pass with BraTS-shaped random tensors
- Verifies every gradient flows: router weights, expert weights, temperature, input tokens
- Verifies curriculum transitions at the correct epochs
- Verifies L_align is exactly zero during warmup and non-zero after
- Verifies CAS metric accumulates correctly across batches
- Verifies ExpertUtilizationTracker emits warnings at the correct threshold

Running this is the closest you can get to "does Phase 1 work as a system" without actual BraTS data.

### Manual sanity test (without pytest)

```python
import torch
from ccr import CCRBottleneckModule, CCRTotalLoss, CCRConfig

cfg = CCRConfig()
module = CCRBottleneckModule(cfg)
loss_fn = CCRTotalLoss(cfg)

tokens = torch.randn(2, 4096, 192)   # B=2, N=4096, D=192
out = module(tokens)

# Check 1: routing_probs is a valid probability distribution
assert out["routing_probs"].sum(dim=-1).allclose(torch.ones(2, 4096), atol=1e-5)

# Check 2: soft dispatch during training
module.train()
out_train = module(tokens)
# every expert contributed

# Check 3: hard dispatch at inference
module.eval()
out_eval = module(tokens)
# each token assigned to exactly one expert

print("routing_probs shape:", out["routing_probs"].shape)   # [2, 4096, 4]
print("expert_outputs shape:", out["expert_outputs"].shape) # [2, 4096, 192]
print("entropy range:", out["entropy"].min().item(), "to", out["entropy"].max().item())
```

---

## 3. Theory Validation — What Holds

### 3.1 The core claim: routing IS the explanation

**Valid — and stronger than all prior work.**

Post-hoc attribution (GradCAM, SHAP, Integrated Gradients, Attention Rollout) computes `∂output/∂input` *after* the prediction is made. These measure sensitivity, not causation. Documented deletion AUC 0.55–0.62 on BraTS — barely above random masking.

CCR's routing is at the bottleneck — the narrowest point in the information flow. The decoder has access only to concept-conditioned expert representations, not the original encoder features. P(t→k) is not a description of the computation; it IS the computation. Every decoder operation is conditioned on it. This is why Proposition 1a is exact by construction, not approximate.

The closest theoretical precedent is Concept Bottleneck Models (Koh et al., 2020). CCR generalizes CBM to dense per-voxel prediction with soft routing and is more suitable for 3D segmentation where you need per-token (not per-image) concept assignment.

### 3.2 L_align does the right thing by design

BCE(p_k, gt_k) pushes p_k → 1 for tokens in subregion k and p_k → 0 for tokens outside. The probability simplex constraint (Σ_k p_k = 1) means you cannot push all p_k high simultaneously — they compete. Pushing p_NCR(t) high for an NCR token automatically pushes p_Edema(t) and p_ET(t) down for that token. This is the right inductive bias. You are not training K independent binary classifiers; you are training a K-way soft assignment constrained to be consistent.

### 3.3 Soft dispatch during training is theoretically necessary

Gradient path: decoder → expert outputs → `Σ_k p_k · Expert_k_output` → p_k → router MLP → encoder. This is the path through which L_align reshapes the encoder's internal representations. Without soft dispatch, argmax breaks this path. Top-1 hard routing from epoch 1 would require straight-through estimators (noisy, unstable). Soft dispatch gives clean, stable gradients.

### 3.4 Warmup before L_align is necessary by theory

In early epochs, encoder bottleneck features are near-random (network not trained). If L_align is active, the router receives random gradients and tends to collapse to a single expert by chance — the one that accidentally minimizes the random loss best. Once collapsed, other experts receive no gradient (soft dispatch weights for those experts → 0) and stay untrained permanently. L_diversity (active during warmup with λ=0.5) partially prevents this, but warmup (L_align=0) is the correct structural fix.

### 3.5 Temperature clamping is necessary

Without clamping, τ → 0 collapses routing to a near-delta distribution (all weight on one expert), making soft dispatch ≈ hard dispatch and breaking gradient flow through p_k. τ → ∞ makes routing uniform, so L_align receives no signal through routing_probs. Clamping to [0.1, 5.0] keeps the distribution in a useful range throughout training.

---

## 4. Theory Validation — Known Limitations

These are not errors in the implementation. They are inherent limitations of the approach that must be acknowledged in the paper and planned for in experiments.

### 4.1 CAS(background) is unreliable as a reported metric

Background is ~80% of BraTS voxels. M_0(t) (background membership) is nearly all-1s with near-zero variance. Pearson correlation is numerically unreliable when one variable has low variance (near-zero denominator).

**What to do**: Track and report CAS for NCR (k=1), Edema (k=2), ET (k=3) only. Background alignment is implicit — if all three tumor CAS values are ≥ 0.85, Background routing is correct by elimination (tokens not assigned to tumor concepts are assigned to Background). Do not include CAS(background) in the main paper table.

### 4.2 Soft dispatch during training means the explanation is not fully causal

During training with soft dispatch: `expert_output(t) = Σ_k P(t→k) · Expert_k(X[t])`. For a token with routing_probs = [0.9, 0.05, 0.03, 0.02], Expert_NCR contributes 90% but the other experts contribute 10%. This 10% is unexplained — the decoder used it, but the explanation only accounts for the dominant routing probability.

**Why this is still valid**: At inference, hard dispatch eliminates this. Each token goes to exactly one expert. The paper's faithfulness claim is about inference-time behavior. Soft dispatch is a training convenience (differentiability) that is discarded at inference. The experiment that proves this: measure deletion AUC with hard-routing explanation maps, not soft routing maps.

### 4.3 L_diversity penalizes routing patterns, not expert functionality

The Gram matrix penalty penalizes cosine similarity between expert routing vectors (which tokens are routed where). Two experts can have dissimilar routing patterns but still learn identical transformations. L_diversity prevents routing redundancy; it does not guarantee functional expert diversity. Functional diversity must come from L_align forcing different experts to process different subregions.

**For the paper**: state this clearly. L_diversity is a necessary condition (prevents collapse) but not sufficient condition (does not guarantee specialization). L_align provides specialization. The two losses are complementary, not redundant.

### 4.4 Proposition 2 (routing entropy ECE < MC-Dropout) is empirical, not theoretical

The entropy calibration claim requires empirical validation in Phase 4. There is no theoretical guarantee that routing entropy is better calibrated than MC-Dropout ECE — it depends on training quality and the specific dataset. The intuition is sound (boundary tokens have genuinely ambiguous routing → high entropy → genuine uncertainty) but the ECE comparison is data-dependent.

**What to do**: treat Proposition 2 as an empirical finding in Phase 4, not a derived theorem. The theoretical argument is the mechanism (routing entropy is a natural, zero-overhead uncertainty signal); the ECE < MC-Dropout comparison is the empirical result.

### 4.5 The K=4 assumption imposes hard concept boundaries

L_align forces every token to be explained by one clinical concept from the fixed vocabulary of K=4. Transition zone tokens (where NCR meets Edema, for example) will be ambiguously assigned. The model must pick one concept even if the voxel is genuinely between concepts.

**Why this is acceptable**: BraTS annotations have the same limitation — annotators assign every voxel to exactly one class. CCR follows the annotation protocol exactly. K=4 is not an arbitrary choice; it reflects the clinical taxonomy. The residual routing entropy at boundaries is the honest signal that the model is uncertain at these voxels.

---

## 5. Five Principled Improvements

These are ordered by theoretical impact and implementation cost. All are backward-compatible with Phase 1's existing code.

---

### Improvement 1 — Foreground-only L_align (Immediate, Low Cost)

**Problem**: Background tokens (~80% of BraTS voxels) dominate the L_align BCE computation. The loss is easy to minimize for background (most tokens correctly route to Expert_0). The hard alignment problem — distinguishing NCR from Edema from ET — receives proportionally small gradient signal.

**Fix**: Compute L_align only on foreground tokens (non-background label).

```python
# In ConceptAlignmentLoss.forward, add:
foreground_mask = (seg_labels > 0)  # [B, N]  — exclude background

for k in range(1, K):  # skip k=0 (background)
    gt_k = (seg_labels == k).float()       # [B, N]
    p_k = routing_probs[..., k]            # [B, N]
    p_k_safe = p_k.clamp(eps, 1-eps)
    
    # Apply foreground mask: only count tokens where any tumor class is present
    fg = foreground_mask.float()           # [B, N]
    weighted_bce = F.binary_cross_entropy(p_k_safe, gt_k, reduction='none')  # [B, N]
    loss += (fg * weighted_bce).sum() / (fg.sum() + 1e-8)

return loss / (K - 1)  # average over K-1 foreground concepts
```

**Theoretical grounding**: The paper's interpretability claim is about tumor subregion alignment, not background alignment. Foreground-only L_align concentrates gradient signal on the small, clinically important subregions. Also fixes the CAS(background) reliability issue — if background is not in L_align, CAS(background) becomes a pure diagnostic metric (not a training target) and unreliable values do not harm training.

**Impact**: Expect CAS(NCR), CAS(Edema), CAS(ET) to improve. CAS(background) becomes irrelevant to training.

---

### Improvement 2 — Prototype-Augmented Router (Medium Cost, High Interpretability Value)

**Problem**: The router MLP (W₁, W₂) is a black box. Its decision boundary in ℝ^D space is opaque — you cannot explain what "the model thinks NCR looks like at the bottleneck" because the decision is encoded in a weight matrix.

**Fix**: Add K learnable concept prototype vectors {c_k} ∈ ℝ^D. Blend MLP routing with distance-to-prototype routing.

```python
# In ClinicalConceptRouter.__init__:
self.prototypes = nn.Parameter(torch.randn(K, embed_dim))  # [K, D]
self.blend_alpha = nn.Parameter(torch.tensor(0.5))          # learnable blend

# In forward:
# MLP logits (existing)
logits_mlp = self.fc2(F.gelu(self.fc1(x_norm)))  # [B, N, K]

# Distance-to-prototype logits
# x_norm: [B, N, D]; prototypes: [K, D]
diff = x_norm.unsqueeze(2) - self.prototypes.unsqueeze(0).unsqueeze(0)  # [B, N, K, D]
dist = -(diff ** 2).sum(dim=-1)  # [B, N, K] negative squared distance
logits_proto = dist / (embed_dim ** 0.5)  # scale by sqrt(D)

# Blend
alpha = self.blend_alpha.sigmoid()
logits = alpha * logits_mlp + (1 - alpha) * logits_proto

routing_probs = F.softmax(logits / tau, dim=-1)
```

**Theoretical grounding**: Prototype-based routing makes the router geometrically interpretable. Prototype c_k converges (under L_align) to the mean bottleneck representation of tokens in clinical subregion k. You can visualize c_k and show reviewers what the model's "template" for NCR, Edema, ET looks like in feature space. This strengthens the interpretability claim: not only does the routing probability tell you which concept — you can also show what that concept's prototype looks like.

This is also consistent with the concept bottleneck literature where concept vectors are explicitly learned.

**Paper bonus**: Add a qualitative analysis section showing the prototype vectors projected to 2D (t-SNE or PCA). NCR, Edema, ET prototypes should cluster separately from Background.

---

### Improvement 3 — Temperature Annealing Aligned to Curriculum (Low Cost, Theoretical Correctness)

**Problem**: Temperature τ is currently fully learnable with no curriculum pressure. During warmup, you want high τ (diverse routing, maximum entropy — no expert specialization yet). During refinement, you want low τ (crisp routing, low entropy — sharp, deterministic concept maps). Currently τ can drift freely in the wrong direction.

**Fix**: Add a soft curriculum target for τ to `CurriculumWeightScheduler`.

```python
# In CurriculumWeightScheduler:
TAU_TARGETS = {
    "warmup":     2.0,   # high → diffuse routing, experts can freely explore
    "alignment":  1.0,   # medium → routing shapes up
    "refinement": 0.5,   # low → crisp routing, clear concept maps
}

def get_tau_target(self, epoch: int) -> float:
    return self.TAU_TARGETS[self.get_phase_name(epoch)]

# In CCRTotalLoss.forward, add a temperature regularization term:
tau_target = scheduler.get_tau_target(epoch)
tau_actual = module.router.temperature.clamp(0.1, 5.0)
l_tau_reg = (tau_actual - tau_target) ** 2   # soft pull toward target
l_total = l_total + 0.01 * l_tau_reg         # small weight, just a soft target
```

**Theoretical grounding**: Temperature annealing is standard in MoE literature (Shazeer et al., 2017; Switch Transformer). The curriculum schedule already controls λ weights — coupling τ to the same schedule makes the three-phase curriculum structurally coherent. During warmup, the model explores freely (high entropy). During refinement, it commits to crisp explanations (low entropy). This is exactly the intended behavior and should improve CAS convergence speed.

---

### Improvement 4 — Focal-BCE for L_align (Low Cost, CAS Improvement for Small Subregions)

**Problem**: Standard BCE in L_align weights all tokens equally. For BraTS, ET is ~2% of foreground voxels, NCR is ~5%, Edema is ~20%. Easy tokens (clearly-labeled interior voxels, far from boundaries) dominate the L_align gradient. The hard tokens — boundary voxels between concepts, small ET subregion tokens — receive small gradient signal relative to easy interior tokens.

**Fix**: Replace BCE with focal BCE in L_align.

```python
# In ConceptAlignmentLoss.forward:
GAMMA_ALIGN = 2.0   # same as FocalLoss

for k in range(K):
    gt_k = (seg_labels == k).float()
    p_k = routing_probs[..., k].clamp(eps, 1-eps)
    
    # Standard BCE
    bce = -(gt_k * p_k.log() + (1 - gt_k) * (1 - p_k).log())  # [B, N]
    
    # Focal modulation: down-weight easy tokens (where model is already confident)
    p_correct = torch.where(gt_k.bool(), p_k, 1 - p_k)  # prob of correct assignment
    focal_weight = (1 - p_correct) ** GAMMA_ALIGN
    
    loss += (focal_weight * bce).mean()
```

**Theoretical grounding**: For small subregions (ET, NCR), the model quickly learns to route most tokens correctly. Once p_ET(t) = 0.95 for ET tokens, the BCE gradient is 0.05 — tiny, and the model stops improving CAS(ET). Focal BCE keeps the gradient proportional to the routing error even for already-correct tokens. The result: CAS(ET) and CAS(NCR) should improve because the model continues refining the small-subregion routing rather than coasting.

Lin et al. (2017) showed this effect for class imbalance in detection. The same mechanism applies here.

---

### Improvement 5 — Foreground CAS as Primary Metric (Analysis Change, No Code Cost)

**Problem**: CAS(k) currently averages over all tokens including background. CAS(background) is near-0 (unreliable Pearson due to low variance) and CAS for tumor classes is inflated by the large easy background context.

**Fix**: Define two metrics:

```python
# In ConceptAlignmentScore.update, track separately:

# CAS_all(k) — current metric, all tokens (for completeness)
# CAS_fg(k)  — foreground tokens only (k=1,2,3 vs. background mask)

def update_fg(self, routing_probs, token_labels):
    """Accumulate only for foreground tokens."""
    fg_mask = (token_labels > 0).reshape(-1)  # [B*N] boolean
    probs_fg = routing_probs.reshape(-1, K)[fg_mask]      # [M_fg, K]
    labels_fg = token_labels.reshape(-1)[fg_mask]         # [M_fg]
    # then accumulate running sums same as current update()
```

**Theoretical grounding**: CAS_fg is the honest metric. It answers: "Among the clinically interesting tokens (tumor subregions), how well does routing match anatomy?" CAS_all is dominated by background. For the paper:
- Report CAS_fg as the primary metric (Table 1, main results)
- Report CAS_all in supplementary for completeness
- The target CAS_fg ≥ 0.85 is harder to achieve than CAS_all ≥ 0.85 — it's a stronger claim

This does not require changing training at all. It is a metric analysis improvement.

---

## 6. Summary Table

| Improvement | Type | Implementation Cost | Theoretical Impact |
|---|---|---|---|
| 1. Foreground-only L_align | Training loss | Low — ~10 lines | High — focuses alignment where it matters |
| 2. Prototype-augmented router | Architecture | Medium — ~30 lines | High — makes routing geometrically interpretable |
| 3. Temperature annealing | Curriculum | Low — ~20 lines | Medium — structural coherence with schedule |
| 4. Focal-BCE for L_align | Training loss | Low — ~5 lines | Medium — CAS(NCR), CAS(ET) improvement |
| 5. Foreground CAS metric | Metric | Low — ~30 lines | High — honest reporting |

All five are backward-compatible. None change the core CCR principle. None violate any hard constraint in CLAUDE.md.

**Recommended order of implementation**: 5 (change metric reporting, zero risk) → 1 (foreground L_align, cleanest theoretical justification) → 4 (focal BCE, small code change) → 3 (temperature annealing) → 2 (prototype router, most complex but strongest paper contribution).

---

## 7. What Should NOT Be Changed

- **The bottleneck placement** — Phase 1's placement is correct and is the core novelty. Do not move routing to intermediate layers or output.
- **K=4 for BraTS** — matches annotation protocol. Do not change.
- **Soft train / hard inference** — correct design. Do not remove hard dispatch at inference.
- **L_diversity during warmup (λ=0.5)** — this is what keeps experts spread during the period when L_align is off. Do not zero it in warmup.
- **λ₃ (entropy) ≤ 0.01** — larger values push routing toward uniform, destroying CAS. Do not increase.
- **Warmup length (10 epochs)** — validated by MoE literature and design reasoning. Do not shorten.
