# CCR Phase 1 — Architecture and Code Reference

**Package**: `src/ccr/`
**Purpose**: Clinical Concept Routing bottleneck module — router, experts, losses, metrics
**Plan reference**: CCR-Net_Research_Plan.md Sections 5–7

---

## 1. Overview

Phase 1 implements the core CCR module that is plugged into the encoder bottleneck of any
3D medical image segmentation model.  The module replaces the bottleneck with a routing layer
that assigns each image token to one of K clinically-labeled expert networks.

**The fundamental claim**: routing probability P(t→k) IS the per-voxel explanation — not
a post-hoc approximation.  It is causally upstream of all decoder computation.

---

## 2. Package Structure

```
src/ccr/
├── __init__.py                  Top-level API: CCRConfig, CCRBottleneckModule, CCRTotalLoss
├── config/
│   ├── __init__.py
│   └── ccr_config.py            All hyperparameters as typed dataclasses
├── modules/
│   ├── __init__.py
│   ├── router.py                ClinicalConceptRouter  — routing MLP
│   ├── expert.py                ClinicalConceptExpert  — per-concept residual MLP
│   └── dispatcher.py            CCRBottleneckModule    — router + K experts combined
├── losses/
│   ├── __init__.py
│   ├── segmentation.py          DiceLoss + FocalLoss + SegmentationLoss
│   ├── alignment.py             ConceptAlignmentLoss   — L_align (the core CCR loss)
│   ├── diversity.py             ExpertDiversityLoss    — L_diversity
│   ├── entropy.py               EntropyRegularizationLoss — L_entropy_reg
│   ├── boundary.py              BoundaryAwareLoss      — L_boundary
│   └── total.py                 CCRTotalLoss           — curriculum-weighted sum
└── utils/
    ├── __init__.py
    ├── weight_schedule.py       CurriculumWeightScheduler
    └── metrics.py               ConceptAlignmentScore, ExpertUtilizationTracker

tests/
├── conftest.py                  Shared pytest fixtures (BraTS-like shapes)
├── test_router.py
├── test_expert.py
├── test_losses.py
├── test_dispatcher.py
└── integration/
    └── test_pipeline.py         Full training step simulation
```

---

## 3. Data Flow

```
Input: bottleneck_tokens  [B, N, D]
  │
  ▼
ClinicalConceptRouter
  LayerNorm → Linear(D, H) → GELU → Linear(H, K) → / τ → Softmax
  │
  ├── routing_probs  [B, N, K]   ← THE EXPLANATION (P(t→k))
  ├── entropy        [B, N]      ← uncertainty (Proposition 2)
  └── logits         [B, N, K]   ← pre-softmax values
  │
  ▼
CCRBottleneckModule._soft_dispatch (train) / _hard_dispatch (eval)
  │
  ├── soft:  expert_output(t) = Σ_k P(t→k) · Expert_k(X[t])   differentiable
  └── hard:  expert_output(t) = Expert_{argmax k P(t→k)}(X[t]) not differentiable
  │
  ▼
expert_outputs  [B, N, D]   ← concept-conditioned bottleneck features
  │
  ▼  (fed to CCR-Net boundary head OR CCR-Retrofit decoder)
Final segmentation  [B, K, H, W, D]
```

---

## 4. Mathematical Reference

### 4.1 Routing (ClinicalConceptRouter)

```
X̂       = LayerNorm(X)                         [B, N, D]
H        = GELU(X̂ @ W₁ + b₁)                  [B, N, H],  H = D × factor
logits   = H @ W₂ + b₂                         [B, N, K]
τ        = clamp(τ_param, 0.1, 5.0)            scalar
P(t→k)   = softmax(logits / τ, dim=-1)         [B, N, K]
H(t)     = -Σ_k P(t→k) log(P(t→k) + ε)        [B, N]

Σ_k P(t→k) = 1  for all b, t  (probability simplex)
H(t) ∈ [0, log K]
```

### 4.2 Expert Dispatch (ClinicalConceptExpert)

```
Expert_k architecture (residual MLP):
    correction = fc2(GELU(fc1(LayerNorm(X))))   [M, D]
    output     = X + correction                  [M, D]

Soft dispatch (train):
    expert_out(b,t) = Σ_k P(b,t→k) · Expert_k(X[b,t])

Hard dispatch (eval):
    k*(b,t)         = argmax_k P(b,t→k)
    expert_out(b,t) = Expert_{k*(b,t)}(X[b,t])
```

### 4.3 L_seg (SegmentationLoss)

```
L_seg = 0.5 · L_dice + 0.5 · L_focal

Dice (macro-averaged):
    p_c = softmax(logits)[..., c]              [B, *spatial]
    y_c = (labels == c).float()
    dice_c = (2 Σ(p_c · y_c) + ε) / (Σ p_c² + Σ y_c² + ε)
    L_dice = 1 - (1/K) Σ_c dice_c

Focal (Lin et al., 2017):
    p_t     = softmax(logits) at GT class      [B, *spatial]
    FL(t)   = -α (1 - p_t)^γ log(p_t + ε)
    L_focal = mean FL(t)

Params: γ=2.0, α=0.25, ε=1e-5
```

### 4.4 L_align (ConceptAlignmentLoss)

```
gt_k(b,t) = 1 if seg_labels[b,t] == k, else 0    [B, N]
p_k(b,t)  = routing_probs[b,t,k]                  [B, N]

L_align = (1/K) Σ_k BCE(p_k, gt_k)

This is the loss that makes routing interpretable.
Training to minimise L_align forces P(t→k) → 1 for tokens in subregion k
and P(t→k) → 0 for tokens outside it.
After convergence: CAS(k) = Pearson(P(t→k), M_k(t)) ≥ 0.85.
```

### 4.5 L_diversity (ExpertDiversityLoss)

```
mean_k   = (1/BN) Σ_{b,t} P(b,t→k)               [K]
load_bal = (1/K) Σ_k (mean_k - 1/K)²

flat     = P.reshape(B*N, K).T                     [K, B*N]
G        = flat @ flat.T                           [K, K]  (Gram matrix)
cos[i,j] = G[i,j] / (||F_i|| · ||F_j|| + ε)
cos_ovlp = mean of cos[i≠j]

L_diversity = load_bal + cos_ovlp
```

### 4.6 L_entropy_reg (EntropyRegularizationLoss)

```
H(b,t) = -Σ_k P(b,t→k) log(P(b,t→k) + ε)    [B, N]
L_ent  = mean_{b,t} H(b,t)

λ₃ = 0.01  (kept small — see note below)
```

**Important**: λ₃ must remain small (0.01).  Larger values push routing toward
uniform distribution, destroying both CAS and the calibration property (Proposition 2).

### 4.7 L_boundary (BoundaryAwareLoss)

```
Boundary detection (GPU-compatible, no scipy):
    one_hot: labels → [B, K, H, W, D]
    dilated = max_pool3d(one_hot, 3, padding=1)
    eroded  = -max_pool3d(-one_hot, 3, padding=1)
    boundary = (|dilated - eroded| > 0.5).any(dim=1)   [B, H, W, D]
    (optional: widen with additional dilation passes)

Weight map:
    weight(t) = base_weight + boost_factor · is_boundary(t)
    Default: interior = 1.0, boundary = 1.0 + 5.0 = 6.0

Loss:
    L_boundary = mean [ weight(t) · CE(pred_logits, labels) ]
```

### 4.8 Total Loss Curriculum

```
L_total = L_seg + λ₁ L_align + λ₂ L_diversity + λ₃ L_entropy + λ₄ L_boundary

Phase       | Epochs  | λ₁(align) | λ₂(div) | λ₃(ent) | λ₄(bound)
------------|---------|-----------|---------|---------|----------
Warmup      | 1-10    | 0.0       | 0.5     | 0.00    | 0.0
Alignment   | 11-50   | 1.0       | 0.5     | 0.01    | 0.3
Refinement  | 51-80   | 0.5       | 0.1     | 0.01    | 1.0
```

---

## 5. CAS — Concept Alignment Score

```
CAS(k) = Pearson(P(t→k|x),  M_k(t))

where:
  P(t→k|x) = routing probability for concept k at token t, given image x
  M_k(t)   = 1 if GT label at token t is k, else 0

Pearson computed using running sums (no memory blowup over large val sets):
  CAS = (n ΣPM - ΣP ΣM) / (sqrt(n ΣP² - (ΣP)²) · sqrt(n ΣM² - (ΣM)²) + ε)

Target: CAS(k) ≥ 0.85 for all k ∈ {0,1,2,3} on BraTS validation set.
CAS = faithfulness for CCR-Net (Proposition 1a).
```

---

## 6. ExpertUtilizationTracker

Monitors the fraction of tokens routed to each expert each epoch.

**Collapse threshold**: If any expert receives < 5 % of tokens after epoch 20,
`check_collapse()` emits a RuntimeWarning and returns the collapsed expert names.

**Action on collapse**: Re-initialise the collapsed expert's weights.  Do not
reduce K — keeping K = 4 is required for matching BraTS annotation protocol.

---

## 7. Configuration Guide

All hyperparameters are in `src/ccr/config/ccr_config.py`.

```python
from ccr import CCRConfig

# Default BraTS configuration (K=4, embed_dim=192)
cfg = CCRConfig()

# LiTS configuration (K=3, different concept names)
from ccr.config.ccr_config import RouterConfig, ExpertConfig
cfg_lits = CCRConfig(
    router        = RouterConfig(num_concepts=3),
    expert        = ExpertConfig(num_concepts=3),
    concept_names = ("background", "liver", "liver_tumor"),
)

# Change embed_dim for a different backbone (e.g., 768 for ViT-L)
cfg_vit_l = CCRConfig(
    router = RouterConfig(embed_dim=768, num_concepts=4),
    expert = ExpertConfig(embed_dim=768, num_concepts=4),
)
```

---

## 8. How to Run All Tests

From the `AMBETS/` directory:

```bash
# Install dependencies
pip install -r requirements_phase1.txt

# Run all tests with verbose output
python -m pytest tests/ -v --tb=short

# Run with coverage report
python -m pytest tests/ -v --cov=src/ccr --cov-report=term-missing

# Run only unit tests (skip integration)
python -m pytest tests/ -v --ignore=tests/integration/

# Run only the integration pipeline
python -m pytest tests/integration/test_pipeline.py -v

# Run a specific test class
python -m pytest tests/test_router.py::TestProbabilitySimplex -v
```

---

## 9. Key Invariants to Never Break

| Invariant | Where enforced | What breaks if violated |
|---|---|---|
| Σ_k P(t→k) = 1 | Softmax in router | CAS undefined; L_align gradients unstable |
| L_align = 0 during warmup (epochs 1-10) | CurriculumWeightScheduler | Expert collapse before encoder learns features |
| λ₃ (entropy) ≤ 0.01 | LossConfig default | Routing collapses to uniform; Proposition 2 fails |
| CCR at bottleneck only | Architectural decision | Post-hoc if placed at output; not intrinsic |
| K=4 for BraTS | CCRConfig.concept_names | Mismatch with BraTS annotation protocol |
| Hard routing at inference | CCRConfig.hard_routing_inference | Crisp explanation maps unavailable |

---

*Phase 1 complete. Next: Phase 2 — CCR-Net full architecture and training loop.*
