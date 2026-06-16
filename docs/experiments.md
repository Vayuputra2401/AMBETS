# CCR-Net Experiments Reference

**Purpose**: Complete specification of every experiment required for A* submission (MICCAI 2026 + NeurIPS 2026). Each entry states the question being answered, the exact protocol, which metrics to track, what result makes the experiment a success, and which proposition or claim it validates.

**Execution order**: Phase 2 (CCR-Net training) → Phase 3 (CCR-Retrofit) → Phase 4 (main experiments) → Phase 5 (generalization + agent). Ablations run alongside Phase 4.

**Datasets**:
- Primary: BraTS 2021 (1001 train / 125 val / 125 test)
- Cross-year: BraTS 2023 / 2024 (OOD validation, no retraining)
- Generalization: LiTS (K=3 retrain, different organ and modality)

---

## Table of Contents

1. [Training Monitoring (per epoch)](#1-training-monitoring-per-epoch)
2. [Experiment 1 — CAS Measurement](#2-experiment-1--cas-measurement-proposition-1a--1b-prerequisite)
3. [Experiment 2 — Faithfulness (Primary Claim)](#3-experiment-2--faithfulness-primary-claim)
4. [Experiment 3 — Segmentation Accuracy](#4-experiment-3--segmentation-accuracy)
5. [Experiment 4 — Uncertainty Calibration (Proposition 2)](#5-experiment-4--uncertainty-calibration-proposition-2)
6. [Experiment 5 — LLM Hallucination Rate (REP)](#6-experiment-5--llm-hallucination-rate-rep)
7. [Experiment 6 — Generalization to LiTS](#7-experiment-6--generalization-to-lits)
8. [Ablation Studies](#8-ablation-studies)
9. [Paper Tables and Figures](#9-paper-tables-and-figures)
10. [Success Criteria for A* Submission](#10-success-criteria-for-a-submission)

---

## 1. Training Monitoring (per epoch)

These metrics are tracked during training of every model (CCR-Net and all CCR-Retrofit variants). They are not reported as paper results — they are training health indicators. Log them to console and save to checkpoint dict.

### Metrics to log every checkpoint epoch

| Metric | How to compute | Warning threshold | Action |
|--------|---------------|-------------------|--------|
| `CAS_fg(NCR)` | `cas.compute_fg()["necrotic_core"]` | < 0.5 at epoch 50 | check L_align weight, check expert collapse |
| `CAS_fg(Edema)` | `cas.compute_fg()["edema"]` | < 0.5 at epoch 50 | same |
| `CAS_fg(ET)` | `cas.compute_fg()["enhancing_tumor"]` | < 0.5 at epoch 50 | same |
| `L_align` | `losses["align"].item()` | rising after epoch 20 | expert collapse — reinit |
| `L_total` | `losses["total"].item()` | spikes > 2× previous | gradient explosion — lower LR |
| `Dice(WT)` | hard-argmax pred on val | < 0.80 at epoch 50 | check seg loss |
| `Dice(TC)` | hard-argmax pred on val | < 0.65 at epoch 50 | check seg loss |
| `Dice(ET)` | hard-argmax pred on val | < 0.60 at epoch 50 | check seg loss |
| Expert utilization % | `tracker.compute()` | any expert < 10% | warn; < 5% after epoch 20 → reinit |
| `τ_actual` | `model.ccr.router.temperature.item()` | > 3.0 or < 0.2 | temperature annealing not working |
| `τ_target` | `losses["tau_target"]` | — | should decrease phase by phase |
| Current phase | `losses["phase"]` | — | confirm phase transitions at correct epochs |

### L_align convergence signature (healthy training)

```
Epochs 1–10  : L_align = 0.0 (warmup — disabled)
Epoch 11     : L_align jumps to ~0.40–0.60 (first activation)
Epochs 11–20 : L_align decreasing rapidly (~0.05/epoch)
Epochs 20–50 : L_align decreasing slowly (~0.005/epoch)
Epoch 50     : L_align should be < 0.15
Epochs 51–80 : L_align stable or slight further decrease

CAS_fg should mirror this: rises sharply 11–30, plateaus 30–80.
```

If L_align is rising or flat after epoch 20, an expert has collapsed. Run `tracker.compute()` — any expert below 5% needs reinit.

---

## 2. Experiment 1 — CAS Measurement (Proposition 1a / 1b prerequisite)

**Question**: Do routing probabilities actually align with clinical subregion labels? Is the CCR module learning concept-specific routing, or is it routing arbitrarily?

**This is the foundational experiment**. Every other claim rests on CAS ≥ 0.85. Run this first, on both BraTS 2021 val and BraTS 2023 OOD, before reporting any faithfulness numbers.

### 1a. CAS on BraTS 2021 Validation

**Models**: CCR-Net, CCR-Retrofit-nnUNet, CCR-Retrofit-SwinUNETR, CCR-Retrofit-TransBTS

**Protocol**:
```python
cas = ConceptAlignmentScore(num_concepts=4, concept_names=(...))
for batch in val_loader:
    out = model(batch["image"])
    token_labels = downsample_labels_to_tokens(batch["label"], model.get_grid_shape())
    cas.update(out["routing_probs"], token_labels)
cas_fg  = cas.compute_fg()   # PRIMARY: report these
cas_all = cas.compute()      # SUPPLEMENTARY: report in appendix
```

**Metrics to report**:
- `CAS_fg(NCR)`, `CAS_fg(Edema)`, `CAS_fg(ET)` — primary metric, foreground-only Pearson
- `CAS_all(NCR)`, `CAS_all(Edema)`, `CAS_all(ET)` — supplementary (background inflates these)
- Do NOT report `CAS_fg(Background)` — background label variance is near zero → Pearson is unreliable

**Additional CAS-related metrics**:
- AUROC of P(t → k) as binary classifier for subregion k (one vs. rest, all tokens)
  - Threshold-free version of CAS; easier to interpret
  - Compute with `sklearn.metrics.roc_auc_score` on flattened routing_probs vs token_labels
- Dice of hard routing map vs. GT subregion mask (argmax routing as binary segmenter)
  - Shows routing is not just correlated — it can segment without the decoder

**Expected**:
```
CAS_fg(NCR)   ≥ 0.85
CAS_fg(Edema) ≥ 0.85
CAS_fg(ET)    ≥ 0.85
AUROC(NCR)    ≥ 0.90
AUROC(Edema)  ≥ 0.90
AUROC(ET)     ≥ 0.88   (ET is smallest, hardest)
```

**If CAS_fg < 0.70 for any expert**: Training failure. Check: (a) L_align activated at correct epoch, (b) no expert collapse, (c) encoder is not frozen in a bad local minimum.

### 1b. CAS on BraTS 2023 (Cross-Year OOD)

**Protocol**: Load trained CCR-Net checkpoint. Run inference on BraTS 2023 val split. No retraining.

**Metrics**: Same as above (CAS_fg per concept)

**Expected**: CAS_fg ≥ 0.80 on OOD data. Degradation < 0.07 vs. in-distribution.

**What this proves**: Routing semantics are not memorized. The model learned generalizable concept representations that transfer across scanner settings and preprocessing variants.

### 1c. AUROC per Concept (Detailed)

Report AUROC per concept, per model. Shows the routing is a reliable binary classifier for each subregion.

| Model | AUROC(NCR) | AUROC(Edema) | AUROC(ET) |
|-------|-----------|-------------|----------|
| CCR-Net | | | |
| CCR-Retrofit-nnUNet | | | |
| CCR-Retrofit-SwinUNETR | | | |
| CCR-Retrofit-TransBTS | | | |

### 1d. Ablation: No L_align

**Protocol**: Train CCR-Net with all losses EXCEPT L_align (λ₁=0 throughout all 80 epochs).

**Expected**: CAS_fg drops to ~0.10–0.25 (near chance for K=4 uniform routing).

**What this proves**: L_align is the mechanism. Routing doesn't self-organize into clinical concepts without explicit alignment supervision. The CAS improvement in the trained model is causal, not correlational.

---

## 3. Experiment 2 — Faithfulness (Primary Claim)

**Question**: Is routing-based explanation more faithful than any post-hoc attribution method on the same backbones?

**This experiment validates Propositions 1a and 1b and is the centerpiece of the paper.**

### 2a. Post-hoc Baselines to Implement

Apply each to the UNMODIFIED backbone (no CCR), using the exact same backbone architecture as the CCR-Retrofit variant:

| Baseline | Apply to | Library |
|----------|----------|---------|
| GradCAM | nnU-Net (CNN) | pytorch-grad-cam |
| Integrated Gradients | nnU-Net (CNN) | Captum |
| KernelSHAP | nnU-Net (CNN) | SHAP (sample 500 voxels) |
| Attention Rollout | Swin-UNETR | custom (follow Abnar 2020) |
| TokenTM | Swin-UNETR | arXiv:2403.14552 reference implementation |
| GradCAM | Swin-UNETR | pytorch-grad-cam |

Note: apply GradCAM to both CNN and transformer backbone to show it degrades across architectures.

### 2b. Faithfulness Metrics

**Deletion AUC** (primary faithfulness metric, following March 2025 benchmark methodology):

```
Protocol:
  For each case in BraTS 2021 test set:
    For each concept k ∈ {NCR, Edema, ET}:
      1. Get explanation map E_k(x) for concept k (routing prob or saliency)
      2. Sort voxels by E_k descending (most important first)
      3. For thresholds τ ∈ {0.05, 0.10, ..., 0.50}:
           mask = top-τ fraction of voxels by E_k
           x_masked = replace mask with baseline (mean training intensity per modality)
           P_k_masked = model(x_masked).softmax().mean over mask region
           record (τ, drop = P_k_original - P_k_masked)
      4. AUC of drop vs. τ curve
  Average deletion AUC over all cases and concepts

For CCR-Net: use hard-dispatch routing map (model.eval()) as E_k.
For CCR-Retrofit: use routing_probs[:, :, k] reshaped to spatial as E_k.
For post-hoc: use GradCAM/IG/SHAP output as E_k.

Critical: use model.eval() for all CCR methods (hard routing).
The faithfulness claim is about INFERENCE-TIME behavior.
```

**Insertion AUC**:
```
Start from blurred/mean image.
Insert voxels flagged by E_k (most important first).
Measure confidence rise.
AUC of rise vs. fraction inserted.
Higher = more faithful.
```

**Infidelity score** (from Yeh et al. 2019, implemented in Captum):
```
E[||<I(x+η) - I(x), explanation> - (f(x+η) - f(x))||²]
where η is random perturbation
Lower infidelity = more faithful
```

**Sensitivity-n** (how well the explanation predicts the output change under n-feature ablation):
```
For n ∈ {10, 50, 100, 500}:
  Remove n random subsets of voxels
  Measure correlation between Σ_i∈S E_k(i) and f(x) - f(x_S)
Higher correlation = more faithful
```

### 2c. Expected Results Table

| Method | Type | Backbone | Del.AUC ↑ | Ins.AUC ↑ | Infidelity ↓ |
|--------|------|----------|----------|----------|-------------|
| GradCAM | Post-hoc | nnU-Net | ~0.58 | ~0.61 | high |
| Integrated Gradients | Post-hoc | nnU-Net | ~0.60 | ~0.62 | high |
| KernelSHAP | Post-hoc | nnU-Net | ~0.61 | ~0.63 | high |
| Attention Rollout | Post-hoc | Swin-UNETR | ~0.59 | ~0.62 | high |
| TokenTM | Post-hoc | Swin-UNETR | ~0.63 | ~0.64 | moderate |
| GradCAM | Post-hoc | Swin-UNETR | ~0.60 | ~0.62 | high |
| **CCR-Net** | **Intrinsic** | Swin-B | **~0.89** | **~0.87** | **low** |
| **CCR-Retrofit** | **Intrinsic** | nnU-Net | **~0.85** | **~0.84** | **low** |
| **CCR-Retrofit** | **Intrinsic** | Swin-UNETR | **~0.84** | **~0.83** | **low** |
| **CCR-Retrofit** | **Intrinsic** | TransBTS | **~0.83** | **~0.82** | **low** |

The gap (≈ 0.23–0.27 on deletion AUC) is the main result.

### 2d. Verify Proposition 1a (CCR-Net)

**Claim**: `Faithfulness_CCRNet = CAS_fg` (equality, not just ≥)

**Check**: After running Experiment 1 (CAS) and Experiment 2 (Del AUC) on the same val split:
```
|Del_AUC(CCR-Net) - CAS_fg_mean| < 0.03   (within measurement noise)
```

If this holds, Proposition 1a is empirically verified.

### 2e. Verify Proposition 1b (CCR-Retrofit)

**Claim**: `Faithfulness_Retrofit ≥ CAS_fg × (1 - DD)`

**Steps**:
1. Measure `CAS_fg` for each Retrofit model (from Experiment 1)
2. Measure Decoder Divergence (DD):
   ```python
   # DD(k) = 1 - Pearson(softmax(decoder_output)[:, k], routing_probs[:, k])
   # where both are computed at token resolution (16³ grid)
   # downsample decoder output to 16³, compare with routing_probs
   decoder_probs_at_grid = F.interpolate(
       softmax(seg_logits), size=grid_shape, mode="trilinear"
   )  # [B, K, 16, 16, 16]
   routing_probs_spatial = routing_probs.transpose(1,2).reshape(B, K, h, w, d)
   DD_k = 1 - pearson(decoder_probs_at_grid[:, k], routing_probs_spatial[:, k])
   ```
3. Check: `Del_AUC(Retrofit_k) ≥ CAS_fg(k) × (1 - DD(k))` for all k

Report the bound alongside actual faithfulness in the paper. If the bound is tight (actual ≈ bound), Proposition 1b is confirmed.

---

## 4. Experiment 3 — Segmentation Accuracy

**Question**: Does CCR degrade backbone segmentation accuracy?

**This is the cost-accounting experiment.** The paper claims CCR adds interpretability at < 1% Dice cost. This experiment verifies that claim.

### 3a. Primary Accuracy Table

**Metrics**: Dice (WT/TC/ET), HD95 (WT/TC/ET) on BraTS 2021 test set.

Standard BraTS region definitions:
```
WT = Whole Tumor  = pred ∈ {1, 2, 3}
TC = Tumor Core   = pred ∈ {1, 3}
ET = Enhancing T  = pred == 3
```

**Models to evaluate** (BraTS 2021 test set, 125 cases):

| Model | Dice WT | Dice TC | Dice ET | HD95 WT | HD95 TC | HD95 ET |
|-------|---------|---------|---------|---------|---------|---------|
| nnU-Net (no CCR) | | | | | | |
| CCR-Retrofit-nnUNet | | | | | | |
| Swin-UNETR (no CCR) | | | | | | |
| CCR-Retrofit-SwinUNETR | | | | | | |
| TransBTS (no CCR) | | | | | | |
| CCR-Retrofit-TransBTS | | | | | | |
| CCR-Net | | | | | | |

**Success criterion**: CCR-Retrofit Dice within 1% of unmodified backbone for WT, TC, ET.

**Note on CCR-Net accuracy**: CCR-Net is not claiming to beat nnU-Net. It is a new model where faithfulness is the primary constraint. Expect CCR-Net Dice to be 2–4% below SwinUNETR. This is acceptable — report it honestly. The paper's claim is interpretability, not SOTA accuracy.

### 3b. Per-Retrofit Training Mode Comparison (nnU-Net)

Run three training modes for CCR-Retrofit applied to nnU-Net:

| Training Mode | CAS_fg | Dice(WT) | DD | Description |
|--------------|--------|---------|-----|-------------|
| Mode A: Frozen backbone | | | | Fast, preserves backbone exactly |
| Mode B: Joint finetune | | | | Encoder adapts to routing |
| Mode C: Pretrain+finetune | | | | Mode A init → Mode B at lower LR |

**Expected**: Mode C has best CAS/accuracy tradeoff. Use Mode C for all other backbones if confirmed.

### 3c. Lesion-Wise Dice (Supplementary)

Standard BraTS evaluation includes per-lesion Dice (instances, not voxels). Compute and report in supplementary table.

---

## 5. Experiment 4 — Uncertainty Calibration (Proposition 2)

**Question**: Is routing entropy a better-calibrated uncertainty estimate than MC-Dropout?

**Claim (Proposition 2)**: `ECE(routing entropy) < ECE(MC-Dropout, T=10)` with zero inference overhead.

### 4a. Methods to Compare

| Method | Passes | Cost |
|--------|--------|------|
| CCR routing entropy | 1 | 0 overhead |
| MC-Dropout T=5 | 5 | 5× |
| MC-Dropout T=10 | 10 | 10× |
| MC-Dropout T=20 | 20 | 20× |
| Deep Ensembles N=3 | 3 | 3× models |
| Deep Ensembles N=5 | 5 | 5× models |

MC-Dropout baseline: take any trained backbone (nnU-Net), enable dropout at inference, run T stochastic forward passes, compute entropy of prediction distribution across passes.

Deep Ensembles: train 3 and 5 independent nnU-Net models with different random seeds, compute entropy of ensemble predictive distribution.

### 4b. Calibration Protocol

**Expected Calibration Error (ECE)**:
```python
def compute_ece(uncertainty_map, error_map, n_bins=15):
    """
    uncertainty_map: [H, W, D] normalized to [0, 1] (entropy / log(K))
    error_map:       [H, W, D] binary (1 where prediction is wrong)
    Partition into bins by uncertainty level. For each bin:
        accuracy = fraction of correct predictions in bin
        confidence = 1 - mean uncertainty in bin
        |accuracy - confidence| = calibration error for bin
    ECE = weighted mean of calibration errors
    """
```

Compute per-concept ECE (NCR, Edema, ET) and overall ECE.

**Reliability Diagrams** (Figure 4 in paper):
- X axis: uncertainty bin (0 → max entropy)
- Y axis: actual error rate in that bin
- Perfectly calibrated: diagonal line
- Overconfident (underestimates uncertainty): curve above diagonal at low uncertainty
- Plot one curve per method

**AURC (Area Under Risk-Coverage curve)**:
- Sort all voxels by uncertainty
- For each coverage threshold (fraction of voxels with uncertainty below threshold):
  - Risk = error rate on covered voxels
- Lower AURC = uncertainty better correlates with where the model is wrong

### 4c. Expected Results

| Method | ECE(overall) | ECE(ET) | AURC | Inference cost |
|--------|-------------|---------|------|----------------|
| CCR routing entropy | **< 0.06** | **< 0.08** | **low** | **0×** |
| MC-Dropout T=5 | ~0.09 | ~0.11 | moderate | 5× |
| MC-Dropout T=10 | ~0.07 | ~0.09 | moderate | 10× |
| MC-Dropout T=20 | ~0.07 | ~0.09 | moderate | 20× |
| Deep Ensembles N=3 | ~0.06 | ~0.08 | moderate | 3× models |
| Deep Ensembles N=5 | ~0.05 | ~0.07 | low | 5× models |

**Success criterion for Prop 2**: `ECE(routing entropy) < ECE(MC-Dropout T=10)`. Deep Ensembles N=5 may be better calibrated — that is acceptable. The claim is about MC-Dropout, not Deep Ensembles. Be honest: if Deep Ensembles N=5 beats CCR routing entropy, say so explicitly.

### 4d. Clinical Boundary Analysis

High entropy should cluster at tumor boundaries and NCR-ET transition zones. Compute:
- Mean routing entropy inside each subregion (should be low)
- Mean routing entropy at subregion boundaries (should be high)
- Spatial overlap of high-entropy voxels with GT boundary mask

This is not a primary metric — it's a qualitative validation that entropy is clinically meaningful.

---

## 6. Experiment 5 — LLM Hallucination Rate (REP)

**Question**: Does routing evidence (REP) reduce clinical language model hallucination when generating segmentation reports?

**This is Section 8 of the research plan. Uses standalone Gemini 1.5 Pro, not the unpublished NeuroReAct system.**

### 5a. Study Design

**Cases**: 50 randomly sampled BraTS 2021 validation cases (different from the 125 test cases used for Experiments 1–3). Document case IDs for reproducibility.

**Two conditions** (within-subjects, same 50 cases):

```
Condition A (Baseline):
  LLM input = MRI slice images + segmentation mask overlay
  No routing information provided
  LLM generates clinical summary (3–5 sentences)

Condition B (CCR + REP):
  LLM input = MRI slice images + segmentation mask overlay + REP structured evidence
  REP includes: volume per concept, confident voxels, uncertain voxels, boundary zone count
  LLM generates clinical summary (3–5 sentences)
```

**REP format** (constructed from CCR routing output):
```
[CCR Routing Evidence — direct model output, not post-hoc]
Enhancing Tumor:   {confident_voxels} confident voxels (prob > 0.7, entropy < 0.5)
                   {uncertain_voxels} uncertain voxels (entropy > 1.2 bits)
                   Mean routing probability: {mean_prob:.2f}
Necrotic Core:     {confident_voxels} confident voxels
                   Mean routing probability: {mean_prob:.2f}
Peritumoral Edema: {confident_voxels} confident voxels
                   Mean routing probability: {mean_prob:.2f}
Boundary zone:     {boundary_voxels} total voxels with high uncertainty (flagged for clinical review)
Overall confidence: {confidence:.2f} (range 0–1)
```

**System prompt for Condition B**:
```
You are a clinical AI assistant. The following evidence comes from the model's
internal routing decisions — these are direct computational outputs, not post-hoc
approximations. Each number reflects the model's actual routing of voxels to
clinical concept experts. Do not add clinical observations not supported by this
evidence. Flag uncertain regions explicitly.
```

### 5b. Rating Protocol

**Raters**: 2 board-certified radiologists with neuro-oncology experience. Blind to condition.

**Rating unit**: Individual sentences (not full reports).

**Categories**:
- **Grounded**: claim is directly supported by routing evidence OR visible MRI features
- **Partially grounded**: claim is plausible but goes beyond routing evidence
- **Hallucination**: claim contradicts routing evidence OR makes claims routing cannot support

**Inter-rater agreement**: Cohen's κ. Minimum κ = 0.60 for the study to be credible.

**Hallucination rate**: fraction of sentences rated as hallucination, per condition.

### 5c. Additional Ratings

- **Clinical utility** (1–5 Likert scale per report): "How clinically useful is this summary?"
- **Accuracy** (1–5 Likert scale): "How accurately does this reflect the imaging findings?"
- **Conciseness** (1–5 Likert scale): "Is this appropriately concise?"

### 5d. Expected Results

| Condition | Hallucination rate | Clinical utility | Cohen's κ |
|-----------|------------------|-----------------|----------|
| A: Mask-only | ~0.35 | ~3.1 | ≥ 0.60 |
| B: Mask + REP | ~0.20 | ~4.0 | ≥ 0.60 |
| **Reduction** | **~40%** | **+0.9 points** | |

**Success criterion**: ≥ 30% reduction in hallucination rate, p < 0.05 (paired t-test over 50 cases).

### 5e. Ethical Note

The agent does not make clinical decisions. The study is a controlled evaluation of LLM report generation quality. Radiologist ratings are for research only. IRB approval required if radiologists are compensated; check institution requirements.

---

## 7. Experiment 6 — Generalization to LiTS

**Question**: Does the CCR principle generalize to a different organ, modality, and concept set?

**Why**: If CCR only works for BraTS it is a BraTS-specific trick. LiTS generalization proves it is a backbone-agnostic principle.

### 6a. LiTS Setup

**Dataset**: Liver Tumor Segmentation Challenge (131 CT volumes, MICCAI 2017)

**Split**: 70% train / 15% val / 15% test (deterministic, seed=42)

**K=3 experts**:
- 0: Background
- 1: Liver (parenchyma)
- 2: Liver Tumor

**Configuration changes**:
```yaml
ccr:
  router:
    num_concepts: 3
    embed_dim: 192   # unchanged
swin:
  in_channels: 1   # CT is single-channel
  embed_dim: 48    # unchanged
```

### 6b. Protocol

Train CCR-Net and CCR-Retrofit-nnUNet from scratch on LiTS. Use same 80-epoch curriculum. K=3 changes the diversity loss and alignment loss dimensions but not the architecture.

### 6c. Metrics

| Metric | Target |
|--------|--------|
| Dice (Liver) | ≥ 0.94 (LiTS SOTA is ~0.96) |
| Dice (Tumor) | ≥ 0.60 (LiTS tumor is notoriously hard) |
| CAS_fg(Liver) | ≥ 0.80 |
| CAS_fg(Tumor) | ≥ 0.80 |
| Del AUC (CCR-Net routing) | ≥ 0.80 |

**Expected CAS may be lower than BraTS**: BraTS has finer subregion structure (3 tumor subregions). LiTS has only 2 semantic classes (liver and tumor). Routing the 2 foreground concepts correctly may be somewhat easier — but the tumor is much harder to segment, so lower Dice is acceptable.

---

## 8. Ablation Studies

Run all ablations on CCR-Net only (fastest to train). For each ablation, change one variable and retrain on BraTS 2021. Evaluate CAS_fg and Dice on val set.

### Ablation Table

| # | Name | What changes | Metric(s) | Expected observation |
|---|------|-------------|-----------|---------------------|
| A1 | No L_align | λ₁=0 throughout | CAS_fg | Drops to ~0.10–0.25. Proves L_align is necessary. |
| A2 | No L_diversity | λ₂=0 throughout | Expert utilization % | Expert collapse appears. ≥1 expert < 5%. |
| A3 | No L_boundary | λ₄=0 throughout | Dice(ET) | ET Dice drops ~2–4%. ET boundaries matter most. |
| A4 | No warmup | L_align starts epoch 1 | CAS_fg, utilization | Expert collapse during first 10 epochs. CAS ~0.20. |
| A5 | K=2 (WT/BG) | 2 experts only | CAS_fg | CAS may be higher (easier), but loses clinical granularity. Argue K=4 is right. |
| A6 | K=8 (over-specified) | 8 experts, same K=4 GT | CAS_fg | CAS drops (GT only has 4 concepts; extra experts find nothing). |
| A7 | Soft routing always | Hard routing disabled at eval | Del AUC | Del AUC drops ~0.05–0.10. Faithfulness requires hard routing. |
| A8 | No boundary head | Remove BoundaryRefinementHead | Dice(ET) | ET Dice drops ~1–2%. Head contributes marginally at boundaries. |
| A9 | Replace vs. additive (Retrofit) | CCR-Retrofit conditioning mode | CAS_fg + Dice | Additive: better Dice. Replace: better CAS_fg. Tradeoff quantified. |
| A10 | Frozen vs. joint (Retrofit) | Training mode A vs. B vs. C | CAS_fg + Dice | Mode C best overall. |
| A11 | Fixed temperature | temperature_learnable=False | CAS_fg | Lower CAS (~0.05 drop). Annealing helps. |
| A12 | No focal in L_align | focal_gamma=0 | CAS_fg | Lower CAS (~0.03 drop). Focal helps at hard boundaries. |

### How to present ablations in the paper

Table format:
```
Row = model variant
Cols = CAS_fg(NCR), CAS_fg(Edema), CAS_fg(ET), Dice(WT), Dice(TC), Dice(ET)
Bold = full model (best)
```

Ablations A1, A2, A4 are the critical ones — they verify the three mechanisms (L_align, L_diversity, warmup) are each individually necessary. Include all three in the main paper.

A3, A5–A12 go in the supplementary section.

---

## 9. Paper Tables and Figures

A complete list of all tables and figures needed for MICCAI (8-page) and NeurIPS (full) versions.

### Main Paper Tables

**Table 1 — CAS Results** (Experiment 1):
```
Columns: Model | CAS_fg(NCR) | CAS_fg(ED) | CAS_fg(ET) | AUROC(NCR) | AUROC(ED) | AUROC(ET)
Rows: CCR-Net | Retrofit-nnUNet | Retrofit-Swin | Retrofit-TransBTS | No-L_align (ablation)
```

**Table 2 — Faithfulness Comparison** (Experiment 2):
```
Columns: Method | Type | Deletion AUC ↑ | Insertion AUC ↑ | Infidelity ↓
Rows: GradCAM / IG / SHAP / AttRollout / TokenTM (post-hoc)
      CCR-Net / CCR-Retrofit×3 (intrinsic)
```

**Table 3 — Segmentation Accuracy** (Experiment 3):
```
Columns: Model | Dice WT | Dice TC | Dice ET | HD95 WT | HD95 TC | HD95 ET
Rows: nnU-Net (baseline) | CCR-Retrofit-nnUNet
      SwinUNETR (baseline) | CCR-Retrofit-SwinUNETR
      TransBTS (baseline) | CCR-Retrofit-TransBTS
      CCR-Net
```

**Table 4 — Uncertainty Calibration** (Experiment 4):
```
Columns: Method | ECE(overall) | ECE(ET) | AURC | Inference overhead
Rows: Routing entropy / MC-Dropout T={5,10,20} / Deep Ensembles N={3,5}
```

### Main Paper Figures

**Figure 1 — CCR Principle Diagram**:
- Two-panel: Post-hoc pipeline (left) vs. CCR pipeline (right)
- Arrow from routing to both "explanation" and "prediction" simultaneously (right panel)
- Key text: "one forward pass, three outputs"

**Figure 2 — Routing Visualization**:
- 4-panel for one BraTS case:
  - Panel 1: T1c MRI slice
  - Panel 2: GT segmentation (colored)
  - Panel 3: Routing probability maps (per-concept heatmaps at 16³ upsampled)
  - Panel 4: Routing entropy map (uncertainty)
- Show that routing maps visually match GT subregions
- High entropy at NCR-ET boundary (clinically meaningful)

**Figure 3 — Faithfulness Curves**:
- X axis: fraction of voxels masked/inserted
- Y axis: confidence drop (deletion) or rise (insertion)
- One curve per method
- CCR-Net and CCR-Retrofit curves clearly above all post-hoc baselines

**Figure 4 — Reliability Diagrams** (Experiment 4):
- 2×3 grid: 2 rows (deletion AUC / uncertainty), 3 columns (per subregion)
- Or: single reliability diagram comparing all methods
- Routing entropy curve closest to diagonal

**Figure 5 — Hallucination Rate (Section 8)**:
- Bar chart: condition A vs. B, hallucination rate + CI
- Table: clinical utility / accuracy / conciseness Likert scores per condition

**Figure 6 — Ablation Sensitivity**:
- Grouped bar chart: CAS_fg × 3 concepts for each ablation variant
- Shows which ablations cause CAS collapse (A1, A2, A4 most severe)

### Supplementary Tables

**Supp Table 1 — CAS_all** (all tokens including background, for completeness)
**Supp Table 2 — Per-Mode CCR-Retrofit** (frozen vs. joint vs. pretrain+finetune for nnU-Net)
**Supp Table 3 — LiTS Results** (Experiment 6)
**Supp Table 4 — DD per model** (decoder divergence values, used for Prop 1b check)
**Supp Table 5 — Lesion-Wise Dice** (BraTS per-lesion evaluation)
**Supp Table 6 — Full ablation table** (all 12 ablations)
**Supp Table 7 — Sensitivity-n values** (faithfulness at n=10, 50, 100, 500 features removed)

---

## 10. Success Criteria for A* Submission

### Hard gates (paper does not submit without these)

All must be satisfied before submission:

| Criterion | Value |
|-----------|-------|
| CAS_fg(NCR) | ≥ 0.85 on BraTS 2021 val |
| CAS_fg(Edema) | ≥ 0.85 on BraTS 2021 val |
| CAS_fg(ET) | ≥ 0.85 on BraTS 2021 val |
| Deletion AUC (CCR-Net) | ≥ 0.84 |
| Deletion AUC (best Retrofit) | ≥ 0.82 |
| Post-hoc baseline Del AUC | ≤ 0.65 (the gap must be real) |
| CCR-Retrofit Dice degradation | ≤ 1.0% vs. unmodified backbone |
| Prop 1a check: \|Del_AUC - CAS_fg\| | ≤ 0.04 |
| Prop 1b check: Del_AUC ≥ CAS × (1−DD) | Holds for all 3 Retrofit backbones |
| ECE(routing entropy) | < ECE(MC-Dropout T=10) |
| LLM hallucination reduction | ≥ 30% (Experiment 5) |
| Expert utilization | All 4 experts ≥ 10% after epoch 40 |
| Cross-year CAS | CAS_fg ≥ 0.80 on BraTS 2023 (OOD) |
| LiTS CAS | CAS_fg ≥ 0.80 for Liver and Tumor |

### Soft targets (strengthen the paper, not strictly required)

| Target | Value |
|--------|-------|
| CAS_fg | ≥ 0.88 (exceeds stated target) |
| Deletion AUC (CCR-Net) | ≥ 0.87 |
| LLM hallucination reduction | ≥ 40% |
| Cohen's κ (radiologist agreement) | ≥ 0.70 |
| All ablations completed | A1–A12 |
| BraTS 2023 OOD CAS | ≥ 0.83 |

### How to handle failure cases honestly

**If Retrofit faithfulness drops below the bound (Prop 1b violated)**:
- Report it. The paper's credibility depends on honesty.
- Investigate: high DD indicates the decoder is overriding routing → reduce decoder depth or increase λ_align.
- If unfixable in one backbone, report it as a limitation with quantitative DD.

**If ECE(routing entropy) ≥ ECE(MC-Dropout T=10) (Prop 2 fails)**:
- Weaken the claim: "routing entropy provides competitive uncertainty quantification at zero cost."
- Do NOT hide the MC-Dropout comparison. Report both.
- Investigate: check whether the temperature τ is well-calibrated vs. actual error rates.

**If LiTS CAS < 0.80**:
- Report what was achieved. Adjust the generalization claim accordingly.
- Partial generalization ("holds for brain tumor segmentation, partial evidence for liver") is publishable — do not overstate.

**If radiologist Cohen's κ < 0.60**:
- Do not report the hallucination study as a main result.
- Move to supplementary as a preliminary observation.
- Report κ honestly. Low κ means radiologists disagreed on what counts as hallucination — a valid finding in itself.

### Reviewer pre-emption checklist

Before submission, verify you can answer these:

- [ ] "Why is deletion AUC the right faithfulness metric?" → cite March 2025 benchmark; it is the established methodology.
- [ ] "Is the CAS claim circular?" → CAS measured on val, faithfulness on test, different data.
- [ ] "Can you show routing maps qualitatively?" → Figure 2, 3 cases minimum.
- [ ] "What happens if an expert collapses?" → ablation A2 shows the result; L_diversity prevents it.
- [ ] "Why K=4?" → matches BraTS annotation protocol; ablation A5/A6 shows K=4 is the right granularity.
- [ ] "Is CCR-Net clinically useful if Dice is 2% below nnU-Net?" → Yes. We are not claiming SOTA accuracy. We are claiming faithful interpretability. These are different optimization targets.
- [ ] "Is REP just prompt engineering?" → No. The content of REP is derived directly from the model's computational routing decisions. Prompt engineering changes the template; REP changes the grounding evidence. Faithfulness of the grounding is the claim.
- [ ] "Is the CCR bottleneck just a probe?" → No. A probe does not condition downstream computation. CCR expert outputs replace the decoder's input. Measured by DD < 0.3: the decoder IS using the routing signal.

---

*All experiments use the `ai_research` venv: `C:\Users\pathi\envs\ai_research`.*
*Primary dataset: BraTS 2021 at `D:\BraTS2024\training_data1_v2` (BraTS 2024 format, use brats_version="2024").*
*Phase 4 start condition: trained CCR-Net checkpoint (Phase 2 complete ✓) AND at least one CCR-Retrofit checkpoint (Phase 3 started).*
