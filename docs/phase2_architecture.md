# CCR-Net Phase 2 — Complete Technical Reference

**Status**: Complete as of 2026-05-31. 169/169 tests pass.
**Entry point**: `python pipeline/train.py --data_root /path/to/brats`
**Venv**: `C:\Users\pathi\envs\ai_research` (torch 2.10+cu126, monai 1.5.2, einops)

---

## Table of Contents

1. [Overview and Design Goals](#1-overview-and-design-goals)
2. [Complete Data Flow](#2-complete-data-flow)
3. [Encoder: MONAI SwinTransformer](#3-encoder-monai-swintransformer)
4. [CCR Bottleneck Insertion](#4-ccr-bottleneck-insertion)
5. [Decoder: MONAI UNetr Blocks](#5-decoder-monai-unetr-blocks)
6. [Boundary Refinement Head](#6-boundary-refinement-head)
7. [Full CCRNet Model](#7-full-ccrnet-model)
8. [Data Loading and Preprocessing](#8-data-loading-and-preprocessing)
9. [Training Strategy](#9-training-strategy)
10. [Loss Function and Curriculum](#10-loss-function-and-curriculum)
11. [Expert Collapse Monitoring](#11-expert-collapse-monitoring)
12. [Checkpointing](#12-checkpointing)
13. [Evaluation Pipeline](#13-evaluation-pipeline)
14. [Configuration Reference](#14-configuration-reference)
15. [Key Invariants and Hard Constraints](#15-key-invariants-and-hard-constraints)
16. [Implementation Gotchas](#16-implementation-gotchas)

---

## 1. Overview and Design Goals

CCR-Net Phase 2 wires the Phase 1 CCRBottleneckModule into a complete end-to-end trainable model for 3D brain tumor segmentation on BraTS.

**Three design goals:**

1. **No new moving parts at the encoder/decoder.** Use the exact same MONAI implementations from SwinUNETR — the encoder is MONAI's `SwinTransformer`, the decoder uses MONAI's `UnetrUpBlock` / `UnetOutBlock`. The CCR bottleneck is inserted between them, not designed around them.

2. **Zero projection overhead.** MONAI SwinTransformer with `embed_dim=48` naturally produces 192-dim features at stage-2 for 128³ input. Phase 1's CCRBottleneckModule expects `[B, N, 192]`. The shapes match exactly — no linear projection layer is needed.

3. **Phase 1 is untouched.** `src/ccr/` is not modified. Phase 2 only wraps and connects Phase 1 outputs.

**Why Swin-B at stage-2?**

The SwinTransformer with `embed_dim=48` and 4 stages produces features at:

```
stage-0: [B,  48, 64, 64, 64]   — too coarse for CCR (spatial only halved)
stage-1: [B,  96, 32, 32, 32]   — still mostly low-level features
stage-2: [B, 192, 16, 16, 16]   ← CCR insertion point
stage-3: [B, 384,  8,  8,  8]   — too compressed, spatial resolution lost
stage-4: [B, 768,  4,  4,  4]   — bottleneck, only 64 tokens
```

Stage-2 gives 4096 tokens (16³) at 192-dim — the same dimensions CCRBottleneckModule was designed for in Phase 1. The semantic content is high enough for clinical concept separation, and the spatial resolution (16³) is fine enough to represent tumor subregion boundaries.

---

## 2. Complete Data Flow

```
BraTS NIfTI files on disk
       │
       ▼
BraTSDataset.__getitem__()
  MONAI transforms: load → normalize → crop → pad → random crop → augment
  image: [4, 128, 128, 128]  float32   (T1c, T1n, T2w, T2f stacked)
  label: [128, 128, 128]     int64     (0=BG, 1=NCR, 2=Edema, 3=ET)
       │
       ▼
CCRNet.forward(image)
  │
  ├─ SwinEncoder3D
  │    MONAI SwinTransformer, embed_dim=48, depths=(2,2,2,2)
  │    hs[0]:  [B,  48, 64, 64, 64]   stage-0 skip
  │    hs[1]:  [B,  96, 32, 32, 32]   stage-1 skip
  │    hs[2]:  [B, 192, 16, 16, 16]   ← CCR STAGE (stage-2)
  │    hs[3]:  [B, 384,  8,  8,  8]   stage-3 skip
  │    hs[4]:  [B, 768,  4,  4,  4]   bottleneck skip
  │
  ├─ Reshape hs[2] for CCR
  │    [B, 192, 16, 16, 16]
  │      flatten spatial: [B, 192, 4096]
  │      transpose:       [B, 4096, 192]   ← tokens
  │
  ├─ CCRBottleneckModule  (Phase 1 — UNCHANGED)
  │    input:  [B, 4096, 192]
  │    output dict:
  │      expert_outputs: [B, 4096, 192]   per-concept transformed tokens
  │      routing_probs:  [B, 4096, 4]     soft routing (sums to 1 per token)
  │      entropy:        [B, 4096]        routing uncertainty per token
  │      assignments:    [B, 4096]        argmax index (hard, eval only)
  │
  ├─ Reshape expert_outputs back to spatial
  │    [B, 4096, 192] → transpose → [B, 192, 4096] → reshape → [B, 192, 16, 16, 16]
  │    = ccr_spatial (replaces hs[2] in decoder)
  │
  ├─ SwinUNETRDecoder
  │    Skip connections from all encoder stages + CCR spatial
  │    enc0 = UnetrBasicBlock(image)      → [B,  48, 128, 128, 128]
  │    enc1 = UnetrBasicBlock(hs[0])      → [B,  48,  64,  64,  64]
  │    enc2 = UnetrBasicBlock(hs[1])      → [B,  96,  32,  32,  32]
  │    enc3 = UnetrBasicBlock(ccr_spatial)→ [B, 192,  16,  16,  16]  ← POST-CCR
  │    enc4 = UnetrUpBlock(hs[4], hs[3]) → [B, 384,   8,   8,   8]
  │    dec3 = UnetrUpBlock(enc4, enc3)   → [B, 192,  16,  16,  16]
  │    dec2 = UnetrUpBlock(dec3, enc2)   → [B,  96,  32,  32,  32]
  │    dec1 = UnetrUpBlock(dec2, enc1)   → [B,  48,  64,  64,  64]
  │    dec0 = UnetrUpBlock(dec1, enc0)   → [B,  48, 128, 128, 128]
  │    out  = UnetOutBlock(dec0)         → [B,   4, 128, 128, 128]
  │
  └─ BoundaryRefinementHead (residual)
       [B, 4, 128, 128, 128] → [B, 4, 128, 128, 128]

       │
       ▼
output dict:
  seg_logits:    [B, 4, 128, 128, 128]   → CCRTotalLoss.pred_logits
  routing_probs: [B, 4096, 4]             → CCRTotalLoss.routing_probs
  entropy:       [B, 4096]               → uncertainty maps, CAS validation
  assignments:   [B, 4096]               → ExpertUtilizationTracker
       │
       ▼
downsample_labels_to_tokens(label, grid_shape=(16,16,16))
  nearest-neighbor interpolation: [B, 128, 128, 128] → [B, 4096]
  = token_labels                                      → CCRTotalLoss.token_labels
       │
       ▼
CCRTotalLoss.forward(
    pred_logits   = seg_logits,        # [B, 4, 128, 128, 128]
    labels        = label,             # [B, 128, 128, 128]
    routing_probs = routing_probs,     # [B, 4096, 4]
    token_labels  = token_labels,      # [B, 4096]
    tau_current   = model.ccr.router.temperature
)
  → losses['total'].backward()
```

---

## 3. Encoder: MONAI SwinTransformer

**Source**: `monai.networks.nets.swin_unetr.SwinTransformer`
**Wrapper**: `src/ccrnet/models/encoder.py` → `SwinEncoder3D`

### 3.1 Architecture

The SwinTransformer (Swin-B 3D) is the same encoder used inside MONAI's SwinUNETR, pretrained on Task01_BrainTumour from the Medical Segmentation Decathlon.

```
Input: [B, 4, 128, 128, 128]   (4 MRI modalities)
          │
          ▼
PatchEmbedding  (patch_size=2×2×2)
  4 channels → embed_dim=48 features
  Spatial: 128³ → 64³
          │
          ▼
Stage 0  (2 Swin-Transformer blocks, window=7³)
  Output: hs[0] = [B, 48,  64, 64, 64]
          │
   PatchMerging (2× downsample, 48→96)
          ▼
Stage 1  (2 Swin-Transformer blocks)
  Output: hs[1] = [B, 96,  32, 32, 32]
          │
   PatchMerging (96→192)
          ▼
Stage 2  (2 Swin-Transformer blocks)
  Output: hs[2] = [B, 192, 16, 16, 16]  ← CCR INSERTION POINT
          │
   PatchMerging (192→384)
          ▼
Stage 3  (2 Swin-Transformer blocks)
  Output: hs[3] = [B, 384,  8,  8,  8]
          │
   PatchMerging (384→768) [Note: encoder returns 5 outputs]
          ▼
Stage 4  (bottleneck, hs[4])
  Output: hs[4] = [B, 768,  4,  4,  4]
```

Each PatchMerging doubles channels and halves spatial resolution. After 4 stages, the spatial size is 128/2/2/2/2 = 8 ... wait, the patch embed already halves once (128→64), then 3 merges: 64→32→16→8→4. The hs indices go from 0 to 4, giving 5 total outputs.

### 3.2 Swin Attention Mechanism

Each Swin block partitions the 3D feature map into non-overlapping windows of size 7³ and computes self-attention within each window. Alternating layers use a cyclic shift to create cross-window connections.

```
For stage-2 at spatial 16³:
  Window size: 7³
  Number of windows: ceil(16/7)³ = 3³ = 27 windows
  Tokens per window: 7³ = 343
  Self-attention within each window: O(343²) per window, O(27×343²) total
```

This is far more efficient than global attention (which would be O(4096²) = 16M comparisons per head) while preserving receptive field through the shift operation.

### 3.3 Configuration

```python
SwinTransformer(
    in_chans        = 4,            # 4 MRI modalities
    embed_dim       = 48,           # feature_size in SwinUNETR terminology
    window_size     = (7, 7, 7),    # attention window
    patch_size      = (2, 2, 2),    # initial patch embed stride
    depths          = (2, 2, 2, 2), # Swin blocks per stage
    num_heads       = (3, 6, 12, 24),
    mlp_ratio       = 4.0,
    drop_path_rate  = 0.0,
    spatial_dims    = 3,
)
```

**MONAI 1.4+ change**: `img_size` was removed from `__init__`. It is no longer needed since the transformer computes attention relative to the input size dynamically. Pass only the architectural parameters above.

**Required package**: `einops` — MONAI's SwinTransformer uses `rearrange` from einops internally.

### 3.4 Pretrained Weights

MONAI provides SwinUNETR checkpoints pretrained on Task01_BrainTumour (5-fold CV, 484 cases). The checkpoint uses `swinViT.*` as key prefix for encoder weights.

```python
def _load_pretrained(self, path: str) -> None:
    ckpt  = torch.load(path, map_location="cpu")
    state = ckpt.get("state_dict", ckpt)
    # Strip 'swinViT.' prefix, skip decoder keys
    encoder_state = {
        k.replace("swinViT.", ""): v
        for k, v in state.items()
        if k.startswith("swinViT.")
    }
    self.swin.load_state_dict(encoder_state, strict=False)
```

`strict=False` is required — the checkpoint has no decoder keys, so missing keys are expected for the decoder-only buffers. Unexpected keys (decoder weights from the full SwinUNETR) are silently ignored.

---

## 4. CCR Bottleneck Insertion

**Source**: `src/ccrnet/models/ccrnet.py` → `CCRNet.forward()`

### 4.1 The Core Invariant

```
4 × swin.embed_dim == ccr.router.embed_dim
4 × 48             == 192
```

MONAI SwinTransformer with `embed_dim=48` applies 2 PatchMerge operations before stage-2, each doubling channels: 48 → 96 → 192. The CCRBottleneckModule from Phase 1 was designed for `embed_dim=192`. They match exactly.

This is enforced at config construction time in `Phase2Config.__post_init__()`.

### 4.2 Reshape Operations

```python
# Step 1: extract stage-2 features
stage2 = hs[2]                              # [B, 192, 16, 16, 16]
B, C, h, w, d = stage2.shape               # C=192, h=w=d=16

# Step 2: reshape to token sequence for CCRBottleneckModule
tokens = stage2.flatten(2).transpose(1, 2) # [B, 4096, 192]
#   flatten(2): [B, 192, 16*16*16] = [B, 192, 4096]
#   transpose:  [B, 4096, 192]

# Step 3: CCR bottleneck (Phase 1 — untouched)
ccr_out = self.ccr(tokens)

# Step 4: reshape expert_outputs back to spatial
ccr_spatial = (
    ccr_out["expert_outputs"]              # [B, 4096, 192]
    .transpose(1, 2)                       # [B, 192, 4096]
    .reshape(B, C, h, w, d)               # [B, 192, 16, 16, 16]
)
```

The token ordering is Z-major (C-contiguous), matching how MONAI's SwinTransformer internally flattens spatial dimensions. The reshape at step 4 exactly reverses step 2 — no spatial permutation or reordering occurs.

### 4.3 What CCRBottleneckModule Returns

From Phase 1 (`src/ccr/modules/dispatcher.py`):

```
input:  tokens [B, 4096, 192]

routing (ClinicalConceptRouter):
  → routing_probs [B, 4096, 4]  (softmax over K=4 concepts, sums to 1 per token)
  → entropy       [B, 4096]     (H(p) = -Σ p log p, range [0, log(4)])
  → assignments   [B, 4096]     (argmax in eval mode, soft in train mode)

experts (4 × ClinicalConceptExpert):
  each expert processes all tokens, weighted by routing_probs
  → expert_outputs [B, 4096, 192]  = Σ_k p(t→k) × Expert_k(t)  [train]
                                    = Expert_{argmax}(t)          [eval]
```

During training (`model.train()`): soft dispatch — each token is a weighted sum of all 4 expert outputs. This allows gradients to flow to all experts.

During inference (`model.eval()`): hard dispatch — each token is processed exclusively by its highest-probability expert. This makes `assignments` a deterministic routing map: the per-voxel explanation.

### 4.4 Why Replace hs[2] in the Decoder?

The decoder uses `ccr_spatial` (post-CCR features) instead of the raw `hs[2]`. This is the key design choice that makes CCR-Net's faithfulness claim:

- The decoder sees concept-specific features at the 16³ scale
- Concept boundaries learned by L_align propagate into the decoder's skip connection
- The final segmentation is conditioned on the routing — not just correlated with it

If we had kept raw `hs[2]` in the decoder and only used `ccr_out` for the loss, the routing would be a post-hoc explanation. By feeding `ccr_spatial` into the decoder, routing IS the computation.

---

## 5. Decoder: MONAI UNetr Blocks

**Source**: `src/ccrnet/models/decoder.py` → `SwinUNETRDecoder`
**MONAI sources**: `monai.networks.blocks.unetr_block`, `monai.networks.blocks.dynunet_block`

### 5.1 Block Types

**UnetrBasicBlock**: A single residual conv block. Used to refine skip connections before concatenation.
```
Input [B, C_in, H, W, D]
  → Conv3d(C_in → C_out, 3×3×3, padding=1)
  → InstanceNorm3d → ReLU
  → Conv3d(C_out → C_out, 3×3×3, padding=1)
  → + residual (if C_in == C_out, otherwise projected)
Output [B, C_out, H, W, D]
```

**UnetrUpBlock**: Upsamples + concatenates skip + refines.
```
Input: inp [B, C_in, H, W, D], skip [B, C_out, 2H, 2W, 2D]

  TransposedConv3d(C_in → C_out, stride=2)
    inp → [B, C_out, 2H, 2W, 2D]
  cat([upsampled, skip], dim=1)
    → [B, 2*C_out, 2H, 2W, 2D]
  UnetResBlock(2*C_out → C_out)
    → [B, C_out, 2H, 2W, 2D]

Output [B, C_out, 2H, 2W, 2D]
```

**Critical channel constraint**: The skip connection must have exactly `C_out` channels — not `C_in` channels. This determines all channel counts in the decoder design.

**UnetOutBlock**: Final 1×1×1 conv mapping feature channels to class logits.
```
Conv3d(C_in → K, kernel_size=1)   K = num_classes
```

### 5.2 Decoder Architecture (f = embed_dim = 48)

```
                     ENCODER SIDE                     DECODER SIDE
                     ────────────                     ────────────
image [B, 4, 128³]
  │
  └─ enc0 = UnetrBasicBlock(4→f)    [B, 48, 128³] ──────────────────────┐
                                                                          │
hs[0] [B, 48, 64³]                                                        │
  │                                                                       │
  └─ enc1 = UnetrBasicBlock(f→f)    [B, 48, 64³]  ─────────────────┐    │
                                                                     │    │
hs[1] [B, 96, 32³]                                                   │    │
  │                                                                  │    │
  └─ enc2 = UnetrBasicBlock(2f→2f)  [B, 96, 32³]  ────────────┐   │    │
                                                                │   │    │
ccr_spatial [B, 192, 16³]           (POST-CCR)                 │   │    │
  │                                                             │   │    │
  └─ enc3 = UnetrBasicBlock(4f→4f)  [B, 192, 16³] ──────┐    │   │    │
                                                          │    │   │    │
hs[4] [B, 768, 4³]                                        │    │   │    │
hs[3] [B, 384, 8³]                                        │    │   │    │
  │                                                       │    │   │    │
  └─ enc4 = UnetrUpBlock(16f→8f)   [B, 384, 8³]          │    │   │    │
                                          │               │    │   │    │
                                          ▼               │    │   │    │
                             dec3 = UnetrUpBlock(8f→4f)←──┘    │   │    │
                                  [B, 192, 16³]                 │   │    │
                                          │                     │   │    │
                                          ▼                     │   │    │
                             dec2 = UnetrUpBlock(4f→2f)←────────┘   │    │
                                  [B, 96, 32³]                       │    │
                                          │                          │    │
                                          ▼                          │    │
                             dec1 = UnetrUpBlock(2f→f)←─────────────┘    │
                                  [B, 48, 64³]                            │
                                          │                               │
                                          ▼                               │
                             dec0 = UnetrUpBlock(f→f)←────────────────────┘
                                  [B, 48, 128³]
                                          │
                                          ▼
                             out  = UnetOutBlock(f→K)
                                  [B, 4, 128³]
```

### 5.3 Channel Count Table

| Block | in_channels | out_channels | Input spatial | Output spatial |
|-------|-------------|--------------|---------------|----------------|
| enc0 (UnetrBasicBlock) | 4 | 48 | 128³ | 128³ |
| enc1 (UnetrBasicBlock) | 48 | 48 | 64³ | 64³ |
| enc2 (UnetrBasicBlock) | 96 | 96 | 32³ | 32³ |
| enc3 (UnetrBasicBlock) | 192 | 192 | 16³ | 16³ |
| enc4 (UnetrUpBlock) | 768 | 384 | 4³ → 8³ | 8³ |
| dec3 (UnetrUpBlock) | 384 | 192 | 8³ → 16³ | 16³ |
| dec2 (UnetrUpBlock) | 192 | 96 | 16³ → 32³ | 32³ |
| dec1 (UnetrUpBlock) | 96 | 48 | 32³ → 64³ | 64³ |
| dec0 (UnetrUpBlock) | 48 | **48** | 64³ → 128³ | 128³ |
| out (UnetOutBlock) | **48** | 4 | 128³ | 128³ |

**Note on dec0**: Uses `out_channels=f=48`, NOT `f//2=24`. This is because the skip (enc0) has 48 channels, and UnetrUpBlock requires `skip.channels == out_channels`. Using 24 would cause a runtime shape error on the concatenation. The original SwinUNETR uses the same 48→48 final step.

### 5.4 Skip Connection Alignment

The decoder is conditioned on CCR features through `enc3`:
- `enc3 = UnetrBasicBlock(ccr_spatial)` — processes POST-CCR features
- `dec3 = UnetrUpBlock(enc4, enc3)` — fuses bottleneck with CCR-refined 16³ features

Every spatial scale in the decoder is transitively affected by the CCR output, because `dec2` uses `dec3` as input, `dec1` uses `dec2`, and so on. The CCR routing decision propagates through all decoder stages.

---

## 6. Boundary Refinement Head

**Source**: `src/ccrnet/models/boundary_head.py` → `BoundaryRefinementHead`

```
Input: seg_logits [B, K, H, W, D]
          │
          ├─ residual branch (identity)
          │
          └─ refine branch:
               Conv3d(K → 16, 3×3×3, padding=1)  → InstanceNorm3d(16) → ReLU
               Conv3d(16 → K, 1×1×1)
          │
          ▼
       x + refine(x)   [B, K, H, W, D]
```

**Parameter count** (K=4, hidden=16): 
- First conv: 4×16×3×3×3 + 16 = 1,744 params
- Norm: 16×2 = 32 params  
- Second conv: 16×4×1×1×1 + 4 = 68 params
- **Total: ~1,844 params** (well under 100K target)

**Why residual?** The residual connection ensures the head cannot override the routing-derived segmentation. It can only sharpen edges, not re-assign concepts. This preserves the faithfulness property: if the routing says token t belongs to concept k, the boundary head cannot reassign it — it can only improve boundary crispness within the logit space.

**Why not a larger head?** We want the final segmentation to be the decoder output (which is conditioned on routing), not the boundary head. A large head would learn independent segmentation features, decoupling the output from routing and invalidating Proposition 1a.

---

## 7. Full CCRNet Model

**Source**: `src/ccrnet/models/ccrnet.py` → `CCRNet`

### 7.1 Model Construction

```python
class CCRNet(nn.Module):
    _CCR_STAGE = 2

    def __init__(self, config: Phase2Config):
        self.encoder       = SwinEncoder3D(config.swin)
        self.ccr           = CCRBottleneckModule(config.ccr)  # Phase 1
        self.decoder       = SwinUNETRDecoder(
            embed_dim    = config.swin.embed_dim,      # 48
            num_concepts = config.ccr.router.num_concepts,  # 4
        )
        self.boundary_head = BoundaryRefinementHead(
            num_concepts = config.ccr.router.num_concepts,
        )
        self._grid_shape = None   # set after first forward pass
```

### 7.2 Forward Pass

```python
def forward(self, x):
    # 1. Swin encoder → 5 hidden states
    hs = self.encoder(x)                       # list of 5

    # 2. Extract CCR stage, record grid shape
    stage2 = hs[self._CCR_STAGE]               # [B, 192, 16, 16, 16]
    B, C, h, w, d = stage2.shape
    self._grid_shape = (h, w, d)               # (16, 16, 16) for 128³ input

    # 3. Reshape to token sequence
    tokens = stage2.flatten(2).transpose(1, 2) # [B, 4096, 192]

    # 4. CCR bottleneck
    ccr_out = self.ccr(tokens)

    # 5. Reshape back to spatial
    ccr_spatial = (ccr_out["expert_outputs"]
                   .transpose(1, 2)
                   .reshape(B, C, h, w, d))    # [B, 192, 16, 16, 16]

    # 6. Decode
    seg_logits = self.decoder(x, hs, ccr_spatial)

    # 7. Boundary refinement
    seg_logits = self.boundary_head(seg_logits)

    return {
        "seg_logits":    seg_logits,           # [B, 4, 128, 128, 128]
        "routing_probs": ccr_out["routing_probs"],  # [B, 4096, 4]
        "entropy":       ccr_out["entropy"],    # [B, 4096]
        "assignments":   ccr_out["assignments"],# [B, 4096]
    }
```

### 7.3 Parameter Count Estimate (128³, embed_dim=48)

| Component | Parameters |
|-----------|-----------|
| SwinTransformer (Swin-B 3D) | ~28M |
| CCRBottleneckModule | ~8M (4 experts × ~2M each + router) |
| SwinUNETRDecoder | ~6M |
| BoundaryRefinementHead | ~2K |
| **Total** | **~42M** |

These are rough estimates. The SwinTransformer dominates. Exact counts depend on window size and depth settings.

---

## 8. Data Loading and Preprocessing

**Source**: `data/brats_dataset.py`

### 8.1 BraTS Naming Conventions

Two versions are supported, controlled by `DataConfig.brats_version`:

```
BraTS 2024 ("2024"):        BraTS 2021 ("2021"):
  {case}-t1c.nii.gz            {case}_t1ce.nii.gz
  {case}-t1n.nii.gz            {case}_t1.nii.gz
  {case}-t2w.nii.gz            {case}_t2.nii.gz
  {case}-t2f.nii.gz            {case}_flair.nii.gz
  {case}-seg.nii.gz            {case}_seg.nii.gz
```

The existing EDA script (`data/load_brats_sample.py`) uses BraTS 2024 naming.

### 8.2 Label Mapping

BraTS annotation protocol:
```
0 = Background
1 = NCR (Necrotic Core)
2 = ED  (Peritumoral Edema)
3 = ET  (Enhancing Tumor)   ← BraTS 2021 uses 3
4 = ET  (Enhancing Tumor)   ← BraTS 2024 GLI uses 4 (!)
```

BraTS 2024 uses label 4 for ET instead of 3. `RemapBraTS2024Labels` transform maps 4→3 during loading, keeping all splits consistent with `CCRConfig.concept_names = ("background", "necrotic_core", "edema", "enhancing_tumor")` where index = label value.

### 8.3 Patient Split

```python
def split_patients(patient_dirs, val_frac=0.125, test_frac=0.125, seed=42):
    # Deterministic shuffle with seeded RNG
    rng = random.Random(seed)
    dirs = list(patient_dirs)
    rng.shuffle(dirs)
    n = len(dirs)
    n_test = max(1, int(n * test_frac))   # ~12.5%
    n_val  = max(1, int(n * val_frac))    # ~12.5%
    return {
        "test":  dirs[:n_test],
        "val":   dirs[n_test:n_test+n_val],
        "train": dirs[n_test+n_val:],     # ~75%
    }
```

The split is patient-level (not scan-level). All scans from one patient go to the same split. This prevents data leakage.

### 8.4 MONAI Transform Pipeline

All transforms are MONAI dictionary transforms operating on `{"t1c": ..., "t1n": ..., "t2w": ..., "t2f": ..., "label": ...}`.

#### Base transforms (all splits):

```
LoadImaged(keys=["t1c","t1n","t2w","t2f","label"])
  Load 5 NIfTI files into numpy arrays

EnsureChannelFirstd(keys=all)
  [H,W,D] → [1,H,W,D] for each modality

RemapBraTS2024Labels(keys=["label"])    [2024 only]
  label[label==4] = 3

NormalizeIntensityd(keys=image_keys, nonzero=True, channel_wise=True)
  For each of the 4 modalities independently:
    mask = (voxel != 0)
    mean = voxels[mask].mean()
    std  = voxels[mask].std()
    voxels[mask] = (voxels[mask] - mean) / std
  Non-brain (zero) voxels remain at zero.
  Why nonzero=True: BraTS brain volumes have a large skull-stripped background at
  zero. Including those zeros would bias the mean and underestimate std.
  Why channel_wise=True: T1c, T1n, T2w, T2f have completely different intensity
  distributions. A single normalization across all 4 channels would distort relative
  contrasts.

CropForegroundd(keys=all, source_key="label", margin=10)
  Compute bounding box of label > 0, add 10-voxel margin, crop all volumes.
  Removes large empty skull-stripped background, reducing memory and compute.
  Output shape: variable (bounded by original volume size).

SpatialPadd(keys=all, spatial_size=[128,128,128])
  Zero-pad to at least 128³ if the cropped volume is smaller.
  Most BraTS volumes are ~240×240×155 before crop; after crop they are smaller,
  and after padding they are ≥128³.

RandCropByPosNegLabeld(
    keys=all, label_key="label",
    spatial_size=[128,128,128],
    pos=1, neg=1, num_samples=1,
    image_key="t1c", image_threshold=0
)
  Randomly crop a 128³ patch, with equal probability of:
    pos: center falls on a foreground (label>0) voxel
    neg: center falls on a background (label=0) voxel
  This ensures tumor regions are sampled approximately 50% of the time.
  Returns a list of 1 sample (num_samples=1). Code unwraps this list in __getitem__.
```

#### Training-only augmentation:

```
RandFlipd(prob=0.5, spatial_axis=0)   # sagittal flip
RandFlipd(prob=0.5, spatial_axis=1)   # coronal flip
RandFlipd(prob=0.5, spatial_axis=2)   # axial flip
  Brain tumors have no consistent L/R or A/P orientation bias.
  Flipping is anatomically valid for BraTS (unlike e.g. cardiac where L≠R).

RandRotate90d(prob=0.5, max_k=3)
  Random 90° rotation (0, 90, 180, or 270 degrees).

RandScaleIntensityd(keys=image_keys, factors=0.1, prob=0.5)
  Multiply intensity by (1 + U[-0.1, 0.1]).
  Simulates scanner gain variation.

RandShiftIntensityd(keys=image_keys, offsets=0.1, prob=0.5)
  Add U[-0.1, 0.1] to all intensities.
  Simulates scanner offset variation.

RandGaussianNoised(keys=image_keys, prob=0.1, std=0.01)
  Add Gaussian noise N(0, 0.01).
  Low probability to avoid degrading features too aggressively.
```

Val and test splits get NO augmentation (only the base transforms).

### 8.5 Dataset and DataLoader

```python
class BraTSDataset(Dataset):
    def __getitem__(self, idx):
        result = self._cache_dataset[idx]
        # RandCropByPosNegLabeld returns a list when num_samples >= 1
        item = result[0] if isinstance(result, list) else result
        # Concatenate 4 single-channel modalities into [4, H, W, D]
        image = torch.cat([item["t1c"], item["t1n"], item["t2w"], item["t2f"]], dim=0)
        label = item["label"].squeeze(0).long()
        return {"image": image, "label": label, "patient_id": item["patient_id"]}
```

**DataLoader settings:**

| Split | shuffle | drop_last | batch_size |
|-------|---------|-----------|-----------|
| train | True | True | 2 (config) |
| val | False | False | 1 |
| test | False | False | 1 |

`pin_memory=True` for CUDA-accelerated transfers. `drop_last=True` for train to avoid partial-batch issues with BatchNorm/InstanceNorm.

---

## 9. Training Strategy

**Source**: `pipeline/train.py`

### 9.1 Optimizer and Scheduler

```python
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr           = 1e-4,
    weight_decay = 1e-5,
)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max  = 80,    # total epochs
    eta_min = 1e-6,
)
```

**Why AdamW over SGD?**
- AdamW decouples weight decay from the gradient update, preventing weight decay from interacting with the adaptive gradient scaling
- More robust to learning rate choice on transformer architectures
- Standard for SwinUNETR and similar MONAI models

**Why CosineAnnealingLR?**
- Smooth decay avoids abrupt LR changes that could disrupt the three-phase curriculum
- `eta_min=1e-6` ensures the model keeps learning slowly during the refinement phase
- LR curve from epoch 1 to 80:

```
Epoch:  1     10    20    40    60    80
LR:   1e-4  ~9e-5 ~7e-5 ~5e-5 ~2e-5 1e-6
```

### 9.2 Mixed Precision Training

```python
scaler = GradScaler(enabled=amp and device.type == "cuda")

with autocast(enabled=amp and device.type == "cuda"):
    out    = model(image)
    losses = loss_fn(...)

scaler.scale(losses["total"]).backward()
scaler.unscale_(optimizer)
clip_grad_norm_(model.parameters(), grad_clip=1.0)
scaler.step(optimizer)
scaler.update()
```

AMP (Automatic Mixed Precision) uses float16 for forward pass and backward pass where safe, keeping float32 for loss scaling and optimizer states. On modern GPUs (A100, V100), this roughly doubles throughput and halves memory.

`GradScaler` prevents float16 gradient underflow by scaling up the loss before backward and scaling down before the optimizer step.

### 9.3 Gradient Clipping

```python
clip_grad_norm_(model.parameters(), max_norm=1.0)
```

Applied after `scaler.unscale_()` so the unscaled gradients are clipped. Prevents occasional gradient spikes during early training when the encoder is still adjusting to the CCR routing signal.

### 9.4 Three-Phase Curriculum

The curriculum matches Phase 1 exactly (unchanged from `CCRConfig.curriculum`):

```
Phase 1 — Warmup (epochs 1–10):
  λ_align = 0.0      ← L_align is DISABLED
  λ_div   = 0.5
  λ_ent   = 0.0
  λ_bound = 0.0
  τ_target = 2.0     (diffuse routing — experts explore freely)

  Purpose: Let the encoder learn useful visual features before routing pressure
  is applied. If L_align starts at epoch 1, routing gradients arrive before the
  encoder can provide discriminative features, and all experts collapse to the
  same attractor.

Phase 2 — Alignment (epochs 11–50):
  λ_align = 1.0      ← L_align is ACTIVE
  λ_div   = 0.5
  λ_ent   = 0.01
  λ_bound = 0.3
  τ_target = 1.0     (routing sharpens as CAS improves)

  Purpose: Force routing to align with clinical concept labels. The most
  important phase — CAS_fg should rise monotonically here.

Phase 3 — Refinement (epochs 51–80):
  λ_align = 0.5      ← reduced (routing mostly stable)
  λ_div   = 0.1
  λ_ent   = 0.01
  λ_bound = 1.0      ← L_boundary raised (boundary precision)
  τ_target = 0.5     (routing nearly deterministic — crisp expert maps)

  Purpose: Fine-tune boundary segmentation without disrupting routing.
  BoundaryAwareLoss boosts cross-entropy weight at tumor boundary voxels.
```

### 9.5 Token Label Downsampling

```python
def downsample_labels_to_tokens(labels, grid_shape):
    # labels: [B, H, W, D]  int64
    labels_f = labels.float().unsqueeze(1)              # [B, 1, H, W, D]
    down = F.interpolate(labels_f, size=grid_shape, mode="nearest")
    return down.long().squeeze(1).reshape(labels.shape[0], -1)  # [B, N]
```

**Why nearest-neighbor?** Bilinear/trilinear interpolation would create fractional label values (e.g., 1.5) at class boundaries, which are meaningless as class indices. Nearest-neighbor preserves integer class membership and naturally implements a majority-vote-like behavior (the center voxel of a block determines the token's label).

**Grid shape**: `model.get_grid_shape()` returns the stage-2 spatial dimensions (16, 16, 16) for 128³ input. Always call this after a forward pass, not before.

### 9.6 Epoch Loop Structure

```python
for epoch in range(start_epoch, 81):
    loss_fn.set_epoch(epoch)      # update curriculum weights + τ target
    model.train()

    for batch in train_loader:
        image = batch["image"].to(device)
        label = batch["label"].to(device)

        optimizer.zero_grad()
        with autocast(...):
            out          = model(image)
            token_labels = downsample_labels_to_tokens(label, model.get_grid_shape())
            losses       = loss_fn(
                pred_logits   = out["seg_logits"],
                labels        = label,
                routing_probs = out["routing_probs"],
                token_labels  = token_labels,
                tau_current   = model.ccr.router.temperature,
            )
        scaler.scale(losses["total"]).backward()
        scaler.unscale_(optimizer)
        clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        tracker.update(out["assignments"])

    # End of epoch
    collapsed = tracker.check_collapse(epoch)
    if collapsed:
        reinitialize_experts(model.ccr, collapsed)
    tracker.reset()
    scheduler.step()

    if epoch % checkpoint_every == 0:
        val_metrics = validate(model, val_loader, cas, device)
        save_checkpoint(model, optimizer, scheduler, epoch, val_metrics, config, ckpt_dir)
```

---

## 10. Loss Function and Curriculum

**Source**: Phase 1 → `src/ccr/losses/total.py` → `CCRTotalLoss`

The total loss is:

```
L_total = L_seg + λ₁·L_align + λ₂·L_diversity + λ₃·L_entropy_reg + λ₄·L_boundary
        + τ_reg_weight·(τ_actual − τ_target)²
```

### 10.1 Component Details

**L_seg** (always weight 1.0):
```
L_seg = 0.5 × DiceLoss + 0.5 × FocalLoss(γ=2.0, α=0.25)
```
Dice handles class imbalance (small ET region). Focal down-weights easy background voxels.

**L_align** (weight λ₁, zero during warmup):
```
For each foreground token t (label_t > 0) and concept k=1..K-1:
  L_align_k(t) = FocalBCE(routing_prob_k(t), M_k(t))
  M_k(t) = 1 if label_t == k, else 0
L_align = mean over (t, k) pairs
```
Focal-BCE with γ=2.0 downweights tokens where routing is already correct, focusing gradient on hard boundary cases. Foreground-only (labels > 0) because background tokens would dominate gradient signal — the important claim is about tumor subregion routing.

**L_diversity** (weight λ₂):
```
Encourage expert routing vectors to be dissimilar:
L_diversity = Σ_{j≠k} |cos_sim(r_j, r_k)|  / (K*(K-1))
where r_k = mean routing vector for concept k
```
Prevents routing collapse where all experts become identical.

**L_entropy_reg** (weight λ₃ ≤ 0.01):
```
L_entropy = -mean H(routing_probs)   (maximize entropy = spread routing)
```
Regularizes against overconfident routing during warmup/alignment. **Keep λ₃ ≤ 0.01** — larger values push toward uniform routing which destroys CAS and violates Proposition 2.

**L_boundary** (weight λ₄, raised to 1.0 in refinement):
```
boundary_mask = morphological_dilation(class_boundaries, iters=3)
weight_map[boundary_mask] += boost_factor   (boost_factor=5.0)
L_boundary = cross_entropy(pred_logits, labels, weight=weight_map)
```
Concentrates segmentation loss on the clinically important tumor boundaries. Implemented without scipy — uses 3D max-pooling as morphological dilation.

**L_tau_reg** (soft temperature annealing):
```
L_tau = τ_reg_weight × (τ_actual − τ_target)²
```
Penalizes deviation of the learnable temperature from its phase-appropriate target. This creates a soft pull: the temperature is not forced to the target, but any deviation costs.

### 10.2 Curriculum Weight Schedule

| Epoch | Phase | λ_align | λ_div | λ_ent | λ_bound | τ_target |
|-------|-------|---------|-------|-------|---------|---------|
| 1–10  | Warmup | 0.0 | 0.5 | 0.00 | 0.0 | 2.0 |
| 11–50 | Alignment | 1.0 | 0.5 | 0.01 | 0.3 | 1.0 |
| 51–80 | Refinement | 0.5 | 0.1 | 0.01 | 1.0 | 0.5 |

### 10.3 Monitoring During Training

Log at every checkpoint epoch:
- `L_align` — must decrease monotonically after epoch 11. If it rises after epoch 20, check for expert collapse.
- `CAS_fg(NCR)`, `CAS_fg(Edema)`, `CAS_fg(ET)` — target ≥ 0.85 by epoch 80
- Expert utilization % per concept — should stay above 5% after epoch 20
- Current τ actual vs τ target — should track within ±0.3

---

## 11. Expert Collapse Monitoring

**Source**: Phase 1 → `src/ccr/utils/metrics.py` → `ExpertUtilizationTracker`

```python
tracker = ExpertUtilizationTracker(
    num_concepts       = 4,
    concept_names      = ("background", "necrotic_core", "edema", "enhancing_tumor"),
    collapse_threshold = 5,   # percent
)

# After each batch:
tracker.update(out["assignments"])   # [B, N] integer assignments

# After each epoch:
collapsed = tracker.check_collapse(epoch)
# Returns list of concept_names with utilization < 5% after warmup
if collapsed:
    reinitialize_experts(model.ccr, collapsed)   # reset fc2 weights
tracker.reset()
```

**Collapse threshold**: 5% of tokens after epoch 20. An expert with <5% utilization has effectively learned nothing and is being ignored by the router.

**Reinitialization**: `reinitialize_experts()` resets the `fc2` weight of collapsed experts using `xavier_uniform(gain=0.01)`, the same initialization as the original `_init_weights()`. This resets the expert to a near-identity mapping, giving it a fresh start without discarding all learned representations.

**Why only fc2?** The `fc1` (input projection) likely contains useful feature extraction. Resetting only `fc2` (output projection) disrupts the dead attractor while preserving some learned structure.

---

## 12. Checkpointing

**Source**: `src/ccrnet/utils/checkpoint.py`

### 12.1 Save

```python
torch.save({
    "epoch":           epoch,
    "model_state":     model.state_dict(),
    "optimizer_state": optimizer.state_dict(),
    "scheduler_state": scheduler.state_dict(),
    "metrics":         {"cas_fg": ..., "dice": ...},
    "config":          config,          # Phase2Config object
}, f"{ckpt_dir}/epoch_{epoch:04d}.pth")
```

### 12.2 Load

```python
ckpt = torch.load(path, map_location="cpu", weights_only=False)
# weights_only=False required for PyTorch 2.6+ when checkpoint contains
# custom Python objects (Phase2Config dataclass). Safe for our own checkpoints.
model.load_state_dict(ckpt["model_state"])
```

**PyTorch 2.6+ note**: The default for `torch.load` changed from `weights_only=False` to `weights_only=True` in PyTorch 2.6. Since we store the `Phase2Config` object in the checkpoint (not just tensors), `weights_only=True` would fail with a security error. Using `weights_only=False` is safe here because we only load our own checkpoints.

Alternative: store `config.to_yaml()` string instead of the object, then reconstruct on load. Avoids the security warning entirely but requires extra handling.

### 12.3 Resume Training

```
python pipeline/train.py --data_root /path/to/brats --resume checkpoints/epoch_0050.pth
```

Restores model weights, optimizer state, and scheduler state. Training resumes from `saved_epoch + 1`.

---

## 13. Evaluation Pipeline

**Source**: `pipeline/evaluate.py`

```
python pipeline/evaluate.py --checkpoint checkpoints/epoch_0080.pth --split val
python pipeline/evaluate.py --checkpoint checkpoints/epoch_0080.pth --split test
```

### 13.1 Metrics

**Dice** (WT/TC/ET — standard BraTS reporting):
```
WT = Whole Tumor   = labels {1, 2, 3}
TC = Tumor Core    = labels {1, 3}    (NCR + ET)
ET = Enhancing Tumor = label {3}
```

**HD95** (95th-percentile Hausdorff distance, mm):
- Computed per-class using `scipy.ndimage.distance_transform_edt`
- Returns inf if either the prediction or ground truth mask is empty

**CAS_fg** (primary CCR metric, computed on routing_probs vs token_labels):
- `cas.update(out["routing_probs"], token_labels)` called per batch
- `cas.compute_fg()` after full dataset pass
- Target: CAS_fg ≥ 0.85 for NCR, Edema, ET

### 13.2 Routing Map Export

For up to 5 sample cases, saves routing probability maps as NIfTI:
```
evals/val/{patient_id}_routing_background.nii.gz
evals/val/{patient_id}_routing_necrotic_core.nii.gz
evals/val/{patient_id}_routing_edema.nii.gz
evals/val/{patient_id}_routing_enhancing_tumor.nii.gz
```

Each file is a 16³ volume (the CCR grid, not the full 128³). Values in [0,1], summing to 1 per voxel across the 4 files. These are the per-voxel explanation maps — they show how each spatial location was routed to clinical concepts.

---

## 14. Configuration Reference

**Source**: `configs/brats_phase2.yaml`, `src/ccrnet/config/phase2_config.py`

```yaml
swin:
  img_size: [128, 128, 128]     # target crop size (encoder sees this size)
  in_channels: 4                # MRI modalities
  embed_dim: 48                 # stage-2 = 4×48 = 192 = ccr.router.embed_dim
  window_size: [7, 7, 7]        # Swin attention window
  patch_size: [2, 2, 2]         # initial patch embed stride
  depths: [2, 2, 2, 2]          # Swin blocks per stage
  num_heads: [3, 6, 12, 24]     # attention heads per stage
  drop_path_rate: 0.0           # stochastic depth rate
  pretrained_path: ""           # set to SwinUNETR .pt from MONAI model zoo

data:
  data_root: ""                 # REQUIRED: one sub-dir per patient
  brats_version: "2024"         # "2024" or "2021"
  spatial_size: [128, 128, 128]
  val_fraction: 0.125           # 12.5% of patients
  test_fraction: 0.125          # 12.5% of patients
  num_workers: 4
  seed: 42                      # controls patient split (deterministic)

training:
  batch_size: 2
  learning_rate: 1.0e-4
  weight_decay: 1.0e-5
  grad_clip: 1.0
  amp: true                     # GradScaler + autocast on CUDA
  checkpoint_dir: "checkpoints/"
  checkpoint_every: 5
  log_every: 10

ccr:                            # Phase 1 config — do not change unless ablating
  router:
    embed_dim: 192              # must equal 4 × swin.embed_dim
    num_concepts: 4
    use_prototypes: true
    temperature_learnable: true
  curriculum:
    warmup_end_epoch: 10
    alignment_end_epoch: 50
    total_epochs: 80
    tau_warmup: 2.0
    tau_alignment: 1.0
    tau_refinement: 0.5
```

---

## 15. Key Invariants and Hard Constraints

These are non-negotiable constraints. Violating them either causes a runtime error, an incorrect result, or invalidates the paper's claims.

**Architectural invariant:**
```
ccr.router.embed_dim == 4 × swin.embed_dim
192 == 4 × 48
```
Enforced by `Phase2Config.__post_init__()`. Changing `swin.embed_dim` without changing `ccr.router.embed_dim` (or vice versa) raises `ValueError`.

**CCR stage:**
```
_CCR_STAGE = 2   (index into hidden_states_out)
```
Swin stage-2 is the only stage that produces 192-dim at 16³ spatial. Stages 0, 1, 3, 4 produce different dimensions and would require projection layers.

**Curriculum warmup:**
```
Do NOT activate L_align before epoch 11.
```
Starting alignment loss at epoch 1 causes expert collapse. The encoder has not learned discriminative features yet, so routing gradients send all experts to the same attractor.

**Entropy weight:**
```
λ_entropy ≤ 0.01
```
Larger values push routing toward uniform distribution. This destroys CAS (routing becomes uninformative) and violates Proposition 2.

**Token labels must be downsampled with nearest-neighbor:**
```
F.interpolate(..., mode="nearest")
```
Not bilinear. Fractional label values are meaningless for discrete class indices.

**Faithfulness claim requires hard routing at inference:**
```
model.eval()   → hard dispatch (argmax)
model.train()  → soft dispatch (weighted sum)
```
Deletion AUC and all faithfulness experiments must be run with `model.eval()`. The paper's claim is about inference-time behavior. Soft dispatch during training is a training trick, not the claimed mechanism.

**dec0 uses out_channels=f (not f//2):**
The last decoder upsampling block must match the enc0 skip (48 channels). Using `f//2=24` causes a channel mismatch in the concatenation at runtime.

---

## 16. Implementation Gotchas

These are bugs or surprising behaviors encountered during implementation.

### MONAI 1.4+ removed img_size from SwinTransformer.__init__

Old (MONAI ≤ 1.3):
```python
SwinTransformer(img_size=(128,128,128), in_chans=4, embed_dim=48, ...)
```
New (MONAI ≥ 1.4):
```python
SwinTransformer(in_chans=4, embed_dim=48, ...)  # no img_size
```
The transformer handles variable input sizes dynamically. Our encoder.py is written for MONAI ≥ 1.4.

### MONAI SwinTransformer requires einops

MONAI's SwinTransformer uses `from einops import rearrange` internally. This is not automatically installed with monai. Always include `einops>=0.6.0` in requirements.

### RandCropByPosNegLabeld returns a list

MONAI's `RandCropByPosNegLabeld` with `num_samples >= 1` wraps the output in a list. Code must unwrap it:
```python
result = self._cache_dataset[idx]
item = result[0] if isinstance(result, list) else result
```

### torch.load requires weights_only=False for custom objects

PyTorch 2.6+ changed `torch.load` default to `weights_only=True`. Loading checkpoints that contain Python dataclass objects (like `Phase2Config`) fails with a security error. Use `weights_only=False` for our own checkpoints:
```python
ckpt = torch.load(path, map_location="cpu", weights_only=False)
```

### BraTS 2024 uses label 4 for ET

BraTS 2021 uses label 3 for ET. BraTS 2024 GLI uses label 4. The `RemapBraTS2024Labels` transform converts 4→3 during loading, keeping the label space consistent with `CCRConfig.concept_names`.

### UnetrBasicBlock requires matching patch_embed patch size

`SpatialPadd` must pad to at least `spatial_size` before `RandCropByPosNegLabeld`. If the padded volume is smaller than 128³, MONAI's SwinTransformer will fail silently or produce wrong shapes.

### Phase2Config.__post_init__ enforces total_epochs==80

The curriculum total_epochs is fixed at 80. Test fixtures that set a different `total_epochs` will fail. Tests must use `total_epochs=80` (the curriculum boundaries can be adjusted for faster test cycles).

---

*Phase 2 complete. Next: Phase 3 — CCR-Retrofit (frozen backbone insertion, DD measurement).*
