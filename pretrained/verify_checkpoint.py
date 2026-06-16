"""
verify_checkpoint.py — Inspect a SwinUNETR pretrained checkpoint.

Usage:
    python pretrained/verify_checkpoint.py pretrained/model_swinvit.pt

Prints:
  - All key prefixes present in the checkpoint
  - Count of swinViT.* keys (should be > 0 for encoder loading to work)
  - Whether the checkpoint is loadable into SwinEncoder3D
"""

import sys
import torch

path = sys.argv[1] if len(sys.argv) > 1 else "pretrained/model_swinvit.pt"

print(f"Loading: {path}")
ckpt = torch.load(path, map_location="cpu", weights_only=False)

# Get state dict
state = ckpt.get("state_dict", ckpt)
if not isinstance(state, dict):
    print(f"WARNING: checkpoint top-level type is {type(ckpt)}, not dict")
    state = ckpt

all_keys = sorted(state.keys())
print(f"\nTotal keys: {len(all_keys)}")

# Show prefix distribution
from collections import Counter
prefixes = Counter(k.split(".")[0] for k in all_keys)
print("\nKey prefixes:")
for prefix, count in prefixes.most_common():
    print(f"  {prefix}: {count} keys")

# Count encoder keys
# Apply the three remappings used in SwinEncoder3D._load_pretrained
mapped = {}
for k, v in state.items():
    k2 = k.removeprefix("module.")
    k2 = k2.replace("mlp.fc1", "mlp.linear1").replace("mlp.fc2", "mlp.linear2")
    mapped[k2] = v

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from ccrnet.config.phase2_config import SwinConfig
from ccrnet.models.encoder import SwinEncoder3D

cfg = SwinConfig()
enc = SwinEncoder3D(cfg)
model_sd = enc.swin.state_dict()

# Channel-inflate patch_embed if needed
pe_key = "patch_embed.proj.weight"
if pe_key in mapped and mapped[pe_key].shape != model_sd[pe_key].shape:
    c_ckpt  = mapped[pe_key].shape[1]
    c_model = model_sd[pe_key].shape[1]
    mapped[pe_key] = mapped[pe_key].repeat(1, c_model, 1, 1, 1)[:, :c_model] / c_ckpt
    print(f"  patch_embed inflated: {c_ckpt}-ch -> {c_model}-ch")

missing, unexpected = enc.swin.load_state_dict(mapped, strict=False)
n_model  = len(model_sd)
n_loaded = n_model - len(missing)
print(f"\nLoad result:")
print(f"  Matched keys   : {n_loaded}/{n_model}  ({100*n_loaded/n_model:.1f}%)")
print(f"  Missing keys   : {len(missing)}")
print(f"  Unexpected keys: {len(unexpected)}  (SSL heads — expected)")
if len(missing) == 0:
    print("  ALL MODEL KEYS MATCHED — checkpoint is fully compatible.")
else:
    print(f"  First missing: {missing[:3]}")
