"""
expert_diagnostics.py — why did the routing intervention do nothing?

The causal intervention (pipeline/routing_intervention.py) found that re-routing tokens to a
different expert barely moves the prediction. That is either (a) an artifact of the experiment,
(b) the experts being functionally interchangeable, or (c) the decoder not needing the
bottleneck at all. This script distinguishes them, because the three have very different
consequences for the paper's central claim.

Measurements
------------
1. RESIDUAL SCALE  ||c_k(h)|| / ||h||
   Each expert computes `h + c_k(h)` with c initialised near zero (gain=0.01). If c stays
   small, every expert is approximately the identity and routing cannot matter regardless of
   what the router decides.

2. PAIRWISE EXPERT DIVERGENCE  ||E_a(h) - E_b(h)|| / ||h||
   The direct test of functional redundancy. Nothing in the objective forces experts apart:
   L_align shapes the ROUTER's probabilities, L_diversity shapes ROUTING VECTORS -- neither
   constrains what an expert computes. This is the quantity that would have to be non-trivial
   for the intervention to have anything to act on.

3. BOTTLENECK NECESSITY
   How much does the decoder actually depend on the CCR bottleneck at all?
     - bypass      : feed the decoder the RAW encoder tokens (skip the experts entirely)
     - zero        : remove the CCR contribution entirely
   If Dice barely moves when the bottleneck is destroyed, the decision does not live there
   and no amount of expert specialisation would make routing causally decisive.

   In DIRECT MODE the module reaches the output through two paths -- the expert features
   the decoder consumes, and an additive routing term -- so two more modes are reported:
     - zero_expert : only the decoder's bottleneck input removed
     - zero_routing: only the additive routing term removed. THIS is the direct-mode
                     number: it isolates how much the routing itself drives the prediction.
   Ablating only `expert_outputs` (what "zero" meant before direct mode existed) would
   leave the routing term intact and wrongly report the bottleneck as unnecessary.

Together these say whether the null intervention result is fixable (experts collapsed to
identity -> the objective needs a term that separates them) or structural (the decoder routes
around the bottleneck via skips -> the architecture cannot support the claim as stated).

Usage
-----
    python pipeline/expert_diagnostics.py \
        --checkpoint ~/checkpoints/20260621_120750/epoch_0080.pth \
        --model ccrnet --config configs/brats_phase2.yaml --env gcp \
        --split test --n_cases 20
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent / "data"))
sys.path.insert(0, str(Path(__file__).parent))

from ccrnet.config.phase2_config import Phase2Config
from ccrnet.config.env_config import apply_env
from ccrnet.models.factory import build_model, ARCHES
from ccrnet.utils.checkpoint import load_checkpoint
from brats_dataset import get_dataloader


def _dice(pred: torch.Tensor, gt: torch.Tensor, k: int) -> float:
    p, g = (pred == k), (gt == k)
    denom = float(p.sum() + g.sum())
    if denom == 0:
        return float("nan")
    return float(2.0 * (p & g).sum() / denom)


def main() -> None:
    _default_config = str(Path(__file__).parent.parent / "configs" / "brats_phase2.yaml")
    ap = argparse.ArgumentParser(description="Diagnose the null routing-intervention result")
    ap.add_argument("--checkpoint", type=str, required=True)
    ap.add_argument("--env",        type=str, default="local")
    ap.add_argument("--config",     type=str, default=_default_config)
    ap.add_argument("--split",      type=str, default="test", choices=["val", "test"])
    ap.add_argument("--n_cases",    type=int, default=20)
    ap.add_argument("--model",      type=str, default="ccrnet", choices=list(ARCHES))
    ap.add_argument("--save_dir",   type=str, default="")
    ap.add_argument("--device",     type=str, default="")
    args = ap.parse_args()

    config = Phase2Config.from_yaml(args.config)
    apply_env(config, env=args.env, config_dir=Path(__file__).parent.parent / "configs")
    device = torch.device(args.device if args.device else
                          ("cuda" if torch.cuda.is_available() else "cpu"))
    save_dir = args.save_dir or str(
        Path(config.training.checkpoint_dir) / "evals" / f"{args.split}_expert_diag")
    os.makedirs(save_dir, exist_ok=True)

    model = build_model(config, args.model).to(device)
    start_epoch, _ = load_checkpoint(args.checkpoint, model)
    model.eval()
    ccr = getattr(model, "ccr", None)
    if ccr is None:
        raise SystemExit("Model has no CCR module.")
    names = config.ccr.concept_names
    K = len(ccr.experts)
    # zero        : both CCR paths removed -- total bottleneck necessity (comparable to V0/V2)
    # zero_expert : only the decoder's bottleneck input removed
    # zero_routing: only the additive routing term removed -- THE direct-mode number, it
    #               measures how much the routing itself drives the prediction
    direct = bool(getattr(config.ccr, "direct_mode", False))
    modes = (("bypass", "zero_expert", "zero_routing", "zero") if direct
             else ("bypass", "zero"))
    print(f"Loaded checkpoint (epoch {start_epoch - 1}) on {device} [{args.model}], K={K}")

    loader = get_dataloader(
        data_root=config.data.data_root, split=args.split, batch_size=1,
        num_workers=config.data.num_workers, spatial_size=config.data.spatial_size,
        brats_version=config.data.brats_version, val_fraction=config.data.val_fraction,
        test_fraction=config.data.test_fraction, seed=config.data.seed,
    )
    n_cases = min(args.n_cases, len(loader.dataset)) if args.n_cases > 0 else len(loader.dataset)

    # capture the tokens entering the CCR module
    grabbed: Dict[str, torch.Tensor] = {}
    ccr.register_forward_pre_hook(lambda m, inp: grabbed.__setitem__("tokens", inp[0].detach()))

    res_scale: List[np.ndarray] = []
    pair_div: List[np.ndarray] = []
    dice_rows: List[Dict[str, float]] = []

    seen = 0
    for batch in tqdm(loader, total=n_cases, desc="  diagnostics"):
        if seen >= n_cases:
            break
        image = batch["image"].to(device)
        label = batch["label"].to(device)[0]

        with torch.no_grad():
            out = model(image)
            h = grabbed["tokens"][0]                       # [N, D] tokens into CCR
            hn = h.norm(dim=-1).clamp_min(1e-8)            # [N]

            # 1. residual scale per expert, and 2. pairwise divergence
            eouts = [ccr.experts[k](h) for k in range(K)]  # each [N, D]
            res_scale.append(np.array(
                [float(((eouts[k] - h).norm(dim=-1) / hn).mean()) for k in range(K)]))
            pd = np.full((K, K), np.nan)
            for a in range(K):
                for b in range(K):
                    pd[a, b] = float(((eouts[a] - eouts[b]).norm(dim=-1) / hn).mean())
            pair_div.append(pd)

            # 3. bottleneck necessity
            base_pred = out["seg_logits"].argmax(1)[0]
            row = {f"dice_base_{names[k]}": _dice(base_pred, label, k) for k in range(1, K)}

            # In direct_mode the CCR module reaches the output through TWO paths: the
            # expert features the decoder consumes, AND an additive routing term
            # (seg_logits = upsample(effective_logits) + refine_scale * decoder_out).
            # Patching only `expert_outputs` would leave the routing term intact, so a
            # "zeroed" bottleneck would still drive the prediction and we would wrongly
            # conclude the bottleneck is unnecessary -- the exact opposite of the truth.
            # So ablate the paths separately, and together.
            for mode in modes:
                orig_forward = ccr.forward

                def patched(tokens, _orig=orig_forward, _mode=mode):
                    o = dict(_orig(tokens))
                    if _mode == "bypass":
                        o["expert_outputs"] = tokens
                    elif _mode in ("zero_expert", "zero"):
                        o["expert_outputs"] = torch.zeros_like(tokens)
                    if _mode in ("zero_routing", "zero"):
                        eff = o.get("effective_logits")
                        if eff is not None:
                            o["effective_logits"] = torch.zeros_like(eff)
                    return o

                ccr.forward = patched
                try:
                    pred = model(image)["seg_logits"].argmax(1)[0]
                finally:
                    ccr.forward = orig_forward
                for k in range(1, K):
                    row[f"dice_{mode}_{names[k]}"] = _dice(pred, label, k)

            dice_rows.append(row)
        seen += 1

    res_scale_m = np.nanmean(np.stack(res_scale), axis=0)
    pair_div_m = np.nanmean(np.stack(pair_div), axis=0)

    print("\n=== 1. Residual scale  ||E_k(h) - h|| / ||h||  (how far each expert moves a token) ===")
    for k in range(K):
        print(f"  {names[k]:>18s}  {res_scale_m[k]:.4f}")

    print("\n=== 2. Pairwise expert divergence  ||E_a(h) - E_b(h)|| / ||h|| ===")
    print(f"{'':>18s}" + "".join(f"{names[b][:9]:>11s}" for b in range(K)))
    for a in range(K):
        print(f"{names[a][:17]:>18s}" + "".join(f"{pair_div_m[a, b]:>11.4f}" for b in range(K)))

    print("\n=== 3. Bottleneck necessity: Dice when the CCR output is replaced ===")
    print(f"{'':>18s}{'base':>10s}" + "".join(f"{m:>13s}" for m in modes))
    summary: Dict = {"residual_scale": {names[k]: float(res_scale_m[k]) for k in range(K)},
                     "pairwise_divergence": pair_div_m.tolist(),
                     "concept_names": list(names), "n_cases": seen,
                     "checkpoint": args.checkpoint}
    for k in range(1, K):
        b = float(np.nanmean([r[f"dice_base_{names[k]}"] for r in dice_rows]))
        cells = {m: float(np.nanmean([r[f"dice_{m}_{names[k]}"] for r in dice_rows]))
                 for m in modes}
        print(f"{names[k][:17]:>18s}{b:>10.4f}" + "".join(f"{cells[m]:>13.4f}" for m in modes))
        summary[f"dice_{names[k]}"] = {"base": b, **cells}

    json_path = os.path.join(save_dir, f"{args.split}_expert_diag.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nJSON: {json_path}")
    print("\nReading the result:")
    print("  residual scale ~0      -> experts are the identity; routing has no lever")
    print("  pairwise div  ~0       -> experts functionally redundant (the L_diversity gap)")
    print("  bypass/zero Dice ~base -> the decoder does not need the bottleneck at all")


if __name__ == "__main__":
    main()
