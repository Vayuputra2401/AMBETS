# Causal sweep V2 — training record (run `20260817_044755`)

**Config:** `configs/causal/segresnet_v2.yaml` — SegResNet, `expert.residual=false`,
`ccr.skip_gate=0.5`. Both leaks identified in `evals/diagnostics_202608/RESULTS.md` are
closed: the expert returns `c_k(h)` alone (no residual carrying `h` past the routing), and
the decoder's bypassing skips are attenuated to half.

80 epochs, ~6.6 min/epoch, 04:47 -> 13:57 on the L4. `metrics.json` here is the full
per-epoch record pulled from the VM. Checkpoints remain at
`~/checkpoints/20260817_044755/` and `gs://research-brats/checkpoints/20260817_044755/`.

## Validation trajectory

| ep | CAS ncr/ed/et | Dice WT/TC/ET | util_ncr | seg loss |
|---|---|---|---|---|
| 5  | -0.025 / 0.097 / 0.106 | 0.842 / 0.676 / 0.671 | 0.242 | 0.1688 |
| 10 | -0.035 / 0.084 / 0.121 | 0.871 / 0.747 / 0.731 | 0.242 | 0.1453 |
| 15 | **0.689 / 0.915 / 0.898** | 0.846 / 0.759 / 0.745 | 0.063 | 0.1394 |
| 20 | 0.698 / 0.925 / 0.903 | 0.885 / 0.780 / 0.765 | 0.063 | 0.1321 |
| 30 | 0.757 / 0.923 / 0.906 | 0.894 / 0.803 / 0.790 | 0.063 | 0.1250 |
| 45 | 0.750 / 0.931 / 0.914 | 0.907 / 0.845 / 0.834 | 0.063 | 0.1171 |
| 60 | 0.782 / 0.937 / 0.922 | 0.912 / 0.844 / 0.829 | 0.063 | 0.1115 |
| 70 | 0.768 / 0.937 / 0.922 | 0.915 / 0.853 / 0.838 | 0.063 | 0.1096 |
| 80 | 0.765 / 0.937 / 0.921 | 0.914 / 0.853 / 0.838 | 0.063 | 0.1088 |

Final (val, ep80): CAS **0.765 / 0.937 / 0.921**, Dice **0.914 / 0.853 / 0.838**.
V0 for reference (test, ep80): CAS 0.802 / 0.962 / 0.954, Dice 0.927 / 0.883 / 0.869.

## What this establishes

1. **The gated architecture trains.** This was the first run where the expert output *is*
   the bottleneck signal rather than a correction to it, so a dead or diverging seg loss was
   a live risk (it is why `_init_weights` uses gain=1.0 when `residual=False`). Seg loss fell
   monotonically 0.305 -> 0.109 with no instability.
2. **Alignment survived closing both leaks.** CAS was at floor through warm-up
   (-0.03 / 0.08 / 0.12 at ep10 — routing unsupervised), then jumped to 0.689 / 0.915 / 0.898
   the moment `L_align` activated at epoch 11. Final CAS is within 0.04 of V0 on every
   concept. Gating the residual and halving the skips did NOT cost the concept structure,
   which was the main risk going in.
3. **Utilization is stable and interpretable.** NCR sat at 24.2% during warm-up (near-uniform,
   i.e. meaningless) and snapped to 6.3% at epoch 11, holding there for 70 epochs — the router
   learning true concept frequency rather than collapsing. Never approached the 5% expert
   re-init threshold.
4. **Dice converged ~1.5-3 points below V0.** WT 0.914 vs 0.927, TC 0.853 vs 0.883, ET 0.838
   vs 0.869. The curve flattened after ~epoch 45; the epoch-61 refinement phase gave only a
   small bump (TC 0.844 -> 0.853). **Caveat: these are VAL numbers and V0's are TEST** — the
   apples-to-apples comparison requires `evaluate.py --split test` on this checkpoint.

## What it does NOT yet establish

**Whether closing the leaks bought any causal control.** That is the whole point of the sweep
and it is not in this file. It comes from `routing_intervention.py`: the off-diagonal
label-flip fraction, which was **0.002** for V0. A ~2-point Dice cost is a bargain if that
number moves substantially and a bad trade if it does not.

Pending on this checkpoint (~1h GPU):

```bash
CKPT=~/checkpoints/20260817_044755/epoch_0080.pth
CFG=configs/causal/segresnet_v2.yaml
python pipeline/evaluate.py           --checkpoint $CKPT --model segresnet --config $CFG --env gcp --split test
python pipeline/routing_intervention.py --checkpoint $CKPT --model segresnet --config $CFG --env gcp --split test --n_cases 0 --save_dir ~/checkpoints/evals/v2_intervention
python pipeline/expert_diagnostics.py --checkpoint $CKPT --model segresnet --config $CFG --env gcp --split test --n_cases 40 --save_dir ~/checkpoints/evals/v2_expert_diag
```
