"""
stats.py — Bootstrap confidence intervals + paired significance (W5).

Post-processing only (no GPU, no model). Reads the per-patient CSVs produced by
evaluate.py (`{split}_results.csv`) and attribution_baselines.py
(`{split}_baselines.csv`) and produces:

  1. Bootstrap 95% CIs for every per-patient CCR metric (Dice, DD, deletion/insertion
     AUC in both region windows) — turns single point estimates into interval estimates.
  2. Paired CCR-vs-baseline comparisons for deletion/insertion AUC (per method, concept,
     and region window): mean paired difference, 95% bootstrap CI of the difference, and
     a Wilcoxon signed-rank p-value on the matched (same-patient) cases.
  3. Optional multi-seed aggregation: mean +/- std across several runs' summary.json
     (the seed-variance rows for the headline metrics).

Why: a ranked-venue reviewer expects uncertainty, not bare point estimates, and a
significance test behind "CCR beats the baselines" rather than "the number is bigger."

Usage
-----
    # CIs + paired tests from the local evals/ tree:
    python pipeline/stats.py --evals_dir evals --split test

    # Explicit CSV paths:
    python pipeline/stats.py \
        --ccr_csv evals/test/test_results.csv \
        --baseline_csv evals/test_baselines/test_baselines.csv \
        --out evals/test/test_stats.json

    # Multi-seed variance across runs (headline scalar metrics):
    python pipeline/stats.py --seed_summaries \
        evals/seed42/test_summary.json evals/seed123/test_summary.json evals/seed7/test_summary.json
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

CONCEPTS = ("necrotic_core", "edema", "enhancing_tumor")
REGION_MODES = ("gt", "pred")
# Method taxonomy — mirrors the paper's three-way split (see docs/ccr_explainability_positioning.md):
#   post-hoc saliency : one map per input, structurally cannot emit per-concept explanations
#   output-level      : the model's own segmentation posterior — discriminates concepts, but it
#                       IS the prediction (circular; DD = 0 by construction), not an explanation
#   CCR routing       : concept-labelled AND causally upstream of the prediction
# segsoftmax is compared against, but is deliberately NOT called a "baseline" — keeping the
# constants separate stops it being described as a post-hoc method anywhere downstream.
BASELINE_METHODS = ("gradcam", "gradient", "ig", "occlusion")
OUTPUT_METHODS = ("segsoftmax", "segsoftmax_16")
COMPARISON_METHODS = BASELINE_METHODS + OUTPUT_METHODS
ALIGN_METHODS = ("ccr",) + COMPARISON_METHODS
N_BOOT = 10000
ALPHA = 0.05


# ---------------------------------------------------------------------------
# IO
# ---------------------------------------------------------------------------

def _to_float(x: str) -> float:
    if x is None:
        return float("nan")
    s = str(x).strip().lower()
    if s in ("", "nan", "none"):
        return float("nan")
    try:
        return float(s)
    except ValueError:
        return float("nan")


def load_csv(path: str) -> List[Dict[str, object]]:
    """Read a per-patient CSV into a list of dicts (patient_id str, rest float)."""
    rows: List[Dict[str, object]] = []
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            row: Dict[str, object] = {}
            for k, v in r.items():
                row[k] = v if k == "patient_id" else _to_float(v)
            rows.append(row)
    return rows


def column(rows: List[Dict[str, object]], key: str) -> np.ndarray:
    return np.array([row.get(key, float("nan")) for row in rows], dtype=float)


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def bootstrap_ci(
    values: np.ndarray, n_boot: int = N_BOOT, alpha: float = ALPHA, seed: int = 0
) -> Tuple[float, float, float, int]:
    """Percentile bootstrap CI of the mean over finite values. Returns (mean, lo, hi, n)."""
    v = values[np.isfinite(values)]
    n = int(v.size)
    if n == 0:
        return float("nan"), float("nan"), float("nan"), 0
    if n == 1:
        return float(v[0]), float(v[0]), float(v[0]), 1
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    boot_means = v[idx].mean(axis=1)
    lo = float(np.percentile(boot_means, 100 * alpha / 2))
    hi = float(np.percentile(boot_means, 100 * (1 - alpha / 2)))
    return float(v.mean()), lo, hi, n


def paired_compare(
    ccr: np.ndarray, base: np.ndarray, n_boot: int = N_BOOT, alpha: float = ALPHA, seed: int = 0
) -> Optional[Dict[str, float]]:
    """
    Paired CCR - baseline comparison on matched cases (aligned index-wise).

    Keeps pairs where BOTH values are finite. Returns mean diff, its bootstrap CI, a
    Wilcoxon signed-rank two-sided p-value, and n_pairs. None if <2 usable pairs.
    """
    mask = np.isfinite(ccr) & np.isfinite(base)
    a, b = ccr[mask], base[mask]
    n = int(a.size)
    if n < 2:
        return None
    diff = a - b
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    boot = diff[idx].mean(axis=1)
    lo = float(np.percentile(boot, 100 * alpha / 2))
    hi = float(np.percentile(boot, 100 * (1 - alpha / 2)))

    pval = float("nan")
    try:
        from scipy.stats import wilcoxon
        if np.any(diff != 0):
            pval = float(wilcoxon(a, b).pvalue)
    except Exception:
        pval = float("nan")

    return {
        "mean_ccr": float(a.mean()),
        "mean_baseline": float(b.mean()),
        "mean_diff": float(diff.mean()),
        "ci_lo": lo,
        "ci_hi": hi,
        "wilcoxon_p": pval,
        "n_pairs": n,
        "significant_95": bool(np.isfinite(lo) and np.isfinite(hi) and (lo > 0 or hi < 0)),
    }


# ---------------------------------------------------------------------------
# Reports
# ---------------------------------------------------------------------------

def ccr_confidence_intervals(ccr_rows: List[Dict[str, object]]) -> Dict[str, Dict[str, float]]:
    """Bootstrap CI for every numeric per-patient CCR metric of interest."""
    keys = [k for k in ccr_rows[0].keys() if k != "patient_id"]
    interesting = [
        k for k in keys
        if k.startswith(("dice_", "dd_", "del_auc_", "ins_auc_")) or k == "ece"
    ]
    out: Dict[str, Dict[str, float]] = {}
    for k in interesting:
        mean_v, lo, hi, n = bootstrap_ci(column(ccr_rows, k))
        out[k] = {"mean": mean_v, "ci_lo": lo, "ci_hi": hi, "n": n}
    return out


def paired_vs_baselines(
    ccr_rows: List[Dict[str, object]],
    base_rows: List[Dict[str, object]],
) -> Dict[str, Dict[str, float]]:
    """Match on patient_id and compare CCR vs each baseline for del/ins x region x concept."""
    ccr_by_id = {r["patient_id"]: r for r in ccr_rows}
    base_by_id = {r["patient_id"]: r for r in base_rows}
    common = [pid for pid in base_by_id if pid in ccr_by_id]  # baselines are the subset

    results: Dict[str, Dict[str, float]] = {}
    for metric in ("del", "ins"):
        ccr_prefix = f"{metric}_auc"      # CCR key: del_auc_{region}_{concept}
        for region in REGION_MODES:
            for concept in CONCEPTS:
                ccr_key = f"{ccr_prefix}_{region}_{concept}"
                ccr_vals = np.array([ccr_by_id[p].get(ccr_key, float("nan")) for p in common], float)
                for method in COMPARISON_METHODS:
                    base_key = f"{method}_{metric}_{region}_{concept}"
                    if not any(base_key in r for r in base_rows):
                        continue
                    base_vals = np.array([base_by_id[p].get(base_key, float("nan")) for p in common], float)
                    cmp = paired_compare(ccr_vals, base_vals)
                    if cmp is not None:
                        results[f"{metric}|{region}|{concept}|ccr_vs_{method}"] = cmp
    return results


def merge_align_rows(paths: List[str]) -> List[Dict[str, object]]:
    """
    Column-wise union of several per-patient CSVs, joined on patient_id.

    Used for BOTH the alignment CSVs and the del/ins baseline CSVs, because in each case no
    single run holds every method: the original sweeps produced ccr+gradcam+gradient+ig+
    occlusion, while the segsoftmax fix (2026-08-06) produced segsoftmax separately. All cover
    the same 168 test cases (same seed, same split), so a join reconstructs the full table for
    free instead of re-running IG/occlusion on the GPU.

    The shared `ccr_*` columns are re-measured in every run. They come from the same
    deterministic forward on the same checkpoint, so they must agree; a mismatch means the
    files came from different checkpoints and the merge would be silently wrong. We keep the
    first file's value and warn loudly rather than averaging over an inconsistency.
    """
    merged: Dict[object, Dict[str, object]] = {}
    order: List[object] = []
    conflicts = 0
    for path in paths:
        for row in load_csv(path):
            pid = row.get("patient_id")
            if pid not in merged:
                merged[pid] = dict(row)
                order.append(pid)
                continue
            for key, val in row.items():
                if key == "patient_id":
                    continue
                prev = merged[pid].get(key)
                if key not in merged[pid]:
                    merged[pid][key] = val
                elif (isinstance(prev, float) and isinstance(val, float)
                      and math.isfinite(prev) and math.isfinite(val)
                      and abs(prev - val) > 1e-6):
                    conflicts += 1
    if conflicts:
        print(f"[warn] {conflicts} conflicting cell(s) across alignment CSVs — kept the first "
              f"file's values. Check that every CSV came from the SAME checkpoint.")
    return [merged[p] for p in order]


def alignment_analysis(align_rows: List[Dict[str, object]]) -> Dict[str, object]:
    """Concept-discriminability (foreground AUROC): per-method 95% CI + paired CCR-vs-baseline.

    Per-patient scalar per method = mean over concepts of {method}_auroc_fg_{concept} (nanmean).
    """
    per_method: Dict[str, np.ndarray] = {}
    for m in ALIGN_METHODS:
        vals = []
        for r in align_rows:
            cvals = [r.get(f"{m}_auroc_fg_{c}", float("nan")) for c in CONCEPTS]
            cvals = [v for v in cvals if isinstance(v, (int, float)) and not math.isnan(v)]
            vals.append(float(np.mean(cvals)) if cvals else float("nan"))
        per_method[m] = np.array(vals, float)

    out: Dict[str, object] = {"per_method": {}, "ccr_vs": {}}
    for m in ALIGN_METHODS:
        mean_v, lo, hi, n = bootstrap_ci(per_method[m])
        out["per_method"][m] = {"mean": mean_v, "ci_lo": lo, "ci_hi": hi, "n": n}
    for m in COMPARISON_METHODS:
        cmp = paired_compare(per_method["ccr"], per_method[m])
        if cmp is not None:
            out["ccr_vs"][m] = cmp
    return out


def multiseed_aggregate(summary_paths: List[str]) -> Dict[str, Dict[str, float]]:
    """mean +/- std across runs for the headline scalar metrics in each summary.json."""
    summaries = []
    for p in summary_paths:
        with open(p) as f:
            summaries.append(json.load(f))

    def _scalar(s: dict, key: str) -> float:
        v = s.get(key)
        if isinstance(v, dict):
            v = v.get("mean")
        return float(v) if v is not None else float("nan")

    keys = set()
    for s in summaries:
        for k, v in s.items():
            if k.startswith(("dice_", "cas_fg_", "auroc_", "del_auc_", "ins_auc_", "dd_")) or k == "routing_ece":
                keys.add(k)

    out: Dict[str, Dict[str, float]] = {}
    for k in sorted(keys):
        vals = np.array([_scalar(s, k) for s in summaries], float)
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            continue
        out[k] = {"mean": float(vals.mean()), "std": float(vals.std(ddof=0)), "n_seeds": int(vals.size)}
    return out


# ---------------------------------------------------------------------------
# Pretty-print
# ---------------------------------------------------------------------------

def _fmt(x: float, p: int = 3) -> str:
    return "nan" if (x is None or (isinstance(x, float) and math.isnan(x))) else f"{x:.{p}f}"


def print_report(cis: Dict, paired: Dict, seeds: Optional[Dict], align: Optional[Dict] = None) -> None:
    print("\n===== CCR per-patient metrics — bootstrap 95% CI =====")
    print(f"{'metric':32s} {'mean':>8s}  {'95% CI':>18s}   n")
    for k, s in cis.items():
        print(f"{k:32s} {_fmt(s['mean']):>8s}  [{_fmt(s['ci_lo'])}, {_fmt(s['ci_hi'])}]   {s['n']}")

    print("\n===== CCR vs baselines — paired difference (matched cases) =====")
    print(f"{'comparison':44s} {'dMean':>7s} {'95% CI of diff':>20s} {'wilcox p':>10s}  n  sig")
    for k, s in paired.items():
        sig = "*" if s.get("significant_95") else " "
        print(f"{k:44s} {_fmt(s['mean_diff']):>7s} "
              f"[{_fmt(s['ci_lo'])}, {_fmt(s['ci_hi'])}] {_fmt(s['wilcoxon_p'],4):>10s}  "
              f"{s['n_pairs']:>3d}  {sig}")

    if align:
        print("\n===== Concept-discriminability (fg AUROC, mean over concepts) — 95% CI =====")
        for m, s in align["per_method"].items():
            print(f"  {m:10s} {_fmt(s['mean'])}  [{_fmt(s['ci_lo'])}, {_fmt(s['ci_hi'])}]  n={s['n']}")
        print("  -- CCR vs baseline (paired) --")
        for m, s in align["ccr_vs"].items():
            sig = "*" if s.get("significant_95") else " "
            print(f"  ccr_vs_{m:10s} dMean={_fmt(s['mean_diff'])} "
                  f"[{_fmt(s['ci_lo'])}, {_fmt(s['ci_hi'])}]  p={_fmt(s['wilcoxon_p'], 4)}  "
                  f"n={s['n_pairs']}  {sig}")

    if seeds:
        print("\n===== Multi-seed variance (mean +/- std across seeds) =====")
        for k, s in seeds.items():
            print(f"{k:28s} {_fmt(s['mean'])} +/- {_fmt(s['std'])}  (n_seeds={s['n_seeds']})")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="Bootstrap CIs + paired significance (W5)")
    ap.add_argument("--evals_dir", type=str, default="evals",
                    help="Root evals dir; auto-locates {split}/{split}_results.csv and "
                         "{split}_baselines/{split}_baselines.csv.")
    ap.add_argument("--split", type=str, default="test", choices=["val", "test"])
    ap.add_argument("--ccr_csv", type=str, default="", help="Override CCR per-patient CSV path.")
    ap.add_argument("--baseline_csv", type=str, nargs="*", default=[],
                    help="Baseline del/ins CSVs to join on patient_id (default: auto-discover "
                         "{split}_baselines/ and {split}_baselines_segsoftmax/ under --evals_dir). "
                         "Multiple because segsoftmax was run separately from the post-hoc four.")
    ap.add_argument("--align_csv", type=str, nargs="*", default=[],
                    help="Alignment CSVs to join on patient_id (default: auto-discover "
                         "{split}_alignment/ and {split}_alignment_segsoftmax/ under --evals_dir). "
                         "All must come from the same checkpoint.")
    ap.add_argument("--seed_summaries", type=str, nargs="*", default=[],
                    help="Optional list of per-seed summary.json paths for variance rows.")
    ap.add_argument("--out", type=str, default="", help="Output JSON path.")
    args = ap.parse_args()

    ccr_csv = args.ccr_csv or os.path.join(args.evals_dir, args.split, f"{args.split}_results.csv")
    # As with the alignment CSVs, no single baseline run holds every method: the post-hoc four
    # and segsoftmax are produced by separate invocations, so join them on patient_id.
    base_csvs = args.baseline_csv or [
        os.path.join(args.evals_dir, f"{args.split}_baselines", f"{args.split}_baselines.csv"),
        os.path.join(args.evals_dir, f"{args.split}_baselines_segsoftmax",
                     f"{args.split}_baselines.csv"),
    ]
    base_csvs = [p for p in base_csvs if os.path.exists(p)]

    report: Dict[str, object] = {"ccr_csv": ccr_csv, "baseline_csvs": base_csvs, "split": args.split}

    cis: Dict = {}
    paired: Dict = {}
    if os.path.exists(ccr_csv):
        ccr_rows = load_csv(ccr_csv)
        cis = ccr_confidence_intervals(ccr_rows)
        report["ccr_confidence_intervals"] = cis
        if base_csvs:
            base_rows = merge_align_rows(base_csvs)
            paired = paired_vs_baselines(ccr_rows, base_rows)
            report["paired_vs_baselines"] = paired
        else:
            print("[warn] no baseline CSV found — skipping paired tests.")
    else:
        print(f"[warn] CCR CSV not found: {ccr_csv} — skipping CIs/paired tests.")

    align: Optional[Dict] = None
    # No single run holds every method (see merge_align_rows), so collect all alignment CSVs
    # for this split and join them on patient_id.
    align_csvs = args.align_csv or [
        p for p in (
            os.path.join(args.evals_dir, f"{args.split}_alignment", f"{args.split}_alignment.csv"),
            os.path.join(args.evals_dir, f"{args.split}_alignment_segsoftmax",
                         f"{args.split}_alignment.csv"),
        ) if os.path.exists(p)
    ]
    align_csvs = [p for p in align_csvs if os.path.exists(p)]
    if align_csvs:
        align = alignment_analysis(merge_align_rows(align_csvs))
        report["concept_alignment"] = align
        report["align_csvs"] = align_csvs

    seeds: Optional[Dict] = None
    if args.seed_summaries:
        seeds = multiseed_aggregate(args.seed_summaries)
        report["multiseed"] = seeds

    print_report(cis, paired, seeds, align)

    out_path = args.out or os.path.join(args.evals_dir, args.split, f"{args.split}_stats.json")
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nStats JSON: {out_path}")


if __name__ == "__main__":
    main()
