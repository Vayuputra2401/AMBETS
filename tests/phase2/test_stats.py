"""
test_stats.py — Tests for pipeline/stats.py (W5 bootstrap CIs + paired significance).

Pure-numpy / CPU. No model, no data.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT / "pipeline"))

from stats import bootstrap_ci, paired_compare  # noqa: E402


def test_bootstrap_ci_brackets_mean():
    vals = np.array([0.80, 0.82, 0.79, 0.81, 0.83, 0.78, 0.80])
    mean, lo, hi, n = bootstrap_ci(vals, n_boot=2000, seed=0)
    assert n == 7
    assert lo <= mean <= hi
    assert math.isclose(mean, float(vals.mean()), rel_tol=1e-9)
    assert hi - lo < 0.2   # tight-ish CI for low-variance data


def test_bootstrap_ci_ignores_nan():
    vals = np.array([0.5, np.nan, 0.7, np.nan, 0.6])
    mean, lo, hi, n = bootstrap_ci(vals, n_boot=1000, seed=1)
    assert n == 3
    assert math.isclose(mean, np.nanmean(vals), rel_tol=1e-9)


def test_bootstrap_ci_all_nan():
    mean, lo, hi, n = bootstrap_ci(np.array([np.nan, np.nan]))
    assert n == 0
    assert math.isnan(mean) and math.isnan(lo) and math.isnan(hi)


def test_paired_compare_detects_clear_difference():
    ccr = np.array([0.90, 0.85, 0.88, 0.92, 0.87, 0.90, 0.89])
    base = np.array([0.50, 0.55, 0.52, 0.48, 0.50, 0.53, 0.51])
    cmp = paired_compare(ccr, base, n_boot=3000, seed=0)
    assert cmp is not None
    assert cmp["n_pairs"] == 7
    assert cmp["mean_diff"] > 0.3
    assert cmp["ci_lo"] > 0.0                 # CI excludes zero
    assert cmp["significant_95"] is True


def test_paired_compare_matches_only_finite_pairs():
    ccr = np.array([0.9, np.nan, 0.8, 0.85])
    base = np.array([0.5, 0.4, np.nan, 0.5])
    cmp = paired_compare(ccr, base, n_boot=500, seed=0)
    assert cmp is not None
    assert cmp["n_pairs"] == 2   # only indices 0 and 3 are finite in both


def test_paired_compare_too_few_pairs_returns_none():
    ccr = np.array([0.9, np.nan, np.nan])
    base = np.array([0.5, 0.4, 0.3])
    assert paired_compare(ccr, base) is None
