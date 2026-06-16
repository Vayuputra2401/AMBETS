"""
test_metrics.py — Unit tests for ConceptAlignmentScore (CAS_all + CAS_fg)
and ExpertUtilizationTracker.

Tests are ordered from structural → mathematical → behavioural → edge cases.
"""

import math
import pytest
import torch
import torch.nn.functional as F

from ccr.utils.metrics import ConceptAlignmentScore, ExpertUtilizationTracker


# ---------------------------------------------------------------------------
# Shared test setup
# ---------------------------------------------------------------------------

CONCEPT_NAMES = ("background", "necrotic_core", "edema", "enhancing_tumor")
K = 4
B, N = 2, 256


@pytest.fixture
def cas():
    return ConceptAlignmentScore(num_concepts=K, concept_names=CONCEPT_NAMES)


@pytest.fixture
def routing_probs_random():
    torch.manual_seed(42)
    return F.softmax(torch.randn(B, N, K), dim=-1)


@pytest.fixture
def token_labels_brats():
    """Approximate BraTS class balance: 80% BG, 8% NCR, 7% Edema, 5% ET."""
    labels = torch.zeros(B, N, dtype=torch.long)
    labels[:, 204:224] = 1   # NCR
    labels[:, 224:242] = 2   # Edema
    labels[:, 242:256] = 3   # ET
    return labels


# ---------------------------------------------------------------------------
# ConceptAlignmentScore — basic structure
# ---------------------------------------------------------------------------

class TestCASStructure:

    def test_compute_returns_all_concepts(self, cas, routing_probs_random, token_labels_brats):
        cas.update(routing_probs_random, token_labels_brats)
        result = cas.compute()
        assert set(result.keys()) == set(CONCEPT_NAMES)

    def test_compute_fg_returns_foreground_concepts_only(
        self, cas, routing_probs_random, token_labels_brats
    ):
        """CAS_fg must NOT include background (k=0)."""
        cas.update(routing_probs_random, token_labels_brats)
        result_fg = cas.compute_fg()
        assert "background" not in result_fg
        assert set(result_fg.keys()) == {"necrotic_core", "edema", "enhancing_tumor"}

    def test_reset_clears_accumulators(self, cas, routing_probs_random, token_labels_brats):
        cas.update(routing_probs_random, token_labels_brats)
        cas.reset()
        # After reset, Pearson should be 0.0 (n < 2)
        result = cas.compute()
        for v in result.values():
            assert v == 0.0

    def test_values_in_range(self, cas, routing_probs_random, token_labels_brats):
        cas.update(routing_probs_random, token_labels_brats)
        for v in cas.compute().values():
            assert -1.0 - 1e-6 <= v <= 1.0 + 1e-6
        for v in cas.compute_fg().values():
            assert -1.0 - 1e-6 <= v <= 1.0 + 1e-6

    def test_mismatched_concept_names_raises(self):
        with pytest.raises(ValueError):
            ConceptAlignmentScore(num_concepts=4, concept_names=("a", "b"))


# ---------------------------------------------------------------------------
# ConceptAlignmentScore — CAS_all correctness
# ---------------------------------------------------------------------------

class TestCASAllCorrecness:

    def test_perfect_routing_gives_cas_near_one(self):
        """
        When routing_probs is exactly one-hot matching token labels,
        CAS_all(k) should be near 1.0 for all k.
        """
        cas    = ConceptAlignmentScore(num_concepts=K, concept_names=CONCEPT_NAMES)
        labels = torch.zeros(1, 200, dtype=torch.long)
        labels[:, 50:100]  = 1
        labels[:, 100:150] = 2
        labels[:, 150:]    = 3

        # Perfect one-hot routing
        probs = torch.zeros(1, 200, K)
        for k in range(K):
            probs[labels == k, k] = 1.0

        cas.update(probs, labels)
        result = cas.compute()
        for k, name in enumerate(CONCEPT_NAMES):
            assert result[name] > 0.9, (
                f"CAS_all({name}) = {result[name]:.4f} — expected > 0.9 for perfect routing"
            )

    def test_random_routing_gives_low_cas(self, cas, routing_probs_random, token_labels_brats):
        """Random routing should give CAS values close to zero."""
        cas.update(routing_probs_random, token_labels_brats)
        result = cas.compute()
        # Not every value must be near zero, but average should be low
        avg_abs = sum(abs(v) for v in result.values()) / len(result)
        assert avg_abs < 0.5, f"Expected low CAS for random routing, got avg={avg_abs:.3f}"

    def test_accumulation_across_batches(self):
        """Updating twice with the same data should give the same CAS as once with 2× data."""
        cas_once  = ConceptAlignmentScore(num_concepts=K, concept_names=CONCEPT_NAMES)
        cas_twice = ConceptAlignmentScore(num_concepts=K, concept_names=CONCEPT_NAMES)

        torch.manual_seed(5)
        probs  = F.softmax(torch.randn(B, N, K), dim=-1)
        labels = torch.zeros(B, N, dtype=torch.long)
        labels[:, 200:] = 1

        cas_once.update(probs, labels)

        cas_twice.update(probs, labels)
        cas_twice.update(probs, labels)

        # compute() from twice should equal once (same data, Pearson is invariant to
        # scale of n when all batches are identical)
        for name in CONCEPT_NAMES:
            assert abs(cas_once.compute()[name] - cas_twice.compute()[name]) < 1e-5


# ---------------------------------------------------------------------------
# ConceptAlignmentScore — CAS_fg correctness (Improvement 5)
# ---------------------------------------------------------------------------

class TestCASFgCorrectness:

    def test_perfect_foreground_routing_gives_fg_cas_near_one(self):
        """
        Perfect routing on foreground tokens → CAS_fg ≈ 1 for all foreground concepts.
        """
        cas    = ConceptAlignmentScore(num_concepts=K, concept_names=CONCEPT_NAMES)
        labels = torch.zeros(1, 300, dtype=torch.long)
        labels[:, 100:150] = 1   # NCR
        labels[:, 150:200] = 2   # Edema
        labels[:, 200:250] = 3   # ET
        # tokens 0:100 and 250:300 are background

        probs = torch.zeros(1, 300, K)
        for k in range(K):
            probs[labels == k, k] = 1.0

        cas.update(probs, labels)
        fg_result = cas.compute_fg()

        for name in ("necrotic_core", "edema", "enhancing_tumor"):
            assert fg_result[name] > 0.9, (
                f"CAS_fg({name}) = {fg_result[name]:.4f} — expected > 0.9"
            )

    def test_cas_fg_strictly_excludes_background_tokens(self):
        """
        CAS_fg should differ from CAS_all when background tokens are poorly aligned.

        Setup: routing_probs correctly aligns foreground concepts but assigns
        all background tokens randomly → CAS_all is diluted, CAS_fg stays high.
        """
        cas = ConceptAlignmentScore(num_concepts=K, concept_names=CONCEPT_NAMES)

        N_total = 500
        labels  = torch.zeros(1, N_total, dtype=torch.long)
        labels[:, 300:350] = 1  # NCR  (50 tokens)
        labels[:, 350:400] = 2  # Edema(50 tokens)
        labels[:, 400:450] = 3  # ET   (50 tokens)
        # tokens 0:300 and 450:500 = background

        # Foreground tokens: perfect routing
        probs = torch.zeros(1, N_total, K)
        for k in range(K):
            probs[labels == k, k] = 1.0

        # Background tokens: random routing (noisy)
        torch.manual_seed(7)
        probs[:, :300, :]  = F.softmax(torch.randn(1, 300, K), dim=-1)
        probs[:, 450:, :]  = F.softmax(torch.randn(1, 50, K),  dim=-1)
        # Renormalise background part (already softmax, but set foreground to ensure sum=1)
        # For background tokens, the random routing does not match labels perfectly

        cas.update(probs, labels)
        all_result = cas.compute()
        fg_result  = cas.compute_fg()

        # CAS_fg for NCR should be higher than CAS_all for NCR
        # because CAS_all is diluted by the random background routing
        for name in ("necrotic_core", "edema", "enhancing_tumor"):
            assert fg_result[name] >= all_result[name] - 1e-3, (
                f"CAS_fg({name})={fg_result[name]:.4f} should be ≥ "
                f"CAS_all({name})={all_result[name]:.4f}"
            )

    def test_cas_fg_background_key_absent(self, cas, routing_probs_random, token_labels_brats):
        """compute_fg() must not return 'background' key."""
        cas.update(routing_probs_random, token_labels_brats)
        assert "background" not in cas.compute_fg()

    def test_cas_fg_all_foreground_batch_gives_nonzero(self):
        """Even when entire batch is foreground, CAS_fg is computed correctly."""
        cas    = ConceptAlignmentScore(num_concepts=K, concept_names=CONCEPT_NAMES)
        labels = torch.ones(1, 100, dtype=torch.long)   # all NCR
        labels[:, 50:] = 2                               # half NCR, half Edema

        probs = torch.zeros(1, 100, K)
        probs[:, :50, 1] = 1.0   # NCR tokens → expert 1
        probs[:, 50:, 2] = 1.0   # Edema tokens → expert 2

        cas.update(probs, labels)
        fg_result = cas.compute_fg()
        # NCR and Edema should be near 1; ET has no tokens (returns 0)
        assert fg_result["necrotic_core"] > 0.9
        assert fg_result["edema"] > 0.9


# ---------------------------------------------------------------------------
# ExpertUtilizationTracker
# ---------------------------------------------------------------------------

class TestExpertUtilizationTracker:

    def test_uniform_utilization(self):
        """
        When tokens are split equally across K experts, utilization ≈ 1/K each.
        """
        tracker = ExpertUtilizationTracker(
            num_concepts=K, concept_names=CONCEPT_NAMES, warmup_end_epoch=10
        )
        # 400 tokens: 100 per expert
        assignments = torch.tensor([k for k in range(K) for _ in range(100)])
        assignments = assignments.unsqueeze(0)   # [1, 400]
        tracker.update(assignments)
        util = tracker.compute()
        for name in CONCEPT_NAMES:
            assert abs(util[name] - 0.25) < 1e-6

    def test_no_collapse_warning_during_warmup(self):
        """check_collapse must return empty list during warmup (epoch ≤ 10)."""
        tracker = ExpertUtilizationTracker(
            num_concepts=K, concept_names=CONCEPT_NAMES, warmup_end_epoch=10
        )
        # All tokens go to expert 0 (collapsed)
        assignments = torch.zeros(1, 100, dtype=torch.long)
        tracker.update(assignments)
        assert tracker.check_collapse(epoch=5) == []

    def test_collapse_detected_after_warmup(self):
        """After warmup, experts with < 5% utilization should be flagged."""
        tracker = ExpertUtilizationTracker(
            num_concepts=K, concept_names=CONCEPT_NAMES, warmup_end_epoch=10
        )
        # All 400 tokens go to expert 0
        assignments = torch.zeros(1, 400, dtype=torch.long)
        tracker.update(assignments)

        import warnings
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            collapsed = tracker.check_collapse(epoch=15)

        assert len(collapsed) == 3  # experts 1, 2, 3 are collapsed
        assert "background" not in collapsed
        assert all(name in collapsed for name in ("necrotic_core", "edema", "enhancing_tumor"))
        assert len(w) == 3  # one warning per collapsed expert

    def test_reset_clears_counts(self):
        tracker = ExpertUtilizationTracker(
            num_concepts=K, concept_names=CONCEPT_NAMES, warmup_end_epoch=10
        )
        tracker.update(torch.zeros(1, 100, dtype=torch.long))
        tracker.reset()
        util = tracker.compute()
        assert all(v == 0.0 for v in util.values())

    def test_compute_before_update_returns_zeros(self):
        tracker = ExpertUtilizationTracker(
            num_concepts=K, concept_names=CONCEPT_NAMES, warmup_end_epoch=10
        )
        util = tracker.compute()
        assert all(v == 0.0 for v in util.values())
