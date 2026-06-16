"""
metrics.py — CAS computation and expert utilization tracking.

Two metric classes:

1. ConceptAlignmentScore
   Computes two CAS variants per concept k, accumulated over the full validation set:

   CAS_all(k) — all tokens:
       CAS_all(k) = Pearson(P(t→k|x), M_k(t))  over all tokens t
       Inflated by background dominance (~80% of BraTS voxels).
       Report in supplementary for completeness.

   CAS_fg(k) — foreground-only (Improvement 5, docs/phase1_analysis.md §5):
       CAS_fg(k) = Pearson(P(t→k|x), M_k(t))  over non-background tokens only
       k ∈ {1, 2, ..., K-1}  (background concept k=0 is excluded)
       This is the honest primary metric — it answers "among tumor tokens, how
       well does routing match the clinical annotation?"
       Background has near-zero label variance → unreliable Pearson → excluded.
       Report as primary metric in paper.

2. ExpertUtilizationTracker
   Monitors per-expert token utilization percentages across batches.
   Detects expert collapse (< 5 % of tokens after warmup) and flags for
   expert re-initialisation.

Plan reference:
  - CAS definition: Section 5.1
  - Expert collapse threshold: Section 16.1 ("re-init if < 5 % after epoch 20")
  - Faithfulness = CAS (Proposition 1a): Section 5.2
  - CAS_fg as primary metric: docs/phase1_analysis.md §5, Improvement 5
"""

from __future__ import annotations

import warnings
from typing import Dict, List, Optional, Tuple

import torch


# ---------------------------------------------------------------------------
# Concept Alignment Score
# ---------------------------------------------------------------------------


class ConceptAlignmentScore:
    """
    Computes CAS_all(k) and CAS_fg(k) = Pearson(P(t→k|x), M_k(t)) per concept,
    over the full validation set.

    CAS(k) is the Pearson correlation between:
      P(t→k|x)  — routing probability for expert k at token t, given image x
      M_k(t)    — binary ground-truth subregion membership (1 if GT label = k)

    CAS ∈ [-1, 1]; a well-trained CCR model achieves CAS_fg ≥ 0.85 for k=1,2,3.

    Two variants tracked simultaneously:
      CAS_all(k) — all tokens, all concepts (background included)
      CAS_fg(k)  — foreground tokens only (labels > 0), concepts k=1..K-1
                   This is the correct primary metric.  Background has near-zero
                   label variance making Pearson unreliable.

    Implementation uses running sums to avoid storing all predictions in memory.
    Pearson is computed from five running statistics per concept:
        n        — total number of tokens accumulated
        sum_p    — Σ P(t→k)
        sum_m    — Σ M_k(t)
        sum_pp   — Σ P(t→k)²
        sum_mm   — Σ M_k(t)²
        sum_pm   — Σ P(t→k) × M_k(t)

    Pearson(P, M) = (n × Σ PM − Σ P × Σ M) /
                   (sqrt(n × Σ P² − (Σ P)²) × sqrt(n × Σ M² − (Σ M)²) + ε)

    Parameters
    ----------
    num_concepts : int   K
    concept_names : Tuple[str, ...]   length K, for dict keys in compute()

    Usage
    -----
        cas = ConceptAlignmentScore(num_concepts=4, concept_names=(...))
        for batch in val_loader:
            routing_probs, token_labels = ...
            cas.update(routing_probs, token_labels)
        scores_all = cas.compute()          # CAS_all — all tokens
        scores_fg  = cas.compute_fg()       # CAS_fg  — foreground only (primary)
        cas.reset()
    """

    _PEARSON_EPS: float = 1e-8
    _BG_CLASS:    int   = 0

    def __init__(
        self,
        num_concepts: int,
        concept_names: Tuple[str, ...],
    ) -> None:
        if len(concept_names) != num_concepts:
            raise ValueError(
                f"len(concept_names)={len(concept_names)} != num_concepts={num_concepts}"
            )
        self.num_concepts  = num_concepts
        self.concept_names = concept_names
        self._reset_accumulators()

    def _reset_accumulators(self) -> None:
        # CAS_all running sums — all tokens, all concepts
        self._n:      List[float] = [0.0] * self.num_concepts
        self._sum_p:  List[float] = [0.0] * self.num_concepts
        self._sum_m:  List[float] = [0.0] * self.num_concepts
        self._sum_pp: List[float] = [0.0] * self.num_concepts
        self._sum_mm: List[float] = [0.0] * self.num_concepts
        self._sum_pm: List[float] = [0.0] * self.num_concepts

        # CAS_fg running sums — foreground tokens only (labels > 0), concepts k>0
        self._n_fg:      List[float] = [0.0] * self.num_concepts
        self._sum_p_fg:  List[float] = [0.0] * self.num_concepts
        self._sum_m_fg:  List[float] = [0.0] * self.num_concepts
        self._sum_pp_fg: List[float] = [0.0] * self.num_concepts
        self._sum_mm_fg: List[float] = [0.0] * self.num_concepts
        self._sum_pm_fg: List[float] = [0.0] * self.num_concepts

    def update(
        self,
        routing_probs: torch.Tensor,
        token_labels:  torch.Tensor,
    ) -> None:
        """
        Accumulate CAS_all and CAS_fg statistics from one batch.

        Parameters
        ----------
        routing_probs : Tensor [B, N, K]   soft routing probabilities
        token_labels  : Tensor [B, N]      integer GT labels per token
        """
        B, N, K = routing_probs.shape
        assert K == self.num_concepts
        assert token_labels.shape == (B, N)

        probs  = routing_probs.detach().cpu()        # [B, N, K]
        labels = token_labels.detach().cpu().long()  # [B, N]

        flat_probs  = probs.reshape(-1, K)           # [B*N, K]
        flat_labels = labels.reshape(-1)             # [B*N]
        n_tokens    = float(flat_labels.numel())

        # Foreground mask: tokens with non-background GT label
        fg_mask = (flat_labels != self._BG_CLASS)    # [B*N] boolean
        n_fg    = float(fg_mask.sum().item())

        for k in range(K):
            p_k = flat_probs[:, k]                              # [B*N]
            m_k = (flat_labels == k).float()                    # [B*N]

            # CAS_all accumulation — all tokens
            self._n[k]      += n_tokens
            self._sum_p[k]  += p_k.sum().item()
            self._sum_m[k]  += m_k.sum().item()
            self._sum_pp[k] += (p_k * p_k).sum().item()
            self._sum_mm[k] += (m_k * m_k).sum().item()
            self._sum_pm[k] += (p_k * m_k).sum().item()

            # CAS_fg accumulation — foreground tokens, foreground concepts only.
            # Background concept (k=0) is excluded: its label variance is near-zero
            # (80%+ of tokens are background → M_0 ≈ 1 everywhere → unreliable Pearson).
            if k > self._BG_CLASS and n_fg > 0:
                p_k_fg = p_k[fg_mask]    # [n_fg]
                m_k_fg = m_k[fg_mask]    # [n_fg]

                self._n_fg[k]      += n_fg
                self._sum_p_fg[k]  += p_k_fg.sum().item()
                self._sum_m_fg[k]  += m_k_fg.sum().item()
                self._sum_pp_fg[k] += (p_k_fg * p_k_fg).sum().item()
                self._sum_mm_fg[k] += (m_k_fg * m_k_fg).sum().item()
                self._sum_pm_fg[k] += (p_k_fg * m_k_fg).sum().item()

    def _pearson_from_sums(
        self,
        n: float, sum_p: float, sum_m: float,
        sum_pp: float, sum_mm: float, sum_pm: float,
    ) -> float:
        """Compute Pearson correlation from running sums."""
        if n < 2:
            return 0.0
        numerator = n * sum_pm - sum_p * sum_m
        var_p = max(n * sum_pp - sum_p * sum_p, 0.0)
        var_m = max(n * sum_mm - sum_m * sum_m, 0.0)
        denominator = (var_p ** 0.5) * (var_m ** 0.5)
        if denominator < self._PEARSON_EPS:
            return 0.0
        return float(numerator / (denominator + self._PEARSON_EPS))

    def compute(self) -> Dict[str, float]:
        """
        Compute CAS_all(k) for each concept from accumulated statistics.

        All tokens contribute (including background).  Background CAS is
        unreliable due to low label variance and should not be cited as
        evidence of routing quality — report in supplementary only.

        Returns
        -------
        dict : concept_name → CAS_all value ∈ [-1, 1]
        """
        return {
            name: self._pearson_from_sums(
                self._n[k], self._sum_p[k], self._sum_m[k],
                self._sum_pp[k], self._sum_mm[k], self._sum_pm[k],
            )
            for k, name in enumerate(self.concept_names)
        }

    def compute_fg(self) -> Dict[str, float]:
        """
        Compute CAS_fg(k) for foreground concepts (k=1..K-1) over foreground tokens.

        This is the PRIMARY metric for the paper.

        CAS_fg answers: "Among non-background tokens, how well does the routing
        probability for concept k correlate with the ground-truth subregion mask?"
        This is the honest measure of CCR's clinical alignment claim.

        Background (k=0) is excluded from both the token set and the concept set:
          - Token exclusion: background tokens have near-zero label variance for
            foreground concepts → unreliable Pearson denominator.
          - Concept exclusion: CAS_fg(background) is undefined under this definition;
            background routing quality is implied when all tumor CAS_fg ≥ 0.85.

        Returns
        -------
        dict : concept_name → CAS_fg value ∈ [-1, 1]
               Only contains entries for k=1..K-1.
               key "background" is absent (use compute() for CAS_all(background)).
        """
        result: Dict[str, float] = {}
        for k in range(1, self.num_concepts):   # skip k=0 (background)
            name = self.concept_names[k]
            result[name] = self._pearson_from_sums(
                self._n_fg[k], self._sum_p_fg[k], self._sum_m_fg[k],
                self._sum_pp_fg[k], self._sum_mm_fg[k], self._sum_pm_fg[k],
            )
        return result

    def reset(self) -> None:
        """Clear all accumulated statistics."""
        self._reset_accumulators()


# ---------------------------------------------------------------------------
# Expert Utilization Tracker
# ---------------------------------------------------------------------------


class ExpertUtilizationTracker:
    """
    Tracks per-expert token utilization across training batches.

    Expert collapse detection (plan Section 16.1):
        If any expert receives < min_utilization_pct % of all tokens after
        the warmup phase ends (default threshold: 5 %), the model is flagged
        for expert re-initialisation of the collapsed experts.

    Parameters
    ----------
    num_concepts     : int
    concept_names    : Tuple[str, ...]
    warmup_end_epoch : int   epoch number after which collapse is a concern
    min_utilization_pct : float   collapse threshold (default 5.0 %)

    Usage
    -----
        tracker = ExpertUtilizationTracker(4, concept_names, warmup_end_epoch=10)
        for batch in train_loader:
            tracker.update(assignments)   # assignments: [B, N] integer
        stats   = tracker.compute()
        flagged = tracker.check_collapse(epoch=current_epoch)
        tracker.reset()
    """

    def __init__(
        self,
        num_concepts: int,
        concept_names: Tuple[str, ...],
        warmup_end_epoch: int = 10,
        min_utilization_pct: float = 5.0,
    ) -> None:
        self.num_concepts          = num_concepts
        self.concept_names         = concept_names
        self._warmup_end           = warmup_end_epoch
        self._min_pct              = min_utilization_pct
        self._counts: List[int]    = [0] * num_concepts
        self._total: int           = 0

    def update(self, assignments: torch.Tensor) -> None:
        """
        Accumulate hard routing assignment counts.

        Parameters
        ----------
        assignments : Tensor [B, N]   integer expert indices in [0, K)
        """
        flat = assignments.detach().cpu().reshape(-1)  # [B*N]
        self._total += flat.numel()
        for k in range(self.num_concepts):
            self._counts[k] += (flat == k).sum().item()

    def compute(self) -> Dict[str, float]:
        """
        Return per-expert utilization as a fraction of total tokens.

        Returns
        -------
        dict : concept_name → fraction ∈ [0, 1]
        """
        if self._total == 0:
            return {name: 0.0 for name in self.concept_names}
        return {
            name: self._counts[k] / self._total
            for k, name in enumerate(self.concept_names)
        }

    def check_collapse(self, epoch: int) -> List[str]:
        """
        Detect collapsed experts (< min_utilization_pct % of tokens).

        Only meaningful after warmup ends.  Emits a warning for each
        collapsed expert and returns their names.

        Parameters
        ----------
        epoch : int   current training epoch (1-indexed)

        Returns
        -------
        List[str]   names of collapsed experts (empty if none)
        """
        if epoch <= self._warmup_end:
            return []

        utilization = self.compute()
        threshold   = self._min_pct / 100.0
        collapsed   = [
            name
            for name, frac in utilization.items()
            if frac < threshold
        ]

        for name in collapsed:
            frac_pct = utilization[name] * 100.0
            warnings.warn(
                f"[CCR] Expert collapse detected at epoch {epoch}: "
                f"'{name}' receives only {frac_pct:.2f}% of tokens "
                f"(threshold: {self._min_pct:.1f}%).  "
                f"Consider re-initialising this expert's weights.",
                RuntimeWarning,
                stacklevel=2,
            )

        return collapsed

    def reset(self) -> None:
        """Clear all accumulated counts."""
        self._counts = [0] * self.num_concepts
        self._total  = 0
