"""
selection policy — swappable concurvity gate for the Stage-2 prune sweep.

Defines the SelectionPolicy protocol and two concrete implementations:

    NoGate          — always accepts; used for arm B.
    ConcurvityGate  — adj-R² gate; used for arm C.

The protocol is the only thing stage2_select knows about. To swap in a
different strategy (e.g. post-finetune iterative removal), implement
SelectionPolicy and pass it to stage2_select — no other file changes needed.
"""

from __future__ import annotations
from typing import Protocol
import numpy as np


class SelectionPolicy(Protocol):
    """Protocol for Stage-2 candidate acceptance decisions.

    stage2_select calls should_accept once per candidate, in decreasing
    contribution order. The policy may accumulate state across calls (e.g.
    the growing accepted-pair basis for the concurvity gate).
    """

    def should_accept(
        self,
        candidate_vec: np.ndarray,
        accepted_vecs: list[np.ndarray],
        main_vecs: list[np.ndarray],
    ) -> bool:
        """Return True if this candidate should be added to the accepted set.

        Args:
            candidate_vec: (N,) block-trained output of this pair on the pool.
            accepted_vecs: output vectors of already-accepted interactions,
                in acceptance order. Empty on the first call.
            main_vecs: output vectors of all main effects on the pool.
                Fixed for the duration of the sweep.

        Returns:
            True → accept (append to accepted set and record val loss).
            False → skip (excluded from survivors, never reconsidered).
        """
        ...


class NoGate:
    """Always accepts every candidate. Used for arm B (no concurvity filter)."""

    def should_accept(
        self,
        candidate_vec: np.ndarray,
        accepted_vecs: list[np.ndarray],
        main_vecs: list[np.ndarray],
    ) -> bool:
        return True


class ConcurvityGate:
    """Adj-R² concurvity gate. Skips a candidate if its output is too linearly
    redundant with the mains and the already-accepted interactions.

    Used for arm C (with_concurvity_filter=True).
    """

    def __init__(self, threshold: float) -> None:
        """
        Args:
            threshold: Maximum allowed adj-R² concurvity score. Candidates
                above this are skipped. Sourced from hp.concurvity_threshold.
        """
        self.threshold = threshold

    def should_accept(
        self,
        candidate_vec: np.ndarray,
        accepted_vecs: list[np.ndarray],
        main_vecs: list[np.ndarray],
    ) -> bool:
        """Return False if candidate_vec is too redundant with the current basis.

        Basis = column-stack of main_vecs + accepted_vecs (the GROWING accepted
        set, NOT the full candidate set). Calls concurvity_adjr2 from
        na2m.utils.concurvity — the single shared formula.

        """
        from na2m.utils.concurvity import concurvity_adjr2

        all_basis = main_vecs + accepted_vecs
        basis = np.column_stack(all_basis) if all_basis else np.empty((len(candidate_vec), 0))
        score = concurvity_adjr2(candidate_vec, basis)
        return score <= self.threshold