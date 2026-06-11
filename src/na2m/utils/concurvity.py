"""
concurvity — the ONE adjusted-R² concurvity formula, shared by gate and diagnostic.

A single source of truth so the Stage-2 selection
gate and the post-hoc headline diagnostic can never drift apart:

    * Stage-2 gate (selection-time, arm C) — fit_na2m.stage2_select scores a
      candidate's BLOCK-TRAINED output against {mains + already-accepted
      interactions} on the 80% pool; skips the candidate if the score > τ.
    * Headline diagnostic (deployment-time) — reduce.concurvity_summary scores
      each retained interaction of the FINE-TUNED model against all OTHER fitted
      components on the 80% pool. Same formula; NOT re-gated, so for arm C it may
      exceed τ because fine-tuning moved the geometry.

Pure numpy, no torch and no model handle — callers pass already-evaluated RAW
output vectors. Centering is handled by the OLS intercept (fit WITH intercept ⇔
center every column over THIS sample), so callers pass raw vectors.
"""

import numpy as np


def concurvity_adjr2(target_vec: np.ndarray, basis_vecs: np.ndarray) -> float:
    """Adjusted R² of one term's output regressed on a basis of other terms.

    OLS of `target_vec` on `basis_vecs` WITH an intercept; return adjusted R².
    Equivalent to the observed-concurvity index: how much of this term is
    redundant with (linearly explainable by) the other fitted components.

    Args:
        target_vec: (N,) or (N, 1) RAW output vector of the term under test.
        basis_vecs: (N, p) matrix whose columns are the RAW output vectors of the
            regressor terms (mains + the relevant other interactions). p columns,
            recomputed by the caller each call since the active set varies.

    Returns:
        Adjusted R² in (-inf, 1]. By convention return 0.0 when basis_vecs has no
        columns (p == 0): a term with no competitors has zero concurvity.

    TODO:
        - Flatten target to (N,); design = [1 | basis_vecs] (intercept column).
        - Solve least squares (np.linalg.lstsq); residual SS / total SS → R².
        - adj = 1 - (1 - R²) * (N - 1) / (N - p - 1); guard N - p - 1 <= 0.
        - p == 0 → return 0.0.
    """
    raise NotImplementedError


def concurvity_score(term_id, term_vectors: dict, *, exclude_self: bool = True) -> float:
    """Concurvity of one term vs ALL other terms in a {term_id: vector} dict.

    Thin convenience wrapping concurvity_adjr2: assemble the basis from every
    OTHER term's vector (self excluded) and score. Used by the headline
    diagnostic; the Stage-2 gate builds its (growing, accepted-only) basis
    directly and calls concurvity_adjr2.

    Args:
        term_id: the term under test (key into term_vectors).
        term_vectors: {term_id: (N,) or (N,1) RAW vector}, all on the SAME sample
            (the 80% pool for the diagnostic).
        exclude_self: if True (default) drop term_id from the basis.

    Returns:
        Adjusted R² of term_id against the other terms.

    TODO:
        - basis = column-stack of term_vectors[t] for t != term_id (if exclude_self).
        - return concurvity_adjr2(term_vectors[term_id], basis).
    """
    raise NotImplementedError