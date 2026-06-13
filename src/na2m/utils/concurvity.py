"""
concurvity — the ONE adjusted-R² concurvity formula, shared by gate and diagnostic.

Centering is handled by the OLS intercept (fit WITH intercept ⇔
center every column over THIS sample), so callers can pass raw vectors.
"""

import numpy as np

TermId = tuple[str, int] | tuple[str, int, int]  # ("main", j) or ("inter", j, k)


def concurvity_adjr2(target_vec: np.ndarray, basis_vecs: np.ndarray) -> float:
    """OLS of target_vec on basis_vecs. Computes adjusted r-squared, which gets used as multivariate concurvity metric. 

    Args:
        target_vec (np.ndarray): Output vector of the term under test
        basis_vecs (np.ndarray): Output vector of other terms to check concurvity against. 

    Returns:
        float: Adjusted R-squared. 
        0.0 if basic_vecs is empty. 
    """

    # OLS with intercept
    y = target_vec.flatten()
    X = basis_vecs
    N, p = X.shape

    if p == 0:
        return 0.0

    X_design = np.column_stack([np.ones(N), X])
    beta, _, _, _ = np.linalg.lstsq(X_design, y, rcond=None)
    y_hat = X_design @ beta

    ss_res = ((y - y_hat) ** 2).sum()
    ss_tot = ((y - y.mean()) ** 2).sum()

    if ss_tot == 0.0:
        return 0.0

    r2 = 1.0 - ss_res / ss_tot
    adj_r2 = 1.0 - (1.0 - r2) * (N - 1) / (N - p - 1)
    return float(adj_r2)


def concurvity_score(term_id: TermId, term_vectors: dict[TermId, np.ndarray]) -> float:
    """Concurvity of one term against all other terms.

    Convenience wrapper for the post-hoc diagnostic in the reducer, which
    works from a {term_id: output_vector} stored dict. The Stage-2 gate calls
    concurvity_adjr2 directly since it already has the vectors split out in memory.

    Args:
        term_id: Key of the term under test.
        term_vectors: All subnet vecotrs, Dict mapping every term_id to its (N,) output vector
            (raw, evaluated on the same sample for all terms).

    Returns:
        Adjusted R² of term_id regressed on all other terms.
    """
    target = term_vectors[term_id]
    other_vecs = [v for t, v in term_vectors.items() if t != term_id]
    N = target.flatten().shape[0]
    basis = np.column_stack(other_vecs) if other_vecs else np.empty((N, 0))
    return concurvity_adjr2(target, basis)