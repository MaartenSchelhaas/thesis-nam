"""
extract — pull the durable MEASURES out of a live NA2M before discarding it.

Per (arm, fold, seed) the harness trains a model, extracts measures in-process,
writes them to disk, and lets the model go out of scope. We persist the
MEASURES, never the model parameters.

HARD CONSTRAINTS:
    - PRECONDITION: model is in eval mode AND best weights have been restored.
    - Store RAW term outputs — never pre-centered. Centering happens per-metric
      in the reducer.
    - Key everything by term_id, never positionally.
    - Concurvity uses interaction VECTORS over the 80% pool (extracted here);
      interaction grid curves are optional (off by default).
"""


def extract_measures(model, X_pool, X_test, grids, feature_meta, *, include_inter_curves: bool = False) -> dict:
    """Extract the durable measures from a trained NA2M.

    Args:
        model: Trained NA2M, eval mode, best weights restored.
        X_pool: The fold's 80% train pool (term vectors + concurvity are evaluated here).
        X_test: The fold's test slice (logits / accuracy only).
        grids: Per-feature evaluation grids (from make_grid), keyed by feature index.
        feature_meta: Per-feature metadata (type/levels).
        include_inter_curves: If True, also extract interaction shape curves
            (off by default — headline is main-effect stability).

    Returns:
        dict with:
            "curves": term_id -> raw output on the grid. Main terms always;
                      interaction curves only if include_inter_curves. Numerical
                      → (G,), categorical → (n_levels,).
            "term_vectors": term_id -> raw output on the FULL X_pool, for ALL terms
                      (mains + active interactions). RAW (not centered).
            "logits": model.predict(X_test) → (N_test,).
            "pairs": model.active_pairs().

    TODO:
        - assert model is in eval mode and (document) best weights restored.
        - Under torch.no_grad():
            * curves: evaluate each main term on its grid via iter_terms (and
              interaction terms only if include_inter_curves).
            * term_vectors: evaluate every term on X_pool, store RAW.
            * logits: model.predict(X_test).
            * pairs: model.active_pairs().
        - Key all dicts by term_id; convert tensors to numpy for persistence.
    """
    raise NotImplementedError
