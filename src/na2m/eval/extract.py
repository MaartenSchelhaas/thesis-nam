"""
extract — pull the durable MEASURES out of a live NA2M before discarding it.

Per (arm, fold, seed) the harness trains a model, extracts measures in-process,
writes them to disk, and lets the model go out of scope. We persist the
MEASURES, never the model parameters.

HARD CONSTRAINTS:
    - PRECONDITION: model is in eval mode AND best weights have been restored.
    - Store RAW term outputs — never pre-centered. Centering happens per-metric
      in the reducer (over whichever evaluation set that metric uses).
    - Key everything by term_id, never positionally.
    - Two distinct evaluation sets, stored separately and NOT interchangeable:
        * term_vectors_pool : ALL terms on X_pool  -> consumed by CONCURVITY
          (Kovács observed-concurvity index is a property of the fit on the
           training data the model saw).
        * term_vectors_test : ALL terms on X_test  -> consumed by STABILITY
          (cross-seed comparison on a shared, held-out set drawn from the data
           distribution; density is intrinsic to the sample, no grid, no weights).
"""


def extract_measures(model, X_pool, X_test, grids, feature_meta, *, include_inter_curves: bool = False) -> dict:
    """Extract the durable measures from a trained NA2M.

    Args:
        model: Trained NA2M, eval mode, best weights restored.
        X_pool: The fold's 80% train pool. Concurvity term vectors are evaluated here.
        X_test: The fold's test slice. Stability term vectors AND logits are evaluated here.
        grids: Per-feature evaluation grids (from make_grid), keyed by feature index.
               Used ONLY for plotting curves, never for the stability metric.
        feature_meta: Per-feature metadata (type/levels).
        include_inter_curves: If True, also extract interaction shape curves
            (off by default — plotting aid only).

    Returns:
        dict with:
            "curves": term_id -> raw output on the grid (PLOTTING ONLY). Main terms
                      always; interaction curves only if include_inter_curves.
                      Numerical → (G,), categorical → (n_levels,).
            "term_vectors_pool": term_id -> raw output on FULL X_pool, ALL terms
                      (mains + active interactions). RAW. For CONCURVITY.
            "term_vectors_test": term_id -> raw output on FULL X_test, ALL terms
                      (mains + active interactions). RAW. For STABILITY.
            "logits": model.predict(X_test) → (N_test,).
            "pairs": model.active_pairs().

    TODO:
        - assert model.training is False; (document) best weights already restored.
        - Under torch.no_grad():
            * curves (plotting): evaluate each main term on its grid via iter_terms
              (and interaction terms only if include_inter_curves).
            * term_vectors_pool: evaluate every term (iter_terms) on X_pool, store RAW.
            * term_vectors_test: evaluate every term (iter_terms) on X_test, store RAW.
              SAME term set and SAME term_id keys as the pool dict — both cover all
              mains + all active interactions, so concurvity and stability index the
              same terms.
            * logits: model.predict(X_test).
            * pairs: model.active_pairs().
        - iter_terms returns CENTERED outputs; store the RAW per-term values instead
          (raw_main / raw_inter, before the center subtraction) so the reducer can
          re-center per metric over the correct evaluation set. Do NOT store the
          model's deployment centering here.
        - Key all dicts by term_id; convert tensors to numpy for persistence.
    """
    raise NotImplementedError