"""
FAST interaction screening — thin wrapper around the EBM / `interpret` FAST scorer.

GAMI-Net uses the FAST heuristic (Lou et al.) to rank candidate interaction
pairs cheaply before any interaction subnet is trained. This wraps the native
scorer from the `interpret` package (NativeEBM.fast_interaction_score /
get_interaction_list) the GAMI-Net reference relies on.

Input: the trained MAIN model's residual-equivalent predictions on train/val
(logit-transform for classification). Output: a ranked list of (j, k) pairs.

NOTE: the native scorer ships a compiled lib (.so/.dll) that may fail to install.
verify_fast_available() probes it first; a pure-Python fallback stub is provided
so the pipeline can degrade gracefully.

Heavy imports (`interpret`) are done INSIDE functions so this module imports
cleanly even when the package is absent.
"""
def verify_fast_available() -> bool:
    """Check that the native FAST scorer imports and runs.

    Returns:
        True if the native `interpret` FAST path is usable, False otherwise.
    """
    try:
        from interpret.utils._measure_interactions import measure_interactions  # noqa: F401
        return True
    except Exception:
        return False


def fast_screen(main_model, X, y, task: str) -> list[tuple[int, int]]:
    """Rank candidate interaction pairs via the native FAST scorer.

    Args:
        main_model: The trained main-only NA2M (for residual-equivalent preds).
        X: Feature matrix to score on (train/val of the 80% pool).
        y: Targets.
        task: 'classification' (logit-transform residuals) or 'regression'.

    Returns:
        Ranked list of (j, k) pairs, best first.
    """
    import numpy as np
    import torch
    from interpret.utils._measure_interactions import measure_interactions

    if not isinstance(X, np.ndarray):
        X = np.asarray(X)
    if not isinstance(y, np.ndarray):
        y = np.asarray(y)

    main_model.eval()
    with torch.no_grad():
        logits, _ = main_model(torch.as_tensor(X, dtype=torch.float32))
        init_score = logits.squeeze(-1).cpu().numpy()

    objective = "log_loss" if task == "classification" else "rmse"
    ranked = measure_interactions(X, y, init_score=init_score, objective=objective)
    return [pair for pair, _ in ranked]


def fast_screen_fallback(main_model, X, y, task: str) -> list[tuple[int, int]]:
    """Pure-Python FAST fallback when the native lib is unavailable.

    Fits a shallow model per candidate pair on the main-model residuals and
    scores pairs by goodness-of-fit.

    Args:
        main_model: The trained main-only NA2M.
        X: Feature matrix.
        y: Targets.
        task: 'classification' or 'regression'.

    Returns:
        Ranked list of (j, k) pairs, best first.

    TODO (STUB — implement only if verify_fast_available() returns False):
        - Compute residuals from main_model.
        - For each (j, k): fit a shallow model on columns (j, k) → residuals.
        - Score by fit (e.g. residual variance explained); rank descending.
    """
    raise NotImplementedError
