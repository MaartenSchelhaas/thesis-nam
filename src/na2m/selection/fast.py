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

    TODO:
        - Try importing the interpret native EBM FAST entry point.
        - Optionally run a tiny smoke score to confirm the compiled lib loads.
        - Return False (do not raise) on any ImportError / OSError.
    """
    raise NotImplementedError


def fast_screen(main_model, X, y, task: str) -> list[tuple[int, int]]:
    """Rank candidate interaction pairs via the native FAST scorer.

    Args:
        main_model: The trained main-only NA2M (for residual-equivalent preds).
        X: Feature matrix to score on (train/val of the 80% pool).
        y: Targets.
        task: 'classification' (logit-transform residuals) or 'regression'.

    Returns:
        Ranked list of (j, k) pairs, best first.

    TODO:
        - Compute residual-equivalent predictions from main_model (logit for clf).
        - Call the interpret FAST scorer (get_interaction_list /
          NativeEBM.fast_interaction_score) on (X, residuals).
        - Return the ranked (j, k) list.
    """
    raise NotImplementedError


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
