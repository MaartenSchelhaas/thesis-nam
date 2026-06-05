"""
reduce — cheap, re-runnable, model-agnostic metric computation from stored measures.

Reads the measure files written by run.py (never touches a live model) and
computes the headline evaluation metrics. Because measures are stored RAW,
centering happens HERE, per metric.

HARD CONSTRAINTS:
    - Center by the data-distribution mean over the 80% pool, per metric.
    - Stability / concurvity are computed on the 80% pool, NOT the test fold.
    - Concurvity regressors are the MAIN-effect vectors, not all terms.
    - Pool main-effect instability over the K MAIN terms only (shared across arms).
"""


def main_effect_instability(measures, arm: str) -> float:
    """Across-seed instability of the main effects for one arm.

    Per main term: center each seed's f-vector by its own data-mean (over the
    80% pool), take the pointwise across-seed SD, average over points. Pool over
    the K MAIN terms only.

    Args:
        measures: Stored measures keyed by (arm, fold, seed).
        arm: Arm to compute for ('A', 'B', 'C', or 'Bprime').

    Returns:
        Scalar instability (lower = more stable). Compare C vs B and C vs B′.

    TODO:
        - For each main term: stack seed f-vectors; center each by its 80%-pool mean;
          pointwise across-seed SD; average over points.
        - Average over the K main terms (and folds as specified).
    """
    raise NotImplementedError


def bootstrap_gap(measures, arm_c: str, arm_baseline: str, n_boot: int = 2000) -> tuple[float, float]:
    """Bootstrap CI for the instability gap (C − baseline) by resampling seeds.

    Args:
        measures: Stored measures keyed by (arm, fold, seed).
        arm_c: The treatment arm ('C').
        arm_baseline: The baseline arm ('B' or 'Bprime').
        n_boot: Number of bootstrap replicates.

    Returns:
        (lo, hi): 2.5 / 97.5 percentiles of the bootstrapped gap.

    TODO:
        - Resample SEEDS with replacement (same indices within each fold).
        - Recompute the C − baseline instability gap per replicate.
        - Return the 2.5 / 97.5 percentiles.
    """
    raise NotImplementedError


def concurvity_summary(measures, arm: str) -> dict:
    """Concurvity of interaction terms against the main-effect vectors.

    Per interaction term: adj-R²(term_vec ~ all MAIN-term vectors) on the 80%
    pool. Report max & mean over active pairs, averaged over seeds & folds.
    Skip arm A (no interactions).

    Args:
        measures: Stored measures keyed by (arm, fold, seed).
        arm: Arm to compute for ('B', 'C', 'Bprime').

    Returns:
        dict with at least {"max": ..., "mean": ...}.

    TODO:
        - Regressors are the MAIN-effect vectors only (not all terms).
        - Per interaction: adj-R² of its vector on the main-vector design matrix.
        - Aggregate max & mean over pairs, then average over seeds & folds.
    """
    raise NotImplementedError


def accuracy_summary(measures) -> dict:
    """Single-model and ensemble accuracy per arm, plus fine-tune pass counts.

    Args:
        measures: Stored measures keyed by (arm, fold, seed), including logits,
            test labels, and fine_tune_pass_count.

    Returns:
        dict per arm with single-model accuracy (mean over seeds, then folds),
        an ensemble reference (mean logits over seeds), and fine_tune_pass_count.

    TODO:
        - Single-model: metric per (fold, seed) from logits + test labels;
          mean over seeds, then folds.
        - Ensemble: average logits over seeds within a fold, then metric.
        - Report fine_tune_pass_count per arm (for non-inferiority context).
    """
    raise NotImplementedError
