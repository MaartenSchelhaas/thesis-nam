"""
reduce — cheap, re-runnable, model-agnostic metric computation from stored measures.

Reads the measure files written by run.py (never touches a live model) and
computes the headline evaluation metrics. Because measures are stored RAW,
centering happens HERE, per metric, over that metric's own evaluation set.

HARD CONSTRAINTS:
    - STABILITY is computed on the TEST fold (term_vectors_test); CONCURVITY on
      the 80% pool (term_vectors_pool). These use different stored fields and are
      NOT interchangeable.
    - Center each vector by its OWN mean over the metric's evaluation set before
      use (test-fold mean for stability, pool mean for concurvity), since the
      per-subnet mean is not identifiable.
    - No density weighting: stability is averaged over actual test datapoints, so
      data density is intrinsic to the sample.
    - Concurvity regressors are ALL OTHER fitted components (mains + other active
      interactions, self excluded) — matches the stage-4 pruning criterion exactly.
    - main_effect_instability is the CLEAN headline (mains only, shared across
      arms). interaction_instability is reported separately and carries an
      identifiability caveat (see its docstring).
"""


def main_effect_instability(measures, arm: str) -> float:
    """Across-seed instability of the MAIN effects for one arm (headline metric).

    Per main term: take each seed's RAW test-fold f-vector (term_vectors_test),
    center it by its own test-fold mean, take the pointwise across-seed SD,
    average over the test points. Pool over the K MAIN terms only — these are
    shared across all arms, so this number is directly comparable across arms.

    Mains-only is INTENTIONAL: main effects have no additive marginals to slosh,
    so centered cross-seed disagreement is clean shape instability. Interactions
    need the extra caveat handled in interaction_instability.

    Args:
        measures: Stored measures keyed by (arm, fold, seed).
        arm: Arm to compute for ('A', 'B', 'C', or 'Bprime').

    Returns:
        Scalar instability (lower = more stable). Compare C vs B and C vs B′.

    TODO:
        - Read term_vectors_test (NOT pool). For each ("main", k):
          stack seed vectors; center each by its own test-fold mean;
          pointwise across-seed SD; average over test points.
        - Average over the K main terms (and folds as specified).
    """
    raise NotImplementedError


def interaction_instability(measures, arm: str) -> float:
    """Across-seed instability of the INTERACTION terms for one arm (caveated).

    Same computation as main_effect_instability but over ("inter", j, k) terms
    from term_vectors_test, centered by their own test-fold means.

    CAVEAT (state alongside any reported value): a centered interaction surface
    still carries additive marginal components that can shift between the
    interaction and the main effects across seeds without changing the total
    function. This metric therefore conflates genuine shape instability with that
    identifiability reshuffling. Isolating the pure interaction would require a
    functional-ANOVA purification of the surfaces before comparison, left to
    future work. Read these numbers with that mixing in mind; they are NOT as
    clean as the main-effect headline.

    Active pairs differ across seeds/arms, so define the term set explicitly:
    compute instability only over pairs PRESENT in all seeds being compared
    (intersection), or report per-pair and aggregate — decide and document, do
    not silently average over a varying term set.

    Args:
        measures: Stored measures keyed by (arm, fold, seed).
        arm: Arm to compute for ('B', 'C', 'Bprime'). Skip 'A' (no interactions).

    Returns:
        Scalar instability over interaction terms (lower = more stable).

    TODO:
        - Read term_vectors_test. Select ("inter", j, k) terms.
        - Resolve the varying-active-set issue above (intersection of pairs across
          the seeds being compared is the simplest defensible choice).
        - Center each by its own test-fold mean; pointwise across-seed SD;
          average over test points; aggregate over the chosen pair set.
    """
    raise NotImplementedError


def bootstrap_gap(measures, arm_c: str, arm_baseline: str, n_boot: int = 2000, *, metric=main_effect_instability) -> tuple[float, float]:
    """Bootstrap CI for the instability gap (C − baseline) by resampling seeds.

    Args:
        measures: Stored measures keyed by (arm, fold, seed).
        arm_c: The treatment arm ('C').
        arm_baseline: The baseline arm ('B' or 'Bprime').
        n_boot: Number of bootstrap replicates.
        metric: Which instability function to bootstrap (default: the main-effect
            headline). Pass interaction_instability to bootstrap that gap instead.

    Returns:
        (lo, hi): 2.5 / 97.5 percentiles of the bootstrapped gap.

    TODO:
        - Resample SEEDS with replacement (same resampled indices within each fold).
        - Per replicate: recompute metric(resampled, arm_c) − metric(resampled, arm_baseline).
        - Return the 2.5 / 97.5 percentiles.
    """
    raise NotImplementedError


def concurvity_summary(measures, arm: str) -> dict:
    """Concurvity of each interaction term against all OTHER fitted components.

    Per interaction term (j,k): adj-R²(term_vec ~ all OTHER fitted components)
    on the 80% POOL (term_vectors_pool), where "all other" = every main effect +
    every other active interaction (self excluded). Matches eq. (concurvity-filter)
    and the stage-4 pruning criterion exactly. Report max & mean over active
    pairs, averaged over seeds & folds. Skip arm A (no interactions).

    Pool, not test fold: the observed-concurvity index of Kovács is a property of
    the fit on the training data, so it is measured where the model was fit.

    Args:
        measures: Stored measures keyed by (arm, fold, seed).
        arm: Arm to compute for ('B', 'C', 'Bprime').

    Returns:
        dict with at least {"max": ..., "mean": ...}.

    TODO:
        - Read term_vectors_pool (NOT test). For each interaction (j,k):
          regressors = all main term vectors + all other active interaction
          vectors (exclude (j,k) itself).
        - adj-R² fitted WITH an intercept (equivalent to pre-centering);
          p = K + (n_active_pairs - 1), recomputed each time (pairs differ
          across seeds/folds).
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