"""
reduce — compute evaluation metrics from stored measures, without touching any model.

Stability is computed on subnet_vectors_test (test fold); concurvity on
subnet_vectors_pool (training pool). Both fields store RAW uncentered outputs —
centering happens here, per metric, over the correct evaluation set.
"""

import numpy as np


def main_effect_instability(measures, arm: str) -> float:
    """Across-seed instability of the main effects (headline metric).

    For each main term, center each seed's test-fold output vector by its own mean,
    compute the pointwise across-seed standard deviation, then average over test
    points and over all main terms. Lower is more stable.

    Args:
        measures: Stored measures keyed by (arm, fold, seed).
        arm: 'A', 'B', or 'C'.

    Returns:
        Scalar instability. Headline comparison: C vs B.
    """
    raise NotImplementedError


def interaction_instability(measures, arm: str) -> float:
    """Across-seed instability of the interaction terms (caveated).

    Same computation as main_effect_instability but over ("inter", j, k) terms.
    Only pairs present in ALL seeds being compared are included (intersection).

    Note: interaction instability conflates genuine shape disagreement with
    identifiable additive shifts between a term and its parent main effects.
    Report alongside that caveat.

    Args:
        measures: Stored measures keyed by (arm, fold, seed).
        arm: 'B' or 'C'.

    Returns:
        Scalar instability over the shared interaction terms.
    """
    raise NotImplementedError


def bootstrap_gap(measures, arm_c: str, arm_baseline: str, n_boot: int = 2000, *, metric=main_effect_instability) -> tuple[float, float]:
    """Bootstrap CI for the instability gap (arm_c − arm_baseline) by resampling seeds.

    Args:
        measures: Stored measures keyed by (arm, fold, seed).
        arm_c: Treatment arm ('C').
        arm_baseline: Baseline arm ('B').
        n_boot: Number of bootstrap replicates.
        metric: Instability function to bootstrap.

    Returns:
        (lo, hi): 2.5 / 97.5 percentile interval of the gap.
    """
    raise NotImplementedError


def concurvity_summary(measures, arm: str) -> dict:
    """Post-hoc concurvity of each retained interaction against all other fitted components.

    For each active pair (j, k): adj-R² of its pool output vector regressed on all
    main effect vectors + all other active interaction vectors. Measured on the training
    pool using the same formula as the Stage-2 gate (na2m.utils.concurvity), but applied
    after fine-tuning — so arm C is not guaranteed to stay below the gate threshold.

    Args:
        measures: Stored measures keyed by (arm, fold, seed).
        arm: 'B' or 'C' (arm A has no interactions).

    Returns:
        dict with at least {"max": ..., "mean": ...}, averaged over seeds and folds.
    """
    raise NotImplementedError


def accuracy_summary(measures) -> dict:
    """Single-model and ensemble accuracy per arm, plus mean retained interaction count.

    Args:
        measures: Stored measures keyed by (arm, fold, seed).

    Returns:
        dict per arm with single-model accuracy (mean over seeds then folds), ensemble
        accuracy (average logits within fold then score), and mean n_terms.
    """
    raise NotImplementedError
