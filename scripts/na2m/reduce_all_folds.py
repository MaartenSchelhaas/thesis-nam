"""
reduce_all_folds.py — cross-fold aggregate of all reducer metrics.
Run from repo root: python -m scripts.na2m.reduce_all_folds

Each section reports the cross-fold mean ± std of per-fold scalars.
Per-fold detail (individual pair tables, shape plots) lives in reduce_fold0.py.

"""
from pathlib import Path

import numpy as np

from na2m.utils.config import load_na2m_config
from na2m.eval.reduce import (
    accuracy_summary,
    concurvity_summary,
    load_fold,
    main_effect_instability,
    mean_pairwise_jaccard,
    selection_frequencies,
)

ALL_ARMS   = ("mains", "gaminet", "concurvity", "regularized")
INTER_ARMS = ("gaminet", "concurvity", "regularized")


def _section(title: str) -> None:
    print(f"\n{'─' * 60}")
    print(f"  {title}")
    print(f"{'─' * 60}")


def _format_mean_std(mean, std) -> str:
    """Format a mean ± std pair as a fixed-width string.

    Args:
        mean: Cross-fold mean (accepts float or numpy scalar).
        std: Cross-fold standard deviation (accepts float or numpy scalar).

    Returns:
        String of the form "X.XXXX ± X.XXXX".
    """
    return f"{float(mean):.4f} ± {float(std):.4f}"


def _print_instability(fold_measures_list: list) -> None:
    """Print cross-fold mean ± std of main-effect instability per arm.

    Args:
        fold_measures_list: One FoldMeasures dict per fold, from load_fold.
    """
    _section("Main-effect instability  [cross-fold mean ± std, lower = more stable]")
    for arm in ALL_ARMS:
        vals = [
            main_effect_instability(fold, arm)
            for fold in fold_measures_list
            if fold.get(arm)
        ]
        if not vals:
            continue
        print(f"  {arm:12s}  {_format_mean_std(np.mean(vals), np.std(vals, ddof=1))}  (n_folds={len(vals)})")


def _print_selection(fold_measures_list: list) -> None:
    """Print cross-fold mean ± std of interaction selection metrics per arm.

    Reports mean pairwise Jaccard, number of unique pairs ever selected, and
    mean number of pairs selected per run. Individual pair frequency tables
    are omitted at this aggregation level.

    Args:
        fold_measures_list: One FoldMeasures dict per fold, from load_fold.
    """
    _section("Interaction selection  [cross-fold mean ± std]")
    for arm in INTER_ARMS:
        jaccards  = []
        n_uniques = []
        n_pairs   = []
        for fold in fold_measures_list:
            runs = fold.get(arm)
            if not runs:
                continue
            jaccards.append(mean_pairwise_jaccard(fold, arm))
            n_uniques.append(len(selection_frequencies(fold, arm)))
            n_pairs.append(float(np.mean([len(r["pairs"]) for r in runs])))

        if not jaccards:
            continue
        print(
            f"  {arm:12s}"
            f"  jaccard={_format_mean_std(np.mean(jaccards), np.std(jaccards, ddof=1))}"
            f"  n_unique_pairs={np.mean(n_uniques):.1f} ± {np.std(n_uniques, ddof=1):.1f}"
            f"  mean_n_pairs={np.mean(n_pairs):.1f} ± {np.std(n_pairs, ddof=1):.1f}"
        )


def _print_concurvity(fold_measures_list: list) -> None:
    """Print cross-fold mean ± std of post-hoc concurvity summary per arm.

    The per-fold concurvity_summary already aggregates 20 runs into a scalar.
    Here those fold-level scalars are averaged across folds. The within-fold
    mean_std field is discarded — fold-to-fold variance is the relevant estimator.

    Args:
        fold_measures_list: One FoldMeasures dict per fold, from load_fold.
    """
    _section("Post-hoc concurvity  [cross-fold mean ± std of per-fold summary]")
    for arm in ALL_ARMS:
        summaries = [
            concurvity_summary(fold, arm)
            for fold in fold_measures_list
            if fold.get(arm)
        ]
        if not summaries:
            continue
        for part in ("mains", "interactions", "total"):
            means  = [s[part]["mean"]         for s in summaries if not np.isnan(s[part]["mean"])]
            fracs  = [s[part]["frac_concurve"] for s in summaries if not np.isnan(s[part]["frac_concurve"])]
            maxes  = [s[part]["max"]           for s in summaries if not np.isnan(s[part]["max"])]
            if not means:
                continue
            ddof = 1 if len(means) > 1 else 0
            print(
                f"  {arm:12s}  [{part:12s}]"
                f"  mean={_format_mean_std(np.mean(means), np.std(means, ddof=ddof))}"
                f"  frac={_format_mean_std(np.mean(fracs), np.std(fracs, ddof=ddof))}"
                f"  max={_format_mean_std(np.mean(maxes), np.std(maxes, ddof=ddof))}"
            )


def _print_performance(fold_measures_list: list, task: str, metric_name: str) -> None:
    """Print cross-fold mean ± std of predictive performance per arm.

    Single-run and ensemble metrics are averaged across folds. The within-fold
    single_std (run-to-run variance) is dropped; the ± here is the cross-fold std.

    Args:
        fold_measures_list: One FoldMeasures dict per fold, from load_fold.
        task: "classification" or "regression" — passed to accuracy_summary.
        metric_name: Display label, e.g. "auroc" or "rmse".
    """
    _section(f"Predictive performance  [{metric_name}]  [cross-fold mean ± std]")
    per_fold_acc = [accuracy_summary(fold, task=task) for fold in fold_measures_list]
    for arm in ALL_ARMS:
        singles   = [acc[arm]["single"]      for acc in per_fold_acc if arm in acc]
        ensembles = [acc[arm]["ensemble"]     for acc in per_fold_acc if arm in acc]
        n_pairs   = [acc[arm]["mean_n_pairs"] for acc in per_fold_acc if arm in acc]
        if not singles:
            continue
        ddof = 1 if len(singles) > 1 else 0
        print(
            f"  {arm:12s}"
            f"  single={_format_mean_std(np.mean(singles), np.std(singles, ddof=ddof))}"
            f"  ensemble={_format_mean_std(np.mean(ensembles), np.std(ensembles, ddof=ddof))}"
            f"  mean_n_pairs={np.mean(n_pairs):.1f} ± {np.std(n_pairs, ddof=ddof):.1f}"
        )


def main() -> None:
    # ── Edit these before running ────────────────────────────────────────────
    BASE_DIR = Path("runs/compas_na2m")
    RUN_MODE = "fixed"
    N_RUNS   = 20
    N_FOLDS  = 5
    # ────────────────────────────────────────────────────────────────────────

    fold_measures_list = [
        load_fold(BASE_DIR / f"fold_{i}", run_mode=RUN_MODE, n_runs=N_RUNS)
        for i in range(N_FOLDS)
    ]

    task = load_na2m_config(str(BASE_DIR / "fold_0" / "mains_tuned_config.yaml")).task
    metric_name = "auroc" if task == "classification" else "rmse"

    print(f"\nAggregated over {N_FOLDS} folds, {N_RUNS} runs each, run_mode={RUN_MODE!r}")

    _print_instability(fold_measures_list)
    _print_selection(fold_measures_list)
    _print_concurvity(fold_measures_list)
    _print_performance(fold_measures_list, task, metric_name)

    print()


if __name__ == "__main__":
    main()
