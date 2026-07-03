"""
reduce — compute evaluation metrics from stored measures, without touching any model.

All metrics are pure functions of measures.pt files. Load one fold with load_fold,
pass the result to any metric function. Centering happens here per-metric — never
in the extractor. To aggregate across folds, call load_fold once per fold and
average the results yourself in the calling script.
"""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import numpy as np
import torch

from na2m.utils.concurvity import concurvity_score

# {arm_name: [run_0_dict, run_1_dict, ...]} — one entry per completed run in the fold.
# Each run dict is a Measures dict as defined in eval/extract.py:
#   subnet_vectors_pool / subnet_vectors_test: {SubnetId: (N,) ndarray}  — RAW, uncentered
#   logits:  (N_test,) ndarray
#   pairs:   list[tuple[int,int]]
#   curves:  {("main", j): {"inputs": ndarray, "outputs": ndarray}}
#   y_test:  (N_test,) ndarray
FoldMeasures = dict[str, list[dict]]

ARM_NAMES = ("mains", "gaminet", "concurvity", "regularized")


def load_fold(
    fold_dir: Path,
    run_mode: str = "fixed",
    arm_names: tuple[str, ...] = ARM_NAMES,
    n_runs: int | None = None,
) -> FoldMeasures:
    """Load all completed measures.pt files for one fold.

    A run is considered complete only if its 'done' sentinel exists. Incomplete
    runs are silently skipped. If n_runs is given and fewer done runs are found
    for any arm, a warning is printed but loading continues.

    Args:
        fold_dir: The fold's base directory (e.g. runs/compas_na2m/fold_0).
        run_mode: The run-mode subdirectory, default "fixed".
        arm_names: Which arms to load; defaults to all three.
        n_runs: Expected number of runs per arm (for the completeness warning).

    Returns:
        FoldMeasures keyed by arm name. Arms with zero done runs are included
        as empty lists.
    """
    result: FoldMeasures = {}
    mode_dir = fold_dir / run_mode

    for arm in arm_names:
        arm_dir = mode_dir / arm
        completed = []
        run_i = 0
        while True:
            run_dir = arm_dir / f"run_{run_i}"
            if not run_dir.exists():
                break
            if (run_dir / "done").exists():
                measures = torch.load(
                    run_dir / "measures.pt",
                    map_location="cpu",
                    weights_only=False,
                )
                completed.append(measures)
            run_i += 1

        if n_runs is not None and len(completed) < n_runs:
            print(
                f"WARNING: {arm} has {len(completed)}/{n_runs} completed runs"
                f" in {mode_dir}"
            )

        result[arm] = completed

    return result


def main_effect_instability(
    fold_measures: FoldMeasures,
    arm: str,
) -> float:
    """Across-seed instability of the main effects.
    Lower = more stable.

    Args:
        fold_measures: Output of load_fold.
        arm: "mains", "gaminet", or "concurvity".

    Returns:
        Scalar instability. For comparing: concurvity-aware na2m vs gaminet.
    """
    runs = fold_measures[arm]

    main_ids_per_run = [
        {sid for sid in run["subnet_vectors_test"] if sid[0] == "main"}
        for run in runs
    ]
    common_ids = set.intersection(*main_ids_per_run)

    term_instabilities = []
    for term_id in common_ids:
        centered = []
        for run in runs:
            vec = run["subnet_vectors_test"][term_id].astype(np.float64)
            centered.append(vec - vec.mean())

        matrix = np.stack(centered, axis=0)  # (n_runs, N_test)
        var_per_point = matrix.var(axis=0, ddof=1)  # (N_test,)
        term_instabilities.append(float(np.sqrt(var_per_point.mean())))

    return float(np.mean(term_instabilities))


def selection_frequencies(
    fold_measures: FoldMeasures,
    arm: str,
) -> dict[tuple[int, int], float]:
    """For each pair ever selected, the fraction of runs that selected it.

    A pair selected in 20/20 runs has frequency 1.0; one selected in 1/20
    has frequency 0.05. Pairs never selected do not appear in the result.

    Args:
        fold_measures: Output of load_fold.
        arm: "gaminet" or "concurvity".

    Returns:
        {(j, k): fraction} sorted descending by frequency.
        Empty dict if no run selected any pair.
    """
    runs = fold_measures[arm]
    n_runs = len(runs)

    counts: dict[tuple[int, int], int] = {}
    for run in runs:
        for pair in run["pairs"]:
            key = tuple(pair)
            counts[key] = counts.get(key, 0) + 1

    freqs = {pair: count / n_runs for pair, count in counts.items()}
    return dict(sorted(freqs.items(), key=lambda kv: kv[1], reverse=True))


def mean_pairwise_jaccard(
    fold_measures: FoldMeasures,
    arm: str,
) -> float:
    """Mean Jaccard similarity between all pairs of runs' selected interaction sets.

    For every pair of runs (i, j): J(i,j) = |set_i ∩ set_j| / |set_i ∪ set_j|,
    averaged over all (n_runs choose 2) pairs. Comparable across arms with
    different mean pair counts, and degrades gracefully since each term only
    needs its own two runs to agree, unlike an all-runs intersection.

    Args:
        fold_measures: Output of load_fold.
        arm: Which arm this gets ran on. 

    Returns:
        Mean pairwise Jaccard in [0, 1]. nan if fewer than 2 runs.
        A run with zero pairs contributes 0 to any pair it is in (J = 0 when
        one run selected something and the other did not).
    """
    runs = fold_measures[arm]
    if len(runs) < 2:
        return float("nan")

    pair_sets = [set(tuple(p) for p in run["pairs"]) for run in runs]

    jaccards = []
    for i in range(len(pair_sets)):
        for j in range(i + 1, len(pair_sets)):
            union = pair_sets[i] | pair_sets[j]
            if not union:
                jaccards.append(1.0)
                continue
            intersection = pair_sets[i] & pair_sets[j]
            jaccards.append(len(intersection) / len(union))

    return float(np.mean(jaccards))



def concurvity_summary(
    fold_measures: FoldMeasures,
    arm: str,
) -> dict:
    """Post-hoc concurvity of main effects and interactions against all other fitted terms.

    Uses subnet_vectors_pool. For each subnet, regresses its output vector on all
    other subnets via OLS adj-R² (concurvity_score). Main effects are included
    for all arms; interactions only for arms B and C.

    For mains: each main is regressed on all other mains + selected interactions,
    so arm A gives a mains-only baseline while B/C show whether interactions
    increase main concurvity.

    Scores are aggregated per run (mean over that run's terms), then mean ± std
    across runs. frac_concurve is the number of concurve terms in that category
    divided by the TOTAL number of subnets in the run (mains + interactions).

    Args:
        fold_measures: Output of load_fold.
        arm: "mains", "gaminet", or "concurvity".

    Returns:
        {
          "mains":        {"mean", "mean_std", "frac_concurve", "frac_concurve_std", "max"},
          "interactions": {"mean", "mean_std", "frac_concurve", "frac_concurve_std", "max"},
        }
        interactions values are nan for arm A (no interactions selected).
    """
    THRESHOLD = 0.5

    def _summarise(per_run_means, per_run_fracs, all_scores):
        if not per_run_means:
            nan = float("nan")
            return {"mean": nan, "mean_std": nan,
                    "frac_concurve": nan, "frac_concurve_std": nan, "max": nan}
        return {
            "mean":              float(np.mean(per_run_means)),
            "mean_std":          float(np.std(per_run_means, ddof=1)),
            "frac_concurve":     float(np.mean(per_run_fracs)),
            "frac_concurve_std": float(np.std(per_run_fracs, ddof=1)),
            "max":               float(np.max(all_scores)),
        }

    main_run_means, main_run_fracs, main_all = [], [], []
    inter_run_means, inter_run_fracs, inter_all = [], [], []
    total_run_means, total_run_fracs, total_all = [], [], []

    for run in fold_measures[arm]:
        pool = run["subnet_vectors_pool"]
        main_ids = [sid for sid in pool if sid[0] == "main"]
        inter_ids = [sid for sid in pool if sid[0] == "inter"]
        n_total = len(main_ids) + len(inter_ids)

        main_scores = [concurvity_score(sid, pool) for sid in main_ids]
        main_run_means.append(float(np.mean(main_scores)))
        main_run_fracs.append(sum(s > THRESHOLD for s in main_scores) / n_total)
        main_all.extend(main_scores)

        if inter_ids:
            inter_scores = [concurvity_score(sid, pool) for sid in inter_ids]
            inter_run_means.append(float(np.mean(inter_scores)))
            inter_run_fracs.append(sum(s > THRESHOLD for s in inter_scores) / n_total)
            inter_all.extend(inter_scores)
        else:
            inter_scores = []

        all_scores = main_scores + inter_scores
        total_run_means.append(float(np.mean(all_scores)))
        total_run_fracs.append(sum(s > THRESHOLD for s in all_scores) / n_total)
        total_all.extend(all_scores)

    return {
        "mains":        _summarise(main_run_means, main_run_fracs, main_all),
        "interactions": _summarise(inter_run_means, inter_run_fracs, inter_all),
        "total":        _summarise(total_run_means, total_run_fracs, total_all),
    }


def accuracy_summary(
    fold_measures: FoldMeasures,
    task: str = "classification",
) -> dict:
    """Per-arm single-run and ensemble performance metric, plus mean interaction count.

    Mirrors the training metrics in na2m/training/metrics.py:
    - classification → AUROC (higher is better)
    - regression     → RMSE  (lower is better)

    Algorithm:
    1. Single-run metric per arm:
       - For each run: compute metric from raw logits vs y_test → scalar.
       - Mean over runs.
    2. Ensemble metric per arm:
       - classification: average sigmoid(logits) across runs, then AUROC.
       - regression: average raw logits across runs, then RMSE.
       y_test is shared within a fold (fixed outer test split).
    3. Mean pair count per arm: mean(len(run["pairs"])) over runs.

    Args:
        fold_measures: Output of load_fold.
        task: "classification" or "regression".

    Returns:
        {arm_name: {"single": float, "ensemble": float, "mean_n_pairs": float}}
        for each arm present in fold_measures.
    """
    from sklearn.metrics import roc_auc_score

    result = {}
    for arm, runs in fold_measures.items():
        if not runs:
            continue

        y_test = runs[0]["y_test"]

        if task == "classification":
            single_vals = []
            for run in runs:
                probs = 1.0 / (1.0 + np.exp(-run["logits"]))
                single_vals.append(roc_auc_score(run["y_test"], probs))

            mean_probs = np.mean(
                [1.0 / (1.0 + np.exp(-run["logits"])) for run in runs],
                axis=0,
            )
            ensemble = float(roc_auc_score(y_test, mean_probs))

        else:  # regression → RMSE
            def _rmse(preds, y):
                return float(np.sqrt(np.mean((preds - y) ** 2)))

            single_vals = [_rmse(run["logits"], run["y_test"]) for run in runs]
            mean_preds = np.mean([run["logits"] for run in runs], axis=0)
            ensemble = _rmse(mean_preds, y_test)

        mean_n_pairs = float(np.mean([len(run["pairs"]) for run in runs]))
        result[arm] = {
            "single": float(np.mean(single_vals)),
            "single_std": float(np.std(single_vals, ddof=1)),
            "ensemble": ensemble,
            "mean_n_pairs": mean_n_pairs,
        }

    return result


def shape_plots(
    fold_measures: FoldMeasures,
    arm: str,
    feature_meta,
) -> Figure:
    """Per-feature shape plots: faint per-run lines + thick ensemble mean, shared y-axis.

    Uses the pre-centered curves stored in measures["curves"] — no recentering
    needed. The x-axis values are already in real units (stored by the extractor).
    The thick line is the pointwise mean across runs (ensemble average).

    Algorithm:
    1. Collect all ("main", j) curve ids from the first run's curves dict.
    2. Compute global y-limits: min/max of all outputs across all runs and features.
    3. For each feature subplot:
       a. Plot each run's outputs (faint, low alpha).
       b. Compute mean curve: mean across all run outputs at each grid point.
       c. Plot mean (thick).
       d. Set shared ylim, axhline at 0, title from feature_meta[j].name.
       e. For categorical features: xticks = level labels (stored as inputs strings).
    4. fig.tight_layout(); return fig.

    Args:
        fold_measures: Output of load_fold.
        arm: Which arm's runs to plot.
        feature_meta: FeatureMeta list from preprocess() — used for subplot titles.

    Returns:
        matplotlib Figure.
    """
    # TODO: interaction surface curves deferred; only main effects plotted here.
    runs = fold_measures[arm]
    main_ids = sorted(
        [sid for sid in runs[0]["curves"] if sid[0] == "main"],
        key=lambda s: s[1],
    )
    n = len(main_ids)

    all_outputs = [
        run["curves"][sid]["outputs"]
        for run in runs
        for sid in main_ids
    ]
    y_min = float(min(o.min() for o in all_outputs))
    y_max = float(max(o.max() for o in all_outputs))

    fig, axes = plt.subplots(1, n, figsize=(3 * n, 4), sharey=True)
    if n == 1:
        axes = [axes]

    for ax, sid in zip(axes, main_ids):
        j = sid[1]
        inputs = runs[0]["curves"][sid]["inputs"]  # identical across runs
        run_outputs = []
        for run in runs:
            outputs = run["curves"][sid]["outputs"]
            ax.plot(inputs, outputs, color="steelblue", alpha=0.2, linewidth=0.8)
            run_outputs.append(outputs)

        mean_curve = np.mean(np.stack(run_outputs, axis=0), axis=0)
        ax.plot(inputs, mean_curve, color="steelblue", linewidth=2.0)

        ax.set_ylim(y_min, y_max)
        ax.axhline(0, color="k", linewidth=0.5, linestyle="--")
        ax.set_title(feature_meta[j].name)

        if not np.issubdtype(np.array(inputs).dtype, np.floating):
            ax.set_xticks(range(len(inputs)))
            ax.set_xticklabels(inputs, rotation=45, ha="right")

    fig.suptitle(arm)
    fig.tight_layout()
    return fig
