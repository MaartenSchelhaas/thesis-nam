"""
reduce_fold.py — compute and display all reducer metrics for one fold.
Run from repo root: python -m scripts.reduce.reduce_fold
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from na2m.data.compas import CompasDataset
from na2m.data.california_housing import CaliforniaHousingDataset
from na2m.utils.config import load_na2m_config
from na2m.eval.reduce import (
    accuracy_summary,
    concurvity_summary,
    load_fold,
    main_effect_instability,
    mean_pairwise_jaccard,
    selection_frequencies,
    shape_plots,
)

ALL_ARMS   = ("mains", "gaminet", "concurvity", "regularized")
INTER_ARMS = ("gaminet", "concurvity", "regularized")


def _section(title: str) -> None:
    print(f"\n{'─' * 60}")
    print(f"  {title}")
    print(f"{'─' * 60}")


def _print_instability(fold: dict) -> None:
    """Print per-arm main-effect instability for this fold.

    Args:
        fold: FoldMeasures dict for this fold, from load_fold.
    """
    _section("Main-effect instability  [lower = more stable]")
    for arm in ALL_ARMS:
        if fold.get(arm):
            val = main_effect_instability(fold, arm)
            print(f"  {arm:12s}  {val:.6f}")


def _print_selection(fold: dict) -> None:
    """Print per-arm interaction selection frequencies + mean pairwise Jaccard.

    Args:
        fold: FoldMeasures dict for this fold, from load_fold.
    """
    _section("Interaction selection frequencies  [fraction of runs selecting each pair]")
    for arm in INTER_ARMS:
        if not fold.get(arm):
            continue
        freqs = selection_frequencies(fold, arm)
        mpj = mean_pairwise_jaccard(fold, arm)
        if not freqs:
            print(f"  {arm:12s}  no pairs selected")
            continue
        print(f"  {arm:12s}  mean_pairwise_jaccard={mpj:.3f}  n_unique_pairs={len(freqs)}")
        for pair, freq in freqs.items():
            n = round(freq * len(fold[arm]))
            print(f"    {str(pair):12s}  {freq:.2f}  ({n}/{len(fold[arm])} runs)")


def _print_concurvity(fold: dict) -> None:
    """Print per-arm post-hoc concurvity summary for this fold.

    Args:
        fold: FoldMeasures dict for this fold, from load_fold.
    """
    _section("Post-hoc concurvity (adj-R², fine-tuned model on pool, threshold=0.5)")
    for arm in ALL_ARMS:
        if not fold.get(arm):
            continue
        s = concurvity_summary(fold, arm)
        for part in ("mains", "interactions", "total"):
            p = s[part]
            if np.isnan(p["mean"]):
                continue
            print(
                f"  {arm:12s}  [{part:12s}]"
                f"  mean={p['mean']:.4f} ± {p['mean_std']:.4f}"
                f"  frac_concurve={p['frac_concurve']:.3f} ± {p['frac_concurve_std']:.3f}"
                f"  max={p['max']:.4f}"
            )


def _print_performance(fold: dict, task: str) -> None:
    """Print per-arm predictive performance for this fold.

    Args:
        fold: FoldMeasures dict for this fold, from load_fold.
        task: "classification" or "regression" — passed to accuracy_summary.
    """
    metric_name = "auroc" if task == "classification" else "rmse"
    _section(f"Predictive performance  [{metric_name}]")
    acc = accuracy_summary(fold, task=task)
    for arm, vals in acc.items():
        print(
            f"  {arm:12s}  single={vals['single']:.4f} ± {vals['single_std']:.4f}"
            f"  ensemble={vals['ensemble']:.4f}"
            f"  mean_n_pairs={vals['mean_n_pairs']:.1f}"
        )


def _save_shape_plots(fold: dict, feature_meta, plot_dir: Path) -> None:
    """Render + save per-arm shape plots for this fold.

    Args:
        fold: FoldMeasures dict for this fold, from load_fold.
        feature_meta: FeatureMeta list from preprocess().
        plot_dir: Directory to save shape_<arm>.png into (created if absent).
    """
    _section("Shape plots")
    plot_dir.mkdir(parents=True, exist_ok=True)
    for arm in ALL_ARMS:
        if fold.get(arm):
            fig = shape_plots(fold, arm, feature_meta)
            out = plot_dir / f"shape_{arm}.png"
            fig.savefig(out, dpi=120, bbox_inches="tight")
            plt.close(fig)
            print(f"  saved {out}")


def main() -> None:
    # ── Edit these before running ────────────────────────────────────────────
    # --- Dataset (swap these three lines to switch) ---
    DATASET      = CompasDataset()
    DATASET_PATH = r"datasets/raw/compas-scores-two-years.csv"
    BASE_DIR     = Path("runs/compas_na2m")
    # DATASET      = CaliforniaHousingDataset()
    # DATASET_PATH = None  # ignored — CaliforniaHousingDataset fetches via sklearn
    # BASE_DIR     = Path("runs/california_housing_na2m")
    # ------------------------------------------------
    FOLD_IDX = 0
    RUN_MODE = "fixed"
    N_RUNS   = 20
    # ────────────────────────────────────────────────────────────────────────

    fold_dir = BASE_DIR / f"fold_{FOLD_IDX}"

    config = load_na2m_config(str(fold_dir / "mains_tuned_config.yaml"))
    task = config.task

    df = DATASET.load(DATASET_PATH)
    _, _, feature_meta = DATASET.preprocess(df)

    fold = load_fold(fold_dir, run_mode=RUN_MODE, n_runs=N_RUNS)

    for arm, runs in fold.items():
        print(f"  {arm}: {len(runs)} runs loaded")

    _print_instability(fold)
    _print_selection(fold)
    _print_concurvity(fold)
    _print_performance(fold, task)
    _save_shape_plots(fold, feature_meta, fold_dir / RUN_MODE / "plots")

    print()


if __name__ == "__main__":
    main()
