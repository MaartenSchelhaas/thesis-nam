"""
reduce_fold0.py — compute and display all reducer metrics for fold_0.
Run from repo root: python -m scripts.reduce.reduce_fold0
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from na2m.data.compas import CompasDataset
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

DATASET      = CompasDataset()
DATASET_PATH = r"datasets/raw/compas-scores-two-years.csv"
FOLD_DIR     = Path("runs/compas_na2m/fold_0")
RUN_MODE     = "fixed"
N_RUNS       = 20


def _section(title: str) -> None:
    print(f"\n{'─' * 60}")
    print(f"  {title}")
    print(f"{'─' * 60}")


def main() -> None:
    df = DATASET.load(DATASET_PATH)
    _, _, feature_meta = DATASET.preprocess(df)

    config = load_na2m_config(str(FOLD_DIR / "mains_tuned_config.yaml"))
    task = config.task

    fold = load_fold(FOLD_DIR, run_mode=RUN_MODE, n_runs=N_RUNS)

    for arm, runs in fold.items():
        print(f"  {arm}: {len(runs)} runs loaded")

    # --- Main-effect instability (headline) ---
    _section("Main-effect instability  [lower = more stable]")
    for arm in ("mains", "gaminet", "concurvity", "regularized"):
        if fold.get(arm):
            val = main_effect_instability(fold, arm)
            print(f"  {arm:12s}  {val:.6f}")

    # --- Selection frequencies + mean pairwise Jaccard ---
    _section("Interaction selection frequencies  [fraction of runs selecting each pair]")
    for arm in ("gaminet", "concurvity", "regularized"):
        if fold.get(arm):
            freqs = selection_frequencies(fold, arm)
            mpj = mean_pairwise_jaccard(fold, arm)
            if not freqs:
                print(f"  {arm:12s}  no pairs selected")
                continue
            print(f"  {arm:12s}  mean_pairwise_jaccard={mpj:.3f}  n_unique_pairs={len(freqs)}")
            for pair, freq in freqs.items():
                n = round(freq * len(fold[arm]))
                print(f"    {str(pair):12s}  {freq:.2f}  ({n}/{len(fold[arm])} runs)")

    # --- Post-hoc concurvity ---
    _section("Post-hoc concurvity (adj-R², fine-tuned model on pool, threshold=0.5)")
    for arm in ("mains", "gaminet", "concurvity", "regularized"):
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

    # --- AUROC / RMSE ---
    metric_name = "auroc" if task == "classification" else "rmse"
    _section(f"Predictive performance  [{metric_name}]")
    acc = accuracy_summary(fold, task=task)
    for arm, vals in acc.items():
        print(
            f"  {arm:12s}  single={vals['single']:.4f} ± {vals['single_std']:.4f}"
            f"  ensemble={vals['ensemble']:.4f}"
            f"  mean_n_pairs={vals['mean_n_pairs']:.1f}"
        )

    # --- Shape plots → fold's run-mode folder ---
    _section("Shape plots")
    plot_dir = FOLD_DIR / RUN_MODE / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    for arm in ("mains", "gaminet", "concurvity", "regularized"):
        if fold.get(arm):
            fig = shape_plots(fold, arm, feature_meta)
            out = plot_dir / f"shape_{arm}.png"
            fig.savefig(out, dpi=120, bbox_inches="tight")
            plt.close(fig)
            print(f"  saved {out}")

    print()


if __name__ == "__main__":
    main()
