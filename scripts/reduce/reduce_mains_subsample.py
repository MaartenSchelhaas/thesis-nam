"""
reduce_mains_subsample.py — ensemble accuracy + CI for the mains-only subsample evaluation.

Loads measures.pt from evaluate_na2m_mains.py output (subsample/mains/run_i/),
calls accuracy_summary per fold (which handles the logit averaging), then aggregates
the 5 ensemble values into a point estimate + 95% CI — matching the Agarwal et al. protocol.

run with: python -m scripts.reduce.reduce_mains_subsample
"""

import numpy as np
from pathlib import Path

from na2m.eval.reduce import load_fold, accuracy_summary
from na2m.utils.config import load_na2m_config


def main() -> None:
    # ------------------------------ PARAMS ------------------------------ #
    BASE_DIR = Path("runs/compas_na2m")
    N_FOLDS  = 5
    N_RUNS   = 20
    # -------------------------------------------------------------------- #

    # Read task from fold_0 config (same for all folds).
    task = load_na2m_config(str(BASE_DIR / "fold_0" / "mains_tuned_config.yaml")).task

    fold_ensemble_metrics = []

    for fold_idx in range(N_FOLDS):
        fold_dir = BASE_DIR / f"fold_{fold_idx}"
        fold_measures = load_fold(
            fold_dir,
            run_mode="subsample",
            arm_names=("mains",),
            n_runs=N_RUNS,
        )
        summary = accuracy_summary(fold_measures, task=task)
        mains = summary["mains"]
        ensemble = mains["ensemble"]
        fold_ensemble_metrics.append(ensemble)
        print(
            f"fold {fold_idx}:  ensemble={ensemble:.4f}"
            f"  single={mains['single']:.4f} ± {mains['single_std']:.4f}"
            f"  n_runs={len(fold_measures['mains'])}"
        )

    fold_ensemble_metrics = np.array(fold_ensemble_metrics)
    mean = float(fold_ensemble_metrics.mean())
    std  = float(fold_ensemble_metrics.std(ddof=1))
    ci95 = 1.96 * std / np.sqrt(N_FOLDS)

    print(f"\nMains subsample ({N_FOLDS} folds × {N_RUNS} runs)")
    print(f"  mean ± std : {mean:.4f} ± {std:.4f}")
    print(f"  95% CI     : {mean:.4f} ± {ci95:.4f}  [{mean - ci95:.4f}, {mean + ci95:.4f}]")


if __name__ == "__main__":
    main()