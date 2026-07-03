"""
evaluate_na2m_mains.py — K-fold × n_runs subsample evaluation for NA2M main effects (Arm A only).

Orchestrates folders, seeds and data/datasplits. Hands that of to the tuning
scripts, and the model_runner.py

Trains n_runs subsampled models per fold (each on a fresh random train/val split of the pool)
and stores measures.pt for each run. Metric computation (ensemble + CI) is handled separately
by the reducer (src/na2m/eval/reduce.py), following the store-everything → reducer boundary.

If mains_tuned_config.yaml already exists in fold_k/, tuning is skipped.
Otherwise, hyperparameter search runs automatically (tune_main_na2m.tune_fold).

Output layout:
    BASE_DIR/
        fold_k/
            mains_tuned_config.yaml      ← shared; tuned once if absent
            subsample/
                mains/
                    run_i/  {model.pt, measures.pt, done}
                    ...

The subsample/mains/run_i/ path is the same one run_na2m_eval.py uses in
subsample mode — shared done sentinels prevent duplicate training if both
scripts target the same BASE_DIR.

run with: python -m scripts.eval.evaluate_na2m_mains
"""

from pathlib import Path

import numpy as np
from sklearn.model_selection import KFold, train_test_split

from na2m.data.compas import CompasDataset
from na2m.data.california_housing import CaliforniaHousingDataset
from na2m.utils.config import load_na2m_config, load_na2m_search_config
from scripts.eval.model_runner import run_main_effects
from scripts.tuning.tune_main_na2m import tune_fold
from scripts.eval.run_na2m_eval import (
    derive_run_seeds,
    derive_fold_split_seeds,
    fold_inner_split,
)


# --------------------------------------------------------------------------- #
# One fold: tune + store config, run n_runs subsample models                   #
# --------------------------------------------------------------------------- #

def run_fold(
    fixed_params: dict,
    search_space: dict,
    X: np.ndarray,
    y: np.ndarray,
    feature_meta,
    pool_idx: np.ndarray,
    test_idx: np.ndarray,
    tune_dir: Path,
    run_dir: Path,
    *,
    fold_idx: int,
    n_runs: int,
    fold_split_seed: int,
    run_seeds: list[int],
) -> None:
    """Tune once if needed, train n_runs subsample models and store their measures.

    Args:
        fixed_params: Non-search fields from load_na2m_search_config.
        search_space: Full search space dict.
        X, y: Full dataset as numpy arrays.
        feature_meta: FeatureMeta list from preprocess().
        pool_idx: This fold's pool indices (~80%).
        test_idx: This fold's test slice indices.
        tune_dir: Shared fold dir for tuned configs (base_dir/fold_k/).
        run_dir: Subsample fold dir for run outputs (base_dir/fold_k/subsample/).
        fold_idx: Fold number (study names / logging).
        n_runs: Number of subsample runs.
        fold_split_seed: This fold's split seed (keys the tuning split).
        run_seeds: The n_runs init/subsample seeds for this fold.
    """
    tune_dir.mkdir(parents=True, exist_ok=True)
    run_dir.mkdir(parents=True, exist_ok=True)

    # Tuning split is always deterministic from fold_split_seed, not run-mode-dependent.
    val_frac_of_pool = fixed_params["val_frac"] / (1 - fixed_params["test_frac"])
    tune_idx, tune_val_idx = train_test_split(
        pool_idx,
        test_size=val_frac_of_pool,
        random_state=fold_split_seed,
    )

    mains_config_path = tune_dir / "mains_tuned_config.yaml"
    if not mains_config_path.exists():
        mains_space = {k: v for k, v in search_space.items() if k != "marginal_clarity"}
        tune_fold(
            fixed_params=fixed_params,
            search_space=mains_space,
            feature_meta=feature_meta,
            X_train=X[tune_idx],
            y_train=y[tune_idx],
            X_val=X[tune_val_idx],
            y_val=y[tune_val_idx],
            output_path=mains_config_path,
            study_name=f"fold_{fold_idx}_main_search",
        )

    mains_config = load_na2m_config(str(mains_config_path))
    val_frac_of_pool = mains_config.val_frac / (1 - mains_config.test_frac)

    mains_run_dir = run_dir / "mains"

    for run_i in range(n_runs):
        run_seed = run_seeds[run_i]
        run_out = mains_run_dir / f"run_{run_i}"

        if (run_out / "done").exists():
            print(f"  [fold {fold_idx}] run {run_i} already done, skipping.")
            continue

        run_out.mkdir(parents=True, exist_ok=True)

        train_idx, val_idx = fold_inner_split(
            pool_idx,
            val_frac_of_pool,
            run_mode="subsample",
            fold_split_seed=fold_split_seed,
            run_seed=run_seed,
        )
        run_main_effects(
            mains_config, feature_meta,
            X[train_idx], y[train_idx],
            X[val_idx], y[val_idx],
            X[test_idx], y[test_idx],
            run_seed, run_out,
        )


# --------------------------------------------------------------------------- #
# Outer k-fold loop                                                             #
# --------------------------------------------------------------------------- #

def evaluate_na2m_mains(
    search_config_path: str,
    X: np.ndarray,
    y: np.ndarray,
    feature_meta,
    base_dir: Path,
    *,
    n_runs: int,
    n_folds: int,
) -> None:
    """Outer k-fold loop: tune once per fold, then train n_runs subsample models.

    Stores measures.pt per run. Metric computation (ensemble + CI) is left to
    the reducer (src/na2m/eval/reduce.py).

    Args:
        search_config_path: Search YAML (fixed params + search spaces).
        X, y: Full dataset as numpy arrays.
        feature_meta: FeatureMeta list from preprocess().
        base_dir: Root output dir.
        n_runs: Number of subsample runs per fold.
        n_folds: Number of outer CV folds.
    """
    fixed_params, search_space = load_na2m_search_config(search_config_path)

    fold_split_seeds = derive_fold_split_seeds(fixed_params["seed"], n_folds)
    run_seeds        = derive_run_seeds(fixed_params["seed"], n_folds, n_runs)

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=fixed_params["seed"])
    for fold_idx, (pool_idx, test_idx) in enumerate(kf.split(X)):
        print(f"\n--- Fold {fold_idx} ---")
        tune_dir = base_dir / f"fold_{fold_idx}"
        run_dir  = base_dir / f"fold_{fold_idx}" / "subsample"
        run_fold(
            fixed_params, search_space, X, y, feature_meta,
            pool_idx, test_idx,
            tune_dir=tune_dir,
            run_dir=run_dir,
            fold_idx=fold_idx,
            n_runs=n_runs,
            fold_split_seed=fold_split_seeds[fold_idx],
            run_seeds=run_seeds[fold_idx],
        )


def main() -> None:
    # ------------------------------ PARAMS ------------------------------ #
    # --- Dataset (swap these two blocks to switch) ---
    SEARCH_CONFIG_PATH = r"C:\Users\maart\OneDrive\Documenten\Universiteit\Scriptie\python_repo\thesis-nam\configs\compas_na2m_search.yaml"
    DATASET = CompasDataset();  DATASET_NAME = "compas"
    # SEARCH_CONFIG_PATH = r"C:\Users\maart\OneDrive\Documenten\Universiteit\Scriptie\python_repo\thesis-nam\configs\california_housing_na2m_search.yaml"
    # DATASET      = CaliforniaHousingDataset()
    # DATASET_NAME = "california_housing"
    # ------------------------------------------------
    N_RUNS   = 20
    N_FOLDS  = 5
    BASE_DIR = Path(f"runs/{DATASET_NAME}_na2m")
    FRESH    = False  # True → delete fold_k/subsample/mains/ dirs for a clean restart
    # -------------------------------------------------------------------- #

    if FRESH:
        import shutil
        for fold_idx in range(N_FOLDS):
            stale = BASE_DIR / f"fold_{fold_idx}" / "subsample" / "mains"
            if stale.exists():
                shutil.rmtree(stale)
                print(f"Deleted {stale}")

    fixed_params, _ = load_na2m_search_config(SEARCH_CONFIG_PATH)
    df = DATASET.load(fixed_params.get("dataset_path"))
    X, y, feature_meta = DATASET.preprocess(df)

    evaluate_na2m_mains(
        SEARCH_CONFIG_PATH, X, y, feature_meta, BASE_DIR,
        n_runs=N_RUNS,
        n_folds=N_FOLDS,
    )

if __name__ == "__main__":
    main()