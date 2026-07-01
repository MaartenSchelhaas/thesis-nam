"""
run_na2m_eval.py — k-fold × n_runs evaluation loop for all three NA2M arms.

Call stack (each level owns one responsibility):
    evaluate_na2m  → outer k-fold loop, seed derivation
      └ run_fold   → tune + store configs once per fold, loop over runs
          └ run_arms → train mains once, branch arms B/C/D off the same checkpoint

run_mode controls the inner train/val split strategy:
    "fixed"     → same split every run within a fold (used for stability evaluation)
    "subsample" → fresh split per run (Agarwal-style ensemble)

Resume is granular: each done sentinel (written last by model_runner) marks one
arm/run as complete. Tuned configs are shared across run modes; run outputs are
mode-specific, so switching run_mode reruns all models without re-tuning.

Output layout:
    runs/compas/
        fold_k/
            mains_tuned_config.yaml      ← shared across run modes
            gaminet_tuned_config.yaml
            concurvity_tuned_config.yaml
            <run_mode>/
                mains/run_i/      model.pt, measures.pt, done
                gaminet/run_i/    measures.pt, done
                concurvity/run_i/ measures.pt, done

run with: python -m scripts.na2m.run_na2m_eval                
"""

from pathlib import Path

import numpy as np
from sklearn.model_selection import KFold, train_test_split

from na2m.data.compas import CompasDataset
from na2m.data.california_housing import CaliforniaHousingDataset
from na2m.utils.config import load_na2m_config, load_na2m_search_config
from scripts.na2m.model_runner import (
    run_main_effects,
    run_arm,
    _ARM_FLAGS,
)
from scripts.na2m.tune_main_na2m import tune_fold
from scripts.na2m.tune_clarity import tune_clarity_fold
from scripts.na2m.tune_concurvity_reg import concurvity_reg_fold


# --------------------------------------------------------------------------- #
# Seed derivation — one master seed, clean init/split independence             #
# --------------------------------------------------------------------------- #

def derive_run_seeds(
    master: int,
    n_folds: int,
    n_runs: int,
) -> list[list[int]]:
    """Per-(fold, run) init/optimization seeds, independent across the whole grid.

    Args:
        master: The single master seed the whole evaluation derives from.
        n_folds: Number of outer CV folds.
        n_runs: Number of runs per fold.

    Returns:
        run_seeds[fold][run] — the seed handed to run_main_effects/run_arm so init
        + optimization order vary per run (and, in subsample mode, the resample).

    """
    root = np.random.SeedSequence(master)
    fold_sequences = root.spawn(n_folds)

    run_seeds = []
    for fold_ss in fold_sequences:
        run_sequences = fold_ss.spawn(n_runs)
        fold_run_seeds = []
        for run_ss in run_sequences:
            fold_run_seeds.append(int(run_ss.generate_state(1)[0]))
        run_seeds.append(fold_run_seeds)

    return run_seeds


def derive_fold_split_seeds(
    master: int,
    n_folds: int,
) -> list[int]:
    """One split seed per fold — keys the fold's FIXED inner train/val split.

    Spawned from a DIFFERENT branch than the run seeds (master + 1) so the data
    split never moves when an init seed changes — the split contract.

    Args:
        master: The single master seed the whole evaluation derives from.
        n_folds: Number of outer CV folds.

    Returns:
        One int per fold, used as random_state for the fixed inner split AND for
        the per-fold tuning split.
    """
    root = np.random.SeedSequence(master + 1)
    fold_sequences = root.spawn(n_folds)

    fold_split_seeds = []
    for fold_ss in fold_sequences:
        fold_split_seeds.append(int(fold_ss.generate_state(1)[0]))

    return fold_split_seeds


# --------------------------------------------------------------------------- #
# Split policy                                                                 #
# --------------------------------------------------------------------------- #

def fold_inner_split(
    pool_idx: np.ndarray,
    val_frac_of_pool: float,
    *,
    run_mode: str,
    fold_split_seed: int,
    run_seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Resolve the train/val split for one run, per run_mode.

    fixed     → key off fold_split_seed → SAME split every run (split contract).
    subsample → key off run_seed → a fresh split per run (Agarwal ensemble).

    Args:
        pool_idx: The fold's 80% pool indices to split into train/val.
        val_frac_of_pool: val_frac / (1 - test_frac).
        run_mode: "fixed" | "subsample".
        fold_split_seed: This fold's split seed (used in fixed mode).
        run_seed: This run's seed (used in subsample mode).

    Returns:
        (train_idx, val_idx) — subsets of pool_idx.

    """
    if run_mode == "fixed":
        random_state = fold_split_seed
    else:
        random_state = run_seed

    train_idx, val_idx = train_test_split(
        pool_idx,
        test_size=val_frac_of_pool,
        random_state=random_state,
    )
    return train_idx, val_idx


# --------------------------------------------------------------------------- #
# One run: train mains once, branch all arms                                   #
# --------------------------------------------------------------------------- #

def run_arms(
    X: np.ndarray,
    y: np.ndarray,
    feature_meta,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    tune_dir: Path,
    run_dir: Path,
    run_i: int,
    run_seed: int,
) -> None:
    """One run: ensure the mains exist, then produce every interaction arm — resumably.

    Mains are trained once per run, persisted, then B and C deepcopy from that
    trained state. All configs are loaded from tune_dir (shared across run modes);
    run outputs (model.pt, measures.pt, done) are written under run_dir (mode-specific).

    Args:
        X, y: Full dataset as numpy arrays.
        feature_meta: FeatureMeta list from preprocess().
        train_idx, val_idx: This run's train / val indices (from fold_inner_split).
        test_idx: The fold's test slice (measure extraction inside model_runner).
        tune_dir: Shared fold dir holding all tuned config YAML files.
        run_dir: Mode-specific fold dir; per-arm run outputs are written here.
        run_i: Run index (used to name the run subdirectory).
        run_seed: Seed for mains init + optimization, passed to run_main_effects/run_arm.

    """
    mains_config = load_na2m_config(str(tune_dir / "mains_tuned_config.yaml"))

    # --- Arm A (mains-only): train once per run, persist model ---
    mains_dir = run_dir / "mains" / f"run_{run_i}"
    mains_dir.mkdir(parents=True, exist_ok=True)

    if not (mains_dir / "done").exists():
        run_main_effects(
            mains_config, feature_meta,
            X[train_idx], y[train_idx],
            X[val_idx], y[val_idx],
            X[test_idx], y[test_idx],
            run_seed, mains_dir,
        )

    # --- Arms B, C, D: branch from the persisted mains checkpoint ---
    mains_model_path = mains_dir / "model.pt"

    for arm, (with_interactions, with_concurvity_filter, _) in _ARM_FLAGS.items():
        if not with_interactions:
            continue

        arm_config_yaml = tune_dir / f"{arm}_tuned_config.yaml"
        if not arm_config_yaml.exists():
            print(
                f"[run_arms] Skipping arm '{arm}': {arm_config_yaml.name} not found. "
                f"For 'regularized': inspect the lambda_2 sweep plot and run "
                f"confirm_regularized_arm.py with fold_dir={tune_dir}"
            )
            continue

        arm_dir = run_dir / arm / f"run_{run_i}"
        if (arm_dir / "done").exists():
            continue

        arm_dir.mkdir(parents=True, exist_ok=True)
        arm_config = load_na2m_config(str(arm_config_yaml))
        run_arm(
            arm_config, mains_model_path, feature_meta,
            X[train_idx], y[train_idx],
            X[val_idx], y[val_idx],
            X[test_idx], y[test_idx],
            run_seed, arm_dir,
            with_interactions=with_interactions,
            with_concurvity_filter=with_concurvity_filter,
        )


# --------------------------------------------------------------------------- #
# One fold: tune + store config, then loop runs                                #
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
    run_mode: str,
    n_runs: int,
    fold_split_seed: int,
    run_seeds: list[int],
) -> None:
    """Everything that happens within ONE fold.

    1. Tune + STORE hyperparameter configs once into tune_dir (shared across run modes).
    2. Load the mains tuned config as NA2MConfig.
    3. Loop n_runs: resolve the split (fold_inner_split), then run_arms into run_dir.

    Args:
        fixed_params: Non-search fields from load_na2m_search_config.
        search_space: Full search space dict (mains + marginal_clarity).
        X, y: Full dataset as numpy arrays.
        feature_meta: FeatureMeta list from preprocess().
        pool_idx: This fold's 80% pool indices.
        test_idx: This fold's test slice indices.
        tune_dir: Shared fold dir for tuned configs (base_dir/fold_k/).
        run_dir: Mode-specific fold dir for run outputs (base_dir/run_mode/fold_k/).
        fold_idx: Fold number (study names / logging).
        run_mode: "fixed" | "subsample".
        n_runs: Number of runs in this fold.
        fold_split_seed: This fold's split seed (keys the fixed inner split + tuning).
        run_seeds: The n_runs init seeds for this fold.
    """
    tune_dir.mkdir(parents=True, exist_ok=True)
    run_dir.mkdir(parents=True, exist_ok=True)

    # --- Tuning split (always deterministic from fold_split_seed; not run-mode-dependent) ---
    val_frac_of_pool = fixed_params["val_frac"] / (1 - fixed_params["test_frac"])
    tune_idx, tune_val_idx = train_test_split(
        pool_idx,
        test_size=val_frac_of_pool,
        random_state=fold_split_seed,
    )

    # --- Stage 1: main-effects tuning (once per fold, shared across run modes) ---
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

    # --- Stage 2: per-arm clarity tuning (each arm guarded independently) ---
    # Arms with with_concurvity_reg=True (arm D) skip Optuna clarity tuning —
    # their lambda_2 is found via grid sweep in tune_concurvity_reg.py instead.
    for arm, (with_interactions, with_concurvity_filter, with_concurvity_reg) in _ARM_FLAGS.items():
        if not with_interactions or with_concurvity_reg:
            continue
        arm_config_path = tune_dir / f"{arm}_tuned_config.yaml"
        if arm_config_path.exists():
            continue
        tune_clarity_fold(
            mains_config_path=mains_config_path,
            n_trials=fixed_params["clarity_n_trials"],
            search_spec=search_space["marginal_clarity"],
            feature_meta=feature_meta,
            X_train=X[tune_idx],
            y_train=y[tune_idx],
            X_val=X[tune_val_idx],
            y_val=y[tune_val_idx],
            output_path=arm_config_path,
            with_concurvity_filter=with_concurvity_filter,
            study_name=f"fold_{fold_idx}_{arm}_clarity_search",
        )

    # --- Arm D lambda_2 grid sweep (writes CSV + plot; does NOT write final config) ---
    sweep_csv = tune_dir / "regularized_lambda2_sweep.csv"
    lambda2_grid = fixed_params.get("lambda2_grid", [])
    n_sweep_seeds = fixed_params.get("n_sweep_seeds", 1)
    gaminet_config_path = tune_dir / "gaminet_tuned_config.yaml"
    if lambda2_grid and gaminet_config_path.exists() and not sweep_csv.exists():
        concurvity_reg_fold(
            mains_config_path=mains_config_path,
            gaminet_config_path=gaminet_config_path,
            lambda2_grid=lambda2_grid,
            n_sweep_seeds=n_sweep_seeds,
            feature_meta=feature_meta,
            X_tune=X[tune_idx],
            y_tune=y[tune_idx],
            X_tune_val=X[tune_val_idx],
            y_tune_val=y[tune_val_idx],
            out_dir=tune_dir,
        )

    mains_config = load_na2m_config(str(mains_config_path))
    val_frac_of_pool = mains_config.val_frac / (1 - mains_config.test_frac)

    for run_i in range(n_runs):
        run_seed = run_seeds[run_i]
        train_idx, val_idx = fold_inner_split(
            pool_idx, val_frac_of_pool,
            run_mode=run_mode,
            fold_split_seed=fold_split_seed,
            run_seed=run_seed,
        )
        run_arms(
            X, y, feature_meta,
            train_idx, val_idx, test_idx,
            tune_dir=tune_dir,
            run_dir=run_dir,
            run_i=run_i,
            run_seed=run_seed,
        )


# --------------------------------------------------------------------------- #
# Outer k-fold loop                                                            #
# --------------------------------------------------------------------------- #

def evaluate_na2m(
    search_config_path: str,
    X: np.ndarray,
    y: np.ndarray,
    feature_meta,
    base_dir: Path,
    *,
    run_mode: str,
    n_runs: int,
    n_folds: int,
) -> None:
    """Outer k-fold loop: load search config, derive seeds, split into folds, run each.

    Loads fixed_params and search_space once; passes them to run_fold which tunes
    per fold and constructs the NA2MConfig from the tuned result. No metric computed here.

    Args:
        search_config_path: Search YAML (fixed params + search spaces).
        X, y: Full dataset as numpy arrays.
        feature_meta: FeatureMeta list from preprocess().
        base_dir: Root output dir; run_mode subdir is created under it.
        run_mode: "fixed" | "subsample".
        n_runs: Number of runs per fold.
        n_folds: Number of outer CV folds.
    """
    assert run_mode in {"fixed", "subsample"}
    out_dir = base_dir / run_mode

    fixed_params, search_space = load_na2m_search_config(search_config_path)

    fold_split_seeds = derive_fold_split_seeds(fixed_params["seed"], n_folds)
    run_seeds        = derive_run_seeds(fixed_params["seed"], n_folds, n_runs)

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=fixed_params["seed"])
    for fold_idx, (pool_idx, test_idx) in enumerate(kf.split(X)):
        tune_dir = base_dir / f"fold_{fold_idx}"
        run_dir  = base_dir / f"fold_{fold_idx}" / run_mode
        run_fold(
            fixed_params, search_space, X, y, feature_meta,
            pool_idx, test_idx,
            tune_dir=tune_dir,
            run_dir=run_dir,
            fold_idx=fold_idx,
            run_mode=run_mode,
            n_runs=n_runs,
            fold_split_seed=fold_split_seeds[fold_idx],
            run_seeds=run_seeds[fold_idx],
        )


def main() -> None:
    # ------------------------------ PARAMS ------------------------------ #
    # --- Dataset (swap these two lines to switch) ---
    # SEARCH_CONFIG_PATH = r"C:\Users\maart\OneDrive\Documenten\Universiteit\Scriptie\python_repo\thesis-nam\configs\compas_na2m_search.yaml"
    # DATASET = CompasDataset();  DATASET_NAME = "compas"
    SEARCH_CONFIG_PATH = r"C:\Users\maart\OneDrive\Documenten\Universiteit\Scriptie\python_repo\thesis-nam\configs\california_housing_na2m_search.yaml"
    DATASET      = CaliforniaHousingDataset()
    DATASET_NAME = "california_housing"
    # ------------------------------------------------
    RUN_MODE  = "fixed"        # "fixed" | "subsample"
    N_RUNS    = 20
    N_FOLDS   = 5
    BASE_DIR  = Path(f"runs/{DATASET_NAME}_na2m")
    FRESH     = False          # True → delete BASE_DIR/<run_mode> first
    # -------------------------------------------------------------------- #

    if FRESH and (BASE_DIR / RUN_MODE).exists():
        import shutil
        shutil.rmtree(BASE_DIR / RUN_MODE)

    fixed_params, _ = load_na2m_search_config(SEARCH_CONFIG_PATH)
    df = DATASET.load(fixed_params.get("dataset_path"))
    X, y, feature_meta = DATASET.preprocess(df)

    evaluate_na2m(
        SEARCH_CONFIG_PATH, X, y, feature_meta, BASE_DIR,
        run_mode=RUN_MODE,
        n_runs=N_RUNS,
        n_folds=N_FOLDS,
    )

if __name__ == "__main__":
    main()
