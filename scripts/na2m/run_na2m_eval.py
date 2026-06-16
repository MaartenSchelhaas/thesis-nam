"""
run_na2m_eval.py — the NA2M k-fold × n_runs evaluation ORCHESTRATOR.

Owns WHERE the model runs and HOW outputs are laid out: fold/run loops, the
train/val split policy, seed derivation, folder layout. The actual model runs +
measure extraction live in model_runner.py (run_main_effects / load_main_effects /
run_arm); pure compute lives in src/na2m. This file never builds or holds a model.

Modular layering (each level does one thing, top → bottom):

    evaluate_na2m   outer k-fold loop: derive seeds, split into folds, call run_fold
      └ run_fold    everything within ONE fold: tune + store config once, loop runs
          └ run_arms    one run: train mains ONCE, branch all arms (run_arm × N)
              └ fold_inner_split   resolve train/val split for the run (fixed | subsample)

Two user-facing axes (PARAMETERS, set in main(); never NA2MConfig fields):
    RUN_MODE  "fixed"     → one train/val split per fold (split contract; stability)
              "subsample" → fresh split per run (Agarwal-style ensemble)
    arms      all three run every (fold, run): the mains are trained once and arms
              B/C branch off them — see train._ARM_FLAGS.

Extraction is owned by model_runner: run_main_effects / run_arm extract + persist
each arm's measures while the model is in memory, then write a per-arm `done`
sentinel LAST. This file only checks those sentinels to drive resume — it never
reloads a model to extract from it.

Resume contract (granular, NOT coarse fold/run flags):
    - tuned_config.yaml present  → skip tuning for the fold.
    - mains/done present         → skip run_main_effects; just load_main_effects.
    - <arm>/done present         → skip that arm.
This lets an interrupted run continue AND a newly-added arm get picked up on a
rerun of an otherwise-complete run.

Output layout:
    BASE_DIR/<run_mode>/fold_k/
        tuned_config.yaml
        run_i/
            mains/       model.pt, measures.pt, done   (arm A == the mains model)
            gaminet/     measures.pt, done             (arm B)
            concurvity/  measures.pt, done             (arm C)
        done
"""

from pathlib import Path

import numpy as np
from sklearn.model_selection import KFold, train_test_split

from na2m.data.data_utils import load_compas, preprocess
from na2m.utils.config import NA2MConfig, load_na2m_config, load_na2m_search_config
from scripts.na2m.model_runner import (
    run_main_effects,
    load_main_effects,
    run_arm,
    _ARM_FLAGS,
)
from scripts.na2m.tune_na2m import tune_fold
from scripts.na2m.tune_clarity import load_clarity_search_config, tune_clarity_fold


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

    TODO:
        - root = np.random.SeedSequence(master).
        - per fold_ss in root.spawn(n_folds): spawn n_runs children,
          int(child.generate_state(1)[0]) each.
    """
    raise NotImplementedError


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

    TODO:
        - root = np.random.SeedSequence(master + 1).
        - [int(ss.generate_state(1)[0]) for ss in root.spawn(n_folds)].
    """
    raise NotImplementedError


# --------------------------------------------------------------------------- #
# Per-fold tuning                                                              #
# --------------------------------------------------------------------------- #

def tune_fold_two_stage(
    search_config_path: str,
    feature_meta,
    X_tune: np.ndarray,
    y_tune: np.ndarray,
    X_tune_val: np.ndarray,
    y_tune_val: np.ndarray,
    tuned_config_path: Path,
    *,
    fold_idx: int,
) -> Path:
    """Tune one fold: Stage 1 mains always; Stage 2 clarity (single λ for B and C).

    Both stages tune on the SAME fixed inner split (passed in). Thin wrapper over
    tune_fold (Stage 1, pruning on) and tune_clarity_fold (Stage 2, pruning off).
    Writes the complete tuned config (mains hp + λ) that every arm loads.

    Args:
        search_config_path: Search YAML with both search spaces + trial budgets.
        feature_meta: FeatureMeta list from preprocess().
        X_tune, y_tune: Fixed inner train split.
        X_tune_val, y_tune_val: Fixed inner val split.
        tuned_config_path: Stage 1 writes here; Stage 2 rewrites λ in place.
        fold_idx: For unique Optuna study names.

    Returns:
        tuned_config_path after both stages.

    TODO:
        - fixed_params, search_space = load_na2m_search_config(search_config_path).
        - tune_fold(fixed_params, search_space, feature_meta, X_tune, y_tune,
                    X_tune_val, y_tune_val, tuned_config_path,
                    study_name=f"fold_{fold_idx}_main_search").
        - n_trials, spec = load_clarity_search_config(search_config_path).
        - tune_clarity_fold(tuned_config_path, n_trials, spec, feature_meta,
                            X_tune, y_tune, X_tune_val, y_tune_val,
                            study_name=f"fold_{fold_idx}_clarity_search").
        - return tuned_config_path.
    """
    raise NotImplementedError


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

    TODO:
        - rs = fold_split_seed if run_mode == "fixed" else run_seed.
        - return train_test_split(pool_idx, test_size=val_frac_of_pool, random_state=rs).
    """
    raise NotImplementedError


# --------------------------------------------------------------------------- #
# One run: train mains once, branch all arms                                   #
# --------------------------------------------------------------------------- #

def run_arms(
    config: NA2MConfig,
    X: np.ndarray,
    y: np.ndarray,
    feature_meta,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    run_dir: Path,
    run_seed: int,
) -> None:
    """One run: ensure the mains exist, then produce every interaction arm — resumably.

    Mains are deterministic given (split, run_seed), so they are trained once
    (run_main_effects, which also extracts arm A), persisted, and every interaction
    arm branches off them via run_arm. All arms use the same run_seed so B and C are
    paired — only the gate differs. Each model_runner call extracts + flags its own
    `done`; this function only checks those sentinels to skip completed work, so an
    interrupted run continues and a newly-added arm is picked up on rerun.

    The fold's 80% pool is train_idx ∪ val_idx, so model_runner reconstructs it from
    the train/val slices — no pool_idx needed here.

    Args:
        config: NA2MConfig loaded from the fold's tuned_config.yaml.
        X, y: Full dataset as numpy arrays.
        feature_meta: FeatureMeta list from preprocess().
        train_idx, val_idx: This run's train / val indices (from fold_inner_split).
        test_idx: The fold's test slice (measure extraction inside model_runner).
        run_dir: Output dir for this run (per-arm subdirs live under it).
        run_seed: Seed for mains + all arms of this run.

    TODO:
        - mains_dir = run_dir / "mains".
        - if not (mains_dir / "done").exists():
              run_main_effects(config, feature_meta, X[train_idx], y[train_idx],
                               X[val_idx], y[val_idx], X[test_idx], y[test_idx],
                               run_seed, mains_dir).
        - mains = load_main_effects(config, feature_meta, X.shape[1], mains_dir / "model.pt").
        - for arm, (wi, wc) in _ARM_FLAGS.items():
              if not wi: continue            # arm A handled by run_main_effects
              arm_dir = run_dir / arm
              if (arm_dir / "done").exists(): continue
              run_arm(config, mains, feature_meta, X[train_idx], y[train_idx],
                      X[val_idx], y[val_idx], X[test_idx], y[test_idx],
                      run_seed, arm_dir, with_interactions=wi, with_concurvity_filter=wc).
    """
    raise NotImplementedError


# --------------------------------------------------------------------------- #
# One fold: tune + store config, then loop runs                                #
# --------------------------------------------------------------------------- #

def run_fold(
    config: NA2MConfig,
    search_config_path: str | None,
    X: np.ndarray,
    y: np.ndarray,
    feature_meta,
    pool_idx: np.ndarray,
    test_idx: np.ndarray,
    fold_dir: Path,
    *,
    fold_idx: int,
    run_mode: str,
    n_runs: int,
    fold_split_seed: int,
    run_seeds: list[int],
) -> None:
    """Everything that happens within ONE fold.

    1. Tune + STORE the hyperparameter config once (on the fold's fixed inner
       split), so every run in this fold trains with the same hyperparameters.
    2. Loop n_runs: resolve the split (fold_inner_split), then run_arms.

    Args:
        config: Base NA2MConfig (overwritten by the fold's tuned config if tuning).
        search_config_path: Search YAML for tuning; None → use `config` as-is.
        X, y: Full dataset as numpy arrays.
        feature_meta: FeatureMeta list from preprocess().
        pool_idx: This fold's 80% pool indices.
        test_idx: This fold's test slice indices.
        fold_dir: This fold's output dir.
        fold_idx: Fold number (study names / logging).
        run_mode: "fixed" | "subsample".
        n_runs: Number of runs in this fold.
        fold_split_seed: This fold's split seed (keys the fixed inner split + tuning).
        run_seeds: The n_runs init seeds for this fold (run_seeds[fold]).

    TODO:
        - skip whole fold if fold_dir/"done" exists.
        - val_frac_of_pool = config.val_frac / (1 - config.test_frac).
        - # Hyperparameter config — done & stored ONCE at fold start:
          fixed inner split via fold_inner_split(pool_idx, ..., run_mode="fixed",
              fold_split_seed=fold_split_seed, run_seed=fold_split_seed)  # fixed for tuning
          tuned_config_path = fold_dir / "tuned_config.yaml"
          if search_config_path: tune_fold_two_stage(... X[tune], ... tuned_config_path,
              fold_idx=fold_idx); config = load_na2m_config(tuned_config_path).
        - for run_i, run_seed in enumerate(run_seeds):
              run_dir = fold_dir / f"run_{run_i}"; skip if run_dir/"done".
              train_idx, val_idx = fold_inner_split(pool_idx, val_frac_of_pool,
                  run_mode=run_mode, fold_split_seed=fold_split_seed, run_seed=run_seed).
              run_arms(config, X, y, feature_meta, train_idx, val_idx,
                       test_idx, run_dir, run_seed).
        - touch fold_dir/"done".
    """
    raise NotImplementedError


# --------------------------------------------------------------------------- #
# Outer k-fold loop                                                            #
# --------------------------------------------------------------------------- #

def evaluate_na2m(
    config: NA2MConfig,
    search_config_path: str | None,
    X: np.ndarray,
    y: np.ndarray,
    feature_meta,
    base_dir: Path,
    *,
    run_mode: str,
    n_runs: int,
    n_folds: int,
    seed: int,
) -> None:
    """Outer k-fold loop: derive seeds, split into folds, run each fold.

    Runs ALL arms (mains trained once per run, B/C branch off it). No metric is
    computed here.

    Args:
        config: Base NA2MConfig (model/training hp only).
        search_config_path: Search YAML for per-fold tuning; None → skip tuning.
        X, y: Full dataset as numpy arrays.
        feature_meta: FeatureMeta list from preprocess().
        base_dir: Root output dir; run_mode is appended to the path.
        run_mode: "fixed" | "subsample".
        n_runs: Number of runs per fold.
        n_folds: Number of outer CV folds.
        seed: The single master seed; folds + per-run init seeds derive from it.

    TODO:
        - assert run_mode in {"fixed", "subsample"}.
        - out_dir = base_dir / run_mode.
        - fold_split_seeds = derive_fold_split_seeds(seed, n_folds);
          run_seeds        = derive_run_seeds(seed, n_folds, n_runs).
        - KFold(n_folds, shuffle=True, random_state=config.fold_seed).split(X)
          → (pool_idx, test_idx) per fold.
        - for fold_idx, (pool_idx, test_idx):
              run_fold(config, search_config_path, X, y, feature_meta, pool_idx,
                       test_idx, fold_dir=out_dir / f"fold_{fold_idx}",
                       fold_idx=fold_idx, run_mode=run_mode, n_runs=n_runs,
                       fold_split_seed=fold_split_seeds[fold_idx],
                       run_seeds=run_seeds[fold_idx]).
    """
    raise NotImplementedError


def main() -> None:
    # ------------------------------ PARAMS ------------------------------ #
    SEARCH_CONFIG_PATH = r"C:\Users\maart\OneDrive\Documenten\Universiteit\Scriptie\python_repo\thesis-nam\configs\compas_na2m_search.yaml"
    RUN_MODE  = "fixed"        # "fixed" | "subsample"
    N_RUNS    = 20
    N_FOLDS   = 5
    SEED      = 42             # master seed: folds + per-run init seeds derive from it
    BASE_DIR  = Path("runs/na2m_eval")
    FRESH     = False          # True → delete BASE_DIR/<run_mode> first
    # -------------------------------------------------------------------- #

    # TODO (wiring only):
    #   - if FRESH and (BASE_DIR / RUN_MODE).exists(): shutil.rmtree(it).
    #   - fixed_params, _ = load_na2m_search_config(SEARCH_CONFIG_PATH).
    #   - df = load_compas(fixed_params["dataset_path"]); X, y, feature_meta = preprocess(df).
    #   - config = NA2MConfig(**non-search fields)  OR load a base tuned yaml.
    #   - evaluate_na2m(config, SEARCH_CONFIG_PATH, X, y, feature_meta, BASE_DIR,
    #         run_mode=RUN_MODE, n_runs=N_RUNS, n_folds=N_FOLDS, seed=SEED).
    raise NotImplementedError


if __name__ == "__main__":
    main()
