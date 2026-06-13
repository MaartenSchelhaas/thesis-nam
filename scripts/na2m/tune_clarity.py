"""
tune_clarity.py — Optuna search for NA2M clarity_regularization (arm B).

Run AFTER tune_na2m.py. Loads the tuned main-effects config, searches only
clarity_regularization over the full arm-B pipeline (Stages 1-3), and writes
the best value back into the same YAML. Arm C then reuses that same config.

No intermediate pruning: clarity_regularization has no effect during Stage 1,
so intermediate Stage-1 val metrics give no useful signal for pruning this
parameter. Every trial runs the full 3-stage pipeline to completion.

Usage:
    python scripts/na2m/tune_clarity.py
    Edit the variables at the top of main() to point at your configs.
"""

import copy
import numpy as np
from pathlib import Path

import optuna
import torch
import yaml
from torch.utils.data import DataLoader

from na2m.data.data_utils import load_compas, preprocess, split
from na2m.models.na2m import NA2M
from na2m.training.fit_na2m import fit_na2m
from na2m.utils.config import load_na2m_config
from nam.data.dataset import NAMDataset
from nam.training.metrics import auroc


def load_clarity_search_config(search_yaml_path: str) -> tuple[int, dict]:
    """Extract clarity tuning settings from the main NA2M search YAML.

    Args:
        search_yaml_path: Path to compas_na2m_search.yaml.

    Returns:
        (n_trials, search_spec) where search_spec has keys type/low/high.
    """
    with open(search_yaml_path) as f:
        raw = yaml.safe_load(f)
    return raw["clarity_n_trials"], raw["clarity_search_space"]


def objective_clarity(
    trial: optuna.Trial,
    config,
    feature_meta,
    X_train,
    y_train,
    X_val,
    y_val,
    search_spec: dict,
) -> float:
    """Sample clarity_regularization, run full arm B, return val AUROC.

    Args:
        trial: Current Optuna trial.
        config: NA2MConfig loaded from the tuned main-effects YAML.
                Copied internally so the original is not mutated.
        feature_meta: FeatureMeta list from preprocess().
        X_train, y_train: Training split.
        X_val, y_val: Validation split.
        search_spec: Dict with keys type/low/high for clarity_regularization.

    Returns:
        Validation AUROC after full arm-B training.
    """
    clarity = trial.suggest_float(
        "clarity_regularization",
        search_spec["low"],
        search_spec["high"],
        log=True,
    )

    cfg = copy.deepcopy(config)
    cfg.clarity_regularization = clarity

    X_pool = np.concatenate([X_train, X_val])
    y_pool = np.concatenate([y_train, y_val])

    train_loader = DataLoader(
        NAMDataset(X_train, y_train),
        batch_size=cfg.batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        NAMDataset(X_val, y_val),
        batch_size=cfg.batch_size,
        shuffle=False,
    )
    pool_loader = DataLoader(
        NAMDataset(X_pool, y_pool),
        batch_size=cfg.batch_size,
        shuffle=False,
    )

    model = NA2M(
        num_features=X_train.shape[1],
        feature_meta=feature_meta,
        num_units=cfg.num_units,
        hidden_sizes=cfg.hidden_sizes,
        dropout=cfg.dropout,
        feature_dropout=cfg.feature_dropout,
        activation=cfg.activation,
        inter_units=cfg.inter_units,
        inter_hidden=cfg.inter_hidden,
    )

    # Full arm-B pipeline: Stages 1 + 2 + 3.
    # trial is not forwarded: Stage-1 intermediate metrics are independent of
    # clarity_regularization, so pruning based on them would be misleading.
    fit_na2m(
        model,
        train_loader,
        val_loader,
        pool_loader,
        cfg,
        with_interactions=True,
        with_concurvity_filter=False,
    )

    model.eval()
    all_logits, all_targets = [], []
    with torch.no_grad():
        for X_batch, y_batch, _ in val_loader:
            logits, _ = model(X_batch)
            all_logits.append(logits)
            all_targets.append(y_batch)
    return float(auroc(torch.cat(all_logits), torch.cat(all_targets)))


def tune_clarity_fold(
    tuned_config_path: Path,
    n_trials: int,
    search_spec: dict,
    feature_meta,
    X_train,
    y_train,
    X_val,
    y_val,
    *,
    study_name: str = "clarity_search",
) -> Path:
    """Run Optuna over clarity_regularization and write the best value back into the YAML.

    Reads the tuned main-effects config, keeps all main-effect hyperparameters
    fixed, and searches only clarity_regularization. The best value is written
    back into tuned_config_path so it becomes the single complete config used
    by both arm B and arm C.

    Args:
        tuned_config_path: YAML written by tune_na2m.tune_fold.
                           Updated in place with best clarity_regularization.
        n_trials: Number of Optuna trials.
        search_spec: Dict with keys type/low/high for clarity_regularization.
        feature_meta: FeatureMeta list from preprocess().
        X_train, y_train: Training split (same split used for main-effects tuning).
        X_val, y_val: Validation split (same split used for main-effects tuning).
        study_name: Optuna study name.

    Returns:
        tuned_config_path after updating.
    """
    config = load_na2m_config(str(tuned_config_path))

    study = optuna.create_study(
        study_name=study_name,
        direction="maximize",
        sampler=optuna.samplers.TPESampler(),
        # No pruner: Stage-1 metrics are independent of clarity_regularization.
    )

    study.optimize(
        lambda trial: objective_clarity(
            trial,
            config,
            feature_meta,
            X_train,
            y_train,
            X_val,
            y_val,
            search_spec,
        ),
        n_trials=n_trials,
    )

    best_clarity = study.best_trial.params["clarity_regularization"]

    with open(tuned_config_path) as f:
        raw = yaml.safe_load(f)
    raw["clarity_regularization"] = best_clarity
    with open(tuned_config_path, "w") as f:
        yaml.dump(raw, f, default_flow_style=False, sort_keys=False)

    print(f"Best clarity metric       : {study.best_trial.value:.4f}")
    print(f"Best clarity_regularization={best_clarity:.6f} written to {tuned_config_path}")
    return tuned_config_path


def main() -> None:
    # --- Edit these ---
    _SEARCH_CONFIG = r"C:\Users\maart\OneDrive\Documenten\Universiteit\Scriptie\python_repo\thesis-nam\configs\compas_na2m_search.yaml"
    _TUNED_CONFIG  = r"C:\Users\maart\OneDrive\Documenten\Universiteit\Scriptie\python_repo\thesis-nam\configs\compas-scores-two-years_na2m_tuned.yaml"
    # ------------------

    n_trials, search_spec = load_clarity_search_config(_SEARCH_CONFIG)

    config = load_na2m_config(_TUNED_CONFIG)
    df = load_compas(config.dataset_path)
    X, y, feature_meta = preprocess(df)
    X_train, X_val, X_test, y_train, y_val, y_test = split(
        X,
        y,
        config.val_frac,
        config.test_frac,
        seed=getattr(config, "seed", 42),
    )

    dataset_name = Path(config.dataset_path).stem
    tune_clarity_fold(
        tuned_config_path=Path(_TUNED_CONFIG),
        n_trials=n_trials,
        search_spec=search_spec,
        feature_meta=feature_meta,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        study_name=f"{dataset_name}_clarity_search",
    )


if __name__ == "__main__":
    main()