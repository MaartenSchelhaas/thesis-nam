"""
tune_na2m.py — Optuna hyperparameter search for NA2M main effects (arm A).

Tunes Stage-1 hyperparameters (lr, dropout, regularization, activation) with
interactions disabled. clarity_regularization is NOT in this search space — it
is tuned separately by tune_clarity.py once the main-effects config is fixed.

Pruning works: trial is threaded into the Stage-1 Trainer so Optuna's
MedianPruner can cut bad trials early on intermediate validation metrics.

Usage:
    python scripts/na2m/tune_na2m.py
    Edit the variables at the top of main() to point at your search YAML.
"""

import numpy as np
from pathlib import Path

import optuna
import torch
import yaml
from torch.utils.data import DataLoader

from na2m.data.shared import split
from na2m.data.compas import CompasDataset
from na2m.models.na2m import NA2M
from na2m.training.fit_na2m import fit_na2m
from na2m.utils.config import NA2MConfig, load_na2m_search_config
from na2m.data.dataset import NAMDataset
from na2m.training.metrics import auroc, rmse


def suggest_hyperparams(
    trial: optuna.Trial,
    search_space: dict,
) -> dict:
    """Sample one value per entry in search_space.

    Args:
        trial: Current Optuna trial.
        search_space: Search-space spec from the YAML search_space block.

    Returns:
        Dict of sampled hyperparameter values.
    """
    params = {}
    for name, spec in search_space.items():
        if spec["type"] == "float_log":
            params[name] = trial.suggest_float(name, spec["low"], spec["high"], log=True)
        elif spec["type"] == "float":
            params[name] = trial.suggest_float(name, spec["low"], spec["high"])
        elif spec["type"] == "categorical":
            params[name] = trial.suggest_categorical(name, spec["choices"])
        elif spec["type"] == "int":
            params[name] = trial.suggest_int(name, spec["low"], spec["high"])
    return params


def objective(
    trial: optuna.Trial,
    fixed_params: dict,
    search_space: dict,
    feature_meta,
    X_train,
    y_train,
    X_val,
    y_val,
) -> float:
    """Optuna objective: sample hyperparams, run Stage 1, return val AUROC.

    trial is forwarded to fit_na2m so the Stage-1 Trainer reports intermediate
    metrics and the MedianPruner can fire.
    """
    trial_params = suggest_hyperparams(trial, search_space)

    _na2m_fields = set(NA2MConfig.__dataclass_fields__)
    config = NA2MConfig(
        **{k: v for k, v in fixed_params.items() if k in _na2m_fields},
        **trial_params,
    )

    X_pool = np.concatenate([X_train, X_val])
    y_pool = np.concatenate([y_train, y_val])

    train_loader = DataLoader(
        NAMDataset(X_train, y_train),
        batch_size=config.batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        NAMDataset(X_val, y_val),
        batch_size=config.batch_size,
        shuffle=False,
    )
    pool_loader = DataLoader(
        NAMDataset(X_pool, y_pool),
        batch_size=config.batch_size,
        shuffle=False,
    )

    model = NA2M(
        num_features=X_train.shape[1],
        feature_meta=feature_meta,
        num_units=config.num_units,
        hidden_sizes=config.hidden_sizes,
        dropout=config.dropout,
        feature_dropout=config.feature_dropout,
        activation=config.activation,
        inter_units=config.inter_units,
        inter_hidden=config.inter_hidden,
    )

    fit_na2m(
        model,
        train_loader,
        val_loader,
        pool_loader,
        config,
        with_interactions=False,
        with_concurvity_filter=False,
        trial=trial,
    )

    model.eval()
    device = next(model.parameters()).device
    all_logits, all_targets = [], []
    with torch.no_grad():
        for X_batch, y_batch, _ in val_loader:
            logits, _ = model(X_batch.to(device))
            all_logits.append(logits.cpu())
            all_targets.append(y_batch)
    logits = torch.cat(all_logits)
    targets = torch.cat(all_targets)
    if config.task == "regression":
        return float(rmse(logits, targets))
    return float(auroc(logits, targets))


def save_best_config(
    study: optuna.Study,
    fixed_params: dict,
    output_path: Path,
) -> None:
    """Write the best trial's parameters as an NA2MConfig-compatible YAML.

    clarity_regularization is kept at its fixed value (0.0) — it will be
    overwritten by tune_clarity.tune_clarity_fold in a second pass.
    """
    _na2m_fields = set(NA2MConfig.__dataclass_fields__)
    best_params = {k: v for k, v in fixed_params.items() if k in _na2m_fields}
    best_params.update(study.best_trial.params)
    best_params["clarity_regularization"] = 0.0  # placeholder; tune_clarity_fold overwrites

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        yaml.dump(best_params, f, default_flow_style=False, sort_keys=False)

    print(f"Best metric : {study.best_trial.value:.4f}")
    print(f"Best config saved to {output_path}")


def tune_fold(
    fixed_params: dict,
    search_space: dict,
    feature_meta,
    X_train,
    y_train,
    X_val,
    y_val,
    output_path: Path,
    *,
    study_name: str = "fold_search",
) -> Path:
    """Run an Optuna study on one train/val split and save the best main-effects config.

    Args:
        fixed_params: Non-tunable config fields from load_na2m_search_config.
        search_space: Search-space spec from the YAML search_space block.
        feature_meta: FeatureMeta list from preprocess().
        X_train, y_train: Training split.
        X_val, y_val: Validation split.
        output_path: Where to write the best config YAML.
        study_name: Optuna study name.

    Returns:
        output_path after saving.
    """
    direction = "minimize" if fixed_params.get("task") == "regression" else "maximize"
    study = optuna.create_study(
        study_name=study_name,
        direction=direction,
        sampler=optuna.samplers.TPESampler(),
        pruner=optuna.pruners.MedianPruner(),
    )

    study.optimize(
        lambda trial: objective(
            trial,
            fixed_params,
            search_space,
            feature_meta,
            X_train,
            y_train,
            X_val,
            y_val,
        ),
        n_trials=fixed_params["n_trials"],
    )

    save_best_config(study, fixed_params, output_path)
    return output_path


def main() -> None:
    # --- Edit these ---
    _CONFIG = r"C:\Users\maart\OneDrive\Documenten\Universiteit\Scriptie\python_repo\thesis-nam\configs\compas_na2m_search.yaml"
    DATASET = CompasDataset()
    DATASET_NAME = "compas"
    # ------------------

    fixed_params, search_space = load_na2m_search_config(_CONFIG)

    df = DATASET.load(fixed_params.get("dataset_path"))
    X, y, feature_meta = DATASET.preprocess(df)
    X_train, X_val, X_test, y_train, y_val, y_test = split(
        X,
        y,
        fixed_params["val_frac"],
        fixed_params["test_frac"],
        fixed_params.get("seed", 42),
        stratify=(DATASET.TASK == "classification"),
    )

    output_path = Path(_CONFIG).parent / f"{DATASET_NAME}_na2m_tuned.yaml"

    tune_fold(
        fixed_params,
        search_space,
        feature_meta,
        X_train,
        y_train,
        X_val,
        y_val,
        output_path,
        study_name=f"{dataset_name}_na2m_main_search",
    )


if __name__ == "__main__":
    main()