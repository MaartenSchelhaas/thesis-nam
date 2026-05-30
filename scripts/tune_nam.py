"""
tune_nam.py — Optuna hyperparameter search for NAM.

Reads a search-space YAML (see configs/compas_search.yaml) that combines
fixed NAMConfig fields with a `search_space` section defining how each
tunable parameter should be sampled. Runs an Optuna study (TPE sampler +
MedianPruner) and writes the best found configuration to a YAML file
compatible with scripts/train.py.

Usage:
    python scripts/tune_nam.py
    Change _CONFIG for different dataset.
"""

from pathlib import Path

import optuna
import yaml
from torch.utils.data import DataLoader

from nam.data.data_utils import load_compas, preprocess, split
from nam.data.dataset import NAMDataset
from nam.models.nam import NAM
from nam.training.trainer import Trainer
from nam.utils.config import load_search_config


def suggest_hyperparams(trial: optuna.Trial, search_space: dict) -> dict:
    """Sample one value per search-space entry using the appropriate Optuna suggest call."""
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
    X_train,
    y_train,
    X_val,
    y_val,
) -> float:
    """Optuna objective: sample hyperparams, train NAM, return validation metric."""
    trial_params = suggest_hyperparams(trial, search_space)

    model = NAM(
        num_features=X_train.shape[1],
        num_units=fixed_params["num_units"],
        hidden_sizes=fixed_params["hidden_sizes"],
        dropout=trial_params["dropout"],
        feature_dropout=trial_params["feature_dropout"],
        activation=trial_params["activation"],
    )

    train_loader = DataLoader(
        NAMDataset(X_train, y_train),
        batch_size=fixed_params["batch_size"],
        shuffle=True,
    )
    val_loader = DataLoader(
        NAMDataset(X_val, y_val),
        batch_size=fixed_params["batch_size"],
        shuffle=False,
    )

    trainer = Trainer(
        model=model,
        lr=trial_params["lr"],
        decay_rate=fixed_params["decay_rate"],
        output_regularization=trial_params["output_regularization"],
        l2_regularization=trial_params["l2_regularization"],
        task=fixed_params["task"],
        num_epochs=fixed_params["num_epochs"],
        patience=fixed_params["patience"],
        val_check_interval=fixed_params["val_check_interval"],
    )

    trainer.train(train_loader, val_loader, trial)
    return trainer.best_val_metric


def save_best_config(study: optuna.Study, fixed_params: dict, output_path: Path):
    """Write the best trial's parameters as a NAMConfig-compatible YAML."""
    best_params = {**fixed_params, **study.best_trial.params}
    best_params.pop("n_trials", None)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        yaml.dump(best_params, f, default_flow_style=False, sort_keys=False)

    print(f"Best metric : {study.best_trial.value:.4f}")
    print(f"Best config saved to {output_path}")

_CONFIG = r"C:\Users\maart\OneDrive\Documenten\Universiteit\Scriptie\python_repo\thesis-nam\configs\compas_search.yaml"

def main():
    # --- Load config ---
    fixed_params, search_space = load_search_config(_CONFIG)

    # --- Data  ---
    df = load_compas(fixed_params["dataset_path"])
    X, y, _ = preprocess(df)
    X_train, X_val, X_test, y_train, y_val, y_test = split(
        X, y, fixed_params["val_frac"], fixed_params["test_frac"], fixed_params["seed"]
    )

    # --- Study ---
    dataset_name = Path(fixed_params["dataset_path"]).stem
    db_path = Path("runs/optuna") / f"{dataset_name}.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)

    direction = "maximize" if fixed_params["task"] == "classification" else "minimize"

    study = optuna.create_study(
        study_name=f"{dataset_name}_search",
        storage=f"sqlite:///{db_path}",
        load_if_exists=True,
        direction=direction,
        sampler=optuna.samplers.TPESampler(),
        pruner=optuna.pruners.MedianPruner(),
    )

    study.optimize(
        lambda trial: objective(
            trial, fixed_params, search_space, X_train, y_train, X_val, y_val
        ),
        n_trials=fixed_params["n_trials"],
    )

    # --- Save best config ---
    output_path = Path(_CONFIG).parent / f"{dataset_name}_tuned.yaml"
    save_best_config(study, fixed_params, output_path)


if __name__ == "__main__":
    main()