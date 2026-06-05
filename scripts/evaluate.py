"""
K-fold cross validation with run ensembling.

Folder structure:
    base_dir/
        fold_0/
            run_0/
                best.pt
                predictions.pt
            run_1/
                best.pt
                predictions.pt
            fold_metric.json
        fold_1/
            ...
        results.json
"""

import json
import numpy as np
import torch
from pathlib import Path
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader

from nam.data.data_utils import load_compas, preprocess
from nam.data.dataset import NAMDataset
from nam.utils.config import NAMConfig
from nam.models.nam import NAM
from nam.training.metrics import auroc, rmse
from nam.utils.config import load_config, load_search_config
from scripts.train import run_single, build_model
from scripts.tune_nam import tune_fold

def load_model(config: NAMConfig, checkpoint_path: Path, num_features: int) -> NAM:
    """Load the model from stored best parameters

    Args:
        config (NAMConfig): Config file
        checkpoint_path (Path): Path to the stored model parameters
        num_features (int): Amount of features in the dataset

    Returns:
        NAM: Model filled with trained parameters, in eval mode.
    """
    model = build_model(config, num_features=num_features)
    model.load_state_dict(torch.load(checkpoint_path,weights_only=True))
    model.eval()
    return model

def predict(model: NAM, loader: DataLoader) -> torch.Tensor:
    """Forward pass to calculate logits/predictions

    Args:
        model (NAM): Trained NAM
        loader (DataLoader): Test dataloader

    Returns:
        torch.Tensor: Prediction logits
    """
    all_predictions = []
    with torch.no_grad():
        for X_batch, _, _ in loader:
            predictions, _ = model(X_batch)
            all_predictions.append(predictions)
    return torch.cat(all_predictions)


def ensemble_predictions(fold_dir: Path, n_runs: int) -> torch.Tensor:
    """Load predictions.pt from each run and average them.

    Args:
        fold_dir: fold directory containing run_X subfolders
        n_runs: number of runs to ensemble

    Returns:
        torch.Tensor: averaged logits over each run, shape (n,)
    """
    all_run_logits = []

    for i in range(n_runs):
        path = fold_dir / f'run_{i}' / 'predictions.pt'
        logits = torch.load(path, weights_only=True)
        all_run_logits.append(logits)

    return torch.stack(all_run_logits).mean(dim=0)


def evaluate_fold(ensembled_preds: torch.Tensor, targets: torch.Tensor, task: str) -> float:
    """Compute metric for one fold given ensembled predictions.
    
    Args:
        ensembled_preds: averaged logits shape (n,)
        targets: ground truth shape (n,)
        task: 'classification' or 'regression'

    Returns:
        float: AUC or RMSE
    """
    if task == "classification":
        return auroc(ensembled_preds, targets)
    else:
        return rmse(ensembled_preds, targets)




def save_fold_metric(fold_dir: Path, metric: float, n_runs:int) -> None:
    """Save fold metric to fold_metric.json.
    
    Args:
        fold_dir: fold directory
        metric: AUC or RMSE for this fold
        n_runs: number of runs used for ensembling
    """

    fold_dir.mkdir(parents=True, exist_ok=True)
    record = {"metric": metric, "n_runs": n_runs}
    with open(fold_dir / "fold_metric.json", "w") as f:
        json.dump(record, f)



def collect_results(base_dir: Path, n_folds: int, n_runs: int) -> dict:
    """Read fold_metric.json from each fold directory and aggregate.

    Args:
        base_dir: root run directory containing fold_X subdirectories
        n_folds:  expected number of folds
        n_runs:   number of runs per fold (stored in results for reference)

    Returns:
        dict with keys: fold_metrics, mean, std
    """
    fold_metrics = []
    for fold_idx in range(n_folds):
        path = base_dir / f'fold_{fold_idx}' / 'fold_metric.json'
        with open(path) as f:
            fold_metrics.append(json.load(f)['metric'])

    mean = float(np.mean(fold_metrics))
    std  = float(np.std(fold_metrics))

    record = {
        "mean": mean,
        "std": std,
        "fold_metrics": fold_metrics,
        "n_runs": n_runs,
        "n_folds": n_folds,
    }
    with open(base_dir / "results.json", "w") as f:
        json.dump(record, f)

    return {"fold_metrics": fold_metrics, "mean": mean, "std": std}


def evaluate_kfold(
    search_config_path: str,
    X, y,
    base_dir: Path,
    n_runs: int,
    n_folds: int = 5,
) -> dict:
    """
    Args:
        search_config_path: Path to the search space YAML (used for per-fold tuning).
        X, y:     Full dataset as numpy arrays.
        base_dir: Root dir e.g. runs/kfold_5fold_20runs/
        n_runs:   Number of random train/val subsamples per fold.
        n_folds:  Number of CV folds.

    Returns:
        dict with keys: fold_metrics, mean, std
    """
    from sklearn.model_selection import train_test_split

    fixed_params, search_space = load_search_config(search_config_path)
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=fixed_params["seed"])
    num_features = X.shape[1]

    for fold_idx, (train_val_idx, test_idx) in enumerate(kf.split(X)):
        print(f"\n--- Fold {fold_idx} ---")
        fold_dir = base_dir / f'fold_{fold_idx}'
        fold_dir.mkdir(parents=True, exist_ok=True)

        if (fold_dir / 'done').exists():
            print(f"Fold {fold_idx} already complete, skipping.")
            continue

        X_test, y_test = X[test_idx], y[test_idx]
        y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

        val_frac_of_trainval = fixed_params["val_frac"] / (1 - fixed_params["test_frac"])

        # Tune hyperparameters once per fold on a single train/val split
        fold_config_path = fold_dir / 'tuned_config.yaml'
        if not fold_config_path.exists():
            tune_train_idx, tune_val_idx = train_test_split(train_val_idx, test_size=val_frac_of_trainval)
            X_tune_train, y_tune_train = X[tune_train_idx], y[tune_train_idx]
            X_tune_val,   y_tune_val   = X[tune_val_idx],   y[tune_val_idx]
            tune_fold(fixed_params, search_space, X_tune_train, y_tune_train,
                      X_tune_val, y_tune_val, fold_config_path,
                      study_name=f"fold_{fold_idx}_search")
        fold_config = load_config(str(fold_config_path))

        test_loader = DataLoader(NAMDataset(X_test, y_test), batch_size=fold_config.batch_size, shuffle=False)

        #For this fold, do n_runs amount of subsamples
        for i in range(n_runs):
            run_dir = fold_dir / f'run_{i}'
            run_dir.mkdir(parents=True, exist_ok=True)

            if (run_dir / 'done').exists():
                print(f"  Run {i} already complete, skipping.")
                continue

            train_idx, val_idx = train_test_split(train_val_idx, test_size=val_frac_of_trainval)
            X_train, y_train = X[train_idx], y[train_idx]
            X_val,   y_val   = X[val_idx],   y[val_idx]

            run_single(fold_config, X_train, y_train, X_val, y_val, run_dir)

            model = load_model(fold_config, run_dir / 'best.pt', num_features)
            preds = predict(model, test_loader)
            torch.save(preds, run_dir / 'predictions.pt')
            (run_dir / 'done').touch()

        ensembled_preds = ensemble_predictions(fold_dir, n_runs)
        metric = evaluate_fold(ensembled_preds=ensembled_preds, targets=y_test_tensor, task=fold_config.task)
        save_fold_metric(fold_dir, metric, n_runs)
        print(f"Fold {fold_idx} metric: {metric:.4f}")
        (fold_dir / 'done').touch()

    results = collect_results(base_dir, n_folds, n_runs)
    print(f"\nFinal: mean={results['mean']:.4f}, std={results['std']:.4f}")
    return results


if __name__ == '__main__':
    SEARCH_CONFIG_PATH = r"configs/compas_search.yaml"
    BASE_DIR           = Path("runs/kfold_5fold_20runs")
    N_RUNS             = 5
    N_FOLDS            = 5
    FRESH              = False  # set True to delete BASE_DIR and start over

    if FRESH and BASE_DIR.exists():
        import shutil
        shutil.rmtree(BASE_DIR)
        print(f"Deleted {BASE_DIR} for fresh run.")

    fixed_params, _ = load_search_config(SEARCH_CONFIG_PATH)
    df = load_compas(fixed_params["dataset_path"])
    X, y, _ = preprocess(df)

    results = evaluate_kfold(
        search_config_path=SEARCH_CONFIG_PATH,
        X=X, y=y,
        base_dir=BASE_DIR,
        n_runs=N_RUNS,
        n_folds=N_FOLDS,
    )