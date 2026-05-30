"""
K-fold cross validation with optional seed ensembling.

Folder structure:
    base_dir/
        fold_0/
            seed_42/
                best.pt
                predictions.pt
            seed_7/
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
from nam.utils.config import load_config
from scripts.train import run_single, build_model

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
    model.load_state_dict(torch.load(checkpoint_path))
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


def ensemble_predictions(fold_dir: Path, seeds: list[int]) -> torch.Tensor:
    """Load predictions.pt from each seed and average them.
    
    Args:
        seed_dir: fold directory containing seed_X subfolders
        seeds: list of seeds to ensemble

    Returns:
        torch.Tensor: averaged logits over each seperate seed, shape (n,)
    """
    #Go for a given fold past each seed, average logits
    all_seed_logits = []

    for seed in seeds:
        path = fold_dir / f'seed_{seed}' / 'predictions.pt'
        logits = torch.load(path)
        all_seed_logits.append(logits)

    return torch.stack(all_seed_logits).mean(dim=0)   


def evaluate_fold(ensembled_preds: torch.Tensor, targets: torch.Tensor, task: str) -> float:
    """Compute metric for one fold given ensembled predictions.
    
    Args:
        ensembled_preds: averaged logits shape (n,)
        targets: ground truth shape (n,)
        task: 'classification' or 'regression'

    Returns:
        float: AUC or RMSE
    """
    # TODO: compute and return auroc or rmse depending on task
    raise NotImplementedError


def save_fold_metric(fold_dir: Path, metric: float, seeds: list[int]) -> None:
    """Save fold metric to fold_metric.json.
    
    Args:
        fold_dir: fold directory
        metric: AUC or RMSE for this fold
        seeds: seeds used for ensembling
    """
    # TODO: build dict with metric, seeds, save as fold_metric.json
    raise NotImplementedError


def save_results(base_dir: Path, fold_metrics: list[float], seeds: list[int], n_folds: int) -> None:
    """Save overall results to results.json.
    
    Args:
        base_dir: run directory
        fold_metrics: list of per-fold metrics
        seeds: seeds used
        n_folds: number of folds
    """
    # TODO: build dict with mean, std, fold_metrics, seeds, n_folds
    # TODO: save as base_dir / 'results.json'
    raise NotImplementedError


def evaluate_kfold(
    config,
    X, y,
    base_dir: Path,
    seeds: list[int],
    n_folds: int = 5,
) -> dict:
    """
    Args:
        config:   NAMConfig
        X, y:     Full dataset as numpy arrays
        base_dir: Root dir e.g. runs/kfold_5fold_1seed/
        seeds:    e.g. [42] or [42, 7]
        n_folds:  Number of CV folds

    Returns:
        dict with keys: fold_metrics, mean, std
    """
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=config.seed)
    fold_metrics = []

    for fold_idx, (train_val_idx, test_idx) in enumerate(kf.split(X)):
        print(f"\n--- Fold {fold_idx} ---")

        fold_dir = base_dir / f'fold_{fold_idx}'

        # TODO: slice X, y into X_test, y_test using test_idx
        # TODO: split train_val_idx into train and val using config.val_frac
        # TODO: build test_loader

        for seed in seeds:
            seed_dir = fold_dir / f'seed_{seed}'
            # TODO: call run_single with correct splits, seed_dir, seed
            # TODO: get predictions with predict() on test_loader
            # TODO: save predictions to seed_dir / 'predictions.pt'

        # TODO: call ensemble_predictions to average across seeds
        # TODO: call evaluate_fold to get metric
        # TODO: call save_fold_metric
        
        print(f"Fold {fold_idx} metric: {metric:.4f}")
        fold_metrics.append(metric)

    # TODO: call save_results
    # TODO: compute mean, std, print

    return {'fold_metrics': fold_metrics, 'mean': mean, 'std': std}


if __name__ == '__main__':
    CONFIG_PATH = r"configs/compas-scores-two-years_tuned.yaml"
    BASE_DIR    = Path("runs/kfold_5fold_1seed")
    SEEDS       = [42]
    N_FOLDS     = 5

    config = load_config(CONFIG_PATH)
    df = load_compas(config.dataset_path)
    X, y, _ = preprocess(df)

    results = evaluate_kfold(
        config=config,
        X=X, y=y,
        base_dir=BASE_DIR,
        seeds=SEEDS,
        n_folds=N_FOLDS,
    )