"""
Training entry point
Train a single NAM model for a given data split and seed.
Saves model weights to run_dir/best.pt.
"""

import random
import numpy as np
import torch
from pathlib import Path
from torch.utils.data import DataLoader

from nam.models.nam import NAM
from nam.data.data_utils import load_compas, preprocess, split
from nam.data.dataset import NAMDataset
from nam.training.trainer import Trainer
from nam.utils.config import load_config


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def build_model(config, num_features: int) -> NAM:
    model = NAM(
    num_features=num_features,
    num_units=config.num_units,
    hidden_sizes=config.hidden_sizes,
    dropout=config.dropout,
    feature_dropout=config.feature_dropout,
    activation=config.activation,
    )
    return model

def run_single(
    config,
    X_train, y_train,
    X_val,   y_val,
    run_dir: Path,
    seed: int | None = None
) -> None:
    """
    Core unit of work: one seed, one data split.
    Reused by kfold.

    Returns:
        float: test metric
    """
    if seed is not None:
        set_seed(seed)

    run_dir.mkdir(parents=True, exist_ok=True)

    train_loader = DataLoader(NAMDataset(X_train, y_train), batch_size=config.batch_size, shuffle=True)
    val_loader   = DataLoader(NAMDataset(X_val,   y_val),   batch_size=config.batch_size, shuffle=False)

    num_features = X_train.shape[1]
    model = build_model(config,num_features)

    # --- Train ---
    trainer = Trainer(
        model=model,
        lr=config.lr,
        decay_rate=config.decay_rate,
        output_regularization=config.output_regularization,
        l2_regularization=config.l2_regularization,
        task=config.task,
        num_epochs=config.num_epochs,
        patience=config.patience,
        val_check_interval=config.val_check_interval,
    )

    trainer.train(train_loader, val_loader)
    trainer.save_model(run_dir / 'best.pt')

if __name__ == '__main__':
    CONFIG_PATH = r"configs/compas-scores-two-years_tuned.yaml"
    config = load_config(CONFIG_PATH)

    RUN_DIR = Path("runs/manual_run")
    seed = config.seed

    df = load_compas(config.dataset_path)
    X, y, _ = preprocess(df)
    X_train, X_val, X_test, y_train, y_val, y_test = split(
        X, y, config.val_frac, config.test_frac, config.seed
    )

    run_single(config, X_train, y_train, X_val, y_val, run_dir=RUN_DIR, seed=seed)

