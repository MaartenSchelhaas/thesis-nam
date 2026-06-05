import random
import numpy as np
import torch
from torch.utils.data import DataLoader

from nam.models.nam import NAM
from nam.data.data_utils import load_compas, preprocess, split
from nam.data.dataset import NAMDataset
from nam.training.trainer import Trainer
from nam.utils.config import NAMConfig, load_config

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

# --- Config ---
#config = load_config(r"C:\Users\maart\OneDrive\Documenten\Universiteit\Scriptie\python_repo\thesis-nam\configs\compas_baseline.yaml")

config = load_config(r"configs\compas-scores-two-years_tuned.yaml")

set_seed(config.seed)

# --- Data ---
df = load_compas(config.dataset_path)
X, y, feature_names = preprocess(df)
X_train, X_val, X_test, y_train, y_val, y_test = split(
    X, y, config.val_frac, config.test_frac, config.seed
)

train_loader = DataLoader(NAMDataset(X_train, y_train), batch_size=config.batch_size, shuffle=True)
val_loader   = DataLoader(NAMDataset(X_val,   y_val),   batch_size=config.batch_size, shuffle=False)
test_loader  = DataLoader(NAMDataset(X_test,  y_test),  batch_size=config.batch_size, shuffle=False)

# --- Model ---
model = NAM(
    num_features=X_train.shape[1],
    num_units=config.num_units,
    hidden_sizes=config.hidden_sizes,
    dropout=config.dropout,
    feature_dropout=config.feature_dropout,
    activation=config.activation,
)

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

# --- Evaluate ---
test_metric = trainer.evaluate(test_loader)
print(f"Test {config.task} metric: {test_metric:.4f}")