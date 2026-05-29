"""
NAMConfig — central hyperparameter store.

A single dataclass passed into the model, trainer, and loss function
so all settings live in one place and can be serialised to YAML/JSON
for experiment reproducibility.

To run an experiment: edit configs/compas_baseline.yaml and pass it
to scripts/train.py — the trainer loads it into a NAMConfig instance.
"""

from dataclasses import dataclass, field
import yaml

@dataclass
class NAMConfig:
    dataset_path: str = ""

    # --- Model architecture ---
    num_units: int = 64
    # Width of the activation layer (ExU/LinReLU) in each FeatureNN.

    hidden_sizes: list = field(default_factory=lambda: [64, 32])
    # Widths of hidden Dense layers after the activation layer.
    # Empty list → shallow (activation + output only).

    activation: str = "exu"
    # Activation layer type: 'exu' or 'relu'.

    dropout: float = 0.5
    # Dropout probability applied after each hidden layer in FeatureNN.

    feature_dropout: float = 0.0
    # Probability of dropping an entire feature output before summation in NAM.
    # 0.0 = no feature dropout.

    # --- Regularisation ---
    output_regularization: float = 0.0
    # Penalty weight on the squared magnitude of individual feature outputs.
    # Encourages sparse / small per-feature contributions.

    l2_regularization: float = 0.0
    # L2 weight decay applied to all model parameters.

    # --- Optimiser & schedule ---
    lr: float = 1e-3
    # Initial Adam learning rate.

    decay_rate: float = 0.995
    # Multiplicative LR decay applied every epoch via StepLR(gamma=decay_rate).

    # --- Data split ---
    val_frac: float = 0.15
    test_frac: float = 0.15
    seed: int = 42

    # --- Training loop ---
    batch_size: int = 1024
    num_epochs: int = 1000
    patience: int = 60
    # Early stopping: stop if val metric does not improve for this many epochs.

    val_check_interval: int = 10
    # Evaluate on validation set every N epochs.

    # --- Task ---
    task: str = "classification"
    # 'classification' → binary cross-entropy loss + AUROC metric
    # 'regression'     → MSE loss + RMSE metric

def load_config(path: str) -> NAMConfig:
    with open(path) as f:
        raw = yaml.safe_load(f)
    return NAMConfig(**raw)