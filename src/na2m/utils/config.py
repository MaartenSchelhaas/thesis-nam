"""
NA2MConfig — central hyperparameter store for the NA2M extension.

Kept SEPARATE from nam.utils.config.NAMConfig: the two overlap on the
optimizer/training fields but NA2M adds a substantial surface (interaction
subnets, staged selection/pruning, the concurvity filter, and the k-fold × seed
× arm evaluation harness). Sharing one dataclass would force NAM-specific config
to carry na2m-only fields, so we fork.

Shared model-agnostic primitives (NAMDataset, metrics, load_compas/split) are
imported from `nam`; only the configuration diverges here.

Defaults below are placeholders — tune via configs/*.yaml and load_na2m_config().
"""

from dataclasses import dataclass, field

import yaml


@dataclass
class NA2MConfig:
    dataset_path: str = ""

    # ------------------------------------------------------------------ #
    # Shared subnet architecture                                           #
    # (applies to both main-effect and interaction subnets)               #
    # ------------------------------------------------------------------ #

    num_units: int = 64
    # Width of the activation layer (ExU/LinReLU).
    # Interaction subnets use Linear(2, num_units) as their input layer.

    hidden_sizes: list = field(default_factory=lambda: [64, 32])
    # Widths of hidden Dense layers after the activation layer.
    # Empty list → shallow (activation layer + output only).

    activation: str = "exu"
    # Activation layer type: 'exu' or 'relu'.

    dropout: float = 0.5
    # Dropout probability after each hidden layer.

    feature_dropout: float = 0.0
    # Probability of zeroing an entire term (main or interaction) before
    # the additive summation in NA2M. 0.0 = disabled.

    # ------------------------------------------------------------------ #
    # Regularisation                                                       #
    # ------------------------------------------------------------------ #

    output_regularization: float = 0.0
    # Penalty on squared per-term outputs (mains + interactions).

    l2_regularization: float = 0.0
    # L2 weight decay applied to all model parameters.

    clarity_regularization: float = 0.0
    # Coefficient λ on the GAMI-Net marginal-clarity penalty (stage 3).
    # The penalty enforces E[g_ij(x_i, x_j) | x_i] ≈ 0 and
    # E[g_ij(x_i, x_j) | x_j] ≈ 0, keeping interactions zero-mean in each
    # marginal so they don't absorb main-effect signal.
    # 0.0 = disabled (pure NAM behaviour, no interaction correction).

    # ------------------------------------------------------------------ #
    # Optimiser & schedule                                                 #
    # ------------------------------------------------------------------ #

    lr: float = 1e-3
    # Initial Adam learning rate.

    decay_rate: float = 0.995
    # Multiplicative LR decay per epoch (StepLR, gamma=decay_rate).

    # ------------------------------------------------------------------ #
    # Data split                                                           #
    # ------------------------------------------------------------------ #

    val_frac: float = 0.15
    test_frac: float = 0.15
    seed: int = 42

    # ------------------------------------------------------------------ #
    # Training loop                                                        #
    # ------------------------------------------------------------------ #

    batch_size: int = 1024
    num_epochs: int = 1000
    patience: int = 60
    # Early stopping patience (in epochs, checked every val_check_interval).

    val_check_interval: int = 10
    # Validate every N epochs.

    # ------------------------------------------------------------------ #
    # Staged interaction selection & pruning (stages 2–4)                  #
    # ------------------------------------------------------------------ #

    top_m: int = 10
    # Number of top FAST-ranked candidate pairs added before pruning.

    eta_prune: float = 0.0
    # Cumulative validation-loss threshold η for the stage-2 sweep that
    # prunes weak interactions after block training.
    # 0.0 = keep all top_m pairs (no sweep pruning).

    block_train_epochs: int = 100
    # Epochs for the interaction-only block-training step (stage 2,
    # main-effect weights frozen).

    finetune_epochs: int = 100
    # Epochs for the joint fine-tuning pass (stage 3, and each stage-4
    # re-fine-tune after a concurvity removal).

    # ------------------------------------------------------------------ #
    # Concurvity filter (stage 4)                                          #
    # ------------------------------------------------------------------ #

    concurvity_threshold: float = 0.5
    # Remove the worst-offending interaction pair while any pairwise
    # concurvity statistic exceeds this value.

    max_concurvity_iters: int = 10
    # Hard cap on remove-and-refit iterations.

    # ------------------------------------------------------------------ #
    # Evaluation harness                                                   #
    # ------------------------------------------------------------------ #

    k_folds: int = 5
    fold_seed: int = 42
    # Seed for the outer k-fold split (fixed across runs → same folds).

    seeds: list = field(default_factory=lambda: [0, 1, 2, 3, 4])
    # Per-fold replicate seeds (controls optimisation variance; one per fold).

    grid_size: int = 100
    # G: number of grid points per numerical feature for shape-curve
    # extraction and the FAST interaction-strength approximation.

    # ------------------------------------------------------------------ #
    # Task                                                                 #
    # ------------------------------------------------------------------ #

    task: str = "classification"
    # 'classification' → binary cross-entropy + AUROC
    # 'regression'     → MSE + RMSE


def load_na2m_config(path: str) -> NA2MConfig:
    """Load an NA2MConfig from a YAML file.

    Args:
        path: Path to the YAML config file.

    Returns:
        Populated NA2MConfig instance.
    """
    with open(path) as f:
        raw = yaml.safe_load(f)
    return NA2MConfig(**raw)


def load_na2m_search_config(path: str) -> tuple[dict, dict]:
    """Load a tuning config: fixed settings + an Optuna search space.

    The YAML carries a top-level `search_space` block (same schema as
    NAM's load_search_config) that is popped out from the fixed settings.

    Args:
        path: Path to the search YAML.

    Returns:
        (fixed_settings, search_space) dicts.
    """
    with open(path) as f:
        raw = yaml.safe_load(f)
    search_space = raw.pop("search_space", {})
    return raw, search_space