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

    # --- Main bank architecture (FeatureNN for num, CategNet for cat) ---
    num_units: int = 64
    # Width of the main FeatureNN activation layer.

    hidden_sizes: list = field(default_factory=lambda: [64, 32])
    # Hidden Dense widths after the activation layer in each main FeatureNN.

    activation: str = "exu"
    # Main FeatureNN activation: 'exu' or 'relu'.

    dropout: float = 0.5
    # Dropout inside main subnets.

    feature_dropout: float = 0.0
    # Probability of dropping an entire term (main or interaction) before summation.

    # --- Interaction subnet architecture (InteractionNN, 2-input ReLU MLP) ---
    inter_units: int = 64
    # Width of the first InteractionNN layer (Linear(2, inter_units)).

    inter_hidden: list = field(default_factory=lambda: [64, 32])
    # Hidden Dense widths after the first layer in each InteractionNN.

    inter_dropout: float = 0.5
    # Dropout inside interaction subnets.

    # --- Regularisation ---
    output_regularization: float = 0.0
    # Penalty on squared per-term outputs.

    l2_regularization: float = 0.0
    # L2 weight decay.

    clarity_regularization: float = 0.0
    # Coefficient on the GAMI-Net marginal-clarity penalty (NA2M.clarity_loss),
    # wired into stage3 fine-tuning. 0.0 = off.

    # --- Optimiser & schedule ---
    lr: float = 1e-3
    decay_rate: float = 0.995

    # --- Staged selection / pruning (stages 2–4) ---
    top_m: int = 10
    # Number of top FAST-ranked candidate pairs to add before pruning.

    eta_prune: float = 0.0
    # η threshold for the cumulative validation-loss sweep that prunes interactions.

    block_train_epochs: int = 100
    # Epochs for the interactions-only block-training step (mains frozen).

    finetune_epochs: int = 100
    # Epochs per fine-tune pass (stage 3, and each stage-4 re-fine-tune).

    concurvity_threshold: float = 0.5
    # Stage 4 removes the worst pair while any concurvity exceeds this.

    max_concurvity_iters: int = 10
    # Cap on stage-4 remove-and-refit iterations.

    # --- Training loop ---
    batch_size: int = 1024
    num_epochs: int = 1000
    patience: int = 60
    val_check_interval: int = 10

    # --- Evaluation harness (k-fold × seed × arm) ---
    k_folds: int = 5
    fold_seed: int = 42
    # Seed for the outer fold split (fixed; folds = data/accuracy variation).

    seeds: list = field(default_factory=lambda: [0, 1, 2, 3, 4])
    # Per-fold replicate seeds (optimization/instability variation; bootstrap units).

    grid_size: int = 100
    # G: number of grid points per numerical feature (for shape-curve extraction).

    # --- Task ---
    task: str = "classification"
    # 'classification' → BCE + AUROC; 'regression' → MSE + RMSE.


def load_na2m_config(path: str) -> NA2MConfig:
    """Load an NA2MConfig from a YAML file.

    Args:
        path: Path to the YAML config.

    Returns:
        Populated NA2MConfig instance.
    """
    with open(path) as f:
        raw = yaml.safe_load(f)
    return NA2MConfig(**raw)


def load_na2m_search_config(path: str) -> tuple[dict, dict]:
    """Load a tuning config: fixed settings + an Optuna search space.

    Mirrors nam.utils.config.load_search_config — the YAML carries a top-level
    `search_space` block popped out from the fixed settings.

    Args:
        path: Path to the search YAML.

    Returns:
        (fixed_settings, search_space) dicts.
    """
    with open(path) as f:
        raw = yaml.safe_load(f)
    search_space = raw.pop("search_space", {})
    return raw, search_space
