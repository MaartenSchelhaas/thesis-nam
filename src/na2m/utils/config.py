"""
NA2MConfig — hyperparameters for the NA2M model and training pipeline.
Defaults are placeholders; tune via configs/*.yaml and load_na2m_config().
"""

from dataclasses import dataclass, field

import yaml


@dataclass
class NA2MConfig:
    dataset_path: str = ""

    # ------------------------------------------------------------------ #
    # Main subnet architecture                                             #
    # ------------------------------------------------------------------ #

    num_units: int = 64
    hidden_sizes: list = field(default_factory=lambda: [64, 32])
    activation: str = "exu"
    dropout: float = 0.5

    # ------------------------------------------------------------------ #
    # Interaction subnet architecture                                      #
    # ------------------------------------------------------------------ #

    inter_units: int = 32
    inter_hidden: list = field(default_factory=lambda: [])
    feature_dropout: float = 0.0

    # ------------------------------------------------------------------ #
    # Regularisation                                                       #
    # ------------------------------------------------------------------ #

    output_regularization: float = 0.0
    l2_regularization: float = 0.0
    clarity_regularization: float = 0.0  # λ_1 on the GAMI-Net marginal-clarity penalty (stage 3)
    concurvity_regularization: float = 0.0  # λ_2 on the Siems et al. R_perp pairwise concurvity penalty (stage 3, arm D only)

    # ------------------------------------------------------------------ #
    # Optimiser                                                            #
    # ------------------------------------------------------------------ #

    lr: float = 1e-3
    decay_rate: float = 0.995

    # ------------------------------------------------------------------ #
    # Data split                                                           #
    # ------------------------------------------------------------------ #

    val_frac: float = 0.15
    test_frac: float = 0.15
    seed: int = 42
    pool_val_frac: float = 0.15  # inner train/val split of the fold's 80% pool; keyed off the fold, never the run seed

    # ------------------------------------------------------------------ #
    # Training loop                                                        #
    # ------------------------------------------------------------------ #

    batch_size: int = 1024
    num_epochs: int = 1000
    patience: int = 60
    val_check_interval: int = 10

    # ------------------------------------------------------------------ #
    # Stage 2 — interaction selection                                      #
    # ------------------------------------------------------------------ #

    top_m: int = 10                  # FAST candidate pairs to block-train before the sweep
    eta_prune: float = 0.0           # predictive-contribution gate tolerance (0 = cut at argmin)
    block_train_epochs: int = 1000   # epochs for the joint interaction block-training step
    finetune_epochs: int = 100       # epochs for the single Stage-3 fine-tune

    # ------------------------------------------------------------------ #
    # Concurvity gate (arm C only)                                        #
    # ------------------------------------------------------------------ #

    concurvity_filter: bool = True       # False → arm B; True → arm C
    concurvity_threshold: float = 0.5   # adj-R² threshold; candidates above this are skipped

    # ------------------------------------------------------------------ #
    # Evaluation                                                           #
    # ------------------------------------------------------------------ #

    k_folds: int = 5
    fold_seed: int = 42
    seeds: list = field(default_factory=lambda: [0, 1, 2, 3, 4])
    grid_size: int = 256  # grid points per numerical feature for shape curves

    # ------------------------------------------------------------------ #
    # Task                                                                 #
    # ------------------------------------------------------------------ #

    task: str = "classification"  # 'classification' or 'regression'


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
    """Load a tuning config: fixed params + search space.

    Args:
        path: Path to the search YAML.

    Returns:
        (fixed_params, search_space) where fixed_params holds all non-search fields
        (including study knobs like n_trials / clarity_n_trials) and search_space
        holds the per-parameter specs to pass to Optuna.
    """
    with open(path) as f:
        raw = yaml.safe_load(f)
    search_space = raw.pop("search_space", {})
    return raw, search_space
