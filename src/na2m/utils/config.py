"""
NA2MConfig — central hyperparameter store for the NA2M extension.

Kept SEPARATE from nam.utils.config.NAMConfig: the two overlap on the
optimizer/training fields but NA2M adds a substantial surface (interaction
subnets, staged selection/pruning, the concurvity GATE, and the k-fold × seed
× arm evaluation harness). Sharing one dataclass would force NAM-specific config
to carry na2m-only fields, so we fork.

Pipeline shape):
    Stage 1  train mains, center.
    Stage 2  FAST screen → block-train top-M → ONE forward prune sweep applying
             (a) the concurvity gate [only when concurvity_filter=True] and
             (b) the predictive-contribution gate.
    Stage 3  ONE joint fine-tune (mains + survivors) with the clarity penalty,
             then re-center.
Arms A/B/C differ ONLY by `with_interactions` and `concurvity_filter`; both B
and C fine-tune EXACTLY ONCE. There is no per-removal retraining anywhere.

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
    # Main subnet architecture                                             #
    # ------------------------------------------------------------------ #

    num_units: int = 64
    # Width of the activation layer (ExU/LinReLU).

    hidden_sizes: list = field(default_factory=lambda: [64, 32])
    # Widths of hidden Dense layers after the activation layer.
    # Empty list → shallow (activation layer + output only).

    activation: str = "exu"
    # Activation layer type: 'exu' or 'relu'.

    dropout: float = 0.5
    # Dropout probability after each hidden layer (shared with interaction subnets).

    # ------------------------------------------------------------------ #
    # Interaction subnet architecture                                      #
    # ------------------------------------------------------------------ #

    inter_units: int = 32
    # Width of the interaction FeatureNN activation layer.
    # Shallower than main subnets — interactions capture residual signal.

    inter_hidden: list = field(default_factory=lambda: [])
    # Hidden layer widths for interaction subnets. Default: shallow (no hidden layers).

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
    # val_frac/test_frac/seed drive the simple (non-k-fold) split() helper only.

    pool_val_frac: float = 0.15
    # Internal early-stopping split of the fold's 80% pool into (pool_train,
    # pool_val), used for early stopping AND the Stage-2 η-prune sweep.
    # SPLIT CONTRACT: this split is keyed off the FOLD (fold_seed), never off the
    # replicate `seed`. Varying the replicate seed must change ONLY init +
    # optimization, never which rows land in pool_train vs pool_val. Do not
    # reseed this split per replicate.

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
    # Stage 2 — interaction selection & the single prune sweep             #
    # ------------------------------------------------------------------ #

    top_m: int = 10
    # Number of top FAST-ranked candidate pairs block-trained before the sweep.

    eta_prune: float = 0.0
    # Tolerance η for the predictive-contribution gate of the Stage-2 prune
    # sweep. The sweep records validation loss after each accepted candidate
    # (added in DECREASING contribution order); the loss sequence is then
    # min-max normalized and the cut is the SMALLEST index whose normalized loss
    # is within η of the minimum (fallback to argmin if the range is degenerate).
    # 0.0 = cut exactly at the argmin (no slack toward simpler models).

    block_train_epochs: int = 100
    # Epochs for the interaction-only block-training step (Stage 2,
    # main-effect weights frozen). All top_m subnets are trained JOINTLY here,
    # once, before the sweep — the sweep itself NEVER retrains.

    finetune_epochs: int = 100
    # Epochs for the SINGLE joint fine-tuning pass (Stage 3). Both arm B and
    # arm C fine-tune exactly once; there is no per-removal re-fine-tune.

    # ------------------------------------------------------------------ #
    # Concurvity gate (Stage 2 prune sweep — arm C only)                   #
    # ------------------------------------------------------------------ #

    concurvity_filter: bool = True
    # Master switch separating arm B from arm C.
    #   False → arm B: the concurvity gate NEVER fires (vanilla GAMI-Net).
    #   True  → arm C: the gate is active during the Stage-2 prune sweep.
    # This flag is the ONLY difference between B and C. Block training, the
    # predictive-contribution gate, the single fine-tune, and centering are all
    # identical across the two arms.

    concurvity_threshold: float = 0.5
    # Gate threshold τ. During the sweep (arm C only), a candidate whose
    # block-trained output regresses on {all mains + all already-ACCEPTED
    # interactions} with adjusted R² > τ is SKIPPED and never reconsidered.
    # NOTE: this gates SELECTION only. The deployed (fine-tuned) model is NOT
    # guaranteed ≤ τ — fine-tuning moves the geometry; see concurvity_summary.

    # ------------------------------------------------------------------ #
    # Evaluation harness                                                   #
    # ------------------------------------------------------------------ #

    k_folds: int = 5
    fold_seed: int = 42
    # Seed for the outer k-fold split (fixed across runs → same folds).

    seeds: list = field(default_factory=lambda: [0, 1, 2, 3, 4])
    # Replicate seeds run within EVERY fold (the cross-seed sample the stability
    # metric reduces over). A seed controls ONLY initialization + optimization
    # order; per the split contract it must NOT influence any data split. Term
    # stability across these seeds is the headline metric, so vary nothing but
    # the seed between replicates of a fold.

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