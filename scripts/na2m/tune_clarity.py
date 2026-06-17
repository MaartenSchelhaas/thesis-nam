"""
tune_clarity.py — Optuna search for NA2M clarity_regularization (arm B).

Run AFTER tune_na2m.py. Loads the Stage-1 tuned main-effects config and searches
ONLY clarity_regularization, then writes the best value back into the same YAML.
Arm C later reuses that same config unchanged.

KEY DESIGN — train the mains ONCE, then sweep only Stage 2/3
------------------------------------------------------------
clarity_regularization has NO effect on Stage 1: the marginal-clarity penalty
acts on interactions, which don't exist yet during the mains fit (stage1_main
hardcodes clarity_lambda=0.0). So retraining the mains from scratch on every
clarity trial would recompute the SAME main effects again and again — pure wasted
compute, and it would inject mains-init noise into a comparison that is supposed
to isolate clarity.

Instead:
    1. Train the main bank ONCE with the fixed Stage-1 hyperparameters
       (stage1_main) and snapshot the resulting model.
    2. Each trial starts from a fresh deepcopy of that snapshot and runs ONLY
       Stage 2 (block-train + select) and Stage 3 (fine-tune) with the trial's
       clarity_regularization. The trials then differ ONLY by clarity.

Why a fresh deepcopy per trial (not the shared snapshot): Stage 3 fine-tunes ALL
params, including the mains. Mutating the shared snapshot would leak trial n's
fine-tuned mains into trial n+1. The deepcopy gives every trial the same clean
post-Stage-1 starting point.

Why Stage 2 is inside the trial (not done once like Stage 1): the Stage-2 block
trainer uses clarity_lambda, so different clarity can block-train different
interaction subnets and therefore SELECT a different pair set. Selection is part
of what clarity influences, so it must be redone per trial.

No pruning: with Stage 1 lifted out of the trial there are no clarity-dependent
intermediate Stage-1 metrics to prune on; each trial just runs Stage 2/3 to
completion.

The real per-(fold, run) evaluation in run_single.py still runs the FULL pipeline
via fit_na2m — this train-once shortcut is a TUNING optimisation only.

Usage:
    python scripts/na2m/tune_clarity.py
    Edit the variables at the top of main() to point at your configs.
"""

import copy
from pathlib import Path

import numpy as np
import optuna
import torch
import yaml
from torch.utils.data import DataLoader

from na2m.data.dataset import NAMDataset
from na2m.models.na2m import NA2M
from na2m.training.fit_na2m import stage1_main, stage2_select, stage3_finetune
from na2m.selection.policy import NoGate, ConcurvityGate
from na2m.training.metrics import auroc
from na2m.utils.config import NA2MConfig, load_na2m_config


def load_clarity_search_config(search_yaml_path: str) -> tuple[int, dict]:
    """Extract clarity tuning settings from the main NA2M search YAML.

    Args:
        search_yaml_path: Path to compas_na2m_search.yaml.

    Returns:
        (n_trials, search_spec) where search_spec has keys type/low/high.
    """
    with open(search_yaml_path) as f:
        raw = yaml.safe_load(f)
    return raw["clarity_n_trials"], raw["clarity_search_space"]


# --------------------------------------------------------------------------- #
# Small helpers                                                                #
# --------------------------------------------------------------------------- #

def _build_na2m(config: NA2MConfig, num_features: int, feature_meta) -> NA2M:
    """Construct a fresh NA2M from a config (interactions empty at entry)."""
    return NA2M(
        num_features=num_features,
        feature_meta=feature_meta,
        num_units=config.num_units,
        hidden_sizes=config.hidden_sizes,
        dropout=config.dropout,
        feature_dropout=config.feature_dropout,
        activation=config.activation,
        inter_units=config.inter_units,
        inter_hidden=config.inter_hidden,
    )


def _build_loaders(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    batch_size: int,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Build the train / val / pool loaders once (shared across all trials).

    pool = train ∪ val — the reference sample used for centering.
    """
    X_pool = np.concatenate([X_train, X_val])
    y_pool = np.concatenate([y_train, y_val])
    train_loader = DataLoader(NAMDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(NAMDataset(X_val, y_val), batch_size=batch_size, shuffle=False)
    pool_loader = DataLoader(NAMDataset(X_pool, y_pool), batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, pool_loader


def _val_auroc(model: NA2M, val_loader: DataLoader) -> float:
    """Validation AUROC of the current model (eval mode, no grad)."""
    model.eval()
    device = next(model.parameters()).device
    all_logits, all_targets = [], []
    with torch.no_grad():
        for X_batch, y_batch, _ in val_loader:
            logits, _ = model(X_batch.to(device))
            all_logits.append(logits.cpu())
            all_targets.append(y_batch)
    return float(auroc(torch.cat(all_logits), torch.cat(all_targets)))


# --------------------------------------------------------------------------- #
# The clarity objective — Stage 2/3 only, starting from the fixed mains        #
# --------------------------------------------------------------------------- #

def objective_clarity(
    trial: optuna.Trial,
    stage1_snapshot: NA2M,
    config: NA2MConfig,
    train_loader: DataLoader,
    val_loader: DataLoader,
    pool_loader: DataLoader,
    search_spec: dict,
    with_concurvity_filter: bool,
) -> float:
    """Sample clarity_regularization, run Stage 2/3 from the fixed mains, return val AUROC.

    Stage 1 is NOT run here — it was run once by the caller and captured in
    stage1_snapshot. This trial only adds + selects interactions (Stage 2) and
    fine-tunes (Stage 3), so it differs from every other trial ONLY by clarity.

    Args:
        trial: Current Optuna trial.
        stage1_snapshot: Model with mains trained + centered (post Stage 1). NOT
            mutated — a fresh deepcopy is taken per trial.
        config: Stage-1 tuned config. Copied internally so clarity is set per trial
            without touching the original.
        train_loader, val_loader, pool_loader: Shared loaders (built once).
        search_spec: Dict with keys type/low/high for clarity_regularization.
        with_concurvity_filter: If True use ConcurvityGate in Stage 2 (arm C);
            otherwise NoGate (arm B). Affects which pairs survive and therefore
            the optimal clarity λ.

    Returns:
        Validation AUROC after Stage 2 + Stage 3 with this trial's clarity.
    """
    clarity = trial.suggest_float(
        "clarity_regularization",
        search_spec["low"],
        search_spec["high"],
        log=True,
    )

    cfg = copy.deepcopy(config)
    cfg.clarity_regularization = clarity

    model = copy.deepcopy(stage1_snapshot)

    if with_concurvity_filter:
        policy = ConcurvityGate(cfg.concurvity_threshold)
    else:
        policy = NoGate()

    stage2_select(model, train_loader, val_loader, pool_loader, cfg, selection_policy=policy)
    stage3_finetune(model, train_loader, val_loader, pool_loader, cfg)

    return _val_auroc(model, val_loader)


# --------------------------------------------------------------------------- #
# Fold-level driver                                                            #
# --------------------------------------------------------------------------- #

def tune_clarity_fold(
    mains_config_path: Path,
    n_trials: int,
    search_spec: dict,
    feature_meta,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    output_path: Path,
    *,
    with_concurvity_filter: bool,
    study_name: str = "clarity_search",
) -> Path:
    """Train the mains once, search clarity_regularization over Stage 2/3, write result.

    Reads the Stage-1 mains config, trains the main bank ONCE, then runs an Optuna
    study where each trial restarts from that fixed snapshot and runs only Stage 2/3.
    The best clarity_regularization is written into output_path (a copy of the mains
    config extended with the arm's tuned λ). Each arm gets its own output_path.

    Args:
        mains_config_path: YAML written by tune_main_na2m.tune_fold (mains config, read-only).
        n_trials: Number of Optuna trials.
        search_spec: Dict with keys type/low/high for clarity_regularization.
        feature_meta: FeatureMeta list from preprocess().
        X_train, y_train: Training split (the fold's fixed inner train).
        X_val, y_val: Validation split (the fold's fixed inner val).
        output_path: Where to write the arm's complete tuned config (mains hp + λ).
        with_concurvity_filter: True → ConcurvityGate in Stage 2 (arm C); False → NoGate (arm B).
        study_name: Optuna study name.

    Returns:
        output_path after writing.
    """
    config = load_na2m_config(str(mains_config_path))
    num_features = X_train.shape[1]

    train_loader, val_loader, pool_loader = _build_loaders(
        X_train, y_train, X_val, y_val, config.batch_size
    )

    print(f"[clarity] Training main effects once (Stage 1) for study '{study_name}'...")
    stage1_model = _build_na2m(config, num_features, feature_meta)
    stage1_main(stage1_model, train_loader, val_loader, pool_loader, config)
    stage1_model.eval()
    stage1_snapshot = copy.deepcopy(stage1_model)
    print("[clarity] Stage-1 mains snapshot ready. Searching clarity over Stage 2/3...")

    study = optuna.create_study(
        study_name=study_name,
        direction="maximize",
        sampler=optuna.samplers.TPESampler(),
    )
    study.optimize(
        lambda trial: objective_clarity(
            trial,
            stage1_snapshot,
            config,
            train_loader,
            val_loader,
            pool_loader,
            search_spec,
            with_concurvity_filter,
        ),
        n_trials=n_trials,
    )

    best_clarity = study.best_trial.params["clarity_regularization"]

    # Write mains config + tuned clarity into the arm's output path.
    with open(mains_config_path) as f:
        raw = yaml.safe_load(f)
    raw["clarity_regularization"] = best_clarity
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        yaml.dump(raw, f, default_flow_style=False, sort_keys=False)

    print(f"Best clarity val AUROC    : {study.best_trial.value:.4f}")
    print(f"Best clarity_regularization={best_clarity:.6f} written to {output_path}")
    return output_path

