"""
tune_concurvity_reg.py — lambda_2 grid sweep for NA2M arm D (concurvity regularizer).

Run AFTER tune_clarity.py has written gaminet_tuned_config.yaml. Trains the
main effects once, then for each (lambda_2, seed) pair runs Stage 2 (NoGate,
identical to arm B) and Stage 3 with lambda_1 fixed from the gaminet config and
lambda_2 from the grid. Records val_loss and R_perp at each grid point.

Does NOT write regularized_tuned_config.yaml — a human must inspect the
tradeoff plot and run confirm_regularized_arm.py to commit a value.

The elbow rule uses the same eta_prune tolerance as Stage 2's η-cut. This
gives a single consistent threshold for both the interaction-selection tradeoff
and the loss-vs-concurvity tradeoff.

KEY DESIGN — identical to tune_clarity.py's train-once strategy:
    1. Train the main bank ONCE from mains_tuned_config.yaml (stage1_main).
    2. Per (lambda_2, seed): deepcopy the snapshot, run Stage 2 (NoGate) and
       Stage 3 with the given lambda_2, record val_loss + R_perp.
Stage 1 is not re-run per grid point — it is identical across all lambda_2
values and would only inject noise if repeated.

Usage:
    python scripts/na2m/tune_concurvity_reg.py
    Edit the variables at the top of main() to point at your fold directory.
"""

import copy
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from na2m.data.dataset import NAMDataset
from na2m.models.na2m import NA2M
from na2m.training.fit_na2m import stage1_main, stage2_select, stage3_finetune, _eta_cut
from na2m.selection.policy import NoGate
from na2m.utils.config import NA2MConfig, load_na2m_config


# --------------------------------------------------------------------------- #
# Small helpers (mirror tune_clarity.py)                                       #
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
    """Build the train / val / pool loaders once (shared across all grid points).

    pool = train ∪ val — the reference sample used for centering.
    """
    X_pool = np.concatenate([X_train, X_val])
    y_pool = np.concatenate([y_train, y_val])
    train_loader = DataLoader(NAMDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(NAMDataset(X_val, y_val), batch_size=batch_size, shuffle=False)
    pool_loader = DataLoader(NAMDataset(X_pool, y_pool), batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, pool_loader


def _val_loss(
    model: NA2M,
    val_loader: DataLoader,
    task: str,
) -> float:
    """Validation loss (BCE for classification, MSE for regression) on the full val set.

    Args:
        model: NA2M in eval mode with best weights loaded.
        val_loader: Validation split loader.
        task: 'classification' or 'regression'.

    Returns:
        Mean loss over the val set.
    """
    # TODO: no_grad loop over val_loader collecting (logits, targets)
    # TODO: compute BCE (classification) or MSE (regression) in torch, return float
    raise NotImplementedError


def _compute_r_perp(
    model: NA2M,
    val_loader: DataLoader,
) -> float:
    """R_perp (mean absolute pairwise Pearson correlation of component outputs) on the full val set.

    Uses the full val set rather than a single batch for a stable diagnostic value.
    Computation is in numpy — this is a reporting metric, not a gradient computation.

    Args:
        model: NA2M in eval mode.
        val_loader: Validation split loader.

    Returns:
        R_perp in [0, 1].
    """
    # TODO: no_grad loop over val_loader; for each batch collect main_outputs(x) + inter_outputs(x)
    # TODO: concatenate across batches to get (N_val, p) matrix per component, then stack
    # TODO: if p < 2, return 0.0
    # TODO: compute np.corrcoef(F.T) -> (p, p); fill NaN with 0
    # TODO: extract upper triangle, abs, mean -> return float
    raise NotImplementedError


# --------------------------------------------------------------------------- #
# Fold-level grid sweep                                                        #
# --------------------------------------------------------------------------- #

def sweep_lambda2_fold(
    mains_config_path: Path,
    gaminet_config_path: Path,
    lambda2_grid: list[float],
    n_sweep_seeds: int,
    feature_meta,
    X_pool: np.ndarray,
    y_pool: np.ndarray,
    out_dir: Path,
    *,
    eta_prune: float = 0.0,
) -> Path:
    """Grid sweep over lambda_2 for a single fold. Writes CSV + tradeoff plot.

    Does NOT write regularized_tuned_config.yaml — human confirmation is required.
    Run confirm_regularized_arm.py after inspecting the plot.

    lambda_1 (clarity_regularization) is copied from gaminet_config_path and held
    fixed throughout — it is not re-tuned for arm D.

    Args:
        mains_config_path: Stage-1 tuned config (architecture + training HP).
        gaminet_config_path: Arm B tuned config — source of lambda_1.
        lambda2_grid: Candidate lambda_2 values to sweep (log-spaced recommended).
        n_sweep_seeds: Number of random seeds per grid point (for variance estimate).
        feature_meta: FeatureMeta list from preprocess().
        X_pool: The fold's 80% pool (train ∪ val); split internally for tuning.
        y_pool: Pool labels.
        out_dir: Fold config dir; receives regularized_lambda2_sweep.csv and .png.
        eta_prune: Tolerance for the elbow rule (same value as Stage 2's η-cut).

    Returns:
        Path to the written CSV.
    """
    # TODO: load mains_config, gaminet_config; extract lambda_1 = gaminet_config.clarity_regularization
    # TODO: split X_pool/y_pool into train/val using mains_config.pool_val_frac and mains_config.seed
    # TODO: build loaders
    # TODO: train Stage 1 once -> stage1_snapshot = deepcopy(model)
    # TODO: derive n_sweep_seeds seeds from np.random.SeedSequence(mains_config.seed).spawn(n_sweep_seeds)
    # TODO: outer loop over lambda2_grid, inner loop over seeds:
    #   - build trial config (deepcopy mains_config, set clarity_regularization=lambda_1, concurvity_regularization=lambda_2)
    #   - set seed (torch + numpy + random)
    #   - model = deepcopy(stage1_snapshot)
    #   - stage2_select(model, ..., selection_policy=NoGate())
    #   - stage3_finetune(model, ...)
    #   - record val_loss, r_perp
    # TODO: write CSV with columns lambda2, seed, val_loss, r_perp
    # TODO: compute per-lambda_2 mean val_loss across seeds
    # TODO: apply _eta_cut(mean_val_losses_sorted, eta_prune) -> auto_lambda2
    # TODO: write two-panel tradeoff plot (val_loss vs lambda_2, r_perp vs lambda_2; log x-axis; mean±std; vertical dashed at auto_lambda2)
    # TODO: print auto-elbow value, plot path, instructions for confirm_regularized_arm.py
    # TODO: return CSV path
    raise NotImplementedError


# --------------------------------------------------------------------------- #
# Standalone entry point                                                       #
# --------------------------------------------------------------------------- #

def main() -> None:
    # ── Edit these before running ───────────────────────────────────────────
    fold_dir    = Path(r"runs/compas/fold_0")
    config_path = Path(r"configs/compas_na2m_search.yaml")
    # ────────────────────────────────────────────────────────────────────────

    with open(config_path) as f:
        raw = yaml.safe_load(f)

    lambda2_grid  = raw.get("lambda2_grid", [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0])
    n_sweep_seeds = raw.get("n_sweep_seeds", 3)
    eta_prune     = raw.get("eta_prune", 0.0)

    # TODO: load X, y, feature_meta from fixed_params["dataset_path"]
    # TODO: reconstruct pool_idx for the correct fold (requires k-fold split — see run_na2m_eval.py)
    # TODO: call sweep_lambda2_fold(...)
    raise NotImplementedError


if __name__ == "__main__":
    main()
