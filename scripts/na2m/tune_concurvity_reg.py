"""
tune_concurvity_reg.py — lambda_2 grid sweep for NA2M arm D (concurvity regularizer).

Run AFTER tune_clarity.py has written gaminet_tuned_config.yaml. Trains the
main effects once, then for each (lambda_2, seed) pair runs Stage 2 (NoGate,
identical to arm B) and Stage 3 with lambda_1 fixed from the gaminet config and
lambda_2 from the grid. Records val_loss and R_perp at each grid point.

Does NOT write regularized_tuned_config.yaml — a human must inspect the
tradeoff plot and run confirm_regularized_arm.py to commit a value.

The elbow rule uses the same eta_prune tolerance as Stage 2's η-cut. Losses
are passed in descending lambda_2 order so _eta_cut returns the largest lambda_2
still within eta of the minimum loss (the same "fewest constraint" logic as
Stage 2's "fewest removed pairs").

KEY DESIGN — identical to tune_clarity.py's train-once strategy:
    1. Train the main bank ONCE from mains_tuned_config.yaml (stage1_main).
    2. Per (lambda_2, seed): deepcopy the snapshot, run Stage 2 (NoGate) and
       Stage 3 with the given lambda_2, record val_loss + R_perp.
Stage 1 is not re-run per grid point — it is identical across all lambda_2
values and would only inject noise if repeated.

The fold's tuning split (X_tune / X_tune_val) is passed in by run_fold, the
same arrays tune_clarity_fold already uses for this fold. This guarantees
lambda_2 is evaluated on the same split as lambda_1.
"""

import copy
import csv
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from na2m.data.dataset import NAMDataset
from na2m.models.na2m import NA2M
from na2m.training.fit_na2m import stage1_main, stage2_select, stage3_finetune, _eta_cut
from na2m.training.losses import base_loss
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
    val_loader   = DataLoader(NAMDataset(X_val,   y_val),   batch_size=batch_size, shuffle=False)
    pool_loader  = DataLoader(NAMDataset(X_pool,  y_pool),  batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, pool_loader


def _val_loss(
    model: NA2M,
    val_loader: DataLoader,
    task: str,
) -> float:
    """Validation loss (BCE for classification, MSE for regression) on the full val set.

    Args:
        model: NA2M in eval mode.
        val_loader: Validation split loader.
        task: 'classification' or 'regression'.

    Returns:
        Mean loss over the val set.
    """
    model.eval()
    device = next(model.parameters()).device
    all_logits, all_targets = [], []
    with torch.no_grad():
        for X_batch, y_batch, _ in val_loader:
            logits, _ = model(X_batch.to(device))
            all_logits.append(logits.cpu())
            all_targets.append(y_batch.cpu())
    logits  = torch.cat(all_logits)
    targets = torch.cat(all_targets).float()
    weights = torch.ones_like(targets)
    return float(base_loss(logits, targets, weights, task))


def _compute_r_perp(
    model: NA2M,
    val_loader: DataLoader,
) -> float:
    """Mean absolute pairwise Pearson correlation of component outputs on the full val set.

    Uses the full val set rather than a single batch for a stable diagnostic value.
    Computation is in numpy — this is a reporting metric, not a gradient computation.

    Args:
        model: NA2M in eval mode.
        val_loader: Validation split loader.

    Returns:
        R_perp in [0, 1]. Returns 0.0 when fewer than 2 components are active.
    """
    model.eval()
    device = next(model.parameters()).device
    collected = None

    with torch.no_grad():
        for X_batch, _, _ in val_loader:
            X_batch = X_batch.to(device)
            outputs = model.main_outputs(X_batch) + model.inter_outputs(X_batch)
            if len(outputs) < 2:
                return 0.0
            batch_matrix = torch.cat(outputs, dim=1).cpu().numpy()  # (batch, p)
            if collected is None:
                collected = batch_matrix
            else:
                collected = np.concatenate([collected, batch_matrix], axis=0)

    if collected is None or collected.shape[1] < 2:
        return 0.0

    F = collected.T                              # (p, N_val)
    p = F.shape[0]
    corr = np.corrcoef(F)                        # (p, p)
    corr = np.nan_to_num(corr, nan=0.0)
    row_idx, col_idx = np.triu_indices(p, k=1)
    return float(np.mean(np.abs(corr[row_idx, col_idx])))


# --------------------------------------------------------------------------- #
# Fold-level grid sweep                                                        #
# --------------------------------------------------------------------------- #

def concurvity_reg_fold(
    mains_config_path: Path,
    gaminet_config_path: Path,
    lambda2_grid: list[float],
    n_sweep_seeds: int,
    feature_meta,
    X_tune: np.ndarray,
    y_tune: np.ndarray,
    X_tune_val: np.ndarray,
    y_tune_val: np.ndarray,
    out_dir: Path,
    *,
    eta_prune: float = 0.0,
) -> Path:
    """Grid sweep over lambda_2 for a single fold. Writes CSV + tradeoff plot.

    Does NOT write regularized_tuned_config.yaml — human confirmation is required.
    Run confirm_regularized_arm.py after inspecting the plot.

    lambda_1 (clarity_regularization) is copied from gaminet_config_path and held
    fixed throughout — it is not re-tuned for arm D.

    The fold's tuning split is passed in as already-sliced arrays (X_tune /
    X_tune_val), matching the exact split used by tune_clarity_fold for this fold.

    Args:
        mains_config_path: Stage-1 tuned config (architecture + training HP).
        gaminet_config_path: Arm B tuned config — source of lambda_1.
        lambda2_grid: Candidate lambda_2 values to sweep (log-spaced recommended).
        n_sweep_seeds: Number of random seeds per grid point (for variance estimate).
        feature_meta: FeatureMeta list from preprocess().
        X_tune, y_tune: Fold's inner training split (same arrays used by tune_clarity_fold).
        X_tune_val, y_tune_val: Fold's inner validation split.
        out_dir: Fold config dir; receives regularized_lambda2_sweep.csv and .png.
        eta_prune: Tolerance for the elbow rule (same value as Stage 2's η-cut).

    Returns:
        Path to the written CSV.
    """
    mains_config = load_na2m_config(str(mains_config_path))

    with open(gaminet_config_path) as f:
        gaminet_raw = yaml.safe_load(f)
    lambda_1 = gaminet_raw["clarity_regularization"]

    num_features = X_tune.shape[1]
    train_loader, val_loader, pool_loader = _build_loaders(
        X_tune, y_tune, X_tune_val, y_tune_val, mains_config.batch_size
    )

    print("[sweep_lambda2] Training Stage 1 once...")
    torch.manual_seed(mains_config.seed)
    np.random.seed(mains_config.seed % (2**31))
    random.seed(mains_config.seed)
    stage1_model = _build_na2m(mains_config, num_features, feature_meta)
    stage1_main(stage1_model, train_loader, val_loader, pool_loader, mains_config)
    stage1_model.eval()
    stage1_snapshot = copy.deepcopy(stage1_model)
    print(f"[sweep_lambda2] Stage-1 snapshot ready. Sweeping {len(lambda2_grid)} lambda_2 values "
          f"x {n_sweep_seeds} seeds = {len(lambda2_grid) * n_sweep_seeds} trials...")

    seed_sequence = np.random.SeedSequence(mains_config.seed)
    sweep_seeds = [int(ss.generate_state(1)[0]) for ss in seed_sequence.spawn(n_sweep_seeds)]

    rows = []
    for lambda_2 in lambda2_grid:
        for seed in sweep_seeds:
            torch.manual_seed(seed)
            np.random.seed(seed % (2**31))
            random.seed(seed)

            trial_config = copy.deepcopy(mains_config)
            trial_config.clarity_regularization    = lambda_1
            trial_config.concurvity_regularization = lambda_2

            model = copy.deepcopy(stage1_snapshot)
            stage2_select(model, train_loader, val_loader, pool_loader, trial_config, selection_policy=NoGate())
            stage3_finetune(model, train_loader, val_loader, pool_loader, trial_config)
            model.eval()

            loss   = _val_loss(model, val_loader, mains_config.task)
            r_perp = _compute_r_perp(model, val_loader)
            rows.append({"lambda2": lambda_2, "seed": seed, "val_loss": loss, "r_perp": r_perp})
            print(f"  lambda_2={lambda_2:.5f}  seed={seed}  val_loss={loss:.4f}  r_perp={r_perp:.4f}")

    # Write CSV
    csv_path = out_dir / "regularized_lambda2_sweep.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["lambda2", "seed", "val_loss", "r_perp"])
        writer.writeheader()
        writer.writerows(rows)

    # Per-lambda_2 mean/std over seeds, sorted ascending for plotting
    sorted_grid = sorted(lambda2_grid)
    mean_losses, std_losses, mean_r_perps, std_r_perps = [], [], [], []
    for lam in sorted_grid:
        subset  = [r for r in rows if r["lambda2"] == lam]
        losses  = [r["val_loss"] for r in subset]
        r_perps = [r["r_perp"]   for r in subset]
        mean_losses.append(np.mean(losses))
        std_losses.append(np.std(losses))
        mean_r_perps.append(np.mean(r_perps))
        std_r_perps.append(np.std(r_perps))

    # Elbow: pass losses in descending lambda_2 order so _eta_cut returns the
    # largest lambda_2 still within eta of the minimum loss.
    descending_losses = list(reversed(mean_losses))
    cut_idx      = _eta_cut(descending_losses, eta_prune)
    auto_lambda2 = list(reversed(sorted_grid))[cut_idx]

    # Two-panel tradeoff plot
    x              = np.array(sorted_grid)
    mean_losses_a  = np.array(mean_losses)
    std_losses_a   = np.array(std_losses)
    mean_r_perps_a = np.array(mean_r_perps)
    std_r_perps_a  = np.array(std_r_perps)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    ax1.semilogx(x, mean_losses_a, marker="o")
    ax1.fill_between(x, mean_losses_a - std_losses_a, mean_losses_a + std_losses_a, alpha=0.2)
    ax1.axvline(auto_lambda2, linestyle="--", color="red", label=f"auto λ₂={auto_lambda2}")
    ax1.set_xlabel("λ₂")
    ax1.set_ylabel("val loss")
    ax1.set_title("Val loss vs λ₂")
    ax1.legend()

    ax2.semilogx(x, mean_r_perps_a, marker="o")
    ax2.fill_between(x, mean_r_perps_a - std_r_perps_a, mean_r_perps_a + std_r_perps_a, alpha=0.2)
    ax2.axvline(auto_lambda2, linestyle="--", color="red", label=f"auto λ₂={auto_lambda2}")
    ax2.set_xlabel("λ₂")
    ax2.set_ylabel("R_perp")
    ax2.set_title("R_perp vs λ₂")
    ax2.legend()

    fig.tight_layout()
    plot_path = out_dir / "regularized_lambda2_sweep.png"
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)

    print(f"\n[sweep_lambda2] Auto-elbow lambda_2 = {auto_lambda2}")
    print(f"[sweep_lambda2] Plot:  {plot_path}")
    print(f"[sweep_lambda2] CSV:   {csv_path}")
    print(f"[sweep_lambda2] Inspect the plot, then confirm with:")
    print(f"    python scripts/na2m/confirm_regularized_arm.py")
    print(f"    (set fold_dir={out_dir}, lambda2=<your choice>)")

    return csv_path