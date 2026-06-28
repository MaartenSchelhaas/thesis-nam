"""
confirm_regularized_arm.py — write regularized_tuned_config.yaml for arm D.

Run AFTER concurvity_reg_fold has written the sweep CSV and tradeoff plot for
each fold. Inspect each fold's plot, fill in lambda2_per_fold below, and run.

- None         → not yet confirmed (skipped, no file written)
- missing CSV  → sweep not yet run for that fold (warning printed)

What this writes per confirmed fold:
    regularized_tuned_config.yaml
        All fields from mains_tuned_config.yaml, plus:
            clarity_regularization      — copied from gaminet_tuned_config.yaml (lambda_1)
            concurvity_regularization   — the confirmed lambda_2 value

    regularized_config_meta.yaml
        Source-tracking only (not loaded by NA2MConfig):
            confirmed_lambda2    — the value that was written
            lambda2_auto_pick    — elbow suggestion computed from the sweep CSV
            lambda1_from_gaminet — the clarity_regularization value copied from gaminet
            reason               — optional human note
            confirmed_at         — ISO timestamp

Usage:
    Edit the variables at the top of main() then run:
        python scripts/na2m/confirm_regularized_arm.py
"""

import csv
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml

from na2m.training.fit_na2m import _eta_cut


def _auto_pick_from_csv(sweep_csv: Path, eta_prune: float) -> float:
    """Compute the elbow lambda_2 suggestion from the sweep CSV.

    Reads (lambda2, seed, val_loss, r_perp) rows, averages val_loss per lambda_2,
    and applies the same eta_prune elbow rule used during the sweep (descending
    lambda_2 order so _eta_cut returns the largest lambda_2 within tolerance).

    Args:
        sweep_csv: Path to regularized_lambda2_sweep.csv.
        eta_prune: Tolerance for the elbow rule.

    Returns:
        Auto-suggested lambda_2 value.
    """
    rows = []
    with open(sweep_csv, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({"lambda2": float(row["lambda2"]), "val_loss": float(row["val_loss"])})

    sorted_grid = sorted(set(r["lambda2"] for r in rows))
    mean_losses = []
    for lam in sorted_grid:
        losses = [r["val_loss"] for r in rows if r["lambda2"] == lam]
        mean_losses.append(float(np.mean(losses)))

    descending_losses = list(reversed(mean_losses))
    cut_idx = _eta_cut(descending_losses, eta_prune)
    return list(reversed(sorted_grid))[cut_idx]


def _confirm_fold(
    fold_dir: Path,
    lambda2: float,
    eta_prune: float,
    reason: str,
    overwrite: bool,
) -> None:
    """Write regularized_tuned_config.yaml and meta file for one fold."""
    mains_config_yaml   = fold_dir / "mains_tuned_config.yaml"
    gaminet_config_yaml = fold_dir / "gaminet_tuned_config.yaml"
    sweep_csv           = fold_dir / "regularized_lambda2_sweep.csv"
    output_config_yaml  = fold_dir / "regularized_tuned_config.yaml"
    output_meta_yaml    = fold_dir / "regularized_config_meta.yaml"

    for p in [mains_config_yaml, gaminet_config_yaml, sweep_csv]:
        if not p.exists():
            raise FileNotFoundError(
                f"{p} not found. Run concurvity_reg_fold for this fold first."
            )

    if output_config_yaml.exists() and not overwrite:
        raise FileExistsError(
            f"{output_config_yaml} already exists. Set overwrite=True to replace it."
        )

    with open(mains_config_yaml) as f:
        mains_raw = yaml.safe_load(f)

    with open(gaminet_config_yaml) as f:
        gaminet_raw = yaml.safe_load(f)
    lambda_1 = gaminet_raw["clarity_regularization"]

    auto_pick = _auto_pick_from_csv(sweep_csv, eta_prune)

    output_config = mains_raw | {
        "clarity_regularization":    lambda_1,
        "concurvity_regularization": lambda2,
    }
    with open(output_config_yaml, "w") as f:
        yaml.dump(output_config, f, default_flow_style=False, sort_keys=False)

    meta = {
        "confirmed_lambda2":    lambda2,
        "lambda2_auto_pick":    auto_pick,
        "lambda1_from_gaminet": lambda_1,
        "reason":               reason,
        "confirmed_at":         datetime.now(timezone.utc).isoformat(),
    }
    with open(output_meta_yaml, "w") as f:
        yaml.dump(meta, f, default_flow_style=False, sort_keys=False)

    print(f"  lambda_1 (from gaminet) : {lambda_1}")
    print(f"  lambda_2 auto-pick      : {auto_pick}")
    print(f"  lambda_2 confirmed      : {lambda2}")
    print(f"  Written: {output_config_yaml}")
    print(f"  Written: {output_meta_yaml}")


def main() -> None:
    # ── Edit these before running ───────────────────────────────────────────
    base_dir    = Path(r"runs/compas_na2m")
    config_path = Path(r"configs/compas_na2m_search.yaml")
    n_folds     = 5
    reason      = ""      # optional note explaining the choice (applied to all folds)
    overwrite   = False   # set True to overwrite existing configs

    # Set lambda_2 per fold after inspecting each fold's sweep plot.
    # Leave as None to skip that fold (not yet confirmed).
    lambda2_per_fold = {
        0: None,
        1: None,
        2: None,
        3: None,
        4: None,
    }
    # ────────────────────────────────────────────────────────────────────────

    with open(config_path) as f:
        raw = yaml.safe_load(f)
    eta_prune = raw.get("eta_prune", 0.0)

    for fold_idx in range(n_folds):
        fold_dir  = base_dir / f"fold_{fold_idx}"
        sweep_csv = fold_dir / "regularized_lambda2_sweep.csv"
        lambda2   = lambda2_per_fold.get(fold_idx)

        if not sweep_csv.exists():
            print(f"[fold_{fold_idx}] WARNING: sweep CSV not found — run concurvity_reg_fold first.")
            continue

        if lambda2 is None:
            auto_pick = _auto_pick_from_csv(sweep_csv, eta_prune)
            print(f"[fold_{fold_idx}] Not yet confirmed. Auto-pick suggestion: {auto_pick}  "
                  f"(plot: {fold_dir / 'regularized_lambda2_sweep.png'})")
            continue

        print(f"[fold_{fold_idx}] Confirming lambda_2={lambda2}...")
        _confirm_fold(fold_dir, lambda2, eta_prune, reason, overwrite)


if __name__ == "__main__":
    main()