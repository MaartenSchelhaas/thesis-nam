"""
confirm_regularized_arm.py — write regularized_tuned_config.yaml for arm D.

Run AFTER tune_concurvity_reg.py has written the sweep CSV and tradeoff plot.
Inspect the plot, choose a lambda_2 value, set it below, and run this script.

What this writes:
    regularized_tuned_config.yaml
        All fields from mains_tuned_config.yaml, plus:
            clarity_regularization  — copied from gaminet_tuned_config.yaml (lambda_1)
            concurvity_regularization — the confirmed lambda_2 value set below

    regularized_config_meta.yaml
        Source-tracking only (not loaded by NA2MConfig):
            confirmed_lambda2    — the value that was written
            lambda2_auto_pick    — elbow suggestion recomputed from the sweep CSV
            lambda1_from_gaminet — the clarity_regularization value copied from gaminet
            reason               — optional human note
            confirmed_at         — ISO timestamp

Usage:
    Edit the variables at the top of main() then run:
        python scripts/na2m/confirm_regularized_arm.py
"""

import csv
from datetime import datetime
from pathlib import Path

import yaml

from na2m.training.fit_na2m import _eta_cut


def _recompute_auto_pick(sweep_csv: Path, eta_prune: float) -> float:
    """Recompute the elbow suggestion from the sweep CSV.

    Reads (lambda2, seed, val_loss) rows, averages val_loss per lambda_2 (ascending),
    and applies the same eta_prune elbow rule used during the sweep.

    Args:
        sweep_csv: Path to regularized_lambda2_sweep.csv.
        eta_prune: Tolerance for the elbow rule.

    Returns:
        Auto-suggested lambda_2 value.
    """
    # TODO: read CSV; group by lambda2; compute mean val_loss per lambda2
    # TODO: sort by lambda2 ascending
    # TODO: apply _eta_cut(mean_val_losses, eta_prune) -> cut index
    # TODO: return sorted_lambda2_grid[cut_index]
    raise NotImplementedError


def main() -> None:
    # ── Edit these before running ───────────────────────────────────────────
    fold_dir  = Path(r"runs/compas/fold_0")
    lambda2   = 0.01    # confirmed lambda_2 value (inspect the sweep plot first)
    reason    = ""      # optional note explaining the choice
    overwrite = False   # set True to overwrite an existing config
    # ────────────────────────────────────────────────────────────────────────

    mains_config_yaml    = fold_dir / "mains_tuned_config.yaml"
    gaminet_config_yaml  = fold_dir / "gaminet_tuned_config.yaml"
    sweep_csv            = fold_dir / "regularized_lambda2_sweep.csv"
    output_config_yaml   = fold_dir / "regularized_tuned_config.yaml"
    output_meta_yaml     = fold_dir / "regularized_config_meta.yaml"

    # --- Guard: required inputs must exist ---
    for p in [mains_config_yaml, gaminet_config_yaml, sweep_csv]:
        if not p.exists():
            raise FileNotFoundError(
                f"{p} not found. Run tune_concurvity_reg.py for this fold first."
            )

    if output_config_yaml.exists() and not overwrite:
        raise FileExistsError(
            f"{output_config_yaml} already exists. Set overwrite=True to replace it."
        )

    # TODO: load mains_config_yaml via yaml.safe_load
    # TODO: load gaminet_config_yaml via yaml.safe_load; extract lambda_1 = gaminet_raw["clarity_regularization"]
    # TODO: load eta_prune from the search YAML (or fall back to 0.0) for _recompute_auto_pick
    # TODO: auto_pick = _recompute_auto_pick(sweep_csv, eta_prune)

    # TODO: build output config dict: mains_raw | {"clarity_regularization": lambda_1, "concurvity_regularization": lambda2}
    # TODO: write output_config_yaml

    # TODO: build meta dict: confirmed_lambda2, lambda2_auto_pick, lambda1_from_gaminet, reason, confirmed_at
    # TODO: write output_meta_yaml

    # TODO: print confirmation summary (lambda_1, lambda_2, auto_pick, paths written)
    raise NotImplementedError


if __name__ == "__main__":
    main()