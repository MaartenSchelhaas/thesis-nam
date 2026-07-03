"""
confirm_regularized_arm.py — write regularized_tuned_config.yaml for arm D.

Run AFTER tune_concurvity_reg.py has written the sweep CSV + plot for each
fold. Inspect each fold's plot, set lambda2_per_fold in main(), then run:
    python -m scripts.tuning.confirm_regularized_arm

Per confirmed fold, writes regularized_tuned_config.yaml (mains config +
clarity_regularization copied from gaminet_tuned_config.yaml + the confirmed
concurvity_regularization) and regularized_config_meta.yaml (for tracking only:
confirmed value, reason, timestamp). Folds left as None in lambda2_per_fold are skipped; 
folds missing a sweep CSV print a warning and are skipped.
"""

from datetime import datetime, timezone
from pathlib import Path

import yaml


def _confirm_fold(
    fold_dir: Path,
    lambda2: float,
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

    output_config = mains_raw | {
        "clarity_regularization":    lambda_1,
        "concurvity_regularization": lambda2,
    }
    with open(output_config_yaml, "w") as f:
        yaml.dump(output_config, f, default_flow_style=False, sort_keys=False)

    meta = {
        "confirmed_lambda2":    lambda2,
        "lambda1_from_gaminet": lambda_1,
        "reason":               reason,
        "confirmed_at":         datetime.now(timezone.utc).isoformat(),
    }
    with open(output_meta_yaml, "w") as f:
        yaml.dump(meta, f, default_flow_style=False, sort_keys=False)

    print(f"  lambda_1 (from gaminet) : {lambda_1}")
    print(f"  lambda_2 confirmed      : {lambda2}")
    print(f"  Written: {output_config_yaml}")
    print(f"  Written: {output_meta_yaml}")


def main() -> None:
    # ── Edit these before running ───────────────────────────────────────────
    base_dir  = Path(r"runs/compas_na2m")
    n_folds   = 5
    reason    = ""      # optional note explaining the choice (applied to all folds)
    overwrite = False   # set True to overwrite existing configs

    # Set lambda_2 per fold after inspecting each fold's sweep plot.
    # Leave as None to skip that fold (not yet confirmed).
    lambda2_per_fold = {
        0: 1,
        1: 0.1,
        2: 1,
        3: 1,
        4: 1,
    }
    # ────────────────────────────────────────────────────────────────────────

    for fold_idx in range(n_folds):
        fold_dir  = base_dir / f"fold_{fold_idx}"
        sweep_csv = fold_dir / "regularized_lambda2_sweep.csv"
        lambda2   = lambda2_per_fold.get(fold_idx)

        if not sweep_csv.exists():
            print(f"[fold_{fold_idx}] WARNING: sweep CSV not found — run concurvity_reg_fold first.")
            continue

        if lambda2 is None:
            print(f"[fold_{fold_idx}] Not yet confirmed — inspect: {fold_dir / 'regularized_lambda2_sweep.png'}")
            continue

        print(f"[fold_{fold_idx}] Confirming lambda_2={lambda2}...")
        _confirm_fold(fold_dir, lambda2, reason, overwrite)


if __name__ == "__main__":
    main()