"""
evaluate_nam.py — Evaluate the NA2M main-effects-only arm (arm A) on COMPAS.

Runs Stage 1 only (main effects, no interactions) and reports test AUROC.
Use this to verify that the main-effect subnet inside NA2M reproduces the
standalone NAM from src/nam/ before removing the duplicate.

Edit the variables at the top of main() to point at your tuned arm-A config:
    _TUNED_CONFIG — path to the YAML produced by tune_na2m.py with
                    _WITH_INTERACTIONS = False
    _SEED         — must match the SEED used in the comparison notebook.

Usage:
    python scripts/na2m/evaluate_nam.py
"""

import numpy as np
import torch
from pathlib import Path
from torch.utils.data import DataLoader

from na2m.data.data_utils import load_compas, preprocess, split
from na2m.models.na2m import NA2M
from na2m.training.fit_na2m import fit_na2m
from na2m.utils.config import load_na2m_config
from nam.data.dataset import NAMDataset
from nam.training.metrics import auroc


def evaluate(
    config,
    feature_meta,
    X_train,
    y_train,
    X_val,
    y_val,
    X_test,
    y_test,
    *,
    seed: int,
) -> float:
    """Train arm A (Stage 1 only) and return test AUROC.

    Args:
        config: Populated NA2MConfig.
        feature_meta: FeatureMeta list from preprocess().
        X_train, y_train: Training split.
        X_val, y_val: Validation split.
        X_test, y_test: Held-out test split.
        seed: Replicate seed for init + optimisation.

    Returns:
        Test AUROC as a float.
    """
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    train_loader = DataLoader(
        NAMDataset(X_train, y_train),
        batch_size=config.batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        NAMDataset(X_val, y_val),
        batch_size=config.batch_size,
        shuffle=False,
    )
    pool_loader = DataLoader(
        NAMDataset(np.concatenate([X_train, X_val]), np.concatenate([y_train, y_val])),
        batch_size=config.batch_size,
        shuffle=False,
    )
    test_loader = DataLoader(
        NAMDataset(X_test, y_test),
        batch_size=config.batch_size,
        shuffle=False,
    )

    model = NA2M(
        num_features=X_train.shape[1],
        feature_meta=feature_meta,
        num_units=config.num_units,
        hidden_sizes=config.hidden_sizes,
        dropout=config.dropout,
        feature_dropout=config.feature_dropout,
        activation=config.activation,
        inter_units=config.inter_units,
        inter_hidden=config.inter_hidden,
    )

    fit_na2m(
        model,
        train_loader,
        val_loader,
        pool_loader,
        config,
        with_interactions=False,
        with_concurvity_filter=False,
    )

    model.eval()
    all_logits, all_targets = [], []
    with torch.no_grad():
        for X_batch, y_batch, _ in test_loader:
            logits, _ = model(X_batch)
            all_logits.append(logits)
            all_targets.append(y_batch)

    return float(auroc(torch.cat(all_logits), torch.cat(all_targets)))


def main() -> None:
    # --- Edit these ---
    _TUNED_CONFIG = r"C:\Users\maart\OneDrive\Documenten\Universiteit\Scriptie\python_repo\thesis-nam\configs\compas-scores-two-years_na2m_tuned.yaml"
    _SEED = 42
    # ------------------

    config = load_na2m_config(_TUNED_CONFIG)

    df = load_compas(config.dataset_path)
    X, y, feature_meta = preprocess(df)

    X_train, X_val, X_test, y_train, y_val, y_test = split(
        X,
        y,
        config.val_frac,
        config.test_frac,
        seed=_SEED,
    )
    print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

    auc = evaluate(
        config,
        feature_meta,
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        y_test,
        seed=_SEED,
    )
    print(f"\nNA2M arm A (mains only) — test AUROC: {auc:.4f}")
    print("Compare this to the standalone NAM test AUROC to verify equivalence.")


if __name__ == "__main__":
    main()
