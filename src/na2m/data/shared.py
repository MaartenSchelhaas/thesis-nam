"""
na2m/data/shared.py — data utilities shared across all dataset implementations.

Contains FeatureMeta (per-feature metadata) and make_grid (evaluation grid
builder). Both are dataset-agnostic; every dataset class in this package
imports from here.
"""

import numpy as np
from dataclasses import dataclass
from sklearn.model_selection import train_test_split


@dataclass
class FeatureMeta:
    """Per-feature metadata produced by a dataset's preprocess() method.

    Attributes:
        type:     'num' or 'cat'.
        name:     Human-readable feature name.
        n_levels: (cat only) number of distinct levels — needed by CategNet.__init__.
        levels:   (cat only) ordered list of original labels; index == integer code.
        min:      (num only) data minimum in real units (before scaling).
        max:      (num only) data maximum in real units (before scaling).
    """
    type: str
    name: str
    n_levels: int | None = None
    levels: list | None = None
    min: float | None = None
    max: float | None = None


def split(
    X: np.ndarray,
    y: np.ndarray,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
    seed: int = 42,
    *,
    stratify: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split into train / val / test subsets.

    Test is split off first, then val is split from the remainder so the
    fractions are relative to the full dataset.

    Args:
        X:         Feature matrix, shape (n_samples, n_features).
        y:         Target vector, shape (n_samples,).
        val_frac:  Fraction of total data for validation.
        test_frac: Fraction of total data for test.
        seed:      Random seed.
        stratify:  If True, stratify splits by y. Set False for regression targets.

    Returns:
        (X_train, X_val, X_test, y_train, y_val, y_test)
    """
    strat = y if stratify else None
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=test_frac, stratify=strat, random_state=seed
    )
    strat_inner = y_trainval if stratify else None
    val_frac_of_remainder = val_frac / (1.0 - test_frac)
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval,
        test_size=val_frac_of_remainder,
        stratify=strat_inner,
        random_state=seed,
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def make_grid(feature_meta: list[FeatureMeta], j: int, G: int = 256) -> np.ndarray:
    """Build the evaluation grid for feature j in model space.

    Returns the grid the subnet is evaluated on. Inverse-transform to real units
    is the caller's responsibility (extract_measures applies it for curve storage).

    Args:
        feature_meta: Per-feature metadata list.
        j: Feature index.
        G: Number of grid points for numerical features (ignored for categorical).

    Returns:
        1-D grid in model space:
            num → (G,) float32, linspace [0, 1]
            cat → (n_levels,) float32, integer codes 0 … n_levels-1
    """
    meta = feature_meta[j]
    if meta.type == "num":
        return np.linspace(0, 1, G, dtype=np.float32)
    assert meta.n_levels is not None, f"categorical feature '{meta.name}' missing n_levels"
    return np.arange(meta.n_levels, dtype=np.float32)