"""
Route 2 data utilities — integer-coded categorical encoding for NA2M.

A separate dataset path from data_utils.py (the frozen one-hot NAM path).
Here categoricals are INTEGER-CODED (one column each, one CategNet subnet per
categorical feature) and numericals are MinMax-scaled to [0, 1]. Do NOT one-hot
in this path.

Each feature carries a `feature_meta` record describing its type and the
information the harness needs to build grids and inverse-transform back to real
units for plotting.

Reference for column selection: data_utils.py (NUMERICAL_FEATURES /
CATEGORICAL_FEATURES / TARGET_COLUMN).
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class FeatureMeta:
    """Per-feature metadata for the route-2 dataset path.

    Attributes:
        type: 'num' or 'cat'.
        name: Human-readable feature name.
        levels: (cat only) ordered list of original labels, index = integer code.
        n_levels: (cat only) number of distinct levels.
        min: (num only) data minimum in real units (for grid + inverse-transform).
        max: (num only) data maximum in real units.
    """
    type: str
    name: str
    levels: list | None = None
    n_levels: int | None = None
    min: float | None = None
    max: float | None = None


def load_compas(path: str) -> pd.DataFrame:
    """
    Load the raw COMPAS CSV from disk (reuse data_utils.load_compas logic).

    Args:
        path: Path to the raw CSV.

    Returns:
        Raw DataFrame with all original columns intact.

    TODO:
        - Reuse the existing loader logic from nam.data.data_utils.load_compas.
    """
    raise NotImplementedError


def preprocess_route2(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, list[FeatureMeta]]:
    """
    Clean and encode the COMPAS DataFrame into integer-coded model-ready arrays.

    Numericals: MinMaxScaler to [0, 1]; KEEP the fitted scaler (harness needs it
    per fold for inverse-transform). Categoricals: LabelEncoder per column, ONE
    column each — do NOT one-hot. X columns are integer-coded-cat + scaled-num.

    Args:
        df: Raw DataFrame from load_compas().

    Returns:
        X            : np.ndarray, shape (n_samples, n_features).
        y            : np.ndarray, shape (n_samples,), int (0 or 1).
        feature_meta : list[FeatureMeta], one per column of X, in column order.

    TODO:
        - Select numerical + categorical + target columns (see data_utils.py).
        - Drop rows with missing values.
        - Numericals: fit MinMaxScaler to [0, 1]; record real-unit min/max in meta.
        - Categoricals: LabelEncoder per column → integer codes; record ordered levels.
        - Assemble X (column order must match the returned feature_meta order).
        - Decide where the fitted scaler is surfaced to the harness (return / attach).
        - Return (X, y, feature_meta).
    """
    raise NotImplementedError


def make_grid(feature_meta: list[FeatureMeta], j: int, G: int = 100) -> np.ndarray:
    """
    Build the evaluation grid for feature j.

    Numerical → linspace over [0, 1] (model space) with a stored real-unit axis
    available via inverse-transform. Categorical → arange(n_levels).

    Args:
        feature_meta: Per-feature metadata list.
        j: Feature index.
        G: Number of grid points for numerical features.

    Returns:
        1-D grid array: (G,) for numerical, (n_levels,) for categorical.

    TODO:
        - Branch on feature_meta[j].type.
        - num: np.linspace(0, 1, G); keep a real-unit axis via inverse-transform.
        - cat: np.arange(feature_meta[j].n_levels).
    """
    raise NotImplementedError


def density_weights(
    X_pool: np.ndarray,
    feature_meta: list[FeatureMeta],
    j: int,
    grid: np.ndarray,
) -> np.ndarray:
    """
    Compute density weights for feature j over its grid, on the 80% train pool.

    Numerical → normalized histogram of X_pool[:, j] over grid bins.
    Categorical → normalized level frequencies.

    Computed ON THE 80% POOL, per fold.

    Args:
        X_pool: The fold's 80% train pool feature matrix.
        feature_meta: Per-feature metadata list.
        j: Feature index.
        grid: The grid returned by make_grid for feature j.

    Returns:
        Normalized weight vector aligned to `grid`.

    TODO:
        - num: histogram X_pool[:, j] over grid bins, normalize to sum 1.
        - cat: count level frequencies, normalize to sum 1.
    """
    raise NotImplementedError
