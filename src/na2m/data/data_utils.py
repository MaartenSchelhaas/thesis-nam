"""
na2m/data/data_utils.py — preprocessing for the NA2M route-2 dataset path.

Categoricals are INTEGER-CODED (one column per feature, one CategNet subnet
per feature). Numericals are MinMaxScaled to [0, 1]. Do NOT one-hot here —
CategNet handles that internally.

Stands alone: does not import from nam.data.data_utils.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from dataclasses import dataclass


NUMERICAL_FEATURES: list[str] = [
    "age",
    "priors_count",
    "length_of_stay",
]

CATEGORICAL_FEATURES: list[str] = [
    "c_charge_degree",
    "race",
    "sex",
]

TARGET_COLUMN: str = "two_year_recid"


@dataclass
class FeatureMeta:
    """Per-feature metadata produced by preprocess().

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


def load_compas(path: str) -> pd.DataFrame:
    """Load the raw COMPAS CSV from disk.

    Args:
        path: Path to the raw CSV.

    Returns:
        Raw DataFrame with all original columns intact.
    """
    return pd.read_csv(path)


def preprocess(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, list[FeatureMeta]]:
    """Clean and encode the COMPAS DataFrame for the NA2M route-2 path.

    Numericals: MinMaxScaler → [0, 1], stored as float32.
    Categoricals: LabelEncoder per column → integer codes, stored as int64.
    Column order in X: numericals first, then categoricals (matches feature_meta).

    Args:
        df: Raw DataFrame from load_compas().

    Returns:
        X:            np.ndarray, shape (n_samples, n_features), float32.
                      Categorical codes are small integers stored as float32 (lossless);
                      CategNet must cast its input column to .long() before F.one_hot.
        y:            np.ndarray, shape (n_samples,), int (0 or 1).
        feature_meta: list[FeatureMeta], one entry per column of X, in column order.
    """
    columns = NUMERICAL_FEATURES + CATEGORICAL_FEATURES + [TARGET_COLUMN]
    df['length_of_stay'] = (pd.to_datetime(df['c_jail_out']) - pd.to_datetime(df['c_jail_in'])).dt.days
    df = df[columns].dropna().reset_index(drop=True)

    # Separate feature groups and target
    X_num = df[NUMERICAL_FEATURES]
    X_cat = df[CATEGORICAL_FEATURES]
    y = df[TARGET_COLUMN].to_numpy()

    # Scale numerical features
    num_scaler = MinMaxScaler()
    X_num_scaled = num_scaler.fit_transform(X_num).astype(np.float32)

    num_meta = [
        FeatureMeta(type="num", name=name, min=float(num_scaler.data_min_[i]), max=float(num_scaler.data_max_[i]))
        for i, name in enumerate(NUMERICAL_FEATURES)
    ]

    # Integer-code categorical features (one column per feature, CategNet handles one-hot internally)
    cat_meta = []
    X_cat_cols = []
    for name in CATEGORICAL_FEATURES:
        le = LabelEncoder()
        codes = np.array(le.fit_transform(X_cat[name]), dtype=np.float32)
        X_cat_cols.append(codes[:, np.newaxis])
        cat_meta.append(FeatureMeta(type="cat", name=name, n_levels=int(len(le.classes_)), levels=le.classes_.tolist()))

    X_cat_encoded = np.concatenate(X_cat_cols, axis=1)

    # Build the full feature matrix
    X = np.concatenate([X_num_scaled, X_cat_encoded], axis=1)
    feature_meta = num_meta + cat_meta

    return X, y, feature_meta


def split(
    X: np.ndarray,
    y: np.ndarray,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split into stratified train / val / test subsets.

    Args:
        X:         Feature matrix, shape (n_samples, n_features).
        y:         Target vector, shape (n_samples,).
        val_frac:  Fraction of total data for validation.
        test_frac: Fraction of total data for test.
        seed:      Random seed.

    Returns:
        (X_train, X_val, X_test, y_train, y_val, y_test)
    """
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=test_frac, stratify=y, random_state=seed
    )
    val_frac_of_remainder = val_frac / (1.0 - test_frac)
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval,
        test_size=val_frac_of_remainder,
        stratify=y_trainval,
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


def density_weights(
    X_pool: np.ndarray,
    feature_meta: list[FeatureMeta],
    j: int,
    grid: np.ndarray,
) -> np.ndarray:
    """
    #TODO: Ignore for now, implement at the end if necessary.
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
