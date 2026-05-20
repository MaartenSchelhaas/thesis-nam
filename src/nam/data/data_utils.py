"""
Data utilities for the COMPAS recidivism dataset.

Handles loading, preprocessing, and splitting the raw CSV into
train/val/test numpy arrays ready to be wrapped in NAMDataset.

Reference: original_neural_additive_models/nam_train.py for the
TensorFlow preprocessing pipeline we are mirroring.

Preprocessing conventions (matching the original):
- Numerical features: MinMaxScaler → [0, 1]
- Categorical features: OneHotEncoder (drop first to avoid collinearity)
- Target: binary int (0 = no recidivism, 1 = recidivism)
"""

import numpy as np
import pandas as pd


def load_compas(path: str) -> pd.DataFrame:
    """
    Load the raw COMPAS CSV from disk.

    Args:
        path: Path to the raw CSV, e.g. 'datasets/raw/compas-scores-two-years.csv'

    Returns:
        Raw DataFrame with all original columns intact.

    TODO:
        - pd.read_csv(path)
        - Return as-is (no filtering here)
    """
    raise NotImplementedError


def preprocess(df: pd.DataFrame):
    """
    Clean and encode the COMPAS DataFrame into model-ready arrays.

    Steps:
        1. Select relevant feature columns and target column
        2. Drop rows with missing values
        3. Scale numerical columns with MinMaxScaler to [0, 1]
        4. Encode categorical columns with OneHotEncoder
        5. Concatenate encoded arrays horizontally
        6. Extract binary target (two_year_recid)

    Args:
        df: Raw DataFrame from load_compas()

    Returns:
        X            : np.ndarray, shape (n_samples, n_features), float32
        y            : np.ndarray, shape (n_samples,), int (0 or 1)
        feature_names: list[str], one name per column in X
                       (needed later for per-feature interpretability plots)

    TODO:
        - Define NUMERICAL_FEATURES and CATEGORICAL_FEATURES lists
          (refer to original_neural_additive_models for column selection)
        - Fit MinMaxScaler on numerical columns, transform
        - Fit OneHotEncoder on categorical columns, transform
        - Build feature_names: numerical names + encoder.get_feature_names_out()
        - Return (X.astype(np.float32), y, feature_names)
    """
    raise NotImplementedError


def split(
    X: np.ndarray,
    y: np.ndarray,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
    seed: int = 42,
):
    """
    Split arrays into train / val / test subsets with stratification.

    Stratified so class balance is preserved across all three sets —
    important for binary classification on imbalanced data.

    Args:
        X:         Feature matrix, shape (n_samples, n_features)
        y:         Target vector, shape (n_samples,)
        val_frac:  Fraction of total data reserved for validation
        test_frac: Fraction of total data reserved for test
        seed:      Random seed for reproducibility

    Returns:
        (X_train, X_val, X_test, y_train, y_val, y_test) — all np.ndarray

    TODO:
        - Use sklearn train_test_split with stratify=y
        - Stage 1: split off test set → test_frac of total
        - Stage 2: split remainder → val_frac_of_remainder = val_frac / (1 - test_frac)
        - Return six arrays in order: train, val, test
    """
    raise NotImplementedError

