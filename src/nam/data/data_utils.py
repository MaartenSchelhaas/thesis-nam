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
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.model_selection import train_test_split

#TODO: Bij gebruik van meerdere datasets, abstract base class maken om te implementeren. 

NUMERICAL_FEATURES: list[str] = [
    'age',
    'priors_count',
    'length_of_stay'
]

CATEGORICAL_FEATURES: list[str] = [
    'c_charge_degree',
    'race',
    'sex',
]

TARGET_COLUMN: str = 'two_year_recid'


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

    df = pd.read_csv(path)
    return df



def preprocess(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, list[str]]:
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
    
    columns = NUMERICAL_FEATURES + CATEGORICAL_FEATURES + [TARGET_COLUMN]
    df['length_of_stay'] = (pd.to_datetime(df['c_jail_out']) - pd.to_datetime(df['c_jail_in'])).dt.days
    df = df[columns].dropna().reset_index(drop=True)
    
    # Separate feature groups and target
    X_num = df[NUMERICAL_FEATURES]
    X_cat = df[CATEGORICAL_FEATURES]
    y = df[TARGET_COLUMN].to_numpy()
    

    #Scale numeric features
    num_scaler = MinMaxScaler()
    X_num_scaled = num_scaler.fit_transform(X_num).astype(np.float32)
    
    #One-hot encoding for categorical features
    cat_encoder = OneHotEncoder(sparse_output=False,dtype=np.float32)
    X_cat_encoded = cat_encoder.fit_transform(X_cat)

    # Build the full feature matrix
    X = np.concatenate([X_num_scaled, X_cat_encoded], axis=1)

    # Build human-readable feature names (for plots)
    cat_feature_names = cat_encoder.get_feature_names_out(CATEGORICAL_FEATURES).tolist()
    feature_names = NUMERICAL_FEATURES + cat_feature_names

    return X, y, feature_names
    



def split(
    X: np.ndarray,
    y: np.ndarray,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Split arrays into train / val / test subsets with stratification.

    Stratified splits preserve class balance across all three sets
    
    Args:
        X:         Feature matrix, shape (n_samples, n_features)
        y:         Target vector, shape (n_samples,)
        val_frac:  Fraction of total data reserved for validation
        test_frac: Fraction of total data reserved for test
        seed:      Random seed for reproducibility

    Returns:
        (X_train, X_val, X_test, y_train, y_val, y_test) — all np.ndarray
    """
    # Stage 1: split off test set
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y,
        test_size=test_frac,
        stratify=y,
        random_state=seed,
    )

    # Stage 2: split train/val from the remaining data
    # val_frac is a fraction of the *total* data, so we need to adjust
    val_frac_of_remainder = val_frac / (1.0 - test_frac)
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval,
        test_size=val_frac_of_remainder,
        stratify=y_trainval,
        random_state=seed,
    )

    return X_train, X_val, X_test, y_train, y_val, y_test

