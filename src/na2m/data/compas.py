"""
na2m/data/compas.py — dataset class for the COMPAS recidivism dataset.

Categoricals are INTEGER-CODED (one column per feature, one CategNet subnet
per feature). Numericals are MinMaxScaled to [0, 1]. Do NOT one-hot here —
CategNet handles that internally.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

from na2m.data.shared import FeatureMeta


class CompasDataset:
    """Load and preprocess the COMPAS recidivism CSV for NA2M.

    Usage:
        ds = CompasDataset()
        df = ds.load(path)
        X, y, feature_meta = ds.preprocess(df)
    """

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
    TASK: str = "classification"

    def load(self, path: str | None = None) -> pd.DataFrame:
        """Load the raw COMPAS CSV from disk.

        Args:
            path: Path to the raw CSV. Required — raises ValueError if None.

        Returns:
            Raw DataFrame with all original columns intact.
        """
        if path is None:
            raise ValueError("CompasDataset.load() requires a path — set dataset_path in your search config.")
        return pd.read_csv(path)

    def preprocess(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, list[FeatureMeta]]:
        """Clean and encode the COMPAS DataFrame for the NA2M route-2 path.

        Numericals: MinMaxScaler → [0, 1], stored as float32.
        Categoricals: LabelEncoder per column → integer codes, stored as int64.
        Column order in X: numericals first, then categoricals (matches feature_meta).

        Args:
            df: Raw DataFrame from load().

        Returns:
            X:            np.ndarray, shape (n_samples, n_features), float32.
                          Categorical codes are small integers stored as float32 (lossless);
                          CategNet must cast its input column to .long() before F.one_hot.
            y:            np.ndarray, shape (n_samples,), int (0 or 1).
            feature_meta: list[FeatureMeta], one entry per column of X, in column order.
        """
        columns = self.NUMERICAL_FEATURES + self.CATEGORICAL_FEATURES + [self.TARGET_COLUMN]
        df['length_of_stay'] = (pd.to_datetime(df['c_jail_out']) - pd.to_datetime(df['c_jail_in'])).dt.days
        df = df[columns].dropna().reset_index(drop=True)

        X_num = df[self.NUMERICAL_FEATURES]
        X_cat = df[self.CATEGORICAL_FEATURES]
        y = df[self.TARGET_COLUMN].to_numpy()

        num_scaler = MinMaxScaler()
        X_num_scaled = num_scaler.fit_transform(X_num).astype(np.float32)

        num_meta = [
            FeatureMeta(type="num", name=name, min=float(num_scaler.data_min_[i]), max=float(num_scaler.data_max_[i]))
            for i, name in enumerate(self.NUMERICAL_FEATURES)
        ]

        cat_meta = []
        X_cat_cols = []
        for name in self.CATEGORICAL_FEATURES:
            le = LabelEncoder()
            codes = np.array(le.fit_transform(X_cat[name]), dtype=np.float32)
            X_cat_cols.append(codes[:, np.newaxis])
            cat_meta.append(FeatureMeta(type="cat", name=name, n_levels=int(len(le.classes_)), levels=le.classes_.tolist()))

        X_cat_encoded = np.concatenate(X_cat_cols, axis=1)
        X = np.concatenate([X_num_scaled, X_cat_encoded], axis=1)
        feature_meta = num_meta + cat_meta

        return X, y, feature_meta