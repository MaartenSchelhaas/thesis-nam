"""
na2m/data/california_housing.py — dataset class for the California Housing dataset.

All 8 features are numerical; no categoricals. Target is median house value
(continuous), so this is a regression task. Data is fetched directly from
sklearn — no CSV needed.
"""

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import MinMaxScaler

from na2m.data.shared import FeatureMeta


class CaliforniaHousingDataset:
    """Load and preprocess the California Housing dataset from sklearn.

    Usage:
        ds = CaliforniaHousingDataset()
        df = ds.load()            # path argument is accepted but ignored
        X, y, feature_meta = ds.preprocess(df)
    """

    NUMERICAL_FEATURES: list[str] = [
        "MedInc",
        "HouseAge",
        "AveRooms",
        "AveBedrms",
        "Population",
        "AveOccup",
        "Latitude",
        "Longitude",
    ]

    CATEGORICAL_FEATURES: list[str] = []

    TARGET_COLUMN: str = "MedHouseVal"
    TASK: str = "regression"

    def load(self, path: str | None = None) -> pd.DataFrame:
        """Fetch the California Housing dataset from sklearn and return as a DataFrame.

        Args:
            path: Accepted for interface compatibility with other dataset classes;
                  always ignored.

        Returns:
            DataFrame with columns matching NUMERICAL_FEATURES + TARGET_COLUMN.
        """
        bunch = fetch_california_housing(as_frame=False)
        df = pd.DataFrame(bunch.data, columns=self.NUMERICAL_FEATURES)
        df[self.TARGET_COLUMN] = bunch.target
        return df

    def preprocess(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, list[FeatureMeta]]:
        """Scale the California Housing DataFrame for NA2M.

        All features are numerical; MinMaxScaler → [0, 1], stored as float32.
        Target is raw median house value (float32), no transformation applied.

        Args:
            df: DataFrame returned by load().

        Returns:
            X:            np.ndarray, shape (n_samples, 8), float32.
            y:            np.ndarray, shape (n_samples,), float32.
            feature_meta: list[FeatureMeta], one entry per column of X, in column order.
        """
        df = df[self.NUMERICAL_FEATURES + [self.TARGET_COLUMN]].dropna().reset_index(drop=True)

        num_scaler = MinMaxScaler()
        X = num_scaler.fit_transform(df[self.NUMERICAL_FEATURES]).astype(np.float32)

        feature_meta = [
            FeatureMeta(type="num", name=name, min=float(num_scaler.data_min_[i]), max=float(num_scaler.data_max_[i]))
            for i, name in enumerate(self.NUMERICAL_FEATURES)
        ]

        y = df[self.TARGET_COLUMN].to_numpy(dtype=np.float32)

        return X, y, feature_meta