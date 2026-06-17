"""
NAMDataset — generic PyTorch Dataset wrapper for tables.

Converts preprocessed numpy arrays (X, y) into a torch Dataset so they
can be fed to a DataLoader for batching and shuffling.
"""

import numpy as np
import torch
from torch.utils.data import Dataset


class NAMDataset(Dataset):
    """
    Wraps feature matrix X and target vector y as a torch Dataset.

    Args:
        X:       Feature matrix, shape (n_samples, n_features). numpy float32.
        y:       Target vector, shape (n_samples,). Binary 0/1 for classification.
        weights: Optional per-sample weights, shape (n_samples,).
                 Defaults to all-ones. Pass 0 for samples with missing labels.
        feature_meta: Meta-data for NA2M with different categorical encoding

    Each __getitem__ returns:
        features : FloatTensor of shape (n_features,)
        target   : FloatTensor scalar
        weight   : FloatTensor scalar
    """

    def __init__(
            self,
            X: np.ndarray,
            y: np.ndarray,
            weights: np.ndarray | None = None,
            feature_meta: list | None = None):

        if np.isnan(y).any():
            raise ValueError("y contains NaN values — run dropna() before creating NAMDataset")

        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        self.feature_meta = feature_meta

        if weights is None:
            # Default: all ones — every sample is equally weighted.
            w = torch.ones_like(self.y)
        else:
            w = torch.tensor(weights, dtype=torch.float32)

        self.w = w

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor,torch.Tensor,torch.Tensor]:
        return self.X[idx], self.y[idx], self.w[idx]