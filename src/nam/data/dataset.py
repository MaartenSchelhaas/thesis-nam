"""
NAMDataset — generic PyTorch Dataset wrapper for tabular data.

Converts preprocessed numpy arrays (X, y) into a torch Dataset so they
can be fed to a DataLoader for batching and shuffling.

Supports optional per-sample weights. A weight of 0 means the sample
contributes nothing to the loss — used to handle NaN labels in future
multitask extensions without changing the training loop.
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

    Each __getitem__ returns:
        features : FloatTensor of shape (n_features,)
        target   : FloatTensor scalar
        weight   : FloatTensor scalar
    """

    def __init__(self, X: np.ndarray, y: np.ndarray, weights: np.ndarray = None):
        # TODO: cast X to float32 tensor
        # TODO: cast y to float32 tensor
        # TODO: if weights is None, create all-ones float32 tensor of shape (n_samples,)
        #       otherwise cast provided weights to float32 tensor
        # TODO: assign to self.X, self.y, self.weights
        raise NotImplementedError

    def __len__(self) -> int:
        # TODO: return number of samples (first dimension of self.X)
        raise NotImplementedError

    def __getitem__(self, idx: int):
        # TODO: return tuple (self.X[idx], self.y[idx], self.weights[idx])
        raise NotImplementedError
