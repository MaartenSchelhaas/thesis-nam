"""
CategNet — a single-categorical-feature subnet for NA2M (route 2).

Input is an INTEGER level index (not one-hot in the dataset path). Internally
the index is one-hot encoded to n_levels, then a single linear layer maps to a
scalar effect.

Bias handling: NO per-subnet bias that competes with the model-wide intercept.
Keep the per-level effects, let the global NA2M bias own the intercept.

Output shape: (batch_size, 1) — mirrors FeatureNN so terms concatenate.

Reference: GamiNet-master CategNet (TensorFlow).
"""

import torch
import torch.nn as nn


class CategNet(nn.Module):
    """
    Single-categorical-feature subnet: integer level index → scalar effect.
    """

    def __init__(self, n_levels: int):
        """Initialize the categorical subnet.

        Args:
            n_levels (int): Number of distinct levels for this categorical feature.

        TODO:
            - Store n_levels.
            - Single Linear(n_levels, 1, bias=False) — global bias owns the intercept.
        """
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: one-hot the level index, then linear to a scalar.

        Args:
            x (torch.Tensor): Integer level indices, shape (batch_size, 1) or (batch_size,).

        Returns:
            torch.Tensor: Per-level effect, shape (batch_size, 1).

        TODO:
            - Cast x to long, flatten to (batch_size,).
            - F.one_hot(x, num_classes=self.n_levels) → float.
            - Linear layer → (batch_size, 1).
        """
        raise NotImplementedError
