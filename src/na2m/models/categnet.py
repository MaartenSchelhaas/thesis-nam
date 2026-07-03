"""
CategNet — single-categorical-feature subnet for NA2M.

Takes an integer level index, one-hot encodes it internally (F.one_hot),
and maps it to a scalar effect via a single Linear(n_levels, 1, bias=False).

This is equivalent to FeatureNN reduced to just its output layer. No hidden
layers are needed: one-hot inputs are mutually exclusive, so a network of any
depth still collapses to one learned scalar per level.

No output bias — the global NA2M bias owns the intercept.
Output shape: (batch_size, 1) — identical to FeatureNN so terms concatenate.

Reference: GAMI-Net's CategNet (Yang et al., 2021), reimplemented in PyTorch.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CategNet(nn.Module):
    """Single-categorical-feature subnet: integer level index → scalar effect.
    
    Equivalent to FeatureNN reduced to just its output layer, with n_levels 
    input dimensions instead of 1. One learned scalar weight per level,
    trained via backprop — gradients flow only to the active level per sample.
    No hidden layers needed: one-hot inputs are mutually exclusive so hidden
    layers cannot learn cross-level interactions.
    """

    def __init__(self, n_levels: int):
        """
        Args:
            n_levels (int): Number of distinct levels for this categorical feature.
                            Equivalent to in_features in FeatureNN's output layer.
        """
        super().__init__()
        self.n_levels = n_levels
        # Mirrors FeatureNN's output layer: Linear(current_width, 1, bias=False)
        # but current_width = n_levels since no hidden layers are needed
        self.output_layer = nn.Linear(in_features=n_levels, out_features=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Integer level indices, shape (batch_size, 1) or (batch_size,).

        Returns:
            torch.Tensor: Per-level effect, shape (batch_size, 1).
        """
        x = x.long().flatten()                               # (batch_size,)
        x = F.one_hot(x, num_classes=self.n_levels).float() # (batch_size, n_levels)
        return self.output_layer(x)                          # (batch_size, 1)