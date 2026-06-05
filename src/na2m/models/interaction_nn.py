"""
InteractionNN — a pairwise (2-input) interaction subnet for NA2M.

A plain 2-input ReLU MLP (NOT ExU). One InteractionNN per selected feature pair
(j, k). First layer Linear(2, num_units); final Linear(width, 1, bias=False) so
the global NA2M bias owns the intercept.

Output shape: (batch_size, 1) — mirrors the main-term subnets so terms concat.

Reference: GamiNet-master interaction subnetwork (TensorFlow).
"""

import torch
import torch.nn as nn


class InteractionNN(nn.Module):
    """
    Pairwise interaction subnet: maps (x_j, x_k) → scalar effect f_jk.
    """

    def __init__(self, num_units: int, hidden_sizes: list, dropout: float):
        """Initialize the interaction subnet.

        Args:
            num_units (int): Width of the first layer (Linear(2, num_units) + ReLU).
            hidden_sizes (list): Hidden layer widths after the first layer.
            dropout (float): Dropout probability applied after each hidden layer.

        TODO:
            - First layer: Linear(2, num_units) + ReLU (+ Dropout).
            - Hidden layers: Linear + ReLU + Dropout per width in hidden_sizes.
            - Final layer: Linear(width, 1, bias=False).
            - Wrap in nn.Sequential.

        CATEGORICAL-INPUT TODO (flag, do not silently pick):
            When a feature in the pair is categorical, its input column is a LEVEL
            INDEX, not a [0,1] scalar. Two candidate designs:
              (a) embed the level index (nn.Embedding) before the 2-input net, or
              (b) pass the raw index through as a float.
            These are NOT equivalent. Decide explicitly when wiring na2m.inter_outputs.
        """
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the 2-input MLP.

        Args:
            x (torch.Tensor): Paired inputs, shape (batch_size, 2).

        Returns:
            torch.Tensor: Interaction effect, shape (batch_size, 1).

        TODO:
            - return self.model(x).
        """
        raise NotImplementedError
