"""
FeatureNN — a single-feature neural network subnet.

Each input feature gets its own FeatureNN. This is what makes NAM
additive and interpretable: feature i only ever sees x[:, i], so its
output is a learned function purely of that one variable.

Architecture:
    activation_layer (ExU or LinReLU, in=1 → num_units)
        → [Linear(num_units → num_units) + ReLU + Dropout] × len(hidden_sizes)
        → Linear(num_units → 1, bias=False)

The final linear has no bias because the NAM model adds a global
learnable bias after summing all feature outputs.

Output shape: (batch_size, 1) — kept as column so NAM can torch.cat
across features into (batch_size, num_features).

Reference: original_neural_additive_models/models.py  FeatureNN
PyTorch reference: nam-main-multitask/nam-main/nam/models/featurenn.py
"""

import torch
import torch.nn as nn
from .activation.exu import ExU
from .activation.relu import LinReLU


class FeatureNN(nn.Module):
    """
    Single-feature subnet: maps scalar input x_i → scalar output f_i(x_i).

    Args:
        num_units:    Width of the activation layer (and hidden layers).
        hidden_sizes: List of hidden layer widths after the activation layer.
                      Empty list → shallow network (activation layer + output only).
        dropout:      Dropout probability applied after each hidden layer.
        activation:   'exu' or 'relu' — selects the activation layer type.
    """

    def __init__(
        self,
        num_units: int = 64,
        hidden_sizes: list = [64, 32],
        dropout: float = 0.5,
        activation: str = "exu",
    ):
        super().__init__()
        # TODO: instantiate activation layer:
        #       ExU(in_features=1, num_units=num_units) if activation == 'exu'
        #       LinReLU(in_features=1, num_units=num_units) if activation == 'relu'

        # TODO: build hidden layers as nn.Sequential or nn.ModuleList:
        #       for each size in hidden_sizes:
        #           Linear(num_units → size) + ReLU + Dropout(dropout)
        #       (update num_units to track current width between layers)

        # TODO: output layer: Linear(current_width → 1, bias=False)

        # TODO: store dropout rate for use in forward if needed
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch_size, 1)  — a single feature column
        # TODO: pass through activation layer
        # TODO: pass through each hidden layer (Linear + ReLU + Dropout)
        # TODO: pass through output layer
        # TODO: return output of shape (batch_size, 1)
        raise NotImplementedError
