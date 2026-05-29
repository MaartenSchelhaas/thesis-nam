"""
FeatureNN — a single-feature neural network subnet.

Each input feature gets its own FeatureNN.

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
from .activation import ExU, LinReLU


class FeatureNN(nn.Module):
    """
    Single-feature subnet: maps scalar input x_i → scalar output f_i(x_i).
    """

    def __init__(
        self,
        num_units: int = 64,
        hidden_sizes: list = [64, 32],
        dropout: float = 0.5,
        activation: str = "exu",
    ):
        """Initialize the feature model

        Args:
            num_units (int, optional): Width of the activation layer. Defaults to 64.
            hidden_sizes (list, optional):  List of hidden layer widths after the activation layer.
                                            Empty list → shallow network (activation layer + output only).
                                            Defaults to [64, 32]
            dropout (float, optional): Dropout probability applied after each hidden layer. Defaults to 0.5.
            activation (str, optional): 'exu' or 'relu', selects the activation layer type. Defaults to "exu".
        """
        super().__init__()
        
        self.dropout = nn.Dropout(p=dropout)
        layers = []

        #First layer
        if activation == "exu":
            layers.append(ExU(in_features=1, out_features=num_units))
        elif activation == "relu":
            layers.append(LinReLU(in_features=1, out_features=num_units))
        else:
            raise ValueError(f"Unknown activation '{activation}'. Use 'exu' or 'relu'.")
        layers.append(self.dropout)

        #Hidden layers
        current_width = num_units
        for size in hidden_sizes:
            layers.append(nn.Linear(in_features=current_width, out_features=size))
            layers.append(nn.ReLU())
            layers.append(self.dropout)
            current_width = size

        #Last layer
        layers.append(nn.Linear(in_features=current_width, out_features=1,bias=False))

        self.model = nn.Sequential(*layers)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass trough the model

        Args:
            x (torch.Tensor): Input value to pass through the model

        Returns:
            torch.Tensor: Individual contribution of the input value to the prediction. 
        """
        return self.model(x)
