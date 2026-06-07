"""
FeatureNN — single-feature neural network subnet for NAM.

Architecture:
    activation_layer (ExU or LinReLU, in=in_features → num_units)
        → [Linear(current_width → size) + ReLU + Dropout] × len(hidden_sizes)
        → Linear(current_width → 1, bias=False)

No output bias — the global NAM bias owns the intercept.
Output shape: (batch_size, 1) — identical across all subnets so NAM can
torch.cat across features into (batch_size, num_features).

Reference: original_neural_additive_models/models.py FeatureNN
"""

import torch
import torch.nn as nn
from nam.models.activation import ExU, LinReLU


class FeatureNN(nn.Module):
    """Single-feature subnet: maps scalar input x_i → scalar output f_i(x_i)."""

    # TODO: moving_mean tracking for the marginal clarity penalty (NA2M stage 3).
    # Pattern mirrors GamiNet-master NAMNet.call() — track subnet_mean during
    # training, update a moving_mean buffer, expose it to NA2M.clarity_loss().
    # Implement here and in CategNet at the same time so both expose the same
    # interface. Do not add until clarity_loss() is being implemented.

    def __init__(
        self,
        num_units: int = 64,
        hidden_sizes: list = [64, 32],
        dropout: float = 0.5,
        activation: str = "exu",
        in_features: int = 1,
    ):
        """
        Args:
            num_units (int): Width of the activation layer. Defaults to 64.
            hidden_sizes (list): Hidden layer widths after the activation layer.
                                 Empty list → shallow (activation layer + output only).
            dropout (float): Dropout probability after each hidden layer. Defaults to 0.5.
            activation (str): 'exu' or 'relu'. Defaults to 'exu'.
            in_features (int): Input width of the activation layer. Defaults to 1.
        """
        super().__init__()

        self.dropout = nn.Dropout(p=dropout)
        layers = []

        # First layer
        if activation == "exu":
            layers.append(ExU(in_features=in_features, out_features=num_units))
        elif activation == "relu":
            layers.append(LinReLU(in_features=in_features, out_features=num_units))
        else:
            raise ValueError(f"Unknown activation '{activation}'. Use 'exu' or 'relu'.")
        layers.append(self.dropout)

        # Hidden layers
        current_width = num_units
        for size in hidden_sizes:
            layers.append(nn.Linear(in_features=current_width, out_features=size))
            layers.append(nn.ReLU())
            layers.append(self.dropout)
            current_width = size

        # Output layer — no bias, global NAM bias owns the intercept
        layers.append(nn.Linear(in_features=current_width, out_features=1, bias=False))

        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input tensor, shape (batch_size, in_features).

        Returns:
            torch.Tensor: Per-feature effect, shape (batch_size, 1).
        """
        return self.model(x)