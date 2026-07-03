"""
ExU activation layer — Agarwal et al. (2021), Neural Additive Models.
"""

import torch
import torch.nn as nn


class ExU(nn.Module):
    """Exu activation layer"""

    def __init__(self, in_features: int, out_features: int):
        """Initialize the ExU with random parameters

        Args:
            in_features (int): Amount of input features
            out_features (int): Amount of output features

        Attributes:
            w (nn.Parameter): Weight matrix of shape (in_features, out_features).
            b (nn.Parameter): Bias vector of shape (in_features,).
        """
        super().__init__()
        self.w = nn.Parameter(torch.empty(in_features, out_features))
        self.b = nn.Parameter(torch.empty(in_features))

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Function to initialize the weights and biases.
        """
        nn.init.trunc_normal_(self.w, mean=4.0, std=0.5,a=3, b=5)
        nn.init.trunc_normal_(self.b, std=0.5)

    def forward(self, x: torch.Tensor, n: int =1) -> torch.Tensor:
        """Forward pass for ExU.
        Max value between 0 and n.

        Args:
            x (torch.Tensor): Input of shape (batch_size, in_features).
            n (int): Activation value cap


        Returns:
            torch.Tensor: Output of shape (batch_size, out_features).
        """

        output = (x - self.b) @ torch.exp(self.w)
        output = torch.relu(output)
        output = torch.clamp(output,0,n)
        return output
