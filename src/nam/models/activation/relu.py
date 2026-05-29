"""
LinReLU activation layer — the ReLU alternative to ExU.

A simpler activation: linear transformation followed by ReLU.


Reference: original_neural_additive_models/models.py  ActivationLayer (ReLU branch)
PyTorch reference: nam-main-multitask/nam-main/nam/models/activation/relu.py
"""

import torch
import torch.nn as nn
 

class LinReLU(nn.Module):
    """LinReLU activation layer — linear transformation followed by ReLU."""

    def __init__(self, in_features: int, out_features: int):
        """Initialize the Lin ReLU with random parameters

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
        nn.init.xavier_normal_(self.w)  
        nn.init.trunc_normal_(self.b, std=0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass

        Args:
            x (torch.Tensor): Input of shape (batch_size, in_features).

        Returns:
            torch.Tensor: Output of shape (batch_size, out_features).
        """
        output = torch.relu((x - self.b) @ self.w)
        return output
        

  