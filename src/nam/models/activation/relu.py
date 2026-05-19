"""
LinReLU activation layer — the ReLU alternative to ExU.

A simpler activation: linear transformation followed by ReLU.
Used when ExU's high-sensitivity initialization is not desired,
or as an ablation baseline to compare against ExU.

Forward pass:
    out = relu((x - bias) @ weight)

Weight init: Xavier uniform (glorot_uniform in the TF reference)
Bias init: truncated normal N(mean=0, std=0.5)

Reference: original_neural_additive_models/models.py  ActivationLayer (ReLU branch)
PyTorch reference: nam-main-multitask/nam-main/nam/models/activation/relu.py
"""

import torch
import torch.nn as nn


class LinReLU(nn.Module):
    """
    LinReLU activation: (x - bias) @ weight, then ReLU.

    Args:
        in_features: Input dimensionality (1 for a single-feature subnet).
        num_units:   Number of output units (width of this layer).
    """

    def __init__(self, in_features: int, num_units: int):
        super().__init__()
        # TODO: define self.weight as nn.Parameter, shape (in_features, num_units)
        #       init with nn.init.xavier_uniform_
        # TODO: define self.bias as nn.Parameter, shape (in_features,)
        #       init with truncated normal N(mean=0, std=0.5)
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TODO: compute (x - self.bias) @ self.weight
        # TODO: apply torch.relu
        # TODO: return result
        raise NotImplementedError
