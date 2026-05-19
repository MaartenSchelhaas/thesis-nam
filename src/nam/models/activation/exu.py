"""
ExU activation layer — the key building block of NAM.

Introduced in the original NAM paper (Agarwal et al., 2021).
ExU stands for "Exp-centered Unit": it applies a learned exponential
transformation before clipping, giving the network high sensitivity
to small input changes (useful for learning sharp feature shapes).

Forward pass:
    out = relu1((x - bias) @ exp(weight))
    where relu1 = clamp(., min=0, max=1)

Weight init: truncated normal N(mean=4.0, std=0.5)
    — large positive values so exp(weight) >> 1, creating high initial
    sensitivity that the optimizer can dial back as needed.
Bias init: truncated normal N(mean=0, std=0.5)
    — centers the input transformation near zero.

Reference: original_neural_additive_models/models.py  ActivationLayer (ExU branch)
PyTorch reference: nam-main-multitask/nam-main/nam/models/activation/exu.py
"""

import torch
import torch.nn as nn


class ExU(nn.Module):
    """
    ExU activation: (x - bias) @ exp(weight), clipped to [0, 1].

    Args:
        in_features: Input dimensionality (1 for a single-feature subnet).
        num_units:   Number of output units (width of this layer).
    """

    def __init__(self, in_features: int, num_units: int):
        super().__init__()
        # TODO: define self.weight as nn.Parameter, shape (in_features, num_units)
        #       init with truncated normal N(mean=4.0, std=0.5)
        # TODO: define self.bias as nn.Parameter, shape (in_features,)
        #       init with truncated normal N(mean=0, std=0.5)
        # Hint: torch.nn.init has no truncated_normal — sample from normal and
        #       clamp to ±2*std (common approximation used in the reference code)
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TODO: compute (x - self.bias) @ torch.exp(self.weight)
        # TODO: clamp result to [0, 1]  (ReLU-1 / relu1)
        # TODO: return clamped tensor
        raise NotImplementedError
