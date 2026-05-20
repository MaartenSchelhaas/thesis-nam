"""
NAM — Neural Additive Model.

Aggregates one FeatureNN per input feature into an additive model:

    NAM(x) = bias + Σ_i  f_i(x_i)

where each f_i is an independent FeatureNN that sees only feature i.
This additivity is what makes NAM interpretable: you can plot each
f_i separately to understand its contribution.

Feature dropout: during training, randomly zero out entire feature
outputs (not individual neurons) before summation. This regularises
the model and prevents any single feature from dominating.

Returns a tuple (output, feature_outputs):
    output         : (batch_size,)        — final prediction logit
    feature_outputs: (batch_size, n_feats) — per-feature contributions
                     needed by the output_penalty regularisation term in the loss.

Design note for future interaction extension:
    Adding pairwise interactions only requires appending InteractionNN
    outputs to feature_outputs before the sum — no other changes needed.

Reference: original_neural_additive_models/models.py  NAM
PyTorch reference: nam-main-multitask/nam-main/nam/models/nam.py
"""

import torch
import torch.nn as nn
from .feature_nn import FeatureNN
from ..utils.config import NAMConfig


class NAM(nn.Module):
    """
    Neural Additive Model: one FeatureNN per input feature.

    Args:
        num_features: Number of input features (= number of FeatureNNs to create).
        config:       NAMConfig dataclass with all hyperparameters.
    """

    def __init__(self, num_features: int, config: NAMConfig):
        super().__init__()
        self.num_features = num_features
        self.config = config

        # TODO: create one FeatureNN per feature, stored in nn.ModuleList
        #       so PyTorch tracks all parameters automatically
        #       each FeatureNN receives: num_units, hidden_sizes, dropout, activation from config

        # TODO: feature-level dropout: nn.Dropout(config.feature_dropout)
        #       this drops entire feature outputs, not individual neurons

        # TODO: learnable scalar bias: nn.Parameter(torch.zeros(1))
        #       added to the sum of all feature outputs

        raise NotImplementedError

    def forward(self, x: torch.Tensor):
        # x shape: (batch_size, num_features)
        # TODO: for each feature i, slice x[:, i:i+1] and pass through feature_nns[i]
        #       result per feature: (batch_size, 1)

        # TODO: torch.cat all feature outputs along dim=1
        #       → feature_outputs: (batch_size, num_features)

        # TODO: apply feature dropout to feature_outputs

        # TODO: sum across features (dim=1) + self.bias
        #       → output: (batch_size,)

        # TODO: return (output, feature_outputs)
        raise NotImplementedError
