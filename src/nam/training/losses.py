"""
Loss functions for NAM training.

The total loss has three components:
    1. Base loss: BCE (classification) or MSE (regression), per-sample weighted.
    2. Output penalty: penalises large individual feature contributions.
       Encourages the model to spread signal across features rather than
       concentrating it in one, improving interpretability.
    3. L2 regularisation: standard weight decay across all parameters.

Components 2 and 3 are controlled by config.output_regularization and
config.l2_regularization respectively. Setting both to 0 gives plain BCE/MSE.

Reference: original_neural_additive_models/graph_builder.py  penalized_loss
PyTorch reference: nam-main-multitask/nam-main/nam/trainer/losses.py
"""

import torch
import torch.nn as nn
from ..utils.config import NAMConfig


def penalized_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    weights: torch.Tensor,
    fnn_outputs: torch.Tensor,
    model: nn.Module,
    config: NAMConfig,
) -> torch.Tensor:
    """
    Compute the full regularised NAM loss.

    Args:
        logits:      Raw model output, shape (batch_size,).
        targets:     Binary targets, shape (batch_size,).
        weights:     Per-sample weights, shape (batch_size,).
                     Weight 0 means the sample is excluded from the loss.
        fnn_outputs: Per-feature outputs stacked by NAM.forward(),
                     shape (batch_size, num_features).
                     Used for the output penalty term.
        model:       The NAM instance (needed to iterate parameters for L2).
        config:      NAMConfig with regularisation coefficients and task type.

    Returns:
        Scalar loss tensor (differentiable).

    TODO:
        1. Base loss:
           - classification: nn.BCEWithLogitsLoss(reduction='none') → multiply by weights → mean
           - regression:     nn.MSELoss(reduction='none')           → multiply by weights → mean

        2. Output penalty (if config.output_regularization > 0):
           - mean(fnn_outputs ** 2) across batch and features
           - multiply by config.output_regularization

        3. L2 penalty (if config.l2_regularization > 0):
           - sum(param ** 2 for param in model.parameters())
           - divide by model.num_features to normalise
           - multiply by config.l2_regularization

        4. Return sum of all active components
    """
    raise NotImplementedError
