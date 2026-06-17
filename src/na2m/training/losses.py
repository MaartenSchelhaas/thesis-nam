"""
Loss functions for NA2M training.

The total loss has three components:
    1. Base loss: BCE (classification) or MSE (regression), per-sample weighted.
    2. Output penalty: penalises large individual feature contributions.
    3. L2 regularisation: standard weight decay across all parameters.

"""

import torch
import torch.nn as nn
from na2m.models.na2m import NA2M

def penalized_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    weights: torch.Tensor,
    fnn_outputs: torch.Tensor,
    model: NA2M,
    output_regularization: float,
    l2_regularization: float,
    task: str,
) -> torch.Tensor:
    """Compute the full regularised NA2M loss.

    Args:
        logits (torch.Tensor): Raw model output, shape (batch_size,).
        targets (torch.Tensor): Binary targets, shape (batch_size,).
        weights (torch.Tensor): Per-sample weights, shape (batch_size,).
                                Weight 0 means the sample is excluded from the loss.
        fnn_outputs (torch.Tensor): Per-feature outputs stacked by NA2M.forward(),
                                    shape (batch_size, num_features).
                                    Used for the output penalty term.
        model (NA2M): The NA2M instance (needed to iterate parameters for L2).
        output_regularization (float): Coefficient λ1 scaling the output penalty term.
        l2_regularization (float): Coefficient λ2 scaling the L2 weight decay term.
        task (str): 'classification' for BCE loss, 'regression' for MSE loss.

    Returns:
        torch.Tensor: Scalar loss tensor
    """

    num_features = model.num_features

    normal_loss = base_loss(logits=logits, targets=targets, weights=weights, task=task)
    out_pen = output_penalty(fnn_outputs=fnn_outputs, lambda1=output_regularization)
    l2_pen = l2_penalty(model=model, num_features=num_features, lambda2=l2_regularization)

    return normal_loss + out_pen + l2_pen

def base_loss(logits: torch.Tensor, targets: torch.Tensor, weights: torch.Tensor, task: str) -> torch.Tensor:
    """Compute weighted BCE (classification) or MSE (regression) loss.

    Args:
        logits (torch.Tensor): Raw model output, shape (batch_size,).
        targets (torch.Tensor): Ground truth labels, shape (batch_size,).
        weights (torch.Tensor): Per-sample weights, shape (batch_size,). 0 excludes a sample.
        task (str): 'classification' for BCE, 'regression' for MSE.

    Returns:
        torch.Tensor: Scalar loss tensor.
    """
    loss_fn: nn.Module = nn.BCEWithLogitsLoss(reduction='none') if task == 'classification' else nn.MSELoss(reduction='none')
    loss: torch.Tensor = loss_fn(logits, targets)
    return (loss * weights).mean()


def output_penalty(fnn_outputs: torch.Tensor, lambda1: float) -> torch.Tensor:
    """Calculate output penalty, this penalizes large individual feature subnet
    outputs.
    This is analagous to calculating the mean (squared) subnet output for each observation,
    and then taking the mean over each observation.
    Args:
        fnn_outputs (torch.Tensor): Per-feature subnet outputs, shape (batch_size, num_features).

    Returns:
        torch.Tensor: Scalar penalty tensor.
    """
    return (fnn_outputs ** 2).mean() * lambda1

def l2_penalty(model: nn.Module, num_features: int, lambda2: float) -> torch.Tensor:
    """Penalise large model weights (weight decay).

    Sums squared values of all learnable parameters, normalises by num_features,
    and scales by lambda2.

    Args:
        model (nn.Module): NA2M instance whose parameters are regularised.
        num_features (int): Number of feature subnets, used for normalisation.
        lambda2 (float): L2 regularization coefficient.

    Returns:
        torch.Tensor: Scalar penalty tensor.
    """
    # Initialise the accumulator on the model's device. The frozen NAM copy used
    # torch.tensor(0.0) (CPU), which raises a "two devices" error once params live
    # on CUDA — NAM only escaped it by running on CPU.
    device = next(model.parameters()).device
    l2_sum = torch.zeros((), device=device)

    for param in model.parameters():
        squared = param.pow(2)
        param_sum = squared.sum()
        l2_sum = l2_sum + param_sum

    normalized = l2_sum / num_features
    return normalized * lambda2