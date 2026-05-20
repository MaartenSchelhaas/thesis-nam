"""
Evaluation metrics for NAM.

Thin wrappers that accept raw logits (not probabilities) so the caller
never has to remember to apply sigmoid before computing metrics.

Classification: AUROC — threshold-free, appropriate for imbalanced binary data.
Regression:     RMSE  — same units as the target, easy to interpret.
"""

import torch
import numpy as np


def auroc(logits: torch.Tensor, targets: torch.Tensor) -> float:
    """
    Compute Area Under the ROC Curve from raw logits.

    Args:
        logits:  Raw model output, shape (n,). Will be sigmoid-ed internally.
        targets: Binary labels, shape (n,). Values in {0, 1}.

    Returns:
        AUROC score as a Python float in [0, 1].

    TODO:
        - Convert logits to probabilities: torch.sigmoid(logits)
        - Detach and move both tensors to CPU numpy arrays
        - Use sklearn.metrics.roc_auc_score(targets_np, probs_np)
        - Return the float score
    """
    raise NotImplementedError


def rmse(preds: torch.Tensor, targets: torch.Tensor) -> float:
    """
    Compute Root Mean Squared Error.

    Args:
        preds:   Model predictions, shape (n,).
        targets: Ground truth values, shape (n,).

    Returns:
        RMSE as a Python float.

    TODO:
        - torch.sqrt(torch.mean((preds - targets) ** 2))
        - Return as Python float (.item())
    """
    raise NotImplementedError
