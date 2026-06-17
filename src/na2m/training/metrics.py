"""
Evaluation metrics for NA2M.

Thin wrappers that accept raw logits (not probabilities) so the caller
never has to remember to apply sigmoid before computing metrics.

Classification: AUROC
Regression:     RMSE

Copied verbatim from src/nam/training/metrics.py to keep src/na2m/ standalone.
No edits.
"""

import torch
import numpy as np
from sklearn.metrics import roc_auc_score


def auroc(logits: torch.Tensor, targets: torch.Tensor) -> float:
    """
    Compute Area Under the ROC Curve from raw logits.

    Args:
        logits(torch.Tensor):  Raw model output, shape (n,). Will be sigmoid-ed internally.
        targets(torch.Tensor): Binary labels, shape (n,). Values in {0, 1}.

    Returns:
        float: AUROC score as a Python float in [0, 1].

    """

    #Convert sigmoid to binary, and make it numpy for sci-kit learn
    class_pred = torch.sigmoid(logits).detach().cpu().numpy()
    real_values = targets.detach().cpu().numpy()

    auc_score = roc_auc_score(real_values,class_pred)
    return float(auc_score)



def rmse(preds: torch.Tensor, targets: torch.Tensor) -> float:
    """
    Compute Root Mean Squared Error.

    Args:
        preds (torch.Tensor):   Model predictions, shape (n,).
        targets(torch.Tensor): Ground truth values, shape (n,).

    Returns:
        float: RMSE as a Python float.
    """

    return torch.sqrt(torch.mean((preds.detach().cpu() - targets.detach().cpu())**2)).item()