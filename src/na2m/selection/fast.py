"""
FAST interaction screening — thin wrapper around the EBM / `interpret` FAST scorer.

"""

from interpret.utils._measure_interactions import measure_interactions
import numpy as np
import torch
from na2m.models.na2m import NA2M

def fast_screen(main_model: NA2M, X: np.ndarray, y: np.ndarray, task: str) -> list[tuple[int, int]]:
    """Rank candidate interaction pairs via the native FAST scorer.

    Args:
        main_model: The trained main-only NA2M (for residual-equivalent preds).
        X: Feature matrix to score on (train/val of the 80% pool).
        y: Targets.
        task: 'classification' (logit-transform residuals) or 'regression'.

    Returns:
        Ranked list of (j, k) pairs, best first.
    """
    
    if not isinstance(X, np.ndarray):
        X = np.asarray(X)
    if not isinstance(y, np.ndarray):
        y = np.asarray(y)

    device = next(main_model.parameters()).device
    main_model.eval()
    with torch.no_grad():
        logits, _ = main_model(torch.as_tensor(X, dtype=torch.float32).to(device))
        init_score = logits.squeeze(-1).cpu().numpy()

    objective = "log_loss" if task == "classification" else "rmse"
    ranked = measure_interactions(X, y, init_score=init_score, objective=objective)
    return [pair for pair, _ in ranked]
