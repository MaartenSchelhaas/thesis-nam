"""
NA2M Trainer — thin subclass of the NAM Trainer with marginal-clarity penalty.

The only difference from the NAM Trainer is the clarity_lambda parameter, which
enables the GAMI-Net marginal-clarity penalty during interaction block-training
(Stage 2) and joint fine-tuning (Stage 3). Set clarity_lambda=0.0 for Stage 1
(mains only, no interactions exist yet).
"""

import torch
from torch.utils.data import DataLoader

from nam.training.trainer import Trainer as NAMTrainer
from nam.training.losses import penalized_loss
from na2m.models.na2m import NA2M


class Trainer(NAMTrainer):
    """NAM Trainer extended with the marginal-clarity penalty for NA2M stages 2 and 3."""

    def __init__(self, *args, clarity_lambda: float = 0.0, **kwargs):
        """Initialise the NA2M Trainer.

        All arguments are forwarded to the NAM Trainer. The only addition is
        clarity_lambda, which scales the marginal-clarity penalty in _train_epoch.

        Args:
            *args: Positional arguments forwarded to NAMTrainer.__init__.
            clarity_lambda: Coefficient for the marginal-clarity penalty.
                            0.0 (default) disables it — correct for Stage 1.
                            Stages 2 and 3 pass hp.clarity_lambda.
            **kwargs: Keyword arguments forwarded to NAMTrainer.__init__.
        """
        super().__init__(*args, **kwargs)
        self.model: NA2M
        self.clarity_lambda = clarity_lambda

    def _train_epoch(self, loader: DataLoader) -> float:
        """One training epoch with optional marginal-clarity penalty.

        Identical to NAMTrainer._train_epoch but adds clarity_lambda *
        model.clarity_loss(X_batch) to the loss when clarity_lambda > 0.

        Args:
            loader: Training DataLoader.

        Returns:
            Mean loss per batch over the full epoch.
        """
        self.model.train()
        epoch_loss = 0.0
        for X_batch, y_batch, weights in loader:
            X_batch, y_batch, weights = (
                X_batch.to(self.device),
                y_batch.to(self.device),
                weights.to(self.device),
            )
            self.optimizer.zero_grad()
            predictions, fnn_outputs = self.model(X_batch)

            loss = penalized_loss(
                logits=predictions,
                targets=y_batch,
                weights=weights,
                fnn_outputs=fnn_outputs,
                model=self.model, #type: ignore
                output_regularization=self.output_regularization,
                l2_regularization=self.l2_regularization,
                task=self.task,
            )

            if self.clarity_lambda > 0.0 and hasattr(self.model, "clarity_loss"):
                loss = loss + self.clarity_lambda * self.model.clarity_loss(X_batch)

            loss.backward()
            self.optimizer.step()
            epoch_loss += loss.item()

        return epoch_loss / len(loader)