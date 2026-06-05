"""
Trainer — training loop with early stopping.

Handles training a single NAM instance: forward pass, backpropagation,
learning rate scheduling, validation, and early stopping. The best model
state (by validation metric) is retained in memory after training.

Usage:
    trainer = Trainer(model=model, lr=0.001, ...)
    trainer.train(train_loader, val_loader)
    test_auc = trainer.evaluate(test_loader)
"""

from pathlib import Path
import copy
import json, shutil

import torch
from torch.utils.data import DataLoader

from nam.models.nam import NAM
from .losses import penalized_loss
from .metrics import auroc, rmse
from nam.utils.device import get_device

import optuna


class Trainer:
    """
    Manages the training loop, validation, checkpointing, and early stopping.
    """

    def __init__(
        self,
        model: NAM,
        lr: float,
        decay_rate: float,
        output_regularization: float,
        l2_regularization: float,
        task: str,
        num_epochs: int,
        patience: int,
        val_check_interval: int,
        run_dir: str | None = None,
        params=None,
    ):
        """Initialise the NAM Trainer.

        Args:
            model (NAM): Already-initialised NAM instance to train.
            lr (float): Initial Adam learning rate.
            decay_rate (float): Multiplicative LR decay applied every epoch (StepLR gamma).
            output_regularization (float): Coefficient for the feature output penalty term.
            l2_regularization (float): Coefficient for L2 weight decay.
            task (str): 'classification' or 'regression'
            num_epochs (int): Maximum number of training epochs.
            patience (int): Early stopping, stop if val metric doesn't improve for this many epochs.
            val_check_interval (int): Evaluate on validation set every N epochs.
            run_dir (Path): Directory to save checkpoints and metrics (created before passing in).
            params: Optional iterable of parameters to optimize. Defaults to
                model.parameters(). The NA2M orchestrator passes a parameter subset
                (e.g. interaction-only params) when staging; rebuild the Trainer (or
                its optimizer) after any structural model change.
        """
        self.model = model
        self.output_regularization = output_regularization
        self.l2_regularization = l2_regularization
        self.task = task
        self.num_epochs = num_epochs
        self.patience = patience
        self.val_check_interval = val_check_interval


        if run_dir is not None:
            self.run_dir = Path(run_dir)
            self.checkpoint_dir = self.run_dir / "checkpoints"
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        else:
            self.run_dir = None
            self.checkpoint_dir = None

<<<<<<< HEAD
        self.device = get_device()
        self.model = model.to(self.device)

        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
=======
        self.optimizer = torch.optim.Adam(params if params is not None else model.parameters(), lr=lr)
>>>>>>> d8b36df (na2m scaffoloding)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=1, gamma=decay_rate)

        self.best_val_metric = 0.0 if task == "classification" else float("inf")
        self.best_model_state = copy.deepcopy(model.state_dict())
        self.epochs_without_improvement = 0
        self.start_epoch = 0

    # ------------------------------------------------------------------
    # Core training methods
    # ------------------------------------------------------------------

    def _train_epoch(self, loader: DataLoader) -> float:
        """Run one full pass over the training set.

        Iterates all batches: forward pass → penalized loss → backward → optimizer step.

        Args:
            loader: Training DataLoader.

        Returns:
            Float: Mean loss per batch over the full epoch.
        """
        self.model.train()
        epoch_loss = 0.0
        for X_batch, y_batch, weights in loader:
            X_batch, y_batch, weights = X_batch.to(self.device), y_batch.to(self.device), weights.to(self.device)
            self.optimizer.zero_grad()
            predictions, fnn_outputs = self.model(X_batch)
            loss = penalized_loss(logits=predictions,
                                targets=y_batch,
                                weights=weights,
                                fnn_outputs=fnn_outputs,
                                model = self.model,
                                output_regularization=self.output_regularization,
                                l2_regularization=self.l2_regularization,
                                task = self.task)
            loss.backward()
            self.optimizer.step()
            epoch_loss += loss.item()

        return epoch_loss/ len(loader)
    

    def _val_epoch(self, loader: DataLoader) -> float:
        """Calculate validation metric for this epoch. 

        Args:
            loader (DataLoader): Validation dataloader

        Returns:
            float: Return validation metric for the specific task
        """
        self.model.eval()
        all_predictions = []
        all_targets = []
        with torch.no_grad():
            for X_batch, y_batch, _ in loader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                predictions, _ = self.model(X_batch)
                all_predictions.append(predictions)
                all_targets.append(y_batch)

        val_predictions = torch.cat(all_predictions)
        val_targets = torch.cat(all_targets)
        return self._compute_metric(val_predictions, val_targets)
    
    def _compute_metric(self, predictions: torch.Tensor, targets: torch.Tensor) -> float:
        """Compute task-appropriate validation metric.

        Args:
            predictions (torch.Tensor): Raw model logits, shape (n,).
            targets (torch.Tensor): Ground truth labels, shape (n,).

        Returns:
            AUROC for classification, RMSE for regression.
        """
        if self.task == "classification":
            return auroc(predictions, targets)
        else:
            return rmse(predictions, targets)
        
    def _is_improved(self, metric: float) -> bool:
        """Check if the current validation metric is an improvement over the best so far.

        Args:
            metric (float): Current validation metric.

        Returns:
            bool: True if improved, False otherwise.
        """
        if self.task == "classification":
            return metric > self.best_val_metric  # higher AUROC is better
        else:
            return metric < self.best_val_metric  # lower RMSE is better

    # ------------------------------------------------------------------
    # Saving model
    # ------------------------------------------------------------------

    def save_model(self, path: Path):
        """Save best model to disk

        Args:
            path (str): Path where to store the model in
        """

        torch.save(self.best_model_state,path)


    def _save_checkpoint(self, epoch: int, is_best: bool = False):
        """
        Save model + optimizer + scheduler state to disk.

        Not implemented for now, compas doesnt take that long.
        """
        raise NotImplementedError

    def _log_metrics(self, epoch: int, train_loss: float, val_metric: float):
        """Append one JSON line to metrics.jsonl.
        
        Not implemented for now, compas doesnt take that long.
        """
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def train(self, train_loader:DataLoader, val_loader:DataLoader, trial: optuna.Trial | None =None):
        """Run the full training loop with validation and early stopping.

        Trains for up to num_epochs epochs. Validates every val_check_interval epochs,
        saves the best model state, and stops early if val metric does not improve
        for patience epochs.

        Args:
            train_loader: DataLoader for training data.
            val_loader: DataLoader for validation data.
        """

        for epoch in range(self.start_epoch, self.num_epochs):
            epoch_loss = self._train_epoch(train_loader)

            #Run validation for every val_check_interval_epoch
            if (epoch + 1) % self.val_check_interval == 0:
                
                metric = self._val_epoch(val_loader)

                #For early pruning during hyperparameter tuning.
                if trial is not None:
                    trial.report(metric, epoch)
                    if trial.should_prune():
                        raise optuna.exceptions.TrialPruned()

                if self._is_improved(metric):
                    self.best_val_metric = metric
                    self.best_model_state = copy.deepcopy(self.model.state_dict())
                    self.epochs_without_improvement = 0
                else:
                    #We only do this for every val_check_interval epoch.
                    self.epochs_without_improvement += self.val_check_interval

                if self.epochs_without_improvement >= self.patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break

            if epoch == 0 or (epoch + 1) % 100 == 0:
                print(f"Epoch {epoch+1}/{self.num_epochs}| Epoch loss = {epoch_loss:.4f} | best={self.best_val_metric:.4f}")

            self.scheduler.step()
        
    def evaluate(self, loader: DataLoader) -> float:
        """Evaluate the models performence on the input test data

        Args:
            loader (DataLoader): Test dataset data loader

        Returns:
            float: Metric
        """
        self.model.load_state_dict(self.best_model_state)
        return self._val_epoch(loader)

    def load_best(self):
        """Restore the best (by val metric) weights into the model in place.

        Use this in the NA2M harness before any extraction — do NOT rely on
        evaluate()'s side effect for restoring best weights.
        """
        self.model.load_state_dict(self.best_model_state)

    def resume(self, checkpoint_path: str):
        """
        Load a checkpoint and prepare to continue training from that epoch.

        Call this before train() to resume an interrupted run:
            trainer.resume('runs/.../checkpoints/epoch_50.pt')
            trainer.train(train_dataset, val_dataset)

        TODO:
            - torch.load(checkpoint_path)
            - model.load_state_dict(checkpoint['model_state'])
            - optimizer.load_state_dict(checkpoint['optimizer_state'])
            - scheduler.load_state_dict(checkpoint['scheduler_state'])
            - self.start_epoch = checkpoint['epoch'] + 1
            - self.best_val_metric = checkpoint['best_val_metric']

        Not yet necessary for COMPAS.

        """
        raise NotImplementedError
