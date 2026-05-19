"""
Trainer — training loop with checkpointing and early stopping.

Designed to be interruptable: every epoch writes a checkpoint so training
can be resumed exactly from where it left off with --resume.

Run layout (created automatically):
    runs/<run_id>/
        config.yaml           copy of the config used for this run
        metrics.jsonl         one JSON line per epoch, appended (survives crashes)
        checkpoints/
            epoch_<N>.pt      full checkpoint every epoch
        best.pt               overwritten whenever val metric improves

Checkpoint format (dict saved with torch.save):
    {
        'epoch':            int,
        'model_state':      model.state_dict(),
        'optimizer_state':  optimizer.state_dict(),
        'scheduler_state':  scheduler.state_dict(),
        'config':           dataclasses.asdict(config),
        'best_val_metric':  float,
    }

Reference: nam-main-multitask/nam-main/nam/trainer/trainer.py
"""

import json
import shutil
from dataclasses import asdict
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ..utils.config import NAMConfig
from .losses import penalized_loss
from .metrics import auroc, rmse


class Trainer:
    """
    Manages the training loop, validation, checkpointing, and early stopping.

    Args:
        model:   NAM instance to train.
        config:  NAMConfig with all hyperparameters.
        run_dir: Path to the run directory (e.g. runs/20240101_120000/).
                 Created by scripts/train.py before instantiating Trainer.
    """

    def __init__(self, model: nn.Module, config: NAMConfig, run_dir: Path):
        self.model = model
        self.config = config
        self.run_dir = Path(run_dir)
        self.checkpoint_dir = self.run_dir / "checkpoints"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # TODO: initialise Adam optimizer: torch.optim.Adam(model.parameters(), lr=config.lr)
        # TODO: initialise StepLR scheduler: step_size=1, gamma=config.decay_rate
        #       (decays lr by decay_rate every epoch)

        self.best_val_metric = None  # set to None until first validation
        self.epochs_without_improvement = 0
        self.start_epoch = 0

        raise NotImplementedError

    # ------------------------------------------------------------------
    # Core training methods
    # ------------------------------------------------------------------

    def _train_epoch(self, loader: DataLoader) -> float:
        """
        Run one full pass over the training set.

        Returns:
            Mean training loss over all batches (float).

        TODO:
            - model.train()
            - Iterate loader: (features, targets, weights) per batch
            - Forward: output, fnn_outputs = model(features)
            - Loss: penalized_loss(output, targets, weights, fnn_outputs, model, config)
            - Backward + optimizer.step() + optimizer.zero_grad()
            - Accumulate loss, return mean
        """
        raise NotImplementedError

    def _val_epoch(self, loader: DataLoader):
        """
        Evaluate on validation set without gradient computation.

        Returns:
            (val_loss, val_metric) where val_metric is AUROC or RMSE depending on task.

        TODO:
            - model.eval()
            - torch.no_grad() context
            - Collect all logits and targets across batches
            - Compute loss with penalized_loss
            - Compute metric: auroc() for classification, rmse() for regression
        """
        raise NotImplementedError

    def _save_checkpoint(self, epoch: int, is_best: bool = False):
        """
        Save model + optimizer + scheduler state to disk.

        Always saves to checkpoints/epoch_<epoch>.pt.
        If is_best=True, also copies to best.pt.

        TODO:
            - Build checkpoint dict (see module docstring for format)
            - torch.save to self.checkpoint_dir / f'epoch_{epoch}.pt'
            - If is_best: shutil.copy to self.run_dir / 'best.pt'
        """
        raise NotImplementedError

    def _log_metrics(self, epoch: int, train_loss: float, val_loss: float, val_metric: float):
        """
        Append one JSON line to metrics.jsonl.

        Appending (not overwriting) means the file survives an interrupted run
        and can be read incrementally to plot training curves.

        TODO:
            - Build dict: {epoch, train_loss, val_loss, val_metric}
            - Open self.run_dir / 'metrics.jsonl' in append mode
            - json.dumps(dict) + newline
        """
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def train(self, train_dataset, val_dataset):
        """
        Full training loop from self.start_epoch to config.num_epochs.

        Flow per epoch:
            1. _train_epoch
            2. Every val_check_interval epochs: _val_epoch
               a. _log_metrics
               b. _save_checkpoint (is_best if improved)
               c. Check early stopping patience
            3. scheduler.step()

        Early stopping: if val metric does not improve for config.patience epochs,
        print a message and break.

        TODO:
            - Create DataLoaders for train_dataset and val_dataset
              (shuffle=True for train, False for val, batch_size from config)
            - Loop epochs from self.start_epoch to config.num_epochs
            - Implement the flow above
            - Print progress each val check (epoch, train_loss, val_metric)
        """
        raise NotImplementedError

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
        """
        raise NotImplementedError
