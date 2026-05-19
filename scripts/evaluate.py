"""
Evaluation entry point.

Loads a saved checkpoint and evaluates on the held-out test set.
Prints the test metric (AUROC or RMSE) and per-feature output statistics
(mean, std) to give a first look at feature importances.

Usage:
    python scripts/evaluate.py \
        --checkpoint runs/<run_id>/best.pt \
        --data datasets/raw/compas-scores-two-years.csv

TODO:
    - argparse: --checkpoint (required str), --data (required str)
    - torch.load(checkpoint_path) to recover model state + config dict
    - Reconstruct NAMConfig from checkpoint['config'] (dataclasses.asdict format)
    - Load and preprocess data: load_compas() → preprocess() → split()
      (use same seed as training so test split is identical)
    - Rebuild NAM(num_features, config) and load checkpoint['model_state']
    - model.eval() + torch.no_grad()
    - Forward pass on test set, compute metric
    - Print test metric
    - Print per-feature mean absolute output (proxy for feature importance)
      shape: (num_features,) — use feature_names from preprocess() as labels
"""

# TODO: imports


def main():
    # TODO: implement as described in module docstring
    raise NotImplementedError


if __name__ == "__main__":
    main()
