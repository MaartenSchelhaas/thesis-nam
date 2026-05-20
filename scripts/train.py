"""
Training entry point.

Usage:
    # Fresh run
    python scripts/train.py --config configs/compas_baseline.yaml

    # Resume interrupted run
    python scripts/train.py --config configs/compas_baseline.yaml \
        --resume runs/<run_id>/checkpoints/epoch_50.pt

What this script does:
    1. Parse --config YAML into a NAMConfig instance
    2. Generate a unique run_id (datetime + short config hash)
    3. Create runs/<run_id>/ and copy the config there for reproducibility
    4. Load and preprocess the COMPAS dataset
    5. Wrap splits in NAMDataset
    6. Instantiate NAM model and Trainer
    7. If --resume: call trainer.resume(checkpoint_path)
    8. Call trainer.train(train_dataset, val_dataset)
    9. Print final val metric

TODO:
    - argparse: --config (required str), --resume (optional str)
    - Load YAML with PyYAML (import yaml)
    - Build NAMConfig from the YAML dict
    - Generate run_id: datetime.now().strftime('%Y%m%d_%H%M%S')
    - Create Path('runs') / run_id, copy config file there with shutil.copy
    - Call load_compas(), preprocess(), split() from nam.data.data_utils
    - Wrap arrays in NAMDataset (from nam.data.dataset)
    - num_features = X_train.shape[1]
    - Instantiate NAM(num_features, config) and Trainer(model, config, run_dir)
    - If args.resume: trainer.resume(args.resume)
    - trainer.train(train_dataset, val_dataset)
"""

# TODO: imports (argparse, yaml, pathlib, shutil, datetime, torch)
# TODO: imports from src/nam (data_utils, dataset, nam, config, trainer)


def main():
    # TODO: implement as described in module docstring
    raise NotImplementedError


if __name__ == "__main__":
    main()
