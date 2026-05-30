# thesis-nam

Reimplementation of Neural Additive Models (NAMs) from scratch in PyTorch, written as part of a bachelor's thesis. The original paper used TensorFlow — that implementation lives in `original_neural_additive_models/` as read-only reference.

## Installation

```bash
pip install -e ".[dev]"
```

Requires Python 3.10+. Core dependencies: `torch`, `numpy`, `pandas`, `scikit-learn`, `matplotlib`, `optuna`, `pyyaml`.

## Project Structure

```
thesis-nam/
├── src/nam/                  # installable package (src layout)
│   ├── models/               # NAM, FeatureNN, activation layers
│   ├── data/                 # data loading + preprocessing
│   ├── training/             # trainer, losses, metrics
│   └── utils/                # config dataclass
├── configs/
│   ├── compas_baseline.yaml  # fixed hyperparams from the paper
│   ├── compas_search.yaml    # search space definition for tuning
│   └── compas_tuned.yaml     # best params found by tune_nam.py (generated)
├── scripts/
│   ├── train.py              # train a single run from a config YAML
│   ├── evaluate.py           # evaluate a saved checkpoint on test data
│   └── tune_nam.py           # Optuna hyperparameter search
├── runs/                     # training outputs: checkpoints, metrics (gitignored)
├── notebooks/                # experiments, exploration, figures
├── tests/                    # pytest tests
└── datasets/                 # raw/ CSV files (gitignored)
```

Primary dataset: COMPAS recidivism scores (place CSV in `datasets/raw/`).

## Usage

