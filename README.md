# thesis-nam

Reimplementation of Neural Additive Models (NAMs) from scratch in PyTorch, written as part of a master's thesis. The original paper used TensorFlow — that implementation lives in `original_neural_additive_models/` as read-only reference.

## Installation

```bash
pip install -e ".[dev]"
```

Requires Python 3.8+. Core dependencies: `torch`, `numpy`, `pandas`, `scikit-learn`, `matplotlib`.

## Project Structure

```
thesis-nam/
├── src/nam/                  # installable package (src layout)
│   ├── models/               # NAM, FeatureNN, activation layers
│   ├── data/                 # data loading + preprocessing
│   ├── training/             # trainer, losses
│   └── utils/                # metrics
├── tests/                    # pytest tests
├── notebooks/                # experiments, exploration, figures
├── configs/                  # YAML experiment configs
├── scripts/                  # thin CLI entry points (train.py, eval.py)
└── datasets/                 # raw/ and processed/ CSV files (gitignored)
```

Primary dataset: COMPAS recidivism scores (place CSV in `datasets/raw/`).
