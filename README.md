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
├── src/nam/                  # NAM reproduction package (frozen baseline)
│   ├── models/               # NAM, FeatureNN, activation layers
│   ├── data/                 # data loading + preprocessing
│   ├── training/             # trainer, losses, metrics
│   └── utils/                # config dataclass
├── src/na2m/                 # NA2M extension (concurvity-aware, GAMI-Net-style)
│   ├── models/               # CategNet, InteractionNN, NA2M (mains + interactions)
│   ├── data/                 # route-2 path: integer-coded cats, grids, density
│   ├── training/             # fit_na2m staged orchestrator + derive (B′) (building blocks)
│   ├── selection/            # FAST interaction screen (interpret/EBM wrapper)
│   ├── eval/                 # extract measures + reduce metrics (building blocks)
│   └── utils/                # NA2MConfig (own config; shares primitives w/ nam)
├── configs/
│   ├── compas_baseline.yaml  # fixed hyperparams from the paper
│   ├── compas_search.yaml    # search space definition for tuning
│   └── compas_tuned.yaml     # best params found by tune_nam.py (generated)
├── scripts/                  # drivers that use the models (logic lives in src/)
│   ├── nam/                  # NAM drivers: train, evaluate, tune_nam, run_nam
│   └── na2m/                 # NA2M drivers: run_na2m_eval (k-fold×seed×arm loop)
├── runs/                     # training outputs: checkpoints, metrics (gitignored)
├── notebooks/                # experiments, exploration, figures
├── tests/                    # pytest tests
└── datasets/                 # raw/ CSV files (gitignored)
```

Primary dataset: COMPAS recidivism scores (place CSV in `datasets/raw/`).

## Usage

