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
│   ├── models/               # CategNet, NA2M (type-aware mains + dynamic pairwise interactions)
│   ├── data/                 # route-2 path: integer-coded cats, grids, density
│   ├── training/             # fit_na2m staged orchestrator (stages 1–3)
│   ├── selection/            # FAST interaction screen (interpret/EBM wrapper)
│   ├── eval/                 # extract measures + reduce metrics (building blocks)
│   └── utils/                # NA2MConfig + concurvity (shared adj-R² helper)
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

## NA2M extension

`src/nam/` is the frozen NAM reproduction. `src/na2m/` is the thesis contribution:
a GAMI-Net-style additive model with pairwise interactions, extended with a
**concurvity filter** that suppresses interaction terms which are redundant with
the rest of the fitted model. The research question is whether that filter buys
more *stable* (reproducible-across-seeds) interpretable structure without
sacrificing accuracy.

### The model

`NA2M` is one backbone:

```
y = b + Σ_j f_j(x_j) + Σ_(j,k) f_jk(x_j, x_k)
```

a type-aware main bank (`FeatureNN` per numerical, `CategNet` per integer-coded
categorical) plus pairwise interaction subnets that are **built dynamically when
selected** (no pre-allocation/masking). Terms are keyed by `term_id` — `("main",
j)` or `("inter", j, k)`, where `j, k` are feature indices — never positionally.

### Three arms, two flags

A single pipeline serves all experiment arms; they differ only by two flags:

| Arm | `with_interactions` | `with_concurvity_filter` | Meaning |
|-----|--------------------|--------------------------|---------|
| A   | `False`            | —                        | mains-only baseline |
| B   | `True`             | `False`                  | full GAMI-Net |
| C   | `True`             | `True`                   | GAMI-Net + concurvity gate |

B and C run an **identical** pipeline; the *only* difference is whether the
concurvity gate fires. Both fine-tune **exactly once**.

### Training pipeline (`fit_na2m`, three stages)

1. **Stage 1 — mains.** Train the main bank; center each subnet to zero pool-mean
   (fold the mean into the bias, prediction-invariant).
2. **Stage 2 — select.** FAST-screen candidate pairs, block-train the top-M
   interaction subnets jointly (mains frozen), then a **single forward prune
   sweep** in decreasing-contribution order applying two gates:
   - *concurvity gate* (arm C only) — skip a candidate whose block-trained output
     regresses on {mains + already-accepted interactions} with adj-R² above
     threshold; skipped terms are never reconsidered;
   - *predictive-contribution gate* — min-max-normalize the accepted candidates'
     validation-loss sequence and cut at the smallest index within tolerance η.
3. **Stage 3 — fine-tune.** Jointly fine-tune the retained mains + interactions
   once with the marginal-clarity penalty; re-center.

(There is deliberately no iterative post-fine-tune removal — concurvity is a
selection-time gate, so B and C stay byte-identical except for the gate.)

### Evaluation: store-everything → reducer

Models are **ephemeral**. Per `(arm, fold, seed)` the harness trains once and
`extract_measures` persists **raw** per-term output vectors (on both the pool and
the held-out test fold), test logits, and the selected pair set. Every headline
metric — term stability across seeds, selection-set Jaccard, the adj-R²
concurvity diagnostic, and accuracy vs. term-count — is then a **pure function of
those stored measures**, computed in `eval/reduce.py`. Changing or adding a
metric never requires retraining. The k-fold split is keyed off the fold so that
varying the seed varies only initialization, isolating optimization variance.

The build status of each piece is tracked in `docs/todo.md`.

## Usage

