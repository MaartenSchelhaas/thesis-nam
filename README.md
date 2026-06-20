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
├── src/na2m/                 # NA2M extension (concurvity-aware, GAMI-Net-style)
│   ├── models/               # CategNet, NA2M (type-aware mains + dynamic pairwise interactions)
│   ├── data/                 # route-2 path: integer-coded cats, grids, density
│   ├── training/             # fit_na2m staged orchestrator (stages 1–3)
│   ├── selection/            # FAST interaction screen + selection policy
│   ├── eval/                 # extract measures + reduce metrics (building blocks)
│   └── utils/                # NA2MConfig + concurvity (shared adj-R² helper)
├── configs/                    
│   └── compas_na2m_search.yaml #Config file for hyperparameter search space.
├── scripts/                  # drivers that use the models (logic lives in src/)
│   ├── nam/                  # NAM drivers: train, evaluate, tune_nam, run_nam (old reproduction)
│   └── na2m/
│       ├── run_na2m_eval.py  # main entry point: k-fold × n_runs × 3-arm loop
│       ├── model_runner.py   # train one arm, extract measures, write done sentinel
│       ├── tune_main_na2m.py # Optuna Stage-1 hp search (lr, dropout, arch)
│       ├── tune_clarity.py   # Optuna clarity_regularization search over Stage 2/3
│       └── reduce_fold0.py   # compute + print all metrics for one fold; save shape plots
├── runs/                     # training outputs: checkpoints, metrics (gitignored)
├── notebooks/                # experiments, exploration, figures
├── tests/                    # pytest tests, empty
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

Per `(arm, fold, seed)` the evaluator trains once and immediately extracts
everything needed from the live in memory model. `extract_measures` (`src/na2m/eval/extract.py`)
stores four things to `measures.pt`: raw per-subnet output vectors on the pool
(for the post-hoc concurvity score) and the test fold (for stability); summed
test logits (for accuracy); the selected pair list; and centered main-effect shape
curves evaluated on a fixed real-unit grid. The model is then discarded — arms B
and C do not write a `model.pt`.

Every headline metric is a **pure function of those stored measures**, computed
from measures on disk by `eval/reduce.py`. `load_fold` walks a fold directory and returns a
dict of completed run dicts; metric functions (`main_effect_instability`,
`mean_pairwise_jaccard`, `concurvity_summary`, `accuracy_summary`, `shape_plots`)
take that dict and return scalars or figures without touching any model.
Changing or adding a metric never requires retraining.


### Run output layout

All outputs land under `runs/<dataset>/` (gitignored). Tuned hyperparameter
configs are shared across run modes; run outputs are mode-specific:

```
runs/compas/
    fold_k/
        mains_tuned_config.yaml       ← best main-effects hp (arch + lr + reg)
        gaminet_tuned_config.yaml     ← mains hp + tuned clarity λ for arm B
        concurvity_tuned_config.yaml  ← mains hp + tuned clarity λ for arm C
        <run_mode>/                   ← "fixed" or "subsample"
            mains/run_i/
                model.pt              ← trained mains checkpoint (arm A)
                measures.pt           ← extracted per-term outputs + test logits
                done                  ← sentinel: written last, guards resume
            gaminet/run_i/
                measures.pt
                done
            concurvity/run_i/
                measures.pt
                done
```

Tuning is done once per fold regardless of how many run modes you evaluate.
Switching from `fixed` to `subsample` reuses the tuned configs and only
re-runs the model training. Resume is fully granular: the `done` sentinel is
written last, so a crashed run continues exactly where it left off on rerun.

## Usage

