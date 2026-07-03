# thesis-nam

Reimplementation of Neural Additive Models (NAMs) from Agarwal et al. (2021) in PyTorch, written as part of a bachelor's thesis. Extended for interaction effects based on GAMI-Net (Yang et al., 2021), with a concurvity filter based on Kovács's (2022) multivariate concurvity measure, and a concurvity regularizer from Siems et al. (2023).

## Installation

```bash
pip install -e .
```

Requires Python 3.10+. Core dependencies: `torch`, `numpy`, `pandas`, `scikit-learn`, `matplotlib`, `optuna`, `pyyaml`. This is enough to run every script in `scripts/`.

To also run the notebooks (or use `pytest`/`ruff`/`mypy`), install the `dev` extra instead:

```bash
pip install -e ".[dev]"
```

## Project Structure

```
thesis-nam/
├── src/na2m/                 # NA2M model + pure compute (the thesis contribution)
│   ├── models/               # CategNet, Feature Subnet, NA2M
│   ├── data/                 # data processing per data set, data utils
│   ├── training/             # fit_na2m staged orchestrator (stages 1–3)
│   ├── selection/            # FAST interaction screen + selection policy
│   ├── eval/                 # extract measures + reduce metrics (building blocks)
│   └── utils/                # NA2MConfig + concurvity (shared adj-R² helper)
├── configs/                    
│   └── compas_na2m_search.yaml #Config file for hyperparameter search space.
├── scripts/                  # evaluation orchestration that drives src/na2m
│   ├── tuning/                # per-arm hyperparameter search, called by scripts/eval
│   │   ├── tune_main_na2m.py          # Stage-1 hp search: main effects only (arm A)
│   │   ├── tune_clarity.py            # Stage-2/3 hp search: clarity_regularization (arms B/C)
│   │   ├── tune_concurvity_reg.py     # lambda_2 grid sweep across seeds for arm D; writes plot + CSV only
│   │   └── confirm_regularized_arm.py # after inspecting the sweep plot, commit a lambda_2 value → writes arm D's tuned config
│   ├── eval/                  # orchestrate + seed training runs; models are only ever built in model_runner.py
│   │   ├── run_na2m_eval.py           # main entry point: k-fold × n_runs loop over all four arms
│   │   ├── evaluate_na2m_mains.py     # Agarwal reproduction: mains-only, fresh train/val subsample per run
│   │   └── model_runner.py            # instantiates NA2M and calls into src/na2m; both eval scripts above only orchestrate + set seeds
│   └── reduce/                 # aggregate stored measures into headline numbers/plots
│       ├── reduce_fold0.py            # print all metrics for a single fold (currently hardcoded to fold_0)
│       ├── reduce_all_folds.py        # aggregate metrics across all folds
│       └── reduce_mains_subsample.py  # ensemble accuracy + 95% CI for the Agarwal mains-only reproduction
├── runs/                     # training outputs: checkpoints, metrics (Thesis results included)
├── notebooks/                # Used during development, could contain stale code. 
├── tests/                    # pytest tests, empty
└── datasets/                 # raw/ CSV files (COMPAS included)
```

Primary dataset: COMPAS recidivism scores (place CSV in `datasets/raw/`).

## NA2M extension

`src/na2m/` is the thesis contribution: a GAMI-Net-style additive model with
pairwise interactions, extended with a **concurvity filter** and a
**concurvity regularizer** that each suppress interaction terms redundant with
the rest of the fitted model, in different ways. The research question is
whether either buys more *stable* (reproducible-across-seeds) interpretable
structure without sacrificing accuracy.

### The model

`NA2M` is one backbone:

```
y = b + Σ_j f_j(x_j) + Σ_(j,k) f_jk(x_j, x_k)
```

a type-aware main bank (`FeatureNN` per numerical, `CategNet` per integer-coded
categorical) plus pairwise interaction subnets that are **built dynamically when
selected** (no pre-allocation/masking). Terms are keyed by `term_id` — `("main",
j)` or `("inter", j, k)`, where `j, k` are feature indices — never positionally.

### Four arms

A single pipeline serves all experiment arms; they differ only by two flags plus
one scalar:

| Arm | `with_interactions` | `with_concurvity_filter` | `concurvity_regularization` | Meaning |
|-----|--------------------|--------------------------|------------------------------|---------|
| A   | `False`            | —                        | —                            | mains-only baseline |
| B   | `True`             | `False`                  | `0.0`                        | full GAMI-Net |
| C   | `True`             | `True`                   | `0.0`                        | GAMI-Net + concurvity gate |
| D   | `True`             | `False`                  | `lambda_2 > 0`                | GAMI-Net + concurvity regularizer |

B, C, and D run an **identical** training pipeline; the only differences are
whether the Stage-2 concurvity gate fires (arm C) and whether Stage 2/3 add an
R_perp regularization penalty to the loss (arm D, weighted by `lambda_2`). All
three fine-tune **exactly once**.

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
   Arm D's R_perp penalty (`concurvity_regularization`) is added directly to the
   loss throughout Stage 2/3 instead of gating candidates.
3. **Stage 3 — fine-tune.** Jointly fine-tune the retained mains + interactions
   once with the marginal-clarity penalty (and, for arm D, the R_perp penalty);
   re-center.


### Evaluation: store-everything → reducer

Per `(arm, fold, seed)` the evaluator trains once and, while the model is still
live in memory, immediately extracts everything evaluation will ever need —
so the model never has to be reloaded later. `extract_measures`
(`src/na2m/eval/extract.py`) stores four things to `measures.pt`: raw per-subnet
output vectors on the pool (for the post-hoc concurvity score) and the test fold
(for stability); summed test logits (for accuracy); the selected pair list; and
centered main-effect shape curves evaluated on a fixed real-unit grid. The model
is then discarded — arms B, C, and D do not write a `model.pt`.

Every headline metric is a **pure function of stored measures**, computed by
`eval/reduce.py` (called from `scripts/reduce/`). `load_fold` loads a fold's
run dicts; metric functions (`main_effect_instability`, `mean_pairwise_jaccard`,
`concurvity_summary`, `accuracy_summary`, `selection_frequencies`, `shape_plots`)
turn them into scalars or figures — no model is ever reloaded, and adding a
metric never requires retraining.


### Run output layout

All outputs land under `runs/<dataset>_na2m/` (currently included). Tuned hyperparameter
configs are shared across run modes; run outputs are mode-specific:

```
runs/compas_na2m/
    fold_k/
        mains_tuned_config.yaml       ← best main-effects hp (arch + lr + reg)
        gaminet_tuned_config.yaml     ← mains hp + tuned clarity λ for arm B
        concurvity_tuned_config.yaml  ← mains hp + tuned clarity λ for arm C
        regularized_lambda2_sweep.csv ← per-(lambda_2, seed) sweep results (tune_concurvity_reg.py)
        regularized_lambda2_sweep.png ← tradeoff plot to inspect before confirming arm D
        regularized_tuned_config.yaml ← mains hp + confirmed lambda_2 for arm D (written by confirm_regularized_arm.py)
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
            regularized/run_i/
                measures.pt
                done
```

Tuning is done once per fold regardless of how many run modes you evaluate.
Switching from `fixed` to `subsample` reuses the tuned configs and only
re-runs the model training. Resume is fully granular: the `done` file is
written last, so a crashed run continues exactly where it left off on rerun.

## Usage

Main thesis results (all four arms, k-fold × n_runs):

Change the config config file for the given datset if you changes in the search space, amound of fold or runs. 
Then:

```bash
python -m scripts.eval.run_na2m_eval
```

Arm D needs a manual step in between: `run_na2m_eval` runs the `lambda_2` grid
sweep and prints where the plot + CSV landed, but will not train arm D until a
value is confirmed. Inspect the sweep plot, then run:

```bash
python -m scripts.tuning.confirm_regularized_arm
```

(edit the variables at the top of `main()` to point at the fold and chosen
`lambda_2`), then rerun `python -m scripts.eval.run_na2m_eval` to train and
evaluate arm D with the confirmed config.

Agarwal-style mains-only reproduction (subsampled train/val split per run):

```bash
python -m scripts.eval.evaluate_na2m_mains
```

If `run_na2m_eval` has already tuned a fold (`mains_tuned_config.yaml` exists),
tuning is skipped and this just runs in subsample mode, storing measures.
Otherwise it does its own tuning, mirroring `run_na2m_eval` closely.

Once that's done, aggregate the reproduction result:

```bash
python -m scripts.reduce.reduce_mains_subsample
```
