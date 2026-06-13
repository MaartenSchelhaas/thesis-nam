"""
run_na2m_eval.py — driver for the NA2M evaluation experiment.

The k-fold × seed × arm train+extract loop. This is the orchestration DRIVER
(the analog of tune_nam.py's study loop): it imports the reusable building
blocks from src/na2m and src/nam and wires them together. All real logic
(staged training, measure extraction, metric reduction) lives in src — this
script only sequences it.

Usage:
    python scripts/na2m/run_na2m_eval.py --config configs/na2m_eval.yaml

Outer structure:
    Fixed before loops: global per-feature grids; fold split (own seed).
    for fold:
        tune hp on the 80% pool (cache); compute density on the 80% pool;
        derive the internal pool train/val split ONCE here (keyed off the FOLD,
        shared by every seed — SPLIT CONTRACT);
        persist density, feature_meta, fitted scaler, test labels.
        for seed:
            for arm in [A, B, C]:
                skip-if-measures-exist (resumability);
                fit_na2m(fold=fold, with_interactions, with_concurvity_filter)
                → load_best → extract_measures → save measures.
    Arms are {A, B, C}; there is NO B′ arm (count-matched control dropped — the
    in-sweep concurvity gate breaks the B⊃C nesting it relied on). Models are
    ephemeral: extract measures, then let each model go out of scope.

HARD CONSTRAINTS (see src/na2m for the full list):
    - Concurvity/stability on the 80% pool; test fold opened ONCE (accuracy only).
    - Bootstrap units are SEEDS; folds are data/accuracy variation.
    - SPLIT CONTRACT: the internal pool split is keyed off the FOLD, never the
      seed; seed varies only init + optimization. Do not reseed it per replicate.
"""

import argparse

# Shared primitives from the frozen NAM package
from nam.data.dataset import NAMDataset  # noqa: F401  (used once implemented)

# NA2M building blocks
from na2m.utils.config import load_na2m_config  # noqa: F401
from na2m.data.data_utils import (  # noqa: F401
    load_compas,
    preprocess,
    make_grid,
    density_weights,
)
from na2m.training.fit_na2m import fit_na2m  # noqa: F401
from na2m.eval.extract import extract_measures  # noqa: F401


def main():
    """Parse the config and run the full k-fold × seed × arm experiment.

    TODO:
        - argparse: --config (required str).
        - load_na2m_config(args.config).
        - load_compas → preprocess → build global per-feature grids (fixed).
        - K-fold split with its own seed (80% pool / 20% test per fold).
        - for fold:
            * tune hp on the 80% pool (cache to disk); density_weights;
              persist density, feature_meta, fitted scaler, test labels.
            * derive the internal pool train/val split ONCE (keyed off fold) so it
              is identical for every seed of this fold (SPLIT CONTRACT).
            * for seed:
                - for arm in [A, B, C], with flags:
                    A → with_interactions=False, with_concurvity_filter=False
                    B → with_interactions=True,  with_concurvity_filter=False
                    C → with_interactions=True,  with_concurvity_filter=True
                  · skip if measures for (arm, fold, seed) already exist.
                  · set_seed(seed) → build a fresh NA2M (init is now seeded)
                    → fit_na2m(model, train_loader, val_loader, pool_loader, hp, <arm flags>)
                    (fit_na2m does NOT seed — caller owns reproducibility).
                  · extract_measures; save measures.
                  · let the model go out of scope (no model held across arms).
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to NA2M eval config YAML")
    args = parser.parse_args()  # noqa: F841  (used once implemented)

    raise NotImplementedError


if __name__ == "__main__":
    main()
