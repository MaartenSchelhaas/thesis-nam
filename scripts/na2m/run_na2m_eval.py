"""
run_na2m_eval.py — driver for the NA2M evaluation experiment.

The k-fold × seed × arm train+extract loop. This is the orchestration DRIVER
(the analog of tune_nam.py's study loop): it imports the reusable building
blocks from src/na2m and src/nam and wires them together. All real logic
(staged training, measure extraction, B′ derivation, metric reduction) lives in
src — this script only sequences it.

Usage:
    python scripts/na2m/run_na2m_eval.py --config configs/na2m_eval.yaml

Outer structure:
    Fixed before loops: global per-feature grids; fold split (own seed).
    for fold:
        tune hp on the 80% pool (cache); compute density on the 80% pool;
        persist density, feature_meta, fitted scaler, test labels.
        for seed:
            record the internal split.
            for arm in [A, B, C]:
                skip-if-measures-exist (resumability);
                fit_na2m → load_best → extract_measures → save measures;
                record fine_tune_pass_count; keep B's model until B′ is derived.
            derive_b_prime(B, C) → extract → save B′ measures.
    Keep models ephemeral except B (held for B′).

HARD CONSTRAINTS (see src/na2m for the full list):
    - Concurvity/stability on the 80% pool; test fold opened ONCE (accuracy only).
    - Bootstrap units are SEEDS; folds are data/accuracy variation.
    - B′ is derived, not trained. Models ephemeral except B (held for B′).
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
from na2m.training.derive import derive_b_prime  # noqa: F401
from na2m.eval.extract import extract_measures  # noqa: F401


def main():
    """Parse the config and run the full k-fold × seed × arm experiment.

    TODO:
        - argparse: --config (required str).
        - load_na2m_config(args.config).
        - load_compas → preprocess_route2 → build global per-feature grids (fixed).
        - K-fold split with its own seed (80% pool / 20% test per fold).
        - for fold:
            * tune hp on the 80% pool (cache to disk); density_weights;
              persist density, feature_meta, fitted scaler, test labels.
            * for seed:
                - record the internal split.
                - for arm in [A, B, C]:
                    · skip if measures for (arm, fold, seed) already exist.
                    · build a fresh NA2M; fit_na2m(arm flags); load_best; eval;
                      extract_measures; save measures; record fine_tune_pass_count.
                    · keep B's model in memory until B′ is derived.
                - derive_b_prime(B_model, C_result, ...); extract; save B′ measures.
        - Keep models ephemeral except B (held for B′).
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to NA2M eval config YAML")
    args = parser.parse_args()  # noqa: F841  (used once implemented)

    raise NotImplementedError


if __name__ == "__main__":
    main()
