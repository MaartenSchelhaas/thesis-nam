"""
run_na2m_eval.py — K-fold × run evaluation for NA2M arm A (mains only).

Mirrors scripts/nam/evaluate.py exactly so the result is directly comparable
to the original NAM reproduction. Each run within a fold gets a fresh random
train/val subsample of the pool — same methodology as the NAM baseline.

Use this to verify that the NA2M main-effect subnet reproduces the standalone
NAM from Agarwal et al. before extending to arms B and C.

Folder layout:
    base_dir/
        fold_0/
            tuned_config.yaml
            run_0/
                predictions.pt
                done
            run_1/
                ...
            fold_metric.json
            done
        fold_1/
            ...
        results.json
"""

import json
import random

import numpy as np
import torch
from pathlib import Path
from sklearn.model_selection import KFold, train_test_split
from torch.utils.data import DataLoader

from na2m.data.data_utils import load_compas, preprocess
from na2m.models.na2m import NA2M
from na2m.training.fit_na2m import fit_na2m
from na2m.utils.config import load_na2m_config, load_na2m_search_config
from nam.data.dataset import NAMDataset
from nam.training.metrics import auroc
from scripts.na2m.tune_na2m import tune_fold
from scripts.na2m.tune_clarity import load_clarity_search_config, tune_clarity_fold


def _build_model(config, num_features: int, feature_meta) -> NA2M:
    return NA2M(
        num_features=num_features,
        feature_meta=feature_meta,
        num_units=config.num_units,
        hidden_sizes=config.hidden_sizes,
        dropout=config.dropout,
        feature_dropout=config.feature_dropout,
        activation=config.activation,
        inter_units=config.inter_units,
        inter_hidden=config.inter_hidden,
    )


def _predict(model: NA2M, loader: DataLoader) -> torch.Tensor:
    model.eval()
    all_logits = []
    with torch.no_grad():
        for X_batch, _, _ in loader:
            logits, _ = model(X_batch)
            all_logits.append(logits)
    return torch.cat(all_logits)


def _ensemble(run_dirs: list) -> torch.Tensor:
    logits = [
        torch.load(d / "predictions.pt", weights_only=True, map_location="cpu")
        for d in run_dirs
    ]
    return torch.stack(logits).mean(dim=0)


def evaluate_na2m_kfold(
    search_config_path: str,
    X,
    y,
    feature_meta,
    base_dir: Path,
    n_runs: int,
    n_folds: int = 5,
) -> dict:
    """Run the full k-fold × run evaluation for arm A (mains only).

    Mirrors evaluate_kfold in scripts/nam/evaluate.py exactly:
    each run within a fold draws a fresh random train/val subsample.

    Args:
        search_config_path: Path to the search YAML (arm-A tuning).
        X, y: Full dataset as numpy arrays.
        feature_meta: FeatureMeta list from preprocess().
        base_dir: Root output dir, e.g. runs/na2m_arm_a_5fold_20runs/.
        n_runs: Number of random train/val subsamples per fold.
        n_folds: Number of outer CV folds.

    Returns:
        dict with mean, std, fold_metrics.
    """
    fixed_params, search_space = load_na2m_search_config(search_config_path)
    clarity_n_trials, clarity_search_spec = load_clarity_search_config(search_config_path)
    kf = KFold(
        n_splits=n_folds,
        shuffle=True,
        random_state=fixed_params.get("seed", 42),
    )
    num_features = X.shape[1]

    for fold_idx, (pool_idx, test_idx) in enumerate(kf.split(X)):
        print(f"\n--- Fold {fold_idx} ---")
        fold_dir = base_dir / f"fold_{fold_idx}"
        fold_dir.mkdir(parents=True, exist_ok=True)

        if (fold_dir / "done").exists():
            print(f"Fold {fold_idx} already complete, skipping.")
            continue

        X_test, y_test = X[test_idx], y[test_idx]
        y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

        val_frac_of_pool = fixed_params["val_frac"] / (1 - fixed_params["test_frac"])

        tune_idx, tune_val_idx = train_test_split(pool_idx, test_size=val_frac_of_pool)

        # Step 1: tune main-effects hyperparameters once per fold (arm A).
        tuned_config_path = fold_dir / "tuned_config.yaml"
        if not tuned_config_path.exists():
            tune_fold(
                fixed_params,
                search_space,
                feature_meta,
                X[tune_idx],
                y[tune_idx],
                X[tune_val_idx],
                y[tune_val_idx],
                tuned_config_path,
                study_name=f"fold_{fold_idx}_main_search",
            )
            print(f"Fold {fold_idx}: main-effects config saved.")

        # Step 2: tune clarity_regularization (arm B, full pipeline).
        # TODO: enable once arms B and C are added to the run loop below.
        # tune_clarity_fold(
        #     tuned_config_path,
        #     clarity_n_trials,
        #     clarity_search_spec,
        #     feature_meta,
        #     X[tune_idx], y[tune_idx],
        #     X[tune_val_idx], y[tune_val_idx],
        #     study_name=f"fold_{fold_idx}_clarity_search",
        # )

        config = load_na2m_config(str(tuned_config_path))

        test_loader = DataLoader(
            NAMDataset(X_test, y_test),
            batch_size=config.batch_size,
            shuffle=False,
        )

        run_dirs = []
        for i in range(n_runs):
            run_dir = fold_dir / f"run_{i}"
            run_dir.mkdir(parents=True, exist_ok=True)
            run_dirs.append(run_dir)

            if (run_dir / "done").exists():
                print(f"  Run {i} already complete, skipping.")
                continue

            # Fresh random subsample each run — mirrors NAM evaluate.py exactly.
            train_idx, val_idx = train_test_split(pool_idx, test_size=val_frac_of_pool)
            X_train, y_train = X[train_idx], y[train_idx]
            X_val,   y_val   = X[val_idx],   y[val_idx]
            X_pool   = np.concatenate([X_train, X_val])
            y_pool   = np.concatenate([y_train, y_val])

            train_loader = DataLoader(
                NAMDataset(X_train, y_train),
                batch_size=config.batch_size,
                shuffle=True,
            )
            val_loader = DataLoader(
                NAMDataset(X_val, y_val),
                batch_size=config.batch_size,
                shuffle=False,
            )
            pool_loader = DataLoader(
                NAMDataset(X_pool, y_pool),
                batch_size=config.batch_size,
                shuffle=False,
            )

            model = _build_model(config, num_features, feature_meta)
            fit_na2m(
                model,
                train_loader,
                val_loader,
                pool_loader,
                config,
                with_interactions=False,
                with_concurvity_filter=False,
            )

            preds = _predict(model, test_loader)
            torch.save(preds, run_dir / "predictions.pt")
            (run_dir / "done").touch()
            print(f"  Run {i} done.")

        ensembled = _ensemble(run_dirs)
        metric = float(auroc(ensembled, y_test_tensor))
        print(f"  Fold {fold_idx} AUROC (ensembled): {metric:.4f}")

        with open(fold_dir / "fold_metric.json", "w") as f:
            json.dump({"metric": metric, "n_runs": n_runs}, f)
        (fold_dir / "done").touch()

    fold_metrics = []
    for fold_idx in range(n_folds):
        with open(base_dir / f"fold_{fold_idx}" / "fold_metric.json") as f:
            fold_metrics.append(json.load(f)["metric"])

    results = {
        "mean":         float(np.mean(fold_metrics)),
        "std":          float(np.std(fold_metrics)),
        "fold_metrics": fold_metrics,
        "n_runs":       n_runs,
        "n_folds":      n_folds,
    }
    with open(base_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nNA2M arm A (mains only): mean={results['mean']:.4f}, std={results['std']:.4f}")
    return results


if __name__ == "__main__":
    # --- Edit these ---
    SEARCH_CONFIG_PATH = r"C:\Users\maart\OneDrive\Documenten\Universiteit\Scriptie\python_repo\thesis-nam\configs\compas_na2m_search.yaml"
    BASE_DIR = Path("runs/na2m_arm_a_5fold_20runs")
    N_RUNS   = 20
    N_FOLDS  = 5
    FRESH    = False
    # ------------------

    if FRESH and BASE_DIR.exists():
        import shutil
        shutil.rmtree(BASE_DIR)
        print(f"Deleted {BASE_DIR} for fresh run.")

    fixed_params, _ = load_na2m_search_config(SEARCH_CONFIG_PATH)
    df = load_compas(fixed_params["dataset_path"])
    X, y, feature_meta = preprocess(df)

    evaluate_na2m_kfold(
        search_config_path=SEARCH_CONFIG_PATH,
        X=X,
        y=y,
        feature_meta=feature_meta,
        base_dir=BASE_DIR,
        n_runs=N_RUNS,
        n_folds=N_FOLDS,
    )