"""
model_runner.py — the NA2M model layer: build, train, save, load, and extract.

Owns everything about a model on disk. "Running" a model means producing it
(by training from a seed) or reconstructing it (by loading from disk — the same
model, deterministically), then extracting + storing its measures. The
orchestrator (run_na2m_eval.py) only decides WHERE things go and in WHAT order; it
never builds a model itself.

The main effects are deterministic given (split, seed), so they are trained ONCE,
persisted, and every arm branches off that base:

    run_main_effects(..., out_dir=mains_dir)     # train mains, save model.pt, extract arm A
    mains = load_main_effects(..., mains_dir/"model.pt")
    run_arm(mains, ..., out_dir=arm_dir, <flags>)# continue from mains, extract arm B/C

run_arm deepcopies the mains base, so it is safe to call repeatedly on the same
model. A future method is just another set of flags.

Persistence split: the mains model is saved as a state_dict (mains-only, so its
`main_centers` buffer + params are a COMPLETE snapshot — the dynamic interaction
dicts are empty). Arm models are NOT persisted; we store their MEASURES instead.
extract_measures itself (src/na2m/eval/extract.py) still needs implementing — the
_extract_and_save hook below is where it plugs in.
"""

import copy
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from na2m.models.na2m import NA2M
from na2m.data.data_utils import load_compas, preprocess, split, make_grid
from na2m.data.dataset import NAMDataset
from na2m.training.fit_na2m import fit_na2m
from na2m.eval.extract import extract_measures
from na2m.utils.config import load_na2m_config

# arm name -> (with_interactions, with_concurvity_filter). Single source of truth
# for which arms exist and the order they run in.
_ARM_FLAGS: dict[str, tuple[bool, bool]] = {
    "mains":      (False, False),  # arm A — mains only
    "gaminet":    (True,  False),  # arm B — GAMI-Net, no gate
    "concurvity": (True,  True),   # arm C — GAMI-Net + concurvity gate
}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def build_model(config, num_features: int, feature_meta) -> NA2M:
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


def _build_loaders(config, X_train, y_train, X_val, y_val):
    """Train / val / pool loaders. pool = train ∪ val — the centering reference."""
    train_loader = DataLoader(
        NAMDataset(X_train, y_train), batch_size=config.batch_size, shuffle=True
    )
    val_loader = DataLoader(
        NAMDataset(X_val, y_val), batch_size=config.batch_size, shuffle=False
    )
    pool_loader = DataLoader(
        NAMDataset(np.concatenate([X_train, X_val]), np.concatenate([y_train, y_val])),
        batch_size=config.batch_size,
        shuffle=False,
    )
    return train_loader, val_loader, pool_loader


def _extract_and_save(
    model: NA2M,
    X_pool: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    feature_meta,
    config,
    out_dir: Path,
) -> None:
    """Extract one model's measures (it is in memory here) and persist them, then flag done.

    Called by run_main_effects (arm A) and run_arm (arm B/C). The orchestrator only
    chose out_dir; everything written here is model_runner's job. The `done`
    sentinel is written LAST so a crash mid-extract leaves the arm un-flagged and a
    rerun redoes it.

    Args:
        model: Trained model (eval mode, best weights restored) — in memory.
        X_pool: The fold's 80% pool (= train ∪ val) — concurvity term vectors.
        X_test: The fold's test slice — stability term vectors + logits.
        y_test: Test labels — stored for reduce.accuracy_summary.
        feature_meta: FeatureMeta list (term set + grids).
        config: NA2MConfig (grid_size).
        out_dir: This arm's output dir.

    TODO:
        - out_dir.mkdir(parents=True, exist_ok=True)
        - grids = {j: make_grid(feature_meta, j, config.grid_size)
                   for j in range(model.num_features)}
        - measures = extract_measures(model, X_pool, X_test, grids, feature_meta)
        - measures["y_test"] = y_test           # accuracy_summary needs test labels
        - torch.save(measures, out_dir / "measures.pt")
        - (out_dir / "done").touch()            # LAST — marks this arm complete
    """
    raise NotImplementedError


def run_main_effects(
    config,
    feature_meta,
    X_train, y_train,
    X_val, y_val,
    X_test, y_test,
    seed: int,
    out_dir: Path,
) -> NA2M:
    """Train + center the mains once, persist the model, extract arm-A measures.

    This is the shared base for every arm (and is itself arm A). The mains model is
    saved to out_dir/"model.pt" so arms can reload + deepcopy from it later.

    Args:
        config: NA2MConfig with model + training hyperparameters.
        feature_meta: FeatureMeta list from preprocess().
        X_train, y_train: Training split.
        X_val, y_val: Validation split.
        X_test, y_test: Test slice — for the arm-A measure extraction.
        seed: Init/optimization seed (governs the mains, deterministically).
        out_dir: The mains output dir; receives model.pt, measures.pt, done.

    Returns:
        The trained mains-only NA2M (also persisted to out_dir/"model.pt").
    """
    set_seed(seed)
    train_loader, val_loader, pool_loader = _build_loaders(config, X_train, y_train, X_val, y_val)

    model = build_model(config, X_train.shape[1], feature_meta)
    fit_na2m(
        model,
        train_loader,
        val_loader,
        pool_loader,
        config,
        with_interactions=False,
        with_concurvity_filter=False,
    )

    # Persist the mains base (state_dict is a complete snapshot for mains-only).
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out_dir / "model.pt")

    # Arm A == the mains model: extract + store its measures, flag done.
    X_pool = np.concatenate([X_train, X_val])
    _extract_and_save(model, X_pool, X_test, y_test, feature_meta, config, out_dir)
    return model


def load_main_effects(
    config,
    feature_meta,
    num_features: int,
    path: Path,
) -> NA2M:
    """Rebuild a mains NA2M and load its weights — the base to deepcopy arms from.

    Used on resume / when a new arm is added later: instead of retraining the mains,
    reconstruct them from disk so arms branch off the SAME base.

    Args:
        config: NA2MConfig with the model architecture fields.
        feature_meta: FeatureMeta list from preprocess().
        num_features: Number of input columns.
        path: Path to the saved mains state_dict (out_dir/"model.pt").

    Returns:
        The reconstructed mains NA2M (eval mode).
    """
    model = build_model(config, num_features, feature_meta)
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    model.eval()
    return model


def run_arm(
    config,
    mains_model: NA2M,
    feature_meta,
    X_train, y_train,
    X_val, y_val,
    X_test, y_test,
    seed: int,
    out_dir: Path,
    *,
    with_interactions: bool,
    with_concurvity_filter: bool,
) -> dict:
    """Run ONE arm by continuing from an already-trained mains model, then extract.

    deepcopies the mains base (so the base is never mutated and arms stay
    independent), seeds the interaction stages, runs from Stage 2 onward via
    fit_na2m(mains_pretrained=True), then extracts + stores the arm's measures.

    Args:
        config: NA2MConfig with model + training hyperparameters.
        mains_model: Trained mains base (from run_main_effects/load_main_effects). NOT modified.
        feature_meta: FeatureMeta list from preprocess().
        X_train, y_train: Training split (same split used for the mains).
        X_val, y_val: Validation split (same split used for the mains).
        X_test, y_test: Test slice — for measure extraction.
        seed: Seed for the interaction stages (same across arms → B and C paired).
        out_dir: This arm's output dir; receives measures.pt, done.
        with_interactions: False → arm A; True → arm B/C.
        with_concurvity_filter: False → arm B (NoGate); True → arm C (gate on).

    Returns:
        fit_na2m's result dict: {"model": NA2M, "active_pairs": [...]}.
    """
    set_seed(seed)
    model = copy.deepcopy(mains_model)
    train_loader, val_loader, pool_loader = _build_loaders(config, X_train, y_train, X_val, y_val)

    result = fit_na2m(
        model,
        train_loader,
        val_loader,
        pool_loader,
        config,
        with_interactions=with_interactions,
        with_concurvity_filter=with_concurvity_filter,
        mains_pretrained=True,
    )

    X_pool = np.concatenate([X_train, X_val])
    _extract_and_save(result["model"], X_pool, X_test, y_test, feature_meta, config, out_dir)
    return result


if __name__ == "__main__":
    # Manual smoke test (requires extract_measures to be implemented).
    CONFIG_PATH = r"C:\Users\maart\OneDrive\Documenten\Universiteit\Scriptie\python_repo\thesis-nam\configs\compas-scores-two-years_na2m_tuned.yaml"
    config = load_na2m_config(CONFIG_PATH)
    seed = config.seed

    df = load_compas(config.dataset_path)
    X, y, feature_meta = preprocess(df)
    X_train, X_val, X_test, y_train, y_val, y_test = split(
        X, y, config.val_frac, config.test_frac, config.seed
    )

    out_root = Path("runs/model_runner_smoke")
    mains_dir = out_root / "mains"

    # Train the mains once (saves model.pt + extracts arm-A measures), then branch.
    run_main_effects(config, feature_meta, X_train, y_train, X_val, y_val,
                     X_test, y_test, seed, mains_dir)
    mains = load_main_effects(config, feature_meta, X.shape[1], mains_dir / "model.pt")

    for arm, (with_interactions, with_concurvity_filter) in _ARM_FLAGS.items():
        if not with_interactions:
            continue  # arm A already handled by run_main_effects
        run_arm(config, mains, feature_meta, X_train, y_train, X_val, y_val,
                X_test, y_test, seed, out_root / arm,
                with_interactions=with_interactions,
                with_concurvity_filter=with_concurvity_filter)
