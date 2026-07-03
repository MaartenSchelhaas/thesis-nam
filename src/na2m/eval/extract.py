"""
extract — after training a model, pull out everything we need for evaluation and save it to disk.

After each model is trained we immediately extract all the numbers we will ever need from it,
save them to measures.pt, and throw the model away. This means evaluation is fully offline:
every metric is computed from the saved files, no model is ever reloaded.

We store three kinds of things:
  - Raw per-subnet output vectors on the pool and test set, for concurvity and stability metrics.
  - The summed forward-pass logits on the test set, for accuracy.
  - Centered shape curves evaluated on a feature grid, for the shape plots.
"""

from typing import TypedDict

import numpy as np
import torch

# A subnet identifier. Mirrors concurvity.SubnetId; defined here so `eval` does not
# depend on `utils` for a type alias.
SubnetId = tuple[str, int] | tuple[str, int, int]  # ("main", j) | ("inter", j, k)


class MainCurve(TypedDict):
    """1-D shape curve for a main effect (the Agarwal per-run line)."""
    inputs: np.ndarray   # real-unit axis; (G,) continuous, (n_levels,) categorical
    outputs: np.ndarray  # centered curve, same length as inputs


class InterCurve(TypedDict):
    """2-D shape surface for an interaction (heatmap; opt-in, plotting only)."""
    input1: np.ndarray   # axis for feature j, (G_j,)
    input2: np.ndarray   # axis for feature k, (G_k,)
    outputs: np.ndarray  # (G_j, G_k) surface from the meshgrid evaluation


class Density(TypedDict, total=False):
    """Marginal data density of ONE main feature, for the shape-plot bands.

    Continuous → {edges, scores} (histogram); categorical → {levels, scores}.
    """
    edges: np.ndarray    # continuous: bin edges, (B+1,)
    levels: np.ndarray   # categorical: level values, (n_levels,)
    scores: np.ndarray   # density/frequency, (B,) continuous or (n_levels,) cat


class Measures(TypedDict):
    """The on-disk schema of one model's measures.pt.

    Every reduce.py metric is a pure function of these fields. subnet_vectors_pool /
    subnet_vectors_test share the SAME keys. y_test is the ONLY field not populated
    here — model_runner._extract_and_save adds it before saving.
    """
    subnet_vectors_pool: dict[SubnetId, np.ndarray]  # per subnet on X_pool -> concurvity
    subnet_vectors_test: dict[SubnetId, np.ndarray]  # per subnet on X_test -> stability
    logits: np.ndarray                               # (N_test,) forward()[0] on X_test
    pairs: list[tuple[int, int]]                     # active interactions; [] for arm A
    curves: dict[SubnetId, MainCurve | InterCurve]   # shape plots (mains always)
    density: dict[int, Density]                      # per MAIN feature index, from X_pool
    y_test: np.ndarray                               # (N_test,) labels — added downstream


def _extract_raw_vectors(
    model,
    all_subnet_ids: list[SubnetId],
    X_pool_tensor: torch.Tensor,
    X_test_tensor: torch.Tensor,
) -> tuple[dict, dict, np.ndarray]:
    """Extract raw (uncentered) per-subnet output vectors and logits.

    Returns RAW outputs — centering happens in the reducer, not here, because
    the correct reference sample differs per metric:
        subnet_vectors_pool -> concurvity  (OLS intercept centers over the pool)
        subnet_vectors_test -> stability   (test-fold mean centers over the test set)

    Args:
        model: Trained NA2M in eval mode.
        all_subnet_ids: All subnet ids to extract (mains + active interactions).
        X_pool_tensor: Full training pool as a tensor on the model's device.
        X_test_tensor: Test fold as a tensor on the model's device.

    Returns:
        (subnet_vectors_pool, subnet_vectors_test, logits)
    """
    subnet_vectors_pool: dict[SubnetId, np.ndarray] = {}
    subnet_vectors_test: dict[SubnetId, np.ndarray] = {}

    with torch.no_grad():
        for subnet_id in all_subnet_ids:
            pool_raw = model.raw_subnet_output(subnet_id, X_pool_tensor)  # (N_pool, 1)
            subnet_vectors_pool[subnet_id] = pool_raw.squeeze(1).cpu().numpy()

            test_raw = model.raw_subnet_output(subnet_id, X_test_tensor)  # (N_test, 1)
            subnet_vectors_test[subnet_id] = test_raw.squeeze(1).cpu().numpy()

        # model(x) returns (out, dropout_out); [0] is the per-sample scalar logit.
        # Not recoverable from subnet vectors — centering offsets are baked into _bias.
        logits = model(X_test_tensor)[0].cpu().numpy()  # (N_test,)

    return subnet_vectors_pool, subnet_vectors_test, logits


def _extract_shape_curves(
    model,
    grids: dict[int, np.ndarray],
    feature_meta,
    device: torch.device,
) -> dict[SubnetId, MainCurve]:
    """Extract centered shape-plot curves for all main effects.

    Evaluates each main subnet on its feature grid using pool-centering
    (centered_subnet_output subtracts main_centers). The stored x-axis (inputs)
    is already in real units so the plotting function needs no feature_meta.

    Args:
        model: Trained NA2M in eval mode.
        grids: Per-feature model-space grids from make_grid, keyed by feature index.
        feature_meta: FeatureMeta list — used for the inverse-transform of x-axis values.
        device: Device the model lives on (tensors must match).

    Returns:
        Dict keyed by ("main", j) with {"inputs": real_unit_axis, "outputs": centered_curve}.
    """
    curves: dict[SubnetId, MainCurve] = {}

    with torch.no_grad():
        for feature_idx in range(model.num_features):
            subnet_id = ("main", feature_idx)
            meta = feature_meta[feature_idx]

            # Grid in model space: num → 256 pts in [0, 1]; cat → integer codes 0..n_levels-1
            grid = grids[feature_idx]

            # Reshape to (G, 1) — the column shape the subnet was trained with.
            x_col = torch.tensor(grid, device=device).unsqueeze(1)              # (G, 1)
            outputs = model.centered_subnet_output(subnet_id, x_col)            # (G, 1)
            outputs = outputs.squeeze(1).cpu().numpy()                          # (G,)

            # Convert model-space grid to real units for the stored x-axis.
            if meta.type == "num":
                inputs = grid * (meta.max - meta.min) + meta.min               # inverse MinMaxScaler
            else:
                inputs = np.array(meta.levels)                                  # original category strings

            curves[subnet_id] = {"inputs": inputs, "outputs": outputs}

    return curves


def extract_measures(
    model,
    X_pool: np.ndarray,
    X_test: np.ndarray,
    grids: dict[int, np.ndarray],
    feature_meta,
    *,
    include_inter_curves: bool = False,
) -> dict:
    """Extract the durable measures from a trained NA2M (everything but y_test).

    Args:
        model: Trained NA2M, eval mode, best weights restored.
        X_pool: The fold's 80% train pool. Concurvity term vectors are extracted here.
        X_test: The fold's test slice. Stability term vectors and logits extracted here.
        grids: Per-feature evaluation grids from make_grid, keyed by feature index.
        feature_meta: Per-feature metadata (type / n_levels / inverse-transform).
        include_inter_curves: Reserved for 2-D interaction surface curves (not yet implemented).

    Returns:
        A Measures dict WITHOUT y_test (the caller adds it before saving to disk).
    """
    assert not model.training, "model must be in eval mode before extracting measures"

    device = model._bias.device
    X_pool_tensor = torch.from_numpy(X_pool).to(device)
    X_test_tensor = torch.from_numpy(X_test).to(device)

    main_ids = [("main", j) for j in range(model.num_features)]
    inter_ids = [("inter", j, k) for j, k in model.active_interaction_pairs()]
    all_subnet_ids = main_ids + inter_ids

    subnet_vectors_pool, subnet_vectors_test, logits = _extract_raw_vectors(
        model, all_subnet_ids, X_pool_tensor, X_test_tensor,
    )

    curves = _extract_shape_curves(model, grids, feature_meta, device)

    pairs = model.active_interaction_pairs()

    return {
        "subnet_vectors_pool": subnet_vectors_pool,
        "subnet_vectors_test": subnet_vectors_test,
        "logits": logits,
        "pairs": pairs,
        "curves": curves,
        "density": {},  #TODO not implemented
    }
