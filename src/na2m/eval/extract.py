"""
extract — pull the durable MEASURES out of a live NA2M before discarding it.

Per (arm, fold, seed) the harness trains a model, extracts measures in-process,
writes them to disk, and lets the model go out of scope. We persist the
MEASURES, never the model parameters.

HARD CONSTRAINTS:
    - PRECONDITION: model is in eval mode AND best weights have been restored.
    - Store RAW term outputs — never pre-centered. Centering happens per-metric
      in the reducer (over whichever evaluation set that metric uses).
    - Key everything by subnet_id, never positionally.
    - Two distinct evaluation sets, stored separately and NOT interchangeable:
        * subnet_vectors_pool : ALL terms on X_pool  -> consumed by CONCURVITY
          (Kovács observed-concurvity index is a property of the fit on the
           training data the model saw).
        * subnet_vectors_test : ALL terms on X_test  -> consumed by STABILITY
          (cross-seed comparison on a shared, held-out set drawn from the data
           distribution; density is intrinsic to the sample, no grid, no weights).
    - logits and subnet_vectors_test are DIFFERENT extractions that share X_test:
      term vectors are per-term + RAW (stability); logits are the summed forward
      pass (accuracy). Neither is recoverable from the other.

The Measures TypedDict below is the single authoritative schema for what lands in
measures.pt — produced here, completed by scripts/na2m/model_runner._extract_and_save
(which adds y_test), and consumed by src/na2m/eval/reduce.py.
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
        X_pool: The fold's 80% train pool. Concurvity term vectors AND the density
            bands are computed here (property of the fit / data the model saw).
        X_test: The fold's test slice. Stability term vectors AND logits here.
        grids: Per-feature evaluation grids (from make_grid), keyed by feature index.
            Shared & deterministic across runs so curves overlay. PLOTTING ONLY.
        feature_meta: Per-feature metadata (type / n_levels / inverse-transform).
        include_inter_curves: If True, also extract the 2-D interaction surfaces
            (off by default — heavier, plotting aid only).

    Returns:
        A Measures dict WITHOUT y_test (the caller adds it). See the Measures
        TypedDict for the full schema and per-field shapes.

    TODO:
        - assert model.training is False  (best weights already restored upstream).
        - Term set = [("main", j) for j in range(model.num_features)]
                     + [("inter", j, k) for (j, k) in model.active_interaction_pairs()].
        - Under torch.no_grad():
          * subnet_vectors_pool[sid] = model.raw_subnet_output(sid, X_pool) for every
            subnet, stored RAW (numpy, 1-D). For CONCURVITY.
          * subnet_vectors_test[sid] = model.raw_subnet_output(sid, X_test) for every
            subnet, stored RAW (numpy, 1-D). SAME keys as pool. For STABILITY.
          * logits = model(X_test)[0]  (eval mode → dropout off → deterministic).
          * pairs = model.active_interaction_pairs().
          * curves: via iter_subnets() fns (CENTERED deployment path — these feed a
            plot, not a metric):
              - ("main", j):  evaluate on grids[j]  -> MainCurve {inputs, outputs}.
                inputs = grids[j] in real units (from make_grid's inverse-transform).
              - ("inter", j, k): ONLY if include_inter_curves. meshgrid(grids[j],
                grids[k]); evaluate the interaction fn on the flattened (G_j*G_k, 2)
                input; reshape to (G_j, G_k) -> InterCurve {input1, input2, outputs}.
          * density: per MAIN feature j from X_pool, in real units:
              - continuous: np.histogram(real_values_j, bins=...) -> {edges, scores}.
              - categorical: per-level frequency over n_levels      -> {levels, scores}.
            Keyed by feature INDEX (not term_id) — reused on main plots and on the
            margins of interaction plots.
        - RAW (uncentered) term vectors via raw_subnet_output — NOT iter_subnets, NOT the
          model's deployment centering — so the reducer re-centers per metric
          (test-fold mean for stability, OLS intercept for concurvity).
        - Convert all tensors to numpy for persistence; key subnet dicts by subnet_id.
    """
    assert not model.training, "model must be in eval mode before extracting measures"

    # Build the full list of subnet IDs: all main effects, then active interactions.
    # For arm A (mains only), active_interaction_pairs() returns [] so inter_ids is empty.
    main_ids = [("main", j) for j in range(model.num_features)]
    inter_ids = [("inter", j, k) for j, k in model.active_interaction_pairs()]
    all_subnet_ids = main_ids + inter_ids

    # Move the input arrays to the same device the model lives on.
    device = model._bias.device
    X_pool_tensor = torch.from_numpy(X_pool).to(device)
    X_test_tensor = torch.from_numpy(X_test).to(device)

    # Extract raw per-subnet output vectors for BOTH evaluation sets.
    # We call raw_subnet_output once per subnet per set and immediately convert to numpy.
    # RAW means no centering — the reducer applies the correct centering per metric:
    #   subnet_vectors_pool -> concurvity (OLS intercept centers over the pool)
    #   subnet_vectors_test -> stability  (test-fold mean centers over the test set)
    subnet_vectors_pool: dict[SubnetId, np.ndarray] = {}
    subnet_vectors_test: dict[SubnetId, np.ndarray] = {}

    with torch.no_grad():
        for subnet_id in all_subnet_ids:
            pool_raw = model.raw_subnet_output(subnet_id, X_pool_tensor)  # (N_pool, 1)
            subnet_vectors_pool[subnet_id] = pool_raw.squeeze(1).cpu().numpy()

            test_raw = model.raw_subnet_output(subnet_id, X_test_tensor)  # (N_test, 1)
            subnet_vectors_test[subnet_id] = test_raw.squeeze(1).cpu().numpy()

        # Logits: summed forward pass on the test set (eval mode -> dropout off -> deterministic).
        # model(x) returns (out, dropout_out); [0] is the per-sample scalar prediction.
        # These are NOT recoverable from the per-subnet vectors because centering offsets
        # are baked into the bias during training — don't try to reconstruct them.
        logits_tensor = model(X_test_tensor)[0]  # (N_test,)
        logits = logits_tensor.cpu().numpy()

    # The active pair list is a plain Python list — no tensor, no no_grad needed.
    # Empty for arm A; populated for arms B and C after the Stage-2 prune sweep.
    pairs = model.active_interaction_pairs()

    # curves and density are intentionally empty — shape plot extraction is deferred.
    # y_test is intentionally absent — _extract_and_save adds it before torch.save.
    return {
        "subnet_vectors_pool": subnet_vectors_pool,
        "subnet_vectors_test": subnet_vectors_test,
        "logits": logits,
        "pairs": pairs,
        "curves": {},
        "density": {},
    }