"""
fit_na2m — three-stage training pipeline for one NA2M arm on one (fold, seed).

Stage 1: train main effects, center.
Stage 2: FAST screen → block-train top-M interactions (mains frozen) → single
         forward prune sweep with a predictive-contribution gate and, for arm C,
         a concurvity gate.
Stage 3: unfreeze all, fine-tune once with the clarity penalty, re-center.

Arms differ only by two flags: with_interactions and with_concurvity_filter.
Arm A skips stages 2-3 entirely. Arms B and C run the same pipeline; C fires
the concurvity gate inside the sweep. Both fine-tune exactly once.
"""
from na2m.models.na2m import NA2M
from na2m.training.trainer import Trainer
from na2m.utils.config import NA2MConfig
from na2m.selection.policy import SelectionPolicy, NoGate, ConcurvityGate
from na2m.selection.fast import fast_screen
from na2m.data.dataset import NAMDataset
from typing import cast
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader


def fit_na2m(
    model: NA2M,
    train_loader: DataLoader,
    val_loader: DataLoader,
    pool_loader: DataLoader,
    config: NA2MConfig,
    *,
    with_interactions: bool,
    with_concurvity_filter: bool,
    mains_pretrained: bool = False,
    trial=None,
) -> None:
    """Run the staged NA2M training pipeline for one arm. Mutates model in place.

    Set the RNG seed and build the data loaders before calling — this function
    never re-seeds or re-splits. The caller must deepcopy the mains model before
    passing it for arms B or C, since fit_na2m modifies it in place.

    Args:
        model: NA2M to train. For arm A: freshly initialised. For arms B/C: a
               deepcopy of the already-trained mains model.
        train_loader: Training split (same split used for all seeds within a fold).
        val_loader: Validation split (same split used for all seeds within a fold).
        pool_loader: Full 80% pool — centering reference.
        config: Hyperparameters.
        with_interactions: False → arm A (stage 1 only).
        with_concurvity_filter: True → arm C (concurvity gate in stage 2); False → arm B.
        mains_pretrained: If True, skip stage 1 (model already has trained mains).
        trial: Optuna trial for stage-1 pruning; ignored when mains_pretrained=True.
    """
    if not mains_pretrained:
        stage1_main(model, train_loader, val_loader, pool_loader, config, trial=trial)

    if not with_interactions:
        model.eval()
        return

    policy: SelectionPolicy = (
        ConcurvityGate(threshold=config.concurvity_threshold)
        if with_concurvity_filter
        else NoGate()
    )

    stage2_select(model, train_loader, val_loader, pool_loader, config, selection_policy=policy)
    stage3_finetune(model, train_loader, val_loader, pool_loader, config)

    model.eval()


def stage1_main(
    model: NA2M,
    train_loader: DataLoader,
    val_loader: DataLoader,
    pool_loader: DataLoader,
    config: NA2MConfig,
    *,
    trial=None,
) -> None:
    """Train the main bank (Trainer over all params), restore best, then center.

    clarity_lambda is 0.0 here — no interactions exist yet so the penalty is a no-op.

    Args:
        model: NA2M instance (interactions empty at entry).
        train_loader: Internal training split loader (fold-keyed).
        val_loader: Internal validation split loader (fold-keyed).
        pool_loader: Full 80% pool loader — reference sample for centering.
        config: NA2MConfig with training hyperparameters.
    """
    stage_1_trainer = Trainer(
        model=model,
        lr=config.lr,
        decay_rate=config.decay_rate,
        output_regularization=config.output_regularization,
        l2_regularization=config.l2_regularization,
        task=config.task,
        num_epochs=config.num_epochs,
        patience=config.patience,
        val_check_interval=config.val_check_interval,
        clarity_lambda=0.0,
    )
    stage_1_trainer.train(train_loader, val_loader, trial)
    stage_1_trainer.load_best()
    model.center_main_effects(pool_loader)


# ---------------------------------------------------------------------------
# Private helpers for stage2_select: pure functions of stored output vectors
# ---------------------------------------------------------------------------

def _collect_outputs(
    model: NA2M,
    loader: DataLoader,
) -> tuple[list[np.ndarray], dict[str, np.ndarray], np.ndarray]:
    """One no-grad eval pass: collect every active term's centered output vector.

    Args:
        model: NA2M instance; switched to eval mode internally.
        loader: DataLoader yielding (X, y, weights) batches.

    Returns:
        main_vecs:  list of main subnet output vectors
        inter_vecs: list of interaction subnet output vectors
                    dict "j,k" vector (centered). Empty if no interactions.
        targets:    vector of target labels.
    """
    was_training = model.training
    model.eval()
    device = next(model.parameters()).device

    n_features = model.num_features
    pairs = model.active_interaction_pairs()

    main_lists: list[list[np.ndarray]] = [[] for _ in range(n_features)]
    inter_lists: dict[str, list[np.ndarray]] = {f"{j},{k}": [] for j, k in pairs}
    target_list: list[np.ndarray] = []

    with torch.no_grad():
        for X_batch, y_batch, _ in loader:
            X_batch = X_batch.to(device)
            main_out_list = model.main_outputs(X_batch)
            for j, out in enumerate(main_out_list):
                main_lists[j].append(out.squeeze(1).cpu().numpy())

            inter_out_list = model.inter_outputs(X_batch)
            for (j, k), out in zip(pairs, inter_out_list):
                inter_lists[f"{j},{k}"].append(out.squeeze(1).cpu().numpy())

            target_list.append(y_batch.cpu().numpy())

    if was_training:
        model.train()

    main_vecs = [np.concatenate(main_lists[j]) for j in range(n_features)]
    inter_vecs = {f"{j},{k}": np.concatenate(inter_lists[f"{j},{k}"]) for j, k in pairs}
    targets = np.concatenate(target_list)
    return main_vecs, inter_vecs, targets


def _rank_by_contribution(
    inter_vecs: dict[str, np.ndarray],
    pairs: list[tuple[int, int]],
) -> list[tuple[int, int]]:
    """Rank pairs by variance of their centered output on the train split, descending.

    Args:
        inter_vecs: dict "j,k" -> (N,) centered output array (train split).
        pairs: Candidate pairs to rank.

    Returns:
        Pairs sorted by output variance, highest first.
    """
    return sorted(
        pairs,
        key=lambda p: float(np.var(inter_vecs[f"{p[0]},{p[1]}"])),
        reverse=True,
    )


def _partial_val_loss(
    val_main_sum: np.ndarray,
    val_inter_vecs: dict[str, np.ndarray],
    accepted: list[tuple[int, int]],
    bias: float,
    targets: np.ndarray,
    task: str,
) -> float:
    """Val loss of {mains + accepted interactions} — no model calls, pure numpy.

    Because the model is additive, the partial prediction is just the sum of
    the pre-collected centered output vectors for the terms we want to include.
    Non-accepted subnets (still in inter_nns) contribute nothing here.

    Args:
        val_main_sum: (N,) sum of all centered main outputs on the val split.
        val_inter_vecs: dict "j,k" -> (N,) centered inter output on val split.
        accepted: Pairs whose subnets contribute to this partial prediction.
        bias: model._bias scalar value.
        targets: (N,) val labels.
        task: 'classification' or 'regression'.

    Returns:
        Cross-entropy loss (classification) or MSE (regression).
    """
    pred = val_main_sum + bias
    for j, k in accepted:
        pred = pred + val_inter_vecs[f"{j},{k}"]

    pred_t = torch.from_numpy(pred).float()
    targets_t = torch.from_numpy(targets).float()

    if task == "classification":
        return float(F.binary_cross_entropy_with_logits(pred_t, targets_t))
    else:
        return float(F.mse_loss(pred_t, targets_t))


def _eta_cut(
    val_losses: list[float],
    eta: float,
) -> int:
    """Return the fewest accepted interactions whose val loss is within eta of the minimum.

    Args:
        val_losses: Loss sequence; index 0 = mains only, index k = k interactions.
        eta: Tolerance (hp.eta_prune). 0 -> keep argmin; approaching 1 -> keep fewer.

    Returns:
        Number of interactions to keep (0 means mains only, no interactions).
    """
    losses = np.array(val_losses)
    best = float(np.min(losses))
    loss_range = float(np.max(losses) - best)

    if loss_range > 0:
        # Normalize each loss to [0, 1]: 0 = best, 1 = worst.
        # Find the first index (fewest interactions) within eta of the minimum.
        normalized = (losses - best) / loss_range
        candidates = np.where(normalized < eta)[0]
        if len(candidates) > 0:
            return int(candidates[0])

    # Degenerate: all losses identical (interactions added no signal) → keep none.
    return 0


def stage2_select(
    model: NA2M,
    train_loader: DataLoader,
    val_loader: DataLoader,
    pool_loader: DataLoader,
    hp: NA2MConfig,
    *,
    selection_policy: SelectionPolicy,
) -> None:
    """FAST screen → block-train top-M interactions jointly → single forward prune sweep.

    The sweep ranks block-trained subnets by output variance, applies the selection
    policy (concurvity gate for arm C, no-op for arm B), then η-prunes the accepted
    prefix by val loss. No retraining at any point; the model fine-tunes once in stage 3.

    Args:
        model: NA2M with mains trained and centered.
        train_loader: Internal training split (fold-keyed).
        val_loader: Internal validation split (fold-keyed).
        pool_loader: Full 80% pool — for centering and concurvity basis.
        hp: Hyperparameters (top_m, eta_prune, block_train_epochs, ...).
        selection_policy: NoGate() for arm B; ConcurvityGate(threshold) for arm C.
    """
    # --- FAST screen ---
    pool_dataset = cast(NAMDataset, pool_loader.dataset)
    X_pool = pool_dataset.X.numpy()
    y_pool = pool_dataset.y.numpy()
    ranked_pairs = fast_screen(model, X_pool, y_pool, task=hp.task)
    top_m = ranked_pairs[: hp.top_m]
    print(f"[stage2] FAST screen: top-{len(top_m)} pairs selected from {len(ranked_pairs)} candidates")

    # --- Add interactions + block-train (mains + _bias frozen) ---
    model.add_interactions(top_m)
    model.set_main_trainable(False)
    print(f"[stage2] Block-training {len(top_m)} interaction subnets for {hp.block_train_epochs} epochs...")

    block_trainer = Trainer(
        model=model,
        lr=hp.lr,
        decay_rate=hp.decay_rate,
        output_regularization=hp.output_regularization,
        l2_regularization=hp.l2_regularization,
        task=hp.task,
        num_epochs=hp.block_train_epochs,
        patience=hp.patience,
        val_check_interval=hp.val_check_interval,
        clarity_lambda=hp.clarity_regularization,
    )
    block_trainer.train(train_loader, val_loader)
    block_trainer.load_best()
    print(f"[stage2] Block-train done. Best val metric: {block_trainer.best_val_metric:.4f}")

    model.center_interactions(pool_loader, fold_bias=False)

    # --- Collect centered per-term output vectors (one pass each; no model calls during sweep) ---
    _, train_inter_vecs, _ = _collect_outputs(model, train_loader)
    val_main_vecs, val_inter_vecs, val_targets = _collect_outputs(model, val_loader)
    pool_main_vecs, pool_inter_vecs, _ = _collect_outputs(model, pool_loader)

    # Sum all centered main outputs on val split — fixed for the entire sweep
    val_main_sum: np.ndarray = np.sum(np.stack(val_main_vecs, axis=0), axis=0)
    bias = model._bias.item()

    # --- Rank interactions by contribution: variance of centered output on train split, descending ---
    ranked = _rank_by_contribution(train_inter_vecs, top_m)
    print(f"[stage2] Contribution ranking (variance, descending):")
    for j, k in ranked:
        var = float(np.var(train_inter_vecs[f"{j},{k}"]))
        print(f"  ({j},{k})  var={var:.6f}")

    # --- Single forward prune sweep (eval only — no retrain) ---
    accepted: list[tuple[int, int]] = []
    accepted_pool_vecs: list[np.ndarray] = []

    # Index 0 = baseline: mains only, no interactions
    baseline_loss = _partial_val_loss(val_main_sum, val_inter_vecs, [], bias, val_targets, hp.task)
    val_losses: list[float] = [baseline_loss]
    print(f"[stage2] Sweep baseline val loss (mains only): {baseline_loss:.5f}")

    for pair in ranked:
        candidate_vec = pool_inter_vecs[f"{pair[0]},{pair[1]}"]
        if not selection_policy.should_accept(candidate_vec, accepted_pool_vecs, pool_main_vecs):
            print(f"  pair {pair} — SKIPPED by concurvity gate")
            continue

        accepted.append(pair)
        accepted_pool_vecs.append(candidate_vec)
        loss = _partial_val_loss(val_main_sum, val_inter_vecs, accepted, bias, val_targets, hp.task)
        val_losses.append(loss)
        print(f"  pair {pair} — accepted  val_loss={loss:.5f}")

    # --- η-prune: keep the fewest accepted interactions within eta of the best val loss ---
    cut = _eta_cut(val_losses, hp.eta_prune)
    survivors = accepted[:cut]
    print(f"[stage2] eta-prune: val_losses={[round(l,5) for l in val_losses]}  cut={cut}  survivors={len(survivors)}")

    # --- Drop non-survivors (concurvity-skipped + predictive-cut) and fold survivors ---
    survivors_set = set(survivors)
    for pair in top_m:
        if pair not in survivors_set:
            model.remove_interaction(*pair)
    print(f"[stage2] Final active pairs: {model.active_interaction_pairs()}")

    # Unfreeze everything — stage 3 fine-tunes all params and owns the final centering
    model.set_main_trainable(True)


def stage3_finetune(
    model: NA2M,
    train_loader: DataLoader,
    val_loader: DataLoader,
    pool_loader: DataLoader,
    hp: NA2MConfig,
) -> None:
    """Fine-tune all params once with the clarity penalty, then recenter.

    This is the model's only fine-tune, for both arm B and arm C.

    Args:
        model: NA2M instance (interactions selected & pruned, all params unfrozen).
        train_loader: Internal training split loader (fold-keyed).
        val_loader: Internal validation split loader (fold-keyed).
        pool_loader: Full 80% pool loader — reference sample for recentering.
        hp: NA2MConfig with training hyperparameters.
    """
    finetune_trainer = Trainer(
        model=model,
        lr=hp.lr,
        decay_rate=hp.decay_rate,
        output_regularization=hp.output_regularization,
        l2_regularization=hp.l2_regularization,
        task=hp.task,
        num_epochs=hp.finetune_epochs,
        patience=hp.patience,
        val_check_interval=hp.val_check_interval,
        clarity_lambda=hp.clarity_regularization,
    )
    finetune_trainer.train(train_loader, val_loader)
    finetune_trainer.load_best()

    model.center_main_effects(pool_loader)
    model.center_interactions(pool_loader, fold_bias=True)


# ----------------------------------------------------------------------------
# REMOVED: stage4_concurvity (iterative remove-and-re-fine-tune).
#
# The old methodology scored interactions on the FINE-TUNED model, removed the
# most concurve pair, and RE-FINE-TUNED, looping to a fixed point (capped by
# max_concurvity_iters). That path is deleted. The ONLY place re-fine-tuning was
# assumed is gone: concurvity is now a SELECTION-TIME gate inside stage2_select's
# single sweep, and the model fine-tunes exactly once in stage3_finetune.
#
# If you find any remaining caller that loops fine-tuning or references
# max_concurvity_iters / a per-removal Trainer rebuild, it is leftover from the
# old design and should be removed.
# ----------------------------------------------------------------------------