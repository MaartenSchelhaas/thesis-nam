"""
fit_na2m — staged-training orchestrator for the NA2M arms.

Drives the GAMI-Net-style training pipeline on ONE (fold, seed) for ONE arm,
mutating a dynamic NA2M model in place. Staging lives here — there is NO
separate TrainerNA2M; each stage builds/rebuilds a Trainer over the right
parameter subset.

Stages (THREE — the old Stage-4 iterative concurvity removal is GONE):
    1. stage1_main    — train the main bank; center.
    2. stage2_select  — FAST screen → top-M → add interactions → block-train all
                        of them jointly (mains frozen) → ONE forward prune sweep
                        applying two gates in a single pass:
                          (a) concurvity gate  [arm C only]  — skip a candidate
                              whose block-trained output is too redundant with
                              the mains + already-ACCEPTED interactions;
                          (b) predictive-contribution gate — min-max cut on the
                              accepted candidates' validation-loss sequence.
    3. stage3_finetune — unfreeze → fine-tune all params ONCE with the
                        marginal-clarity penalty → re-center.

Arm mapping (the ONLY differences are the two flags):
    Arm A → with_interactions=False                       → stage1 only.
    Arm B → with_interactions=True,  filter=False         → stages 1–3, gate OFF.
    Arm C → with_interactions=True,  filter=True          → stages 1–3, gate ON.
B and C run an IDENTICAL pipeline; arm C merely fires the concurvity gate inside
the Stage-2 sweep. Both fine-tune EXACTLY ONCE.

HARD CONSTRAINTS honoured here:
    - Rebuild the optimizer (new Trainer / params) after EVERY structural change.
    - Restore best weights (trainer.load_best) before returning / extracting.
    - SPLIT CONTRACT: the internal pool train/val split is keyed off `fold`
      (hp.fold_seed), NEVER off `seed`. `seed` controls init + optimization only.
      Do not reseed the split per replicate — the stability metric depends on the
      data being identical across seeds of a fold.
    - NO per-removal re-fine-tune exists anywhere. The model fine-tunes once
      (Stage 3) for both arms. (See the removed-stage flag at the bottom.)
"""
from na2m.models.na2m import NA2M
from na2m.training.trainer import Trainer
from na2m.utils.config import NA2MConfig
from na2m.selection.policy import SelectionPolicy, NoGate, ConcurvityGate
import torch
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
) -> dict:
    """Run the staged NA2M training pipeline for one arm on one (fold, seed).

    Reproducibility contract (caller's responsibility, NOT this function's):
        - Set the RNG seed (torch, numpy, random) BEFORE creating the model and
          calling fit_na2m. This covers weight initialization and optimization
          stochasticity in one shot.
        - Build the train/val/pool loaders once per fold (keyed off fold, not
          seed) and pass them in. This function never re-splits or re-seeds.

    Args:   
        model: Freshly initialised NA2M with the seed already applied (interactions
               empty at entry).
        train_loader: Internal training split loader (fold-keyed, shared across seeds).
        val_loader: Internal validation split loader (fold-keyed, shared across seeds).
        pool_loader: Full 80% pool loader — reference sample for centering.
        config: NA2MConfig with model + training hyperparameters.
        with_interactions: If False → arm A (stage1 only).
        with_concurvity_filter: If True → arm C (concurvity gate ON in stage 2);
            if False → arm B. Has NO effect when with_interactions is False.

    Returns:
        dict with:
            "model": the trained NA2M (best weights restored, eval mode),
            "active_pairs": model.active_interaction_pairs() (the final S2
                selection set — consumed per-seed by the Jaccard eval).

    TODO:
        - stage2_select(..., with_concurvity_filter=with_concurvity_filter).
        - stage3_finetune(...)   # the SINGLE fine-tune, both arms.
        - model.eval(); assemble and return the result dict.
    """
    stage1_main(model, train_loader, val_loader, pool_loader, config)

    if not with_interactions:
        model.eval()
        return {
            "model": model,
            "active_pairs": model.active_interaction_pairs(),  # []
        }

    policy: SelectionPolicy = (
        ConcurvityGate(threshold=config.concurvity_threshold)
        if with_concurvity_filter
        else NoGate()
    )

    raise NotImplementedError  # stage2_select(..., selection_policy=policy)


def stage1_main(model: NA2M,
                train_loader: DataLoader,
                val_loader: DataLoader,
                pool_loader: DataLoader,
                config: NA2MConfig) -> None:
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
    stage_1_trainer.train(train_loader, val_loader)
    stage_1_trainer.load_best()
    model.center_main_effects(pool_loader)



def stage2_select(model, train_loader, val_loader, pool_loader, hp, *, selection_policy: SelectionPolicy) -> None:
    """FAST screen → block-train top-M → ONE forward prune sweep (two gates).

    The sweep is a SINGLE pass in decreasing contribution order; it NEVER retrains
    and NEVER re-fine-tunes. Arm B and arm C run identical code here — the only
    difference is the selection_policy passed in by the caller.

    Args:
        model: NA2M instance (mains trained + centered).
        train_loader: Internal training split loader (fold-keyed).
        val_loader: Internal validation split loader (fold-keyed).
        pool_loader: Full 80% pool loader — used for FAST screen, concurvity
            basis vectors, and re-centering.
        hp: Hyperparameters (top_m, eta_prune, block-train epochs).
        selection_policy: Decides per-candidate acceptance during the sweep.
            Pass NoGate() for arm B, ConcurvityGate(threshold) for arm C.
            Swap in any SelectionPolicy implementation to change the strategy.

        # Block-train Trainer: clarity ON, SAME coefficient as stage 3
        # (hp.clarity_regularization).

    TODO:
        - FAST: fast_screen(main_model=model, X, y, task) -> ranked (j,k); take top_m.
        - add_interactions(top_m); set_main_trainable(False); ALSO freeze _bias for
          block-train (it is about to be re-centered; do not let it absorb signal);
          REBUILD optimizer over interaction params only.
        - Block-train ALL top_m subnets JOINTLY, once, with
          loss = task_loss + hp.clarity_regularization * model.clarity_loss(x)
          (same coefficient as stage 3, matches GAMI-Net train_interaction). load_best.
        - center_interactions(X_pool, fold_bias=False)   # per-term offsets HELD,
          _bias untouched, so any candidate later skipped/excluded leaves no orphan
          bias and cannot contaminate the sweep's validation loss.
        - Contribution ranking = VARIANCE of each centered block-trained interaction
          output vector on the TRAIN split (GAMI-Net moving_norm, w_i=1), descending.

        --- SINGLE FORWARD PRUNE SWEEP (eval only, no retrain) ---
        - accepted = []        # ordered list of (j,k) that passed the gate(s)
        - val_losses = []      # val loss AFTER accepting each candidate
        - For cand in ranking (decreasing contribution):
            * SELECTION GATE — delegate to the policy, which carries all strategy
              details (threshold, formula, state):
                if not selection_policy.should_accept(cand, cand_vec, accepted_vecs, main_vecs):
                    continue   # SKIP — never reconsidered, not added to `accepted`.
            * accepted.append(cand)
            * val_losses.append( val loss of {mains + accepted} on val_loader )
                # Compute from summed CENTERED per-term outputs + _bias, NOT a full
                # forward(): non-accepted top_m subnets are still present but must
                # contribute nothing to this measurement.
        - PREDICTIVE-CONTRIBUTION GATE (after the sweep, on the accepted prefix):
            losses = val_losses; lo = min(losses); rng = max(losses) - lo
            if rng > 0 and any((losses - lo)/rng <= hp.eta_prune):
                cut = first index with (losses[i]-lo)/rng <= hp.eta_prune
            else:
                cut = argmin(losses)              # degenerate-range fallback
            survivors = accepted[: cut + 1]        # CHECK off-by-one vs reference prune

        - DROP every top_m pair NOT in `survivors` (both concurvity-skipped and
          predictive-cut): remove_interaction each (nothing folded yet → clean
          delete); REBUILD optimizer.
        - center_interactions(X_pool, fold_bias=True)   # fold survivors ONCE now
          that the surviving set is fixed.
        - unfreeze _bias (stage 3 trains it).
    """
    #We have the model with trained main effects. Now, on those residuals, we should compute the Top M fast ranking interaction terms
    




    raise NotImplementedError


def stage3_finetune(model, train_loader, val_loader, X_pool, hp) -> None:
    """Unfreeze all params; fine-tune ONCE with the clarity penalty; recenter.

    This is the model's ONLY fine-tune, for both arm B and arm C. There is no
    re-fine-tune anywhere downstream.

    Args:
        model: NA2M instance (interactions selected & pruned, survivors folded).
        train_loader: Internal training split loader (fold-keyed).
        val_loader: Internal validation split loader (fold-keyed).
        X_pool: 80% pool features — the reference sample for re-centering.
        hp: Hyperparameters (clarity coefficient, fine-tune epochs).

    TODO:
        - set_main_trainable(True); ensure _bias trainable; REBUILD optimizer over ALL params.
        - Fine-tune: loss = task_loss + hp.clarity_regularization * model.clarity_loss(x);
          load_best.
        - model.center_main_effects(X_pool)
        - model.center_interactions(X_pool, fold_bias=True)   # re-center, matches
          GAMI-Net fine_tune_all. (Post-fine-tune geometry has moved, so the deployed
          concurvity is NOT guaranteed ≤ threshold even for arm C — that is measured
          post-hoc by concurvity_summary, it is not re-gated here.)
    """
    raise NotImplementedError


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