"""
fit_na2m — staged-training orchestrator for the NA2M arms.

Drives the GAMI-Net-style training pipeline on ONE (fold, seed) for ONE arm,
mutating a dynamic NA2M model in place. Staging lives here — there is NO
separate TrainerNA2M; each stage builds/rebuilds a Trainer over the right
parameter subset.

Stages:
    1. stage1_main      — train the main bank.
    2. stage2_select    — FAST screen → top-M → add interactions → block-train
                          (mains frozen) → η-prune by validation sweep.
    3. stage3_finetune  — unfreeze → fine-tune all params with the marginal-clarity
                          penalty wired in.
    4. stage4_concurvity— (filter arm only) iteratively remove the worst concurve
                          pair on the 80% pool until all ≤ 0.5; count passes.

Arm mapping:
    Arm A → with_interactions=False                  → stage1 only.
    Arm B → with_interactions=True,  filter=False    → stages 1–3 (one fine-tune).
    Arm C → with_interactions=True,  filter=True     → stages 1–4.

HARD CONSTRAINTS honoured here:
    - Rebuild the optimizer (new Trainer / params) after EVERY structural change.
    - Restore best weights (trainer.load_best) before returning / extracting.
    - Concurvity is computed on the FINAL fine-tuned subnets over the 80% pool.
    - B′ is NOT trained here — it is derived later by the harness from B and C.
"""


def fit_na2m(
    model,
    X_pool,
    y_pool,
    hp,
    seed,
    *,
    with_interactions: bool,
    with_concurvity_filter: bool,
) -> dict:
    """Run the staged NA2M training pipeline for one arm on one (fold, seed).

    Args:
        model: Freshly initialised NA2M (interactions empty at entry).
        X_pool: The fold's 80% train pool features.
        y_pool: The fold's 80% train pool targets.
        hp: Hyperparameters for this fold (model + training settings).
        seed: Reproducibility seed for this replicate (init + optimization).
        with_interactions: If False → arm A (stage1 only).
        with_concurvity_filter: If True → run stage4 (arm C).

    Returns:
        dict with at least:
            "model": the trained NA2M (best weights restored, eval mode),
            "active_pairs": model.active_pairs(),
            "fine_tune_pass_count": int (1 for B, ≥1 for C, 0/NA for A).

    TODO:
        - Seed RNGs from `seed`.
        - Internal train/val split of the 80% pool (for early stopping & η-prune).
        - stage1_main(...).
        - If not with_interactions: restore best, return arm-A result.
        - stage2_select(...) → stage3_finetune(...).
        - If with_concurvity_filter: stage4_concurvity(...).
        - Restore best weights, set eval(), assemble and return the result dict.
    """
    raise NotImplementedError


def stage1_main(model, train_loader, val_loader, hp) -> None:
    """Train the main bank (Trainer over main params), restore best. Center

    Args:
        model: NA2M instance.
        train_loader: Internal training split loader.
        val_loader: Internal validation split loader.
        hp: Hyperparameters.
        # stage-1 Trainer: clarity_lambda defaults to 0.0 (no interactions exist yet).

    TODO:
        - Trainer over model.parameters() (only mains exist); train; load_best.
        - Freeze _bias during this stage is NOT needed (mains-only, bias trains fine).
        - model.center_main_effects(X_pool)   # fold per-subnet mean into _bias.
    """
    raise NotImplementedError


def stage2_select(model, train_loader, val_loader, X_pool, y_pool, hp) -> None:
    """FAST screen → top-M → add interactions → block-train → η-prune.

    Args:
        model: NA2M instance (mains trained).
        train_loader: Internal training split loader.
        val_loader: Internal validation split loader.
        X_pool: 80% pool features (for the FAST residual screen).
        y_pool: 80% pool targets.
        hp: Hyperparameters (M, η, block-train epochs).

        # Fine-tune Trainer: clarity ON, same coefficient as stage 2.
        # clarity_lambda=hp.clarity_lambda


    TODO:
        - FAST: fast_screen(main_model=model, X, y, task) -> ranked (j,k); top_M.
        - add_interactions(top_M); set_main_trainable(False); ALSO freeze _bias for
          block-train (it's about to be re-centered; let it not absorb signal);
          REBUILD optimizer over interaction params only; block-train; load_best.
        - add_interactions(top_M); set_main_trainable(False); ALSO freeze _bias for
          block-train; REBUILD optimizer over interaction params only.
        - Block-train with loss = task_loss + hp.clarity_lambda * model.clarity_loss(x),
          SAME coefficient as stage 3 (matches GAMI-Net train_interaction). The penalty
          acts on the frozen-mean parents and the training interaction: it shapes f_jk
          toward orthogonality with its parents even at selection time. load_best.
        - NOTE: parents are centered (end of stage 1) but interactions are NOT yet
          centered here, so the f_jk factor carries a nonzero mean and the penalty is
          slightly approximate at this point; this matches the reference, which centers
          interactions only after block-training (center_interactions below).
        - center_interactions(X_pool, fold_bias=False)   # per-term zero-mean, bias untouched.
        - Contribution ranking = VARIANCE of each centered interaction output vector
          on the TRAIN split (GAMI-Net moving_norm with w_i=1), descending.
        - η-prune sweep on VAL, cumulative adds in ranking order, EVAL ONLY (no retrain):
            losses=[l_0..l_M]; lo=min; rng=max-min;
            if rng>0 and any((losses-lo)/rng < hp.loss_threshold):
                k = first such index           # min-max-normalized rule (NOT (1+η)min)
            else: k = argmin(losses)
            survivors = ranking[:k]            # CHECK off-by-one vs reference prune
        - remove_interaction for the dropped pairs (nothing folded yet → clean delete);
          REBUILD optimizer.
        - center_interactions(X_pool, fold_bias=True)     # fold survivors once.
        - unfreeze _bias (stage 3 trains it).
    """
    raise NotImplementedError


def stage3_finetune(model, train_loader, val_loader, hp) -> None:
    """Unfreeze all params; fine-tune with the marginal-clarity penalty; recenter.

    Args:
        model: NA2M instance (interactions selected & pruned).
        train_loader: Internal training split loader.
        val_loader: Internal validation split loader.
        hp: Hyperparameters (clarity-penalty coefficient, fine-tune epochs).

        # Block-train Trainer: clarity ON, same coefficient as stage 3.
        # clarity_lambda=hp.clarity_lambda


    TODO:
        - set_main_trainable(True); ensure _bias trainable; REBUILD optimizer over ALL params.
        - Fine-tune: loss = task_loss + hp.clarity_lambda * model.clarity_loss(x); load_best.
        - model.center_main_effects(X_pool)
        - model.center_interactions(X_pool, fold_bias=True)   # re-center, matches GAMI-Net fine_tune_all.
    """
    raise NotImplementedError


def stage4_concurvity(model, train_loader, val_loader, X_pool, y_pool, hp) -> int:
    """Iteratively remove the worst concurve pair until all ≤ 0.5. re-fine-tune each pass.

    Args:
        model: NA2M instance (fine-tuned).
        train_loader: Internal training split loader.
        val_loader: Internal validation split loader.
        X_pool: 80% pool features (concurvity is computed here).
        y_pool: 80% pool targets.
        hp: Hyperparameters (iteration cap).

    Returns:
        fine_tune_pass_count: number of fine-tune passes performed (for the
        accuracy non-inferiority comparison; B does one, C may do several).

    TODO:
        - passes = 0
        - Loop (cap hp.max_concurvity_iters):
            * scores = concurvity per active pair on X_pool, basis = ALL OTHER terms
              (mains + other active interactions, exclude self); adj-R² with intercept;
              p = K + (n_active_pairs - 1), recomputed each iter; score the RAW fitted
              vector (purification = future work).
            * worst_pair = max by (score, key) for DETERMINISTIC tie-break.
            * if worst_score <= 0.5: break.
            * remove_interaction(worst_pair)  (subtract-back handled inside); REBUILD opt.
            * stage3_finetune(...)  (re-fine-tune + re-center); passes += 1
        - Loop NEVER re-screens or re-adds (active set only shrinks).
        - return passes
    """
    raise NotImplementedError
