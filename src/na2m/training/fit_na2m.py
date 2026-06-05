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
    """Train the main bank (Trainer over main params), restore best.

    Args:
        model: NA2M instance.
        train_loader: Internal training split loader.
        val_loader: Internal validation split loader.
        hp: Hyperparameters.

    TODO:
        - Trainer over model.parameters() (only mains exist yet); train; load_best.
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

    TODO:
        - FAST screen on main-model residual-equivalent predictions (nam.selection.fast)
          → ranked (j,k) → top-M.
        - model.add_interactions(top_M); REBUILD optimizer over interaction params only
          (set_main_trainable(False)); block-train.
        - η-prune: cumulative validation-loss sweep over ranked kept pairs
          (evaluation only, NO retrain); remove_interaction for the dropped ones;
          REBUILD optimizer after removals.
    """
    raise NotImplementedError


def stage3_finetune(model, train_loader, val_loader, hp) -> None:
    """Unfreeze all params; fine-tune with the marginal-clarity penalty.

    Args:
        model: NA2M instance (interactions selected & pruned).
        train_loader: Internal training split loader.
        val_loader: Internal validation split loader.
        hp: Hyperparameters (clarity-penalty coefficient, fine-tune epochs).

    TODO:
        - set_main_trainable(True); REBUILD optimizer over ALL params.
        - Fine-tune with model.clarity_loss(x) added to the loss; load_best.
    """
    raise NotImplementedError


def stage4_concurvity(model, train_loader, val_loader, X_pool, y_pool, hp) -> int:
    """Iteratively remove the worst concurve pair until all ≤ 0.5.

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
        - Loop (capped): compute concurvity on fine-tuned subnets over the 80% pool.
        - If max concurvity > 0.5: remove_interaction the single worst pair;
          REBUILD optimizer; re-fine-tune (reuse stage3_finetune); increment count.
        - Else break (fixed point). Return the pass count.
    """
    raise NotImplementedError
