"""
derive — build arm B′ (the count-matched control) from arms B and C.

B′ is DERIVED, never trained as an independent arm: take arm B's trained model,
truncate its interactions to the NUMBER arm C kept, then re-fine-tune. Built
AFTER B and C exist for a given (fold, seed). This is reusable model logic the
eval driver calls — it lives in `src`, not in the script.
"""

from .fit_na2m import stage3_finetune  # B′ re-fine-tunes via the same stage-3 step


def derive_b_prime(b_model, c_result, X_pool, y_pool, hp, seed) -> dict:
    """Derive arm B′ from arm B's model and arm C's kept-interaction count.

    B′ = arm B truncated to len(c_result["active_pairs"]) interactions, then
    re-fine-tuned. NEVER trained from scratch.

    Args:
        b_model: Arm B's trained NA2M (held in memory for this seed).
        c_result: Arm C's fit_na2m result (read its active-pair count).
        X_pool: The fold's 80% pool features.
        y_pool: The fold's 80% pool targets.
        hp: Hyperparameters.
        seed: The replicate seed.

    Returns:
        dict mirroring a fit_na2m result for B′ (model, active_pairs,
        fine_tune_pass_count) — for extraction, then discard.

    TODO:
        - n_keep = len(c_result["active_pairs"]).
        - Truncate B's interactions to the n_keep best by B's selection ranking;
          remove_interaction the rest; REBUILD optimizer.
        - Re-fine-tune via stage3_finetune; restore best; eval.
        - Assemble and return the result dict.
    """
    raise NotImplementedError
