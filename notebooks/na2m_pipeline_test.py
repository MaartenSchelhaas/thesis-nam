"""
na2m_pipeline_test.py — end-to-end test of the full NA2M pipeline.

Runs 2 folds (KFold minimum) with 3 seeds and all three arms.
Patches the search config in memory and writes a temporary YAML so the original
config on disk is never modified:
    - n_trials          20 → 10   (main-effects optuna trials)
    - clarity_n_trials  15 → 5    (clarity optuna trials)
    - num_epochs        100 → 20  (main + fine-tune stages)
    - block_train_epochs 100 → 20 (interaction block-train stage)
    - finetune_epochs   100 → 20  (stage-3 fine-tune)
    - patience          50  → 10

Writes to runs/pipeline_test/ so it never touches runs/compas/.

Run from the project root:
    python -m notebooks.na2m_pipeline_test
"""

from pathlib import Path

import torch
import yaml

from na2m.data.data_utils import load_compas, preprocess
from na2m.utils.config import load_na2m_search_config
from scripts.na2m.run_na2m_eval import evaluate_na2m

# --------------------------------------------------------------------------- #
# Parameters                                                                    #
# --------------------------------------------------------------------------- #

SEARCH_CONFIG_PATH = (
    r"C:\Users\maart\OneDrive\Documenten\Universiteit\Scriptie"
    r"\python_repo\thesis-nam\configs\compas_na2m_search.yaml"
)
BASE_DIR = Path("runs/pipeline_test")
RUN_MODE = "fixed"
N_RUNS   = 3
N_FOLDS  = 2   # KFold requires at least 2; both folds run


# --------------------------------------------------------------------------- #
# Patch config in memory, write to a temp YAML (original never touched)        #
# --------------------------------------------------------------------------- #

fixed_params, search_space = load_na2m_search_config(SEARCH_CONFIG_PATH)

fixed_params["n_trials"]            = 10
fixed_params["clarity_n_trials"]    = 5
fixed_params["num_epochs"]          = 100
fixed_params["block_train_epochs"]  = 100
fixed_params["finetune_epochs"]     = 100
fixed_params["patience"]            = 10

# Reconstruct the full YAML dict (load_na2m_search_config pops search_space,
# so we put it back before dumping).
patched_config_path = BASE_DIR / "patched_search_config.yaml"
BASE_DIR.mkdir(parents=True, exist_ok=True)

patched_raw = dict(fixed_params)
patched_raw["search_space"] = search_space

with open(patched_config_path, "w") as f:
    yaml.dump(patched_raw, f)


# --------------------------------------------------------------------------- #
# Data                                                                          #
# --------------------------------------------------------------------------- #

df = load_compas(fixed_params["dataset_path"])
X, y, feature_meta = preprocess(df)
print(f"Dataset: {X.shape[0]} samples, {X.shape[1]} features")


# --------------------------------------------------------------------------- #
# Run                                                                           #
# --------------------------------------------------------------------------- #

evaluate_na2m(
    str(patched_config_path),
    X, y,
    feature_meta,
    BASE_DIR,
    run_mode=RUN_MODE,
    n_runs=N_RUNS,
    n_folds=N_FOLDS,
)


# --------------------------------------------------------------------------- #
# Verification: check expected files exist and print basic shapes               #
# --------------------------------------------------------------------------- #

print("\n--- Verification ---")

arm_names = ["mains", "gaminet", "concurvity"]

for fold_idx in range(N_FOLDS):
    run_dir = BASE_DIR / f"fold_{fold_idx}" / RUN_MODE
    for arm in arm_names:
        for run_i in range(N_RUNS):
            arm_dir = run_dir / arm / f"run_{run_i}"

            done_path     = arm_dir / "done"
            measures_path = arm_dir / "measures.pt"

            done_ok     = done_path.exists()
            measures_ok = measures_path.exists()

            if not done_ok or not measures_ok:
                print(f"  MISSING  fold_{fold_idx}/{arm}/run_{run_i}  (done={done_ok}, measures={measures_ok})")
                continue

            measures = torch.load(measures_path, weights_only=False)

            print(
                f"  OK  fold_{fold_idx}/{arm}/run_{run_i}"
                f"  subnets={len(measures['subnet_vectors_pool'])}"
                f"  logits={measures['logits'].shape}"
                f"  y_test={measures['y_test'].shape}"
                f"  pairs={len(measures['pairs'])}"
            )

print("\nPipeline test complete.")
