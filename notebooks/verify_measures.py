"""
Verify that measures.pt contains the expected keys and shapes.
Run from the repo root: python -m notebooks.verify_measures
"""
import numpy as np
import torch
from pathlib import Path


ARMS = [
    ("mains",      "runs/pipeline_test/fold_0/fixed/mains/run_0/measures.pt"),
    ("gaminet",    "runs/pipeline_test/fold_0/fixed/gaminet/run_0/measures.pt"),
    ("concurvity", "runs/pipeline_test/fold_0/fixed/concurvity/run_0/measures.pt"),
]


def _section(title: str) -> None:
    print(f"\n{'─'*60}")
    print(f"  {title}")
    print(f"{'─'*60}")


def verify(path: Path, arm_label: str) -> None:
    _section(f"{arm_label}  ←  {path}")

    if not path.exists():
        print(f"  FILE NOT FOUND — run the pipeline first")
        return

    m = torch.load(path, map_location="cpu", weights_only=False)
    print(f"\nkeys: {list(m.keys())}")

    # --- Raw subnet vectors ---
    for field in ("subnet_vectors_pool", "subnet_vectors_test"):
        vecs = m[field]
        print(f"\n{field}: {len(vecs)} term(s)")
        for sid, vec in vecs.items():
            print(f"  {sid}  shape={vec.shape}  range=[{vec.min():.6f}, {vec.max():.6f}]  std={vec.std():.6f}")

    # --- Test logits ---
    logits = m["logits"]
    print(f"\nlogits: shape={logits.shape}  range=[{logits.min():.4f}, {logits.max():.4f}]")

    # --- Selected pairs ---
    print(f"\npairs: {m['pairs']}")

    # --- Shape-plot curves ---
    curves = m.get("curves", {})
    if not curves:
        print(f"\ncurves: EMPTY — was the extractor updated before this run?")
    else:
        print(f"\ncurves: {len(curves)} main effect(s)")
        for sid, curve in curves.items():
            inp = curve["inputs"]
            out = curve["outputs"]
            x_desc = (
                f"[{inp[0]!r} … {inp[-1]!r}]" if len(inp) > 2
                else str(inp.tolist())
            )
            print(f"  {sid}  inputs={inp.shape} dtype={inp.dtype}  x={x_desc}")
            print(f"          outputs={out.shape}  range=[{out.min():.4f}, {out.max():.4f}]")
            assert inp.shape == out.shape, f"inputs/outputs length mismatch for {sid}"
        print("  [OK] inputs.shape == outputs.shape for all curves")

    # --- Labels ---
    y = m["y_test"]
    unique, counts = np.unique(y, return_counts=True)
    balance = {int(k): int(v) for k, v in zip(unique, counts)}
    print(f"\ny_test: shape={y.shape}  class_balance={balance}")


def main() -> None:
    for arm_label, rel_path in ARMS:
        verify(Path(rel_path), arm_label)
    print()


if __name__ == "__main__":
    main()
