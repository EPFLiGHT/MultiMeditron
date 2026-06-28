#!/usr/bin/env python3
"""
Generate benchmark split files for BENCHMARK_SPECS datasets.

For each benchmark dataset, the nemo val.jsonl is split deterministically
into three parts:
  - mlp_train.jsonl  : 50% of val  → trains the MLP probe during Optuna HPO
  - bench_eval.jsonl : 25% of val  → Optuna evaluation metric
  - holdout.jsonl    : 25% of val  → final evaluation (use only once)

The nemo train.jsonl is left untouched as the general training split.
TRAIN_ONLY_SPECS are not processed here (all their data goes to training).

Output: benchmark_splits/nemo/<dataset_name>/{mlp_train,bench_eval,holdout}.jsonl

Usage:
    python scripts/build_splits.py
    python scripts/build_splits.py --dry-run
"""


import argparse
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src" / "multimeditron" / "experts"))
from dataset_specs import BENCHMARK_SPECS


def split_val(val_jsonl, out_dir, seed=42, dry_run=False):
    lines = val_jsonl.read_text().splitlines()
    rng = random.Random(seed)
    rng.shuffle(lines)

    n = len(lines)
    n_mlp = n // 2
    n_bench = n // 4
    splits = {
        "mlp_train.jsonl": lines[:n_mlp],
        "bench_eval.jsonl": lines[n_mlp : n_mlp + n_bench],
        "holdout.jsonl": lines[n_mlp + n_bench :],
    }

    if not dry_run:
        out_dir.mkdir(parents=True, exist_ok=True)

    for fname, rows in splits.items():
        out_path = out_dir / fname
        if dry_run:
            print(f"  [dry-run] {out_path}  ({len(rows)} rows)")
        else:
            out_path.write_text("\n".join(rows) + "\n")
            print(f"  {out_path}  ({len(rows)} rows)")

    return {k: len(v) for k, v in splits.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be written without creating files",
    )
    args = parser.parse_args()

    out_root = Path(__file__).parents[1] / "benchmark_splits" / "nemo"

    print(f"Output root: {out_root}")
    print(f"Seed: {args.seed}")
    print()

    summary = []
    for spec in BENCHMARK_SPECS:
        print(f"[{spec.name}]  val_jsonl = {spec.val_jsonl}")
        if not spec.val_jsonl.exists():
            print(f"  WARNING: val_jsonl not found, skipping")
            continue

        out_dir = out_root / spec.name
        counts = split_val(
            spec.val_jsonl, out_dir, seed=args.seed, dry_run=args.dry_run
        )
        summary.append((spec.name, counts))
        print()

    print("=== Summary ===")
    print(f"{'dataset':15s}  {'mlp_train':>9}  {'bench_eval':>10}  {'holdout':>7}")
    for name, counts in summary:
        print(
            f"{name:15s}  "
            f"{counts.get('mlp_train.jsonl', 0):9d}  "
            f"{counts.get('bench_eval.jsonl', 0):10d}  "
            f"{counts.get('holdout.jsonl', 0):7d}"
        )


if __name__ == "__main__":
    main()
