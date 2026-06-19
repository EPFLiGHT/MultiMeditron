"""
Split a JSONL file into train / validation sets in a reproducible way.

- Shuffles deterministically with a seed
- Writes two JSONL files
- Dataset-agnostic (works for any JSONL)
"""

import argparse
import random
from pathlib import Path


def split_jsonl(
    input_file: Path,
    train_file: Path,
    val_file: Path,
    val_ratio: float,
    seed: int,
):
    # Read all lines
    with input_file.open("r", encoding="utf-8") as f:
        lines = [ln for ln in f if ln.strip()]

    n_total = len(lines)
    if n_total == 0:
        raise ValueError(f"Input JSONL is empty: {input_file}")

    # Shuffle deterministically
    rng = random.Random(seed)
    rng.shuffle(lines)

    # Compute split sizes
    n_val = int(n_total * val_ratio)
    if n_val == 0 and n_total > 0:
        n_val = 1  # ensure at least one val example

    val_lines = lines[:n_val]
    train_lines = lines[n_val:]

    # Ensure output directories exist
    train_file.parent.mkdir(parents=True, exist_ok=True)
    val_file.parent.mkdir(parents=True, exist_ok=True)

    # Write outputs
    with val_file.open("w", encoding="utf-8") as f:
        f.writelines(val_lines)

    with train_file.open("w", encoding="utf-8") as f:
        f.writelines(train_lines)

    print("=== JSONL split complete ===")
    print(f"Input:  {input_file}")
    print(f"Total:  {n_total}")
    print(f"Train:  {len(train_lines)} → {train_file}")
    print(f"Val:    {len(val_lines)} → {val_file}")
    print(f"Seed:   {seed}")
    print(f"Val ratio: {val_ratio}")


def main():
    parser = argparse.ArgumentParser(description="Split a JSONL file into train/val.")
    parser.add_argument(
        "--input-jsonl",
        type=Path,
        required=True,
        help="Path to input JSONL file",
    )
    parser.add_argument(
        "--train-jsonl",
        type=Path,
        required=True,
        help="Path to output train JSONL file",
    )
    parser.add_argument(
        "--val-jsonl",
        type=Path,
        required=True,
        help="Path to output validation JSONL file",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.1,
        help="Fraction of data to use for validation (default: 0.1)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )

    args = parser.parse_args()

    if not (0.0 < args.val_ratio < 1.0):
        raise ValueError("--val-ratio must be in (0, 1)")

    split_jsonl(
        input_file=args.input_jsonl,
        train_file=args.train_jsonl,
        val_file=args.val_jsonl,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
