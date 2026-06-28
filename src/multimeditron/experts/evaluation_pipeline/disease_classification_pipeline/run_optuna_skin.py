"""
Entry-point Script for Skin Disease CLIP Hyperparameter Optimization.

Usage:
  python run_optuna_skin.py \
    --config CONFIG.yaml \
    --train-jsonl PATH \
    --test-jsonl PATH \
    --image-root PATH \
    [--output PATH]

Notes:
- Saves a single Optuna study pickle to --output (default: study_skin.pkl).
"""

import argparse
import pickle
from pathlib import Path

from skin_benchmark import SkinDiseaseBenchmark
from train_hp_opt import train


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", required=True, help="YAML config for HP tuning (training setup)"
    )

    parser.add_argument("--train-jsonl", required=True, help="Path to training jsonl")
    parser.add_argument(
        "--test-jsonl", required=True, help="Path to validation/test jsonl"
    )
    parser.add_argument(
        "--image-root",
        required=True,
        help="Root directory for images referenced by jsonl",
    )

    parser.add_argument(
        "--output", default="study_skin.pkl", help="Where to save the study pickle"
    )

    args = parser.parse_args()

    skin_bench = SkinDiseaseBenchmark(
        train_jsonl=args.train_jsonl,
        test_jsonl=args.test_jsonl,
        image_root=args.image_root,
    )

    study = train([skin_bench], args.config)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("wb") as f:
        pickle.dump(study, f)

    print(f"study saved: {out_path.resolve()}")


if __name__ == "__main__":
    main()
