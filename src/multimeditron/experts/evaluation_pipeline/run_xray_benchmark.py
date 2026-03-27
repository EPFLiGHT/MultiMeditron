#!/usr/bin/env python3
"""Run XRay benchmark against an already trained model directory."""

import argparse
import os
import sys


def _parse_bool(value: str) -> bool:
    lowered = value.strip().lower()
    if lowered in {"1", "true", "t", "yes", "y"}:
        return True
    if lowered in {"0", "false", "f", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run XRay benchmark for a trained CLIP-like model."
    )
    parser.add_argument(
        "model_path",
        help="Path to a trained model directory compatible with VisionTextDualEncoderModel.from_pretrained.",
    )
    parser.add_argument(
        "--is-lion-model",
        type=_parse_bool,
        default=False,
        help="Set true if the model is a Lion/OpenCLIP variant (default: false).",
    )

    args = parser.parse_args()

    model_path = os.path.abspath(args.model_path)
    if not os.path.exists(model_path):
        print(f"Model path does not exist: {model_path}")
        return 2

    # Ensure local imports (xray_eval.py -> load_from_clip.py etc.) resolve reliably.
    eval_dir = os.path.dirname(os.path.abspath(__file__))
    if eval_dir not in sys.path:
        sys.path.insert(0, eval_dir)

    from xray_eval import XRay_benchmark

    benchmark = XRay_benchmark(is_lion_model=args.is_lion_model)
    score = benchmark.evaluate(model_path)

    print(f"XRay benchmark score: {score}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
