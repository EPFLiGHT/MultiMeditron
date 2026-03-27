from __future__ import annotations

import argparse
import sys
from pathlib import Path

from transformers import VisionTextDualEncoderModel

REPO_ROOT = Path(__file__).resolve().parents[1]
EVAL_PIPELINE_DIR = REPO_ROOT / "src" / "multimeditron" / "experts" / "evaluation_pipeline"

# The benchmark module uses local imports like `from Benchmark import Benchmark`,
# so we add the evaluation_pipeline folder to sys.path before importing it.
sys.path.insert(0, str(EVAL_PIPELINE_DIR))

from ultrasound_new_new_benchmark import (  # noqa: E402
    BodyPartsDataset,
    NUM_CLASSES,
    build_class_weights,
)
from mlp_eval import MLP_eval  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the ultrasound benchmark without launching the full training pipeline.",
    )
    parser.add_argument("model_path", type=Path, help="Path to the trained model checkpoint directory")
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=None,
        help="Optional cache directory for precomputed train/test embeddings",
    )
    parser.add_argument(
        "--rebuild-cache",
        action="store_true",
        help="Recompute embeddings instead of reusing cached .pt files",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run a lighter smoke test with fewer folds, epochs, and hyperparameter trials",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VisionTextDualEncoderModel.from_pretrained(args.model_path)
    model = model.to(device)
    model.eval()

    model_name = args.model_path.name
    use_cache = not args.rebuild_cache

    train_dataset = BodyPartsDataset(
        model=model,
        model_name=model_name,
        split="train",
        cache_root=args.cache_root,
        use_cache=use_cache,
    )
    print(f"Training dataset ready: {len(train_dataset)} samples")

    test_dataset = BodyPartsDataset(
        model=model,
        model_name=model_name,
        split="test",
        cache_root=args.cache_root,
        use_cache=use_cache,
    )
    print(f"Test dataset ready: {len(test_dataset)} samples")

    class_weights = build_class_weights(train_dataset.labels)
    loss = nn.CrossEntropyLoss(weight=class_weights)

    if args.quick:
        benchmark = MLP_eval(
            output_dim=NUM_CLASSES,
            training_set=train_dataset,
            test_set=test_dataset,
            k=2,
            iteration_number=1,
            n_epoch=2,
            loss=loss,
        )
        benchmark.kfold = benchmark.kfold.__class__(n_splits=benchmark.k, shuffle=True, random_state=42)
        benchmark.evaluate = lambda: quick_evaluate(benchmark)
    else:
        benchmark = MLP_eval(
            output_dim=NUM_CLASSES,
            training_set=train_dataset,
            test_set=test_dataset,
            loss=loss,
        )

    result = benchmark.evaluate()
    print(f"Benchmark result: {result}")


def quick_evaluate(benchmark: MLP_eval) -> float:
    learning_rates = [0.001]
    weight_decays = [0.01]
    best_result = -1.0
    best_lr = learning_rates[0]
    best_wd = weight_decays[0]

    for lr in learning_rates:
        for wd in weight_decays:
            kfold_result = benchmark.k_fold_training(lr, wd)
            if kfold_result > best_result:
                best_result = kfold_result
                best_lr = lr
                best_wd = wd

    print(f"Quick mode best params: lr={best_lr}, wd={best_wd}")
    _, best_classifier = benchmark.training(benchmark.train_loader, best_lr, best_wd)
    final_result = benchmark.evaluate_fold(best_classifier, benchmark.test_loader)
    print(f"Quick mode test value: {final_result}")
    return final_result


if __name__ == "__main__":
    main()
