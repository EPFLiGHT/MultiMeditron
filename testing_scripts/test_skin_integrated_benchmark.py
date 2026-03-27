from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path
from random import Random

import torch
from transformers import VisionTextDualEncoderModel

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / 'src' / 'multimeditron' / 'experts'

sys.path.insert(0, str(SRC_ROOT))

from evaluation_pipeline.benchmark_classification.skin_integrated_benchmark import SkinIntegratedBenchmark  # noqa: E402


class Tee:
    def __init__(self, *streams) -> None:
        self.streams = streams

    def write(self, data: str) -> int:
        for stream in self.streams:
            stream.write(data)
            stream.flush()
        return len(data)

    def flush(self) -> None:
        for stream in self.streams:
            stream.flush()


class SmokeSkinIntegratedBenchmark(SkinIntegratedBenchmark):
    def __init__(self, subset_fraction: float, **kwargs) -> None:
        super().__init__(**kwargs)
        self.subset_fraction = subset_fraction

    def _subset_examples(self, examples: list[dict]) -> list[dict]:
        rng = Random(42)
        shuffled = examples.copy()
        rng.shuffle(shuffled)
        subset_size = max(1, int(self.subset_fraction * len(shuffled)))
        return shuffled[:subset_size]

    def load_train_examples(self) -> list[dict]:
        return self._subset_examples(super().load_train_examples())

    def load_test_examples(self) -> list[dict]:
        return self._subset_examples(super().load_test_examples())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Smoke test for the integrated skin benchmark (Skin10 + ISIC).',
    )
    parser.add_argument('model_path', type=Path, help='Path to the trained model checkpoint directory')
    parser.add_argument(
        '--train-jsonl',
        type=Path,
        action='append',
        default=None,
        help='Optional training manifest. Repeat to include multiple manifests.',
    )
    parser.add_argument(
        '--test-jsonl',
        type=Path,
        action='append',
        default=None,
        help='Optional test manifest. Repeat to include multiple manifests.',
    )
    parser.add_argument(
        '--image-root',
        type=Path,
        action='append',
        default=None,
        help='Optional image root. Repeat to include multiple roots.',
    )
    parser.add_argument(
        '--subset-fraction',
        type=float,
        default=0.01,
        help='Fraction of train and test examples to use for the smoke test.',
    )
    parser.add_argument(
        '--cache-root',
        type=Path,
        default=None,
        help='Optional cache directory for precomputed skin embeddings.',
    )
    parser.add_argument(
        '--log-dir',
        type=Path,
        default=REPO_ROOT / 'testing_logs',
        help='Directory where the smoke-test log file should be written.',
    )
    parser.add_argument(
        '--rebuild-cache',
        action='store_true',
        help='Recompute embeddings instead of reusing cached .pt files.',
    )
    parser.add_argument(
        '--full-eval',
        action='store_true',
        help='Run the full benchmark evaluation after the smoke test dataset checks.',
    )
    return parser.parse_args()


def summarize_labels(dataset, split_name: str) -> None:
    labels = dataset.labels.cpu()
    unique, counts = torch.unique(labels, return_counts=True)
    parts = [f'{int(label)}:{int(count)}' for label, count in zip(unique.tolist(), counts.tolist(), strict=True)]
    print(f"{split_name} labels: {' '.join(parts)}")


def setup_logging(log_dir: Path, benchmark_name: str) -> Path:
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
    log_path = log_dir / f'{benchmark_name}_smoke_{timestamp}.log'
    log_file = log_path.open('w', encoding='utf-8')
    sys.stdout = Tee(sys.stdout, log_file)
    sys.stderr = Tee(sys.stderr, log_file)
    return log_path


def main() -> None:
    args = parse_args()

    if not 0 < args.subset_fraction <= 1:
        raise ValueError('--subset-fraction must be in the interval (0, 1].')

    log_path = setup_logging(args.log_dir, 'skin_integrated')
    print(f'Writing log to: {log_path}')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = VisionTextDualEncoderModel.from_pretrained(args.model_path)
    model = model.to(device)
    model.eval()

    benchmark = SmokeSkinIntegratedBenchmark(
        subset_fraction=args.subset_fraction,
        train_jsonls=tuple(args.train_jsonl) if args.train_jsonl else None,
        test_jsonls=tuple(args.test_jsonl) if args.test_jsonl else None,
        image_roots=tuple(args.image_root) if args.image_root else None,
        cache_root=args.cache_root,
    )
    use_cache = not args.rebuild_cache
    model_name = args.model_path.name

    print(f'Testing SkinIntegratedBenchmark with model: {args.model_path}')
    print(f'Train jsonls: {benchmark.train_jsonls}')
    print(f'Test jsonls: {benchmark.test_jsonls}')
    print(f'Image roots: {benchmark.image_roots}')
    print(f'Subset fraction: {benchmark.subset_fraction}')
    print(f'Using cache: {use_cache}')

    train_dataset = benchmark.build_train_dataset(model=model, model_name=model_name, use_cache=use_cache)
    print(f'Train dataset ready: {len(train_dataset)} samples')
    summarize_labels(train_dataset, 'train')

    test_dataset = benchmark.build_test_dataset(model=model, model_name=model_name, use_cache=use_cache)
    print(f'Test dataset ready: {len(test_dataset)} samples')
    summarize_labels(test_dataset, 'test')

    train_sample, train_label = train_dataset[0]
    test_sample, test_label = test_dataset[0]
    print(f'First train sample: shape={tuple(train_sample.shape)} label={int(train_label)}')
    print(f'First test sample: shape={tuple(test_sample.shape)} label={int(test_label)}')

    if args.full_eval:
        result = benchmark.evaluate_model(
            model=model,
            model_name=model_name,
            use_cache=use_cache,
        )
        print(f'Skin integrated benchmark result: {result}')


if __name__ == '__main__':
    main()
