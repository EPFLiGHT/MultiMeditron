from __future__ import annotations

import argparse
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from random import Random

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / 'src' / 'multimeditron' / 'experts'

sys.path.insert(0, str(SRC_ROOT))

from evaluation_pipeline.scin_benchmark import SCINBenchmark  # noqa: E402


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

    def isatty(self) -> bool:
        return any(getattr(stream, 'isatty', lambda: False)() for stream in self.streams)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Smoke test for the SCIN hard-negative benchmark.')
    parser.add_argument('model_path', type=Path, help='Path to the trained model checkpoint directory')
    parser.add_argument(
        '--eval-jsonl',
        type=Path,
        action='append',
        default=None,
        help='Optional evaluation manifest. Repeat to include multiple eval manifests.',
    )
    parser.add_argument(
        '--manifest-jsonl',
        type=Path,
        action='append',
        default=None,
        help='Optional metadata manifest. Repeat to include multiple manifests.',
    )
    parser.add_argument(
        '--subset-fraction',
        type=float,
        default=0.1,
        help='Fraction of SCIN eval items to keep for the smoke test.',
    )
    parser.add_argument(
        '--protocol-cache-path',
        type=Path,
        default=None,
        help='Optional cache file for the hard-negative protocol.',
    )
    parser.add_argument(
        '--log-dir',
        type=Path,
        default=REPO_ROOT / 'testing_logs',
        help='Directory where the smoke-test log file should be written.',
    )
    return parser.parse_args()


def setup_logging(log_dir: Path, benchmark_name: str) -> Path:
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
    log_path = log_dir / f'{benchmark_name}_smoke_{timestamp}.log'
    log_file = log_path.open('w', encoding='utf-8')
    sys.stdout = Tee(sys.stdout, log_file)
    sys.stderr = Tee(sys.stderr, log_file)
    return log_path


def build_subset_jsonl(src_path: Path, subset_fraction: float, out_dir: Path) -> Path:
    import json

    rng = Random(42)
    lines = src_path.read_text(encoding='utf-8').splitlines()
    if not lines:
        raise ValueError(f'Empty JSONL: {src_path}')
    rng.shuffle(lines)
    subset_size = max(4, int(len(lines) * subset_fraction))
    chosen = lines[:subset_size]

    normalized_lines = []
    for line in chosen:
        example = json.loads(line)
        image_value = Path(example['modalities'][0]['value'])
        if not image_value.is_absolute():
            image_value = (src_path.parent / image_value).resolve()
        example['modalities'][0]['value'] = str(image_value)
        normalized_lines.append(json.dumps(example))

    out_path = out_dir / f'{src_path.stem}_subset_{subset_size}.jsonl'
    out_path.write_text(chr(10).join(normalized_lines) + chr(10), encoding='utf-8')
    return out_path


def main() -> None:
    args = parse_args()
    if not 0 < args.subset_fraction <= 1:
        raise ValueError('--subset-fraction must be in the interval (0, 1].')

    log_path = setup_logging(args.log_dir, 'scin')
    print(f'Writing log to: {log_path}')

    eval_jsonls = tuple(args.eval_jsonl) if args.eval_jsonl else SCINBenchmark.default_eval_jsonls
    manifest_jsonls = tuple(args.manifest_jsonl) if args.manifest_jsonl else SCINBenchmark.default_manifest_jsonls

    tmp_root = Path('/tmp/scin_smoke')
    tmp_root.mkdir(parents=True, exist_ok=True)
    subset_eval_jsonls = tuple(build_subset_jsonl(path, args.subset_fraction, tmp_root) for path in eval_jsonls)

    protocol_cache_path = args.protocol_cache_path
    if protocol_cache_path is None:
        protocol_cache_path = tmp_root / f'scin_protocol_{int(args.subset_fraction * 10000)}.json'
        if protocol_cache_path.exists():
            protocol_cache_path.unlink()

    benchmark = SCINBenchmark(
        eval_jsonls=subset_eval_jsonls,
        manifest_jsonls=manifest_jsonls,
        protocol_cache_path=protocol_cache_path,
    )

    print(f'Testing SCINBenchmark with model: {args.model_path}')
    print(f'Eval jsonls: {benchmark.eval_jsonls}')
    print(f'Manifest jsonls: {benchmark.manifest_jsonls}')
    print(f'Subset fraction: {args.subset_fraction}')
    print(f'Protocol cache: {benchmark.protocol_cache_path}')
    print(f'Items: {len(benchmark.items)}')
    print(f'Protocol triples: {len(benchmark.triples)}')
    print(f'Skin groups: {dict(Counter(item["_skin_group"] for item in benchmark.items))}')

    score = benchmark.evaluate(str(args.model_path))
    print(f'SCIN benchmark score: {score}')


if __name__ == '__main__':
    main()
